"""Subagents owned by other teams, called over the network.

* Support Case runs on its own Agent Server (``langgraph.support_case.json``) and is
  called with ``RemoteGraph``. To move it to LangSmith Deployment you only need to set
  ``SUPPORT_CASE_URL`` and ``LANGSMITH_API_KEY``.
* Customer Success is only called over A2A (``customer_success_subagent`` and
  ``fetch_agent_card``), so it could be any agent that supports A2A.

The orchestrator calls both through ``task``, like any local subagent. Only the runnable
behind each one knows how to reach it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import httpx
from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langgraph.pregel.remote import RemoteGraph
from langgraph.runtime import get_runtime
from langsmith.run_helpers import get_current_run_tree

from community_assistant.context import UserContext, require_user_context
from community_assistant.orchestrator.middleware import FAILURE_MESSAGE

logger = logging.getLogger(__name__)

SUPPORT_CASE_URL = os.environ.get("SUPPORT_CASE_URL", "http://127.0.0.1:2025")

SUPPORT_CASE = RemoteGraph(
    "support_case",
    url=SUPPORT_CASE_URL,
    api_key=os.environ.get("LANGSMITH_API_KEY"),
    distributed_tracing=True,
)


async def _call_support_case(state: dict) -> dict:
    """Send the delegated request to Support Case and return its reply.

    A plain RemoteGraph doesn't fit the task tool on its own, so this does three things:

    * Sends the user's identity. The orchestrator's runtime context doesn't cross the
      network by itself, and Support Case checks the user's role.
    * Builds a clean config with a new remote thread. Otherwise the orchestrator's own
      config (including its thread_id) would be forwarded, and the Agent Server rejects
      a request that has both ``configurable`` and ``context``.
    * Converts the reply. RemoteGraph returns messages as plain dicts, and task only
      reads message objects (deepagents#6126).
    """
    user = require_user_context(get_runtime().context)
    out = await SUPPORT_CASE.ainvoke(
        {"messages": state["messages"]},
        {"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user.to_dict(),
    )
    return {"messages": convert_to_messages(out["messages"])}


def support_case_subagent() -> CompiledSubAgent:
    return {
        "name": "support_case",
        "description": "Support team's agent (separate deployment). Looks up case status and creates support cases.",
        "runnable": RunnableLambda(_call_support_case, name="support_case"),
    }


# ---------------------------------------------------------------------------
# Customer Success: A2A
#
# A2A JSON-RPC ``message/send`` over HTTP, so the remote side can be any agent that
# supports A2A. The user context is sent in two places:
#
# * ``message.metadata["user_context"]``: standard A2A, so any agent can read it.
# * ``params.context``: an Agent Server addition that isn't in the A2A spec. The Agent
#   Server turns it into the run's context, so the remote tools see it as
#   ``runtime.context``. Other A2A agents ignore it.
# ---------------------------------------------------------------------------

CUSTOMER_SUCCESS_URL = os.environ.get("CUSTOMER_SUCCESS_URL", "http://127.0.0.1:2026")
# The Agent Server addresses A2A agents by assistant UUID, not graph name
# (``/a2a/customer_success`` is rejected with "Invalid assistant ID: must be a UUID"). A
# graph's default assistant ID is the uuid5 of the graph name in the server's namespace.
_AGENT_SERVER_GRAPH_NAMESPACE = uuid.UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
CUSTOMER_SUCCESS_AGENT_ID = str(uuid.uuid5(_AGENT_SERVER_GRAPH_NAMESPACE, "customer_success"))
A2A_ENDPOINT = f"{CUSTOMER_SUCCESS_URL.rstrip('/')}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"
A2A_TIMEOUT = httpx.Timeout(60.0, connect=5.0)  # a remote agent turn can include model calls


class A2AError(RuntimeError):
    """The A2A call failed: a network error, a JSON-RPC error, or a reply with no text.

    The subagent retries once when this happens. If it fails again, the error comes out of
    the ``task`` call and the orchestrator's tool middleware turns it into the standard
    failure message.
    """


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=A2A_TIMEOUT)


def _auth_headers() -> dict[str, str]:
    """``x-api-key`` for the Customer Success endpoint, only if ``CUSTOMER_SUCCESS_API_KEY`` is set.

    This doesn't fall back to LANGSMITH_API_KEY on purpose. The A2A endpoint might belong
    to a third party, and a key shouldn't be sent somewhere it wasn't meant for.
    """
    key = os.environ.get("CUSTOMER_SUCCESS_API_KEY")
    return {"x-api-key": key} if key else {}


def _tracing_headers() -> dict[str, str]:
    """LangSmith distributed tracing headers for the current run, if there is one.

    The Agent Server copies ``langsmith-trace`` (and ``baggage``) from A2A requests into
    the run's config, the same as on its other run endpoints. The remote graph uses them
    to continue our trace (see ``community_assistant.tracing``), so the remote run shows
    up under this call in LangSmith. Other A2A agents ignore headers they don't recognize.

    These headers go to whatever endpoint is configured. The trace ID isn't secret, but
    ``baggage`` includes your LangSmith project name, so drop these headers if the A2A
    agent belongs to a third party that shouldn't see it.
    """
    run_tree = get_current_run_tree()
    return run_tree.to_headers() if run_tree is not None else {}


def _reply_text(task: dict) -> str:
    """The reply text of a completed A2A task. The Agent Server puts it in the task's artifacts."""
    text = "".join(
        p.get("text", "") for a in task.get("artifacts") or () for p in a.get("parts") or ()
    )
    if not text.strip():
        state = (task.get("status") or {}).get("state")
        raise A2AError(f"remote task returned no text (state {state!r})")
    return text


def _send_message(question: str, user: UserContext) -> str:
    """Send one A2A ``message/send`` and return the reply text. Raises ``A2AError``."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": question}],
                "metadata": {"user_context": user.to_dict()},
            },
            "context": user.to_dict(),  # Agent Server extension -> runtime.context
        },
    }
    headers = {**_auth_headers(), **_tracing_headers()}
    try:
        with _http_client() as client:
            response = client.post(A2A_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:  # ValueError: a non-JSON body
        raise A2AError(f"A2A request failed: {exc}") from exc
    if "error" in body:
        raise A2AError(f"JSON-RPC error: {body['error']}")
    return _reply_text(body["result"])


def _delegate(state: dict) -> dict:
    # This runs inside the orchestrator's ``task`` call, so ``get_runtime()`` returns the
    # orchestrator's runtime, which has the user context.
    try:
        user = require_user_context(get_runtime().context)
    except PermissionError:
        # Without a user we send nothing, and retrying wouldn't help.
        logger.warning("customer_success called without a user context")
        return {"messages": [AIMessage(FAILURE_MESSAGE)]}
    return {"messages": [AIMessage(_send_message(state["messages"][-1].text, user))]}


async def _adelegate(state: dict) -> dict:
    return await asyncio.to_thread(_delegate, state)  # copies the context, so get_runtime() works


def customer_success_subagent() -> CompiledSubAgent:
    return {
        "name": "customer_success",
        "description": (
            "Customer Success team's agent (any stack, reached over A2A). Answers questions "
            "about training and enablement for the user's purchased products. Pass a "
            "self-contained question."
        ),
        "runnable": RunnableLambda(_delegate, afunc=_adelegate, name="customer_success").with_retry(
            retry_if_exception_type=(A2AError,), stop_after_attempt=2, wait_exponential_jitter=False
        ),
    }


def fetch_agent_card() -> dict:
    """Fetch the Customer Success agent card over A2A."""
    with _http_client() as client:
        response = client.get(f"{A2A_ENDPOINT}/.well-known/agent-card.json", headers=_auth_headers())
    response.raise_for_status()
    return response.json()
