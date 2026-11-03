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
from typing import Any

import httpx
from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langgraph.config import get_config
from langgraph.pregel.remote import RemoteGraph
from langgraph.runtime import get_runtime
from langgraph.types import Command, interrupt
from langgraph_sdk.errors import NotFoundError
from langsmith.run_helpers import get_current_run_tree

from community_assistant.context import UserContext, require_user_context
from community_assistant.localization import failure_message

logger = logging.getLogger(__name__)

SUPPORT_CASE_URL = os.environ.get("SUPPORT_CASE_URL", "http://127.0.0.1:2025")

_REMOTE_THREAD_NAMESPACE = uuid.UUID("6b1f3c2e-5a4d-4e8b-9c7a-2f0e1d3b4a59")
_HISTORY_LIMIT = 200  # the server's default page (10) is too short to replay several approvals


class RemoteRunIncomplete(RuntimeError):
    """The remote thread holds a run that stopped before it finished."""


def remote_thread_id(parent_thread_id: str, delegation_ns: str) -> str:
    """The Support Case thread for one delegation (one ``task`` call), UUID-shaped."""
    key = f"{parent_thread_id}|{delegation_ns}|support_case"
    return str(uuid.uuid5(_REMOTE_THREAD_NAMESPACE, key))


# ---------------------------------------------------------------------------
# Support Case: RemoteGraph
# ---------------------------------------------------------------------------


def _ambient_config() -> dict:
    try:
        return get_config()
    except RuntimeError:  # called outside a graph run
        return {}


class ContextForwardingRemoteGraph(RemoteGraph):
    """A RemoteGraph that passes the user's identity and approvals across the network.

    Identity: the runtime context doesn't cross the network by itself. The ``task`` tool
    calls ``invoke(state, config)`` without ``context``, so the remote run wouldn't know
    who the user is. This class sends the user context explicitly, and the remote server
    applies its own access rules to it. The Agent Server rejects a request that has both
    ``configurable`` and ``context``, so the ``configurable`` keys from ``task`` are moved
    into metadata.

    Approvals: with a plain RemoteGraph as the subagent, the remote ``create_case`` approval
    couldn't make it back through ``task``. ``task`` passes no ``thread_id``, so the remote
    run has no thread, and no ``checkpoint_ns``, so RemoteGraph returns the remote interrupt
    as data instead of raising it. Resuming would then send a new message rather than
    ``Command(resume)``. So this class manages the remote thread itself:

    * Each delegation gets its own remote thread, based on the parent ``thread_id`` and the
      ``checkpoint_ns`` of the ``task`` call (``tools:<task id>``). We checked that the ns
      stays the same when that task resumes, changes for each later delegation, and differs
      between parallel ``task`` calls. So old history and old approvals never carry over,
      and parallel calls don't share a thread.
    * Each remote interrupt is raised again in the orchestrator with ``interrupt()``. On
      resume, ``task`` runs again. The remote thread is still paused, so nothing new is
      sent, and the user's decision goes to that thread as ``Command(resume=...)``. If the
      remote agent pauses again (for example, it proposes a new case after a rejection),
      that approval is raised too. LangGraph matches resume values to ``interrupt()`` calls
      in order, so each re-run first replays the remote interrupts that were already
      answered, read from the remote thread's history.
    * Only messages from the current remote turn are returned.
    * If a remote thread has nothing pending, its last turn should have finished. If it
      didn't (the earlier remote run failed partway), this raises ``RemoteRunIncomplete``,
      so the user gets the localized failure message instead of an empty answer.

    Only ``invoke`` and ``ainvoke`` are overridden, since those are what ``task`` calls.
    Calling ``.stream()`` directly behaves like a normal RemoteGraph.
    """

    def _prepare(self, config, context):
        parent = _ambient_config().get("configurable", {})
        meta = {**(config or {}).get("configurable", {}), **(config or {}).get("metadata", {})}
        if parent.get("thread_id"):
            meta["orchestrator_thread_id"] = parent["thread_id"]  # also lands on the remote thread
            thread_id = remote_thread_id(parent["thread_id"], parent.get("checkpoint_ns", ""))
        else:
            thread_id = str(uuid.uuid4())
        # configurable keeps only thread_id, which RemoteGraph pops into the request URL.
        cfg = {"metadata": meta, "configurable": {"thread_id": thread_id}}
        if context is None:
            context = require_user_context(get_runtime().context).to_dict()
        return cfg, context

    @staticmethod
    def _log(history) -> tuple[list, list, dict | None]:
        """(answered interrupt values oldest first, pending values, latest state values)."""
        if not history:
            return [], [], None
        pending_ids = {i.id for i in history[0].interrupts}
        seen, answered = set(), []
        for snapshot in reversed(history):
            for i in snapshot.interrupts:
                if i.id not in seen and i.id not in pending_ids:
                    seen.add(i.id)
                    answered.append(i.value)
        return answered, [i.value for i in history[0].interrupts], history[0].values

    @classmethod
    def _require_finished(cls, latest) -> None:
        """Raise unless the remote thread's latest turn ran to the end with a reply."""
        messages = cls._result(latest.values or {})["messages"]
        replied = any(m.type == "ai" and m.text.strip() for m in messages)
        if latest.next or not replied:
            raise RemoteRunIncomplete(f"remote run stopped before finishing (next={latest.next!r})")

    @staticmethod
    def _pending(out: dict) -> list:
        return [i["value"] for i in out.get("__interrupt__", [])]

    @staticmethod
    def _result(out: dict) -> dict:
        # Only return this turn's messages. The rest of the remote state stays on the remote side.
        messages = convert_to_messages(out.get("messages", []))
        humans = [n for n, m in enumerate(messages) if m.type == "human"]
        return {"messages": messages[humans[-1] + 1 :] if humans else messages}

    def invoke(self, input, config=None, *, context=None, **kwargs):
        cfg, context = self._prepare(config, context)
        try:
            history = list(self.get_state_history(cfg, limit=_HISTORY_LIMIT))
        except NotFoundError:  # a new delegation
            history = []
        answered, pending, out = self._log(history)
        if out is None:
            out = super().invoke({"messages": input["messages"]}, cfg, context=context, **kwargs)
            pending = self._pending(out)
        elif not pending:
            self._require_finished(history[0])
        for value in answered:
            interrupt(value)  # replay: returns the decision already applied remotely
        while pending:
            decision = interrupt(pending[0])  # pauses the orchestrator until the user answers
            out = super().invoke(Command(resume=decision), cfg, context=context, **kwargs)
            pending = self._pending(out)
        return self._result(out)

    async def ainvoke(self, input, config=None, *, context=None, **kwargs):
        cfg, context = self._prepare(config, context)
        try:
            history = [s async for s in self.aget_state_history(cfg, limit=_HISTORY_LIMIT)]
        except NotFoundError:
            history = []
        answered, pending, out = self._log(history)
        if out is None:
            out = await super().ainvoke({"messages": input["messages"]}, cfg, context=context, **kwargs)
            pending = self._pending(out)
        elif not pending:
            self._require_finished(history[0])
        for value in answered:
            interrupt(value)
        while pending:
            decision = interrupt(pending[0])
            out = await super().ainvoke(Command(resume=decision), cfg, context=context, **kwargs)
            pending = self._pending(out)
        return self._result(out)


def support_case_subagent() -> CompiledSubAgent:
    return {
        "name": "support_case",
        "description": "Support team's agent (separate deployment). Looks up case status and creates support cases.",
        "runnable": ContextForwardingRemoteGraph(
            "support_case",
            url=SUPPORT_CASE_URL,
            api_key=os.environ.get("LANGSMITH_API_KEY"),
            distributed_tracing=True,
        ),
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
# Endpoint: ``CUSTOMER_SUCCESS_A2A_URL`` (read on each call) is the agent's full A2A
# endpoint and is used as is. Set it for a real A2A agent (the URL is in its agent card
# or docs) or for a non-default assistant on LangSmith Deployment.
#
# If it isn't set, we build ``{CUSTOMER_SUCCESS_URL}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}``.
# That's a shortcut for local dev. The Agent Server addresses A2A agents by assistant
# UUID, not graph name (``/a2a/customer_success`` is rejected with "Invalid assistant ID:
# must be a UUID"), and a graph's default assistant ID is the uuid5 of the graph name in
# the server's graph namespace.
#
# Agent card: ``CUSTOMER_SUCCESS_AGENT_CARD_URL`` (read on each call) if it's set,
# otherwise the per-assistant card at ``{endpoint}/.well-known/agent-card.json``. The
# Agent Server's root ``/.well-known/agent-card.json`` needs an ``assistant_id`` query
# parameter, so we don't use it as the default.
_AGENT_SERVER_GRAPH_NAMESPACE = uuid.UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
CUSTOMER_SUCCESS_AGENT_ID = str(uuid.uuid5(_AGENT_SERVER_GRAPH_NAMESPACE, "customer_success"))
A2A_TIMEOUT = httpx.Timeout(60.0, connect=5.0)  # a remote agent turn can include model calls

_FAILED_TASK_STATES = {
    *("failed", "rejected", "canceled"),  # A2A spec names
    *("TASK_STATE_FAILED", "TASK_STATE_REJECTED", "TASK_STATE_CANCELED"),  # v1.0 enum names
}
_AGENT_ROLES = {"agent", "ROLE_AGENT"}


class A2AError(RuntimeError):
    """The A2A call failed: a network error, a JSON-RPC error, or a reply with no text.

    The subagent retries once when this happens. If it fails again, the error comes out of
    the ``task`` call and the orchestrator's tool middleware turns it into the localized
    failure message.
    """


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=A2A_TIMEOUT)


def _a2a_endpoint() -> str:
    if endpoint := os.environ.get("CUSTOMER_SUCCESS_A2A_URL"):
        return endpoint
    return f"{CUSTOMER_SUCCESS_URL.rstrip('/')}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"


def _agent_card_url() -> str:
    if url := os.environ.get("CUSTOMER_SUCCESS_AGENT_CARD_URL"):
        return url
    return f"{_a2a_endpoint().rstrip('/')}/.well-known/agent-card.json"


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


def _text(parts) -> str:
    return "".join(p.get("text", "") for p in parts or () if p.get("kind", "text") == "text")


def _reply_text(task: dict) -> str:
    """Text of the agent's final message in the task.

    A completed task normally has the agent's reply as its artifact (A2A section 3.7).
    This also handles agents that reply in the status message, only in the history, or
    with a plain Message instead of a Task.
    """
    if task.get("kind") == "message":
        candidates = [_text(task.get("parts"))]
    else:
        status = task.get("status") or {}
        if status.get("state") in _FAILED_TASK_STATES:
            raise A2AError(f"remote task ended in state {status.get('state')!r}")
        candidates = [_text(a.get("parts")) for a in task.get("artifacts") or ()]
        status_message = status.get("message") or {}
        if status_message.get("role") in _AGENT_ROLES:
            candidates.append(_text(status_message.get("parts")))
        if not any(candidates):
            history = task.get("history") or ()
            candidates = [_text(m.get("parts")) for m in history if m.get("role") in _AGENT_ROLES]
    text = next((t for t in reversed(candidates) if t.strip()), "")
    if not text:
        raise A2AError("remote task returned no text")
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
            response = client.post(_a2a_endpoint(), json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:  # ValueError: a non-JSON body
        raise A2AError(f"A2A request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise A2AError(f"JSON-RPC response is not an object: {type(body).__name__}")
    if "error" in body:
        raise A2AError(f"JSON-RPC error: {body['error']}")
    result = body.get("result")
    task = result.get("task", result) if isinstance(result, dict) else result  # some wrap it
    if not isinstance(task, dict):
        raise A2AError(f"JSON-RPC result is not a Task or Message: {type(task).__name__}")
    return _reply_text(task)


def ask_customer_success(question: str, context: Any) -> str:
    """Ask the Customer Success agent over A2A on behalf of the user in ``context``."""
    try:
        user = require_user_context(context)
    except PermissionError:
        # Without a user we send nothing, and retrying wouldn't help.
        logger.warning("customer_success called without a user context")
        return failure_message("en_US")
    return _send_message(question, user)  # raises A2AError


def _delegate(state: dict) -> dict:
    # This runs inside the orchestrator's ``task`` call, so ``get_runtime()`` returns the
    # orchestrator's runtime, which has the user context.
    question = state["messages"][-1].text
    return {"messages": [AIMessage(ask_customer_success(question, get_runtime().context))]}


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
        response = client.get(_agent_card_url(), headers=_auth_headers())
    response.raise_for_status()
    return response.json()
