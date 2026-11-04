"""Subagents owned by other teams, called over the network.

* Support Case runs on its own Agent Server (``langgraph.support_case.json``) and is
  called with ``RemoteGraph``. To move it to LangSmith Deployment you only need to set
  ``SUPPORT_CASE_URL`` and ``LANGSMITH_API_KEY``.
* Customer Success is only called over A2A, so it could be any agent that supports A2A.

The orchestrator calls both through ``task``, like any local subagent. Only the runnable
behind each one knows how to reach it.
"""

from __future__ import annotations

import os
import uuid

import httpx
from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langgraph.pregel.remote import RemoteGraph
from langgraph.runtime import get_runtime
from langsmith.run_helpers import get_current_run_tree

from community_assistant.context import require_user_context


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
# ---------------------------------------------------------------------------

CUSTOMER_SUCCESS_URL = os.environ.get("CUSTOMER_SUCCESS_URL", "http://127.0.0.1:2026")
# The Agent Server addresses A2A agents by assistant ID, not graph name. A graph's default
# assistant ID is the uuid5 of the graph name in the server's namespace.
_AGENT_SERVER_GRAPH_NAMESPACE = uuid.UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
CUSTOMER_SUCCESS_AGENT_ID = str(uuid.uuid5(_AGENT_SERVER_GRAPH_NAMESPACE, "customer_success"))
A2A_ENDPOINT = f"{CUSTOMER_SUCCESS_URL.rstrip('/')}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"


def _headers() -> dict[str, str]:
    """An API key if CUSTOMER_SUCCESS_API_KEY is set, plus LangSmith trace headers."""
    headers = {}
    if key := os.environ.get("CUSTOMER_SUCCESS_API_KEY"):
        headers["x-api-key"] = key
    if run := get_current_run_tree():
        headers.update(run.to_headers())
    return headers


async def _call_customer_success(state: dict) -> dict:
    """Send the delegated request to Customer Success as one A2A message and return its reply."""
    user = require_user_context(get_runtime().context)
    request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": state["messages"][-1].text}],
                "metadata": {"user_context": user.to_dict()},  # standard A2A: any agent can read it
            },
            "context": user.to_dict(),  # Agent Server addition: becomes the remote runtime.context
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(A2A_ENDPOINT, json=request, headers=_headers())
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"A2A error: {body['error']}")
    task = body["result"]  # an A2A task; the reply is in its artifacts
    reply = "".join(p.get("text", "") for a in task.get("artifacts", []) for p in a["parts"])
    if not reply.strip():
        raise RuntimeError(f"A2A task returned no reply (state {task.get('status', {}).get('state')!r})")
    return {"messages": [AIMessage(reply)]}


def customer_success_subagent() -> CompiledSubAgent:
    return {
        "name": "customer_success",
        "description": (
            "Customer Success team's agent (any stack, reached over A2A). Answers questions "
            "about training and enablement for the user's purchased products. Pass a "
            "self-contained question."
        ),
        "runnable": RunnableLambda(_call_customer_success, name="customer_success"),
    }


def fetch_agent_card() -> dict:
    """Fetch the Customer Success agent card, the standard A2A description of the agent."""
    response = httpx.get(f"{A2A_ENDPOINT}/.well-known/agent-card.json", headers=_headers(), timeout=10)
    response.raise_for_status()
    return response.json()
