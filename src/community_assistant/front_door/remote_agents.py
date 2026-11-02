"""Agents owned by other teams, reached over the network.

* Support Case runs on its own Agent Server (``langgraph.support_case.json``) and
  is reached through ``RemoteGraph``. Moving it to LangSmith Deployment needs only
  ``SUPPORT_CASE_URL`` and ``LANGSMITH_API_KEY``.
* Customer Success is reached only through the A2A protocol (``ask_customer_success``
  and ``fetch_agent_card``), so it can be any A2A-compliant agent, on any stack.
"""

from __future__ import annotations

import logging
import os
import uuid

import httpx
from deepagents import CompiledSubAgent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import convert_to_messages
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
    """RemoteGraph that carries identity and approvals across the network boundary.

    Identity: the parent's runtime context does not travel over the network on its
    own. A deep agent's ``task`` tool calls ``invoke(state, config)`` without
    ``context``, so the remote run would see no user. The user context is forwarded
    here, explicitly and by design. The remote server then enforces its own access
    rules on that context. The Agent Server rejects requests that set both
    ``configurable`` and ``context``, so the task tool's ``configurable`` keys are
    moved into metadata.

    Approvals: we tried the plain RemoteGraph-as-subagent path first, and the remote
    ``create_case`` approval could not round-trip through ``task``. The task tool
    passes no ``thread_id`` (so the remote run is stateless) and no
    ``checkpoint_ns``, so RemoteGraph returns the remote interrupt as data instead
    of raising it. A resume would then send a fresh message, not ``Command(resume)``.
    So this runnable manages the remote thread itself and resumes it with
    ``Command(resume=...)``, and Support Case stays a ``CompiledSubAgent``:

    * One remote thread per delegation, keyed by the parent ``thread_id`` plus the
      ambient ``checkpoint_ns`` of the ``task`` tool call (``tools:<task id>``).
      Verified empirically: the ns is identical when that same task is resumed,
      new for every later delegation, and distinct for parallel ``task`` calls. So
      no stale history or stale approvals carry over, and parallel calls don't
      collide on one remote thread.
    * Each remote interrupt is re-raised in the front door with ``interrupt()``.
      On resume, ``task`` runs again: the remote thread is still paused, so nothing
      new is sent, and the user's decision goes as ``Command(resume=...)`` to that
      same thread. If the remote pauses again (e.g. it re-proposes after a reject),
      that approval is raised too. LangGraph matches resume values to
      ``interrupt()`` calls by order, so on each re-run the remote interrupts
      already answered (read from the remote thread's history) are replayed first.
    * Only messages from the current remote turn are returned.
    * A remote thread found with nothing pending must hold a finished turn. If it
      doesn't (the earlier remote run failed part-way), ``RemoteRunIncomplete`` is
      raised, so the front door reports a localized failure instead of an empty answer.

    Only ``invoke``/``ainvoke`` are overridden: they are what ``task`` calls. A
    direct ``.stream()`` gets plain RemoteGraph behaviour.
    """

    def _prepare(self, config, context):
        parent = _ambient_config().get("configurable", {})
        meta = {**(config or {}).get("configurable", {}), **(config or {}).get("metadata", {})}
        if parent.get("thread_id"):
            meta["front_door_thread_id"] = parent["thread_id"]  # also lands on the remote thread
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
        # Only this turn's conversation crosses back; other remote state stays private.
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
            decision = interrupt(pending[0])  # pauses the front door until the user answers
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
# A2A JSON-RPC ``message/send`` over HTTP, so the remote side can be any
# A2A-compliant agent. The user context is sent twice:
#
# * ``message.metadata["user_context"]``: standard A2A, for any agent on any stack.
# * ``params.context``: an Agent Server extension to A2A (not in the A2A spec). The
#   Agent Server maps it to the run's context, so it arrives as ``runtime.context``
#   in the remote tools. Other A2A agents ignore it.
# ---------------------------------------------------------------------------

CUSTOMER_SUCCESS_URL = os.environ.get("CUSTOMER_SUCCESS_URL", "http://127.0.0.1:2026")
# Endpoint: ``CUSTOMER_SUCCESS_A2A_URL`` (read per call) is the agent's full A2A
# endpoint, used as is. Set it for any real A2A agent (take the URL from its agent
# card or docs) and for a non-default assistant on LangSmith Deployment.
#
# Unset, ``{CUSTOMER_SUCCESS_URL}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}`` is computed. That
# is a local-dev convenience that relies on how the Agent Server derives the
# default assistant's id: it addresses A2A agents by assistant UUID, not graph id
# (verified: ``/a2a/customer_success`` is rejected with "Invalid assistant ID: must
# be a UUID"), and a graph's default assistant is uuid5 of the graph id in the
# server's graph namespace.
#
# Agent card: ``CUSTOMER_SUCCESS_AGENT_CARD_URL`` (read per call) if set, else the
# Agent Server's per-assistant card at ``{endpoint}/.well-known/agent-card.json``.
# (The Agent Server's root ``/.well-known/agent-card.json`` needs an ``assistant_id``
# query parameter, so it is not used as the default.)
_AGENT_SERVER_GRAPH_NAMESPACE = uuid.UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")
CUSTOMER_SUCCESS_AGENT_ID = str(uuid.uuid5(_AGENT_SERVER_GRAPH_NAMESPACE, "customer_success"))
A2A_TIMEOUT = httpx.Timeout(60.0, connect=5.0)  # a remote agent turn can include model calls

_FAILED_TASK_STATES = {
    *("failed", "rejected", "canceled"),  # A2A spec names
    *("TASK_STATE_FAILED", "TASK_STATE_REJECTED", "TASK_STATE_CANCELED"),  # v1.0 enum names
}
_AGENT_ROLES = {"agent", "ROLE_AGENT"}


class A2AError(RuntimeError):
    """The A2A call failed: transport error, JSON-RPC error, or no usable reply.

    The tool raises it, so the front door's tool-retry middleware retries the call
    once and then returns the localized failure message.
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

    Deliberately not defaulted to LANGSMITH_API_KEY: the A2A endpoint may belong to a third
    party, and a key must never go to an endpoint it wasn't meant for.
    """
    key = os.environ.get("CUSTOMER_SUCCESS_API_KEY")
    return {"x-api-key": key} if key else {}


def _tracing_headers() -> dict[str, str]:
    """LangSmith distributed-tracing headers for the current run, if one is active.

    The Agent Server copies ``langsmith-trace`` (and ``baggage``) from A2A requests
    into the run's config, as on its other run routes. The receiving graph turns
    that back into the trace parent (``community_assistant.tracing``), which nests
    the remote run under this tool call in LangSmith. Other A2A agents ignore
    headers they don't know.

    They go to whatever endpoint is configured. The trace id is no secret, but the
    ``baggage`` header names your LangSmith project: drop these headers if the A2A
    agent belongs to a third party you don't want to see that.
    """
    run_tree = get_current_run_tree()
    return run_tree.to_headers() if run_tree is not None else {}


def _text(parts) -> str:
    return "".join(p.get("text", "") for p in parts or () if p.get("kind", "text") == "text")


def _reply_text(task: dict) -> str:
    """Text of the task's final agent message.

    A completed task carries the agent's final reply as its response artifact
    (A2A section 3.7). Agents that answer in the status message, only in history,
    or with a bare Message instead of a Task are covered too.
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


@tool
def ask_customer_success(question: str, runtime: ToolRuntime[UserContext]) -> str:
    """Ask the Customer Success team's agent about training and enablement for the
    user's purchased products. It is a separate system; pass a self-contained question."""
    try:
        user = require_user_context(runtime.context)
    except PermissionError:
        # Fail closed: nothing is sent without an identity, and a retry can't fix that.
        logger.warning("ask_customer_success called without a user context")
        return failure_message("en_US")
    # An A2AError propagates: the tool-retry middleware retries, then localizes it.
    return _send_message(question, user)


def fetch_agent_card() -> dict:
    """Discover the Customer Success agent: GET its A2A agent card."""
    with _http_client() as client:
        response = client.get(_agent_card_url(), headers=_auth_headers())
    response.raise_for_status()
    return response.json()
