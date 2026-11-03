"""Orchestrator middleware.

In a single flat graph, these checks are usually separate nodes. Here they're middleware:

* the ``safety_check`` node becomes ``SafetyMiddleware``
* ``broadness_check`` and its follow-up become ``BroadnessMiddleware``, which uses ``interrupt``
* retry-once-then-error becomes ``FriendlyToolRetryMiddleware``

Pass ``orchestrator_middleware()`` to ``create_deep_agent(middleware=...)``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

FAILURE_MESSAGE = "Sorry, something went wrong on our side. Please try again in a moment."
REFUSAL_MESSAGE = "I can't help with that request. If you need support, I can help you open a case."

# Tools that are safe to call twice. ``task`` isn't one of them: retrying it would run
# the whole subagent again, including things like an approval or a new case. A subagent
# can retry internally when that's safe. Errors from other tools still become
# FAILURE_MESSAGE, they just aren't retried.
RETRIED_TOOLS = ("flaky_community_search",)

UNSAFE_PATTERNS = ("ignore previous instructions", "social security number", "how do i hack")
BROAD_TOPICS = ("payroll", "benefits", "hcm", "financials", "acme")
QUESTION_WORDS = ("how", "what", "why", "where", "when", "can", "does", "is")
CLARIFICATION_QUESTION = (
    "Could you tell me a bit more? For example, which product area and what you're trying to do?"
)


def is_unsafe(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in UNSAFE_PATTERNS)


def is_broad(text: str) -> bool:
    """A short topic-only message (e.g. "payroll") with no question word."""
    words = [w.strip("?.,!:;").lower() for w in text.split()]
    if not words or len(words) > 4:
        return False
    if any(w in QUESTION_WORDS for w in words):
        return False
    return any(t in words for t in BROAD_TOPICS)


def _last_human_text(state: dict[str, Any]) -> str | None:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.text
    return None


def _refusal() -> dict[str, Any]:
    return {"messages": [AIMessage(REFUSAL_MESSAGE)], "jump_to": "end"}


class SafetyMiddleware(AgentMiddleware):
    """Replaces the ``safety_check`` node. Checks the input before the run and the answer after it."""

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        text = _last_human_text(state)
        if text is not None and is_unsafe(text):
            return _refusal()
        return None

    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and is_unsafe(last.text):
            # Same id -> the add_messages reducer replaces the unsafe message.
            return {"messages": [AIMessage(REFUSAL_MESSAGE, id=last.id)]}
        return None


class BroadnessMiddleware(AgentMiddleware):
    """Replaces ``broadness_check`` and its follow-up node.

    If the request is too vague, it pauses the run with ``interrupt`` and asks for more
    detail. When the run resumes, LangGraph runs this hook again and ``interrupt`` returns
    the user's reply. The reply is added to the conversation only once, because the first
    pass never returned a state update.
    """

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        text = _last_human_text(state)
        if text is None or not is_broad(text):
            return None
        reply = interrupt({"type": "clarification", "question": CLARIFICATION_QUESTION})
        reply_text = str(reply)
        if is_unsafe(reply_text):
            return _refusal()
        return {"messages": [HumanMessage(reply_text)]}


class FriendlyToolRetryMiddleware(ToolRetryMiddleware):
    """Retries the tools in ``tools=[...]``, and turns any tool's final error into ``FAILURE_MESSAGE``.

    The built-in ``ToolRetryMiddleware`` only handles the tools it retries, so an error from
    any other tool (such as a failed ``task`` call to a remote subagent) would end the run
    with the raw error. This subclass catches those too, so the user never sees one.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(on_failure="error", **kwargs)

    def _to_message(self, exc: Exception, request: Any) -> ToolMessage:
        logger.exception("tool call failed", exc_info=exc)
        return ToolMessage(
            content=FAILURE_MESSAGE,
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    def wrap_tool_call(self, request, handler) -> ToolMessage | Command[Any]:
        try:
            return super().wrap_tool_call(request, handler)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._to_message(exc, request)

    async def awrap_tool_call(
        self, request, handler: Callable[..., Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        try:
            return await super().awrap_tool_call(request, handler)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._to_message(exc, request)


def orchestrator_middleware() -> list[AgentMiddleware]:
    return [
        SafetyMiddleware(),
        BroadnessMiddleware(),
        FriendlyToolRetryMiddleware(tools=list(RETRIED_TOOLS), max_retries=1, initial_delay=0.2),
    ]
