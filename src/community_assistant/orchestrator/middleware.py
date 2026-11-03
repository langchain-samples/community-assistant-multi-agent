"""Orchestrator middleware.

In a single flat graph, these checks are usually separate nodes. Here they're middleware:

* the ``safety_check`` node becomes ``SafetyMiddleware``
* ``broadness_check`` and its follow-up become ``BroadnessMiddleware``, which uses ``interrupt``
* retry-once-then-error becomes the built-in ``ToolRetryMiddleware`` and ``ToolErrorMiddleware``

Pass ``orchestrator_middleware()`` to ``create_deep_agent(middleware=...)``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

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


def friendly_tool_error(exc: Exception, request: Any) -> str:
    """Log a tool's final error and give the model ``FAILURE_MESSAGE`` instead of the raw error."""
    logger.exception("tool call failed", exc_info=exc)
    return FAILURE_MESSAGE


def orchestrator_middleware() -> list[AgentMiddleware]:
    return [
        SafetyMiddleware(),
        BroadnessMiddleware(),
        # Outer: any tool's error, including a failed ``task`` call, becomes FAILURE_MESSAGE.
        ToolErrorMiddleware(on_error=friendly_tool_error),
        # Inner: retry the safe tools once, then pass the error out to ToolErrorMiddleware.
        ToolRetryMiddleware(
            tools=list(RETRIED_TOOLS), max_retries=1, initial_delay=0.2, on_failure="error"
        ),
    ]
