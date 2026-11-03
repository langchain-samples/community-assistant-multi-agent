"""Front door middleware.

In a single flat graph, these checks are usually separate nodes. Here they're middleware:

* the ``safety_check`` node becomes ``SafetyMiddleware``
* ``broadness_check`` and its follow-up become ``BroadnessMiddleware``, which uses ``interrupt``
* retry-once-then-error becomes ``LocalizedToolRetryMiddleware``

``user_context_prompt`` isn't a check. It just tells the model the user's name and locale.

Pass ``front_door_middleware()`` to ``create_deep_agent(middleware=...)``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ToolRetryMiddleware,
    dynamic_prompt,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt

from community_assistant.context import require_user_context
from community_assistant.localization import REFUSAL_COPY, failure_message

logger = logging.getLogger(__name__)

# Tools that are safe to call twice. ``task`` isn't one of them: retrying it would run
# the whole subagent again, including things like an approval or a new case. A subagent
# can retry internally when that's safe. Errors from other tools still become the
# localized failure message, they just aren't retried.
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
    return {"messages": [AIMessage(REFUSAL_COPY)], "jump_to": "end"}


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
            return {"messages": [AIMessage(REFUSAL_COPY, id=last.id)]}
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


def _locale(request: Any) -> str:
    ctx = getattr(getattr(request, "runtime", None), "context", None)
    try:
        return require_user_context(ctx).locale
    except PermissionError:
        return "en_US"


def localized_tool_error(exc: Exception, request: Any) -> str:
    """Log ``exc`` and return the hard-failure copy in the user's locale."""
    logger.exception("tool call failed", exc_info=exc)
    return failure_message(_locale(request))


class LocalizedToolRetryMiddleware(ToolRetryMiddleware):
    """Retries a failing tool, then returns a ``ToolMessage`` in the user's language.

    ``ToolRetryMiddleware(on_failure=callable)`` only gets the exception, so it can't see
    the user's locale in ``request.runtime.context``. This subclass keeps the built-in
    retry loop (``on_failure="error"`` re-raises after the last attempt) and converts the
    final error here, where the request is available. Tools that aren't in ``tools=[...]``
    aren't retried, but their errors are converted the same way.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(on_failure="error", **kwargs)

    def _to_message(self, exc: Exception, request: Any) -> ToolMessage:
        return ToolMessage(
            content=localized_tool_error(exc, request),
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


@dynamic_prompt
def user_context_prompt(request: ModelRequest) -> str:
    """Adds the user's name and locale to the system prompt, and asks the model to reply in that language."""
    base = request.system_message.text if request.system_message else ""
    try:
        user = require_user_context(request.runtime.context)
    except PermissionError:
        return base
    return (
        f"{base}\n\nCurrent user: {user.display_name}. Locale: {user.locale}. "
        "Reply in this locale's language."
    )


def front_door_middleware() -> list[AgentMiddleware]:
    return [
        user_context_prompt,
        SafetyMiddleware(),
        BroadnessMiddleware(),
        LocalizedToolRetryMiddleware(tools=list(RETRIED_TOOLS), max_retries=1, initial_delay=0.2),
    ]
