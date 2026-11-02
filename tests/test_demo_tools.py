"""The demo-only flaky search tool, alone and behind the front door's retry middleware."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from community_assistant.context import DEMO_USERS, UserContext
from community_assistant.front_door import demo_tools
from community_assistant.front_door.demo_tools import (
    CANNED_RESULT,
    CommunitySearchUnavailable,
    flaky_community_search,
)
from community_assistant.front_door.middleware import front_door_middleware
from community_assistant.localization import FAILURE_COPY


@pytest.fixture(autouse=True)
def _fresh_process():
    demo_tools.reset_flaky_state()
    yield
    demo_tools.reset_flaky_state()


def _search(query: str, thread_id: str | None = None) -> str:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return flaky_community_search.invoke({"query": query}, config)


def test_plain_query_returns_canned_result():
    assert _search("How do I run payroll?") == CANNED_RESULT
    assert _search("How do I run payroll?") == CANNED_RESULT


def test_flaky_fails_only_on_first_call_per_process():
    with pytest.raises(CommunitySearchUnavailable):
        _search("payroll tips [flaky]")
    assert _search("payroll tips [flaky]") == CANNED_RESULT
    assert _search("other [flaky] query") == CANNED_RESULT  # still recovered


def test_flaky_fails_once_per_conversation():
    for thread in ("demo-1", "demo-2"):  # every new demo thread shows the retry again
        with pytest.raises(CommunitySearchUnavailable):
            _search("payroll tips [flaky]", thread)
        assert _search("payroll tips [flaky]", thread) == CANNED_RESULT
    # Outside a run (no thread_id) the state is process-wide, separate from threads.
    with pytest.raises(CommunitySearchUnavailable):
        _search("payroll tips [flaky]")
    assert _search("payroll tips [flaky]") == CANNED_RESULT
    assert _search("payroll tips [flaky]", "demo-1") == CANNED_RESULT


def test_down_always_fails():
    for _ in range(3):
        with pytest.raises(CommunitySearchUnavailable):
            _search("payroll tips [down]")


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _via_front_door(query: str, user: str, thread_id: str = "t") -> ToolMessage:
    call = AIMessage(
        "", tool_calls=[{"name": "flaky_community_search", "args": {"query": query}, "id": "c1"}]
    )
    agent = create_agent(
        FakeModel(messages=iter([call, AIMessage("done")])),
        tools=[flaky_community_search],
        middleware=front_door_middleware(),
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
    )
    out = agent.invoke(
        {"messages": [HumanMessage(f"How do I search for {query}?")]},
        {"configurable": {"thread_id": thread_id}},
        context=DEMO_USERS[user],
    )
    return next(m for m in out["messages"] if isinstance(m, ToolMessage))


def test_flaky_recovers_through_retry_middleware():
    tm = _via_front_door("payroll tips [flaky]", "admin_customer")
    assert tm.content == CANNED_RESULT
    assert tm.status != "error"


def test_each_new_thread_fails_once_then_recovers_through_retry(monkeypatch):
    calls: list[str] = []
    original = demo_tools._flaky_key
    monkeypatch.setattr(demo_tools, "_flaky_key", lambda: calls.append(original()) or calls[-1])
    for thread in ("stage-1", "stage-2"):
        tm = _via_front_door("payroll tips [flaky]", "admin_customer", thread)
        assert tm.content == CANNED_RESULT
    # Two attempts (fail, retry) per thread: the first call in each thread failed again.
    assert calls == ["stage-1", "stage-1", "stage-2", "stage-2"]


def test_down_gives_french_failure_copy_through_retry_middleware():
    tm = _via_front_door("payroll tips [down]", "fr_customer")
    assert tm.content == FAILURE_COPY["fr_FR"]
    assert tm.status == "error"
