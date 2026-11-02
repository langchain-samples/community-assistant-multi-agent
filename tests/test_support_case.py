from __future__ import annotations

import warnings

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from conftest import ToolCallingFake
from community_assistant.context import DEMO_USERS
from community_assistant.support_case import agent as sc
from community_assistant.support_case.agent import (
    DENIED_COPY,
    build_support_case_agent,
    create_case,
)

CREATE_ARGS = {
    "subject": "Payroll run failing",
    "description": "Nightly payroll run errors out.",
    "product": "Acme Payroll",
}


@pytest.fixture(autouse=True)
def _reset_cases():
    sc.reset_cases()
    yield
    sc.reset_cases()


def _agent(*responses):
    return build_support_case_agent(
        ToolCallingFake(responses=list(responses)), checkpointer=MemorySaver()
    )


def _call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _tool_text(result):
    return [m for m in result["messages"] if m.type == "tool"][-1].content


def test_get_case_status_only_own_cases():
    admin = DEMO_USERS["admin_customer"]
    other = DEMO_USERS["partner_readonly"]
    own = _agent(_call("get_case_status", {"case_id": "CS-1042"}), AIMessage(content="ok")).invoke(
        {"messages": [HumanMessage("status?")]},
        {"configurable": {"thread_id": "a"}},
        context=admin,
    )
    assert "Waiting on Acme" in _tool_text(own)
    foreign = _agent(
        _call("get_case_status", {"case_id": "CS-1042"}), AIMessage(content="ok")
    ).invoke(
        {"messages": [HumanMessage("status?")]},
        {"configurable": {"thread_id": "b"}},
        context=other,
    )
    assert _tool_text(foreign) == "No case CS-1042 found for your account."


def test_get_case_status_accepts_dict_context():
    """The Agent Server API sends context as JSON (a dict with list-valued fields).

    LangGraph coerces it to a UserContext before the tool runs (verified: tools see
    UserContext), so the tool's dict handling is only a safeguard. ``UserContext``
    stores the lists as tuples, so the coercion raises no serializer warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (e.g. Pydantic serializer) fails the test
        result = _agent(
            _call("get_case_status", {"case_id": "CS-1042"}), AIMessage(content="ok")
        ).invoke(
            {"messages": [HumanMessage("status?")]},
            {"configurable": {"thread_id": "d"}},
            context=DEMO_USERS["admin_customer"].to_dict(),
        )
    assert "Waiting on Acme" in _tool_text(result)


@pytest.mark.parametrize("ctx", [DEMO_USERS["partner_readonly"], None])
def test_create_case_denied_without_role(ctx):
    # Call the tool body directly (HITL is bypassed) to prove the tool itself fails closed.
    class R:
        context = ctx

    assert create_case.func(runtime=R(), **CREATE_ARGS) == DENIED_COPY
    assert "CS-1043" not in sc.CASES_STORE


def test_create_case_interrupts_for_approval():
    agent = _agent(_call("create_case", CREATE_ARGS), AIMessage(content="done"))
    cfg = {"configurable": {"thread_id": "i"}}
    result = agent.invoke(
        {"messages": [HumanMessage("open a case")]}, cfg, context=DEMO_USERS["admin_customer"]
    )
    assert "__interrupt__" in result
    req = result["__interrupt__"][0].value["action_requests"][0]
    assert req["name"] == "create_case"
    assert "CS-1043" not in sc.CASES_STORE


def test_create_case_approved_creates_case():
    agent = _agent(_call("create_case", CREATE_ARGS), AIMessage(content="Opened CS-1043."))
    cfg = {"configurable": {"thread_id": "p"}}
    agent.invoke(
        {"messages": [HumanMessage("open a case")]}, cfg, context=DEMO_USERS["admin_customer"]
    )
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        cfg,
        context=DEMO_USERS["admin_customer"],
    )
    assert "CS-1043" in _tool_text(result)
    assert sc.CASES_STORE["CS-1043"]["owner"] == DEMO_USERS["admin_customer"].user_id
    assert sc.CASES_STORE["CS-1043"]["subject"] == CREATE_ARGS["subject"]
    # The baseline mock case has the same shape as a created one.
    assert set(sc.CASES_STORE["CS-1042"]) == set(sc.CASES_STORE["CS-1043"])


def test_create_case_rejected_creates_nothing():
    agent = _agent(_call("create_case", CREATE_ARGS), AIMessage(content="Okay, not opened."))
    cfg = {"configurable": {"thread_id": "r"}}
    agent.invoke(
        {"messages": [HumanMessage("open a case")]}, cfg, context=DEMO_USERS["admin_customer"]
    )
    agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "no"}]}),
        cfg,
        context=DEMO_USERS["admin_customer"],
    )
    assert "CS-1043" not in sc.CASES_STORE


def test_partner_readonly_gets_no_approval_prompt_and_no_create_tool():
    offered: list[list[str]] = []

    class RecordingFake(ToolCallingFake):
        def bind_tools(self, tools, **kwargs):
            offered.append([getattr(t, "name", str(t)) for t in tools])
            return self

    model = RecordingFake(
        responses=[AIMessage(content="You don't have permission to create support cases.")]
    )
    agent = build_support_case_agent(model, checkpointer=MemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage("open a case")]},
        {"configurable": {"thread_id": "ro"}},
        context=DEMO_USERS["partner_readonly"],
    )
    assert "__interrupt__" not in result
    assert offered and all("create_case" not in names for names in offered)
    assert "get_case_status" in offered[0]
    assert len(sc.CASES_STORE) == 1


def test_admin_is_offered_create_case():
    offered: list[list[str]] = []

    class RecordingFake(ToolCallingFake):
        def bind_tools(self, tools, **kwargs):
            offered.append([getattr(t, "name", str(t)) for t in tools])
            return self

    agent = build_support_case_agent(
        RecordingFake(responses=[AIMessage(content="hi")]), checkpointer=MemorySaver()
    )
    agent.invoke(
        {"messages": [HumanMessage("hi")]},
        {"configurable": {"thread_id": "ad"}},
        context=DEMO_USERS["admin_customer"],
    )
    assert "create_case" in offered[0]


def test_partner_readonly_is_told_why_create_case_is_missing():
    prompts: list[str] = []

    class PromptRecordingFake(ToolCallingFake):
        def _generate(self, messages, *a, **kw):
            prompts.append(messages[0].content)
            return super()._generate(messages, *a, **kw)

    agent = build_support_case_agent(
        PromptRecordingFake(responses=[AIMessage(content=DENIED_COPY)]), checkpointer=MemorySaver()
    )
    agent.invoke(
        {"messages": [HumanMessage("open a case")]},
        {"configurable": {"thread_id": "ro-note"}},
        context=DEMO_USERS["partner_readonly"],
    )
    assert sc.SYSTEM_PROMPT in prompts[0]  # the agent's own prompt is kept
    assert sc.NO_CREATE_NOTE in prompts[0] and DENIED_COPY in prompts[0]


def test_admin_prompt_has_no_permission_note():
    prompts: list[str] = []

    class PromptRecordingFake(ToolCallingFake):
        def _generate(self, messages, *a, **kw):
            prompts.append(messages[0].content)
            return super()._generate(messages, *a, **kw)

    agent = build_support_case_agent(
        PromptRecordingFake(responses=[AIMessage(content="hi")]), checkpointer=MemorySaver()
    )
    agent.invoke(
        {"messages": [HumanMessage("hi")]},
        {"configurable": {"thread_id": "ad-note"}},
        context=DEMO_USERS["admin_customer"],
    )
    assert sc.NO_CREATE_NOTE not in prompts[0]
