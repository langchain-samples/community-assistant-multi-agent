from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage

from conftest import ToolCallingFake
from community_assistant.context import DEMO_USERS, UserContext
from community_assistant.front_door.knowledge import _last_user_text, build_knowledge_graph, retrieve

PAYROLL_Q = "How do I pay out accrued time off at termination in payroll?"


class RecordingModel(GenericFakeChatModel):
    """Fake model that records every prompt it is given."""

    seen: list = []

    def _generate(self, messages, *a, **kw):
        self.seen.append(messages)
        return super()._generate(messages, *a, **kw)


def _answer_model(text: str = "Here is the answer.") -> RecordingModel:
    return RecordingModel(messages=iter([AIMessage(content=text)]), seen=[])


class RaisingModel(GenericFakeChatModel):
    def _generate(self, *a, **kw):
        raise AssertionError("model must not be called")


def _run(model, ctx: UserContext, question: str) -> dict[str, Any]:
    graph = build_knowledge_graph(model)
    return graph.invoke({"messages": [HumanMessage(question)], "docs": []}, context=ctx)


def test_partner_does_not_see_payroll_docs():
    out = _run(_answer_model(), DEMO_USERS["partner_readonly"], PAYROLL_Q)
    assert out["docs"], "partner should still see HCM docs"
    assert all(d["entitlement"] != "Payroll" for d in out["docs"])


def test_admin_sees_payroll_docs():
    out = _run(_answer_model(), DEMO_USERS["admin_customer"], PAYROLL_Q)
    assert any(d["entitlement"] == "Payroll" for d in out["docs"])


def test_answer_includes_sources_section():
    out = _run(_answer_model(), DEMO_USERS["admin_customer"], PAYROLL_Q)
    last = out["messages"][-1].content
    assert "Sources:" in last
    assert "[1] " in last and " — https://" in last


def test_no_entitled_docs_skips_model():
    ctx = UserContext("u-9", "No Access", ("customer",), ("Recruiting",), ())
    out = _run(RaisingModel(messages=iter([])), ctx, PAYROLL_Q)
    assert out["docs"] == []
    assert (
        out["messages"][-1].content
        == "I couldn't find Community content you have access to on that topic."
    )


def test_context_propagates_when_called_as_subagent():
    answer = _answer_model("Sub answer.")
    orchestrator = ToolCallingFake(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": PAYROLL_Q, "subagent_type": "knowledge"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(
        model=orchestrator,
        subagents=[
            {
                "name": "knowledge",
                "description": "Answers product questions from Community KB.",
                "runnable": build_knowledge_graph(answer),
            }
        ],
        context_schema=UserContext,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(PAYROLL_Q)]}, context=DEMO_USERS["partner_readonly"]
    )
    # The subagent ran with the partner's context: its model prompt has no Payroll docs.
    assert answer.seen, "knowledge subagent model was never called"
    prompt = " ".join(str(m.content) for m in answer.seen[0])
    assert "KB-001" in prompt or "HCM" in prompt or "time off" in prompt.lower()
    assert "KB-003" not in prompt and "KB-004" not in prompt and "KB-005" not in prompt
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs and "Sources:" in tool_msgs[0].content
    assert "kb/KB-004" not in tool_msgs[0].content


def test_list_content_reply_is_flattened_to_text():
    blocks = [{"type": "text", "text": "Configure it per Time Off Plan [1]."}]
    last = _run(_answer_model(blocks), DEMO_USERS["admin_customer"], "time off accrual")["messages"][-1]
    assert last.content.startswith("Configure it per Time Off Plan [1].")
    assert "'type'" not in last.content and "[{" not in last.content


def test_model_written_sources_list_is_replaced_by_linked_one():
    text = "Configure it per plan [1].\n\n**Sources:**\n- [1] Configuring time off accrual for HCM"
    last = _run(_answer_model(text), DEMO_USERS["admin_customer"], "time off accrual")["messages"][-1]
    assert last.content.count("Sources:") == 1
    assert last.content.startswith("Configure it per plan [1].\n\nSources:\n[1] ")
    assert "https://community.example.com/kb/" in last.content


def test_last_user_text_reads_text_from_content_blocks():
    message = HumanMessage(content=[{"type": "text", "text": "off-cycle payroll"}])
    assert _last_user_text([message]) == "off-cycle payroll"


def test_off_cycle_payroll_question_retrieves_the_off_cycle_article():
    docs = retrieve({"messages": [HumanMessage("How do I run an off-cycle payroll?")], "docs": []})["docs"]
    assert docs[0]["id"] == "KB-009" and docs[0]["entitlement"] == "Payroll"
