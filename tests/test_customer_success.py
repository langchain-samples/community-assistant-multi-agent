from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from conftest import ToolCallingFake
from community_assistant.context import DEMO_USERS
from community_assistant.customer_success.agent import (
    NO_PRODUCTS_COPY,
    build_customer_success_agent,
    list_training_for_products,
)


def _run(user):
    call = AIMessage(
        content="", tool_calls=[{"name": "list_training_for_products", "args": {}, "id": "c1"}]
    )
    agent = build_customer_success_agent(
        ToolCallingFake(responses=[call, AIMessage(content="Here is your training.")])
    )
    out = agent.invoke({"messages": [HumanMessage("What training can I take?")]}, context=user)
    return [m for m in out["messages"] if m.type == "tool"][-1].content


def test_training_uses_purchased_products():
    partner = _run(DEMO_USERS["partner_readonly"])  # purchased: Acme HCM only
    assert "TR-HCM-101" in partner and "TR-HCM-210" in partner
    assert "TR-PAY" not in partner

    admin = _run(DEMO_USERS["admin_customer"])  # HCM + Payroll
    assert "TR-HCM-101" in admin and "TR-PAY-220" in admin


def test_training_fails_closed_without_user_context():
    class R:
        context = None

    assert list_training_for_products.func(runtime=R()) == NO_PRODUCTS_COPY


def test_graph_is_named_for_the_agent_card():
    assert build_customer_success_agent(ToolCallingFake(responses=[])).name == "customer_success"
