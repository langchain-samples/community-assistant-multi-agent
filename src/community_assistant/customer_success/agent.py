"""Customer Success agent.

Stands in for a team that does not build on LangGraph. The front door treats it as
any A2A-compliant agent: it discovers it from its agent card and talks to it only
over the A2A protocol (JSON-RPC ``message/send``). That this one happens to be
served by an Agent Server (``langgraph.customer_success.json``, whose built-in A2A
endpoint is used) is an implementation detail the front door does not rely on.

The agent card's name comes from the graph ``name`` below, and its description
from the graph's ``description`` in ``langgraph.customer_success.json``. The
training catalog is mocked in memory.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel

from community_assistant.context import UserContext, require_user_context
from community_assistant.mock_data import TRAINING_CATALOG
from community_assistant.models import get_model
from community_assistant.tracing import continue_caller_trace

NO_PRODUCTS_COPY = (
    "I couldn't find any purchased products on your account, so I have no training to suggest."
)


def _format_course(product: str, course: dict) -> str:
    return (
        f"- {course['id']}: {course['title']} ({product}; {course['format']}, {course['duration']})"
    )


@tool
def list_training_for_products(runtime: ToolRuntime[UserContext]) -> str:
    """List the training courses available for the products the current user has purchased."""
    try:
        user = require_user_context(runtime.context)
    except PermissionError:
        return NO_PRODUCTS_COPY
    lines = [
        _format_course(product, course)
        for product in user.purchased_products
        for course in TRAINING_CATALOG.get(product, [])
    ]
    if not lines:
        return NO_PRODUCTS_COPY
    return "Training for your purchased products:\n" + "\n".join(lines)


SYSTEM_PROMPT = (
    "You are the Acme Customer Success assistant. You help customers find training "
    "and enablement for the products they own. Use list_training_for_products to see "
    "what is available for this user; only recommend courses it returns. Be concise."
)


def build_customer_success_agent(model: BaseChatModel | None = None):
    """Build the agent. The model is resolved here, not at import of this module."""
    return create_agent(
        model or get_model(),
        tools=[list_training_for_products],
        system_prompt=SYSTEM_PROMPT,
        context_schema=UserContext,
        name="customer_success",
    )


graph = build_customer_success_agent()
# What the server serves (see langgraph.customer_success.json): the same graph, joined to the
# caller's LangSmith trace when the request carries one.
make_graph = continue_caller_trace(graph)
