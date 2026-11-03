"""Customer Success agent.

This plays the part of a team that doesn't use LangGraph. The orchestrator only talks
to it over A2A: it reads the agent card and sends JSON-RPC ``message/send`` requests.
It happens to run on an Agent Server here (``langgraph.customer_success.json``), but
the orchestrator doesn't depend on that.

The agent card takes its name from the graph ``name`` below and its description from
``langgraph.customer_success.json``. The training catalog is mock data.
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
# The server loads this (see langgraph.customer_success.json). It's the same graph, but
# its runs are added to the caller's LangSmith trace when the request includes one.
make_graph = continue_caller_trace(graph)
