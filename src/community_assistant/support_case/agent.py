"""Support Case agent.

This plays the part of an agent another team (the Support team) owns and deploys.
The orchestrator calls it over the network with RemoteGraph. It runs on its own port
from ``langgraph.support_case.json``, and the cases are mock data kept in memory.
"""

from __future__ import annotations

import copy

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from community_assistant.context import UserContext, require_user_context
from community_assistant.mock_data import CASES
from community_assistant.models import get_model
from community_assistant.tracing import continue_caller_trace

DENIED_COPY = (
    "You don't have permission to create support cases. Ask your Acme admin for access."
)

# Told to the model when create_case is hidden from a user without the role.
NO_CREATE_NOTE = (
    "This user does not have permission to create support cases. If they ask to open one, "
    f'reply with exactly: "{DENIED_COPY}"'
)

# Mock case store: the starting cases plus any created while the server runs.
CASES_STORE: dict[str, dict] = copy.deepcopy(CASES)
_next_case_number = 1043


@tool
def get_case_status(case_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Look up the status of one of the current user's support cases, e.g. CS-1042."""
    not_found = f"No case {case_id} found for your account."
    try:
        user = require_user_context(runtime.context)
    except PermissionError:
        return not_found
    case = CASES_STORE.get(case_id.strip().upper())
    if case is None or case["owner"] != user.user_id:
        return not_found
    return (
        f"{case['id']}: {case['status']}. {case['subject']} "
        f"(opened {case['opened']}). Latest update: {case['last_update']}"
    )


@tool
def create_case(
    subject: str, description: str, product: str, runtime: ToolRuntime[UserContext]
) -> str:
    """Open a new support case for the current user. Requires user approval."""
    global _next_case_number
    try:
        user = require_user_context(runtime.context)
    except PermissionError:
        return DENIED_COPY
    if "can_create_case" not in user.roles:
        return DENIED_COPY
    case_id = f"CS-{_next_case_number}"
    _next_case_number += 1
    CASES_STORE[case_id] = {
        "id": case_id,
        "owner": user.user_id,
        "status": "Open",
        "subject": subject,
        "description": description,
        "product": product,
        "opened": "2026-11-02",
        "last_update": "Your case has been received and is awaiting triage.",
    }
    return f"Created support case {case_id}: {subject}"


def _can_create(context) -> bool:
    try:
        return "can_create_case" in require_user_context(context).roles
    except PermissionError:
        return False


class RoleBasedToolAccess(AgentMiddleware):
    """Only offers ``create_case`` to users who are allowed to use it.

    The approval interrupt happens right after the model responds, before the tool runs.
    Without this middleware, a user without the role would be asked to approve a call that
    then gets denied. With the tool hidden, they just get a refusal and no approval prompt.
    ``create_case`` still checks the role itself as a backup.
    """

    def _filter(self, request):
        if _can_create(request.runtime.context):
            return request
        tools = [t for t in request.tools if getattr(t, "name", None) != "create_case"]
        # Say why the tool is missing, or the model just reports it "can't" create cases.
        base = request.system_message.content if request.system_message else ""
        note = f"{base}\n\n{NO_CREATE_NOTE}" if base else NO_CREATE_NOTE
        return request.override(tools=tools, system_message=SystemMessage(content=note))

    def wrap_model_call(self, request, handler):
        return handler(self._filter(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._filter(request))


SYSTEM_PROMPT = (
    "You are the Acme Support Case assistant. You can look up the status of the "
    "user's own support cases with get_case_status and open new ones with create_case. "
    "Before calling create_case, make sure you have a subject, a description and the "
    "product; ask for anything missing. The user is asked to approve every new case. "
    "If a tool says the user lacks permission or the case isn't found, relay that "
    "plainly without guessing or working around it. Be concise."
)


def build_support_case_agent(model: BaseChatModel | None = None, checkpointer=None):
    """Build the agent. The model is resolved here, not at import of this module."""
    return create_agent(
        model or get_model(),
        tools=[get_case_status, create_case],
        system_prompt=SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[
            RoleBasedToolAccess(),
            HumanInTheLoopMiddleware(
                interrupt_on={"create_case": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
        checkpointer=checkpointer,
        name="support_case",
    )


graph = build_support_case_agent()
# The server loads this (see langgraph.support_case.json). It's the same graph, but
# its runs are added to the caller's LangSmith trace when the request includes one.
make_graph = continue_caller_trace(graph)
