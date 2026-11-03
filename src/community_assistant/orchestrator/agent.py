"""The orchestrator: the main Deep Agents agent.

It hands each request to one of four subagents. The orchestrator calls them all the
same way, through ``task``, but each is built differently:

* ``knowledge``: a LangGraph workflow (retrieval with citations)
* ``forum_post``: a prompt and tools, with an approval step
* ``support_case``: another team's LangGraph agent on its own server (RemoteGraph)
* ``customer_success``: another team's agent, which could be on any stack (A2A)

Safety checks, clarifying questions and retries are handled by middleware
(``orchestrator_middleware``), not by the prompt.
"""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from community_assistant.context import UserContext
from community_assistant.orchestrator.demo_tools import flaky_community_search
from community_assistant.orchestrator.forum import forum_subagent
from community_assistant.orchestrator.knowledge import knowledge_subagent
from community_assistant.orchestrator.middleware import orchestrator_middleware
from community_assistant.orchestrator.remote_agents import (
    customer_success_subagent,
    support_case_subagent,
)
from community_assistant.models import get_model

ORCHESTRATOR_PROMPT = """\
You are the Acme Community Assistant. You help Acme customers and partners get \
answers, join discussions, and get support. You are friendly, concise and accurate.

How to handle a request:
- Product how-to and troubleshooting questions: delegate to the `knowledge` subagent. \
It answers from Community articles the user is entitled to see and includes numbered \
citations.
- Drafting or publishing a Community forum post: delegate to the `forum_post` subagent \
and ask it to draft and submit the post. Submitting automatically shows the user an \
approval prompt before anything is posted, so don't ask it to return a draft for review.
- Support case status, or opening a new support case: delegate to the `support_case` \
subagent. It is run by the Support team. Pass along a case number, subject, \
description and product when the user has given them.
- Training, learning paths or customer success questions: delegate to the \
`customer_success` subagent. It is run by the Customer Success team. Pass a \
self-contained question.
- To search Community discussion threads (peer conversations, not knowledge \
articles), call `flaky_community_search` with the user's query exactly as written, \
including any bracketed markers such as [flaky] or [down]. Use `knowledge` for \
documented how-to answers, and this tool only for discussion threads.
- Greetings and small talk: answer directly, briefly.

Multi-part questions: call every relevant specialist (in parallel when they are \
independent), then combine the results into one coherent reply. For example, case \
status + product docs + training: call `support_case`, `knowledge` and \
`customer_success`. Keep the numbered citations, and copy the `Sources:` list from \
`knowledge` verbatim; do not paraphrase or reformat it.

Rules:
- Never claim a forum post or support case was created, or give a case number, until \
the subagent or tool has confirmed it. If the user declines an approval, say it was \
not created.
- Never show raw errors, stack traces or tool internals. If a specialist or tool returns \
our standard failure apology (it starts like "Sorry — something went wrong on our side" \
or "Désolé, un problème est survenu"), include that apology copied word for word, \
exactly as returned, for the part that failed: it is already in the user's language, so \
do not reword or translate it. Still give the results of the parts that succeeded. You \
may add one short sentence offering an alternative.
- If a specialist says the user lacks access or permission, relay that plainly. Do not \
work around it.
- Do not invent product facts, case details or training. If you don't have the \
answer, say so.
- Reply in the user's language when their locale is not English (for example French for a French \
locale), including relayed specialist results. The current user's locale is given at the \
end of this prompt.
- You do not need files or a to-do list for this assistant; don't use the filesystem \
tools. Just route, then answer.
"""


def build_orchestrator(
    model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build the orchestrator. Uses ``get_model()`` unless a model is passed."""
    return create_deep_agent(
        model or get_model(),
        tools=[flaky_community_search],
        system_prompt=ORCHESTRATOR_PROMPT,
        middleware=orchestrator_middleware(),
        subagents=[
            knowledge_subagent,
            forum_subagent,
            support_case_subagent(),
            customer_success_subagent(),
        ],
        context_schema=UserContext,
        name="community_orchestrator",
    )


graph = build_orchestrator()
