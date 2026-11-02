"""Knowledge subagent: a compiled LangGraph StateGraph used as a CompiledSubAgent.

Pipeline: retrieve -> filter_by_entitlement -> answer_with_citations.
Entitlement filtering reads the per-request UserContext from the LangGraph
runtime, so the same graph is safe to run standalone or as a Deep Agents subagent.
"""

from __future__ import annotations

import re
from typing import Annotated, TypedDict

from deepagents import CompiledSubAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import get_runtime

from community_assistant.context import UserContext, require_user_context
from community_assistant.mock_data import KB_CORPUS
from community_assistant.models import get_model

NO_ACCESS_MESSAGE = "I couldn't find Community content you have access to on that topic."
TOP_K = 4
_STOPWORDS = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "do", "how", "i", "at"}


class KnowledgeState(TypedDict):
    messages: Annotated[list, add_messages]
    docs: list[dict]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return m.text  # .text: content may be a list of blocks
    return ""


def retrieve(state: KnowledgeState) -> dict:
    """Keyword-overlap scoring over the mock KB (stand-in for a real search index)."""
    q = _tokens(_last_user_text(state["messages"]))
    scored = []
    for doc in KB_CORPUS:
        score = len(q & _tokens(doc["title"] + " " + doc["body"]))
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda s: -s[0])
    return {"docs": [d for _, d in scored[:TOP_K]]}


def filter_by_entitlement(state: KnowledgeState) -> dict:
    """Drop docs the caller is not entitled to. Fails closed without a user context."""
    user: UserContext = require_user_context(get_runtime().context)
    return {"docs": [d for d in state["docs"] if d["entitlement"] in user.entitlements]}


_SOURCES_HEADING = re.compile(r"^\W*sources\W*$", re.IGNORECASE | re.MULTILINE)


def _strip_sources(text: str) -> str:
    """Drop a sources list the model wrote anyway, so the linked one below is the only one."""
    match = None
    for match in _SOURCES_HEADING.finditer(text):
        pass
    return text[: match.start()].rstrip() if match else text.strip()


def _make_answer_node(model: BaseChatModel | None):
    def answer_with_citations(state: KnowledgeState) -> dict:
        docs = state["docs"]
        if not docs:
            return {"messages": [AIMessage(content=NO_ACCESS_MESSAGE)]}
        # Lazy resolution: no API key is needed to import/build the graph.
        llm = model or get_model()
        context = "\n\n".join(f"[{i}] {d['title']}\n{d['body']}" for i, d in enumerate(docs, 1))
        reply = llm.invoke(
            [
                SystemMessage(
                    content="Answer the question using ONLY the numbered Community articles "
                    "below. Cite them inline as [n]. If they do not answer it, say so. Do not "
                    "add a sources list: one with links is appended for you.\n\n"
                    + context
                ),
                HumanMessage(content=_last_user_text(state["messages"])),
            ]
        )
        text = _strip_sources(reply.text)  # .text: content may be a list of blocks
        sources = "\n".join(f"[{i}] {d['title']} — {d['url']}" for i, d in enumerate(docs, 1))
        return {"messages": [AIMessage(content=f"{text}\n\nSources:\n{sources}")]}

    return answer_with_citations


def build_knowledge_graph(model: BaseChatModel | None = None) -> CompiledStateGraph:
    g = StateGraph(KnowledgeState, context_schema=UserContext)
    g.add_node("retrieve", retrieve)
    g.add_node("filter_by_entitlement", filter_by_entitlement)
    g.add_node("answer_with_citations", _make_answer_node(model))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "filter_by_entitlement")
    g.add_edge("filter_by_entitlement", "answer_with_citations")
    g.add_edge("answer_with_citations", END)
    return g.compile()


knowledge_subagent: CompiledSubAgent = {
    "name": "knowledge",
    "description": (
        "Answers product how-to and troubleshooting questions from Community knowledge "
        "articles, with citations. Only uses content the user is entitled to see."
    ),
    "runnable": build_knowledge_graph(),
}
