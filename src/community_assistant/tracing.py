"""Continue the caller's LangSmith trace inside a separately served agent.

The front door sends LangSmith distributed-tracing headers with every remote call
(``RemoteGraph(distributed_tracing=True)`` for Support Case, ``langsmith-trace`` and
``baggage`` on the A2A request for Customer Success). The Agent Server copies them
into the run's ``config["configurable"]`` (``langsmith-trace`` plus the parsed
``langsmith-project`` / ``langsmith-metadata`` / ``langsmith-tags`` baggage), but in
the server version pinned here nothing turns them back into a trace parent: the
remote run was logged as its own, unconnected trace (verified against LangSmith).

``continue_caller_trace(graph)`` closes that gap on the receiving side. It returns a
graph factory, which the Agent Server enters around each run. While the run is in
progress, the forwarded parent is the LangSmith tracing context, so the remote run
nests under the front door's tool call and the whole request is one trace. Requests
without the header (Studio, direct calls) are traced as before.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import langsmith as ls
from langchain_core.runnables import RunnableConfig
from langsmith.run_trees import RunTree

logger = logging.getLogger(__name__)

TRACE_KEY = "langsmith-trace"


def caller_trace_context(config: RunnableConfig | None) -> contextlib.AbstractContextManager:
    """A tracing context parented on the caller's run, or a no-op without the header."""
    configurable = (config or {}).get("configurable") or {}
    parent = configurable.get(TRACE_KEY)
    if not isinstance(parent, str) or not parent:
        return contextlib.nullcontext()
    kwargs: dict[str, Any] = {}
    if project := configurable.get("langsmith-project"):
        kwargs["project_name"] = project
    try:  # parse up front: tracing must never break the run
        kwargs["parent"] = RunTree.from_dotted_order(parent, **kwargs)
    except Exception:
        logger.warning("ignoring malformed %s header; tracing this run on its own", TRACE_KEY)
        return contextlib.nullcontext()
    if isinstance(metadata := configurable.get("langsmith-metadata"), dict):
        kwargs["metadata"] = metadata
    if isinstance(tags := configurable.get("langsmith-tags"), list):
        kwargs["tags"] = tags
    return ls.tracing_context(**kwargs)


def continue_caller_trace(graph):
    """Wrap a compiled graph in an Agent Server graph factory that joins the caller's trace."""

    @contextlib.asynccontextmanager
    async def make_graph(config: RunnableConfig) -> AsyncIterator[Any]:
        with caller_trace_context(config):
            yield graph

    return make_graph
