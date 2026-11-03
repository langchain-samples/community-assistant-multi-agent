"""Add a remote agent's runs to the caller's LangSmith trace.

The orchestrator sends LangSmith distributed tracing headers with every remote call
(``RemoteGraph(distributed_tracing=True)`` for Support Case, and ``langsmith-trace`` and
``baggage`` on the A2A request for Customer Success). The Agent Server copies them into
the run's ``config["configurable"]`` as ``langsmith-trace`` plus the ``langsmith-project``,
``langsmith-metadata`` and ``langsmith-tags`` values from baggage. But in the server
version this repo uses, nothing turns them back into a trace parent, so each remote run
showed up in LangSmith as its own separate trace.

``continue_caller_trace(graph)`` fixes this on the receiving side. It returns a graph
factory that the Agent Server enters for each run. During the run, the caller's trace
is set as the parent, so the remote run appears under the orchestrator's call and the
whole request is one trace. Requests without the header (from Studio, or direct calls)
are traced the same as before.
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
    try:  # parse it first, so a bad header can't break the run
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
