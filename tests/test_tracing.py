import asyncio

import langsmith as ls
from langsmith.run_helpers import get_tracing_context

from community_assistant.tracing import caller_trace_context, continue_caller_trace

TRACE_ID = "01a19dc3-3f39-75bd-a541-d7743ea5f0ee"
PARENT_ID = "01a19dc3-524a-76c6-be0a-5cb235d363e9"
DOTTED = f"20261102T193622053582Z{TRACE_ID}.20261102T193626436321Z{PARENT_ID}"


def _config(**extra):
    return {"configurable": {"langsmith-trace": DOTTED, **extra}}


def test_header_becomes_the_tracing_parent():
    with caller_trace_context(_config(**{"langsmith-metadata": {"scenario": "7-x"}})):
        parent = get_tracing_context()["parent"]
        assert str(parent.trace_id) == TRACE_ID
        assert str(parent.id) == PARENT_ID
        assert get_tracing_context()["metadata"]["scenario"] == "7-x"
    assert get_tracing_context()["parent"] is None


def test_no_header_is_a_no_op():
    with caller_trace_context({"configurable": {}}):
        assert get_tracing_context()["parent"] is None
    with caller_trace_context(None):
        assert get_tracing_context()["parent"] is None


def test_factory_yields_the_graph_inside_the_callers_trace():
    graph = object()
    factory = continue_caller_trace(graph)

    async def run():
        async with factory(_config()) as served:
            assert served is graph
            assert str(get_tracing_context()["parent"].id) == PARENT_ID
        assert get_tracing_context()["parent"] is None

    asyncio.run(run())


def test_child_runs_nest_under_the_forwarded_parent():
    seen = {}

    @ls.traceable
    def remote_step():
        rt = ls.get_current_run_tree()
        seen["trace_id"], seen["parent"] = str(rt.trace_id), str(rt.parent_run_id)

    with ls.tracing_context(enabled="local"), caller_trace_context(_config()):
        remote_step()
    assert seen == {"trace_id": TRACE_ID, "parent": PARENT_ID}


def test_malformed_header_is_ignored_and_the_graph_still_runs():
    graph = object()
    factory = continue_caller_trace(graph)
    bad = {"configurable": {"langsmith-trace": "garbage"}}

    async def run():
        async with factory(bad) as served:
            assert served is graph
            assert get_tracing_context()["parent"] is None

    asyncio.run(run())
    with caller_trace_context(bad):
        assert get_tracing_context()["parent"] is None
