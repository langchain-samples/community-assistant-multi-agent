"""Remote agents: Support Case over RemoteGraph and Customer Success over A2A.

Unit tests only: no server is needed. The live cross-server behaviour is
exercised by ``make smoke`` against the real servers.
"""

from __future__ import annotations

import asyncio
import json
import socket
import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from community_assistant.context import DEMO_USERS, UserContext
from community_assistant.front_door import remote_agents
from community_assistant.front_door.middleware import front_door_middleware
from community_assistant.front_door.remote_agents import (
    CUSTOMER_SUCCESS_AGENT_ID,
    A2AError,
    RemoteRunIncomplete,
    ask_customer_success,
    customer_success_subagent,
    fetch_agent_card,
    remote_thread_id,
    support_case_subagent,
)
from community_assistant.localization import FAILURE_COPY


def test_remote_thread_id_is_per_delegation():
    assert remote_thread_id("t", "tools:1") == remote_thread_id("t", "tools:1")
    assert remote_thread_id("t", "tools:1") != remote_thread_id("t", "tools:2")
    assert remote_thread_id("t", "tools:1") != remote_thread_id("u", "tools:1")
    uuid.UUID(remote_thread_id("t", "tools:1"))


def test_prepare_outside_a_run_with_explicit_context():
    graph = support_case_subagent()["runnable"]
    ctx = DEMO_USERS["admin_customer"].to_dict()
    cfg, context = graph._prepare({"configurable": {"ls_agent_type": "subagent"}}, ctx)
    assert context == ctx
    assert list(cfg["configurable"]) == ["thread_id"]
    assert cfg["metadata"] == {"ls_agent_type": "subagent"}


def _snapshot(messages, next_=()):
    return SimpleNamespace(values={"messages": messages}, next=next_, interrupts=())


_CTX = DEMO_USERS["admin_customer"].to_dict()


@pytest.mark.parametrize(
    "snapshot",
    [
        # The earlier remote run failed part-way: the graph still had nodes to run.
        _snapshot([HumanMessage("Open a case")], next_=("model",)),
        # It stopped without a reply for this turn.
        _snapshot([HumanMessage("Open a case")]),
    ],
    ids=["unfinished", "no-reply"],
)
def test_unfinished_remote_thread_raises_instead_of_empty_answer(monkeypatch, snapshot):
    graph = support_case_subagent()["runnable"]
    monkeypatch.setattr(graph, "get_state_history", lambda cfg, limit: [snapshot])

    async def ahistory(cfg, limit):
        yield snapshot

    monkeypatch.setattr(graph, "aget_state_history", ahistory)
    request = {"messages": [HumanMessage("Open a case")]}
    with pytest.raises(RemoteRunIncomplete):
        graph.invoke(request, context=_CTX)
    with pytest.raises(RemoteRunIncomplete):
        asyncio.run(graph.ainvoke(request, context=_CTX))


def test_finished_remote_thread_returns_its_reply(monkeypatch):
    graph = support_case_subagent()["runnable"]
    done = _snapshot([HumanMessage("status?"), AIMessage("CS-1042: Waiting on Acme.")])
    monkeypatch.setattr(graph, "get_state_history", lambda cfg, limit: [done])
    out = graph.invoke({"messages": [HumanMessage("status?")]}, context=_CTX)
    assert [m.text for m in out["messages"]] == ["CS-1042: Waiting on Acme."]


# ---------------------------------------------------------------------------
# Customer Success over A2A
#
# Unit tests use a closed port or an ``httpx.MockTransport``.
# ---------------------------------------------------------------------------


def _ask_cs(question: str, user: str | None = "admin_customer") -> str:
    return ask_customer_success(question, DEMO_USERS[user] if user else None)


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mock_a2a(monkeypatch, respond):
    """Route the tool's HTTP calls to ``respond(request)``; return the requests seen."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    monkeypatch.setattr(
        remote_agents, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    return seen


def _rpc(**body) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", **body})


def _part(text: str) -> dict:
    return {"kind": "text", "text": text}


def _task(state: str, *, artifact_text: str | None = None, history=(), status_text=None) -> dict:
    task = {"kind": "task", "id": "ctx:run", "contextId": "ctx", "status": {"state": state}}
    if artifact_text is not None:
        task["artifacts"] = [{"artifactId": "a1", "parts": [_part(artifact_text)]}]
    if status_text is not None:
        task["status"]["message"] = {"role": "agent", "parts": [_part(status_text)]}
    task["history"] = list(history)
    return task


def test_a2a_unreachable_raises_a2a_error(monkeypatch):
    monkeypatch.delenv("CUSTOMER_SUCCESS_A2A_URL", raising=False)
    monkeypatch.setattr(remote_agents, "CUSTOMER_SUCCESS_URL", f"http://127.0.0.1:{_closed_port()}")
    with pytest.raises(A2AError):
        _ask_cs("What training can I take?", "fr_customer")


def test_a2a_sends_message_send_with_user_context(monkeypatch):
    seen = _mock_a2a(
        monkeypatch,
        lambda r: _rpc(result=_task("completed", artifact_text="ok")),
    )
    _ask_cs("What training can I take?", "partner_readonly")

    (request,) = seen
    assert request.method == "POST"
    expected_url = f"{remote_agents.CUSTOMER_SUCCESS_URL}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"
    assert str(request.url) == expected_url
    body = json.loads(request.content)
    assert body["jsonrpc"] == "2.0" and body["method"] == "message/send"
    uuid.UUID(body["id"])
    message = body["params"]["message"]
    assert message["role"] == "user"
    uuid.UUID(message["messageId"])
    assert message["parts"] == [{"kind": "text", "text": "What training can I take?"}]
    # Standard A2A field, for any agent; params.context is the Agent Server extension.
    assert message["metadata"] == {"user_context": DEMO_USERS["partner_readonly"].to_dict()}
    assert body["params"]["context"] == DEMO_USERS["partner_readonly"].to_dict()


def test_a2a_returns_text_of_last_agent_message(monkeypatch):
    history = [
        {"role": "user", "parts": [{"kind": "text", "text": "q"}]},
        {"role": "agent", "parts": [{"kind": "text", "text": "Earlier answer"}]},
        {"role": "agent", "parts": [_part("Final "), {"kind": "data", "data": {}}, _part("answer")]},
    ]
    _mock_a2a(
        monkeypatch,
        lambda r: _rpc(result=_task("completed", history=history)),
    )
    assert _ask_cs("q") == "Final answer"

    # The completed task's response artifact is preferred over history.
    _mock_a2a(
        monkeypatch,
        lambda r: _rpc(result=_task("completed", artifact_text="From artifact", history=history)),
    )
    assert _ask_cs("q") == "From artifact"

    # A2A also lets an agent answer message/send with a bare Message.
    message = {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "Direct"}]}
    _mock_a2a(
        monkeypatch,
        lambda r: _rpc(result=message),
    )
    assert _ask_cs("q") == "Direct"


@pytest.mark.parametrize(
    "response",
    [
        _rpc(error={"code": -32602, "message": "bad"}),
        _rpc(result=_task("failed", status_text="Error executing assistant: boom")),
        _rpc(result=_task("completed", artifact_text="")),
        httpx.Response(500, text="oops"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["not", "an", "object"]),
        _rpc(result="just a string"),
        _rpc(result={"task": 42}),
    ],
    ids=[
        "jsonrpc-error", "failed-task", "empty-reply", "http-500", "non-json",
        "list-body", "scalar-result", "scalar-task",
    ],
)
def test_a2a_errors_raise_a2a_error(monkeypatch, response):
    _mock_a2a(monkeypatch, lambda r: response)
    with pytest.raises(A2AError):
        _ask_cs("q", "fr_customer")


class _OneDelegation(BaseChatModel):
    """Delegates once to the customer_success subagent, then answers with its reply."""

    @property
    def _llm_type(self) -> str:
        return "one-delegation"

    def bind_tools(self, tools, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if messages[-1].type == "tool":
            reply = AIMessage(content=messages[-1].text)
        else:
            args = {"description": "What training is available?", "subagent_type": "customer_success"}
            reply = AIMessage(content="", tool_calls=[{"name": "task", "args": args, "id": "c1"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _cs_front_door():
    return create_deep_agent(
        model=_OneDelegation(),
        subagents=[customer_success_subagent()],
        middleware=front_door_middleware(),
        context_schema=UserContext,
    )


def test_a2a_subagent_sends_the_delegated_question_with_user_context(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: _rpc(result=_task("completed", artifact_text="TR-HCM-101")))
    out = _cs_front_door().invoke(
        {"messages": [HumanMessage("training?")]}, context=DEMO_USERS["partner_readonly"]
    )
    assert out["messages"][-1].text == "TR-HCM-101"
    params = json.loads(seen[0].content)["params"]
    assert params["message"]["parts"][0]["text"] == "What training is available?"
    assert params["context"]["user_id"] == DEMO_USERS["partner_readonly"].user_id


def test_a2a_failure_is_retried_once_then_localized_by_middleware(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(503, text="unavailable"))
    out = _cs_front_door().invoke(
        {"messages": [HumanMessage("training?")]}, context=DEMO_USERS["fr_customer"]
    )
    assert len(seen) == 2  # first call + one retry inside the subagent
    tool_message = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert tool_message.content == FAILURE_COPY["fr_FR"]
    assert "503" not in out["messages"][-1].text


def test_a2a_recovers_on_retry_async(monkeypatch):
    responses = iter([httpx.Response(503), _rpc(result=_task("completed", artifact_text="TR-HCM-101"))])
    seen = _mock_a2a(monkeypatch, lambda r: next(responses))
    out = asyncio.run(
        _cs_front_door().ainvoke(
            {"messages": [HumanMessage("training?")]}, context=DEMO_USERS["admin_customer"]
        )
    )
    assert len(seen) == 2
    assert out["messages"][-1].text == "TR-HCM-101"


def test_a2a_subagent_without_user_context_fails_closed(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(500))
    out = _cs_front_door().invoke({"messages": [HumanMessage("training?")]})
    assert out["messages"][-1].text == FAILURE_COPY["en_US"]
    assert seen == []


def test_a2a_without_user_context_fails_closed(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(500))
    assert _ask_cs("q", None) == FAILURE_COPY["en_US"]
    assert seen == []  # nothing is sent without an identity


def test_a2a_forwards_tracing_headers_when_a_run_is_active(monkeypatch):
    TRACE = "20260102T000000000000Zabc.def"

    class FakeRunTree:
        def to_headers(self):
            return {"langsmith-trace": TRACE, "baggage": "langsmith-project=x"}

    seen = _mock_a2a(
        monkeypatch,
        lambda r: _rpc(result=_task("completed", artifact_text="ok")),
    )
    monkeypatch.setattr(remote_agents, "get_current_run_tree", lambda: FakeRunTree())
    _ask_cs("q")
    assert seen[0].headers["langsmith-trace"] == TRACE
    assert seen[0].headers["baggage"] == "langsmith-project=x"

    monkeypatch.setattr(remote_agents, "get_current_run_tree", lambda: None)
    _ask_cs("q")
    assert "langsmith-trace" not in seen[1].headers


def test_customer_success_is_a_compiled_subagent():
    subagent = customer_success_subagent()
    assert subagent["name"] == "customer_success"
    assert "training" in subagent["description"].lower()


def test_a2a_api_key_header_only_when_configured(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: _rpc(result=_task("completed", artifact_text="ok")))
    monkeypatch.delenv("CUSTOMER_SUCCESS_API_KEY", raising=False)
    _ask_cs("q")
    assert "x-api-key" not in seen[0].headers

    monkeypatch.setenv("CUSTOMER_SUCCESS_API_KEY", "cs-key")
    _ask_cs("q")
    assert seen[1].headers["x-api-key"] == "cs-key"


def test_agent_card_api_key_header_only_when_configured(monkeypatch):
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(200, json={"name": "x"}))
    monkeypatch.delenv("CUSTOMER_SUCCESS_API_KEY", raising=False)
    fetch_agent_card()
    assert "x-api-key" not in seen[0].headers

    monkeypatch.setenv("CUSTOMER_SUCCESS_API_KEY", "cs-key")
    fetch_agent_card()
    assert seen[1].headers["x-api-key"] == "cs-key"


def test_customer_success_agent_id_is_the_default_assistant_uuid():
    # The Agent Server's default assistant for graph "customer_success"; the A2A
    # route needs this UUID (the graph id is rejected).
    assert CUSTOMER_SUCCESS_AGENT_ID == "e7c056fa-9a09-568c-bd18-da02b1ed78d5"


def test_a2a_url_env_override_is_used_as_is(monkeypatch):
    # A non-LangGraph A2A agent, or a non-default assistant: its endpoint from its card/docs.
    endpoint = "https://agents.example.test/cs/a2a/7d0c1f0e-0000-4000-8000-000000000001"
    monkeypatch.setenv("CUSTOMER_SUCCESS_A2A_URL", endpoint)
    monkeypatch.delenv("CUSTOMER_SUCCESS_AGENT_CARD_URL", raising=False)
    card = {"name": "customer_success", "description": "training", "url": endpoint}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=card)
        return _rpc(result=_task("completed", artifact_text="ok"))

    seen = _mock_a2a(monkeypatch, respond)
    assert _ask_cs("q") == "ok"
    assert fetch_agent_card() == card
    assert [str(r.url) for r in seen] == [endpoint, f"{endpoint}/.well-known/agent-card.json"]


def test_agent_card_url_env_override(monkeypatch):
    monkeypatch.delenv("CUSTOMER_SUCCESS_A2A_URL", raising=False)
    card_url = "https://agents.example.test/.well-known/agent-card.json"
    monkeypatch.setenv("CUSTOMER_SUCCESS_AGENT_CARD_URL", card_url)
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(200, json={"name": "x"}))
    assert fetch_agent_card() == {"name": "x"}
    assert str(seen[0].url) == card_url


def test_agent_card_url_defaults_to_the_agent_server_per_assistant_card(monkeypatch):
    monkeypatch.delenv("CUSTOMER_SUCCESS_A2A_URL", raising=False)
    monkeypatch.delenv("CUSTOMER_SUCCESS_AGENT_CARD_URL", raising=False)
    seen = _mock_a2a(monkeypatch, lambda r: httpx.Response(200, json={"name": "x"}))
    fetch_agent_card()
    endpoint = f"{remote_agents.CUSTOMER_SUCCESS_URL}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"
    assert str(seen[0].url) == f"{endpoint}/.well-known/agent-card.json"


def test_a2a_url_falls_back_to_local_dev_path(monkeypatch):
    monkeypatch.delenv("CUSTOMER_SUCCESS_A2A_URL", raising=False)
    seen = _mock_a2a(monkeypatch, lambda r: _rpc(result=_task("completed", artifact_text="ok")))
    _ask_cs("q")
    expected = f"{remote_agents.CUSTOMER_SUCCESS_URL}/a2a/{CUSTOMER_SUCCESS_AGENT_ID}"
    assert str(seen[0].url) == expected
