from __future__ import annotations

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from conftest import ToolCallingFake
from community_assistant.front_door import forum
from community_assistant.front_door.forum import forum_subagent

POST_ARGS = {
    "title": "Best practice for accrued PTO payout",
    "body": "Original draft body.",
    "product_area": "Payroll",
}
CFG = {"configurable": {"thread_id": "t-1"}}


@pytest.fixture(autouse=True)
def _clear_posts():
    forum.POSTED.clear()
    yield
    forum.POSTED.clear()


def _agent():
    # One fake model drives parent and subagent; responses are consumed in order:
    # parent task call -> subagent create_post -> subagent final -> parent final.
    model = ToolCallingFake(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Post my PTO question",
                            "subagent_type": "forum_post",
                        },
                        "id": "call_task",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "create_post", "args": POST_ARGS, "id": "call_post"}
                ],
            ),
            AIMessage(content="Subagent finished."),
            AIMessage(content="All done."),
        ]
    )
    return create_deep_agent(
        model=model, subagents=[forum_subagent], checkpointer=MemorySaver()
    )


def _start(agent):
    return agent.invoke({"messages": [HumanMessage("Post this to the forum")]}, CFG)


def test_forum_post_pauses_for_approval():
    out = _start(_agent())
    assert "__interrupt__" in out
    req = out["__interrupt__"][0].value["action_requests"][0]
    assert req["name"] == "create_post"
    assert req["args"]["title"] == POST_ARGS["title"]
    assert forum.POSTED == []


def test_forum_approve_posts():
    agent = _agent()
    _start(agent)
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), CFG)
    assert len(forum.POSTED) == 1
    assert forum.POSTED[0]["body"] == "Original draft body."


def test_forum_reject_does_not_post():
    agent = _agent()
    _start(agent)
    agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "No thanks"}]}),
        CFG,
    )
    assert forum.POSTED == []


def test_forum_edit_posts_edited_body():
    agent = _agent()
    _start(agent)
    edited = {**POST_ARGS, "body": "Edited by the user."}
    agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {"name": "create_post", "args": edited},
                    }
                ]
            }
        ),
        CFG,
    )
    assert len(forum.POSTED) == 1
    assert forum.POSTED[0]["body"] == "Edited by the user."


def test_create_post_returns_slug_url():
    out = forum.create_post.invoke(
        {"title": "Hello, World! 2026", "body": "b", "product_area": "HCM"}
    )
    assert out == "Posted: https://community.example.com/t/hello-world-2026"


def test_prompts_route_drafts_through_the_approval_tool():
    # Live runs showed the subagent returning a text draft instead of calling create_post
    # when the front door asked it to "return the draft for review".
    from community_assistant.front_door.agent import FRONT_DOOR_PROMPT
    from community_assistant.front_door.forum import FORUM_SYSTEM_PROMPT

    assert "never ask for approval in text" in FORUM_SYSTEM_PROMPT
    assert "don't ask it to return a draft for review" in FRONT_DOOR_PROMPT
