"""Helpers ``demo.ipynb`` uses to drive the live orchestrator.

Every scenario runs on a fresh thread on the orchestrator server (default
``http://127.0.0.1:2024``, override with ``ORCHESTRATOR_URL``), with one of the
``DEMO_USERS`` as the run's context. A turn either finishes with an answer or
pauses on an interrupt (a clarification question or an approval request); pass
the user's reply to ``resume``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from langgraph_sdk import get_sync_client

from community_assistant.context import DEMO_USERS

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = "orchestrator"
APPROVE = {"decisions": [{"type": "approve"}]}
REJECT = {"decisions": [{"type": "reject"}]}

_client = None


def client():
    global _client
    if _client is None:
        _client = get_sync_client(url=ORCHESTRATOR_URL)
    return _client


def message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(
        b if isinstance(b, str) else b.get("text", "")
        for b in content
        if isinstance(b, str) or b.get("type") == "text"
    )


@dataclass
class Turn:
    """What one orchestrator turn produced, read back from the thread's state."""

    thread_id: str
    user: str
    scenario: str
    messages: list[dict] = field(default_factory=list)
    interrupts: list[dict] = field(default_factory=list)  # [{"id": ..., "value": ...}]

    @property
    def answer(self) -> str:
        """Text of the last AI message (empty while the turn is paused)."""
        for m in reversed(self.messages):
            if m.get("type") == "ai" and message_text(m).strip():
                return message_text(m)
        return ""

    @property
    def interrupt(self) -> Any:
        """The first pending interrupt's payload, or None."""
        return self.interrupts[0]["value"] if self.interrupts else None

    @property
    def calls(self) -> list[str]:
        """Specialists the orchestrator called, in order: ``task:<subagent>`` or a tool name."""
        out = []
        for m in self.messages:
            for tc in m.get("tool_calls") or ():
                if tc["name"] == "task":
                    out.append(f"task:{tc['args'].get('subagent_type', '?')}")
                else:
                    out.append(tc["name"])
        return out

    def tool_results(self, name: str) -> list[str]:
        """Contents of the ToolMessages returned for tool ``name`` (``task`` for subagents)."""
        return [message_text(m) for m in self.messages if m.get("type") == "tool" and m.get("name") == name]

    @property
    def action_names(self) -> list[str]:
        """Tool names awaiting approval in the pending interrupt (HITL payload)."""
        value = self.interrupt
        if not isinstance(value, dict):
            return []
        return [a.get("name") for a in value.get("action_requests", ())]


def _read_turn(thread_id: str, user: str, scenario: str) -> Turn:
    state = client().threads.get_state(thread_id)
    interrupts = list(state.get("interrupts") or ())
    if not interrupts:  # older servers only report them per task
        for task in state.get("tasks") or ():
            interrupts.extend(task.get("interrupts") or ())
    return Turn(
        thread_id=thread_id,
        user=user,
        scenario=scenario,
        messages=(state.get("values") or {}).get("messages", []),
        interrupts=interrupts,
    )


def run_turn(text: str, *, user: str = "admin_customer", scenario: str, thread_id: str | None = None) -> Turn:
    """Send one user message on a new thread (or ``thread_id``) and wait for it to finish or pause."""
    c = client()
    if thread_id is None:
        thread_id = c.threads.create(metadata={"scenario": scenario})["thread_id"]
    c.runs.wait(
        thread_id,
        ASSISTANT_ID,
        input={"messages": [{"role": "user", "content": text}]},
        context=DEMO_USERS[user].to_dict(),
        metadata={"scenario": scenario},
    )
    return _read_turn(thread_id, user, scenario)


def resume(turn: Turn, value: Any) -> Turn:
    """Answer the pending interrupt (a clarification reply, or ``APPROVE``/``REJECT``)."""
    if not turn.interrupts:
        raise ValueError("this turn is not waiting on an interrupt")
    if len(turn.interrupts) == 1:
        command = {"resume": value}
    else:  # several pending at once: answer each by id with the same value
        command = {"resume": {i["id"]: value for i in turn.interrupts}}
    client().runs.wait(
        turn.thread_id,
        ASSISTANT_ID,
        command=command,
        context=DEMO_USERS[turn.user].to_dict(),
        metadata={"scenario": turn.scenario},
    )
    return _read_turn(turn.thread_id, turn.user, turn.scenario)


def show(turn: Turn) -> None:
    """Pretty-print a turn: specialists called, the pending interrupt, the final answer."""
    print(f"user: {turn.user}    thread: {turn.thread_id}")
    print(f"called: {', '.join(turn.calls) or '(no specialists)'}")
    if turn.interrupt is not None:
        print("\nPAUSED, waiting on the user. Interrupt payload:")
        print(json.dumps(turn.interrupt, indent=2, ensure_ascii=False))
    else:
        print("\nanswer:\n" + turn.answer)
