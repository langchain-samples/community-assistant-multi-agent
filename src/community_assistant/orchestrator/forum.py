"""Forum post subagent: drafts a Community post, waits for the user to approve it, then posts.

This is an example of an agent that completes a task rather than just answering. The
posting tool is fake, but the approval step is real (``interrupt_on``).
"""

from __future__ import annotations

import re

from deepagents import SubAgent
from langchain_core.tools import tool

# Mock "forum": every approved post is recorded here instead of calling a real API.
POSTED: list[dict] = []


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@tool
def create_post(title: str, body: str, product_area: str) -> str:
    """Publish a new post to the Community forum. Requires user approval."""
    POSTED.append({"title": title, "body": body, "product_area": product_area})
    return f"Posted: https://community.example.com/t/{_slug(title)}"


FORUM_SYSTEM_PROMPT = (
    "You help users publish a post to the Community forum. Draft a clear title, "
    "a concise body, and pick the most relevant product area from the user's "
    "request. Then call create_post with the draft in this same turn, even if you "
    "were asked only to draft it or to get approval first: calling create_post is "
    "what shows the user the approval prompt, so never ask for approval in text. "
    "The user reviews it before it is published and may approve, edit, or reject it. If it is "
    "rejected, do not retry; acknowledge and ask what they'd like changed. "
    "After a successful post, share the link."
)

forum_subagent: SubAgent = {
    "name": "forum_post",
    "description": "Drafts and publishes a Community forum post after the user approves it.",
    "system_prompt": FORUM_SYSTEM_PROMPT,
    "tools": [create_post],
    "interrupt_on": {
        "create_post": {"allowed_decisions": ["approve", "edit", "reject"]}
    },
}
