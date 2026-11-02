"""Demo-only tools for showing the front door's retry and failure handling.

``flaky_community_search`` stands in for a community search backend with scripted
outages, keyed by markers in the query:

* ``"[flaky]"``: fails on the first such call in each conversation (thread), then
  works. The front door's tool retry (``max_retries=1``) recovers without the user
  noticing, and every new demo thread shows that retry again. Outside a run there
  is no thread, so one process-wide key is used.
* ``"[down]"``: always fails. After the retry, the user gets the hard-failure copy
  in their locale.

Anything else returns a short canned result.
"""

from __future__ import annotations

from langchain.tools import tool
from langgraph.config import get_config

FLAKY_MARKER = "[flaky]"
DOWN_MARKER = "[down]"

CANNED_RESULT = (
    'Community search: 3 matching threads. Top result: "Tips for a smooth payroll '
    'run" (accepted answer: run the audit report before you submit).'
)

_PROCESS_KEY = "__process__"
_flaky_failed: set[str] = set()  # conversations whose first [flaky] call already failed


class CommunitySearchUnavailable(RuntimeError):
    """The (mock) community search backend did not answer."""


def reset_flaky_state() -> None:
    """Make the next ``[flaky]`` call fail again everywhere, as in a fresh process."""
    _flaky_failed.clear()


def _flaky_key() -> str:
    """The current conversation's thread_id, or a process-wide key outside a run."""
    try:
        thread_id = get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:  # called outside a runnable context
        thread_id = None
    return str(thread_id) if thread_id else _PROCESS_KEY


@tool
def flaky_community_search(query: str) -> str:
    """Search community discussions and answers for the query."""
    if DOWN_MARKER in query:
        raise CommunitySearchUnavailable("community search is down")
    if FLAKY_MARKER in query:
        key = _flaky_key()
        if key not in _flaky_failed:
            _flaky_failed.add(key)
            raise CommunitySearchUnavailable("community search timed out")
    return CANNED_RESULT
