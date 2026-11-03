"""A demo tool for showing how the front door handles tool failures.

``flaky_community_search`` pretends to be a community search backend that sometimes
fails. What it does depends on a marker in the query:

* ``"[flaky]"``: the first call in each conversation (thread) fails and later calls
  work. The front door retries once, so the user doesn't notice, and every new
  thread shows the retry again. Outside a run there's no thread, so a single
  process-wide key is used.
* ``"[down]"``: every call fails. After the retry, the user gets the standard
  failure message in their language.

Any other query returns a short canned result.
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
