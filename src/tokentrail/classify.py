"""Task families and consumer categories.

Both are heuristics and say so. A task's family can always be declared
(``tokentrail tag``), which overrides the guess.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

FAMILIES = ("question", "refactor", "review", "measure")

CATEGORIES = ("subagents", "reviews", "tool_turns", "user_turns")
CATEGORY_LABELS = {
    "subagents": "sub-agents",
    "reviews": "reviews & re-reads",
    "tool_turns": "tool turns",
    "user_turns": "your messages",
}

REVIEW_COMMANDS = frozenset({"review", "code-review", "security-review", "pr-review", "pr-comments"})
_WRITES = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
_READS = frozenset({"Read", "Grep", "Glob", "LS", "NotebookRead"})
_MEASURE_RE = re.compile(
    r"\b(bench\w*|hyperfine|time\s|timeit|perf\b|profil\w*|cProfile|py-spy|coverage|"
    r"pytest\s.*--durations|wc\s|du\s|cloc|tokei|ccusage|tokentrail|lighthouse|ab\s|wrk\s)",
    re.IGNORECASE,
)


def guess_family(command: Optional[str], first_tools: Iterable[tuple[str, dict[str, Any]]]) -> str:
    """Guess from the slash command and the first (up to five) tool calls.

    review   - opened with a review command, or only reads/searches (3+)
    refactor - edits or writes a file early
    measure  - runs benchmarks, profilers, counters
    question - everything else (few or no tool calls)
    """
    if command and command.split(":")[-1].lower() in REVIEW_COMMANDS:
        return "review"
    tools = list(first_tools)
    names = [n for n, _ in tools]
    if any(n in _WRITES for n in names):
        return "refactor"
    for n, inp in tools:
        if n == "Bash" and _MEASURE_RE.search(str(inp.get("command", ""))):
            return "measure"
    if len(names) >= 3 and all(n in _READS for n in names):
        return "review"
    return "question"


def category(trigger: str, family: Optional[str], reread: bool) -> str:
    """Which consumer a call is charged to. First match wins."""
    if trigger == "subagent":
        return "subagents"
    if family == "review" or reread:
        return "reviews"
    if trigger == "tool_call":
        return "tool_turns"
    return "user_turns"
