"""The one format every collector produces.

This module is the contract between collectors (one per source) and
everything downstream (store, report, estimate). A new source — another
agent harness, raw API logs — only has to yield ``UsageRecord`` objects;
nothing downstream changes.

One record is one model call (one API request/response pair).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Trigger = Literal["user_turn", "tool_call", "subagent"]
TRIGGERS: tuple[str, ...] = ("user_turn", "tool_call", "subagent")


@dataclass(frozen=True)
class UsageRecord:
    # --- when / where / what ---
    timestamp: str  # ISO 8601, UTC
    source: str  # collector name, e.g. "claude-code"
    model: str

    # --- input: total = cache_read + cache_write + new ---
    input_total: int
    input_cache_read: int
    input_cache_write: int
    input_new: int

    # --- output: total, of which reasoning when the source reports it ---
    output_total: int
    output_reasoning: Optional[int]

    # --- grouping ---
    turn_id: str  # one model call
    task_id: str  # everything one request of yours caused, sub-agents included
    trigger: Trigger  # what caused this call

    duration_ms: Optional[int]

    # --- optional context; collectors fill what they know ---
    session_id: Optional[str] = None
    project: Optional[str] = None
    agent_id: Optional[str] = None  # set for sub-agent calls
    input_cache_write_1h: int = 0  # part of input_cache_write written with the 1-hour TTL
    # Where output_total comes from. "logged": the call's own log line.
    # "subagent_result": the call's own line was written mid-stream; the
    # figure comes from the sub-agent's result in the parent transcript.
    # Reports mark these, and --logged-only ignores them.
    output_source: str = "logged"
    output_logged: Optional[int] = None  # what the call's own line says (= output_total when logged)
    output_exact: bool = True  # False: logged mid-stream and not recoverable, a lower bound
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trigger not in TRIGGERS:
            raise ValueError(f"unknown trigger {self.trigger!r}")
        parts = self.input_cache_read + self.input_cache_write + self.input_new
        if parts != self.input_total:
            raise ValueError(
                f"input_total {self.input_total} != cache_read + cache_write + new ({parts})"
            )

    @property
    def output_recovered(self) -> int:
        """Output tokens not taken from the call's own log line."""
        if self.output_logged is None:
            return 0
        return self.output_total - self.output_logged

    @property
    def total_tokens(self) -> int:
        return self.input_total + self.output_total


@dataclass
class Task:
    """One request of yours and everything it set off."""

    task_id: str
    session_id: str
    source: str
    started_at: str
    project: Optional[str] = None
    cwd: Optional[str] = None
    family: Optional[str] = None  # question / refactor / review / measure
    family_source: str = "guessed"  # guessed | declared
    command: Optional[str] = None  # slash command that opened the task, if any
    prompt_chars: int = 0  # length only; prompt text is never stored


@dataclass
class FileEvent:
    session_id: str
    task_id: str
    timestamp: str
    path: str
    action: Literal["read", "write", "search"]


@dataclass
class CounterCheck:
    """One model's totals in a session: the source's own counter vs our sum.

    Claude Code keeps an in-memory counter fed by the API's final usage and
    writes it to the transcript as ``cost-state`` records. It is computed by a
    different code path from the per-call lines we read, so agreement means
    the de-duplication is right, and a gap is a measured gap, not a guess.
    """

    model: str
    source_input: int
    source_cache_read: int
    source_cache_write: int
    source_output: int
    ours_input: int
    ours_cache_read: int
    ours_cache_write: int
    ours_output_logged: int
    ours_output: int  # logged + recovered

    @property
    def input_matches(self) -> bool:
        return (self.source_input, self.source_cache_read, self.source_cache_write) == (
            self.ours_input, self.ours_cache_read, self.ours_cache_write)


def normalize_model(name: str) -> str:
    """One model, whatever qualifier a side adds: "claude-opus-5[1m]" -> "claude-opus-5".

    Claude Code's counter can name a model with a bracketed qualifier (the 1M
    context window) where the call lines don't. Used for comparisons only;
    prices and reports keep the name as written.
    """
    return re.sub(r"\[[^\]]*\]$", "", name.strip())


def totals_match(checks: "list[CounterCheck]") -> bool:
    """Input and cache summed over all models, ours vs the counter.

    Summed, so that a model the counter names differently from the call
    lines (e.g. with a context-size suffix) does not read as a mismatch;
    per-model rows are kept for `tokentrail diagnose`.
    """
    def tot(attr: str) -> int:
        return sum(getattr(c, attr) for c in checks)
    return all(tot(f"source_{k}") == tot(f"ours_{k}") for k in ("input", "cache_read", "cache_write"))


def counter_resets(series: "list[list[int]]") -> int:
    """Snapshots where Claude Code's counter went DOWN: it restarted counting
    (a new run of Claude Code on the same session file, e.g. a resumed session)."""
    vals = [e[2] for e in series if len(e) == 3]
    return sum(1 for a, b in zip(vals, vals[1:]) if b < a)


def increments(series: "list[list[int]]") -> dict:
    """Compare what was added between consecutive snapshots of the same counting run.

    Claude Code's counter belongs to a run, not to a file: it can start with usage
    carried in from outside the file, or restart at zero when the session is
    resumed. Differences between two snapshots of one run cancel both, so they
    are the reliable reference. Pairs where the counter went down are skipped.

    series: [calls before the snapshot, ours, counter] per snapshot (input + cache).
    """
    pairs = [(b[0] - a[0], b[1] - a[1], b[2] - a[2])
             for a, b in zip(series, series[1:]) if len(a) == 3 and len(b) == 3 and b[2] >= a[2]]
    first = series[0] if series else None
    return {
        "pairs": len(pairs),
        "calls": sum(p[0] for p in pairs),
        "ours": sum(p[1] for p in pairs),
        "counter": sum(p[2] for p in pairs),
        "deltas": [p[1] - p[2] for p in pairs],  # per interval, ours - counter
        "calls_per_pair": [p[0] for p in pairs],
        # counter minus ours at the first snapshot: > 0 carried in, < 0 not covered
        "offset": (first[2] - first[1]) if first and len(first) == 3 else None,
        "restarts": counter_resets(series),
    }


def gap_shape(series: "list[list[int]]") -> dict:
    """How the gap with Claude Code's counter moved between snapshots of one run.

    The gap already present at the first snapshot is left out (it can be usage
    the counter carried in), and so are intervals where the counter restarted.
    concentrated: the gap moved in at most 2 intervals, or 90% of it did: a few
                  hidden calls.
    spread:       it moved in many intervals: the signature of a counting error.
    none:         it never moved: increments match to the token.
    unknown:      a single snapshot, nothing to compare between.
    """
    if series and len(series[0]) == 2:  # legacy form: [calls, ours - counter]
        series = [[c, g, 0] for c, g in series]
        inc = {"deltas": [b[1] - a[1] for a, b in zip(series, series[1:])],
               "calls_per_pair": [b[0] - a[0] for a, b in zip(series, series[1:])], "pairs": len(series) - 1}
    else:
        inc = increments(series)
    if len(series) < 2 or inc["pairs"] == 0:
        return {"kind": "unknown", "steps": 0, "intervals": 0}
    if inc["pairs"] < 3 and any(inc["deltas"]):
        # one or two intervals: any gap is trivially "concentrated", which says nothing
        return {"kind": "too few intervals", "steps": sum(1 for d in inc["deltas"] if d), "intervals": inc["pairs"]}
    moved = [(abs(d), c) for d, c in zip(inc["deltas"], inc["calls_per_pair"]) if d != 0]
    if not moved:
        return {"kind": "none", "steps": 0, "intervals": inc["pairs"]}
    total = sum(d for d, _ in moved)
    top2 = sum(sorted((d for d, _ in moved), reverse=True)[:2])
    kind = "concentrated" if len(moved) <= 2 or top2 >= 0.9 * total else "spread"
    return {"kind": kind, "steps": len(moved), "intervals": inc["pairs"],
            "calls_in_steps": sum(c for _, c in moved)}


@dataclass
class ParseStats:
    """What a collector read, what it ignored on purpose, what it could not read."""

    files: int = 0
    lines: int = 0
    used: int = 0
    ignored: dict[str, int] = field(default_factory=dict)  # known record types we skip by design
    unreadable: dict[str, int] = field(default_factory=dict)  # bad JSON, unexpected shape
    unknown_types: dict[str, int] = field(default_factory=dict)  # types this version doesn't know
    # unknown types that contain token-count fields: they may hold usage we miss
    unknown_with_tokens: dict[str, int] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)

    def bump(self, bucket: dict[str, int], key: str, n: int = 1) -> None:
        bucket[key] = bucket.get(key, 0) + n

    @property
    def not_understood(self) -> int:
        return sum(self.unreadable.values()) + sum(self.unknown_types.values())

    def merge(self, other: "ParseStats") -> None:
        self.files += other.files
        self.lines += other.lines
        self.used += other.used
        for mine, theirs in (
            (self.ignored, other.ignored),
            (self.unreadable, other.unreadable),
            (self.unknown_types, other.unknown_types),
            (self.unknown_with_tokens, other.unknown_with_tokens),
            (self.versions, other.versions),
        ):
            for k, v in theirs.items():
                mine[k] = mine.get(k, 0) + v
