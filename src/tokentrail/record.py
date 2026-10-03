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
    (a new process on the same session file, e.g. a resumed session)."""
    vals = [e[2] for e in series if len(e) == 3]
    return sum(1 for a, b in zip(vals, vals[1:]) if b < a)


def gap_shape(series: "list[list[int]]") -> dict:
    """How a session's gap with Claude Code's counter built up, snapshot by snapshot.

    series: [calls made before the snapshot, ours - counter] per cost-state snapshot.
    concentrated: the gap appeared between at most 2 snapshots (or 90% of it did):
                  the signature of a few hidden calls.
    spread:       it grew between many snapshots: the signature of a counting error.
    unknown:      a single snapshot, or no gap at all.
    """
    series = [[e[0], e[1] - e[2]] if len(e) == 3 else list(e) for e in series]
    if not series or all(g == 0 for _, g in series):
        return {"kind": "none", "steps": 0, "snapshots": len(series)}
    if len(series) < 2:
        return {"kind": "unknown", "steps": 0, "snapshots": 1}
    deltas = [series[0][1]] + [b[1] - a[1] for a, b in zip(series, series[1:])]
    calls = [series[0][0]] + [b[0] - a[0] for a, b in zip(series, series[1:])]
    moved = [(abs(d), c) for d, c in zip(deltas, calls) if d != 0]
    total = sum(d for d, _ in moved)
    top2 = sum(sorted((d for d, _ in moved), reverse=True)[:2])
    kind = "concentrated" if len(moved) <= 2 or top2 >= 0.9 * total else "spread"
    return {"kind": kind, "steps": len(moved), "snapshots": len(series),
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
