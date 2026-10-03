"""The one format every collector produces.

This module is the contract between collectors (one per source) and
everything downstream (store, report, estimate). A new source — another
agent harness, raw API logs — only has to yield ``UsageRecord`` objects;
nothing downstream changes.

One record is one model call (one API request/response pair).
"""

from __future__ import annotations

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
    output_exact: bool = True  # False when the source logged the count mid-stream
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
class ParseStats:
    """What a collector read, what it ignored on purpose, what it could not read."""

    files: int = 0
    lines: int = 0
    used: int = 0
    ignored: dict[str, int] = field(default_factory=dict)  # known record types we skip by design
    unreadable: dict[str, int] = field(default_factory=dict)  # bad JSON, unexpected shape
    unknown_types: dict[str, int] = field(default_factory=dict)  # types this version doesn't know
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
            (self.versions, other.versions),
        ):
            for k, v in theirs.items():
                mine[k] = mine.get(k, 0) + v
