"""Shared computations over stored records: cost, categories, cache events, tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from .classify import category
from .prices import PriceTable

TTL_SECONDS = {"5m": 300, "1h": 3600}
MIN_CACHEABLE = 4096  # below this a missing cache read is not worth flagging


def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass
class Row:
    """A stored record, enriched for analysis."""

    source: str
    turn_id: str
    session_key: str
    session_id: str
    ts: str
    model: str
    input_total: int
    cache_read: int
    cache_write: int
    cache_write_1h: int
    input_new: int
    output: int
    output_logged: int
    output_source: str
    reasoning: Optional[int]
    output_exact: bool
    task_id: str
    trigger: str
    duration_ms: Optional[int]
    project: str
    agent_id: Optional[str]
    extra: dict
    cost: Optional[float] = None
    family: Optional[str] = None
    category: str = ""

    @property
    def tokens(self) -> int:
        return self.input_total + self.output

    @property
    def recovered(self) -> int:
        """Output tokens not from this call's own log line (0 unless marked)."""
        return self.output - self.output_logged


def enrich(db_rows: Iterable, prices: PriceTable, tasks: dict, logged_only: bool = False) -> list[Row]:
    """logged_only: use each call's own log line for output, ignore recovered figures."""
    out = []
    for r in db_rows:
        extra = json.loads(r["extra"]) if r["extra"] else {}
        row = Row(
            source=r["source"], turn_id=r["turn_id"], session_key=r["session_key"],
            session_id=r["session_id"] or "", ts=r["ts"], model=r["model"],
            input_total=r["input_total"], cache_read=r["input_cache_read"],
            cache_write=r["input_cache_write"], cache_write_1h=r["input_cache_write_1h"],
            input_new=r["input_new"],
            output=r["output_logged"] if logged_only else r["output_total"],
            output_logged=r["output_logged"], output_source=r["output_source"],
            reasoning=r["output_reasoning"],
            output_exact=bool(r["output_exact"]), task_id=r["task_id"], trigger=r["trigger"],
            duration_ms=r["duration_ms"], project=r["project"] or "?", agent_id=r["agent_id"],
            extra=extra,
        )
        row.cost = prices.cost(
            row.model, new=row.input_new, cache_read=row.cache_read,
            cache_write=row.cache_write, cache_write_1h=row.cache_write_1h, output=row.output,
        )
        task = tasks.get((row.source, row.task_id))
        row.family = task["family"] if task else None
        row.category = category(row.trigger, row.family, bool(extra.get("reread")))
        out.append(row)
    return out


# ---------------------------------------------------------------- cache events


@dataclass
class CacheEvent:
    kind: str  # "expired" (TTL ran out) or "break" (prefix changed within TTL)
    session_key: str
    turn_id: str
    ts: str
    lost_tokens: int  # context that had to be written again
    extra_cost: Optional[float]
    gap_s: float


def ttl_of(rows: Sequence[Row]) -> str:
    """TTL Claude Code used most recently in these calls."""
    for r in reversed(rows):
        if r.cache_write_1h:
            return "1h"
        if r.cache_write:
            return "5m"
    return "5m"


def cache_events(main_rows: Sequence[Row], prices: PriceTable) -> list[CacheEvent]:
    """Walk one session's main-thread calls in order and find lost caches.

    After call N, the next call should read about N's input from cache. When
    it reads much less, either the TTL expired, or something early in the
    prompt changed (model switch, tools, system prompt, settings).
    Compaction legitimately resets the prefix and is not counted.
    """
    events: list[CacheEvent] = []
    ttl = "5m"
    for prev, cur in zip(main_rows, main_rows[1:]):
        if prev.cache_write:
            ttl = "1h" if prev.cache_write_1h else "5m"
        expected = prev.input_total
        if expected < MIN_CACHEABLE or cur.extra.get("after_compaction"):
            continue
        if cur.input_total < expected * 0.5:  # context shrank: compaction or /clear
            continue
        if cur.cache_read >= expected * 0.5:
            continue
        t0, t1 = parse_ts(prev.ts), parse_ts(cur.ts)
        gap = (t1 - t0).total_seconds() if t0 and t1 else 0.0
        kind = "expired" if gap > TTL_SECONDS[ttl] else "break"
        lost = expected - cur.cache_read
        p = prices.lookup(cur.model)
        extra = None
        if p:
            mult = prices.write_1h if ttl == "1h" else prices.write_5m
            extra = lost * (p.input * mult - p.cache_read) / 1_000_000
        events.append(CacheEvent(kind, cur.session_key, cur.turn_id, cur.ts, lost, extra, gap))
    return events


# ---------------------------------------------------------------- statistics


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class TaskStats:
    source: str
    task_id: str
    session_key: str
    project: str
    family: Optional[str]
    family_source: str
    command: Optional[str]
    started: str
    turns_main: int = 0
    turns_sub: int = 0
    tokens: int = 0
    output: int = 0
    input_total: int = 0
    first_input: int = 0
    sub_tokens: int = 0
    recovered: int = 0
    cost: float = 0.0
    unpriced: int = 0
    rows: list[Row] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return self.turns_main + self.turns_sub


def group_tasks(rows: Iterable[Row], tasks: dict) -> list[TaskStats]:
    by: dict[tuple[str, str], TaskStats] = {}
    for r in rows:
        key = (r.source, r.task_id)
        t = by.get(key)
        if t is None:
            meta = tasks.get(key, {})
            t = by[key] = TaskStats(
                r.source, r.task_id, r.session_key, r.project, meta.get("family"),
                meta.get("family_source", "guessed"), meta.get("command"),
                meta.get("started_at") or r.ts,
            )
        t.rows.append(r)
        if r.trigger == "subagent":
            t.turns_sub += 1
            t.sub_tokens += r.tokens
        else:
            if t.turns_main == 0:
                t.first_input = r.input_total
            t.turns_main += 1
        t.tokens += r.tokens
        t.output += r.output
        t.recovered += r.recovered
        t.input_total += r.input_total
        if r.cost is None:
            t.unpriced += 1
        else:
            t.cost += r.cost
    return list(by.values())


# ---------------------------------------------------------------- periods


def parse_since(spec: Optional[str], now: Optional[datetime] = None) -> Optional[str]:
    """'7d', '24h', '2w', or an ISO date -> ISO timestamp (UTC)."""
    if not spec:
        return None
    now = now or datetime.now(timezone.utc)
    units = {"h": "hours", "d": "days", "w": "weeks"}
    if spec[-1:] in units and spec[:-1].isdigit():
        dt = now - timedelta(**{units[spec[-1]]: int(spec[:-1])})
    else:
        dt = datetime.fromisoformat(spec)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
