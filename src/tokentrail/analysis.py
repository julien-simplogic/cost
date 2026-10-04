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
# Files Claude Code puts in, or builds, the start of the prompt.
PREFIX_FILES = {"CLAUDE.md", "CLAUDE.local.md", ".mcp.json", "settings.json", "settings.local.json"}
SPREAD_LIMIT = 100  # high / low bound at or above this (two orders of magnitude): no interval is given
# The interval aimed at holding 80% of outcomes. p10-p90 held 89% on real history (turns are
# whole numbers with many ties); p12.5-p87.5, chosen on the older half of that history, is the
# pair closest to 80% (scripts/study.py 3). `score` keeps checking it.
INTERVAL = (12.5, 87.5)
INTERVAL_LABEL = f"p{INTERVAL[0]:g}-p{INTERVAL[1]:g}"


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


@dataclass
class Expiry:
    """A main-thread call made after the cache TTL ran out, same model as the call before."""

    session_key: str
    model: str
    ts: str
    gap_s: float
    ttl: str
    context: int  # the previous call's input: what would have been cached
    read: int  # what was read from cache anyway


def expiries(rows: Iterable[Row]) -> list[Expiry]:
    """Every call after an idle expiry, across sessions. Compaction and model switches are left out."""
    by: dict[str, list[Row]] = {}
    for r in rows:
        if r.trigger != "subagent":
            by.setdefault(r.session_key, []).append(r)
    out = []
    for key, main in by.items():
        ttl = "5m"
        for prev, cur in zip(main, main[1:]):
            if prev.cache_write:
                ttl = "1h" if prev.cache_write_1h else "5m"
            if prev.input_total < MIN_CACHEABLE or cur.model != prev.model or cur.extra.get("after_compaction"):
                continue
            if cur.input_total < 0.5 * prev.input_total:
                continue
            t0, t1 = parse_ts(prev.ts), parse_ts(cur.ts)
            if not (t0 and t1):
                continue
            gap = (t1 - t0).total_seconds()
            if gap > TTL_SECONDS[ttl]:
                out.append(Expiry(key, cur.model, cur.ts, gap, ttl, prev.input_total, cur.cache_read))
    return out


def other_session_active(rows: Iterable[Row], cases: Sequence[Expiry]) -> list[bool]:
    """For each expiry: did another session call the same model within the TTL before it?

    If so, that session may have kept warm a prompt prefix both share (system
    prompt, tools), which would explain cache reads after an expiry.
    """
    import bisect

    by_model: dict[str, list[tuple[datetime, str]]] = {}
    for r in rows:
        t = parse_ts(r.ts)
        if t:
            by_model.setdefault(r.model, []).append((t, r.session_key))
    for v in by_model.values():
        v.sort()
    out = []
    for c in cases:
        calls = by_model.get(c.model, [])
        t = parse_ts(c.ts)
        found = False
        if t:
            i = bisect.bisect_left(calls, (t, ""))
            lo = t - timedelta(seconds=TTL_SECONDS[c.ttl])
            while i > 0 and calls[i - 1][0] >= lo:
                i -= 1
                if calls[i][1] != c.session_key:
                    found = True
                    break
        out.append(found)
    return out


def is_prefix_file(path: str) -> bool:
    p = path.replace("\\", "/")
    return p.rsplit("/", 1)[-1] in PREFIX_FILES or "/.claude/" in f"/{p}"


@dataclass
class BreakCauses:
    """What visibly happened before each cache break, and what followed each prefix-file edit.

    Visible means: in the transcript. A file you edit in your own editor, an MCP
    server that changes its tools, a Claude Code update between two calls: none
    of these is written there, so they land in "no visible cause".
    """

    breaks: int = 0
    model_switch: int = 0
    prefix_file_written: int = 0
    no_visible_cause: int = 0
    break_cost: dict = field(default_factory=lambda: {"model_switch": 0.0, "prefix_file_written": 0.0,
                                                      "no_visible_cause": 0.0})
    prefix_writes: int = 0  # prefix files written by a tool, with a next call to look at
    prefix_writes_then_break: int = 0  # ... followed by a break on that next call

    def add(self, other: "BreakCauses") -> None:
        for k in ("breaks", "model_switch", "prefix_file_written", "no_visible_cause",
                  "prefix_writes", "prefix_writes_then_break"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        for k, v in other.break_cost.items():
            self.break_cost[k] += v


def break_causes(main_rows: Sequence[Row], file_events: Sequence, prices: PriceTable) -> BreakCauses:
    """One session: attribute each cache break to what the transcript shows between two calls.

    A tool call is logged with the call that issued it, so a file written "between"
    calls N and N+1 carries a timestamp from N up to (not including) N+1.
    """
    out = BreakCauses()
    breaks = {e.turn_id: e for e in cache_events(main_rows, prices) if e.kind == "break"}
    writes = sorted((e["ts"] or "", e["path"]) for e in file_events
                    if e["action"] == "write" and is_prefix_file(e["path"]))
    for prev, cur in zip(main_rows, main_rows[1:]):
        written = [p for ts, p in writes if prev.ts <= ts < cur.ts]
        ev = breaks.get(cur.turn_id)
        if written and prev.input_total >= MIN_CACHEABLE and not cur.extra.get("after_compaction"):
            out.prefix_writes += 1
            out.prefix_writes_then_break += int(ev is not None)
        if ev is None:
            continue
        out.breaks += 1
        cause = ("model_switch" if cur.model != prev.model
                 else "prefix_file_written" if written else "no_visible_cause")
        setattr(out, cause, getattr(out, cause) + 1)
        out.break_cost[cause] += ev.extra_cost or 0
    return out


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


def percentile_sorted(xs: Sequence[float], q: float) -> Optional[float]:
    """percentile() on an already sorted list."""
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def distribution(values: Sequence[float], q: tuple[float, float] = INTERVAL) -> Optional[dict]:
    """Low bound, median, high bound (percentiles q), and whether the spread is too wide to give an interval."""
    if not values:
        return None
    xs = sorted(values)
    lo, p50, hi = (percentile_sorted(xs, x) for x in (q[0], 50, q[1]))
    ratio = hi / lo if lo else float("inf")  # type: ignore[operator]
    return {"n": len(xs), "lo": lo, "p50": p50, "hi": hi, "q": list(q), "ratio": ratio,
            "spread": ratio >= SPREAD_LIMIT}


def bounds(d: dict) -> tuple[float, float]:
    """(low, high) of an interval dict; estimates recorded before p12.5-p87.5 used p10/p90 keys."""
    return (d["lo"], d["hi"]) if "lo" in d else (d["p10"], d["p90"])


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
    first_cache_read: int = 0
    rows: list[Row] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return self.turns_main + self.turns_sub

    @property
    def tracked(self) -> bool:
        """Tied to a prompt of yours (records Claude Code logged outside any are not a task)."""
        return not self.task_id.endswith(":untracked")


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
                t.first_cache_read = r.cache_read
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
        try:
            dt = datetime.fromisoformat(spec)
        except ValueError:
            from .errors import TokentrailError

            raise TokentrailError(
                f"Can't read the date {spec!r}.",
                "Use 7d, 24h, 2w, 'all', or a date like 2026-09-01.",
            ) from None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
