"""The estimate scores itself: every estimate is recorded, then compared with what happened.

An estimate is linked to the first task of its session that starts after it
(within LINK_WINDOW_S). Once that task is over, a later task started in the
same session or nothing happened in it for FINISHED_AFTER_S, its outcome is
written next to the estimate and never recomputed: transcripts get cleaned up,
the record stays.

`tokentrail score` then says, over the last N scored estimates, how often the
outcome fell inside the interval and how far the median was off. It also
backtests on the whole history what needs no recorded estimate: the computed
next-call input, the cache state, and the turn intervals each task would have
got from the tasks before it.

Only numbers are stored, never the prompt text.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime, timezone
from typing import Any, Optional

from .analysis import (
    INTERVAL, INTERVAL_LABEL, SPREAD_LIMIT, bounds, expiries, other_session_active, TTL_SECONDS, TaskStats, distribution, enrich, group_tasks, parse_ts, percentile,
    percentile_sorted,
)
from . import metrics
from .classify import FAMILIES
from .prices import PriceTable
from .report import _table
from .store import Store

LINK_WINDOW_S = 1800  # an estimate older than this when its session's next task starts is not about it
FINISHED_AFTER_S = 3600  # a task with no activity for this long is over
START_SLACK_S = 5  # the hook runs just before the prompt is written: allow a little clock jitter
MIN_BASIS = 8  # same as estimate.MIN_FAMILY_SAMPLES
BOOTSTRAP = 400  # session resamples per confidence interval: ~1 s on 4,000 tasks


def record(store: Store, est: dict[str, Any], origin: str) -> int:
    """Keep what the estimate said. Numbers only."""
    c, p = est["computed"], est["predicted"]
    return store.add_prediction(est["made_at"], origin, est["session_key"], {
        "family": p["family"],
        "family_source": p["family_source"],
        "basis_is_family": p["basis_is_family"],
        "basis_kind": p.get("basis_kind"),
        "samples": p["samples"],
        "turns": p["turns"],
        "cost": p["cost"],
        "input_next": c["input_next"],
        "cached": c["cached"],
        "cache_ttl": c["cache_ttl"],
        "seconds_since_last_call": c["seconds_since_last_call"],
        "floor_cost": est["frame"]["floor"]["cost"],
        "prompt_chars": est.get("prompt_chars", 0),
        "model": est["model"],
    })


def settle(store: Store, prices: PriceTable, now: Optional[datetime] = None,
           task_stats: Optional[list[TaskStats]] = None) -> int:
    """Link open estimates to their task, and record the outcome of finished ones. Returns how many changed."""
    if not store.has_open_predictions():
        return 0
    now = now or datetime.now(timezone.utc)
    if task_stats is None:
        tasks = store.tasks()
        task_stats = group_tasks(enrich(store.records(), prices, tasks), tasks)
    by_session: dict[str, list[TaskStats]] = {}
    for t in task_stats:
        if t.tracked and t.turns_main:
            by_session.setdefault(t.session_key, []).append(t)
    for ts in by_session.values():
        ts.sort(key=lambda t: t.started)
    stats = {(t.session_key, t.task_id): t for t in task_stats}
    taken = {(p["session_key"], p["task_id"]) for p in store.predictions() if p["task_id"]}
    changed = 0

    # newest first: when several estimates precede one task, the latest one is about it
    for p in sorted(store.predictions("open"), key=lambda p: p["made_at"], reverse=True):
        made = parse_ts(p["made_at"])
        if made is None:
            store.settle_prediction(p["id"], "unmatched")
            changed += 1
            continue
        task = None
        for t in by_session.get(p["session_key"], []):
            start = parse_ts(t.started)
            if start is None:
                continue
            dt = (start - made).total_seconds()
            if dt < -START_SLACK_S:
                continue
            if dt <= LINK_WINDOW_S:
                task = t
            break
        if task is not None and (p["session_key"], task.task_id) in taken:
            store.settle_prediction(p["id"], "superseded")  # a later estimate is about the same task
            changed += 1
        elif task is not None:
            taken.add((p["session_key"], task.task_id))
            store.settle_prediction(p["id"], "linked", task_id=task.task_id)
            changed += 1
        elif (now - made).total_seconds() > LINK_WINDOW_S:
            store.settle_prediction(p["id"], "unmatched")
            changed += 1

    for p in store.predictions("linked"):
        t = stats.get((p["session_key"], p["task_id"]))
        if t is None:
            continue
        if not _finished(t, by_session.get(p["session_key"], []), now):
            continue
        store.settle_prediction(p["id"], "scored", outcome={
            "turns_main": t.turns_main,
            "turns": t.turns,
            "cost": round(t.cost, 6),
            "unpriced_calls": t.unpriced,
            "first_input": t.first_input,
            "first_cache_read": t.first_cache_read,
            "family_after": t.family,
            "finished_seen_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        changed += 1
    return changed


def _finished(t: TaskStats, same_session: list[TaskStats], now: datetime) -> bool:
    if any(o.started > t.started for o in same_session):
        return True
    last = parse_ts(max(r.ts for r in t.rows))
    return last is not None and (now - last).total_seconds() > FINISHED_AFTER_S


# ------------------------------------------------------------------ score


def build_score(store: Store, prices: PriceTable, last: int = 50) -> dict[str, Any]:
    tasks = store.tasks()
    rows = enrich(store.records(), prices, tasks)
    task_stats = group_tasks(rows, tasks)
    settle(store, prices, task_stats=task_stats)
    preds = store.predictions()
    scored = [p for p in preds if p["status"] == "scored"][-last:]
    return {
        "recorded": {
            "total": len(preds),
            "by_origin": _count(p["origin"] for p in preds),
            "by_status": _count(p["status"] for p in preds),
        },
        "last": last,
        "scored": _score(scored),
        "backtest": {
            "computed": backtest_computed(rows, task_stats, tasks),
            "turns": backtest_turns(task_stats),
        },
    }


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _score(scored: list[dict]) -> dict[str, Any]:
    n = len(scored)
    with_interval = [p for p in scored if p["prediction"].get("turns") and not p["prediction"]["turns"]["spread"]]
    no_interval = [p for p in scored if p["prediction"].get("turns") and p["prediction"]["turns"]["spread"]]
    no_history = n - len(with_interval) - len(no_interval)

    def inside(lo: float, x: float, hi: float) -> bool:
        return lo <= x <= hi

    turns_in = sum(1 for p in with_interval if inside(
        bounds(p["prediction"]["turns"])[0], p["outcome"]["turns_main"], bounds(p["prediction"]["turns"])[1]))
    has_turns = with_interval + no_interval
    turn_err = [abs(p["outcome"]["turns_main"] - p["prediction"]["turns"]["p50"]) for p in has_turns]
    turn_ratio = [p["outcome"]["turns_main"] / p["prediction"]["turns"]["p50"]
                  for p in has_turns if p["prediction"]["turns"]["p50"]]
    priced = [p for p in with_interval if p["prediction"].get("cost") and not p["outcome"]["unpriced_calls"]]
    cost_in = sum(1 for p in priced if inside(
        bounds(p["prediction"]["cost"])[0], p["outcome"]["cost"], bounds(p["prediction"]["cost"])[1]))
    cost_ratio = [p["outcome"]["cost"] / p["prediction"]["cost"]["p50"]
                  for p in priced if p["prediction"]["cost"]["p50"]]
    inp_err = [(p["outcome"]["first_input"] - p["prediction"]["input_next"]) / p["outcome"]["first_input"]
               for p in scored if p["outcome"]["first_input"]]
    cache_ok = sum(1 for p in scored if _cache_as_said(
        p["prediction"]["cached"], p["outcome"]["first_cache_read"], p["outcome"]["first_input"]))
    fam = [p for p in scored if p["prediction"]["family_source"] != "declared"]
    turn_items = [(p["session_key"], bounds(p["prediction"]["turns"])[0], p["outcome"]["turns_main"],
                   bounds(p["prediction"]["turns"])[1], p["prediction"]["turns"]["p50"]) for p in with_interval]
    cost_items = [(p["session_key"], bounds(p["prediction"]["cost"])[0], p["outcome"]["cost"],
                   bounds(p["prediction"]["cost"])[1], p["prediction"]["cost"]["p50"]) for p in priced]
    return {
        "n": n,
        "turns_metrics": metrics.summarize(turn_items, bootstrap=BOOTSTRAP) if turn_items else None,
        "cost_metrics": metrics.summarize(cost_items, bootstrap=BOOTSTRAP) if cost_items else None,
        "with_interval": len(with_interval),
        "no_interval_spread": len(no_interval),
        "no_history": no_history,
        "turns_inside": turns_in,
        "turns_median_abs_error": percentile(turn_err, 50),
        "turns_median_actual_over_p50": percentile(turn_ratio, 50),
        "cost_scored": len(priced),
        "cost_inside": cost_in,
        "cost_median_actual_over_p50": percentile(cost_ratio, 50),
        "input_median_error": percentile(inp_err, 50),
        "input_within_5pct": sum(1 for e in inp_err if abs(e) <= 0.05),
        "input_scored": len(inp_err),
        "cache_as_said": cache_ok,
        "family_guessed": len(fam),
        "family_same_after": sum(1 for p in fam if p["prediction"]["family"] == p["outcome"]["family_after"]),
    }


def _cache_as_said(cached: int, read: int, first_input: int) -> bool:
    """Warm or cold as stated. Warm: 90% or more of what was said to be cached was read.
    Cold: under 10% of the call's input was read (a short shared prefix may survive)."""
    if cached == 0:
        return read < 0.1 * first_input
    return read >= 0.9 * cached


# ------------------------------------------------------------------ backtests


def backtest_computed(rows, task_stats: list[TaskStats], tasks: dict) -> dict[str, Any]:
    """For every prompt of yours that followed a call in the same session: was the
    computed part right? Uses only what estimate would have known at that moment."""
    from .estimate import chars_per_token

    ratio, _ = chars_per_token(rows)
    main_by_session: dict[str, list] = {}
    for r in rows:
        if r.trigger != "subagent":
            main_by_session.setdefault(r.session_key, []).append(r)
    first_of_task = {(t.session_key, t.task_id) for t in task_stats if t.tracked}
    errors, cold, warm = [], [], []
    cold_cases: list[tuple[str, float, float, str]] = []  # (why, read share, gap s, ttl)
    for key, main in main_by_session.items():
        ttl = "5m"
        for prev, cur in zip(main, main[1:]):
            if prev.cache_write:  # the TTL in force is the one of the last write, as estimate reads it
                ttl = "1h" if prev.cache_write_1h else "5m"
            if cur.trigger != "user_turn" or (key, cur.task_id) not in first_of_task or prev.task_id == cur.task_id:
                continue
            if cur.extra.get("after_compaction") or cur.input_total < 0.5 * prev.input_total or not prev.input_total:
                continue
            t0, t1 = parse_ts(prev.ts), parse_ts(cur.ts)
            if not (t0 and t1):
                continue
            gap = (t1 - t0).total_seconds()
            prompt_chars = int((tasks.get((cur.source, cur.task_id)) or {}).get("prompt_chars") or 0)
            predicted = prev.input_total + prev.output + round(prompt_chars / ratio)
            errors.append((cur.input_total - predicted) / cur.input_total)
            if prev.input_total < 4096:
                continue
            share = cur.cache_read / prev.input_total
            if cur.model != prev.model or gap > TTL_SECONDS[ttl]:
                cold.append(share)
                cold_cases.append(("model" if cur.model != prev.model else "idle", share, gap, ttl))
            else:
                warm.append(share)
    missed = [c for c in cold_cases if c[1] >= 0.1]
    return {
        "prompts": len(errors),
        "input_median_error": percentile(errors, 50),
        "input_p10_error": percentile(errors, 10),
        "input_p90_error": percentile(errors, 90),
        "input_within_5pct": sum(1 for e in errors if abs(e) <= 0.05),
        "cold_said": len(cold),
        "cold_was_cold": sum(1 for s in cold if s < 0.1),
        "cold_median_read_share": percentile(cold, 50),
        "cold_read_share_p90": percentile(cold, 90),
        # the "said cold" cases that were not: what they have in common
        "cold_missed": {
            "n": len(missed),
            "model_switch": sum(1 for c in missed if c[0] == "model"),
            "idle_5m": sum(1 for c in missed if c[0] == "idle" and c[3] == "5m"),
            "idle_1h": sum(1 for c in missed if c[0] == "idle" and c[3] == "1h"),
            "fully_warm": sum(1 for c in missed if c[1] >= 0.9),
            "read_share_p50": percentile([c[1] for c in missed], 50),
            "idle_gap_p10_s": percentile([c[2] for c in missed if c[0] == "idle"], 10),
            "idle_gap_p50_s": percentile([c[2] for c in missed if c[0] == "idle"], 50),
            "idle_gap_p90_s": percentile([c[2] for c in missed if c[0] == "idle"], 90),
            "idle_5m_gap_under_1h": sum(1 for c in missed if c[0] == "idle" and c[3] == "5m" and c[2] <= 3600),
        },
        "expiry": _expiry_stats(rows),
        "warm_said": len(warm),
        "warm_was_warm": sum(1 for s in warm if s >= 0.9),
        "warm_median_read_share": percentile(warm, 50),
    }


def _expiry_stats(rows) -> dict[str, Any]:
    """After every idle expiry in the history (not only before a prompt): how much
    was still read from cache, and was another session active just before?"""
    cases = expiries(rows)
    active = other_session_active(rows, cases)
    w = [c for c, a in zip(cases, active) if a]
    wo = [c for c, a in zip(cases, active) if not a]
    return {
        "n": len(cases),
        "read_p10": percentile([c.read for c in cases], 10),
        "read_p50": percentile([c.read for c in cases], 50),
        "read_p90": percentile([c.read for c in cases], 90),
        "other_active": len(w),
        "other_active_read10": sum(1 for c in w if c.read >= 0.1 * c.context),
        "other_active_read_p50": percentile([c.read for c in w], 50),
        "alone": len(wo),
        "alone_read10": sum(1 for c in wo if c.read >= 0.1 * c.context),
        "alone_read_p50": percentile([c.read for c in wo], 50),
    }


def backtest_turns(task_stats: list[TaskStats]) -> dict[str, Any]:
    """Each task's turns against the interval (INTERVAL) the tasks before it would have given.

    Two bases. "all": every earlier task; nothing about the task itself is used.
    "family": earlier tasks of its family, as estimate does; but here the family
    is the one guessed from the task's own first tool calls, which estimate
    cannot know (it guesses from your text), so this one is optimistic.
    """
    ts = sorted((t for t in task_stats if t.tracked and t.turns_main and t.first_input), key=lambda t: t.started)
    seen_all: list[int] = []
    seen_fam: dict[str, list[int]] = {}
    res = {b: {"n": 0, "inside": 0, "spread": 0, "items": []} for b in ("all", "family")}
    for t in ts:
        for b, basis in (("all", seen_all), ("family", seen_fam.get(t.family or "?", []))):
            if len(basis) < MIN_BASIS:
                continue
            p10, p50, p90 = (percentile_sorted(basis, q) for q in (INTERVAL[0], 50, INTERVAL[1]))
            r = res[b]
            if p10 and p90 / p10 >= SPREAD_LIMIT:  # type: ignore[operator]
                r["spread"] += 1
                continue
            r["n"] += 1
            r["inside"] += int(p10 <= t.turns_main <= p90)  # type: ignore[operator]
            r["items"].append((t.session_key, p10, t.turns_main, p90, p50))
        bisect.insort(seen_all, t.turns_main)
        bisect.insort(seen_fam.setdefault(t.family or "?", []), t.turns_main)
    for r in res.values():
        items = r.pop("items")
        r["metrics"] = metrics.summarize(items, bootstrap=BOOTSTRAP) if items else None
        r["median_abs_error"] = r["metrics"]["abs_error"] if items else None
    return res


# ------------------------------------------------------------------ turns per family


def build_turns(store: Store, prices: PriceTable) -> dict[str, Any]:
    tasks = store.tasks()
    ts = [t for t in group_tasks(enrich(store.records(), prices, tasks), tasks)
          if t.tracked and t.turns_main and t.first_input]
    fams = {}
    for fam in list(FAMILIES) + [None]:
        mine = ts if fam is None else [t for t in ts if t.family == fam]
        fams[fam or "all"] = {
            "tasks": len(mine),
            "declared": sum(1 for t in mine if t.family_source == "declared"),
            "turns_main": distribution([t.turns_main for t in mine]),
            "turns_all_calls": distribution([t.turns for t in mine]),
        }
    return {"families": fams, "spread_limit": SPREAD_LIMIT}


def verdict(d: Optional[dict]) -> str:
    if not d:
        return "no tasks"
    if d["n"] < MIN_BASIS:
        return f"too few (estimate needs {MIN_BASIS})"
    if d["spread"]:
        return "too spread: no interval given"
    if d["ratio"] < 4:
        return "tight"
    return "wide"


def render_turns(rep: dict[str, Any]) -> str:
    out = ["tokentrail turns: how many model calls a task takes, by family", ""]
    rows = []
    for fam, d in rep["families"].items():
        m, a = d["turns_main"], d["turns_all_calls"]
        if not m:
            rows.append([fam, "0", "", "", "", "", "", "no tasks"])
            continue
        rows.append([fam, f"{d['tasks']:,}", _n(m["lo"]), _n(m["p50"]), _n(m["hi"]),
                     "x" + (f"{m['ratio']:.0f}" if m["ratio"] != float("inf") else "?"),
                     f"{_n(a['p50'])} / {_n(a['hi'])}", verdict(m)])
    lo_l, hi_l = (f"p{x:g}" for x in INTERVAL)
    out += _table(["family", "tasks", lo_l, "median", hi_l, f"{hi_l}/{lo_l}", f"all calls p50/{hi_l}", ""],
                  rows, {1, 2, 3, 4, 5, 6})
    out.append("")
    out.append(f"  {INTERVAL_LABEL.replace('-', ' / median / ')}: main-thread model calls per task, the ones that each re-read the whole")
    out.append("  context, so the cost of a task is roughly turns x context size. All calls adds sub-agents.")
    out.append(f"  A family whose high bound is {rep['spread_limit']}x its low one or more (two orders of magnitude) gets no")
    out.append("  interval from `estimate`: one that wide would cover everything and say nothing.")
    declared = sum(d["declared"] for f, d in rep["families"].items() if f != "all")
    total = rep["families"]["all"]["tasks"]
    out.append(f"  Families: {declared:,} of {total:,} declared with `tokentrail tag`, the rest guessed from each")
    out.append("  task's first tool calls. `estimate` guesses from your text instead; `score` says how often")
    out.append("  the two agree.")
    return "\n".join(out)


def render_score(rep: dict[str, Any]) -> str:
    r, s, bt = rep["recorded"], rep["scored"], rep["backtest"]
    out = ["tokentrail score: what estimate said vs what happened", ""]
    st = r["by_status"]
    out.append(f"Recorded estimates: {r['total']} ({_fmt_counts(r['by_origin'])}); "
               f"{st.get('scored', 0)} scored, {st.get('linked', 0)} waiting for their task to finish, "
               f"{st.get('open', 0)} waiting for a task to start, {st.get('unmatched', 0)} never followed by a task"
               + (f", {st['superseded']} replaced by a later estimate of the same task" if st.get("superseded") else ""))
    out.append("")
    if not s["n"]:
        out.append("No estimate scored yet. Every `estimate` and every prompt seen by the hook is recorded;")
        out.append("each is scored once the task it preceded is over. Until then the turn estimate is unvalidated.")
    else:
        out.append(f"Last {s['n']} scored estimates")
        n_i = s["with_interval"]
        out.append(f"  turns inside the interval     {_of(s['turns_inside'], n_i)}   (it aims at 80%)")
        out.append(f"  turns, median error           {_num(s['turns_median_abs_error'])} turns; "
                   f"median actual / predicted median x{_num(s['turns_median_actual_over_p50'], 2)}")
        out.append(f"  cost inside its band          {_of(s['cost_inside'], s['cost_scored'])}   "
                   f"median actual / predicted x{_num(s['cost_median_actual_over_p50'], 2)} "
                   "(the band leaves out output and sub-agents)")
        if s.get("turns_metrics"):
            out += metric_lines(s["turns_metrics"], "turns", "    ")
        if s.get("cost_metrics"):
            out.append("  cost band, same measures:")
            out += metric_lines(s["cost_metrics"], "$", "    ")
        if s["no_interval_spread"]:
            out.append(f"  no interval given (spread)    {s['no_interval_spread']}")
        if s["no_history"]:
            out.append(f"  no history to estimate from   {s['no_history']}")
        out.append(f"  next-call input (computed)    median error {_pct(s['input_median_error'])}, "
                   f"within 5%: {_of(s['input_within_5pct'], s['input_scored'])}")
        out.append(f"  cache warm/cold as stated     {_of(s['cache_as_said'], s['n'])}")
        if s["family_guessed"]:
            out.append(f"  family from your text = family from the task's tools: "
                       f"{_of(s['family_same_after'], s['family_guessed'])}")
    c = bt["computed"]
    out.append("")
    out.append("Backtest on your history: the computed part (what estimate would have said before each prompt)")
    if c["prompts"]:
        out.append(f"  next-call input = last input + last answer + your text: {c['prompts']:,} prompts, "
                   f"median error {_pct(c['input_median_error'])} (p10 {_pct(c['input_p10_error'])}, "
                   f"p90 {_pct(c['input_p90_error'])}), within 5%: {_of(c['input_within_5pct'], c['prompts'])}")
        out.append(f"  said cold (idle past the TTL or model switch): {_of(c['cold_was_cold'], c['cold_said'])} "
                   f"read under 10% of the context from cache (median {_share(c['cold_median_read_share'])})")
        m = c["cold_missed"]
        if m["n"]:
            out.append(f"    the {m['n']} that read 10% or more: {m['model_switch']} after a model switch, "
                       f"{m['idle_5m']} idle past a 5m TTL ({m['idle_5m_gap_under_1h']} of them within an hour), "
                       f"{m['idle_1h']} idle past a 1h TTL")
            out.append(f"    they read a median {_share(m['read_share_p50'])} of the context from cache, "
                       f"{m['fully_warm']} read 90% or more; idle time p10 {_dur(m['idle_gap_p10_s'])}, "
                       f"median {_dur(m['idle_gap_p50_s'])}, p90 {_dur(m['idle_gap_p90_s'])}")
        e = c["expiry"]
        if e["n"]:
            out.append(f"  after every idle expiry ({e['n']:,} calls): tokens still read from cache p10 "
                       f"{_tok(e['read_p10'])}, median {_tok(e['read_p50'])}, p90 {_tok(e['read_p90'])}")
            out.append(f"    another session used the same model within the TTL before: {e['other_active']:,} calls, "
                       f"{_of(e['other_active_read10'], e['other_active'])} read 10% or more "
                       f"(median {_tok(e['other_active_read_p50'])} tokens)")
            out.append(f"    no other session active: {e['alone']:,} calls, {_of(e['alone_read10'], e['alone'])} "
                       f"read 10% or more (median {_tok(e['alone_read_p50'])} tokens)")
        out.append(f"  said warm: {_of(c['warm_was_warm'], c['warm_said'])} read 90% or more "
                   f"(median {_share(c['warm_median_read_share'])})")
    else:
        out.append("  no prompt follows an earlier call in the same session yet")
    t = bt["turns"]
    out.append("")
    out.append("Backtest on your history: turn intervals, each task against the tasks before it")
    for b, label in (("all", "basis: all earlier tasks"),
                     ("family", "basis: earlier tasks of its family (optimistic: family guessed from its own tools)")):
        d = t[b]
        line = (f"  {label}: {_of(d['inside'], d['n'])} inside {INTERVAL_LABEL}, median error "
                f"{_num(d['median_abs_error'])} turns")
        if d["spread"]:
            line += f"; {d['spread']} without an interval (too spread)"
        out.append(line)
        if d.get("metrics"):
            out += metric_lines(d["metrics"], "turns", "    ")
    out.append("  (95% intervals in brackets: sessions resampled, not tasks, since tasks of one session are alike)")
    return "\n".join(out)


def metric_lines(m: dict, unit: str, indent: str) -> list[str]:
    """Coverage against nominal, interval score, errors, each with its session-bootstrap 95% interval."""
    def ci(key: str, fmt) -> str:
        c = m.get(key + "_ci")
        return f" [{fmt(c[0])}, {fmt(c[1])}]" if c else ""

    def pct(x):
        return "?" if x is None else f"{100 * x:.0f}%"

    def num(x):
        return "?" if x is None else (f"{x:.2f}" if unit == "$" and abs(x) < 10 else f"{x:.1f}")

    def lg(x):
        return "?" if x is None else f"{x:+.2f}"

    le = m.get("log_error")
    ratio = f" (median estimate x{math.exp(le):.2f} the actual)" if le is not None else ""
    return [
        f"{indent}coverage {pct(m['coverage'])}{ci('coverage', pct)} vs nominal {pct(m['nominal'])}; "
        f"{m['n']} tasks in {m['sessions']} sessions",
        f"{indent}interval score (Winkler, lower is better) {num(m['interval_score'])}{ci('interval_score', num)}; "
        f"median width {num(m['width'])} {unit}",
        f"{indent}median error {num(m['abs_error'])}{ci('abs_error', num)} {unit}; "
        f"median log(estimate/actual) {lg(le)}{ci('log_error', lg)}{ratio}",
    ]


def _fmt_counts(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "none"


def _of(k: int, n: int) -> str:
    return f"{k} of {n}" + (f" ({100 * k / n:.0f}%)" if n else "")


def _num(x: Optional[float], digits: int = 1) -> str:
    return "?" if x is None else f"{x:.{digits}f}"


def _pct(x: Optional[float]) -> str:
    return "?" if x is None else f"{100 * x:+.1f}%"


def _tok(x: Optional[float]) -> str:
    return "?" if x is None else f"{round(x):,}"


def _dur(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return f"{x / 60:.0f} min" if x < 7200 else f"{x / 3600:.1f} h"


def _share(x: Optional[float]) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


def _n(x: Optional[float]) -> str:
    return "?" if x is None else str(round(x))

