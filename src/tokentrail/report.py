"""Command 1: where the tokens went, over a period."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Optional

from .analysis import Row, cache_events, enrich, group_tasks
from .classify import CATEGORIES, CATEGORY_LABELS
from .prices import PriceTable
from .record import counter_resets, gap_shape, increments
from .store import Store


def build(
    store: Store,
    prices: PriceTable,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    project: Optional[str] = None,
    top: int = 10,
    anonymize: bool = False,
    logged_only: bool = False,
) -> dict[str, Any]:
    tasks = store.tasks()
    rows = enrich(store.records(since, until, project), prices, tasks, logged_only)
    names = _Anonymizer(anonymize)

    tot_tokens = sum(r.tokens for r in rows)
    tot_cost = sum(r.cost or 0 for r in rows)

    cats: dict[str, dict[str, float]] = {c: defaultdict(float) for c in CATEGORIES}
    for r in rows:
        c = cats[r.category]
        c["turns"] += 1
        c["tokens"] += r.tokens
        c["cost"] += r.cost or 0
        c["recovered"] += r.recovered
    categories = [
        {
            "key": k,
            "label": CATEGORY_LABELS[k],
            "turns": int(v["turns"]),
            "tokens": int(v["tokens"]),
            "cost": round(v["cost"], 4),
            "recovered_output": int(v["recovered"]),
            "token_share": _share(v["tokens"], tot_tokens),
            "cost_share": _share(v["cost"], tot_cost),
        }
        for k, v in cats.items()
    ]

    by_session: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_session[r.session_key].append(r)
    sessions = []
    events = []
    for key, rs in by_session.items():
        main = [r for r in rs if r.trigger != "subagent"]
        events += cache_events(main, prices)
        cost = sum(r.cost or 0 for r in rs)
        sub = sum(r.cost or 0 for r in rs if r.trigger == "subagent")
        sessions.append({
            "session": names.session(rs[0].session_id),
            "project": names.project(rs[0].project),
            "started": rs[0].ts[:16].replace("T", " "),
            "turns": len(rs),
            "tasks": len({r.task_id for r in rs}),
            "tokens": sum(r.tokens for r in rs),
            "cost": round(cost, 4),
            "subagent_cost_share": _share(sub, cost),
        })
    sessions.sort(key=lambda s: s["cost"], reverse=True)

    task_stats = group_tasks(rows, tasks)
    task_stats.sort(key=lambda t: t.cost, reverse=True)
    top_tasks = [
        {
            "task": t.task_id[:8],
            "project": names.project(t.project),
            "session": names.session(t.rows[0].session_id),
            "started": t.started[:16].replace("T", " "),
            "family": t.family,
            "family_source": t.family_source,
            "command": t.command,
            "turns": t.turns,
            "turns_main": t.turns_main,
            "turns_subagent": t.turns_sub,
            "tokens": t.tokens,
            "recovered_output": t.recovered,
            "cost": round(t.cost, 4),
            "subagent_token_share": _share(t.sub_tokens, t.tokens),
        }
        for t in task_stats[:top]
    ]

    stats = store.parse_stats()
    recovered_rows = [r for r in rows if r.recovered]
    recovered_cost = 0.0
    for r in recovered_rows:
        p = prices.lookup(r.model)
        if p:
            recovered_cost += r.recovered * p.output / 1e6
    return {
        "period": {"since": since, "until": until, "project": project},
        "totals": {
            "turns": len(rows),
            "tasks": len(task_stats),
            "sessions": len(by_session),
            "tokens": tot_tokens,
            "input_new": sum(r.input_new for r in rows),
            "input_cache_read": sum(r.cache_read for r in rows),
            "input_cache_write": sum(r.cache_write for r in rows),
            "output": sum(r.output for r in rows),
            "output_reasoning": sum(r.reasoning or 0 for r in rows),
            "output_logged": sum(r.output_logged for r in rows),
            "output_recovered": sum(r.recovered for r in rows),
            "output_recovered_calls": len(recovered_rows),
            "output_recovered_cost": round(recovered_cost, 4),
            "logged_only": logged_only,
            "cost": round(tot_cost, 4),
            "unpriced_turns": sum(1 for r in rows if r.cost is None),
            "inexact_output_turns": sum(1 for r in rows if not r.output_exact),
            "inexact_output_logged": sum(r.output for r in rows if not r.output_exact),
        },
        "categories": categories,
        "sessions": sessions[:top],
        "tasks": top_tasks,
        "cache": {
            "breaks": sum(1 for e in events if e.kind == "break"),
            "break_cost": round(sum(e.extra_cost or 0 for e in events if e.kind == "break"), 4),
            "expiries": sum(1 for e in events if e.kind == "expired"),
            "expiry_cost": round(sum(e.extra_cost or 0 for e in events if e.kind == "expired"), 4),
        },
        "checks": _checks_summary(store, {r.session_key for r in rows}, names),
        "parsing": {
            "files": stats.files,
            "records": stats.lines,
            "used": stats.used,
            "ignored_by_design": sum(stats.ignored.values()),
            "not_understood": stats.not_understood,
            "unreadable": stats.unreadable,
            "unknown_types": stats.unknown_types,
            "unknown_with_tokens": stats.unknown_with_tokens,
            "versions": stats.versions,
            "not_understood_by_version": _not_understood_by_version(store),
        },
        "prices": {
            "file": prices.path,
            "verified_on": prices.verified_on.isoformat() if prices.verified_on else None,
            "stale": prices.is_stale(),
        },
    }


def _checks_summary(store: Store, session_keys: set[str], names: "_Anonymizer") -> dict[str, Any]:
    """Our sums vs Claude Code's own counter, for sessions in the period that have one."""
    by_session: dict[str, list] = {}
    for c in store.checks():
        if c["session_key"] in session_keys:
            by_session.setdefault(c["session_key"], []).append(c)

    def ic(c, side: str) -> int:
        return c[f"{side}_input"] + c[f"{side}_cache_read"] + c[f"{side}_cache_write"]

    one_sided: dict[str, set[str]] = {"counter": set(), "lines": set()}
    floor: dict[str, Any] = {"sessions": 0, "lines": [0, 0, 0, 0], "gap": [0, 0, 0, 0]}
    input_ok, mismatched = [], []
    for k, cs in by_session.items():
        counter_only = sorted(c["model"] for c in cs if ic(c, "ours") == 0 and ic(c, "source"))
        lines_only = sorted(c["model"] for c in cs if ic(c, "source") == 0 and ic(c, "ours"))
        one_sided["counter"].update(counter_only)
        one_sided["lines"].update(lines_only)
        if _rows_match(cs):
            input_ok.append(k)
            fg = floor_gap("exact", cs, {})
            floor["sessions"] += 1
            floor["lines"] = [a + b for a, b in zip(floor["lines"], fg[0])]
            floor["gap"] = [a + b for a, b in zip(floor["gap"], fg[1])]
            continue
        src = sum(ic(c, "source") for c in cs)
        ours = sum(ic(c, "ours") for c in cs)
        cov = json.loads(cs[0]["coverage"] or "{}")
        status, inc, shape = session_status(cov)
        fg = floor_gap(status, cs, cov)
        if fg:
            floor["sessions"] += 1
            floor["lines"] = [a + b for a, b in zip(floor["lines"], fg[0])]
            floor["gap"] = [a + b for a, b in zip(floor["gap"], fg[1])]
        mismatched.append({
            "status": status,
            "increments": inc,
            "last_calls": cov.get("counter_equals_last_calls"),
            "session": names.session(cs[0]["session_id"]), "project": names.project(cs[0]["project"] or "?"),
            "version": cs[0]["version"],
            "input_gap": ours - src,  # signed: positive = we count more than the counter
            "input_gap_pct": 100 * (ours - src) / src if src else None,
            "output_gap": sum(c["ours_output"] for c in cs) - sum(c["source_output"] for c in cs),
            "shape": shape,
            "counter_resets": inc["restarts"],
            "calls_after_counter": cov.get("main_calls_after_counter", 0) + cov.get("subagent_calls_after_counter", 0),
            "subagent_calls_unlinked": cov.get("subagent_calls_unlinked", 0),
            "models_counter_only": counter_only,
            "models_lines_only": lines_only,
        })
    mismatched.sort(key=lambda m: abs(m["input_gap_pct"] or 0), reverse=True)
    all_c = [c for cs in by_session.values() for c in cs]
    # Counting MORE than Claude Code is an error whatever its shape. Counting less is
    # usage no call line records only if the gap appeared at a few moments (hidden
    # calls); a gap that grows at every snapshot is a counting error too.
    def n(status: str) -> int:
        return sum(1 for m in mismatched if m["status"] == status)
    return {
        "sessions_in_period": len(session_keys),
        "sessions_checkable": len(by_session),
        "sessions_input_exact": len(input_ok),
        "sessions_over": n("over"),
        "sessions_spread": n("spread"),
        "sessions_increments_ok": n("increments_ok"),
        "sessions_run_exact": n("run_exact"),
        "sessions_lines_exact": n("lines_exact"),
        "sessions_single": n("single"),
        "sessions_restart": n("restart"),
        "sessions_under": len(mismatched),
        "models_counter_only": sorted(one_sided["counter"]),
        "models_lines_only": sorted(one_sided["lines"]),
        "floor": floor,
        "mismatched": mismatched,
        "source_output": sum(c["source_output"] for c in all_c),
        "ours_output_logged": sum(c["ours_output_logged"] for c in all_c),
        "ours_output": sum(c["ours_output"] for c in all_c),
    }


def floor_gap(status: str, rows, cov: dict) -> Optional[tuple[list[int], list[int]]]:
    """What the call lines miss, where the comparison means something.

    Returns (lines, counter minus lines) per field [input, cache read, cache write,
    output], or None. Only scopes where both sides cover the same calls count:
    an exact session (all models), the final run a restarted counter equals, and
    what both sides added between snapshots of one run. Absolute totals of other
    sessions mix in usage the counter carried in, or calls it never covered.
    """
    keys = ("input", "cache_read", "cache_write", "output")
    if status in ("exact", "lines_exact"):
        ours = [sum(r[f"ours_{k}"] for r in rows) for k in keys]
        src = [sum(r[f"source_{k}"] for r in rows) for k in keys]
        return ours, [b - a for a, b in zip(ours, src)]
    if status == "run_exact":
        last = cov.get("counter_equals_last_calls") or {}
        if "ours" in last and "counter" in last:
            return last["ours"], [b - a for a, b in zip(last["ours"], last["counter"])]
        return None
    if status == "increments_ok":
        fs = cov.get("field_series") or []
        runs = [(a, b) for a, b in zip(fs, fs[1:]) if sum(b[2][:3]) >= sum(a[2][:3])]
        if not runs:
            return None
        ours = [sum(b[1][i] - a[1][i] for a, b in runs) for i in range(4)]
        src = [sum(b[2][i] - a[2][i] for a, b in runs) for i in range(4)]
        return ours, [b - a for a, b in zip(ours, src)]
    return None


def session_status(cov: dict) -> tuple[str, dict, dict]:
    """How a session that disagrees with Claude Code's counter compares, snapshot to snapshot.

    Claude Code's counter belongs to a run, not to a file (it can carry usage in, or
    restart when a session is resumed): what was added between two snapshots of one
    run is the reliable reference.
    """
    series = cov.get("gap_series") or []
    inc = increments(series)
    shape = gap_shape(series)
    last = cov.get("counter_equals_last_calls")
    if last:
        # the last counter equals, to the token on three fields, the last n calls
        if last["calls"] == last["of"]:
            # every call line: no restart; the rest of the counter is calls no line records
            return "lines_exact", inc, shape
        # only a final run: the counter restarted, and on that run we count exactly like it
        return "run_exact", inc, shape
    if inc["pairs"]:
        if inc["ours"] > inc["counter"]:
            return "over", inc, shape  # more than Claude Code within one run: unexplained
        if shape["kind"] == "spread":
            return "spread", inc, shape
        return "increments_ok", inc, shape
    return ("restart" if inc["restarts"] else "single"), inc, shape


def _fmt_pct(p: Optional[float]) -> str:
    """Signed, and never rounded to a misleading zero."""
    if p is None:
        return "no counter input"
    if p != 0 and abs(p) < 0.01:
        return f"{'+' if p > 0 else '-'}<0.01%"
    return f"{p:+.2f}%"


def _rows_match(rows) -> bool:
    """Input and cache summed over every model of the session, ours vs Claude Code's counter."""
    return all(sum(r[f"source_{k}"] for r in rows) == sum(r[f"ours_{k}"] for r in rows)
               for k in ("input", "cache_read", "cache_write"))


def _not_understood_by_version(store: Store) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in store.sessions():
        st = json.loads(s["stats"])
        n = sum(st.get("unreadable", {}).values()) + sum(st.get("unknown_types", {}).values())
        if n:
            v = s["version"] or "unknown"
            out[v] = out.get(v, 0) + n
    return out


def _share(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


class _Anonymizer:
    """Stable fake names so a report can be shown without client names."""

    def __init__(self, on: bool):
        self.on = on
        self.projects: dict[str, str] = {}

    def project(self, name: str) -> str:
        if not self.on:
            return name
        if name not in self.projects:
            self.projects[name] = f"project-{chr(ord('A') + len(self.projects) % 26)}" + (
                str(len(self.projects) // 26) if len(self.projects) >= 26 else ""
            )
        return self.projects[name]

    def session(self, sid: str) -> str:
        if not self.on:
            return sid[:8]
        return hashlib.sha256(sid.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- rendering


def fmt_tokens(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def fmt_cost(c: Optional[float]) -> str:
    if c is None:
        return "?"
    return f"${c:,.2f}" if c >= 0.995 else f"${c:.3f}"


def _table(headers: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(cells)
        ).rstrip()

    return [line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]


def render(rep: dict[str, Any]) -> str:
    t, p = rep["totals"], rep["parsing"]
    period = rep["period"]
    out = []
    span = f"since {period['since'][:10]}" if period["since"] else "all history"
    if period["until"]:
        span += f" until {period['until'][:10]}"
    if period["project"]:
        span += f", projects matching '{period['project']}'"
    out.append(f"tokentrail report: {span}")
    out.append(verification_banner(rep["checks"]))
    out.append(
        f"{t['sessions']} sessions, {t['tasks']} tasks, {t['turns']} model calls, "
        f"{fmt_tokens(t['tokens'])} tokens, {fmt_cost(t['cost'])} at API prices"
    )
    out.append(
        f"  input: {fmt_tokens(t['input_cache_read'])} cache read, "
        f"{fmt_tokens(t['input_cache_write'])} cache write, {fmt_tokens(t['input_new'])} new; "
        f"output: {fmt_tokens(t['output'])} (of which reasoning {fmt_tokens(t['output_reasoning'])})"
    )
    out.append("")
    out.append("Where it went")
    rows = [
        [c["label"], str(c["turns"]),
         fmt_tokens(c["tokens"]) + ("\u2021" if c["recovered_output"] else " "), f"{c['token_share']:.0f}%",
         fmt_cost(c["cost"]), f"{c['cost_share']:.0f}%"]
        for c in sorted(rep["categories"], key=lambda c: c["cost"], reverse=True)
    ]
    out += _table(["consumer", "calls", "tokens", "share", "cost", "share"], rows, {1, 2, 3, 4, 5})
    out.append("")
    out.append("Most expensive tasks")
    rows = [
        [x["task"], x["project"], x["started"],
         (x["family"] or "?") + ("*" if x["family_source"] == "declared" else ""),
         f"{x['turns']} ({x['turns_subagent']} sub)",
         fmt_tokens(x["tokens"]) + ("\u2021" if x["recovered_output"] else " "), fmt_cost(x["cost"]),
         f"{x['subagent_token_share']:.0f}%"]
        for x in rep["tasks"]
    ]
    out += _table(
        ["task", "project", "started", "family", "calls", "tokens", "cost", "sub-agents"],
        rows, {4, 5, 6, 7},
    )
    out.append("  family: guessed from the first tool calls; * = declared with `tokentrail tag`")
    if rep["totals"]["output_recovered"]:
        out.append("  \u2021 includes output not taken from the call's own log line (see 'How sure' below)")
    out.append("")
    out.append("By session")
    rows = [
        [s["session"], s["project"], s["started"], str(s["tasks"]), str(s["turns"]),
         fmt_tokens(s["tokens"]), fmt_cost(s["cost"]), f"{s['subagent_cost_share']:.0f}%"]
        for s in rep["sessions"]
    ]
    out += _table(
        ["session", "project", "started", "tasks", "calls", "tokens", "cost", "sub-agents"],
        rows, {3, 4, 5, 6, 7},
    )
    c = rep["cache"]
    if c["breaks"] or c["expiries"]:
        out.append("")
        out.append(
            f"Cache: {c['expiries']} expiries (idle past the TTL) cost {fmt_cost(c['expiry_cost'])} extra; "
            f"{c['breaks']} breaks (prefix changed) cost {fmt_cost(c['break_cost'])} extra"
        )
    out.append("")
    out += _render_trust(rep)
    return "\n".join(out)


def verification_banner(ck: dict[str, Any]) -> str:
    """First thing a reader sees: are these totals checked, how did it go, which way are they wrong?

    Hard rules: "Verified" only when every checked session matches Claude Code's
    counter to the token AND every model is on both sides. Counting more than the
    counter between two snapshots of one run, or a gap that keeps growing from
    interval to interval, makes the totals unverifiable. It does not say which
    side is wrong: the counter's meaning across versions is not documented.
    """
    n, k, ok = ck["sessions_in_period"], ck["sessions_checkable"], ck["sessions_input_exact"]
    over, spread = ck.get("sessions_over", 0), ck.get("sessions_spread", 0)
    one_sided = ck.get("models_counter_only", []) + ck.get("models_lines_only", [])
    if k == 0:
        return (f"NOT VERIFIED: none of these {n} sessions carries Claude Code's own counter (cost-state), so "
                "nothing checks the totals below. They miss what call lines never record, so read them as a minimum.")
    unknown_meaning = ("Which side is wrong is not established: what Claude Code's counter means across its "
                       "versions is not documented. See 'How sure' at the end.")
    if over:
        return (f"COULD NOT VERIFY: in {over} of {k} checkable sessions these totals could not be verified against "
                "Claude Code's counter: between two of its snapshots tokentrail counts more than it, which no "
                f"known cause explains. {unknown_meaning}")
    if spread:
        return (f"COULD NOT VERIFY: in {spread} of {k} checkable sessions these totals could not be verified against "
                "Claude Code's counter: the gap grows from interval to interval, which hidden calls don't "
                f"explain. {unknown_meaning}")
    unchecked = f"; {n - k} sessions carry no counter and are unchecked" if k < n else ""
    names = (" Models seen on one side only (not compared): " + ", ".join(one_sided) + ".") if one_sided else ""
    inc_ok, single, restart = (ck.get("sessions_increments_ok", 0), ck.get("sessions_single", 0),
                               ck.get("sessions_restart", 0))
    run_exact, lines_exact = ck.get("sessions_run_exact", 0), ck.get("sessions_lines_exact", 0)
    if inc_ok or single or restart or run_exact or lines_exact:
        parts = [f"exact in {ok}"]
        if lines_exact:
            parts.append(f"every call line exact in {lines_exact}, the counter also holding calls no line records")
        if run_exact:
            parts.append(f"exact on the last run in {run_exact} (the counter restarted; its value equals the "
                         "last calls to the token)")
        if inc_ok:
            parts.append(f"consistent snapshot to snapshot in {inc_ok} (what both sides added between two "
                         "snapshots of one run agrees, apart from calls no line records)")
        if restart:
            parts.append(f"counter restarted in {restart}, nothing comparable")
        if single:
            parts.append(f"a single snapshot that disagrees in {single}, which can't be checked")
        return (f"Partly verified, {k} of {n} sessions carry Claude Code's counter: " + "; ".join(parts) + ". "
                "Never more than the counter between snapshots. The totals below are a MINIMUM: some usage "
                f"(tools' own model calls) is recorded by no call line.{names}{unchecked}")
    if one_sided:
        return f"Partly verified: totals match Claude Code's counter, but not model by model.{names}{unchecked}"
    if k < n:
        return (f"Partly verified: input matches Claude Code's own counter to the token in {k} of {n} "
                f"sessions; the other {n - k} carry no counter and are unchecked.")
    return f"Verified: input matches Claude Code's own counter to the token in all {n} sessions."


def _render_trust(rep: dict[str, Any]) -> list[str]:
    """What the numbers above rest on. Never omitted."""
    t, p, ck = rep["totals"], rep["parsing"], rep["checks"]
    out = ["How sure are these numbers"]
    out.append(
        f"  Read {p['files']} transcript files, {p['records']} records: {p['used']} used, "
        f"{p['ignored_by_design']} skipped by design, {p['not_understood']} not understood"
    )
    if p["not_understood"]:
        detail = {**p["unreadable"], **{f"type:{k}": v for k, v in p["unknown_types"].items()}}
        out.append("    not understood: " + ", ".join(f"{k}={v}" for k, v in sorted(detail.items())))
        if p["unknown_types"]:
            tok = p.get("unknown_with_tokens") or {}
            out.append(
                "    of the unknown record types, " + (
                    "these carry token fields and may hold usage counted nowhere: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(tok.items()))
                    if tok else "none carries a token field (usage, input_tokens, ...): nothing is missing from the totals")
            )
        by_v = p.get("not_understood_by_version") or {}
        if by_v:
            out.append("    by Claude Code version: " + ", ".join(f"{v}={n}" for v, n in sorted(by_v.items())))
    if t["logged_only"]:
        out.append("  Output: as logged on each call's own line (--logged-only); nothing recovered.")
    elif t["output_recovered"]:
        out.append(
            f"  \u2021 Output: {fmt_tokens(t['output_logged'])} as logged on each call's own line, plus "
            f"{fmt_tokens(t['output_recovered'])} recovered for {t['output_recovered_calls']} sub-agent calls "
            f"({fmt_cost(t['output_recovered_cost'])}) from the sub-agent's result in the parent transcript, "
            "because their own line was written mid-stream. Rows marked \u2021 include them; "
            "--logged-only leaves them out."
        )
    if t["inexact_output_turns"]:
        out.append(
            f"  {t['inexact_output_turns']} sub-agent calls were logged mid-stream with nothing to recover from: "
            f"their output ({fmt_tokens(t['inexact_output_logged'])} as logged) is a lower bound. Input is exact."
        )
    if ck["sessions_checkable"]:
        out.append(f"  Checked against Claude Code's own counter (cost-state records) in {ck['sessions_checkable']} of "
                   f"{ck['sessions_in_period']} sessions; the verdict is on the first line.")
        fl = ck["floor"]
        if fl["sessions"]:
            ln, gp = fl["lines"], fl["gap"]
            ic_lines, ic_gap = sum(ln[:3]), sum(gp[:3])
            pin = 100 * ic_gap / ic_lines if ic_lines else 0.0
            pout = 100 * gp[3] / ln[3] if ln[3] else 0.0
            out.append(
                f"    How far below the real usage the totals are: where both sides cover the same calls "
                f"({fl['sessions']} sessions: exact, exact on the last run, or between snapshots of one run), "
                f"Claude Code's counter holds {fmt_tokens(ic_gap)} more input and cache ({pin:+.2f}%) and "
                f"{fmt_tokens(gp[3])} more output ({pout:+.1f}%) than the call lines. That is usage no call line "
                "records (tools that run a model themselves, sub-agent output logged mid-stream), not in the "
                "totals or categories above. Read the totals as a floor by about that much."
            )
        else:
            out.append("    No session lets both sides cover the same calls: how much the call lines miss is unknown.")
        explained = [m for m in ck["mismatched"] if m["status"] in ("lines_exact", "run_exact", "increments_ok")]
        unresolved = [m for m in ck["mismatched"] if m not in explained]
        if explained:
            out.append(f"    {len(explained)} sessions differ from the counter in absolute totals only, and are explained "
                       "above (exact on the last run, every line exact, or consistent between snapshots); "
                       "`tokentrail report --json` lists them.")
        if unresolved:
            out.append("    sessions not resolved, largest first (input + cache, ours minus counter; + = we count more):")
            for m in unresolved[:15]:
                pct = _fmt_pct(m["input_gap_pct"])
                extra = []
                inc, sh = m["increments"], m["shape"]
                if m["status"] == "lines_exact":
                    extra.append("every call line matches the counter to the token; the counter's surplus is "
                                 "calls no line records")
                if m["status"] == "run_exact":
                    lc = m["last_calls"]
                    extra.append(f"the counter equals EXACTLY the last {lc['calls']} of {lc['of']} calls: it covers "
                                 "only a final run, counted exactly alike")
                if inc["pairs"]:
                    d = inc["ours"] - inc["counter"]
                    pct = _fmt_pct(100 * d / inc["counter"]) if inc["counter"] else "n/a"
                    extra.append(f"between snapshots: ours +{inc['ours']:,} vs counter +{inc['counter']:,} "
                                 f"({d:+,}, {pct}" + (", we count MORE: unexplained)" if d > 0 else ")"))
                if inc["offset"]:
                    extra.append(f"at the first snapshot the counter was {inc['offset']:+,} away from this file "
                                 + ("(usage carried in from outside it)" if inc["offset"] > 0 else
                                    "(it covered fewer calls: a restarted run)"))
                if m["counter_resets"]:
                    extra.append(f"counter went DOWN {m['counter_resets']} time(s): Claude Code restarted counting")
                if sh["kind"] == "spread":
                    extra.append(f"GAP GROWS in {sh['steps']} of {sh['intervals']} intervals (unexplained)")
                elif sh["kind"] == "concentrated":
                    extra.append(f"gap moved in {sh['steps']} of {sh['intervals']} intervals (hidden calls)")
                elif m["status"] == "single":
                    extra.append("a single snapshot: can't tell carried-in usage, a restart or an error apart "
                                 "(`tokentrail diagnose` looks for an exact match)")
                if m["calls_after_counter"]:
                    extra.append(f"{m['calls_after_counter']} calls after the counter (not compared)")
                if m["subagent_calls_unlinked"]:
                    extra.append(f"{m['subagent_calls_unlinked']} sub-agent calls not linked to a launch")
                if m["models_counter_only"]:
                    extra.append("models only in the counter: " + ", ".join(m["models_counter_only"]))
                if m["models_lines_only"]:
                    extra.append("models only in the lines: " + ", ".join(m["models_lines_only"]))
                out.append(
                    f"      {m['session']} ({m['project']}, {m['version']}): input {m['input_gap']:+,} ({pct}), "
                    f"output {m['output_gap']:+,}" + (f"; {'; '.join(extra)}" if extra else "")
                )
            if len(unresolved) > 15:
                out.append(f"      ... and {len(unresolved) - 15} more (`tokentrail report --json` lists all)")
    else:
        out.append("  No session in this period carries Claude Code's own counter: totals are unchecked.")
    if t["unpriced_turns"]:
        out.append(f"  {t['unpriced_turns']} calls use a model missing from the price file (cost not counted)")
    pr = rep["prices"]
    out.append(f"  Prices: {pr['file']}, verified {pr['verified_on'] or 'never'}")
    if pr["stale"]:
        out.append("    prices are more than 60 days old: check them, then `tokentrail prices --init` to edit")
    return out


# ---------------------------------------------------------------- check


def build_check(store: Store, prices: PriceTable) -> dict[str, Any]:
    """Everything needed to judge the numbers: per version, per month."""
    tasks = store.tasks()
    rows = enrich(store.records(), prices, tasks)
    sessions = {s["session_key"]: s for s in store.sessions()}
    checks: dict[str, list] = {}
    for c in store.checks():
        checks.setdefault(c["session_key"], []).append(c)

    versions: dict[str, dict[str, int]] = {}
    for key, s in sessions.items():
        v = versions.setdefault(s["version"] or "unknown", defaultdict(int))
        st = json.loads(s["stats"])
        v["sessions"] += 1
        v["records"] += st.get("lines", 0)
        v["not_understood"] += sum(st.get("unreadable", {}).values()) + sum(st.get("unknown_types", {}).values())
        cs = checks.get(key)
        if cs:
            v["checkable"] += 1
            exact = _rows_match(cs)
            v["input_exact"] += int(exact)
            v["output_gap"] += sum(c["source_output"] - c["ours_output"] for c in cs)

    # Does the counter read the same in every version? Sessions that ran entirely on
    # one version answer it without resumes across upgrades muddying the comparison.
    single: dict[str, dict[str, int]] = {}
    for key, s in sessions.items():
        cs = checks.get(key)
        st = json.loads(s["stats"])
        vs = st.get("versions") or {}
        if not cs or len(vs) != 1:
            continue
        d = single.setdefault(next(iter(vs)), defaultdict(int))
        d["sessions"] += 1
        if _rows_match(cs):
            d["exact"] += 1
        else:
            d[session_status(json.loads(s["coverage"] or "{}"))[0]] += 1

    months: dict[str, dict[str, float]] = {}
    for r in rows:
        m = months.setdefault(r.ts[:7], defaultdict(float))
        m["calls"] += 1
        m["input"] += r.input_total
        m["output"] += r.output
        m["tokens"] += r.tokens
        m["cost"] += r.cost or 0
        m["recovered"] += r.recovered
    return {
        "versions": {k: dict(v) for k, v in sorted(versions.items())},
        "single_version": {k: dict(v) for k, v in sorted(single.items())},
        "months": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sorted(months.items())},
    }


def render_check(chk: dict[str, Any]) -> str:
    out = ["tokentrail check", "", "By Claude Code version"]
    rows = [
        [v, str(int(d.get("sessions", 0))), str(int(d.get("records", 0))), str(int(d.get("not_understood", 0))),
         f"{int(d.get('checkable', 0))}", f"{int(d.get('input_exact', 0))}", fmt_tokens(d.get("output_gap", 0))]
        for v, d in chk["versions"].items()
    ]
    out += _table(["version", "sessions", "records", "not understood", "with counter", "input exact",
                   "output gap"], rows, {1, 2, 3, 4, 5, 6})
    out.append("  with counter: sessions carrying Claude Code's own cost-state counter; input exact: of those,")
    out.append("  sessions where input and cache match it to the token; output gap: output the counter has")
    out.append("  and the per-call lines don't (sub-agent calls logged mid-stream).")
    out.append("")
    out.append("Sessions that ran on a single version, with a counter (does the counter read the same everywhere?)")
    rows = [
        [v, str(d.get("sessions", 0)), str(d.get("exact", 0) + d.get("lines_exact", 0)), str(d.get("run_exact", 0)),
         str(d.get("increments_ok", 0)), str(d.get("single", 0) + d.get("restart", 0)),
         str(d.get("over", 0) + d.get("spread", 0))]
        for v, d in chk.get("single_version", {}).items()
    ]
    if rows:
        out += _table(["version", "sessions", "exact (lines)", "exact on last run", "consistent between snapshots",
                       "can't check", "unexplained"], rows, {1, 2, 3, 4, 5, 6})
    else:
        out.append("  none: every session with a counter mixes several Claude Code versions")
    out.append("")
    out.append("By month (compare with your plan's usage page)")
    rows = [
        [m, str(int(d["calls"])), fmt_tokens(d["input"]), fmt_tokens(d["output"]), fmt_tokens(d["tokens"]),
         fmt_cost(d["cost"]) + ("‡" if d.get("recovered") else "")]
        for m, d in chk["months"].items()
    ]
    out += _table(["month", "calls", "input", "output", "tokens", "API-price cost"], rows, {1, 2, 3, 4, 5})
    out.append("  Sessions on this machine only: cloud sessions run on Anthropic's machines, not in these files.")
    return "\n".join(out)
