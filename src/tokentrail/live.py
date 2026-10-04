"""Live display inside Claude Code: a status line and a UserPromptSubmit hook.

Both read the JSON Claude Code sends on stdin (documented at
code.claude.com/docs/en/statusline and /hooks) and the session's own
transcript. Neither can break a session: any failure prints a short
notice (status line) or nothing at all (hook), and the hook always exits 0.

The hook answers with ``systemMessage`` only. Plain stdout from a
UserPromptSubmit hook is added to the model's context, which would cost
tokens on every prompt: the opposite of what this tool is for.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import estimate, paths, predictions, prices
from .collectors import claude_code
from .errors import TokentrailError
from .analysis import TTL_SECONDS, parse_ts
from .record import totals_match
from .ingest import ingest_claude_code
from .report import fmt_cost, fmt_tokens
from .store import Store


def session_files(transcript_path: str) -> claude_code.SessionFiles:
    main = Path(transcript_path)
    subs = sorted((main.parent / main.stem / "subagents").glob("*.jsonl"))
    return claude_code.SessionFiles(main.stem, main, subs, project_dir=main.parent.name)


def _read_stdin_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------------------ status line


def statusline(raw_stdin: str, now: Optional[datetime] = None) -> str:
    """One line: what changes and what you can act on. Problems only when there are some.

    Always: context used, your plan's usage windows when Claude Code passes them
    (Pro/Max, documented as rate_limits), and the session's value at API rates.
    Only when something is wrong: cold or missed cache, a mismatch with Claude
    Code's counter, unread transcript lines, a model missing from the price file.
    Everything that is merely reassuring belongs to `tokentrail check`.
    """
    data = _read_stdin_json(raw_stdin)
    tp = data.get("transcript_path")
    if not isinstance(tp, str) or not Path(tp).is_file():
        return "tokentrail: no transcript yet"
    res = claude_code.parse_session(session_files(tp))
    if not res.records:
        return "tokentrail: no model call yet"
    table = prices.load()
    main = [r for r in res.records if r.trigger != "subagent"]
    last = main[-1] if main else res.records[-1]

    parts = []
    ctx = _obj(data, "context_window")
    pct, size = ctx.get("used_percentage"), ctx.get("context_window_size")
    if isinstance(pct, (int, float)):
        parts.append(f"ctx {pct:.0f}%")
    elif isinstance(size, int) and size > 0:
        parts.append(f"ctx {100 * (last.input_total + last.output_total) / size:.0f}%")
    else:
        parts.append(f"ctx {fmt_tokens(last.input_total + last.output_total)}")

    limits = _obj(data, "rate_limits")
    for key, label in (("five_hour", "5h"), ("seven_day", "7d")):
        used = _obj(limits, key).get("used_percentage")
        if isinstance(used, (int, float)):
            parts.append(f"{label} {used:.0f}%")

    # Cache: computed from the last call's time and the TTL it was written with, nothing estimated.
    # Claude Code's own prompt_cache diagnostics, when present, win over this.
    cache = _obj(data, "prompt_cache")
    expiry = cache_clock(main, table, now)
    cc_warm = cache.get("warm") if cache else None
    if expiry and expiry["warm"] and cc_warm is not False:
        parts.append(f"cache {_left(expiry['left_s'])} left")

    cost, unpriced = 0.0, 0
    for r in res.records:
        c = table.cost(r.model, new=r.input_new, cache_read=r.input_cache_read,
                       cache_write=r.input_cache_write, cache_write_1h=r.input_cache_write_1h,
                       output=r.output_total)
        if c is None:
            unpriced += 1
        else:
            cost += c
    parts.append(f"{fmt_cost(cost)} at API rates")

    alerts = []
    if cc_warm is False and isinstance(cache.get("recache_tokens_if_cold"), int):
        n = cache["recache_tokens_if_cold"]
        alerts.append(f"cache cold: next message re-caches {fmt_tokens(n)}" + _write_cost(last, table, n, expiry))
    elif expiry and not expiry["warm"] and cc_warm is not True:
        alerts.append(f"cache expired: next message re-writes up to {fmt_tokens(expiry['tokens'])}"
                      + _write_cost(last, table, expiry["tokens"], expiry))
    if cache:
        misses = cache.get("misses")
        if isinstance(misses, int) and misses > 0:
            causes = _obj(cache, "last_miss_cause").get("causes")
            why = f" (last: {', '.join(map(str, causes))})" if isinstance(causes, list) and causes else ""
            alerts.append(f"{misses} cache miss{'es' if misses > 1 else ''}{why}")
    elif len(main) >= 2:
        prev = main[-2]
        if prev.input_total >= 4096 and last.input_cache_read < 0.5 * prev.input_total \
                and last.input_total >= 0.5 * prev.input_total:
            alerts.append("cache missed on the last call")
    if res.checks and not totals_match(res.checks):
        alerts.append("totals disagree with Claude Code's counter: run tokentrail check")
    if res.stats.not_understood:
        alerts.append(f"{res.stats.not_understood} lines unread: run tokentrail check")
    if unpriced:
        alerts.append(f"{unpriced} calls unpriced: model missing from the price file")
    return " | ".join(parts + [f"! {a}" for a in alerts])


def cache_clock(main: list, table, now: Optional[datetime] = None) -> Optional[dict]:
    """Time left on the main thread's cache, or that it has expired, and what the next call
    would write: the last call's input plus its answer (your next text not included)."""
    if not main:
        return None
    last = main[-1]
    ttl = "5m"
    for r in reversed(main):
        if r.input_cache_write:
            ttl = "1h" if r.input_cache_write_1h else "5m"
            break
    t = parse_ts(last.timestamp)
    if t is None:
        return None
    now = now or datetime.now(timezone.utc)
    left = TTL_SECONDS[ttl] - (now - t).total_seconds()
    return {"ttl": ttl, "warm": left > 0, "left_s": max(left, 0.0),
            "tokens": last.input_total + last.output_total}


def _write_cost(last, table, tokens: int, expiry: Optional[dict]) -> str:
    p = table.lookup(last.model)
    if not p:
        return ""
    mult = table.write_1h if expiry and expiry["ttl"] == "1h" else table.write_5m
    return f", {fmt_cost(tokens * p.input * mult / 1e6)}"


def _left(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{seconds / 3600:.0f}h"


def _obj(d: Any, key: str) -> dict:
    v = d.get(key) if isinstance(d, dict) else None
    return v if isinstance(v, dict) else {}


# ------------------------------------------------------------------ prompt hook


def prompt_hook(raw_stdin: str) -> Optional[str]:
    """JSON to print for Claude Code, or None to print nothing."""
    data = _read_stdin_json(raw_stdin)
    tp, prompt = data.get("transcript_path"), data.get("prompt")
    if data.get("source") == "system" or not isinstance(tp, str) or not Path(tp).is_file():
        return None
    sf = session_files(tp)
    paths.ensure_data_dir()
    with Store(paths.db_path()) as store:
        # only this session: the rest of the history is whatever earlier runs ingested
        ingest_claude_code(store, Path(tp).parent.parent, sessions=[sf])
        try:
            est = estimate.build(store, prices.load(), estimate.EstimateInput(
                text=prompt if isinstance(prompt, str) else "",
                session=sf.session_id,
                cwd=data.get("cwd") if isinstance(data.get("cwd"), str) else None,
            ))
        except TokentrailError:
            return None  # first prompt of a session: nothing to start from yet
        try:
            predictions.record(store, est, "hook")  # so that `tokentrail score` can grade it later
        except Exception:  # noqa: BLE001 - never lose the message over the bookkeeping
            pass
    return json.dumps({"systemMessage": hook_message(est)})


def hook_message(est: dict[str, Any]) -> str:
    """First what is certain (idle time, cache, what re-writing costs), then the unvalidated estimate."""
    p = est["predicted"]
    lines = ["tokentrail: " + " ".join(est["certain"])]
    if p.get("turns"):
        basis = f"{p['samples']} past '{p['family']}' tasks" if p.get("basis_is_family") else (
            f"all {p['samples']} past tasks")
        t, c = p["turns"], p.get("cost")
        if t["spread"]:
            lines.append(f"  turns too spread to estimate ({basis}): p10 {t['p10']:.0f}, p90 {t['p90']:.0f}")
        else:
            line = f"  unvalidated estimate: {t['p10']:.0f}-{t['p90']:.0f} turns, median {t['p50']:.0f} ({basis})"
            if c:
                line += (f" x context = {fmt_cost(c['p10'])}-{fmt_cost(c['p90'])}, "
                         "output and sub-agents not included")
            lines.append(line)
    lines += [f"  ! {w}" for w in est["warnings"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ setup


def executable() -> str:
    """The command Claude Code should run: this very tokentrail, by absolute path,
    since a venv or pipx install is often not on the PATH Claude Code sees."""
    import shutil
    import sys

    me = Path(sys.argv[0])
    if me.name.startswith("tokentrail") and me.exists():
        return str(me.resolve())
    found = shutil.which("tokentrail")
    return found or "tokentrail"


def setup_snippet(exe: str = "") -> str:
    exe = exe or executable()
    if " " in exe:
        exe = f'"{exe}"'
    return json.dumps({
        "statusLine": {"type": "command", "command": f"{exe} statusline"},
        "hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": f"{exe} hook prompt", "timeout": 10}]}
        ]},
    }, indent=2)


def where_settings() -> str:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return str(Path(base) / "settings.json")
