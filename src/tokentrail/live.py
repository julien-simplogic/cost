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
from pathlib import Path
from typing import Any, Optional

from . import estimate, paths, prices
from .collectors import claude_code
from .errors import TokentrailError
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


def statusline(raw_stdin: str) -> str:
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
    ctx = data.get("context_window") if isinstance(data.get("context_window"), dict) else {}
    pct = ctx.get("used_percentage")
    size = ctx.get("context_window_size")
    if isinstance(pct, (int, float)):
        parts.append(f"ctx {pct:.0f}%")
    elif isinstance(size, int) and size > 0:
        parts.append(f"ctx {100 * (last.input_total + last.output_total) / size:.0f}%")
    else:
        parts.append(f"ctx {fmt_tokens(last.input_total + last.output_total)}")

    total_in = sum(r.input_total for r in res.records)
    if total_in:
        parts.append(f"cache {100 * sum(r.input_cache_read for r in res.records) / total_in:.0f}%")

    cost = 0.0
    unpriced = 0
    for r in res.records:
        c = table.cost(r.model, new=r.input_new, cache_read=r.input_cache_read,
                       cache_write=r.input_cache_write, cache_write_1h=r.input_cache_write_1h,
                       output=r.output_total)
        if c is None:
            unpriced += 1
        else:
            cost += c
    recovered = any(r.output_recovered for r in res.records)
    parts.append(f"{fmt_cost(cost)}{'*' if unpriced else ''}{' (incl. recovered)' if recovered else ''}")

    if res.checks:
        parts.append("counter ok" if all(c.input_matches for c in res.checks) else "COUNTER MISMATCH")
    else:
        parts.append("unverified")
    if res.stats.not_understood:
        parts.append(f"{res.stats.not_understood} unread")
    return "tokentrail | " + " | ".join(parts)


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
    return json.dumps({"systemMessage": hook_message(est)})


def hook_message(est: dict[str, Any]) -> str:
    c, p, f = est["computed"], est["predicted"], est["frame"]
    line = (
        f"tokentrail: next call {c['input_next']:,} tokens in ({c['cached']:,} cached), "
        f"floor {fmt_cost(f['floor']['cost'])}"
    )
    if p.get("cost"):
        lo, hi = p["cost"]
        basis = f"{p['samples']} past '{p['family']}' tasks" if p.get("basis_is_family") else (
            f"all {p['samples']} past tasks, too few '{p['family']}' ones")
        line += f"; this task p10-p90 {fmt_cost(lo)}-{fmt_cost(hi)} ({basis})"
    lines = [line] + [f"  ! {w}" for w in est["warnings"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ setup


def setup_snippet() -> str:
    exe = "tokentrail"
    return json.dumps({
        "statusLine": {"type": "command", "command": f"{exe} statusline"},
        "hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": f"{exe} hook prompt", "timeout": 10}]}
        ]},
    }, indent=2)


def where_settings() -> str:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return str(Path(base) / "settings.json")
