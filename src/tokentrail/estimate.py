"""Command 2: what the next prompt will cost, before you send it.

Two kinds of numbers, kept apart:

computed   the input of the next call: the real input of the session's last
           call (from the transcript), plus what you are adding. And how much
           of it is already cached.
predicted  turns and output, as the 10th-90th percentile of your own past
           tasks of the same family. Never a mean: the tail is long.

Plus the frame: an exact floor (the first call's input, paid whatever
happens) and a ceiling (max_tokens on every turn up to the turn limit).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .analysis import (
    TTL_SECONDS, Row, cache_events, enrich, group_tasks, parse_ts, percentile, ttl_of,
)
from .prices import PriceTable
from .report import fmt_cost, fmt_tokens
from .store import Store

DEFAULT_CHARS_PER_TOKEN = 2.5  # observed 1.8-2.7 on code and JSON with current tokenizers
MIN_CALIBRATION = 10
MIN_FAMILY_SAMPLES = 8
LARGE_CONTEXT = 100_000
PREFIX_FILES = {"CLAUDE.md", "CLAUDE.local.md", ".mcp.json", "settings.json", "settings.local.json"}

_FAMILY_WORDS = {
    "review": r"\b(review|relis|relecture|relire|audit|vérifie|verifie|check the code|code review)\w*",
    "refactor": r"\b(refactor|refonte|réécri|reecri|rewrite|rename|renomme|migrate|migre|implement|implémente|ajoute|add|fix|corrige)\w*",
    "measure": r"\b(measure|mesure|bench\w*|profil\w*|how (fast|slow|many)|combien|count|compte|perf)\b",
}


@dataclass
class EstimateInput:
    text: str = ""
    add_files: tuple[Path, ...] = ()
    session: Optional[str] = None  # session id prefix; default: latest for cwd
    family: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    max_turns: Optional[int] = None
    edits: tuple[str, ...] = ()  # files you are about to change
    cwd: Optional[str] = None
    now: Optional[datetime] = None


def guess_family_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for fam, pat in _FAMILY_WORDS.items():
        if re.search(pat, t):
            return fam
    return "question" if t.strip() else None


def _pick_session(store: Store, prefix: Optional[str], cwd: Optional[str]) -> Optional[str]:
    rows = store.db.execute(
        "SELECT s.session_key, s.session_id, s.cwd, MAX(r.ts) AS last "
        "FROM sessions s JOIN records r ON r.session_key = s.session_key "
        "GROUP BY s.session_key ORDER BY last DESC"
    ).fetchall()
    if prefix:
        hits = [r for r in rows if r["session_id"].startswith(prefix)]
        return hits[0]["session_key"] if hits else None
    if cwd:
        for r in rows:
            if r["cwd"] and os.path.realpath(r["cwd"]) == os.path.realpath(cwd):
                return r["session_key"]
    return rows[0]["session_key"] if rows else None


def chars_per_token(rows: list[Row]) -> tuple[float, int]:
    """Calibrate from history: input growth between consecutive calls vs chars added.

    Claude Code also injects reminders we don't see as chars, so this ratio
    errs low, i.e. the delta errs high. That's the safe side.
    """
    # Each call also adds a fixed overhead (tool-result wrapper, reminders), so
    # small results say little: keep results of 1,000+ chars and weight by size.
    chars = tokens = n = 0
    prev_by_session: dict[str, Row] = {}
    for r in rows:
        if r.trigger == "subagent":
            continue
        prev = prev_by_session.get(r.session_key)
        prev_by_session[r.session_key] = r
        added = int(r.extra.get("added_chars") or 0)
        if prev is None or r.trigger != "tool_call" or added < 1000 or r.extra.get("after_compaction"):
            continue
        grew = r.input_total - prev.input_total - prev.output
        if grew > 0:
            chars, tokens, n = chars + added, tokens + grew, n + 1
    if n < MIN_CALIBRATION:
        return DEFAULT_CHARS_PER_TOKEN, n
    return min(8.0, max(1.5, chars / tokens)), n


def build(store: Store, prices: PriceTable, inp: EstimateInput) -> dict[str, Any]:
    now = inp.now or datetime.now(timezone.utc)
    tasks = store.tasks()
    all_rows = enrich(store.records(), prices, tasks)
    session_key = _pick_session(store, inp.session, inp.cwd or os.getcwd())
    if session_key is None:
        return {"error": "no session found; run Claude Code first, or pass --session"}
    srows = [r for r in all_rows if r.session_key == session_key]
    main = [r for r in srows if r.trigger != "subagent"]
    if not main:
        return {"error": f"session {session_key} has no main-thread calls"}
    last = main[-1]
    model = inp.model or last.model
    price = prices.lookup(model)

    # ------------------------------------------------------------ computed
    ratio, n_cal = chars_per_token(all_rows)
    file_chars = 0
    unreadable_files = []
    for f in inp.add_files:
        try:
            file_chars += len(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            unreadable_files.append(str(f))
    text_tokens = round(len(inp.text) / ratio)
    file_tokens = round(file_chars / ratio)
    base = last.input_total + last.output
    input_next = base + text_tokens + file_tokens

    ttl = ttl_of(main)
    t_last = parse_ts(last.ts)
    gap = (now - t_last).total_seconds() if t_last else float("inf")
    warm = gap <= TTL_SECONDS[ttl] and model == last.model
    cached = last.input_total if warm else 0
    to_write = input_next - cached
    mult = prices.write_1h if ttl == "1h" else prices.write_5m

    def in_cost(cache_read: int, write: int) -> Optional[float]:
        if not price:
            return None
        return (cache_read * price.cache_read + write * price.input * mult) / 1e6

    floor_cost = in_cost(cached, to_write)
    warm_cost = in_cost(last.input_total, input_next - last.input_total)

    # ----------------------------------------------------------- predicted
    family = inp.family
    family_source = "declared" if family else "guessed from your text"
    if family is None:
        family = guess_family_from_text(inp.text)
    # past tasks; the one still running in this session is not history yet
    hist = [t for t in group_tasks(all_rows, tasks) if t.first_input > 0 and t.task_id != last.task_id]
    fam_hist = [t for t in hist if t.family == family]
    basis = fam_hist if len(fam_hist) >= MIN_FAMILY_SAMPLES else hist
    basis_label = (
        f"{len(fam_hist)} past '{family}' tasks" if basis is fam_hist
        else f"all {len(hist)} past tasks ('{family}' has only {len(fam_hist)}, need {MIN_FAMILY_SAMPLES})"
    )

    def band(vals: list[float]) -> Optional[tuple[float, float]]:
        if not vals:
            return None
        return (percentile(vals, 10), percentile(vals, 90))  # type: ignore[return-value]

    turns = band([t.turns for t in basis])
    turns_main = band([t.turns_main for t in basis])
    output = band([t.output for t in basis])
    in_ratio = band([t.input_total / t.first_input for t in basis])
    cost_ratio = band([t.cost / t.first_input for t in basis if not t.unpriced])
    expected = None
    if in_ratio and output:
        expected = {
            "turns": turns, "turns_main": turns_main,
            "input": (in_ratio[0] * input_next, in_ratio[1] * input_next),
            "output": output,
            "cost": (cost_ratio[0] * input_next, cost_ratio[1] * input_next) if cost_ratio else None,
        }

    # -------------------------------------------------------------- ceiling
    max_tokens = (
        inp.max_tokens
        or _int_env("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        or (price.max_output if price else None)
        or 32_000
    )
    max_turns = inp.max_turns or (int(max(t.turns_main for t in basis)) if basis else 1)
    n, m = max_turns, max_tokens
    ceil_in = n * input_next + m * n * (n - 1) // 2
    ceil_out = n * m
    ceil_cost = (ceil_in * price.input * mult + ceil_out * price.output) / 1e6 if price else None

    # ------------------------------------------------------------- warnings
    warnings = []
    if not warm and model == last.model and t_last:
        warnings.append(
            f"Your last call was {_ago(gap)} ago and the {ttl} cache has expired: this call "
            f"writes the whole {fmt_tokens(input_next)}-token context again, "
            f"{fmt_cost((floor_cost or 0) - (warm_cost or 0))} more than with a warm cache."
        )
    elif warm and TTL_SECONDS[ttl] - gap < 60:
        warnings.append(
            f"The {ttl} cache expires in {int(TTL_SECONDS[ttl] - gap)} s: send now or pay "
            f"to re-write {fmt_tokens(input_next)} tokens."
        )
    if model != last.model:
        warnings.append(
            f"Switching model ({last.model} -> {model}) breaks the cache prefix: the whole "
            f"{fmt_tokens(input_next)}-token context is written again."
        )
    for e in inp.edits:
        if Path(e).name in PREFIX_FILES or "/.claude/" in f"/{e}":
            warnings.append(
                f"{e} is part of the prompt prefix: changing it breaks the cache prefix, and "
                f"the next call that reloads it re-writes the whole context "
                f"(~{fmt_tokens(input_next)} tokens, {fmt_cost(in_cost(0, input_next))})."
            )
    breaks = [e for e in cache_events(main, prices) if e.kind == "break"]
    if breaks:
        warnings.append(
            f"This session already broke its cache prefix {len(breaks)} time(s) "
            f"({fmt_cost(sum(e.extra_cost or 0 for e in breaks))} extra): model switches, tool "
            "or settings changes mid-session do that."
        )
    files_note = _files_warning(store, session_key, main, fam_hist, family)
    if files_note:
        warnings.append(files_note)
    if input_next > LARGE_CONTEXT:
        per_turn = in_cost(input_next, 0)
        warnings.append(
            f"Large context ({fmt_tokens(input_next)}): every further turn re-reads it "
            f"(~{fmt_cost(per_turn)} per turn from cache"
            + (f", {int(turns_main[1])} turns at p90" if turns_main else "")
            + "). /compact or a fresh session resets it."
        )
    sub_share = _share([t.sub_tokens for t in fam_hist], [t.tokens for t in fam_hist])
    if fam_hist and sub_share >= 30:
        warnings.append(
            f"On '{family}' tasks, {sub_share:.0f}% of your tokens went to sub-agents. "
            "Each one starts with its own full context."
        )
    if unreadable_files:
        warnings.append("Could not read: " + ", ".join(unreadable_files))

    return {
        "session": last.session_id[:8],
        "project": last.project,
        "model": model,
        "computed": {
            "last_call_input": last.input_total,
            "last_call_output": last.output,
            "your_text_tokens": text_tokens,
            "added_files_tokens": file_tokens,
            "input_next": input_next,
            "cached": cached,
            "to_write": to_write,
            "cache_ttl": ttl,
            "seconds_since_last_call": None if gap == float("inf") else round(gap),
            "chars_per_token": round(ratio, 2),
            "chars_per_token_samples": n_cal,
        },
        "predicted": {
            "family": family,
            "family_source": family_source,
            "basis": basis_label,
            "samples": len(basis),
            **({k: v for k, v in expected.items()} if expected else {}),
        },
        "frame": {
            "floor": {"input": input_next, "cost": floor_cost},
            "ceiling": {
                "input": ceil_in, "output": ceil_out, "cost": ceil_cost,
                "max_tokens": max_tokens, "max_turns": max_turns,
            },
        },
        "warnings": warnings,
    }


def _files_warning(store: Store, session_key: str, main: list[Row], fam_hist, family) -> Optional[str]:
    # only what is still in context: after the last compaction
    since = ""
    for prev, cur in zip(main, main[1:]):
        if cur.extra.get("after_compaction") or cur.input_total < prev.input_total * 0.5:
            since = cur.ts
    events = [e for e in store.file_events(session_key) if (e["ts"] or "") >= since]
    read_at: dict[str, str] = {}
    touched_after: set[str] = set()
    for e in events:
        p = e["path"]
        if e["action"] == "read":
            if p in read_at:
                touched_after.add(p)
            read_at.setdefault(p, e["ts"] or "")
        elif e["action"] == "write" and p in read_at:
            touched_after.add(p)
    loaded = len(read_at)
    unused = loaded - len(touched_after)
    if loaded < 10 or unused / loaded < 0.7:
        return None
    msg = (
        f"{loaded} files are loaded in this session's context; {unused} were read once and "
        "never edited or opened again, yet every call re-reads them."
    )
    rate = _edit_rate(store, {t.task_id for t in fam_hist})
    if rate is not None and len(fam_hist) >= 5:
        msg += f" On '{family}' tasks you edit {rate:.0f}% of the files you read."
    return msg


def _edit_rate(store: Store, task_ids: set[str]) -> Optional[float]:
    reads: dict[str, set[str]] = {}
    writes: dict[str, set[str]] = {}
    for e in store.file_events():
        if e["task_id"] not in task_ids:
            continue
        bucket = reads if e["action"] == "read" else writes if e["action"] == "write" else None
        if bucket is not None:
            bucket.setdefault(e["task_id"], set()).add(e["path"])
    total = sum(len(v) for v in reads.values())
    if not total:
        return None
    edited = sum(len(v & writes.get(k, set())) for k, v in reads.items())
    return 100 * edited / total


def _share(parts: list[int], wholes: list[int]) -> float:
    w = sum(wholes)
    return 100 * sum(parts) / w if w else 0.0


def _int_env(name: str) -> Optional[int]:
    v = os.environ.get(name, "")
    return int(v) if v.isdigit() else None


def _ago(seconds: float) -> str:
    if seconds < 120:
        return f"{int(seconds)} s"
    if seconds < 7200:
        return f"{int(seconds // 60)} min"
    return f"{seconds / 3600:.1f} h"


def render(est: dict[str, Any]) -> str:
    if "error" in est:
        return f"tokentrail estimate: {est['error']}"
    c, p, f = est["computed"], est["predicted"], est["frame"]
    out = [f"tokentrail estimate: session {est['session']} ({est['project']}), {est['model']}", ""]
    out.append("Computed (from the session's last real call)")
    out.append(f"  last call input            {c['last_call_input']:>12,}")
    out.append(f"  + its answer               {c['last_call_output']:>12,}")
    out.append(f"  + your text                {c['your_text_tokens']:>12,}")
    if c["added_files_tokens"]:
        out.append(f"  + added files              {c['added_files_tokens']:>12,}")
    out.append(f"  = next call input          {c['input_next']:>12,}")
    out.append(
        f"    already cached           {c['cached']:>12,}   (TTL {c['cache_ttl']}, last call "
        f"{_ago(c['seconds_since_last_call'] or 0)} ago)"
    )
    out.append(f"    written to cache         {c['to_write']:>12,}")
    if c["your_text_tokens"]:
        out.append(
            f"  Your text is {c['your_text_tokens']:,} tokens; the call sends {c['input_next']:,}: "
            "system prompt, tools, files and history ride along."
        )
    cal = (
        f"calibrated on {c['chars_per_token_samples']} of your calls"
        if c["chars_per_token_samples"] >= MIN_CALIBRATION else "default, too little history to calibrate"
    )
    out.append(f"  (text -> tokens at {c['chars_per_token']} chars/token, {cal}; no network tokenizer)")
    out.append("")
    out.append(f"Predicted (p10-p90 of {p['basis']}; family '{p['family']}', {p['family_source']})")
    if "turns" in p and p["turns"]:
        out.append(f"  model calls                {_band(p['turns'], _n)}   (main thread {_band(p['turns_main'], _n)})")
        out.append(f"  input over the task        {_band(p['input'], fmt_tokens)}")
        out.append(f"  output over the task       {_band(p['output'], fmt_tokens)}")
        if p.get("cost"):
            out.append(f"  cost                       {_band(p['cost'], fmt_cost)}")
    else:
        out.append("  not enough history yet")
    out.append("")
    ce = f["ceiling"]
    out.append("Frame")
    out.append(f"  floor (exact)    {fmt_cost(f['floor']['cost']):>10}   the first call's input, paid whatever happens")
    out.append(
        f"  ceiling          {fmt_cost(ce['cost']):>10}   {ce['max_turns']} turns x {ce['max_tokens']:,} max_tokens, "
        "context re-sent each turn"
    )
    out.append("                                tool results and sub-agents have no fixed cap and are not in it")
    if est["warnings"]:
        out.append("")
        out.append("Warnings")
        for w in est["warnings"]:
            out.append(f"  ! {w}")
    return "\n".join(out)


def _band(b, fmt) -> str:
    if not b:
        return "?"
    lo, hi = b
    return f"{fmt(lo)} - {fmt(hi)}"


def _n(x: float) -> str:
    return str(round(x))
