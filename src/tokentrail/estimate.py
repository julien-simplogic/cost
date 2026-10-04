"""Command 2: what the next message costs, before you send it.

Two kinds of numbers, kept apart:

certain    arithmetic on a state the transcript shows: the input of the next
           call (the real input of the session's last call, plus what you are
           adding), how much of it is still cached given the idle time and the
           cache TTL, and what re-writing the rest costs.
estimated  the number of turns the task will take, as the 10th, 50th and 90th
           percentiles of your own past tasks of the same family, times the
           context size. Output is not predicted. This part is unvalidated until
           `tokentrail score` has compared enough estimates with what happened;
           every estimate is recorded for that (numbers only, never the text).

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
    TTL_SECONDS, Row, cache_events, distribution, enrich, expiries, group_tasks, is_prefix_file,
    parse_ts, percentile, ttl_of,
)
from .prices import PriceTable
from .report import fmt_cost, fmt_tokens
from .errors import TokentrailError
from .store import Store

DEFAULT_CHARS_PER_TOKEN = 2.5  # observed 1.8-2.7 on code and JSON with current tokenizers
MIN_CALIBRATION = 10
MIN_FAMILY_SAMPLES = 8
LARGE_CONTEXT = 100_000

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
    transcripts: Optional[Path] = None  # where past prompts are read from (default: Claude Code's)
    neighbour_budget_s: Optional[float] = None  # the hook gives up on neighbours past this


def guess_family_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for fam, pat in _FAMILY_WORDS.items():
        if re.search(pat, t):
            return fam
    return "question" if t.strip() else None


def _pick_session(store: Store, prefix: Optional[str], cwd: Optional[str]) -> tuple[Optional[str], str]:
    """(session key, how it was chosen)."""
    rows = store.db.execute(
        "SELECT s.session_key, s.session_id, s.cwd, MAX(r.ts) AS last "
        "FROM sessions s JOIN records r ON r.session_key = s.session_key "
        "GROUP BY s.session_key ORDER BY last DESC"
    ).fetchall()
    if prefix:
        hits = [r for r in rows if r["session_id"].startswith(prefix)]
        return (hits[0]["session_key"] if hits else None), "given"
    if cwd:
        here = _norm(cwd)
        for r in rows:
            if r["cwd"] and _norm(r["cwd"]) == here:
                return r["session_key"], "this directory"
    return (rows[0]["session_key"] if rows else None), "latest"


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


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
    session_key, chosen = _pick_session(store, inp.session, inp.cwd or os.getcwd())
    if session_key is None:
        if inp.session:
            raise TokentrailError(f"No session id starts with {inp.session!r}.",
                                  "Session ids are shown by `tokentrail report`.")
        raise TokentrailError("No session to start from.", "Run Claude Code once, then try again.")
    srows = [r for r in all_rows if r.session_key == session_key]
    main = [r for r in srows if r.trigger != "subagent"]
    if not main:
        raise TokentrailError(
            "That session has no call of its own (only sub-agent calls), so there is no last input to start from.",
            "Pick another session with --session.",
        )
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
    all_tasks = [t for t in group_tasks(all_rows, tasks) if t.first_input > 0 and t.tracked]
    hist = [t for t in all_tasks if t.task_id != last.task_id]
    fam_hist = [t for t in hist if t.family == family]
    ranks = task_ranks(all_tasks)
    rank_next = sum(1 for t in all_tasks if t.session_key == session_key) + 1
    slice_next = rank_slice(rank_next)
    rank_hist = [t for t in hist if rank_slice(ranks[t.task_id]) == slice_next]
    near_hist: list = []
    if inp.text.strip() and not inp.family:
        from . import paths, similar

        docs = similar.prompt_words(inp.transcripts or paths.claude_code_dir(), inp.neighbour_budget_s)
        if docs:
            by_id = {t.task_id: t for t in hist}
            ids = similar.nearest(similar.words(inp.text), {i: w for i, w in docs.items() if i in by_id})
            near_hist = [by_id[i] for i in ids]
    # A family guessed from your text is recorded (score checks it) but not used as the
    # basis: on real history, text-guessed families did not separate turns (all medians 4-5,
    # except 'review' at 11 with p10-p90 2-57). A family you declare is used.
    if inp.family and len(fam_hist) >= MIN_FAMILY_SAMPLES:
        basis, basis_label = fam_hist, f"{len(fam_hist)} past '{family}' tasks"
    elif inp.family:
        basis, basis_label = hist, (f"all {len(hist)} past tasks ('{family}' has only {len(fam_hist)}, "
                                    f"need {MIN_FAMILY_SAMPLES})")
    elif near_hist:
        # Your past prompts most alike in words (read in memory, nothing kept): on real history
        # the best sign before sending, where 30 close enough exist (scripts/study.py 9b).
        basis, basis_label = near_hist, f"the {len(near_hist)} past prompts most like this one"
    elif len(rank_hist) >= MIN_RANK_SAMPLES:
        # The one sign that held up in the leak-free backtest (scripts/study.py 6): where the
        # task falls in its session. The first one is the long one (median 22 turns, against
        # 4 to 7 for later ones, on the author's history).
        basis, basis_label = rank_hist, f"{len(rank_hist)} past tasks that were {RANK_WORDS[slice_next]}"
    else:
        basis, basis_label = hist, f"all {len(hist)} past tasks"

    # Turns, not output: the number of main-thread calls decides how many times
    # the context is re-read. Their spread decides whether an interval means anything.
    turns = distribution([t.turns_main for t in basis])
    turns_all = distribution([t.turns for t in basis])
    cost = None
    if turns and price and floor_cost is not None:
        reread = input_next * price.cache_read / 1e6

        def cost_at(n: float) -> float:
            return floor_cost + max(n - 1, 0) * reread  # type: ignore[operator]
        cost = {"lo": cost_at(turns["lo"]), "p50": cost_at(turns["p50"]), "hi": cost_at(turns["hi"])}

    # -------------------------------------------------------------- ceiling
    max_tokens = (
        inp.max_tokens
        or _int_env("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        or (price.max_output if price else None)
        or 32_000
    )
    max_turns = inp.max_turns or (int(max(t.turns_main for t in basis)) if basis else 1)
    turns_basis = "given" if inp.max_turns else ("your max for this basis" if basis else "no history: pass --max-turns")
    n, m = max_turns, max_tokens
    ceil_in = n * input_next + m * n * (n - 1) // 2
    ceil_out = n * m
    ceil_cost = (ceil_in * price.input * mult + ceil_out * price.output) / 1e6 if price else None

    # ------------------------------------------------------------- warnings
    certain = _certain(gap, ttl, warm, model, last, input_next, cached, floor_cost, warm_cost, t_last)
    if not warm and model == last.model and t_last:
        # Measured, not assumed: after an expiry, part of the prompt can still come from cache
        past = [e.read for e in expiries(all_rows) if e.model == model]
        if len(past) >= MIN_CALIBRATION:
            kept = min(int(percentile(past, 50) or 0), input_next)
            if kept:
                certain.append(
                    f"After your {len(past)} past expiries on this model, a median {kept:,} tokens were still "
                    f"read from cache anyway; that would make it {fmt_cost(in_cost(kept, input_next - kept))}.")
    warnings = []
    edited_prefix = [e for e in inp.edits if is_prefix_file(e)]
    if edited_prefix:
        warnings.append(_prefix_edit_note(store, prices, edited_prefix, input_next, in_cost(0, input_next)))
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
            + (f", {int(turns['hi'])} turns at p{turns['q'][1]:g}" if turns else "")
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
        "made_at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "session": last.session_id[:8],
        "session_key": session_key,
        "prompt_chars": len(inp.text),
        "certain": certain,
        "session_chosen": chosen,
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
            "validated": False,
            "rank_in_session": rank_next,
            "basis_kind": ("family" if basis is fam_hist else "neighbours" if basis is near_hist
                           else "rank" if basis is rank_hist else "all"),
            "family": family,
            "family_source": family_source,
            "basis": basis_label,
            "basis_is_family": basis is fam_hist,
            "samples": len(basis),
            "turns": turns,
            "turns_all_calls": turns_all,
            "cost": cost,
            "method": "main-thread turns x next-call context at the cache-read price, plus the first call; "
                      "output and sub-agents not included",
        },
        "frame": {
            "floor": {"input": input_next, "cost": floor_cost},
            "ceiling": {
                "input": ceil_in, "output": ceil_out, "cost": ceil_cost,
                "max_tokens": max_tokens, "max_turns": max_turns, "max_turns_basis": turns_basis,
            },
        },
        "warnings": warnings,
    }


MIN_RANK_SAMPLES = 30  # as the study: a slice with fewer past tasks falls back to all of them
RANK_SLICES = ((1, "1st"), (3, "2-3"), (10, "4-10"), (30, "11-30"))
RANK_WORDS = {"1st": "the first of their session", "2-3": "the 2nd or 3rd of their session",
              "4-10": "the 4th to 10th of their session", "11-30": "the 11th to 30th of their session",
              "31+": "the 31st or later of their session"}


def rank_slice(rank: int) -> str:
    for limit, label in RANK_SLICES:
        if rank <= limit:
            return label
    return "31+"


def task_ranks(task_stats) -> dict[str, int]:
    """Position of each task in its session, 1 for the first."""
    by: dict[str, list] = {}
    for t in task_stats:
        by.setdefault(t.session_key, []).append(t)
    out = {}
    for ts in by.values():
        for i, t in enumerate(sorted(ts, key=lambda t: (t.started, t.task_id)), 1):
            out[t.task_id] = i
    return out


def _certain(gap, ttl, warm, model, last, input_next, cached, floor_cost, warm_cost, t_last) -> list[str]:
    """What the transcript and the cache rules decide alone: no history, no guess."""
    out = []
    if model != last.model:
        out.append(
            f"Switching model ({last.model} -> {model}): nothing is cached for the new model. "
            f"Your next message writes the whole context, {input_next:,} tokens, {fmt_cost(floor_cost)}."
        )
    elif not t_last:
        out.append(f"Your next message sends {input_next:,} tokens; the last call has no timestamp, "
                   "so whether the cache is still warm is unknown.")
    elif not warm:
        out.append(
            f"Idle for {_ago(gap)}: the {ttl} cache has expired. Your next message re-writes the "
            f"context, up to {input_next:,} tokens, {fmt_cost(floor_cost)} "
            f"({fmt_cost((floor_cost or 0) - (warm_cost or 0))} more than with a warm cache)."
        )
    else:
        left = TTL_SECONDS[ttl] - gap
        out.append(
            f"Cache warm ({ttl} TTL, expires in {_ago(left)}): your next message reads {cached:,} of its "
            f"{input_next:,} tokens from cache, {fmt_cost(floor_cost)}."
            + (" Send now or it re-writes them all." if left < 60 else "")
        )
    return out


def _prefix_edit_note(store: Store, prices: PriceTable, edits: list[str], input_next: int,
                      full_write: Optional[float]) -> str:
    """What editing CLAUDE.md or settings did on your own history: measured, not assumed.

    When Claude Code re-reads these files, and where they sit in the prompt, is
    not written in the transcripts, so the position where the prefix breaks is
    not known. What is known: after each time a tool wrote one of them, did the
    next call find its cache?
    """
    from .analysis import BreakCauses, break_causes

    tasks = store.tasks()
    rows = enrich(store.records(), prices, tasks)
    by: dict[str, list[Row]] = {}
    for r in rows:
        if r.trigger != "subagent":
            by.setdefault(r.session_key, []).append(r)
    total = BreakCauses()
    events: dict[str, list] = {}
    for e in store.file_events():
        events.setdefault(e["session_key"], []).append(e)
    for key, main in by.items():
        if key in events:
            total.add(break_causes(main, events[key], prices))
    names = ", ".join(edits)
    worst = f"if it does, the whole {fmt_tokens(input_next)}-token context is written again ({fmt_cost(full_write)})"
    if total.prefix_writes == 0:
        return (f"{names}: whether Claude Code re-reads it mid-session, and where it sits in the prompt, "
                f"is not in the transcripts, and your history has no such edit to measure; {worst}.")
    return (f"{names}: in your history, {total.prefix_writes_then_break} of {total.prefix_writes} edits of such "
            f"files during a session were followed by a cache break on the next call; {worst}.")


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
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86_400:.1f} days"


def render(est: dict[str, Any]) -> str:
    c, p, f = est["computed"], est["predicted"], est["frame"]
    out = [f"tokentrail estimate: session {est['session']} ({est['project']}), {est['model']}"]
    if est.get("session_chosen") == "latest":
        out.append("  no session was started in this directory: using your most recent one (--session to pick)")
    out.append("")
    out += est["certain"]
    out.append("")
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
    out.append(f"Unvalidated estimate: turns from {p['basis']} (family '{p['family']}', {p['family_source']})")
    out += ["  " + line for line in turns_lines(p)]
    out.append("  Not yet compared with outcomes: `tokentrail score` says how past estimates fared.")
    out.append("")
    ce = f["ceiling"]
    out.append("Frame")
    out.append(f"  floor (exact)    {fmt_cost(f['floor']['cost']):>10}   the first call's input, paid whatever happens")
    out.append(
        f"  ceiling          {fmt_cost(ce['cost']):>10}   {ce['max_turns']} turns ({ce['max_turns_basis']}) x "
        f"{ce['max_tokens']:,} max_tokens, context re-sent each turn"
    )
    out.append("                                tool results and sub-agents have no fixed cap and are not in it")
    if est["warnings"]:
        out.append("")
        out.append("Warnings")
        for w in est["warnings"]:
            out.append(f"  ! {w}")
    return "\n".join(out)


def turns_lines(p: dict[str, Any]) -> list[str]:
    """The turn estimate in words; no interval when the history is too spread to give one."""
    t = p.get("turns")
    if not t:
        return [f"no past task to compare with yet ({p['samples']} found); the floor and ceiling below hold."]
    if t["spread"]:
        return [f"turns too spread to estimate: p{t['q'][0]:g} {_n(t['lo'])}, median {_n(t['p50'])}, "
                f"p{t['q'][1]:g} {_n(t['hi'])} "
                f"(x{t['ratio']:.0f}, two orders of magnitude or more); no interval given"]
    out = [f"model calls (main thread)  {_n(t['lo'])} - {_n(t['hi'])}, median {_n(t['p50'])}   "
           f"(p{t['q'][0]:g}-p{t['q'][1]:g}, aimed at 80%)"]
    if p.get("cost"):
        c = p["cost"]
        out.append(f"cost of those turns        {fmt_cost(c['lo'])} - {fmt_cost(c['hi'])}, median {fmt_cost(c['p50'])}"
                   "   (turns x context; output and sub-agents not included)")
    return out


def _n(x: float) -> str:
    return str(round(x))
