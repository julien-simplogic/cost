"""`tokentrail diagnose <session>`: why one session does or doesn't match Claude Code's counter.

Prints numbers and model names only, never prompt or file content, so the
output can be pasted into a bug report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .collectors import claude_code
from .errors import TokentrailError
from .record import counter_resets, gap_shape, increments, totals_match


def raw_counter_records(main: Path) -> list[str]:
    """Every cost-state record of the file as written, with the timestamp of the line before it."""
    out, last_ts = [], "?"
    with main.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            if isinstance(obj.get("timestamp"), str):
                last_ts = obj["timestamp"]
            if obj.get("type") == "cost-state" and isinstance(obj.get("modelUsage"), dict):
                for model, u in obj["modelUsage"].items():
                    if isinstance(u, dict):
                        out.append(
                            f"  line {i:>6}  after {last_ts}  {model:30} in {u.get('inputTokens', 0):>12,}  "
                            f"cache read {u.get('cacheReadInputTokens', 0):>15,}  "
                            f"cache write {u.get('cacheCreationInputTokens', 0):>13,}  out {u.get('outputTokens', 0):>12,}")
    return out


def run(root: Path, prefix: str, raw: bool = False) -> str:
    sessions = claude_code.discover(root)
    hits = [sf for sf in sessions if sf.session_id.startswith(prefix)]
    if not hits:  # ids from `report --anonymize` are hashes of the real ones
        hits = [sf for sf in sessions
                if hashlib.sha256(sf.session_id.encode()).hexdigest().startswith(prefix)]
    if not hits:
        raise TokentrailError(f"No session id starts with {prefix!r}.", "Session ids are shown by `tokentrail report`.")
    if len(hits) > 1:
        raise TokentrailError(f"{len(hits)} sessions start with {prefix!r}.", "Give a longer prefix.")
    sf = hits[0]
    res = claude_code.parse_session(sf)
    st, cov = res.stats, res.coverage
    out = [f"session {sf.session_id[:8]}  Claude Code {', '.join(sorted(st.versions)) or '?'}"]
    out.append(f"  files: 1 main + {len(sf.subagents)} sub-agent "
               f"({cov.get('subagent_files_linked', 0)} linked to the call that launched them)")
    out.append(f"  records: {st.lines} read, {st.used} used, {sum(st.ignored.values())} skipped by design, "
               f"{st.not_understood} not understood")
    for t, n in sorted(st.unknown_types.items()):
        tok = st.unknown_with_tokens.get(t, 0)
        out.append(f"    unknown type {t}: {n}" + (f", {tok} of them carry token fields" if tok else ", no token fields"))
    if len(res.session_ids) > 1:
        out.append(f"  the main file holds records of {len(res.session_ids)} session ids "
                   "(a resumed or forked session repeats its parent's history)")
    main = [r for r in res.records if r.trigger != "subagent"]
    subs = [r for r in res.records if r.trigger == "subagent"]
    out.append(f"  calls: {len(main)} main, {len(subs)} sub-agent "
               f"({sum(1 for r in subs if not r.output_exact)} sub-agent calls logged mid-stream)")
    others = _other_files_with_this_id(sf)
    if others:
        out.append(f"  {len(others)} other session file(s) in this project carry this session's id "
                   f"({', '.join(others[:5])}{', ...' if len(others) > 5 else ''})")
    if not res.checks:
        out.append("  no cost-state counter in this session: nothing to compare with")
        return "\n".join(out)
    if cov.get("counter_model_names") != cov.get("line_model_names"):
        out.append(f"  model names as written: counter {', '.join(cov.get('counter_model_names', []))}; "
                   f"call lines {', '.join(cov.get('line_model_names', []))} (compared without [..] qualifiers)")
    out.append(f"  counter: {cov.get('counter_records', 0)} cost-state records; the last one is compared "
               f"with the {cov.get('calls_covered', 0)} calls made before it")
    out.append(f"    not compared: {cov.get('main_calls_after_counter', 0)} main calls and "
               f"{cov.get('subagent_calls_after_counter', 0)} sub-agent calls after it, "
               f"{cov.get('subagent_calls_unlinked', 0)} sub-agent calls not linked to a launch")
    out.append("")
    head = f"  {'model':34}{'field':13}{'counter':>14}{'ours':>14}{'ours - counter':>16}"
    out += [head, "  " + "-" * (len(head) - 2)]
    for c in res.checks:
        for label, src, ours in (
            ("input", c.source_input, c.ours_input),
            ("cache read", c.source_cache_read, c.ours_cache_read),
            ("cache write", c.source_cache_write, c.ours_cache_write),
            ("output", c.source_output, c.ours_output),
        ):
            out.append(f"  {c.model[:34]:34}{label:13}{src:>14,}{ours:>14,}{ours - src:>+16,}")
    if raw and sf.main:
        out.append("")
        out.append("  raw cost-state records, as written (the comparison uses the LAST one, never a sum):")
        out += raw_counter_records(sf.main)
    verdict = "equal" if totals_match(res.checks) else "different (absolute totals; see what each side added between snapshots)"
    out.append(f"  input + cache summed over models: {verdict}")
    series = cov.get("gap_series") or []
    if series:
        out.append("")
        out.append("  at each counter snapshot (input + cache, all models):")
        out.append(f"  {'#':>4}{'calls before':>14}{'ours':>18}{'counter':>18}{'ours - counter':>18}")
        prev = None
        for i, (calls, ours, src) in enumerate(series, 1):
            note = "   <- counter went DOWN: Claude Code restarted counting" if prev is not None and src < prev else ""
            out.append(f"  {i:>4}{calls:>14,}{ours:>18,}{src:>18,}{ours - src:>+18,}{note}")
            prev = src
        inc = increments(series)
        if inc["pairs"]:
            d = inc["ours"] - inc["counter"]
            out.append(f"  added between snapshots of one run: ours +{inc['ours']:,}, counter +{inc['counter']:,} "
                       f"({d:+,})" + ("  <- we count MORE: unexplained" if d > 0 else ""))
        if inc["offset"]:
            out.append(f"  at the first snapshot the counter is {inc['offset']:+,} away from this file "
                       + ("(usage carried in from outside it)" if inc["offset"] > 0 else "(it covers fewer calls)"))
        fs = cov.get("field_series") or []
        runs = [(a, b) for a, b in zip(fs, fs[1:]) if sum(b[2][:3]) >= sum(a[2][:3])]
        if runs:
            out.append("  added between snapshots of one run, field by field (does the counter count like us?):")
            out.append(f"    {'field':12}{'ours':>16}{'counter':>16}{'ours - counter':>16}")
            for i, label in enumerate(("input", "cache read", "cache write", "output")):
                o = sum(b[1][i] - a[1][i] for a, b in runs)
                c = sum(b[2][i] - a[2][i] for a, b in runs)
                out.append(f"    {label:12}{o:>+16,}{c:>+16,}{o - c:>+16,}")
        m = cov.get("counter_equals_last_calls")
        if m:
            out.append(f"  the last counter equals EXACTLY the last {m['calls']} of {m['of']} calls before it "
                       f"(input, cache read, cache write), from {m['since']}: it counts only a final run")
        elif series and series[-1][1] > series[-1][2]:
            out.append("  no run of final calls adds up exactly to the last counter")
        sh = gap_shape(series)
        out.append(f"  gap shape: {sh['kind']}"
                   + (f" (moved in {sh['steps']} of {sh['intervals']} intervals)" if sh.get("steps") else "")
                   + f"; counter restarts: {counter_resets(series)}")
    return "\n".join(out)


def _other_files_with_this_id(sf: claude_code.SessionFiles) -> list[str]:
    """Session files of the same project whose first records carry this session's id."""
    if not sf.main:
        return []
    found = []
    for f in sorted(sf.main.parent.glob("*.jsonl")):
        if f == sf.main:
            continue
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for _, line in zip(range(30), fh):
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get("sessionId") == sf.session_id:
                        found.append(f.stem[:8])
                        break
        except OSError:
            continue
    return found
