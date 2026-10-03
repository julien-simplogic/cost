"""`tokentrail diagnose <session>`: why one session does or doesn't match Claude Code's counter.

Prints numbers and model names only, never prompt or file content, so the
output can be pasted into a bug report.
"""

from __future__ import annotations

from pathlib import Path

from .collectors import claude_code
from .errors import TokentrailError
from .record import totals_match


def run(root: Path, prefix: str) -> str:
    hits = [sf for sf in claude_code.discover(root) if sf.session_id.startswith(prefix)]
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
    if not res.checks:
        out.append("  no cost-state counter in this session: nothing to compare with")
        return "\n".join(out)
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
    verdict = "match" if totals_match(res.checks) else "MISMATCH"
    out.append(f"  input + cache summed over models: {verdict}")
    return "\n".join(out)
