"""Check the de-duplication claim on your own Claude Code transcripts.

Standalone: Python standard library only, does not import tokentrail, reads
~/.claude/projects (or the folder you pass), writes nothing.

For every session that carries Claude Code's own counter (`cost-state`
records), it compares three numbers over the calls made before the last
counter: the counter itself, the naive sum of `usage` over every line, and
the sum after keeping one line per `message.id`.

    python verify_dedup.py [~/.claude/projects]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIELDS = [("input_tokens", "inputTokens"), ("cache_read_input_tokens", "cacheReadInputTokens"),
          ("cache_creation_input_tokens", "cacheCreationInputTokens"), ("output_tokens", "outputTokens")]


def lines(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def usage_of(o: dict):
    m = o.get("message") if o.get("type") == "assistant" else None
    if not isinstance(m, dict) or not isinstance(m.get("usage"), dict) or m.get("model") == "<synthetic>":
        return None
    return m.get("model"), m.get("id"), m["usage"]


def session(main: Path):
    records = list(lines(main))
    stops = [i for i, o in enumerate(records) if o.get("type") == "cost-state" and isinstance(o.get("modelUsage"), dict)]
    if not stops:
        return None
    stop = stops[-1]
    before = records[:stop]
    # sub-agents launched before the counter was written
    launched = {b.get("id") for o in before if o.get("type") == "assistant"
                for b in (o.get("message", {}).get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"}
    calls = list(before)
    for meta in (main.parent / main.stem / "subagents").glob("*.meta.json"):
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("toolUseId") in launched:
                calls += list(lines(meta.with_name(meta.name.replace(".meta.json", ".jsonl"))))
        except (OSError, ValueError):
            continue
    naive, grouped, lines_per_call = {}, {}, {}
    for o in calls:
        u = usage_of(o)
        if not u:
            continue
        model, mid, usage = u
        n = naive.setdefault(model, [0, 0, 0, 0])
        for i, (k, _) in enumerate(FIELDS):
            n[i] += usage.get(k) or 0
        grouped.setdefault(model, {})[mid] = usage  # one line per message.id
        lines_per_call[mid] = lines_per_call.get(mid, 0) + 1
    counter = {m: [u.get(k) or 0 for _, k in FIELDS] for m, u in records[stop]["modelUsage"].items()}
    grouped = {m: [sum(u.get(k) or 0 for u in ids.values()) for k, _ in FIELDS] for m, ids in grouped.items()}
    return counter, naive, grouped, list(lines_per_call.values())


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude", "projects")).expanduser()
    sessions = checked = exact = over = 0
    tot = {"counter": [0] * 4, "naive": [0] * 4, "grouped": [0] * 4}
    by_model: dict = {}
    all_lines: list[int] = []
    for main_file in sorted(root.glob("*/*.jsonl")):
        sessions += 1
        r = session(main_file)
        if r is None:
            continue
        checked += 1
        counter, naive, grouped, per_call = r
        all_lines += per_call
        # input and cache only: sub-agent output is known to be under-logged
        g3 = sum(sum(v[:3]) for v in grouped.values())
        c3 = sum(sum(v[:3]) for v in counter.values())
        exact += g3 == c3
        over += g3 > c3
        for m in set(counter) | set(grouped):
            pm = by_model.setdefault(m, [0, 0])
            pm[0] += sum(counter.get(m, [0] * 4)[:3])
            pm[1] += sum(grouped.get(m, [0] * 4)[:3])
        for name, per_model in (("counter", counter), ("naive", naive), ("grouped", grouped)):
            for v in per_model.values():
                tot[name] = [a + b for a, b in zip(tot[name], v)]
    print(f"{sessions} sessions under {root}; {checked} carry Claude Code's counter (cost-state).")
    if not checked:
        print("Nothing to compare: your Claude Code version does not write the counter, or there is no session.")
        return 1
    print(f"{'':28}{'input':>14}{'cache read':>16}{'cache write':>16}{'output':>14}")
    for label, key in (("Claude Code's counter", "counter"), ("naive sum of every line", "naive"),
                       ("one line per message.id", "grouped")):
        print(f"{label:28}" + "".join(f"{v:>{w},}" for v, w in zip(tot[key], (14, 16, 16, 14))))
    # the duplication factor: same calls, counted once per line vs once per message.id
    ratio = [n / g if g else 0 for n, g in zip(tot["naive"], tot["grouped"])]
    print(f"{'naive / one per message.id':28}" + "".join(f"{r:>{w}.2f}x" for r, w in zip(ratio, (13, 15, 15, 13))))
    print(f"Sessions where 'one line per message.id' equals the counter for input and cache: {exact} of {checked}; "
          f"above it (would mean overcounting): {over}.")
    print("Input and cache by model (counter vs one line per message.id):")
    for m, (c, g) in sorted(by_model.items()):
        print(f"  {m:34}{c:>16,}{g:>16,}{g - c:>+16,}")
    print("A model the counter has more of usually ran inside a tool (WebFetch reading a page),"
          " which writes no call line.")
    dist = {k: all_lines.count(k) for k in sorted(set(all_lines))}
    print(f"Lines per call: mean {sum(all_lines) / len(all_lines):.2f}, distribution "
          + ", ".join(f"{k} line{'s' if k > 1 else ''}: {n}" for k, n in dist.items()) + ".")
    print("Each ratio above is this lines-per-call figure, weighted by that column's tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
