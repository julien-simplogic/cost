"""Read-only probe: (1) do families separate turns, (2) shape of unexplained cache breaks.
Prints numbers and tool names only (MCP server names hashed)."""
import hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

from tokentrail import paths, prices
from tokentrail.analysis import cache_events, enrich, group_tasks, is_prefix_file, parse_ts, percentile
from tokentrail.collectors import claude_code
from tokentrail.estimate import guess_family_from_text
from tokentrail.store import Store

st = Store(paths.db_path())
table = prices.load()
tasks = st.tasks()
rows = enrich(st.records(), table, tasks)
ts = [t for t in group_tasks(rows, tasks) if t.tracked and t.turns_main and t.first_input]


def anon(name):
    if name.startswith("mcp__"):
        parts = name.split("__")
        return "mcp:" + hashlib.sha256(parts[1].encode()).hexdigest()[:6] + "__" + "__".join(parts[2:])
    return name


def dist(v):
    if len(v) < 2:
        return f"n={len(v)}"
    q = [percentile(v, p) for p in (10, 25, 50, 75, 90)]
    return f"n={len(v):>5}  p10 {q[0]:>4.0f}  p25 {q[1]:>4.0f}  median {q[2]:>4.0f}  p75 {q[3]:>4.0f}  p90 {q[4]:>5.0f}"


print("=== 1. FAMILIES: main-thread turns per task")
print("-- a) family guessed from the task's first tool calls (what `turns` shows; circular:")
print("      'question' means few or no tool calls, so it is short by construction)")
for fam in ("question", "refactor", "review", "measure"):
    print(f"   {fam:9} {dist([t.turns_main for t in ts if t.family == fam])}")

# family from the prompt text: what estimate actually uses. Read prompts from the transcripts, print none.
text_fam = {}
for sf in claude_code.discover(paths.claude_code_dir()):
    if not sf.main:
        continue
    with sf.main.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not isinstance(o, dict) or o.get("type") != "user" or not o.get("promptId") or o.get("isMeta"):
                continue
            c = (o.get("message") or {}).get("content")
            if isinstance(c, list):
                if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                    continue
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            if isinstance(c, str) and o["promptId"] not in text_fam:
                text_fam[o["promptId"]] = guess_family_from_text(c) or "none"
print("-- b) family guessed from your prompt text (what estimate uses before you send)")
for fam in ("question", "refactor", "review", "measure", "none"):
    print(f"   {fam:9} {dist([t.turns_main for t in ts if text_fam.get(t.task_id) == fam])}")
print(f"   tasks with a prompt found: {sum(1 for t in ts if t.task_id in text_fam)} of {len(ts)}")
agree = [(text_fam[t.task_id], t.family) for t in ts if t.task_id in text_fam]
print(f"   text family = tool family: {sum(1 for a, b in agree if a == b)} of {len(agree)}")
print("   cross table (rows: from text, columns: from tools):")
fams = ("question", "refactor", "review", "measure")
print("   " + " " * 10 + "".join(f"{f:>10}" for f in fams))
for a in fams + ("none",):
    print(f"   {a:10}" + "".join(f"{sum(1 for x, y in agree if x == a and y == b):>10}" for b in fams))
print("-- c) 'measure' tasks (from tools): tools in their first 5 calls")
first_cmd = Counter()
for t in ts:
    if t.family != "measure":
        continue
    first_cmd.update(anon(n) for r in t.rows[:5] for n in (r.extra.get("tools") or []))
print("   ", ", ".join(f"{k} {v}" for k, v in first_cmd.most_common(10)))


print()
print("=== 2. CACHE BREAKS WITH NO VISIBLE CAUSE: shape")
main_by = defaultdict(list)
for r in rows:
    if r.trigger != "subagent":
        main_by[r.session_key].append(r)
sub_ts = defaultdict(list)
for r in rows:
    if r.trigger == "subagent":
        sub_ts[r.session_key].append(r.ts)
events_by = defaultdict(list)
for e in st.file_events():
    events_by[e["session_key"]].append(e)
versions = {s["session_key"]: (s["version"] or "?") for s in st.sessions()}

base_tools, brk_tools = Counter(), Counter()
n_pairs = 0
found = []
for key, main in main_by.items():
    idx = {r.turn_id: i for i, r in enumerate(main)}
    writes = [(e["ts"] or "", e["path"]) for e in events_by[key] if e["action"] == "write" and is_prefix_file(e["path"])]
    for prev, cur in zip(main, main[1:]):
        n_pairs += 1
        base_tools.update(set(anon(x) for x in prev.extra.get("tools") or []))
    for ev in cache_events(main, table):
        if ev.kind != "break":
            continue
        i = idx[ev.turn_id]
        prev, cur = main[i - 1], main[i]
        if cur.model != prev.model or any(prev.ts <= t < cur.ts for t, _ in writes):
            continue
        brk_tools.update(set(anon(x) for x in prev.extra.get("tools") or []))
        subs_between = sum(1 for t in sub_ts[key] if prev.ts <= t < cur.ts)
        ttl = "1h" if any(r.cache_write_1h for r in main[:i]) else "5m"
        found.append(dict(key=key, i=i, n=len(main), gap=ev.gap_s, cost=ev.extra_cost or 0, ttl=ttl,
                          trigger=cur.trigger, read=cur.cache_read, prev_in=prev.input_total,
                          grew=cur.input_total - prev.input_total - prev.output, subs=subs_between,
                          new_task=cur.task_id != prev.task_id, ver=versions.get(key, "?")))

print(f"breaks: {len(found)}, extra cost ${sum(f['cost'] for f in found):,.2f}")
per = Counter(f["key"] for f in found)
cost_per = defaultdict(float)
for f in found:
    cost_per[f["key"]] += f["cost"]
top = per.most_common()
print(f"-- concentration: in {len(per)} of {len(main_by)} sessions")
for k in (1, 3, 5, 10):
    print(f"   top {k:>2} sessions hold {sum(n for _, n in top[:k])} breaks, "
          f"${sum(sorted(cost_per.values(), reverse=True)[:k]):,.2f}")
print("   breaks per session (largest first):", ", ".join(str(n) for _, n in top[:20]))
print("   by Claude Code version:", ", ".join(f"{v} {n}" for v, n in sorted(Counter(f["ver"] for f in found).items())))
print("-- when, in the session")
print(f"   call index: {dist([f['i'] for f in found])}")
print(f"   within the first 3 calls: {sum(1 for f in found if f['i'] <= 3)}; "
      f"relative position deciles: {Counter(min(9, int(10 * f['i'] / f['n'])) for f in found).most_common()}")
print(f"   on the first call of a task (your prompt): {sum(1 for f in found if f['new_task'])}; "
      f"mid-task (tool loop): {sum(1 for f in found if not f['new_task'])}")
print(f"   sub-agent calls between the two calls: {sum(1 for f in found if f['subs'])} breaks")
print("-- gap since the previous call, as a share of the TTL")
for ttl in ("5m", "1h"):
    g = [f["gap"] for f in found if f["ttl"] == ttl]
    lim = 300 if ttl == "5m" else 3600
    b = Counter(("<10%" if x < .1 * lim else "10-50%" if x < .5 * lim else "50-90%" if x < .9 * lim else "90-100%") for x in g)
    print(f"   TTL {ttl}: {len(g)} breaks; " + ", ".join(f"{k} {b.get(k, 0)}" for k in ("<10%", "10-50%", "50-90%", "90-100%"))
          + (f"; gap median {statistics.median(g):.0f} s" if g else ""))
print("-- what was still read from cache on the breaking call")
print(f"   tokens read: {dist([f['read'] for f in found])}")
print(f"   read 0: {sum(1 for f in found if f['read'] == 0)}; "
      f"read 15k-35k (the ~24k block): {sum(1 for f in found if 15_000 <= f['read'] <= 35_000)}; "
      f"read more than 35k: {sum(1 for f in found if f['read'] > 35_000)}")
print("-- did the context shrink? (input vs previous input + its answer)")
print(f"   grew as expected (>= 0): {sum(1 for f in found if f['grew'] >= 0)}; shrank: {sum(1 for f in found if f['grew'] < 0)}"
      f"; shrank by: {dist([-f['grew'] for f in found if f['grew'] < 0])}")
print("-- tools used by the call just before the break, vs before any call (lift)")
nb = len(found) or 1
lifts = []
for tool, k in brk_tools.items():
    base = base_tools[tool] / n_pairs
    lifts.append((k / nb / base if base else 0, tool, k, base_tools[tool]))
for lift, tool, k, b in sorted(lifts, reverse=True)[:12]:
    print(f"   {tool:40} before {k:>3} of {len(found)} breaks; before {b:>6} of {n_pairs} calls; lift x{lift:.1f}")
