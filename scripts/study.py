"""Read-only study of how to estimate a task better. `python scripts/study.py <step>`.

Opens tokentrail's database read-only and reads the transcripts read-only. Prompt
text is read in memory to compute lengths and flags, never printed or stored.
Prints numbers only; project names and MCP server names are replaced by hashes.

Steps (numbering of the 4 Oct 2026 plan):
  1  task definition: short follow-ups merged into the previous task
  3  recalibration: which quantiles give 80% in a leak-free backtest
  4  turns still ahead once a task has made k turns
  5  cost rather than turns: context growth per turn, and which cost estimate does best
  6  candidate signs, one at a time, then a shallow quantile tree if several pass
  8  the ~24k block that survives an expiry: what happens between two expiries when it changes
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import bisect
from collections import Counter

from tokentrail import metrics, paths, prices
from tokentrail.analysis import enrich, expiries, group_tasks, parse_ts, percentile, percentile_sorted
from tokentrail.collectors import claude_code

HERE = Path(__file__).resolve().parent
_PUNCT = re.compile(r"[\s.!?,;:…]+$")


def anon(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:6]


# ------------------------------------------------------------------ loading


def open_db() -> sqlite3.Connection:
    db = paths.db_path()
    if not db.exists():
        sys.exit(f"no database at {db}: run `tokentrail ingest` first")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_tasks_meta(con) -> dict:
    out = {}
    for r in con.execute(
        "SELECT t.*, g.family AS declared FROM tasks t "
        "LEFT JOIN task_tags g ON g.source = t.source AND g.task_id = t.task_id"
    ):
        d = dict(r)
        d["family_source"] = "declared" if d["declared"] else "guessed"
        d["family"] = d["declared"] or d["family"]
        out[(d["source"], d["task_id"])] = d
    return out


def load_followups() -> set[str]:
    out = set()
    for line in (HERE / "followups.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(normalize(line))
    return out


def normalize(text: str) -> str:
    return _PUNCT.sub("", " ".join(text.lower().split()))


_ERROR = re.compile(r"Traceback \(most recent call last\)|\b\w*(Error|Exception)\b:|\bFAILED\b|panicked at|"
                    r"^\s+at .+:\d+|error\[E\d+\]|npm ERR!|fatal:", re.MULTILINE)
_PATH = re.compile(r"(?:^|\s)@?[\w./-]+\.[A-Za-z0-9]{1,6}\b")


@dataclass
class Prompt:
    chars: int
    followup: bool
    error: bool
    n_files: int
    command: Optional[str]


def prompt_features(followups: set[str]) -> dict[str, Prompt]:
    """promptId -> features of the prompt that opened the task. The text is not kept."""
    out: dict[str, Prompt] = {}
    for sf in claude_code.discover(paths.claude_code_dir()):
        if not sf.main:
            continue
        with sf.main.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(o, dict) or o.get("type") != "user" or o.get("isMeta"):
                    continue
                pid = o.get("promptId")
                if not pid or pid in out:
                    continue
                c = (o.get("message") or {}).get("content")
                if isinstance(c, list):
                    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                        continue
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                if not isinstance(c, str):
                    continue
                m = re.search(r"<command-name>/?([\w:.-]+)</command-name>", c)
                out[pid] = Prompt(len(c), normalize(c) in followups, bool(_ERROR.search(c)),
                                  len(set(_PATH.findall(c))), m.group(1) if m else None)
    return out


@dataclass
class T:
    """One task, as the study sees it."""

    key: str  # session key
    task_id: str
    started: str
    turns: int  # main-thread calls
    calls: int  # all calls, sub-agents included
    cost: float
    first_input: int
    floor: float  # input cost of the task's first call: the computed part, known before sending
    reread: float  # cost of re-reading the first call's context once from cache
    model: str
    project: str
    family: Optional[str]
    family_source: str
    prompt: Optional[Prompt] = None
    rows: list = field(default_factory=list, repr=False)


def load(with_prompts: bool = True) -> list[T]:
    """Every tracked task with at least one main-thread call, oldest first."""
    con = open_db()
    table = prices.load()
    meta = load_tasks_meta(con)
    rows = enrich(con.execute("SELECT * FROM records ORDER BY ts, rowid").fetchall(), table, meta)
    feats = prompt_features(load_followups()) if with_prompts else {}
    newest = parse_ts(max((r.ts for r in rows), default=""))
    stats = [t for t in group_tasks(rows, meta) if t.tracked and t.turns_main and t.first_input]
    last_of_session = {}
    for t in stats:
        if t.started >= last_of_session.get(t.session_key, ("", None))[0]:
            last_of_session[t.session_key] = (t.started, t.task_id)
    out = []
    for t in stats:
        end = parse_ts(max(r.ts for r in t.rows))
        if last_of_session[t.session_key][1] == t.task_id and newest and end and (newest - end).total_seconds() < 3600:
            continue  # may still be running
        main = [r for r in t.rows if r.trigger != "subagent"]
        f = main[0]
        p = table.lookup(f.model)
        floor = (f.cost - f.output * p.output / 1e6) if (p and f.cost is not None) else 0.0
        reread = f.input_total * p.cache_read / 1e6 if p else 0.0
        out.append(T(t.session_key, t.task_id, t.started, t.turns_main, t.turns, t.cost, t.first_input,
                     floor, reread, f.model, t.project, t.family, t.family_source, feats.get(t.task_id), t.rows))
    out.sort(key=lambda t: (t.started, t.task_id))
    return out


def by_session(tasks: list[T]) -> dict[str, list[T]]:
    out: dict[str, list[T]] = {}
    for t in tasks:
        out.setdefault(t.key, []).append(t)
    return out


# ------------------------------------------------------------------ printing


def dist(vals, label: str = "") -> str:
    if not vals:
        return f"{label:28} n=0"
    q = [percentile(vals, p) for p in (10, 25, 50, 75, 90)]
    return (f"{label:28} n={len(vals):>5}  p10 {q[0]:>6.1f}  p25 {q[1]:>6.1f}  median {q[2]:>6.1f}  "
            f"p75 {q[3]:>6.1f}  p90 {q[4]:>7.1f}")


# ------------------------------------------------------------------ step 1


def merge_followups(tasks: list[T]) -> tuple[list[T], int]:
    """Tasks opened by a short follow-up are folded into the previous task of the same session."""
    merged, folded = [], 0
    for _, ts in by_session(tasks).items():
        cur: Optional[T] = None
        for t in ts:
            if cur is not None and t.prompt and t.prompt.followup:
                cur = T(cur.key, cur.task_id, cur.started, cur.turns + t.turns, cur.calls + t.calls,
                        cur.cost + t.cost, cur.first_input, cur.floor, cur.reread, cur.model, cur.project, cur.family,
                        cur.family_source, cur.prompt, cur.rows + t.rows)
                merged[-1] = cur
                folded += 1
                continue
            cur = t
            merged.append(t)
    merged.sort(key=lambda t: (t.started, t.task_id))
    return merged, folded


def step1(tasks: list[T]) -> None:
    print("=== Step 1: task definition, short follow-ups")
    with_prompt = [t for t in tasks if t.prompt]
    fu = [t for t in with_prompt if t.prompt.followup]
    print(f"tasks: {len(tasks)}; with their opening prompt found in the transcripts: {len(with_prompt)}")
    print(f"opened by a short follow-up (scripts/followups.txt): {len(fu)} "
          f"({100 * len(fu) / max(len(with_prompt), 1):.1f}%); "
          f"prompts of 20 characters or fewer, for scale: {sum(1 for t in with_prompt if t.prompt.chars <= 20)}")
    firsts = {ts[0].task_id for ts in by_session(tasks).values()}
    first_in_session = sum(1 for t in fu if t.task_id in firsts)
    print(f"  of which the first task of their session (nothing to merge into): {first_in_session}")
    print(dist([t.turns for t in fu], "turns of follow-up tasks"))
    merged, folded = merge_followups(tasks)
    print(f"\nmain-thread turns per task, before and after merging ({folded} tasks folded):")
    print(dist([t.turns for t in tasks], "before"))
    print(dist([t.turns for t in merged], "after"))
    print("all calls (sub-agents included):")
    print(dist([t.calls for t in tasks], "before"))
    print(dist([t.calls for t in merged], "after"))
    print("The tool's task definition is unchanged.")


# ------------------------------------------------------------------ backtest helpers

MIN_HISTORY = 8  # as estimate: fewer earlier tasks than this, no interval
BOOT = 400


def leak_free(tasks: list[T], target, quantiles=(10, 50, 90), basis_key=None, min_history: int = MIN_HISTORY):
    """For each task, oldest first, the quantiles of `target` over the tasks before it
    (only those sharing basis_key(task) when given). Returns (task, (q values...))."""
    seen: dict = {}
    out = []
    for t in tasks:
        k = basis_key(t) if basis_key else None
        hist = seen.setdefault(k, [])
        if len(hist) >= min_history:
            out.append((t, tuple(percentile_sorted(hist, q) for q in quantiles)))
        y = target(t)
        if y is not None:
            bisect.insort(hist, y)
    return out


def score_line(label: str, items, unit: str = "turns") -> str:
    m = metrics.summarize(items, bootstrap=BOOT)

    def ci(k, f):
        c = m[k + "_ci"]
        return f"[{f(c[0])}, {f(c[1])}]" if c else ""
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    num = (lambda x: f"${x:.3f}") if unit == "$" else (lambda x: f"{x:.1f}")  # noqa: E731
    lg = lambda x: f"{x:+.2f}"  # noqa: E731
    return (f"{label:30} n={m['n']:>5} ({m['sessions']} sessions)  coverage {pct(m['coverage'])} {ci('coverage', pct)}  "
            f"interval score {num(m['interval_score'])} {ci('interval_score', num)}  width {num(m['width'])}  "
            f"median abs error {num(m['abs_error'])}  median log err {lg(m['log_error'])} {ci('log_error', lg)}")


# ------------------------------------------------------------------ step 3


def step3(tasks: list[T]) -> None:
    print("=== Step 3: recalibration of the turn interval (leak-free: each task from earlier tasks only)")
    qs = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
    rows = leak_free(tasks, lambda t: t.turns, quantiles=[q for q in qs] + [50] + [100 - q for q in qs])
    if not rows:
        print("not enough tasks")
        return
    half = len(rows) // 2
    first, second = rows[:half], rows[half:]

    def items(rs, i):
        return [(t.key, v[i], t.turns, v[len(qs) + 1 + i], v[len(qs)]) for t, v in rs]

    print("coverage by interval, whole history (nominal 80%):")
    for i, q in enumerate(qs):
        cov = metrics.coverage([(lo, y, hi) for _, lo, y, hi, _ in items(rows, i)])
        print(f"  p{q:g}-p{100 - q:g}: {100 * cov:.1f}%")
    covs = [metrics.coverage([(lo, y, hi) for _, lo, y, hi, _ in items(first, i)]) for i in range(len(qs))]
    best = min(range(len(qs)), key=lambda i: abs(covs[i] - metrics.NOMINAL))
    q = qs[best]
    print(f"chosen on the first half ({len(first)} tasks): p{q:g}-p{100 - q:g} "
          f"(coverage there {100 * covs[best]:.1f}%)")
    print("judged on the second half, never used for the choice:")
    print(score_line("p10-p90 (today)", items(second, qs.index(10))))
    print(score_line(f"p{q:g}-p{100 - q:g} (recalibrated)", items(second, best)))
    print("Interval score uses alpha 0.2 for both, the target being 80%. Turns are whole numbers, so")
    print("coverage moves in steps: an exact 80% may not exist.")


# ------------------------------------------------------------------ step 4

KS = (1, 2, 3, 5, 10)


def step4(tasks: list[T]) -> None:
    print("=== Step 4: turns still ahead once a task has made k main-thread turns")
    print("Every task here is finished (a later task or an hour idle ends it), so nothing is censored:")
    print("the empirical distribution is the Kaplan-Meier estimate itself, no correction needed.")
    print(dist([t.turns for t in tasks], "total turns, k=0"))
    for k in KS:
        reached = [t for t in tasks if t.turns >= k]
        if not reached:
            continue
        stop = sum(1 for t in reached if t.turns == k)
        print(dist([t.turns - k for t in reached], f"ahead after k={k}")
              + f"   ends at k: {100 * stop / len(reached):.0f}%")
    print("If the median ahead grows with k, the longer a task has run, the longer it still runs.")
    print("\nleak-free backtest (interval from earlier tasks that had also reached k):")
    for k in (0,) + KS:
        reached = [t for t in tasks if t.turns >= max(k, 1)]
        rows = leak_free(reached, lambda t: t.turns - k)
        if rows:
            print(score_line(f"k={k}", [(t.key, v[0], t.turns - k, v[2], v[1]) for t, v in rows]))
    print("Log error leaves out tasks that end exactly at k (0 turns ahead).")


# ------------------------------------------------------------------ step 5


def growth(t: T) -> list[int]:
    """Tokens added to the context between consecutive main-thread calls of a task
    (the previous call's answer excluded): tool results, plus Claude Code's own additions."""
    main = [r for r in t.rows if r.trigger != "subagent"]
    return [b.input_total - a.input_total - a.output for a, b in zip(main, main[1:])
            if not b.extra.get("after_compaction") and b.input_total >= 0.5 * a.input_total]


def step5(tasks: list[T]) -> None:
    print("=== Step 5: cost rather than turns")
    g = [x for t in tasks for x in growth(t)]
    print(dist(g, "tokens added per turn"))
    print(f"  turns that shrank the context (compaction aside): {sum(1 for x in g if x < 0)} of {len(g)}")
    priced = [t for t in tasks if t.floor > 0 and t.cost > 0]
    print(dist([t.cost for t in priced], "task cost, $"))
    print(dist([t.cost / t.floor for t in priced], "task cost / first-call input"))
    print("\nleak-free backtest, same tasks for every method, interval p10-p90 of the task's total cost:")
    turns_q = {t.task_id: v for t, v in leak_free(priced, lambda t: t.turns)}
    ratio_q = {t.task_id: v for t, v in leak_free(priced, lambda t: t.cost / t.floor)}
    abs_q = {t.task_id: v for t, v in leak_free(priced, lambda t: t.cost)}
    grow_med: dict[str, float] = {}
    seen: list[int] = []
    for t in priced:  # median growth per turn from earlier tasks only
        grow_med[t.task_id] = percentile(seen, 50) if seen else 0.0
        seen.extend(growth(t))
    both = [t for t in priced if t.task_id in turns_q and t.task_id in ratio_q and t.task_id in abs_q]

    def via_turns(t, n, with_growth=False):
        n = max(n, 1)
        extra = 0.0
        if with_growth and t.first_input:
            # each later turn re-reads the context, grown by the median growth per turn
            extra = t.reread / t.first_input * grow_med[t.task_id] * (n - 1) * n / 2
        return t.floor + (n - 1) * t.reread + extra

    methods = {
        "turns x context (tool today)": lambda t: tuple(via_turns(t, n) for n in turns_q[t.task_id]),
        "turns x growing context": lambda t: tuple(via_turns(t, n, True) for n in turns_q[t.task_id]),
        "cost / first input, x this one": lambda t: tuple(r * t.floor for r in ratio_q[t.task_id]),
        "task cost directly": lambda t: abs_q[t.task_id],
    }
    for name, f in methods.items():
        items = []
        for t in both:
            lo, mid, hi = f(t)
            items.append((t.key, lo, t.cost, hi, mid))
        print(score_line(name, items, unit="$"))
    print("Interval score is in dollars here; a lower one is better on the same tasks. Turns-based methods")
    print("leave out output and sub-agents, as the tool does today.")


# ------------------------------------------------------------------ step 6

MIN_BIN = 30  # a slice with fewer earlier tasks than this falls back to all earlier tasks
BUILTIN_COMMANDS = {"review", "compact", "init", "clear", "security-review", "pr-comments", "model", "cost",
                    "help", "config", "memory", "resume", "context", "loop", "simplify", "code-review"}


def _bucket(x: Optional[float], edges: list[tuple[float, str]]) -> Optional[str]:
    if x is None:
        return None
    for limit, label in edges:
        if x <= limit:
            return label
    return edges[-1][1]


def signs(tasks: list[T]) -> dict[str, dict[str, str]]:
    """sign -> task_id -> slice label, from what is known before the task's prompt is sent."""
    prev: dict[str, Optional[T]] = {}
    rank: dict[str, int] = {}
    for ts in by_session(tasks).values():
        for i, t in enumerate(ts):
            prev[t.task_id] = ts[i - 1] if i else None
            rank[t.task_id] = i + 1
    top_projects = {p for p, _ in Counter(t.project for t in tasks).most_common(10)}
    out: dict[str, dict[str, str]] = {k: {} for k in (
        "previous task's turns", "short follow-up", "error pasted", "prompt length", "files mentioned",
        "slash command", "project", "rank in session", "model")}
    for t in tasks:
        p = prev[t.task_id]
        out["previous task's turns"][t.task_id] = "first of session" if p is None else _bucket(
            p.turns, [(1, "1"), (3, "2-3"), (8, "4-8"), (20, "9-20"), (float("inf"), "21+")])
        out["rank in session"][t.task_id] = _bucket(
            rank[t.task_id], [(1, "1st"), (3, "2-3"), (10, "4-10"), (30, "11-30"), (float("inf"), "31+")])
        out["project"][t.task_id] = ("project " + anon(t.project)) if t.project in top_projects else "other"
        out["model"][t.task_id] = t.model
        pr = t.prompt
        if pr is None:
            continue
        out["short follow-up"][t.task_id] = "yes" if pr.followup else "no"
        out["error pasted"][t.task_id] = "yes" if pr.error else "no"
        out["prompt length"][t.task_id] = _bucket(
            pr.chars, [(49, "<50 chars"), (199, "50-199"), (999, "200-999"), (float("inf"), "1000+")])
        out["files mentioned"][t.task_id] = _bucket(pr.n_files, [(0, "0"), (1, "1"), (3, "2-3"), (float("inf"), "4+")])
        c = pr.command
        out["slash command"][t.task_id] = "none" if not c else (
            "/" + c if c.split(":")[-1].lower() in BUILTIN_COMMANDS else "custom " + anon(c))
    return out


def sliced_backtest(tasks: list[T], label: dict[str, str]):
    """Paired leak-free intervals on the tasks that have a slice: by slice (fallback: all earlier
    tasks while the slice has fewer than MIN_BIN), and baseline (all earlier tasks)."""
    seen_all: list[int] = []
    seen: dict[str, list[int]] = {}
    pairs = []
    fallback = 0
    for t in tasks:
        lab = label.get(t.task_id)
        if lab is not None and len(seen_all) >= MIN_HISTORY:
            base = tuple(percentile_sorted(seen_all, q) for q in (10, 50, 90))
            h = seen.get(lab, [])
            if len(h) >= MIN_BIN:
                mine = tuple(percentile_sorted(h, q) for q in (10, 50, 90))
            else:
                mine, fallback = base, fallback + 1
            pairs.append((t, base, mine))
        bisect.insort(seen_all, t.turns)
        if lab is not None:
            bisect.insort(seen.setdefault(lab, []), t.turns)
    return pairs, fallback


def paired_gain(pairs) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Mean interval score, slice minus baseline (negative = the sign helps), with its session CI."""
    diffs = [(t.key, metrics.interval_score(m[0], t.turns, m[2]) - metrics.interval_score(b[0], t.turns, b[2]))
             for t, b, m in pairs]
    stat = lambda xs: metrics.mean([d for _, d in xs])  # noqa: E731
    return stat(diffs), metrics.bootstrap_by_session(diffs, lambda x: x[0], stat, n=BOOT)


def passes(pairs) -> bool:
    gain, ci = paired_gain(pairs)
    cov = metrics.coverage([(m[0], t.turns, m[2]) for t, _, m in pairs])
    return bool(ci and ci[1] < 0 and cov is not None and 0.75 <= cov <= 0.85)


def step6(tasks: list[T]) -> None:
    print("=== Step 6: candidate signs, one at a time (target: main-thread turns of the task)")
    print(f"Slices with fewer than {MIN_BIN} earlier tasks fall back to all earlier tasks. Criterion: interval")
    print("score better than the baseline on the same tasks (95% session CI of the difference below 0)")
    print("and coverage between 75% and 85%.")
    sg = signs(tasks)
    declared = [t for t in tasks if t.family_source == "declared"]
    passed = []
    for name, label in sg.items():
        print(f"\n-- {name} ({len(label)} tasks with a value)")
        groups: dict[str, list[int]] = {}
        for t in tasks:
            if t.task_id in label:
                groups.setdefault(label[t.task_id], []).append(t.turns)
        for lab, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:12]:
            print("   " + dist(v, lab[:28]))
        pairs, fb = sliced_backtest(tasks, label)
        if not pairs:
            continue
        print("   " + score_line("baseline (all earlier)", [(t.key, b[0], t.turns, b[2], b[1]) for t, b, _ in pairs]))
        print("   " + score_line("by this sign", [(t.key, m[0], t.turns, m[2], m[1]) for t, _, m in pairs]))
        gain, ci = paired_gain(pairs)
        ok = passes(pairs)
        print(f"   interval score difference {gain:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]; fell back on {fb} tasks; "
              f"{'PASSES' if ok else 'does not pass'}" if ci else f"   difference {gain}")
        if ok:
            passed.append(name)
    if len(declared) >= MIN_BIN:
        print(f"\n-- declared families ({len(declared)} tasks tagged): declared vs guessed on the same tasks")
        guess = {t.task_id: t.family for t in tasks}
        for lab in sorted({t.family for t in declared}):
            mine = [t.turns for t in declared if t.family == lab]
            print("   " + dist(mine, f"declared {lab}"))
        print(f"   guessed family = declared on {sum(1 for t in declared if guess[t.task_id] == t.family)} "
              f"of {len(declared)} (the guess is overwritten by the tag; agreement needs the raw guess, not kept)")
    else:
        print(f"\n-- declared families: {len(declared)} tasks tagged, fewer than {MIN_BIN}: not enough to compare")
    print(f"\nsigns that pass: {', '.join(passed) or 'none'}")
    if len(passed) >= 2:
        quantile_tree(tasks, {n: sg[n] for n in passed})


def pinball(ys: list[int], preds: tuple[float, float, float]) -> float:
    tot = 0.0
    for q, p in zip((0.1, 0.5, 0.9), preds):
        for y in ys:
            tot += max(q * (y - p), (q - 1) * (y - p))
    return tot


def _quantiles(ys) -> tuple[float, float, float]:
    s = sorted(ys)
    return tuple(percentile_sorted(s, q) for q in (10, 50, 90))  # type: ignore[return-value]


def grow(ts: list[T], labels: dict[str, dict[str, str]], depth: int, path: tuple = ()):
    """Greedy binary split minimizing pinball loss; slices of a sign ordered by their median."""
    ys = [t.turns for t in ts]
    leaf = {"path": path, "n": len(ts), "q": _quantiles(ys)}
    if depth == 0:
        return leaf
    best = None
    here = pinball(ys, leaf["q"])
    for name, lab in labels.items():
        groups: dict[str, list[T]] = {}
        for t in ts:
            groups.setdefault(lab.get(t.task_id, "?"), []).append(t)
        order = sorted(groups, key=lambda g: percentile([t.turns for t in groups[g]], 50))
        for cut in range(1, len(order)):
            left = [t for g in order[:cut] for t in groups[g]]
            right = [t for g in order[cut:] for t in groups[g]]
            if len(left) < MIN_BIN or len(right) < MIN_BIN:
                continue
            loss = (pinball([t.turns for t in left], _quantiles([t.turns for t in left]))
                    + pinball([t.turns for t in right], _quantiles([t.turns for t in right])))
            if loss < here and (best is None or loss < best[0]):
                best = (loss, name, set(order[:cut]), left, right)
    if best is None:
        return leaf
    _, name, lefts, left, right = best
    return {"sign": name, "left": lefts,
            "yes": grow(left, labels, depth - 1, path + (f"{name} in {sorted(lefts)}",)),
            "no": grow(right, labels, depth - 1, path + (f"{name} not in {sorted(lefts)}",))}


def predict(node, t: T, labels):
    while "sign" in node:
        node = node["yes"] if labels[node["sign"]].get(t.task_id, "?") in node["left"] else node["no"]
    return node


def quantile_tree(tasks: list[T], labels: dict[str, dict[str, str]]) -> None:
    print("\n-- shallow quantile tree on the signs that pass (trained on the oldest 70%, judged on the rest)")
    cut = int(len(tasks) * 0.7)
    train, test = tasks[:cut], tasks[cut:]
    base_q = _quantiles([t.turns for t in train])
    print("   " + score_line("baseline (all training tasks)",
                             [(t.key, base_q[0], t.turns, base_q[2], base_q[1]) for t in test]))
    for name, lab in labels.items():  # every passing sign alone, judged on the same held-out tasks
        groups: dict[str, list[int]] = {}
        for t in train:
            groups.setdefault(lab.get(t.task_id, "?"), []).append(t.turns)
        qmap = {g: _quantiles(v) for g, v in groups.items() if len(v) >= MIN_BIN}
        items = []
        for t in test:
            q = qmap.get(lab.get(t.task_id, "?"), base_q)
            items.append((t.key, q[0], t.turns, q[2], q[1]))
        print("   " + score_line(f"alone: {name}"[:30], items))
    for depth in (2, 3):
        tree = grow(train, labels, depth)
        items = []
        for t in test:
            leaf = predict(tree, t, labels)
            items.append((t.key, leaf["q"][0], t.turns, leaf["q"][2], leaf["q"][1]))
        print("   " + score_line(f"tree depth {depth}", items))
    tree = grow(train, labels, 3)
    print("   leaves of the depth-3 tree (each estimate reads: N past tasks like this one):")

    def walk(n):
        if "sign" in n:
            walk(n["yes"])
            walk(n["no"])
        else:
            print(f"     {n['n']:>5} tasks, p10 {n['q'][0]:.0f} / median {n['q'][1]:.0f} / p90 {n['q'][2]:.0f}: "
                  + ("; ".join(n["path"]) or "all"))
    walk(tree)


# ------------------------------------------------------------------ step 8


def _events(path: Path, t0: str, t1: str) -> set[str]:
    """Kinds of things the transcript shows strictly after t0 and up to t1. Names only:
    record types, attachment and system subtypes, tool names (MCP servers hashed)."""
    out: set[str] = set()
    versions, models, cwds, branches = set(), set(), set(), set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not isinstance(o, dict):
                continue
            ts = o.get("timestamp")
            if not isinstance(ts, str) or not (t0 < ts <= t1):
                continue
            typ = o.get("type")
            for k, bag in (("version", versions), ("cwd", cwds), ("gitBranch", branches)):
                if isinstance(o.get(k), str):
                    bag.add(o[k])
            if typ == "attachment" and isinstance(o.get("attachment"), dict):
                out.add(f"attachment:{o['attachment'].get('type')}")
            elif typ == "system":
                out.add(f"system:{o.get('subtype')}")
            elif typ not in ("user", "assistant"):
                out.add(f"record:{typ}")
            if o.get("isCompactSummary"):
                out.add("compaction")
            msg = o.get("message") if isinstance(o.get("message"), dict) else {}
            if typ == "assistant":
                if isinstance(msg.get("model"), str):
                    models.add(msg["model"])
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        n = str(b.get("name"))
                        if n.startswith("mcp__"):
                            out.add("tool:mcp " + anon(n.split("__")[1]))
                        elif n in ("ToolSearch", "Skill", "Task", "Agent", "WebFetch", "WebSearch"):
                            out.add(f"tool:{n}")
    for name, bag in (("Claude Code version changed", versions), ("model changed", models),
                      ("working directory changed", cwds), ("git branch changed", branches)):
        if len(bag) > 1:
            out.add(name)
    return out


def step8(_tasks: list[T]) -> None:
    print("=== Step 8: the block read after an expiry, between two expiries of the same session")
    con = open_db()
    table = prices.load()
    meta = load_tasks_meta(con)
    rows = enrich(con.execute("SELECT * FROM records ORDER BY ts, rowid").fetchall(), table, meta)
    files = {f"{claude_code.SOURCE}:{sf.project_dir}/{sf.session_id}": sf.main
             for sf in claude_code.discover(paths.claude_code_dir()) if sf.main}
    by: dict[tuple, list] = {}
    for e in expiries(rows):
        by.setdefault((e.session_key, e.model), []).append(e)
    pairs = []
    for (key, _), es in by.items():
        es.sort(key=lambda e: e.ts)
        path = files.get(key)
        if path is None:
            continue
        for a, b in zip(es, es[1:]):
            # only the block: an expiry that re-read far more than ~24k (a whole context) is left out
            if a.read > 60_000 or b.read > 60_000:
                continue
            pairs.append((a.read == b.read, abs(b.read - a.read), _events(path, a.ts, b.ts)))
    same = [p for p in pairs if p[0]]
    diff = [p for p in pairs if not p[0]]
    print(f"pairs of consecutive expiries in one session (both reading under 60k): {len(pairs)}; "
          f"same value to the token: {len(same)}; different: {len(diff)}")
    if diff:
        print(dist([d for _, d, _ in diff], "tokens of difference"))
    kinds = Counter(k for _, _, ev in pairs for k in ev)
    print("events seen between the two expiries, share of pairs where the block stayed / changed:")
    print(f"   {'event':48}{'stayed':>12}{'changed':>12}")
    for k, _ in sorted(kinds.items(), key=lambda kv: -abs(
            sum(1 for p in diff if kv[0] in p[2]) / max(len(diff), 1)
            - sum(1 for p in same if kv[0] in p[2]) / max(len(same), 1)))[:25]:
        a = sum(1 for p in same if k in p[2])
        b = sum(1 for p in diff if k in p[2])
        print(f"   {k[:48]:48}{f'{a}/{len(same)}':>12}{f'{b}/{len(diff)}':>12}")
    nothing = sum(1 for p in diff if not p[2])
    print(f"changed with no event of these kinds in between: {nothing} of {len(diff)}")
    print("An event far more frequent in 'changed' than in 'stayed' points at what alters the block;")
    print("it does not prove it.")


STEPS = {"1": step1, "3": step3, "4": step4, "5": step5, "6": step6, "8": step8}


def main(argv: list[str]) -> None:
    wanted = argv or sorted(STEPS)
    tasks = load()
    for s in wanted:
        STEPS[s](tasks)
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
