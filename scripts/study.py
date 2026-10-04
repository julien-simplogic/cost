"""Read-only study of how to estimate a task better. `python scripts/study.py <step>`.

Opens tokentrail's database read-only and reads the transcripts read-only. Prompt
text is read in memory to compute lengths and flags, never printed or stored.
Prints numbers only; project names and MCP server names are replaced by hashes.

Steps (numbering of the 4 Oct 2026 plan):
  1  task definition: short follow-ups merged into the previous task
  3  recalibration: which quantiles give 80% in a leak-free backtest
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

from tokentrail import metrics, paths, prices
from tokentrail.analysis import enrich, group_tasks, parse_ts, percentile, percentile_sorted
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
    out = []
    for t in group_tasks(rows, meta):
        if not t.tracked or not t.turns_main or not t.first_input:
            continue
        main = [r for r in t.rows if r.trigger != "subagent"]
        out.append(T(t.session_key, t.task_id, t.started, t.turns_main, t.turns, t.cost, t.first_input,
                     main[0].model, t.project, t.family, t.family_source, feats.get(t.task_id), t.rows))
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
                        cur.cost + t.cost, cur.first_input, cur.model, cur.project, cur.family,
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
    num = lambda x: f"{x:.1f}"  # noqa: E731
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


STEPS = {"1": step1, "3": step3}


def main(argv: list[str]) -> None:
    wanted = argv or sorted(STEPS)
    tasks = load()
    for s in wanted:
        STEPS[s](tasks)
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
