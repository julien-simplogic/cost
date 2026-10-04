"""Past prompts most alike in words: the one sign that reads what you ask.

The prompts are read from the transcripts each time, kept in memory as sets of
words, and forgotten when the estimate is done: nothing is stored or printed.
On real history (scripts/study.py 9b) the 30 nearest earlier prompts gave
a better interval than all past tasks (interval score -5.3, session CI -8.8 to
-3.1, coverage 80.6%) for the 36% of prompts that had 30 close enough.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Iterable, Optional

from .collectors import claude_code

K = 30  # neighbours
MIN_SIM = 0.1  # cosine below this is not "alike"
COMMON = 0.05  # words found in this share of prompts or more are ignored
_WORD = re.compile(r"[^\W\d_]{3,}")


def words(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


def prompt_words(root: Path, budget_s: Optional[float] = None,
                 sessions: Optional[Iterable[claude_code.SessionFiles]] = None) -> Optional[dict[str, frozenset]]:
    """promptId -> words of the prompt that opened that task. None when the time budget ran out
    (the hook cannot wait): the caller then estimates without neighbours."""
    start = time.monotonic()
    out: dict[str, frozenset] = {}
    for sf in sessions if sessions is not None else claude_code.discover(root):
        if not sf.main:
            continue
        try:
            fh = sf.main.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if budget_s is not None and time.monotonic() - start > budget_s:
                    return None
                if '"promptId"' not in line:  # cheap filter before parsing
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(o, dict) or o.get("type") != "user" or o.get("isMeta"):
                    continue
                pid = o.get("promptId")
                if not isinstance(pid, str) or pid in out:
                    continue
                c = (o.get("message") or {}).get("content")
                if isinstance(c, list):
                    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                        continue
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                if isinstance(c, str):
                    out[pid] = words(c)
    return out


def nearest(query: frozenset, docs: dict[str, frozenset], k: int = K, min_sim: float = MIN_SIM) -> list[str]:
    """Ids of the k documents most alike to the query (TF-IDF cosine), only if k reach min_sim."""
    if not query or not docs:
        return []
    n = len(docs) + 1
    df: dict[str, int] = {}
    for d in docs.values():
        for w in d:
            df[w] = df.get(w, 0) + 1
    for w in query:
        df[w] = df.get(w, 0) + 1
    idf = {w: math.log(n / c) for w, c in df.items() if c < COMMON * n}

    def vec(d: frozenset) -> dict[str, float]:
        v = {w: idf[w] for w in d if w in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    q = vec(query)
    if not q:
        return []
    scores = []
    for pid, d in docs.items():
        v = vec(d)
        s = sum(x * v[w] for w, x in q.items() if w in v)
        if s >= min_sim:
            scores.append((s, pid))
    if len(scores) < k:
        return []
    scores.sort(reverse=True)
    return [pid for _, pid in scores[:k]]
