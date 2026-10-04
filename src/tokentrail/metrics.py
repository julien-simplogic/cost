"""How good an interval estimate is: coverage, interval score, log error, session bootstrap.

Pure functions, no I/O. Used by `tokentrail score` and by scripts/study.py.

coverage        share of outcomes inside [lo, hi], to compare with the nominal
                coverage (80% for p10-p90).
interval score  Winkler / Gneiting-Raftery for a central (1 - alpha) interval:
                width + (2 / alpha) x how far the outcome fell outside. Lower is
                better; it rewards narrow intervals and punishes misses, so a
                wide interval that always covers does not win by default.
log error       log(estimated / actual): 0 is exact, +0.69 is twice too high,
                -0.69 half. Symmetric for over and under, unlike a % error.
bootstrap       resampling whole sessions, not tasks: tasks of one session are
                alike, so resampling tasks would claim more certainty than the
                data holds.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Hashable, Optional, Sequence, TypeVar

T = TypeVar("T")

NOMINAL = 0.80  # p10-p90
ALPHA = 1 - NOMINAL


def inside(lo: float, y: float, hi: float) -> bool:
    return lo <= y <= hi


def coverage(items: Sequence[tuple[float, float, float]]) -> Optional[float]:
    """items: (lo, y, hi)."""
    return sum(1 for lo, y, hi in items if inside(lo, y, hi)) / len(items) if items else None


def interval_score(lo: float, y: float, hi: float, alpha: float = ALPHA) -> float:
    s = hi - lo
    if y < lo:
        s += 2 / alpha * (lo - y)
    elif y > hi:
        s += 2 / alpha * (y - hi)
    return s


def log_error(estimate: float, actual: float) -> Optional[float]:
    if estimate <= 0 or actual <= 0:
        return None
    return math.log(estimate / actual)


def median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def bootstrap_by_session(items: Sequence[T], session: Callable[[T], Hashable],
                         stat: Callable[[list[T]], Optional[float]],
                         n: int = 1000, seed: int = 0, level: float = 0.95) -> Optional[tuple[float, float]]:
    """Percentile confidence interval of stat(items), resampling sessions with replacement."""
    groups: dict[Hashable, list[T]] = {}
    for it in items:
        groups.setdefault(session(it), []).append(it)
    keys = list(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample: list[T] = []
        for k in rng.choices(keys, k=len(keys)):
            sample.extend(groups[k])
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    lo_i = int((1 - level) / 2 * (len(vals) - 1))
    hi_i = int((1 + level) / 2 * (len(vals) - 1))
    return vals[lo_i], vals[hi_i]


def summarize(items: Sequence[tuple], *, bootstrap: int = 1000) -> dict:
    """items: (session, lo, y, hi, median_estimate). The metrics `score` prints, with session CIs."""
    def cov(xs):
        return coverage([(lo, y, hi) for _, lo, y, hi, _ in xs])

    def isc(xs):
        return mean([interval_score(lo, y, hi) for _, lo, y, hi, _ in xs])

    def abs_err(xs):
        return median([abs(m - y) for _, _, y, _, m in xs])

    def log_err(xs):
        return median([e for e in (log_error(m, y) for _, _, y, _, m in xs) if e is not None])

    def width(xs):
        return median([hi - lo for _, lo, _, hi, _ in xs])

    out: dict = {"n": len(items), "sessions": len({it[0] for it in items}), "nominal": NOMINAL}
    for name, f in (("coverage", cov), ("interval_score", isc), ("abs_error", abs_err),
                    ("log_error", log_err), ("width", width)):
        out[name] = f(list(items))
        out[name + "_ci"] = bootstrap_by_session(items, lambda it: it[0], f, n=bootstrap) if bootstrap else None
    return out
