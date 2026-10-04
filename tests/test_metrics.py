"""Interval metrics used by `score`: values checked by hand."""

from __future__ import annotations

import math

import pytest

from tokentrail import metrics


def test_interval_score_is_width_plus_penalty_for_misses():
    # alpha 0.2: a miss costs 2 / 0.2 = 10 per unit outside
    assert metrics.interval_score(2, 5, 10) == 8
    assert metrics.interval_score(2, 1, 10) == pytest.approx(8 + 10 * 1)
    assert metrics.interval_score(2, 13, 10) == pytest.approx(8 + 10 * 3)


def test_a_wide_interval_that_always_covers_does_not_win_by_default():
    ys = [3, 4, 5, 6, 30]
    narrow = sum(metrics.interval_score(3, y, 6) for y in ys)  # misses 30 once
    wide = sum(metrics.interval_score(1, y, 1000) for y in ys)
    assert narrow < wide


def test_coverage_and_log_error():
    assert metrics.coverage([(1, 2, 3), (1, 5, 3), (2, 2, 2), (0, -1, 1)]) == 0.5
    assert metrics.log_error(10, 5) == pytest.approx(math.log(2))
    assert metrics.log_error(5, 10) == pytest.approx(-math.log(2))
    assert metrics.log_error(0, 3) is None


def test_bootstrap_resamples_sessions_not_tasks():
    # one session of 100 identical misses, one of 100 identical hits: per session the
    # coverage is 0 or 1, so resampled sessions give 0, 0.5 or 1, never anything between
    items = [("a", 1, 9, 3)] * 100 + [("b", 1, 2, 3)] * 100
    seen = set()
    for seed in range(5):
        ci = metrics.bootstrap_by_session(items, lambda it: it[0],
                                          lambda xs: metrics.coverage([x[1:] for x in xs]), n=200, seed=seed)
        seen.update(ci)
    assert seen <= {0.0, 0.5, 1.0}
    assert metrics.bootstrap_by_session(items[:3], lambda it: it[0], lambda xs: 1.0) is None  # one session


def test_summarize_reports_every_measure_with_its_interval():
    items = [(f"s{i % 7}", 2, y, 8, 4) for i, y in enumerate([1, 3, 4, 5, 6, 9, 20, 4, 5, 3] * 3)]
    m = metrics.summarize(items, bootstrap=100)
    assert m["n"] == 30 and m["sessions"] == 7 and m["nominal"] == 0.8
    assert m["coverage"] == pytest.approx(21 / 30)
    for k in ("coverage", "interval_score", "abs_error", "log_error", "width"):
        lo, hi = m[k + "_ci"]
        assert lo <= hi
