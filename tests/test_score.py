"""The estimate grades itself, the turn distribution, and what precedes cache breaks."""

from __future__ import annotations

from datetime import timedelta

import pytest

from fake_transcripts import FakeSession, edit
from test_estimate import HISTORY_TURNS, T0, _current, _history
from tokentrail import estimate, paths, predictions, prices, report
from tokentrail.cli import main
from tokentrail.ingest import ingest_claude_code
from tokentrail.store import Store


@pytest.fixture
def store(env):
    st = Store(paths.db_path())
    yield st
    st.close()


def _task(s: FakeSession, turns: int, ctx: int = 60_000) -> None:
    s.prompt("refactor the next thing")
    s.call(new=2, read=ctx, write=900, tools=edit("x.py"))
    for _ in range(turns - 1):
        s.call(read=ctx, write=100, out=120)


def test_an_estimate_is_linked_to_the_next_task_and_scored_once_it_is_over(env, store):
    _history(env.projects)
    s = _current(env.projects)
    ingest_claude_code(store, s.root)
    table = prices.load()
    est = estimate.build(store, table, estimate.EstimateInput(
        text="x" * 40, family="refactor", cwd=s.cwd, now=s.now + timedelta(seconds=20)))
    predictions.record(store, est, "cli")

    s.tick(20)
    _task(s, turns=5)  # inside p10-p90 of HISTORY_TURNS
    s.write()
    ingest_claude_code(store, s.root)
    predictions.settle(store, table, now=s.now + timedelta(minutes=1))
    [p] = store.predictions()
    assert p["status"] == "linked" and p["outcome"] is None  # the task may not be over yet

    s.tick(60)
    _task(s, turns=2)  # a later task in the same session: the first one is over
    s.write()
    ingest_claude_code(store, s.root)
    predictions.settle(store, table, now=s.now + timedelta(minutes=1))
    [p] = store.predictions()
    assert p["status"] == "scored"
    assert p["outcome"]["turns_main"] == 5
    assert p["outcome"]["first_cache_read"] == 60_000

    sc = predictions.build_score(store, table)["scored"]
    assert sc["n"] == 1 and sc["with_interval"] == 1 and sc["turns_inside"] == 1
    assert sc["turns_median_abs_error"] == pytest.approx(abs(5 - est["predicted"]["turns"]["p50"]))
    assert sc["cache_as_said"] == 1


def test_an_estimate_with_no_task_after_it_is_never_scored(env, store):
    _history(env.projects)
    s = _current(env.projects)
    ingest_claude_code(store, s.root)
    table = prices.load()
    est = estimate.build(store, table, estimate.EstimateInput(cwd=s.cwd, now=s.now + timedelta(seconds=20)))
    predictions.record(store, est, "cli")
    predictions.settle(store, table, now=s.now + timedelta(hours=2))
    [p] = store.predictions()
    assert p["status"] == "unmatched"


def test_predictions_survive_a_schema_rebuild(env, store):
    _history(env.projects)
    s = _current(env.projects)
    ingest_claude_code(store, s.root)
    est = estimate.build(store, prices.load(), estimate.EstimateInput(cwd=s.cwd, now=s.now))
    predictions.record(store, est, "cli")
    store.db.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    store.db.commit()
    with Store(paths.db_path()) as again:
        assert len(again.predictions()) == 1
        assert not again.has_records()  # derived tables were dropped, the predictions were not


def test_backtest_uses_only_what_estimate_knew_before_each_task(env, store):
    _history(env.projects)
    ingest_claude_code(store, env.projects)
    tasks = store.tasks()
    from tokentrail.analysis import enrich, group_tasks

    rows = enrich(store.records(), prices.load(), tasks)
    bt = predictions.backtest_turns(group_tasks(rows, tasks))
    # ten tasks, the first eight have too little before them: only tasks 9 and 10 are scored
    assert bt["all"]["n"] == 2
    assert bt["all"]["inside"] == sum(
        1 for i in (8, 9) if _p(sorted(HISTORY_TURNS[:i]), 10) <= HISTORY_TURNS[i] <= _p(sorted(HISTORY_TURNS[:i]), 90))


def _p(xs, q):
    from tokentrail.analysis import percentile_sorted

    return percentile_sorted(xs, q)


def test_computed_part_backtest_on_history(env, store):
    s = FakeSession(env.projects, start=T0)
    s.prompt("first")
    s.call(new=2, write=30_000, out=500)
    s.prompt("x" * 250)  # warm: 30 s later
    s.call(new=2, read=30_000, write=700, out=300)
    s.tick(900)  # past the 5m TTL
    s.prompt("again")
    s.call(new=2, write=31_200, out=100)
    s.write()
    ingest_claude_code(store, env.projects)
    sc = predictions.build_score(store, prices.load())["backtest"]["computed"]
    assert sc["prompts"] == 2
    assert sc["warm_said"] == 1 and sc["warm_was_warm"] == 1
    assert sc["cold_said"] == 1 and sc["cold_was_cold"] == 1
    assert sc["cold_missed"]["n"] == 0


def test_backtest_says_what_the_wrongly_cold_cases_have_in_common(env, store):
    s = FakeSession(env.projects, start=T0)
    s.prompt("first")
    s.call(new=2, write=30_000, out=500)
    s.tick(400)  # past 5 min, yet the cache held
    s.prompt("again")
    s.call(new=2, read=30_000, write=700, out=100)
    s.write()
    ingest_claude_code(store, env.projects)
    sc = predictions.build_score(store, prices.load())["backtest"]["computed"]
    m = sc["cold_missed"]
    assert (sc["cold_said"], sc["cold_was_cold"]) == (1, 0)
    assert (m["n"], m["idle_5m"], m["fully_warm"], m["idle_5m_gap_under_1h"]) == (1, 1, 1, 1)
    text = predictions.render_score(predictions.build_score(store, prices.load()))
    assert "the 1 that read 10% or more: 0 after a model switch, 1 idle past a 5m TTL" in text


def test_turn_distribution_by_family_flags_spread(env, store, capsys):
    _history(env.projects)  # refactor: 2 .. 41 turns, x8 between p10 and p90
    s = FakeSession(env.projects, project="other", start=T0 + timedelta(days=1))
    for n in [1, 1, 1, 1, 2, 30, 90, 150, 200, 300]:
        s.prompt("what does this do")
        s.call(new=2, read=5_000, write=100)
        for _ in range(n - 1):
            s.call(read=5_000, write=10, out=10)
        s.tick(10)
    s.write()
    assert main(["--source-dir", str(env.projects), "turns"]) == 0
    out = capsys.readouterr().out
    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}
    assert "wide" in lines["refactor"]
    assert "too spread: no interval given" in lines["question"]


def test_cache_breaks_are_attributed_to_what_preceded_them(env, store):
    s = FakeSession(env.projects, start=T0)
    s.prompt("go")
    s.call(new=2, write=20_000)
    s.call(new=2, read=20_000, write=100, tools=edit("CLAUDE.md"))
    s.call(new=2, write=20_200)  # break right after a CLAUDE.md edit
    s.call(new=2, read=20_200, write=100, tools=edit("/srv/demo/acme-webshop/.claude/settings.json"))
    s.call(new=2, read=20_300, write=100)  # edited, yet the cache held
    s.call(new=2, write=20_500, model="claude-sonnet-5-5")  # model switch
    s.call(new=2, read=20_500, write=100, model="claude-sonnet-5-5")
    s.call(new=2, write=20_700, model="claude-sonnet-5-5")  # nothing visible
    s.write()
    ingest_claude_code(store, env.projects)
    b = report.build_check(store, prices.load())["cache_breaks"]
    assert (b["breaks"], b["model_switch"], b["prefix_file_written"], b["no_visible_cause"]) == (3, 1, 1, 1)
    assert (b["prefix_writes"], b["prefix_writes_then_break"]) == (2, 1)
    text = report.render_check(report.build_check(store, prices.load()))
    assert "followed by a" in text and "break on that next call: 1" in text


def test_cli_score_runs_and_estimate_records(env, capsys):
    _history(env.projects)
    s = _current(env.projects)
    assert main(["estimate", "--session", s.session_id[:8], "rename foo"]) == 0
    assert main(["estimate", "--no-record", "--session", s.session_id[:8], "rename foo"]) == 0
    capsys.readouterr()
    assert main(["score"]) == 0
    out = capsys.readouterr().out
    assert "Recorded estimates: 1 (cli 1)" in out
    assert "No estimate scored yet" in out
    assert "Backtest on your history" in out


def test_after_an_expiry_what_survives_is_measured_and_tied_to_other_sessions(env, store):
    # two sessions in parallel; session A goes idle for 2 h while B keeps working
    a = FakeSession(env.projects, project="alpha", start=T0)
    b = FakeSession(env.projects, project="beta", start=T0)
    for _ in range(12):
        a.prompt("go")
        a.call(new=2, read=8_000, write=42_000)
        a.tick(7200)  # 2 h idle: the 5m cache is long gone...
        a.prompt("back")
        a.call(new=2, read=8_000, write=42_100)  # ...yet 8k of shared prefix was read
    for _ in range(12 * 37):
        b.call(new=2, read=8_000, write=50)
        b.tick(200)  # b never lets its own cache expire
    a.write(), b.write()
    ingest_claude_code(store, env.projects)
    e = predictions.build_score(store, prices.load())["backtest"]["computed"]["expiry"]
    assert e["n"] == 12 and e["read_p50"] == 8_000
    assert e["other_active"] == 12 and e["other_active_read10"] == 12 and e["alone"] == 0

    est = estimate.build(store, prices.load(), estimate.EstimateInput(
        session=a.session_id[:8], now=a.now + timedelta(hours=3)))
    assert est["certain"][0].startswith("Idle for 3.0 h")
    assert "a median 8,000 tokens were still read from cache anyway" in est["certain"][1]
