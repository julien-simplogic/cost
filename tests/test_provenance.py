"""Output that is not from a call's own log line is marked, counted and checkable."""

from __future__ import annotations

import json
import sqlite3

import pytest

from fake_transcripts import FakeSession, bash
from tokentrail import paths, prices, report
from tokentrail.cli import main
from tokentrail.collectors import claude_code as cc
from tokentrail.ingest import ingest_claude_code
from tokentrail.store import Store

OPUS, HAIKU = "claude-opus-5-5", "claude-haiku-4-5-20251001"


def _session(projects, counter_output_gap: int = 87, counter_input_delta: int = 0):
    """Main thread on Opus, one Haiku sub-agent whose calls are logged mid-stream (3 tokens).

    The sub-agent really produced 90 + 219 tokens; its parent's tool result
    reports 219 for the last call. Claude Code's counter knows the truth.
    """
    s = FakeSession(projects)
    s.prompt("look into it")
    s.call(new=2, write=20_000, out=300, tools=bash("ls"))
    s.subagent([
        {"new": 10, "write": 34_901, "out": 3, "tools": bash("echo probe"), "model": HAIKU},
        {"new": 8, "read": 34_901, "write": 1_496, "out": 3, "model": HAIKU},
    ], final_out=219)
    s.call(read=20_000, write=500, out=400)
    s.cost_state({
        OPUS: {"inputTokens": 2 + 2 + 2 + counter_input_delta, "cacheReadInputTokens": 1000 + 20_000,
               "cacheCreationInputTokens": 20_000 + 200 + 500, "outputTokens": 300 + 80 + 400},
        HAIKU: {"inputTokens": 18, "cacheReadInputTokens": 34_901, "cacheCreationInputTokens": 36_397,
                "outputTokens": 3 + 219 + counter_output_gap},
    })
    s.write()
    return s


def test_logged_value_is_kept_next_to_the_recovered_one(env):
    _session(env.projects)
    [sf] = cc.discover(env.projects)
    subs = [r for r in cc.parse_session(sf).records if r.trigger == "subagent"]
    first, last = subs
    assert (first.output_logged, first.output_total, first.output_source) == (3, 3, "logged")
    assert first.output_exact is False
    assert (last.output_logged, last.output_total, last.output_source) == (3, 219, "subagent_result")
    assert last.output_recovered == 216


def test_counter_check_matches_and_measures_the_gap(env):
    _session(env.projects)
    [sf] = cc.discover(env.projects)
    checks = {c.model: c for c in cc.parse_session(sf).checks}
    assert checks[OPUS].input_matches
    assert checks[OPUS].source_output == checks[OPUS].ours_output == checks[OPUS].ours_output_logged
    h = checks[HAIKU]
    assert h.input_matches
    assert (h.ours_output_logged, h.ours_output, h.source_output) == (6, 222, 309)


def test_calls_after_the_counter_are_not_compared(env):
    s = _session(env.projects)
    s.prompt("more")
    s.call(read=21_000, write=100, out=50)  # after the last cost-state
    s.write()
    [sf] = cc.discover(env.projects)
    checks = {c.model: c for c in cc.parse_session(sf).checks}
    assert checks[OPUS].input_matches


def test_report_marks_recovered_output_and_says_how_much(env, capsys):
    _session(env.projects)
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert "‡" in out
    assert "216 recovered for 1 sub-agent calls" in out
    assert "lower bound" in out
    assert out.splitlines()[1].startswith("Verified")
    # 87 output tokens the sub-agent produced and no line records, out of 1,002 recorded
    assert "holds 0 more input and cache (+0.00%) and 87 more output (+8.7%)" in out


def test_logged_only_ignores_recovered_output(env, capsys):
    _session(env.projects)
    main(["report", "--since", "all", "--json"])
    full = json.loads(capsys.readouterr().out)["totals"]
    main(["report", "--since", "all", "--json", "--logged-only", "--no-ingest"])
    logged = json.loads(capsys.readouterr().out)["totals"]
    assert full["output"] - logged["output"] == 216
    assert full["output_recovered"] == 216 and logged["output_recovered"] == 0
    main(["report", "--since", "all", "--logged-only", "--no-ingest"])
    assert "--logged-only" in capsys.readouterr().out


def test_input_mismatch_is_reported(env, capsys):
    _session(env.projects, counter_input_delta=5)
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert "exact in 0" in out.splitlines()[1]
    # signed and sized: we count 5 fewer input tokens than the counter
    assert "): input -5 (-<0.01%), output -87" in out


def test_check_command(env, capsys):
    _session(env.projects)
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "2.1.288" in out and "2026-09" in out


def test_old_database_is_rebuilt_and_tags_survive(env):
    paths.ensure_data_dir()
    db = sqlite3.connect(str(paths.db_path()))
    db.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO meta VALUES ('schema_version', '1');"
        "CREATE TABLE records (source TEXT, turn_id TEXT);"
        "CREATE TABLE task_tags (source TEXT NOT NULL, task_id TEXT NOT NULL, family TEXT NOT NULL,"
        " PRIMARY KEY (source, task_id));"
        "INSERT INTO task_tags VALUES ('claude-code', 'abc', 'review');"
    )
    db.commit()
    db.close()
    _session(env.projects)
    with Store(paths.db_path()) as st:
        ingest_claude_code(st, env.projects)
        assert len(st.records()) == 5
        assert st.db.execute("SELECT family FROM task_tags").fetchone()[0] == "review"
        rep = report.build(st, prices.load())
    assert rep["totals"]["output_recovered"] == 216


@pytest.mark.parametrize("flag", [[], ["--logged-only"]])
def test_categories_carry_the_mark(env, capsys, flag):
    _session(env.projects)
    main(["report", "--since", "all", "--json", *flag])
    cats = {c["key"]: c for c in json.loads(capsys.readouterr().out)["categories"]}
    assert cats["subagents"]["recovered_output"] == (0 if flag else 216)
