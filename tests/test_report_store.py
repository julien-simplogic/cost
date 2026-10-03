from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from fake_transcripts import FakeSession, bash, edit, read
from tokentrail import paths, prices, report
from tokentrail.cli import main
from tokentrail.ingest import ingest_claude_code
from tokentrail.store import Store


@pytest.fixture
def store(env):
    st = Store(paths.db_path())
    yield st
    st.close()


def _week(projects):
    a = FakeSession(projects, project="acme-webshop")
    a.prompt("refactor cart")
    a.call(new=2, write=30_000, tools=edit("cart.py"))
    a.call(read=30_000, write=2_000, tools=bash("pytest"))
    a.subagent([{"new": 2, "write": 20_000, "tools": bash("ls")},
                {"read": 20_000, "write": 1_000}], final_out=500)
    a.call(read=32_000, write=500, out=900)
    a.prompt("why is checkout slow?")
    a.call(read=33_000, write=300)
    a.write()
    b = FakeSession(projects, project="harbor-api", start=datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc))
    b.prompt("review the auth module", command="code-review")
    b.call(new=2, write=15_000, tools=read("auth.py"))
    b.call(read=15_000, write=4_000)
    b.write()
    return a, b


def test_report_categories_and_tasks(env, store):
    _week(env.projects)
    ingest_claude_code(store, env.projects)
    rep = report.build(store, prices.load(), top=5)
    cats = {c["key"]: c for c in rep["categories"]}
    assert cats["subagents"]["turns"] == 2
    assert cats["reviews"]["turns"] == 2  # the code-review task
    assert cats["user_turns"]["turns"] == 2  # the two prompts in acme-webshop
    assert cats["tool_turns"]["turns"] == 3  # bash, the Agent launch, the follow-up
    assert sum(c["turns"] for c in cats.values()) == rep["totals"]["turns"] == 9
    assert abs(sum(c["cost_share"] for c in cats.values()) - 100) < 0.5

    top = rep["tasks"][0]
    assert top["family"] == "refactor" and top["turns"] == 6 and top["turns_subagent"] == 2
    assert {t["family"] for t in rep["tasks"]} == {"refactor", "question", "review"}
    assert rep["totals"]["inexact_output_turns"] == 1
    assert rep["parsing"]["not_understood"] == 0


def test_cost_matches_the_price_file(env, store):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(new=1_000_000, out=0, thinking=0)
    s.call(read=1_000_000, out=1_000_000, thinking=0)
    s.call(write=1_000_000, out=0, thinking=0, ttl="1h")
    s.write()
    ingest_claude_code(store, env.projects)
    rep = report.build(store, prices.load())
    # opus 5.5: $4 new + ($0.20 read + $20 output) + $4 x 2 (1h write)
    assert rep["totals"]["cost"] == pytest.approx(4 + 0.2 + 20 + 8)


def test_anonymize_hides_project_names(env, store, capsys):
    _week(env.projects)
    main(["report", "--since", "all", "--anonymize"])
    out = capsys.readouterr().out
    assert "acme" not in out and "harbor" not in out
    assert "project-A" in out
    main(["report", "--since", "all", "--json"])
    assert json.loads(capsys.readouterr().out)["totals"]["turns"] == 9


def test_cache_expiry_and_break_detection(env, store):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(new=2, write=50_000)
    s.prompt("back after lunch")
    s.tick(3600)
    s.call(new=2, write=50_200)  # 5m cache long gone
    s.prompt("switch model")
    s.call(new=2, write=50_400, model="claude-sonnet-5-5", seconds=20)  # within TTL, prefix lost
    s.write()
    ingest_claude_code(store, env.projects)
    c = report.build(store, prices.load())["cache"]
    assert c["expiries"] == 1 and c["breaks"] == 1
    assert c["expiry_cost"] > 0 and c["break_cost"] > 0


def test_since_filters_the_period(env, store):
    _week(env.projects)
    ingest_claude_code(store, env.projects)
    rep = report.build(store, prices.load(), since="2026-09-29T00:00:00.000Z")
    assert rep["totals"]["turns"] == 2


# ------------------------------------------------------------------ store


def test_ingest_is_incremental(env, store):
    a, _ = _week(env.projects)
    first = ingest_claude_code(store, env.projects)
    assert (first.sessions_seen, first.sessions_parsed) == (2, 2)
    again = ingest_claude_code(store, env.projects)
    assert again.sessions_parsed == 0
    a.prompt("one more")
    a.call(read=34_000, write=100)
    a.write()
    third = ingest_claude_code(store, env.projects)
    assert third.sessions_parsed == 1
    assert len(store.records()) == 10


def test_resumed_session_does_not_double_count(env, store):
    a = FakeSession(env.projects)
    a.prompt("x")
    a.call(write=1000)
    a.write()
    resumed = FakeSession(env.projects, session_id="resumed-0001")
    resumed.lines = list(a.lines)  # a resumed session repeats its parent's history
    resumed.prompt("continue")
    resumed.call(read=1000, write=10)
    resumed.write()
    ingest_claude_code(store, env.projects)
    assert len(store.records()) == 2


def test_declared_family_survives_reingest(env, store, capsys):
    _week(env.projects)
    ingest_claude_code(store, env.projects)
    task = report.build(store, prices.load())["tasks"][0]["task"]
    assert main(["tag", task, "measure"]) == 0
    main(["ingest", "--force"])
    capsys.readouterr()
    rep = report.build(store, prices.load())
    tagged = [t for t in rep["tasks"] if t["task"] == task][0]
    assert (tagged["family"], tagged["family_source"]) == ("measure", "declared")


# ------------------------------------------------------------------ prices


def test_price_file_is_data_with_a_date(env):
    table = prices.load()
    assert table.verified_on is not None and table.path == "(packaged default)"
    assert table.lookup("claude-haiku-4-5-20251001") is table.models["claude-haiku-4-5"]
    assert table.lookup("claude-opus-5-5") is table.models["claude-opus-5-5"]
    assert table.lookup("claude-opus-5") is table.models["claude-opus-5"]
    assert table.lookup("gpt-something") is None
    assert table.is_stale(today=date(2027, 6, 1))


def test_user_price_file_wins(env, capsys):
    main(["prices", "--init"])
    user = paths.user_prices_path()
    text = user.read_text().replace("input = 4.00", "input = 40.00", 1)
    user.write_text(text.replace("verified_on = 2026-09-25", "verified_on = 2026-10-01"))
    table = prices.load()
    assert table.path == str(user)
    assert table.lookup("claude-opus-5-5").input == 40.0
    assert str(table.verified_on) == "2026-10-01"
