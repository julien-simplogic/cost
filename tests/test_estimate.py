from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fake_transcripts import FakeSession, edit, read
from tokentrail import estimate, paths, prices
from tokentrail.analysis import percentile
from tokentrail.cli import main
from tokentrail.ingest import ingest_claude_code
from tokentrail.store import Store

T0 = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)
HISTORY_TURNS = [2, 3, 3, 4, 5, 6, 8, 12, 20, 41]  # long tail on purpose


def _history(projects):
    """Ten past refactor tasks with known turn counts."""
    s = FakeSession(projects, project="acme-webshop", start=T0)
    ctx = 20_000
    for n in HISTORY_TURNS:
        s.prompt("refactor something")
        s.call(new=2, read=ctx, write=1000, tools=edit("x.py"))
        for _ in range(n - 1):
            s.call(read=ctx, write=200, out=150)
        s.tick(10)
    s.write()


def _current(projects, *, input_last=60_000, out_last=700, ttl="5m", reads=0):
    s = FakeSession(projects, project="harbor-api", start=T0 + timedelta(days=5))
    s.prompt("earlier work")
    for i in range(reads):
        s.call(read=1000, write=100, tools=read(f"src/m{i}.py"))
    s.call(new=3, read=input_last - 1003, write=1000, out=out_last, ttl=ttl)
    s.write()
    return s


@pytest.fixture
def store(env):
    st = Store(paths.db_path())
    yield st
    st.close()


def _est(store, s, **kw):
    kw.setdefault("now", s.now + timedelta(seconds=30))
    kw.setdefault("cwd", s.cwd)
    ingest_claude_code(store, s.root)
    return estimate.build(store, prices.load(), estimate.EstimateInput(**kw))


def test_input_comes_from_the_last_real_call_not_the_text(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, text="x" * 250, family="refactor")
    c = est["computed"]
    assert c["last_call_input"] == 60_000
    assert c["your_text_tokens"] == 100  # 250 chars at the default 2.5 chars/token
    assert c["input_next"] == 60_000 + 700 + 100
    assert c["cached"] == 60_000 and c["to_write"] == 800


def test_floor_is_exact(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, text="x" * 250, family="refactor")
    # opus 5.5: cache read $0.20/MTok, 5m write at 1.25 x $4
    assert est["frame"]["floor"]["cost"] == pytest.approx((60_000 * 0.20 + 800 * 4 * 1.25) / 1e6)


def test_prediction_is_p10_p90_of_the_family_never_a_mean(env, store):
    _history(env.projects)
    s = _current(env.projects)
    p = _est(store, s, family="refactor")["predicted"]
    assert p["samples"] == 10 and p["family_source"] == "declared"
    t = p["turns"]
    assert (t["p10"], t["p50"], t["p90"]) == pytest.approx(
        (percentile(HISTORY_TURNS, 10), percentile(HISTORY_TURNS, 50), percentile(HISTORY_TURNS, 90)))
    mean = sum(HISTORY_TURNS) / len(HISTORY_TURNS)
    assert mean not in (t["p10"], t["p50"], t["p90"])
    assert p["validated"] is False


def test_cost_is_turns_times_context(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, family="refactor")
    t, c, i = est["predicted"]["turns"], est["predicted"]["cost"], est["computed"]["input_next"]
    floor = est["frame"]["floor"]["cost"]
    # opus 5.5 cache read $0.20/MTok: every further turn re-reads the next call's context
    assert c["p90"] == pytest.approx(floor + (t["p90"] - 1) * i * 0.20 / 1e6)


def test_spread_history_gives_no_interval(env, store):
    s = FakeSession(env.projects, project="acme-webshop", start=T0)
    for n in [1, 1, 1, 1, 2, 30, 90, 150, 200, 300]:  # p90 / p10 > 100
        s.prompt("refactor something")
        s.call(new=2, read=5_000, write=100, tools=edit("x.py"))
        for _ in range(n - 1):
            s.call(read=5_000, write=10, out=10)
        s.tick(10)
    s.write()
    cur = _current(env.projects)
    est = _est(store, cur, family="refactor")
    assert est["predicted"]["turns"]["spread"]
    text = estimate.render(est)
    assert "turns too spread to estimate" in text and "no interval given" in text
    assert "cost of those turns" not in text


def test_too_few_family_samples_fall_back_and_say_so(env, store):
    _history(env.projects)
    s = _current(env.projects)
    p = _est(store, s, family="measure")["predicted"]
    assert "'measure' has only 0" in p["basis"]
    assert p["samples"] == 10


def test_ceiling_resends_context_every_turn(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, family="refactor", max_turns=3, max_tokens=1000)
    i = est["computed"]["input_next"]
    ce = est["frame"]["ceiling"]
    assert ce["input"] == 3 * i + 1000 * (0 + 1 + 2)
    assert ce["output"] == 3000
    assert ce["cost"] > est["frame"]["floor"]["cost"]


def test_default_turn_limit_is_your_family_max(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, family="refactor")
    assert est["frame"]["ceiling"]["max_turns"] == max(HISTORY_TURNS)
    assert est["frame"]["ceiling"]["max_tokens"] == 128_000  # opus 5.5 cap from the price file


def test_idle_time_is_stated_first_with_tokens_and_dollars(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, text="x" * 250, now=s.now + timedelta(minutes=48))
    assert est["computed"]["cached"] == 0
    line = est["certain"][0]
    # 60,000 + 700 + 100 written at 1.25 x $4/MTok; warm it would be 60,000 read at $0.20 + 800 written
    full = 60_800 * 4 * 1.25 / 1e6
    warm = (60_000 * 0.20 + 800 * 4 * 1.25) / 1e6
    assert line.startswith("Idle for 48 min: the 5m cache has expired.")
    assert "up to 60,800 tokens, $" + f"{full:.3f}" in line
    assert f"${full - warm:.3f} more than with a warm cache" in line
    text = estimate.render(est)
    assert text.splitlines()[2] == line  # right after the header, before any detail


def test_warm_cache_says_when_it_expires(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, now=s.now + timedelta(seconds=30))
    assert est["certain"][0].startswith("Cache warm (5m TTL, expires in 4 min)")


def test_one_hour_ttl_is_still_warm_after_twelve_minutes(env, store):
    _history(env.projects)
    s = _current(env.projects, ttl="1h")
    est = _est(store, s, now=s.now + timedelta(minutes=12))
    assert est["computed"]["cache_ttl"] == "1h" and est["computed"]["cached"] == 60_000


def test_prefix_breakers_warn(env, store):
    _history(env.projects)
    s = _current(env.projects)
    est = _est(store, s, model="claude-sonnet-5-5", edits=("CLAUDE.md", "src/app.py"))
    w = " ".join(est["warnings"])
    assert "Switching model" in est["certain"][0]
    # where CLAUDE.md sits in the prompt is not in the transcripts: measured, not asserted
    assert "CLAUDE.md: whether Claude Code re-reads it mid-session" in w
    assert "src/app.py" not in w
    assert est["computed"]["cached"] == 0


def test_loaded_but_unused_files_warn(env, store):
    _history(env.projects)
    s = _current(env.projects, reads=12)
    est = _est(store, s)
    assert any("12 files are loaded" in w for w in est["warnings"])


def test_family_is_guessed_from_text_when_not_declared():
    g = estimate.guess_family_from_text
    assert g("relis le module de paiement") == "review"
    assert g("refactor the cart into two classes") == "refactor"
    assert g("how fast is the import?") == "measure"
    assert g("what does this function return?") == "question"


def test_cli_estimate_runs(env, capsys):
    _history(env.projects)
    s = _current(env.projects)
    assert main(["estimate", "--session", s.session_id[:8], "--family", "refactor", "rename foo"]) == 0
    out = capsys.readouterr().out
    assert "Computed" in out and "Unvalidated estimate" in out and "floor (exact)" in out
    assert "Predicted" not in out
    assert "harbor-api" in out  # picked by session id, not by the current directory


def test_a_guessed_family_is_recorded_but_not_used_as_the_basis(env, store):
    _history(env.projects)  # ten 'refactor' tasks, enough for a family basis
    s = _current(env.projects)
    p = _est(store, s, text="refactor the cart")["predicted"]
    assert p["family"] == "refactor" and p["family_source"] == "guessed from your text"
    assert p["basis_is_family"] is False and p["basis"].startswith("all ")
