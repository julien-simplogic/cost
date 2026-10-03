"""Our sums vs Claude Code's counter: direction matters, and every gap is shown with its size."""

from __future__ import annotations

import json

from fake_transcripts import FakeSession, bash
from tokentrail import live
from tokentrail.cli import main
from tokentrail.collectors import claude_code as cc

SECRET = "refactor the BillingGateway for client Zorblax"


def _session(env, extra_counter=None, shrink=0):
    s = FakeSession(env.projects)
    s.prompt(SECRET)
    s.call(new=2, write=30_000, out=200, tools=(("WebFetch", {"url": "https://example.com", "prompt": "x"}),))
    s.call(read=30_000, write=400, out=300)
    usage = {m: {"inputTokens": u[0] - shrink, "cacheReadInputTokens": u[1], "cacheCreationInputTokens": u[2],
                 "outputTokens": u[3]} for m, u in s.true_usage.items()}
    usage.update(extra_counter or {})
    s.cost_state(usage)
    s.write()
    return s


def test_side_calls_counted_only_by_claude_code_are_shown_not_called_a_bug(env, capsys):
    # WebFetch has a small model read the page: Claude Code counts it, no call line records it
    _session(env, extra_counter={"claude-haiku-4-5-20251001": {
        "inputTokens": 144_000, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "outputTokens": 12_000}})
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    banner = out.splitlines()[1]
    assert banner.startswith("Checked against Claude Code's counter in 1 of 1 sessions: never more than it")
    assert "MISMATCH" not in out
    assert "holds 144k input tokens (cache included) that no call line records" in out
    assert "input -144,000" in out and "models only in the counter: claude-haiku-4-5-20251001" in out


def test_counting_more_than_the_counter_is_an_alarm(env, capsys):
    _session(env, shrink=1)  # the counter has 1 input token less than our lines: we overcount
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("MISMATCH: in 1 of 1 checkable sessions tokentrail counts MORE")
    assert "input +1 (+<0.01%)" in out


def test_differently_named_models_still_match_in_total(env, capsys):
    s = FakeSession(env.projects, model="claude-opus-5")
    s.prompt("x")
    s.call(new=2, write=9_000, out=100)
    u = s.true_usage["claude-opus-5"]
    s.cost_state({"claude-opus-5[1m]": {"inputTokens": u[0], "cacheReadInputTokens": u[1],
                                        "cacheCreationInputTokens": u[2], "outputTokens": u[3]}})
    s.write()
    [sf] = cc.discover(env.projects)
    checks = {c.model: c for c in cc.parse_session(sf).checks}
    assert set(checks) == {"claude-opus-5[1m]", "claude-opus-5"}  # neither side vanishes
    main(["report", "--since", "all"])
    assert capsys.readouterr().out.splitlines()[1].startswith("Verified")


def test_unknown_types_say_whether_they_carry_tokens(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.raw(json.dumps({"type": "frame-link", "frame": "a"}))
    s.raw(json.dumps({"type": "bridge-session", "meta": {"usage": {"input_tokens": 50}}}))
    s.write()
    [sf] = cc.discover(env.projects)
    st = cc.parse_session(sf).stats
    assert st.unknown_types == {"frame-link": 1, "bridge-session": 1}
    assert st.unknown_with_tokens == {"bridge-session": 1}
    main(["report", "--since", "all"])
    assert "these carry token fields and may hold usage counted nowhere: bridge-session=1" in capsys.readouterr().out


def test_diagnose_prints_numbers_and_model_names_only(env, capsys):
    s = _session(env, extra_counter={"claude-haiku-4-5-20251001": {
        "inputTokens": 500, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "outputTokens": 50}})
    assert main(["diagnose", s.session_id[:8]]) == 0
    out = capsys.readouterr().out
    assert "claude-haiku-4-5-20251001" in out and "-500" in out and "MISMATCH" in out
    assert "Zorblax" not in out and "BillingGateway" not in out and "example.com" not in out


def test_diagnose_unknown_session(env, capsys):
    _session(env)
    assert main(["diagnose", "nope"]) == 2
    assert "No session id starts with 'nope'" in capsys.readouterr().err


def test_setup_uses_the_running_executable(tmp_path, monkeypatch):
    exe = tmp_path / "venv bin" / "tokentrail"
    exe.parent.mkdir()
    exe.write_text("")
    monkeypatch.setattr("sys.argv", [str(exe), "setup"])
    snippet = json.loads(live.setup_snippet())
    assert snippet["statusLine"]["command"] == f'"{exe.resolve()}" statusline'
    assert snippet["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == f'"{exe.resolve()}" hook prompt'
