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
    assert banner.startswith("Partly verified: never more than Claude Code's counter")
    assert "The totals below are a MINIMUM" in banner
    assert "a single snapshot can't tell a hidden call from an error" in banner  # one cost-state only
    assert "one side only (not compared): claude-haiku-4-5-20251001" in banner
    assert "MISMATCH" not in out
    assert "holds 144k input tokens (cache included) that no call line records" in out


def test_counting_more_than_the_counter_is_an_alarm(env, capsys):
    _session(env, shrink=1)  # the counter has 1 input token less than our lines: we overcount
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("MISMATCH: in 1 of 1 checkable sessions tokentrail counts MORE")
    assert "input +1 (+<0.01%)" in out


def test_a_context_qualifier_does_not_split_a_model(env, capsys):
    # the counter says claude-opus-5[1m], the lines claude-opus-5: same model
    s = FakeSession(env.projects, model="claude-opus-5")
    s.prompt("x")
    s.call(new=2, write=9_000, out=100)
    u = s.true_usage["claude-opus-5"]
    s.cost_state({"claude-opus-5[1m]": {"inputTokens": u[0], "cacheReadInputTokens": u[1],
                                        "cacheCreationInputTokens": u[2], "outputTokens": u[3]}})
    s.write()
    [sf] = cc.discover(env.projects)
    res = cc.parse_session(sf)
    assert [c.model for c in res.checks] == ["claude-opus-5"]
    assert res.coverage["counter_model_names"] == ["claude-opus-5[1m]"]
    main(["report", "--since", "all"])
    assert capsys.readouterr().out.splitlines()[1].startswith("Verified")
    main(["diagnose", s.session_id[:8]])
    assert "compared without [..] qualifiers" in capsys.readouterr().out


def test_a_model_on_one_side_only_never_reads_as_verified(env, capsys):
    # totals match, but the counter attributes the usage to another model
    s = FakeSession(env.projects, model="claude-opus-5")
    s.prompt("x")
    s.call(new=2, write=9_000, out=100)
    u = s.true_usage["claude-opus-5"]
    s.cost_state({"claude-sonnet-5": {"inputTokens": u[0], "cacheReadInputTokens": u[1],
                                      "cacheCreationInputTokens": u[2], "outputTokens": u[3]}})
    s.write()
    main(["report", "--since", "all"])
    banner = capsys.readouterr().out.splitlines()[1]
    assert banner.startswith("Partly verified: totals match Claude Code's counter, but not model by model")
    assert "(not compared): claude-sonnet-5, claude-opus-5." in banner


def test_verified_is_unreachable_with_a_model_on_one_side():
    from itertools import product

    from tokentrail.report import verification_banner

    for k, ok, over, spread, under, unk, c_only, l_only in product(
            (1, 3), (0, 1, 3), (0, 1), (0, 1), (0, 1), (0, 1), ([], ["m[1m]"]), ([], ["m"])):
        ck = {"sessions_in_period": 3, "sessions_checkable": k, "sessions_input_exact": min(ok, k),
              "sessions_over": over, "sessions_spread": spread, "sessions_under": under,
              "sessions_shape_unknown": unk, "models_counter_only": c_only, "models_lines_only": l_only,
              "counter_input": 100, "lines_input": 90}
        banner = verification_banner(ck)
        if c_only or l_only:
            assert not banner.startswith("Verified"), ck


def _snapshots(env, counter_shortfall):
    """Five turns, a counter snapshot after each; counter_shortfall(turn) = tokens the counter has extra."""
    s = FakeSession(env.projects)
    for turn in range(5):
        s.prompt(f"turn {turn}")
        s.call(new=2, read=10_000 * turn, write=10_000, out=100)
        u = s.true_usage["claude-opus-5-5"]
        s.cost_state({"claude-opus-5-5": {"inputTokens": u[0] + counter_shortfall(turn), "cacheReadInputTokens": u[1],
                                          "cacheCreationInputTokens": u[2], "outputTokens": u[3]}})
    s.write()
    return s


def test_a_gap_that_appears_once_is_a_hidden_call(env, capsys):
    _snapshots(env, lambda t: 50_000 if t >= 2 else 0)  # one jump, then stable
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("Partly verified: never more")
    assert "gap appeared between 1 of 5 snapshots (hidden calls)" in out


def test_a_gap_that_grows_at_every_snapshot_is_a_counting_error(env, capsys):
    _snapshots(env, lambda t: 3_000 * (t + 1))  # grows every turn
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("MISMATCH: in 1 of 1 checkable sessions the gap")
    assert "GAP GROWS between 5 of 5 snapshots" in out


def test_gap_shape_rules():
    from tokentrail.record import gap_shape

    assert gap_shape([])["kind"] == "none"
    assert gap_shape([[3, 0], [6, 0]])["kind"] == "none"
    assert gap_shape([[3, -100]])["kind"] == "unknown"
    assert gap_shape([[3, 0], [6, -9000], [9, -9000], [12, -9000]])["kind"] == "concentrated"
    assert gap_shape([[3, -10], [6, -20], [9, -30], [12, -40]])["kind"] == "spread"
    # three moves but one dominates (>= 90%): still concentrated
    assert gap_shape([[3, -1], [6, -2], [9, -1000], [12, -1003]])["kind"] == "concentrated"


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


def test_a_counter_that_goes_down_is_reported_as_a_restart(env, capsys):
    # a resumed session: the file keeps the old calls, Claude Code's counter starts again from 0
    s = FakeSession(env.projects)
    s.prompt("first run")
    s.call(new=2, write=40_000, out=100)
    u = s.true_usage["claude-opus-5-5"]
    s.cost_state({"claude-opus-5-5": {"inputTokens": u[0], "cacheReadInputTokens": u[1],
                                      "cacheCreationInputTokens": u[2], "outputTokens": u[3]}})
    s.prompt("resumed later")
    s.call(new=2, write=5_000, out=50)
    s.cost_state({"claude-opus-5-5": {"inputTokens": 2, "cacheReadInputTokens": 0,
                                      "cacheCreationInputTokens": 5_000, "outputTokens": 50}})
    s.write()
    main(["report", "--since", "all"])
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("MISMATCH: in 1 of 1 checkable sessions tokentrail counts MORE")
    assert "counter went DOWN 1 time(s): it restarted counting" in out
    assert main(["diagnose", s.session_id[:8]]) == 0
    d = capsys.readouterr().out
    assert "<- counter went DOWN: Claude Code restarted counting" in d and "counter restarts: 1" in d


def test_diagnose_accepts_anonymized_ids_and_finds_sibling_files(env, capsys):
    import hashlib

    s = _session(env)
    sibling = FakeSession(env.projects, session_id="0f0f0f0f-sibling")
    sibling.session_id = s.session_id  # records carrying the same session id, in another file
    sibling.prompt("continued")
    sibling.call()
    (s.dir / "0f0f0f0f-sibling.jsonl").write_text("\n".join(sibling.lines) + "\n")
    anon = hashlib.sha256(s.session_id.encode()).hexdigest()[:8]
    assert main(["diagnose", anon]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"session {s.session_id[:8]}")
    assert "1 other session file(s) in this project carry this session's id (0f0f0f0f)" in out


def test_the_counter_side_is_the_last_snapshot_never_a_sum(env, capsys):
    # snapshots are cumulative and Claude Code sometimes writes the same one twice
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(new=2, write=10_000)
    s.counter()
    s.counter()  # written twice in a row, as observed in a real transcript
    s.call(read=10_000, write=500)
    s.counter()
    s.write()
    [sf] = cc.discover(env.projects)
    [c] = cc.parse_session(sf).checks
    last = s.true_usage["claude-opus-5-5"]
    assert (c.source_input, c.source_cache_read, c.source_cache_write) == tuple(last[:3])
    assert (c.ours_input, c.ours_cache_read, c.ours_cache_write) == tuple(last[:3])
    main(["diagnose", s.session_id[:8], "--raw"])
    out = capsys.readouterr().out
    assert out.count("cache write") >= 3 and "uses the LAST one, never a sum" in out
