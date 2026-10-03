"""Status line and prompt hook: driven by the JSON Claude Code sends on stdin."""

from __future__ import annotations

import io
import json
import sys

from conftest import guarded
from fake_transcripts import FakeSession, bash, read
from tokentrail.cli import main


def _feed(monkeypatch, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


def _session(env, counter=True, n_reads=0):
    s = FakeSession(env.projects)
    s.prompt("tidy the cart")
    for i in range(n_reads):
        s.call(read=1000, write=100, tools=read(f"src/m{i}.py"))
    s.call(new=2, read=40_000, write=2_000, out=500, tools=bash("pytest"))
    s.call(read=42_000, write=300, out=400)
    if counter:
        s.counter()
    return s.write(), s


def test_statusline_shows_only_what_changes_when_all_is_well(env, capsys, monkeypatch):
    tp, _ = _session(env)
    _feed(monkeypatch, {"transcript_path": str(tp), "context_window": {"used_percentage": 21.4},
                        "prompt_cache": {"warm": True, "misses": 0}})
    assert main(["statusline"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("ctx 21% | $") and line.endswith(" at API rates")
    for reassurance in ("counter ok", "unverified", "recovered", "cache 9", "!"):
        assert reassurance not in line


def test_statusline_shows_plan_usage_only_when_claude_code_passes_it(env, capsys, monkeypatch):
    tp, _ = _session(env)
    _feed(monkeypatch, {"transcript_path": str(tp), "context_window": {"used_percentage": 5},
                        "rate_limits": {"five_hour": {"used_percentage": 23.5},
                                        "seven_day": {"used_percentage": 41.2}}})
    main(["statusline"])
    assert "| 5h 24% | 7d 41% |" in capsys.readouterr().out
    _feed(monkeypatch, {"transcript_path": str(tp)})  # API key users: no rate_limits
    main(["statusline"])
    out = capsys.readouterr().out
    assert "5h" not in out and "7d" not in out


def test_statusline_raises_problems(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=20_000)
    s.call(read=20_000, write=500)
    s.cost_state({"claude-opus-5-5": {"inputTokens": 999, "cacheReadInputTokens": 0,
                                      "cacheCreationInputTokens": 0, "outputTokens": 0}})
    tp = s.write()
    _feed(monkeypatch, {"transcript_path": str(tp), "context_window": {"context_window_size": 1_000_000},
                        "prompt_cache": {"warm": False, "recache_tokens_if_cold": 20_500, "misses": 2,
                                         "last_miss_cause": {"causes": ["tools_changed"]}}})
    main(["statusline"])
    line = capsys.readouterr().out
    assert "! cache cold: next message re-caches 20k" in line
    assert "! 2 cache misses (last: tools_changed)" in line
    assert "! COUNTER MISMATCH: run tokentrail check" in line


def test_statusline_detects_a_miss_itself_on_versions_without_prompt_cache(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=50_000)
    s.call(new=2, write=50_200)  # the whole context written again
    tp = s.write()
    _feed(monkeypatch, {"transcript_path": str(tp)})
    main(["statusline"])
    assert "! cache missed on the last call" in capsys.readouterr().out


def test_statusline_survives_anything(env, capsys, monkeypatch):
    for payload in ("", "not json", "[1,2]", {"transcript_path": "/nowhere.jsonl"}, {"transcript_path": 3}):
        _feed(monkeypatch, payload)
        assert main(["statusline"]) == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 1 and out.startswith("tokentrail:")


def test_hook_answers_with_a_user_only_message(env, capsys, monkeypatch):
    tp, s = _session(env, n_reads=12)
    _feed(monkeypatch, {"transcript_path": str(tp), "cwd": s.cwd, "prompt": "now split it in two",
                        "hook_event_name": "UserPromptSubmit", "source": "user"})
    assert main(["hook", "prompt"]) == 0
    out = json.loads(capsys.readouterr().out)
    # systemMessage is shown to the user; nothing else, so nothing reaches the model's context
    assert set(out) == {"systemMessage"}
    msg = out["systemMessage"]
    assert msg.startswith("tokentrail: next call ") and "floor $" in msg
    assert "12 files are loaded" in msg


def test_hook_is_silent_when_it_has_nothing_to_say_or_fails(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("first prompt of a session")
    tp = s.write()
    payloads = [
        {"transcript_path": str(tp), "prompt": "hi"},  # no call yet: nothing to start from
        {"transcript_path": str(tp), "prompt": "x", "source": "system"},  # turns Claude Code starts
        "garbage",
        {"transcript_path": "/nowhere.jsonl", "prompt": "x"},
    ]
    for p in payloads:
        _feed(monkeypatch, p)
        assert main(["hook", "prompt"]) == 0
        assert capsys.readouterr().out == ""


def test_live_commands_stay_local_and_in_the_data_dir(env, capsys, monkeypatch):
    tp, s = _session(env)
    with guarded([env.data]) as g:
        _feed(monkeypatch, {"transcript_path": str(tp)})
        main(["statusline"])
        _feed(monkeypatch, {"transcript_path": str(tp), "cwd": s.cwd, "prompt": "x"})
        main(["hook", "prompt"])
        main(["setup"])
    assert g.violations == []
    out = capsys.readouterr().out
    assert '"UserPromptSubmit"' in out and " statusline" in out


def test_setup_snippet_matches_the_documented_shape(env, capsys):
    main(["setup"])
    text = capsys.readouterr().out
    snippet = json.loads(text[text.index("{"): text.rindex("}") + 1])
    # the command is this tokentrail, by absolute path when it can be found
    # (on Windows the executable is tokentrail.EXE)
    assert snippet["statusLine"]["type"] == "command"
    assert _runs_tokentrail(snippet["statusLine"]["command"], "statusline")
    [group] = snippet["hooks"]["UserPromptSubmit"]
    assert group["hooks"][0]["type"] == "command"
    assert _runs_tokentrail(group["hooks"][0]["command"], "hook prompt")


def _runs_tokentrail(command: str, args: str) -> bool:
    from pathlib import PureWindowsPath

    exe, _, rest = command.rpartition(" " + args.split()[0])
    return rest == args[len(args.split()[0]):] and PureWindowsPath(exe.strip('"')).stem.lower() == "tokentrail"
