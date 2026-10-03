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


def test_statusline_shows_context_cache_cost_and_verification(env, capsys, monkeypatch):
    tp, _ = _session(env)
    _feed(monkeypatch, {"transcript_path": str(tp), "context_window": {"used_percentage": 21.4}})
    assert main(["statusline"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("tokentrail | ctx 21% | cache ")
    assert "$" in line and line.endswith("counter ok")
    assert "\n" not in line


def test_statusline_without_counter_says_unverified(env, capsys, monkeypatch):
    tp, _ = _session(env, counter=False)
    _feed(monkeypatch, {"transcript_path": str(tp), "context_window": {"context_window_size": 200_000}})
    main(["statusline"])
    line = capsys.readouterr().out.strip()
    assert "ctx 21%" in line and line.endswith("unverified")  # (42,300 + 400) / 200,000


def test_statusline_survives_anything(env, capsys, monkeypatch):
    for payload in ("", "not json", "[1,2]", {"transcript_path": "/nowhere.jsonl"}, {"transcript_path": 3}):
        _feed(monkeypatch, payload)
        assert main(["statusline"]) == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 1 and out.startswith("tokentrail")


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
    assert '"UserPromptSubmit"' in out and "tokentrail statusline" in out


def test_setup_snippet_matches_the_documented_shape(env, capsys):
    main(["setup"])
    text = capsys.readouterr().out
    snippet = json.loads(text[text.index("{"): text.rindex("}") + 1])
    assert snippet["statusLine"] == {"type": "command", "command": "tokentrail statusline"}
    [group] = snippet["hooks"]["UserPromptSubmit"]
    assert group["hooks"][0]["type"] == "command" and group["hooks"][0]["command"] == "tokentrail hook prompt"
