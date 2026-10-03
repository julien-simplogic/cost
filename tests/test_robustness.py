"""A stranger's machine: every case gives a clear message and a next step, never a traceback."""

from __future__ import annotations

import io
import json
import sys

import pytest

from fake_transcripts import FakeSession, bash
from tokentrail import paths, report
from tokentrail.cli import main
from tokentrail.collectors import claude_code as cc


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    return code, out, err


# ------------------------------------------------------------- nothing to read


def test_no_claude_directory(env, capsys):
    (env.projects).rmdir()
    for cmd in (["report"], ["check"], ["estimate", "hi"], ["ingest"]):
        code, out, err = run(capsys, *cmd)
        assert code == 2 and out == ""
        assert "No Claude Code transcripts found" in err and "does not exist" in err
        assert "CLAUDE_CONFIG_DIR" in err and "--source-dir" in err


def test_empty_projects_directory(env, capsys):
    code, _, err = run(capsys, "report")
    assert code == 2 and "contains no session files" in err


def test_sessions_without_any_model_call(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("hello?")  # Claude never answered
    s.write()
    code, _, err = run(capsys, "report")
    assert code == 1 and "no model call yet" in err


def test_period_without_calls_says_what_the_history_covers(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.write()
    code, _, err = run(capsys, "report", "--since", "2030-01-01")
    assert code == 1 and "2026-09-28 to 2026-09-28" in err and "--since all" in err


# ------------------------------------------------------------- tiny histories


def test_single_three_message_session(env, capsys):
    s = FakeSession(env.projects)
    for text in ("one", "two", "three"):
        s.prompt(text)
        s.call(write=500)
    s.write()
    code, out, _ = run(capsys, "report", "--since", "all")
    assert code == 0 and "3 tasks, 3 model calls" in out
    code, out, _ = run(capsys, "estimate", "--session", s.session_id[:8], "four")
    assert code == 0
    assert "no past task to compare with yet" not in out  # 2 earlier tasks exist
    code, out, _ = run(capsys, "check")
    assert code == 0


def test_first_task_has_no_history_and_says_so(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("only one")
    s.call(write=500)
    s.write()
    code, out, _ = run(capsys, "estimate", "--session", s.session_id[:8], "next")
    assert code == 0
    assert "no past task to compare with yet (0 found)" in out
    assert "no history: pass --max-turns" in out


def test_session_without_tool_calls_or_subagents(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("what is a monad?")
    s.call(write=900, out=700)
    s.prompt("and a functor?")
    s.call(read=900, write=800, out=600)
    s.write()
    code, out, _ = run(capsys, "report", "--since", "all", "--json")
    rep = json.loads(out)
    cats = {c["key"]: c["turns"] for c in rep["categories"]}
    assert cats == {"subagents": 0, "reviews": 0, "tool_turns": 0, "user_turns": 2}
    assert rep["totals"]["inexact_output_turns"] == 0


# ------------------------------------------------------------- verification


def test_report_says_unverified_first_when_no_counter(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=500)
    s.write()
    _, out, _ = run(capsys, "report", "--since", "all")
    assert out.splitlines()[1].startswith("NOT VERIFIED: none of these 1 sessions")


def test_report_says_partly_verified_first(env, capsys):
    a = FakeSession(env.projects, project="acme-webshop")
    a.prompt("x")
    a.call(write=500)
    a.counter()
    a.write()
    b = FakeSession(env.projects, project="harbor-api")
    b.prompt("y")
    b.call(write=500)
    b.write()
    _, out, _ = run(capsys, "report", "--since", "all")
    assert out.splitlines()[1].startswith("Partly verified: input matches")
    assert "in 1 of 2 sessions" in out.splitlines()[1]


def test_report_says_verified_first(env, capsys):
    a = FakeSession(env.projects)
    a.prompt("x")
    a.call(write=500, tools=bash("ls"))
    a.call(read=500, write=50)
    a.counter()
    a.write()
    _, out, _ = run(capsys, "report", "--since", "all")
    assert out.splitlines()[1].startswith("Verified: input matches")


# ------------------------------------------------------------- damaged files


def test_truncated_last_line(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=500)
    s.call(read=500, write=20)
    main_file = s.write()
    text = main_file.read_text()
    main_file.write_text(text + '{"type":"assistant","message":{"id":"msg_cut","us')  # killed mid-write
    code, out, err = run(capsys, "report", "--since", "all")
    assert code == 0 and "2 model calls" in out
    assert "truncated_last_line=1" in out
    assert "cut-off line" in err


def test_non_utf8_bytes_and_unreadable_file(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=500)
    f = s.write()
    f.write_bytes(f.read_bytes() + b'{"type":"user","message":"\xff\xfe"}\n')
    (f.parent / "not-a-file.jsonl").mkdir()  # opening it fails like an unreadable file
    code, out, _ = run(capsys, "report", "--since", "all")
    assert code == 0 and "1 model calls" in out
    assert "file_unreadable=1" in out


def test_main_thread_without_stop_reason_is_not_flagged(env):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=500, stop_reason=None)  # older versions may not write stop_reason
    s.write()
    [sf] = cc.discover(env.projects)
    [r] = cc.parse_session(sf).records
    assert r.output_exact is True


# ------------------------------------------------------------- Windows


def test_windows_working_directory_names_the_project(env):
    s = FakeSession(env.projects)
    s.cwd = "D:\\work\\acme-webshop"
    s.prompt("x")
    s.call()
    s.write()
    [sf] = cc.discover(env.projects)
    assert cc.parse_session(sf).records[0].project == "acme-webshop"
    assert cc.project_name("D:/work/harbor-api") == "harbor-api"
    assert cc.project_name("/srv/demo/lumen-docs") == "lumen-docs"


def test_windows_data_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKENTRAIL_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert paths.data_dir() == tmp_path / "AppData" / "Local" / "tokentrail"


def test_console_that_cannot_print_every_character(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.subagent([{"write": 900, "out": 3, "tools": bash("ls")}, {"read": 900, "write": 10}], final_out=50)
    s.write()
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp437")  # an old Windows console: no double dagger
    monkeypatch.setattr(sys, "stdout", console)
    assert main(["report", "--since", "all"]) == 0
    console.flush()
    assert b"?" in raw.getvalue() and b"tokentrail report" in raw.getvalue()


# ------------------------------------------------------------- bad input, bad state


def test_bad_date(env, capsys):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.write()
    code, _, err = run(capsys, "report", "--since", "last tuesday")
    assert code == 2 and "Can't read the date" in err and "2026-09-01" in err


def test_broken_user_price_file(env, capsys):
    run(capsys, "prices", "--init")
    paths.user_prices_path().write_text("verified_on = 2026-13-45\n[models.x]\ninput = 'a'\n")
    code, _, err = run(capsys, "prices")
    assert code == 2 and "can't be read" in err and "delete it" in err


def test_corrupt_database(env, capsys):
    paths.ensure_data_dir()
    paths.db_path().write_bytes(b"this is not sqlite" * 100)
    code, _, err = run(capsys, "report")
    assert code == 2 and "unreadable" in err and "Delete that file" in err


def test_data_directory_cannot_be_created(env, capsys, monkeypatch):
    def refuse():
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(paths, "ensure_data_dir", refuse)
    code, _, err = run(capsys, "report")
    assert code == 2 and "Cannot create tokentrail's data directory" in err and "TOKENTRAIL_HOME" in err


def test_unexpected_error_is_a_message(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.write()

    def boom(*a, **k):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(report, "build", boom)
    code, _, err = run(capsys, "report")
    assert code == 70 and "unexpected error (ZeroDivisionError: oops)" in err and "--debug" in err
    with pytest.raises(ZeroDivisionError):
        main(["--debug", "report"])


# ------------------------------------------------------------- the standalone check


def test_verify_dedup_script_reproduces_the_claim(env, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "verify_dedup", Path(__file__).parent.parent / "scripts" / "verify_dedup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=5000, tools=bash("ls") + bash("pwd"))  # 3 lines for one call
    s.call(read=5000, write=100)  # 2 lines
    s.counter()
    s.write()
    sys.argv = ["verify_dedup.py", str(env.projects)]
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "1 carry Claude Code's counter" in out
    assert "equals the counter for input and cache: 1 of 1; above it (would mean overcounting): 0" in out
    # call 1 is 3 lines, call 2 is 2 lines, each repeating its usage:
    #   input       (2*3 + 2*2) / (2 + 2)          = 10 / 4        = 2.50x
    #   cache read  (0*3 + 5000*2) / 5000          = 2.00x
    #   cache write (5000*3 + 100*2) / 5100        = 15200 / 5100  = 2.98x
    #   output      (100*3 + 100*2) / 200          = 2.50x
    ratios = [l for l in out.splitlines() if l.startswith("naive / one per message.id")][0].split()[-4:]
    assert ratios == ["2.50x", "2.00x", "2.98x", "2.50x"]


def test_verify_dedup_script_without_counter(env, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "verify_dedup", Path(__file__).parent.parent / "scripts" / "verify_dedup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.write()
    sys.argv = ["verify_dedup.py", str(env.projects)]
    assert mod.main() == 1
    assert "does not write the counter" in capsys.readouterr().out


# ------------------------------------------------------------- nothing waits forever


class _SilentStdin:
    """An open pipe nobody writes to: reading it would block forever."""

    def isatty(self):
        return False

    def read(self, *a):
        raise AssertionError("read from stdin that nobody writes to")


def test_estimate_without_text_does_not_wait_for_stdin(env, capsys, monkeypatch):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(write=500)
    s.write()
    monkeypatch.setattr(sys, "stdin", _SilentStdin())
    code, out, _ = run(capsys, "estimate", "--session", s.session_id[:8])
    assert code == 0 and "Computed" in out


def test_live_commands_typed_by_hand_do_not_wait(env, capsys, monkeypatch):
    class Terminal(_SilentStdin):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Terminal())
    assert main(["statusline"]) == 0
    assert capsys.readouterr().out.startswith("tokentrail:")
    assert main(["hook", "prompt"]) == 0
    assert capsys.readouterr().out == ""


def test_demo_does_not_sit_in_the_folder_it_deletes():
    from pathlib import Path

    demo = (Path(__file__).parent.parent / "scripts" / "demo.py").read_text()
    assert "chdir" not in demo  # Windows cannot delete the current directory
