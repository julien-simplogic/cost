"""The promise "nothing leaves the machine" is tested, not asserted.

1. The analysis path makes no network call (and starts no process that could).
2. tokentrail writes nowhere but its own data directory.
3. The transcripts it reads are left untouched.

Each guard is first shown to catch a violation, so a passing test means
something.
"""

from __future__ import annotations

import ast
import hashlib
import socket
from pathlib import Path

import pytest

from conftest import guarded
from fake_transcripts import FakeSession, bash, edit, read
from tokentrail.cli import main

SRC = Path(__file__).parent.parent / "src" / "tokentrail"


def _populate(projects: Path) -> None:
    s = FakeSession(projects)
    s.prompt("tidy the cart module")
    s.call(new=3, write=20_000, tools=read("src/cart.py"))
    s.call(read=20_000, write=900, tools=edit("src/cart.py"))
    s.subagent([{"new": 2, "write": 9000, "tools": bash("ls")}, {"read": 9000, "write": 300}], final_out=150)
    s.call(read=21_000, write=400)
    s.write()


def _all_commands(env) -> list[list[str]]:
    return [
        ["where"],
        ["ingest"],
        ["ingest", "--force"],
        ["report", "--since", "all"],
        ["report", "--since", "all", "--json", "--anonymize"],
        ["estimate", "--family", "refactor", "--edits", "CLAUDE.md", "make it faster"],
        ["estimate", "--json", "--add", str(env.projects.parent / "settings.json"), "x"],
        ["prices", "--init"],
        ["prices"],
        ["tag", "zzz", "review"],
    ]


def _hashes(root: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


# ------------------------------------------------------------------ network


def test_network_guard_catches_a_connection(env):
    with guarded([env.data]) as g:
        with pytest.raises(RuntimeError):
            socket.create_connection(("192.0.2.1", 443), timeout=0.01)
    assert g.violations


def test_analysis_makes_no_network_call(env, capsys):
    _populate(env.projects)
    with guarded([env.data]) as g:
        for argv in _all_commands(env):
            main(argv)
    assert g.violations == []
    assert "tokentrail report" in capsys.readouterr().out


def test_source_imports_no_network_or_process_module():
    banned = {"socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib",
              "smtplib", "subprocess", "webbrowser", "xmlrpc", "asyncio"}
    found = []
    for f in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            found += [f"{f.name}: {n}" for n in names if n.split(".")[0] in banned]
    assert found == []


# ------------------------------------------------------------------ writes


def test_write_guard_catches_a_stray_write(env):
    with guarded([env.data]) as g:
        with pytest.raises(RuntimeError):
            (env.tmp / "stray.txt").write_text("oops")
    assert g.violations


def test_write_guard_sees_the_tools_own_writes(env):
    """With the data dir *not* allowed, the guard must flag tokentrail's database."""
    _populate(env.projects)
    with guarded([env.tmp / "elsewhere"]) as g:
        with pytest.raises(RuntimeError):
            main(["ingest"])
    assert any("tokentrail" in v for v in g.violations), g.violations


def test_writes_only_in_its_own_data_dir(env):
    _populate(env.projects)
    with guarded([env.data]) as g:
        for argv in _all_commands(env):
            main(argv)
    assert g.violations == []
    assert (env.data / "tokentrail.sqlite3").is_file()
    assert (env.data / "prices.toml").is_file()


def test_transcripts_are_never_modified(env):
    _populate(env.projects)
    before = _hashes(env.home)
    with guarded([env.data]):
        for argv in _all_commands(env):
            main(argv)
    assert _hashes(env.home) == before
