"""Shared fixtures, and the audit-hook guard behind the privacy tests.

Python audit hooks (PEP 578) see every socket, subprocess, file open and
filesystem change made through the interpreter, including from C modules
like sqlite3. The guard turns them into test failures while active.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).parent))

NETWORK_EVENTS = {
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.sendto", "socket.__new__",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
    "webbrowser.open",
}
FS_EVENTS_1 = {  # events whose first argument is a path being changed
    "os.mkdir", "os.remove", "os.rmdir", "os.unlink", "os.chmod", "os.chown", "os.truncate",
    "os.utime", "os.symlink", "os.link", "shutil.rmtree", "shutil.chown",
}
FS_EVENTS_2 = {"os.rename", "os.replace", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
               "shutil.move", "shutil.copytree", "shutil.make_archive"}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class Guard:
    def __init__(self) -> None:
        self.active = False
        self.allowed: list[Path] = []
        self.violations: list[str] = []

    def _allowed(self, p) -> bool:
        if isinstance(p, int) or p is None:
            return True  # an already-open descriptor
        if isinstance(p, bytes):
            p = os.fsdecode(p)
        p = str(p)
        if p == ":memory:" or p == "":
            return True
        rp = Path(os.path.realpath(p))
        return any(rp == a or a in rp.parents for a in self.allowed)

    def hook(self, event: str, args: tuple) -> None:
        if not self.active:
            return
        if event in NETWORK_EVENTS:
            self.violations.append(f"{event} {args[:2]!r}")
            raise RuntimeError(f"network/process call during analysis: {event}")
        bad = None
        if event == "open":
            path, mode, flags = (list(args) + [None, None, 0])[:3]
            writing = bool(mode and any(c in str(mode) for c in "wax+")) or bool((flags or 0) & _WRITE_FLAGS)
            if writing and not self._allowed(path):
                bad = path
        elif event in FS_EVENTS_1 and args and not self._allowed(args[0]):
            bad = args[0]
        elif event in FS_EVENTS_2 and len(args) > 1 and not self._allowed(args[1]):
            bad = args[1]
        elif event == "sqlite3.connect" and args and not self._allowed(args[0]):
            bad = args[0]
        if bad is not None:
            self.violations.append(f"{event} {bad!r}")
            raise RuntimeError(f"write outside the data directory: {event} {bad!r}")


GUARD = Guard()
sys.addaudithook(GUARD.hook)


@contextmanager
def guarded(allowed: list[Path]) -> Iterator[Guard]:
    GUARD.allowed = [Path(os.path.realpath(a)) for a in allowed]
    GUARD.violations = []
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # importing must not write .pyc files either
    GUARD.active = True
    try:
        yield GUARD
    finally:
        GUARD.active = False
        sys.dont_write_bytecode = old


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated HOME, Claude Code dir and tokentrail data dir."""
    home = tmp_path / "home"
    claude = home / ".claude"
    projects = claude / "projects"
    data = tmp_path / "tokentrail-data"
    projects.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("TOKENTRAIL_HOME", str(data))
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.chdir(tmp_path)

    class Env:
        pass

    e = Env()
    e.home, e.projects, e.data, e.tmp = home, projects, data, tmp_path
    return e
