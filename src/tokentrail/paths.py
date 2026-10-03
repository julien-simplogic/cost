"""Where tokentrail reads from and the only place it writes to."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """tokentrail's own directory. Nothing is ever written outside it.

    $TOKENTRAIL_HOME, else the platform's per-user data directory.
    """
    env = os.environ.get("TOKENTRAIL_HOME")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "tokentrail"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "tokentrail"


def db_path() -> Path:
    return data_dir() / "tokentrail.sqlite3"


def user_prices_path() -> Path:
    return data_dir() / "prices.toml"


def claude_code_dir() -> Path:
    """Claude Code's transcript root (read only)."""
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(env).expanduser() if env else Path.home() / ".claude"
    return base / "projects"


def ensure_data_dir() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
