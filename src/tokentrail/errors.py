"""Errors a user can act on. The CLI prints them as a message and a next step,
never as a traceback."""

from __future__ import annotations


class TokentrailError(Exception):
    def __init__(self, message: str, hint: str = "", exit_code: int = 2):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


def no_transcripts(root, exists: bool) -> TokentrailError:
    where = f"{root} does not exist" if not exists else f"{root} contains no session files (*.jsonl)"
    return TokentrailError(
        f"No Claude Code transcripts found: {where}.",
        "If Claude Code uses a custom config directory, set CLAUDE_CONFIG_DIR or pass "
        "--source-dir <its projects folder>. If you have never run Claude Code on this machine "
        "(only claude.ai, the mobile app or cloud sessions), there is nothing local to read: "
        "see 'What v0 does not see' in the README.",
    )
