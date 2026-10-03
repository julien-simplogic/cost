"""The pre-push leak scanner must actually catch leaks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "leak_check", Path(__file__).parent.parent / "scripts" / "leak_check.py"
)
leak_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak_check)  # type: ignore[union-attr]


# Planted leaks are assembled at run time so this file never contains one
# literally (the scanner runs on this repository too).
HOME = "/" + "home/alice"
WIN = "C:" + "\\Users\\alice"
MAIL = "alice.martin" + "@" + "bigcorp-mail.fr"
HOST = "intranet.bigcorp" + ".example-internal.net"


def test_catches_paths_emails_domains_and_denylisted_terms():
    text = "\n".join([
        f"cwd = '{HOME}/clients/bigcorp/app'",
        f"path = r'{WIN}\\code'",
        f"contact: {MAIL}",
        f"see https://{HOST}/wiki",
        "the BigCorp migration",
    ])
    found = leak_check.scan_text(text, "f", ["bigcorp"])
    kinds = " ".join(found)
    assert f"home path {HOME!r}" in kinds
    assert f"home path {WIN!r}" in kinds
    assert f"e-mail {MAIL!r}" in kinds
    assert f"domain {HOST!r}" in kinds
    assert sum("denylisted term" in f for f in found) == 4  # path, e-mail, url, and line 5


def test_allows_what_is_meant_to_be_public():
    text = "\n".join([
        "Co-Authored-By: Claude <noreply@anthropic.com>",
        "https://peps.python.org/pep-0578/ and https://github.com/actions/checkout",
        "~/.claude/projects and /srv/demo/acme-webshop",
        "dev@example.com",
        "+@pytest.fixture in a diff",
    ])
    assert leak_check.scan_text(text, "f", []) == []
