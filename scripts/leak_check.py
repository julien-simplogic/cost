"""Scan the working tree and every commit for things that must not be published.

Looks for: personal absolute paths, e-mail addresses, unknown domains, and
any term from a private denylist (client names, your own handles). The
denylist is never committed: pass it with --denylist, $TOKENTRAIL_LEAK_DENYLIST,
or keep it in .leak-denylist (git-ignored). One term per line, # for comments.

Checks: tracked and untracked (non-ignored) files, then each commit's
author, committer, message and diff.

Exit 1 when anything is found. Run it before every push.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_EMAILS = re.compile(
    r"^(noreply@anthropic\.com|noreply@github\.com|[\w.+-]+@users\.noreply\.github\.com|"
    r"[\w.+-]+@example\.(com|org|net))$",
    re.IGNORECASE,
)
ALLOWED_DOMAINS = {
    "github.com", "githubusercontent.com", "claude.com", "claude.ai", "code.claude.com", "anthropic.com", "docs.anthropic.com",
    "python.org", "docs.python.org", "peps.python.org", "pypi.org", "opensource.org",
    "example.com", "example.org", "keepachangelog.com", "semver.org", "pipx.pypa.io",
}
PATTERNS = {
    "home path": re.compile(r"(?:/home/|/Users/|/root/|[A-Za-z]:\\+Users\\+)[\w.-]+"),
    "email": re.compile(r"\b[A-Za-z0-9][\w.+-]*@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", re.IGNORECASE),
    "url": re.compile(r"https?://([\w.-]+)", re.IGNORECASE),
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def _denylist(path: str | None) -> list[str]:
    candidates = [path, os.environ.get("TOKENTRAIL_LEAK_DENYLIST"), ".leak-denylist"]
    for c in candidates:
        if c and Path(c).is_file():
            terms = []
            for line in Path(c).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    terms.append(line)
            return terms
    return []


def _domain_ok(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def scan_text(text: str, where: str, deny: list[str]) -> list[str]:
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        loc = f"{where}:{n}"
        for m in PATTERNS["home path"].finditer(line):
            found.append(f"{loc}: home path {m.group(0)!r}")
        for m in PATTERNS["email"].finditer(line):
            if not ALLOWED_EMAILS.match(m.group(0)):
                found.append(f"{loc}: e-mail {m.group(0)!r}")
        for m in PATTERNS["url"].finditer(line):
            if not _domain_ok(m.group(1)):
                found.append(f"{loc}: domain {m.group(1)!r}")
        low = line.lower()
        for term in deny:
            if term.lower() in low:
                found.append(f"{loc}: denylisted term {term!r}")
    return found


def scan_tree(deny: list[str]) -> list[str]:
    found = []
    files = _git("ls-files", "-co", "--exclude-standard", "-z").split("\0")
    for f in filter(None, files):
        p = Path(f)
        if not p.is_file():
            continue
        found += scan_text(f, f"(path) {f}", deny)  # file names leak too
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found += scan_text(text, f, deny)
    return found


def scan_history(deny: list[str]) -> tuple[list[str], int]:
    found: list[str] = []
    try:
        commits = _git("rev-list", "--all").split()
    except subprocess.CalledProcessError:
        return found, 0  # no commits yet
    for c in commits:
        meta = _git("show", "-s", "--format=%an <%ae>%n%cn <%ce>%n%B", c)
        found += scan_text(meta, f"commit {c[:10]} (metadata)", deny)
        diff = _git("show", "--format=", "-p", "--no-color", c)
        found += scan_text(diff, f"commit {c[:10]} (diff)", deny)
    return found, len(commits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--denylist", help="file with private terms, one per line")
    ap.add_argument("--tree-only", action="store_true")
    args = ap.parse_args()
    deny = _denylist(args.denylist)
    found = scan_tree(deny)
    commits = 0
    if not args.tree_only:
        in_history, commits = scan_history(deny)
        found += in_history
    for f in sorted(set(found)):
        print(f)
    print(
        f"leak check: {len(set(found))} finding(s); tree + {commits} commit(s) scanned; "
        f"{len(deny)} private term(s) checked",
        file=sys.stderr,
    )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
