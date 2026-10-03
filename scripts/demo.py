"""Build an invented week of Claude Code transcripts and run tokentrail on it.

Produces the sample output in the README. Every name here is made up.
Usage: python scripts/demo.py   (writes only to a temporary directory)
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from fake_transcripts import FakeSession, bash, edit, read  # noqa: E402

PROJECTS = ["acme-webshop", "harbor-api", "lumen-docs"]


def build(projects: Path) -> FakeSession:
    rng = random.Random(7)
    start = datetime(2026, 9, 21, 8, 30, tzinfo=timezone.utc)
    last = None
    for day in range(7):
        for k in range(2):
            proj = PROJECTS[(day + k) % len(PROJECTS)]
            s = FakeSession(projects, project=proj, start=start + timedelta(days=day, hours=4 * k))
            ctx = 25_000
            for _ in range(rng.randint(2, 4)):
                kind = rng.choice(["question", "refactor", "refactor", "review", "measure"])
                s.prompt("...", command="code-review" if kind == "review" else None)
                if kind == "review":
                    for i in range(rng.randint(6, 14)):
                        s.call(read=ctx, write=3_000, out=400, tools=read(f"src/mod{i % 5}.py"))
                        ctx += 3_000
                    n = rng.randint(4, 9)
                    s.subagent([{"write": 30_000, "out": 3, "tools": read("src/a.py")}]
                               + [{"read": 30_000 + 2_000 * i, "write": 2_000, "out": 3, "tools": read(f"src/b{i}.py")}
                                  for i in range(n)]
                               + [{"read": 30_000 + 2_000 * n, "write": 1_000}], final_out=2_500)
                elif kind == "refactor":
                    s.call(read=ctx, write=2_000, out=600, tools=edit("src/core.py"))
                    for _ in range(rng.randint(3, 25)):
                        s.call(read=ctx, write=1_500, out=900, tools=bash("pytest -q"))
                        ctx += 1_500
                    if rng.random() < 0.6:
                        n = rng.randint(2, 6)
                        s.subagent([{"write": 25_000, "out": 3, "tools": bash("rg TODO")}]
                                   + [{"read": 25_000 + 1_500 * i, "write": 1_500, "out": 3, "tools": bash("rg x")}
                                      for i in range(n)]
                                   + [{"read": 25_000 + 1_500 * n, "write": 800}], final_out=1_800)
                elif kind == "measure":
                    s.call(read=ctx, write=800, out=300, tools=bash("hyperfine ./run"))
                    s.call(read=ctx, write=600, out=500)
                else:
                    s.call(read=ctx, write=1_200, out=700)
                ctx += 2_000
                s.tick(rng.choice([60, 200, 900, 2400]))
            s.write()
            last = s
    return last  # type: ignore[return-value]


def main() -> None:
    from tokentrail.cli import main as cli

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        os.environ["CLAUDE_CONFIG_DIR"] = str(t / "claude")
        os.environ["TOKENTRAIL_HOME"] = str(t / "data")
        last = build(t / "claude" / "projects")
        cli(["--source-dir", str(t / "claude" / "projects"), "ingest"])
        print("\n$ tokentrail report --since 2026-09-21\n")
        cli(["report", "--since", "2026-09-21", "--top", "5", "--no-ingest"])
        print("\n$ tokentrail estimate --family refactor \"split the cart service in two\"\n")
        os.chdir(t)
        now = (last.last_call_at + timedelta(minutes=4, seconds=15)).isoformat()
        cli(["estimate", "--no-ingest", "--now", now, "--session", last.session_id[:8], "--family", "refactor",
             "--edits", "CLAUDE.md", "split the cart service in two"])


if __name__ == "__main__":
    main()
