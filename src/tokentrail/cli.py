"""tokentrail command line."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, estimate, paths, prices, report
from .analysis import parse_since
from .classify import FAMILIES
from .collectors import claude_code
from .ingest import ingest_claude_code
from .store import Store


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokentrail",
        description="Where your Claude Code tokens go, and what a prompt will cost before you send it. "
        "Reads local transcripts only; nothing leaves your machine.",
    )
    p.add_argument("--version", action="version", version=f"tokentrail {__version__}")
    p.add_argument("--source-dir", type=Path, help="Claude Code projects dir (default: ~/.claude/projects)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="read new or changed transcripts into the local database")
    s.add_argument("--force", action="store_true", help="re-read everything")

    s = sub.add_parser("report", help="where the tokens went over a period")
    s.add_argument("--since", default="7d", help="7d, 24h, 2w, an ISO date, or 'all' (default: 7d)")
    s.add_argument("--until", help="ISO date (exclusive)")
    s.add_argument("--project", help="only projects whose name contains this")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--anonymize", action="store_true", help="replace project and session names")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-ingest", action="store_true", help="don't read transcripts first")

    s = sub.add_parser("estimate", help="what the next prompt will cost, before sending it")
    s.add_argument("text", nargs="*", help="the prompt you are about to send (or use --file / stdin)")
    s.add_argument("--file", type=Path, help="read the prompt from a file ('-' for stdin)")
    s.add_argument("--add", type=Path, action="append", default=[], metavar="PATH",
                   help="a file you will attach or @-mention (repeatable)")
    s.add_argument("--edits", action="append", default=[], metavar="PATH",
                   help="a file you are about to change (checks whether it breaks the cache prefix)")
    s.add_argument("--session", help="session id prefix (default: latest session in this directory)")
    s.add_argument("--family", choices=FAMILIES, help="declare the kind of task")
    s.add_argument("--model", help="model for the next call (default: the session's)")
    s.add_argument("--max-tokens", type=int, help="output cap per call, for the ceiling")
    s.add_argument("--max-turns", type=int, help="turn limit, for the ceiling (default: your max for this family)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-ingest", action="store_true")
    s.add_argument("--now", help=argparse.SUPPRESS)  # fixed clock, for demos and tests

    s = sub.add_parser("tag", help="declare a task's family (overrides the guess)")
    s.add_argument("task", help="task id or prefix, as shown by `report`")
    s.add_argument("family", choices=FAMILIES)

    s = sub.add_parser("prices", help="show the price file; --init to make an editable copy")
    s.add_argument("--init", action="store_true")

    sub.add_parser("where", help="show where tokentrail reads and writes")
    return p


def _ingest(store: Store, args) -> None:
    root = args.source_dir or paths.claude_code_dir()
    res = ingest_claude_code(store, root, force=getattr(args, "force", False))
    st = res.stats
    if args.cmd == "ingest":
        print(f"{res.sessions_seen} sessions found, {res.sessions_parsed} read (others unchanged)")
        if st and st.lines:
            print(
                f"{st.lines} records: {st.used} used, {sum(st.ignored.values())} skipped by design, "
                f"{st.not_understood} not understood"
            )
    if st and st.not_understood:
        print(
            f"note: {st.not_understood} transcript records could not be understood and were skipped "
            f"({_fmt_counts({**st.unreadable, **{'type:' + k: v for k, v in st.unknown_types.items()}})})",
            file=sys.stderr,
        )
    newer = sorted(v for v in (st.versions if st else {}) if _vkey(v) > _vkey(claude_code.VERIFIED_VERSION))
    if newer:
        print(
            f"note: transcripts from Claude Code {newer[-1]}; parser verified on "
            f"{claude_code.VERIFIED_VERSION}. Counts are tolerant to changes but check the 'not understood' line.",
            file=sys.stderr,
        )


def _vkey(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))


def _fmt_counts(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)

    if args.cmd == "where":
        print(f"reads:  {args.source_dir or paths.claude_code_dir()}")
        print(f"writes: {paths.data_dir()}  (database, your price file; nothing else)")
        return 0

    if args.cmd == "prices":
        if args.init:
            dest = paths.user_prices_path()
            if dest.exists():
                print(f"{dest} already exists; edit it directly")
            else:
                paths.ensure_data_dir()
                dest.write_text(prices.packaged_prices_text(), encoding="utf-8")
                print(f"wrote {dest}; edit it, keep verified_on up to date")
        table = prices.load()
        age = table.age_days()
        print(f"price file: {table.path}")
        print(f"verified on: {table.verified_on} ({age} days ago)" if age is not None else "verified on: never")
        if table.is_stale():
            print("stale: check current prices, then update the file")
        for name, p in table.models.items():
            print(f"  {name:22} in ${p.input:>6.2f}  out ${p.output:>6.2f}  cache read ${p.cache_read:.2f}  /MTok")
        return 0

    paths.ensure_data_dir()
    with Store(paths.db_path()) as store:
        if args.cmd == "tag":
            ids = store.tag(claude_code.SOURCE, args.task, args.family)
            if len(ids) == 1:
                print(f"task {ids[0][:8]} -> {args.family}")
                return 0
            print("no task matches" if not ids else f"{len(ids)} tasks match; give a longer prefix")
            return 1

        if args.cmd == "ingest" or not getattr(args, "no_ingest", False):
            _ingest(store, args)
        if args.cmd == "ingest":
            return 0

        table = prices.load()
        if args.cmd == "report":
            since = None if args.since == "all" else parse_since(args.since)
            until = parse_since(args.until) if args.until else None
            rep = report.build(
                store, table, since=since, until=until, project=args.project,
                top=args.top, anonymize=args.anonymize,
            )
            print(json.dumps(rep, indent=2) if args.json else report.render(rep))
            return 0

        if args.cmd == "estimate":
            text = " ".join(args.text)
            if args.file:
                text = sys.stdin.read() if str(args.file) == "-" else args.file.read_text(encoding="utf-8")
            elif not text and not sys.stdin.isatty():
                text = sys.stdin.read()
            est = estimate.build(store, table, estimate.EstimateInput(
                text=text, add_files=tuple(args.add), session=args.session, family=args.family,
                model=args.model, max_tokens=args.max_tokens, max_turns=args.max_turns,
                edits=tuple(args.edits),
                now=datetime.fromisoformat(args.now) if args.now else None,
            ))
            print(json.dumps(est, indent=2, default=str) if args.json else estimate.render(est))
            return 1 if "error" in est else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
