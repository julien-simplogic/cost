"""tokentrail command line."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, diagnose, estimate, live, paths, prices, report
from .analysis import parse_since
from .classify import FAMILIES
from .collectors import claude_code
from .errors import TokentrailError, no_transcripts
from .ingest import ingest_claude_code
from .store import Store

ISSUES_HINT = "Nothing was changed. Run again with --debug to see details, and please report it."


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokentrail",
        description="Where your Claude Code tokens go, and what a prompt will cost before you send it. "
        "Reads local transcripts only; nothing leaves your machine.",
    )
    p.add_argument("--version", action="version", version=f"tokentrail {__version__}")
    p.add_argument("--source-dir", type=Path, help="Claude Code projects dir (default: ~/.claude/projects)")
    p.add_argument("--debug", action="store_true", help="show full errors instead of a short message")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="read new or changed transcripts into the local database")
    s.add_argument("--force", action="store_true", help="re-read everything")

    s = sub.add_parser("report", help="where the tokens went over a period")
    s.add_argument("--since", default="7d", help="7d, 24h, 2w, an ISO date, or 'all' (default: 7d)")
    s.add_argument("--until", help="ISO date (exclusive)")
    s.add_argument("--project", help="only projects whose name contains this")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--anonymize", action="store_true", help="replace project and session names")
    s.add_argument("--logged-only", action="store_true",
                   help="use only each call's own log line for output (no recovered sub-agent figures)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-ingest", action="store_true", help="don't read transcripts first")

    s = sub.add_parser("check", help="how far to trust the numbers: per Claude Code version, per month")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-ingest", action="store_true")

    s = sub.add_parser("estimate", help="what the next prompt will cost, before sending it")
    s.add_argument("text", nargs="*", help="the prompt you are about to send (or use --file)")
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

    sub.add_parser("statusline", help="Claude Code status line (reads Claude Code's JSON on stdin)")
    s = sub.add_parser("hook", help="Claude Code hooks (read Claude Code's JSON on stdin)")
    s.add_argument("event", choices=["prompt"], help="prompt: UserPromptSubmit")
    sub.add_parser("setup", help="print the settings.json lines that turn on the live display")
    s = sub.add_parser("diagnose", help="one session vs Claude Code's counter, numbers only (safe to paste)")
    s.add_argument("session", help="session id or prefix (the hashed ids of `report --anonymize` work too)")
    s.add_argument("--raw", action="store_true", help="also print every cost-state record as written")
    return p


def _ingest(store: Store, args) -> None:
    root = args.source_dir or paths.claude_code_dir()
    res = ingest_claude_code(store, root, force=getattr(args, "force", False))
    if res.sessions_seen == 0:
        raise no_transcripts(root, root.is_dir())
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
        if st.unreadable.get("truncated_last_line"):
            print(
                f"note: {st.unreadable['truncated_last_line']} session file(s) end in a cut-off line: the session "
                "was killed, or is still being written. Everything before that line was read.",
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
    try:
        # never crash on a console that can't print a character (e.g. Windows cp437)
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    try:
        return _run(args)
    except TokentrailError as e:
        if args.debug:
            raise
        print(f"tokentrail: {e.message}", file=sys.stderr)
        if e.hint:
            print(f"  what to do: {e.hint}", file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - last resort: a message, never a traceback
        if args.debug:
            raise
        print(f"tokentrail: unexpected error ({type(e).__name__}: {e}).", file=sys.stderr)
        print(f"  what to do: {ISSUES_HINT}", file=sys.stderr)
        return 70


def _open_store() -> Store:
    try:
        paths.ensure_data_dir()
    except OSError as e:
        raise TokentrailError(
            f"Cannot create tokentrail's data directory {paths.data_dir()} ({e.strerror or e}).",
            "Set TOKENTRAIL_HOME to a folder you can write to.",
        ) from e
    try:
        return Store(paths.db_path())
    except sqlite3.DatabaseError as e:
        raise TokentrailError(
            f"tokentrail's database {paths.db_path()} is unreadable ({e}).",
            "Delete that file: it only holds data rebuilt from your transcripts on the next run "
            "(families declared with `tokentrail tag` are lost).",
        ) from e


def _claude_code_stdin() -> str:
    """Claude Code writes its JSON and closes stdin. Typed by hand in a terminal,
    there is nothing to wait for."""
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def _run(args) -> int:
    if args.cmd == "statusline":
        # must print one line whatever happens: Claude Code shows it as is
        try:
            print(live.statusline(_claude_code_stdin()))
        except Exception as e:  # noqa: BLE001
            if args.debug:
                raise
            print(f"tokentrail: {type(e).__name__} (run `tokentrail check`)")
        return 0

    if args.cmd == "hook":
        # never block or slow down a prompt, never write to the model's context
        try:
            out = live.prompt_hook(_claude_code_stdin())
        except Exception:  # noqa: BLE001
            if args.debug:
                raise
            out = None
        if out:
            print(out)
        return 0

    if args.cmd == "diagnose":
        print(diagnose.run(args.source_dir or paths.claude_code_dir(), args.session, raw=args.raw))
        return 0

    if args.cmd == "setup":
        print(f"Add this to {live.where_settings()} (merge with what is already there):\n")
        print(live.setup_snippet())
        print("\nThe status line runs after each reply; the hook runs when you send a prompt and shows")
        print("its estimate to you only (systemMessage): nothing is added to the model's context.")
        print("tokentrail never edits that file itself: it writes only to its own data directory.")
        return 0


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

    with _open_store() as store:
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

        if not store.has_records():
            raise TokentrailError(
                "Transcripts were found but hold no model call yet (sessions Claude never answered, "
                "or only records this version of tokentrail doesn't understand).",
                "Use Claude Code for a bit and run this again; `tokentrail ingest` says what was read.",
                exit_code=1,
            )

        table = prices.load()
        if args.cmd == "report":
            since = None if args.since == "all" else parse_since(args.since)
            until = parse_since(args.until) if args.until else None
            rep = report.build(
                store, table, since=since, until=until, project=args.project,
                top=args.top, anonymize=args.anonymize, logged_only=args.logged_only,
            )
            if not rep["totals"]["turns"] and not args.json:
                first, last = store.time_span()
                raise TokentrailError(
                    f"No model call in this period. Your history runs from {first[:10]} to {last[:10]}.",
                    "Widen the period (--since all) or check --project.",
                    exit_code=1,
                )
            print(json.dumps(rep, indent=2) if args.json else report.render(rep))
            return 0

        if args.cmd == "check":
            chk = report.build_check(store, table)
            print(json.dumps(chk, indent=2) if args.json else report.render_check(chk))
            return 0

        if args.cmd == "estimate":
            text = " ".join(args.text)
            # stdin is read only when asked for (--file -): an open but silent stdin
            # (a pipe, a CI runner) would otherwise wait forever
            if args.file:
                text = sys.stdin.read() if str(args.file) == "-" else args.file.read_text(encoding="utf-8")
            est = estimate.build(store, table, estimate.EstimateInput(
                text=text, add_files=tuple(args.add), session=args.session, family=args.family,
                model=args.model, max_tokens=args.max_tokens, max_turns=args.max_turns,
                edits=tuple(args.edits),
                now=datetime.fromisoformat(args.now) if args.now else None,
            ))
            print(json.dumps(est, indent=2, default=str) if args.json else estimate.render(est))
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
