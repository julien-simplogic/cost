"""Local SQLite store. Lives in tokentrail's data directory and nowhere else."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from .record import CounterCheck, FileEvent, ParseStats, Task, UsageRecord

SCHEMA_VERSION = 3
DERIVED_TABLES = ("sessions", "records", "tasks", "file_events", "checks")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    session_id TEXT NOT NULL,
    project TEXT,
    cwd TEXT,
    version TEXT,
    signature TEXT NOT NULL,
    stats TEXT NOT NULL,
    coverage TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS records (
    source TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    input_total INTEGER NOT NULL,
    input_cache_read INTEGER NOT NULL,
    input_cache_write INTEGER NOT NULL,
    input_cache_write_1h INTEGER NOT NULL,
    input_new INTEGER NOT NULL,
    output_total INTEGER NOT NULL,
    output_logged INTEGER,
    output_source TEXT NOT NULL,
    output_reasoning INTEGER,
    output_exact INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    duration_ms INTEGER,
    session_id TEXT,
    project TEXT,
    agent_id TEXT,
    extra TEXT NOT NULL,
    PRIMARY KEY (source, turn_id)
);
CREATE INDEX IF NOT EXISTS records_ts ON records (ts);
CREATE INDEX IF NOT EXISTS records_task ON records (source, task_id);
CREATE INDEX IF NOT EXISTS records_session ON records (session_key);
CREATE TABLE IF NOT EXISTS tasks (
    source TEXT NOT NULL,
    task_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    started_at TEXT,
    project TEXT,
    cwd TEXT,
    family TEXT,
    command TEXT,
    prompt_chars INTEGER,
    PRIMARY KEY (source, task_id)
);
CREATE TABLE IF NOT EXISTS task_tags (
    source TEXT NOT NULL,
    task_id TEXT NOT NULL,
    family TEXT NOT NULL,
    PRIMARY KEY (source, task_id)
);
CREATE TABLE IF NOT EXISTS file_events (
    session_key TEXT NOT NULL,
    task_id TEXT NOT NULL,
    ts TEXT,
    path TEXT NOT NULL,
    action TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS file_events_session ON file_events (session_key);
CREATE TABLE IF NOT EXISTS checks (
    session_key TEXT NOT NULL,
    model TEXT NOT NULL,
    source_input INTEGER, source_cache_read INTEGER, source_cache_write INTEGER, source_output INTEGER,
    ours_input INTEGER, ours_cache_read INTEGER, ours_cache_write INTEGER,
    ours_output_logged INTEGER, ours_output INTEGER
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = self.db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row and row["value"] != str(SCHEMA_VERSION):
            # Everything but declared families is derived from the transcripts:
            # drop it and let the next ingest rebuild it.
            for t in DERIVED_TABLES:
                self.db.execute(f"DROP TABLE IF EXISTS {t}")
        self.db.executescript(SCHEMA)
        self.db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ ingest
    def signature_of(self, session_key: str) -> Optional[str]:
        row = self.db.execute(
            "SELECT signature FROM sessions WHERE session_key = ?", (session_key,)
        ).fetchone()
        return row["signature"] if row else None

    def replace_session(
        self,
        *,
        session_key: str,
        source: str,
        session_id: str,
        project: Optional[str],
        cwd: Optional[str],
        version: Optional[str],
        signature: str,
        stats: ParseStats,
        records: Iterable[UsageRecord],
        tasks: Iterable[Task],
        file_events: Iterable[FileEvent],
        checks: Iterable[CounterCheck] = (),
        coverage: Optional[dict] = None,
    ) -> None:
        db = self.db
        with db:
            for table in ("records", "tasks", "file_events", "checks"):
                db.execute(f"DELETE FROM {table} WHERE session_key = ?", (session_key,))
            db.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (session_key, source, session_id, project, cwd, version, signature,
                 json.dumps(asdict(stats)), json.dumps(coverage or {})),
            )
            # A resumed session can repeat calls already logged by its parent
            # session: (source, turn_id) is unique, so they are counted once.
            db.executemany(
                "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (r.source, r.turn_id, session_key, r.timestamp, r.model, r.input_total,
                     r.input_cache_read, r.input_cache_write, r.input_cache_write_1h,
                     r.input_new, r.output_total,
                     r.output_logged if r.output_logged is not None else r.output_total,
                     r.output_source, r.output_reasoning, int(r.output_exact),
                     r.task_id, r.trigger, r.duration_ms, r.session_id, r.project,
                     r.agent_id, json.dumps(r.extra))
                    for r in records
                ],
            )
            db.executemany(
                "INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (t.source, t.task_id, session_key, t.started_at, t.project, t.cwd,
                     t.family, t.command, t.prompt_chars)
                    for t in tasks
                ],
            )
            db.executemany(
                "INSERT INTO file_events VALUES (?,?,?,?,?)",
                [(session_key, e.task_id, e.timestamp, e.path, e.action) for e in file_events],
            )
            db.executemany(
                "INSERT INTO checks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(session_key, c.model, c.source_input, c.source_cache_read, c.source_cache_write,
                  c.source_output, c.ours_input, c.ours_cache_read, c.ours_cache_write,
                  c.ours_output_logged, c.ours_output) for c in checks],
            )

    def tag(self, source: str, task_prefix: str, family: str) -> list[str]:
        rows = self.db.execute(
            "SELECT task_id FROM tasks WHERE source = ? AND task_id LIKE ?",
            (source, task_prefix + "%"),
        ).fetchall()
        ids = [r["task_id"] for r in rows]
        if len(ids) == 1:
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO task_tags VALUES (?,?,?)", (source, ids[0], family)
                )
        return ids

    # ------------------------------------------------------------ queries
    def records(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        project: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM records WHERE 1=1"
        args: list = []
        if since:
            q += " AND ts >= ?"
            args.append(since)
        if until:
            q += " AND ts < ?"
            args.append(until)
        if project:
            q += " AND project LIKE ?"
            args.append(f"%{project}%")
        if session_key:
            q += " AND session_key = ?"
            args.append(session_key)
        return self.db.execute(q + " ORDER BY ts, rowid", args).fetchall()

    def has_records(self) -> bool:
        return self.db.execute("SELECT 1 FROM records LIMIT 1").fetchone() is not None

    def time_span(self) -> tuple[str, str]:
        row = self.db.execute("SELECT MIN(ts), MAX(ts) FROM records").fetchone()
        return (row[0] or "", row[1] or "")

    def tasks(self) -> dict[tuple[str, str], dict]:
        out = {}
        for r in self.db.execute(
            "SELECT t.*, g.family AS declared FROM tasks t "
            "LEFT JOIN task_tags g ON g.source = t.source AND g.task_id = t.task_id"
        ):
            d = dict(r)
            d["family_source"] = "declared" if d["declared"] else "guessed"
            d["family"] = d["declared"] or d["family"]
            out[(d["source"], d["task_id"])] = d
        return out

    def sessions(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM sessions").fetchall()

    def file_events(self, session_key: Optional[str] = None) -> list[sqlite3.Row]:
        if session_key:
            return self.db.execute(
                "SELECT * FROM file_events WHERE session_key = ? ORDER BY ts, rowid", (session_key,)
            ).fetchall()
        return self.db.execute("SELECT * FROM file_events ORDER BY ts, rowid").fetchall()

    def checks(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT c.*, s.session_id, s.project, s.version, s.coverage FROM checks c "
            "JOIN sessions s ON s.session_key = c.session_key"
        ).fetchall()

    def parse_stats(self) -> ParseStats:
        total = ParseStats()
        for r in self.db.execute("SELECT stats FROM sessions"):
            total.merge(ParseStats(**json.loads(r["stats"])))
        return total
