"""Runs the collectors and loads their output into the store (incremental)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .collectors import claude_code
from .record import ParseStats
from .store import Store


@dataclass
class IngestResult:
    sessions_seen: int = 0
    sessions_parsed: int = 0
    stats: ParseStats | None = None  # for the sessions parsed in this run


def _signature(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def ingest_claude_code(store: Store, root: Path, force: bool = False,
                       sessions: "list[claude_code.SessionFiles] | None" = None) -> IngestResult:
    """sessions: only these (the live hook reads just the current session)."""
    res = IngestResult(stats=ParseStats())
    for sf in sessions if sessions is not None else claude_code.discover(root):
        res.sessions_seen += 1
        key = f"{claude_code.SOURCE}:{sf.project_dir}/{sf.session_id}"
        files = sf.all_files()
        files += [p.with_name(p.stem + ".meta.json") for p in sf.subagents]
        sig = _signature(files)
        if not force and store.signature_of(key) == sig:
            continue
        out = claude_code.parse_session(sf)
        res.sessions_parsed += 1
        res.stats.merge(out.stats)  # type: ignore[union-attr]
        cwd = next((t.cwd for t in out.tasks if t.cwd), None)
        project = next((r.project for r in out.records if r.project), None) or (
            claude_code.project_name(cwd) if cwd else sf.project_dir
        )
        store.replace_session(
            session_key=key,
            source=claude_code.SOURCE,
            session_id=sf.session_id,
            project=project,
            cwd=cwd,
            version=out.version,
            signature=sig,
            stats=out.stats,
            records=out.records,
            tasks=out.tasks,
            file_events=out.file_events,
            checks=out.checks,
        )
    return res
