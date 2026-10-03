"""Collector for Claude Code's local session transcripts.

Claude Code writes one JSONL file per session under
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``, and one file per
sub-agent under ``<session-id>/subagents/agent-<id>.jsonl`` (older versions
kept sub-agent records inline, flagged ``isSidechain``).

The format is not a public contract. This parser is tolerant: anything it
does not understand is counted and skipped, never fatal. Verified against
Claude Code 2.1.288 (see README).

Things the format does that matter for counting:

* One API call is written as several ``assistant`` records (one per content
  block: thinking, text, each tool_use), and every one of them repeats the
  same ``usage``. Summing records double- or triple-counts. We group by
  ``message.id``.
* ``promptId`` on user records ties every call back to the message you typed,
  sub-agents included. That is the task id.
* Sub-agent records are written when streaming starts, so their
  ``output_tokens`` is often a placeholder (``stop_reason`` is null). We know
  because Claude Code's own counter (``cost-state`` records) disagrees with
  them while agreeing to the token with every main-thread call. The final
  call's real usage is in the parent's tool result: we use it, keep the
  logged value next to it, and mark the record (``output_source``).
  Earlier sub-agent calls stay as logged, flagged ``output_exact = False``.
* ``cost-state`` records are checked against our sums (``CounterCheck``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from ..classify import guess_family
from ..record import CounterCheck, FileEvent, ParseStats, Task, UsageRecord

SOURCE = "claude-code"
VERIFIED_VERSION = "2.1.288"

# Record types we know and skip on purpose (they carry no usage).
KNOWN_IGNORED = frozenset(
    {
        "attachment",
        "system",
        "queue-operation",
        "last-prompt",
        "mode",
        "cost-state",
        "atis-latch",
        "summary",
        "file-history-snapshot",
        "custom-title",
        "progress",
    }
)

READ_TOOLS = frozenset({"Read", "NotebookRead"})
WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
SEARCH_TOOLS = frozenset({"Grep", "Glob", "LS"})
SUBAGENT_TOOLS = frozenset({"Agent", "Task"})

_COMMAND_RE = re.compile(r"<command-name>/?([\w:.-]+)</command-name>")
_MAX_CHAIN = 500


@dataclass
class SessionFiles:
    session_id: str
    main: Optional[Path]
    subagents: list[Path] = field(default_factory=list)
    project_dir: str = ""

    def all_files(self) -> list[Path]:
        return ([self.main] if self.main else []) + self.subagents


@dataclass
class SessionResult:
    session_id: str
    records: list[UsageRecord]
    tasks: list[Task]
    file_events: list[FileEvent]
    stats: ParseStats
    version: Optional[str] = None
    checks: list[CounterCheck] = field(default_factory=list)


def discover(root: Path) -> list[SessionFiles]:
    """Find sessions under Claude Code's projects directory."""
    sessions: dict[tuple[str, str], SessionFiles] = {}
    if not root.is_dir():
        return []
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        for main in sorted(project.glob("*.jsonl")):
            key = (project.name, main.stem)
            sessions[key] = SessionFiles(main.stem, main, project_dir=project.name)
        for sub in sorted(project.glob("*/subagents/*.jsonl")):
            sid = sub.parent.parent.name
            key = (project.name, sid)
            sf = sessions.setdefault(key, SessionFiles(sid, None, project_dir=project.name))
            sf.subagents.append(sub)
    return list(sessions.values())


# --------------------------------------------------------------------------
# helpers


def _int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int) and v >= 0:
        return v
    return None


def _parse_ts(s: Any) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    n += len(b["text"])
                elif "content" in b:
                    n += _text_len(b["content"])
            elif isinstance(b, str):
                n += len(b)
        return n
    return 0


def _read_jsonl(path: Path, stats: ParseStats) -> Iterator[dict]:
    stats.files += 1
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        stats.bump(stats.unreadable, "file_unreadable")
        return
    with fh:
        for line in fh:
            if not line.strip():
                continue
            stats.lines += 1
            try:
                obj = json.loads(line)
            except ValueError:
                stats.bump(stats.unreadable, "bad_json")
                continue
            if not isinstance(obj, dict):
                stats.bump(stats.unreadable, "not_an_object")
                continue
            yield obj


@dataclass
class _Node:
    type: str
    parent: Optional[str]
    prompt_id: Optional[str]
    ts: Optional[str]
    kind: str = ""  # prompt | meta | tool_result | assistant | other
    tool_result_ids: tuple[str, ...] = ()
    chars: int = 0
    group: Optional[str] = None


@dataclass
class _Group:
    key: str
    thread: str  # "main" or agent id
    first: dict
    last: dict
    usage: dict
    model: str
    stop_reason: Optional[str]
    tool_uses: list[dict] = field(default_factory=list)
    order: int = 0
    main_line: int = -1  # line in the main file, -1 for a sub-agent file


@dataclass
class _ToolUse:
    id: str
    name: str
    input: dict
    group: str
    thread: str
    ts: Optional[str]
    reread: bool = False


# --------------------------------------------------------------------------


def parse_session(sf: SessionFiles) -> SessionResult:
    stats = ParseStats()
    nodes: dict[str, _Node] = {}
    groups: dict[str, _Group] = {}
    tool_uses: dict[str, _ToolUse] = {}
    prompts: list[tuple[dict, _Node, Optional[str]]] = []
    agent_final_usage: dict[str, dict] = {}  # Agent tool_use id -> final usage
    agent_meta: dict[str, dict] = {}  # agent id -> meta.json
    compaction_after: set[str] = set()  # uuids of compaction boundaries
    version: Optional[str] = None
    cwd: Optional[str] = None
    order = 0
    checkpoints: list[tuple[int, dict]] = []  # (main-file line, cost-state modelUsage)

    def handle(obj: dict, thread_hint: Optional[str], line: int = -1) -> None:
        nonlocal order, version, cwd
        rtype = obj.get("type")
        if not isinstance(rtype, str):
            stats.bump(stats.unreadable, "no_type")
            return
        v = obj.get("version")
        if isinstance(v, str):
            stats.bump(stats.versions, v)
            version = v
        if cwd is None and isinstance(obj.get("cwd"), str):
            cwd = obj["cwd"]
        uuid = obj.get("uuid") if isinstance(obj.get("uuid"), str) else None
        parent = obj.get("parentUuid") if isinstance(obj.get("parentUuid"), str) else None
        pid = obj.get("promptId") if isinstance(obj.get("promptId"), str) else None
        ts = obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None
        thread = thread_hint or (
            obj.get("agentId") if obj.get("isSidechain") and isinstance(obj.get("agentId"), str)
            else ("sidechain" if obj.get("isSidechain") else "main")
        )

        if rtype == "user":
            msg = obj.get("message")
            if not isinstance(msg, dict):
                stats.bump(stats.unreadable, "user_without_message")
                return
            content = msg.get("content")
            node = _Node("user", parent, pid, ts)
            results = [
                b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
            ] if isinstance(content, list) else []
            if results:
                node.kind = "tool_result"
                node.tool_result_ids = tuple(
                    b["tool_use_id"] for b in results if isinstance(b.get("tool_use_id"), str)
                )
                node.chars = sum(_text_len(b.get("content")) for b in results)
                tur = obj.get("toolUseResult")
                if isinstance(tur, dict) and isinstance(tur.get("usage"), dict) and tur.get("agentId"):
                    for tid in node.tool_result_ids:
                        agent_final_usage[tid] = tur["usage"]
            elif isinstance(content, (str, list)):
                node.kind = "meta" if obj.get("isMeta") or obj.get("isCompactSummary") else "prompt"
                node.chars = _text_len(content)
                if node.kind == "prompt" and thread == "main":
                    text = content if isinstance(content, str) else " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                    m = _COMMAND_RE.search(text)
                    prompts.append((obj, node, m.group(1) if m else None))
            else:
                stats.bump(stats.unreadable, "user_unexpected_content")
                return
            if uuid:
                nodes[uuid] = node
            stats.used += 1
            return

        if rtype == "assistant":
            msg = obj.get("message")
            if not isinstance(msg, dict):
                stats.bump(stats.unreadable, "assistant_without_message")
                return
            model = msg.get("model") if isinstance(msg.get("model"), str) else "unknown"
            usage = msg.get("usage")
            if model == "<synthetic>":
                stats.bump(stats.ignored, "synthetic_message")
                if uuid:
                    nodes[uuid] = _Node("assistant", parent, pid, ts, kind="other")
                return
            if not isinstance(usage, dict) or any(
                k in usage and _int(usage[k]) is None
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                          "cache_creation_input_tokens")
            ) or _int(usage.get("input_tokens")) is None or _int(usage.get("output_tokens")) is None:
                stats.bump(stats.unreadable, "assistant_bad_usage")
                if uuid:
                    nodes[uuid] = _Node("assistant", parent, pid, ts, kind="other")
                return
            key = msg.get("id") if isinstance(msg.get("id"), str) else (
                obj.get("requestId") if isinstance(obj.get("requestId"), str) else uuid
            )
            if key is None:
                stats.bump(stats.unreadable, "assistant_without_id")
                return
            g = groups.get(key)
            if g is None:
                order += 1
                g = _Group(key, thread, obj, obj, usage, model, msg.get("stop_reason"), order=order,
                           main_line=line)
                groups[key] = g
            else:
                g.last = obj
                # keep the most complete usage snapshot
                if (_int(usage.get("output_tokens")) or 0) >= (_int(g.usage.get("output_tokens")) or 0):
                    g.usage = usage
                if msg.get("stop_reason"):
                    g.stop_reason = msg["stop_reason"]
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and isinstance(b.get("id"), str):
                        tu = _ToolUse(
                            b["id"], str(b.get("name", "?")),
                            b.get("input") if isinstance(b.get("input"), dict) else {},
                            key, thread, ts,
                        )
                        tool_uses[tu.id] = tu
                        g.tool_uses.append({"id": tu.id, "name": tu.name})
            if uuid:
                nodes[uuid] = _Node("assistant", parent, pid, ts, kind="assistant", group=key)
            stats.used += 1
            return

        if uuid:
            nodes[uuid] = _Node(rtype, parent, pid, ts, kind="other")
        if rtype == "cost-state" and line >= 0 and isinstance(obj.get("modelUsage"), dict):
            checkpoints.append((line, obj["modelUsage"]))
        if rtype in KNOWN_IGNORED:
            stats.bump(stats.ignored, rtype)
            if rtype == "system" and "compact" in str(obj.get("subtype", "")) and uuid:
                compaction_after.add(uuid)
        else:
            stats.bump(stats.unknown_types, rtype)

    if sf.main:
        for line, obj in enumerate(_read_jsonl(sf.main, stats)):
            handle(obj, None, line)
    for sub in sf.subagents:
        agent_id = sub.stem[len("agent-"):] if sub.stem.startswith("agent-") else sub.stem
        meta_path = sub.with_name(sub.stem + ".meta.json")
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(meta, dict):
                    agent_meta[agent_id] = meta
            except (OSError, ValueError):
                stats.bump(stats.unreadable, "bad_agent_meta")
        for obj in _read_jsonl(sub, stats):
            handle(obj, agent_id)

    # ---------------------------------------------------------------- tasks
    def task_of(uuid: Optional[str]) -> Optional[str]:
        seen = 0
        while uuid and seen < _MAX_CHAIN:
            n = nodes.get(uuid)
            if n is None:
                return None
            if n.prompt_id:
                return n.prompt_id
            uuid = n.parent
            seen += 1
        return None

    group_task: dict[str, str] = {}
    for g in sorted(groups.values(), key=lambda g: g.order):
        t = task_of(g.first.get("uuid"))
        if t is None and g.thread not in ("main", "sidechain"):
            meta = agent_meta.get(g.thread, {})
            parent_tu = tool_uses.get(meta.get("toolUseId", ""))
            if parent_tu:
                t = group_task.get(parent_tu.group)
        group_task[g.key] = t or f"{sf.session_id}:untracked"

    # ------------------------------------------- file events and re-reads
    file_events: list[FileEvent] = []
    seen_reads: dict[str, set[str]] = {}
    for g in sorted(groups.values(), key=lambda g: g.order):
        for ref in g.tool_uses:
            tu = tool_uses[ref["id"]]
            path = tu.input.get("file_path") or tu.input.get("notebook_path")
            action = None
            if tu.name in READ_TOOLS and isinstance(path, str):
                action = "read"
                reads = seen_reads.setdefault(tu.thread, set())
                tu.reread = path in reads
                reads.add(path)
            elif tu.name in WRITE_TOOLS and isinstance(path, str):
                action = "write"
            elif tu.name in SEARCH_TOOLS and isinstance(tu.input.get("path"), str):
                action, path = "search", tu.input["path"]
            if action:
                file_events.append(FileEvent(
                    sf.session_id, group_task[g.key],
                    tu.ts or g.first.get("timestamp", ""), path, action,  # type: ignore[arg-type]
                ))

    # ----------------------------------------------- sub-agent final usage
    patched: dict[str, dict] = {}
    for agent_id, meta in agent_meta.items():
        final = agent_final_usage.get(meta.get("toolUseId", ""))
        if not final:
            continue
        mine = [g for g in groups.values() if g.thread == agent_id]
        if not mine:
            continue
        last = max(mine, key=lambda g: g.order)
        same_input = all(
            (_int(last.usage.get(k)) or 0) == (_int(final.get(k)) or 0)
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        if same_input:
            patched[last.key] = final

    # -------------------------------------------------------------- records
    records: list[UsageRecord] = []
    project = Path(cwd).name if cwd else sf.project_dir
    for g in sorted(groups.values(), key=lambda g: g.order):
        usage = patched.get(g.key, g.usage)
        out_logged = _int(g.usage.get("output_tokens")) or 0
        new = _int(usage.get("input_tokens")) or 0
        cread = _int(usage.get("cache_read_input_tokens")) or 0
        cwrite = _int(usage.get("cache_creation_input_tokens")) or 0
        cc = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
        cwrite_1h = min(_int(cc.get("ephemeral_1h_input_tokens")) or 0, cwrite)
        out = _int(usage.get("output_tokens")) or 0
        otd = usage.get("output_tokens_details")
        reasoning = _int(otd.get("thinking_tokens")) if isinstance(otd, dict) else None

        is_sub = g.thread != "main"
        trigger, trig_ids, added_chars, trig_node = _trigger(g, nodes, is_sub)
        trig_tools = [tool_uses[i].name for i in trig_ids if i in tool_uses]
        reread = bool(trig_ids) and all(
            i in tool_uses and tool_uses[i].reread for i in trig_ids
        )
        start = _parse_ts(trig_node.ts) if trig_node else None
        end = _parse_ts(g.last.get("timestamp"))
        duration = int((end - start).total_seconds() * 1000) if start and end and end >= start else None
        after_compaction = _follows_compaction(g, nodes, compaction_after)

        records.append(UsageRecord(
            timestamp=str(g.first.get("timestamp", "")),
            source=SOURCE,
            model=g.model,
            input_total=new + cread + cwrite,
            input_cache_read=cread,
            input_cache_write=cwrite,
            input_new=new,
            output_total=out,
            output_reasoning=reasoning,
            turn_id=g.key,
            task_id=group_task[g.key],
            trigger="subagent" if is_sub else trigger,
            duration_ms=duration,
            session_id=sf.session_id,
            project=project,
            agent_id=g.thread if is_sub else None,
            input_cache_write_1h=cwrite_1h,
            output_source="subagent_result" if g.key in patched else "logged",
            output_logged=out_logged,
            output_exact=g.key in patched or g.stop_reason is not None,
            extra={
                "tools": [t["name"] for t in g.tool_uses],
                "trigger_tools": trig_tools,
                "added_chars": added_chars,
                "reread": reread,
                "after_compaction": after_compaction,
            },
        ))

    # ---------------------------------------------------------------- tasks
    tasks: dict[str, Task] = {}
    for obj, node, command in prompts:
        tid = node.prompt_id or (obj.get("uuid") or "")
        if tid in tasks:
            tasks[tid].prompt_chars += node.chars
            continue
        tasks[tid] = Task(
            task_id=tid, session_id=sf.session_id, source=SOURCE,
            started_at=node.ts or "", project=project, cwd=cwd,
            command=command, prompt_chars=node.chars,
        )
    for r in records:
        if r.task_id not in tasks:
            tasks[r.task_id] = Task(r.task_id, sf.session_id, SOURCE, r.timestamp, project, cwd)
    for t in tasks.values():
        first_tools = []
        for g in sorted(groups.values(), key=lambda g: g.order):
            if group_task[g.key] == t.task_id and g.thread == "main":
                first_tools.extend(tool_uses[ref["id"]] for ref in g.tool_uses)
            if len(first_tools) >= 5:
                break
        t.family = guess_family(t.command, [(tu.name, tu.input) for tu in first_tools[:5]])

    checks = _counter_checks(checkpoints, groups, records, tool_uses, agent_meta)
    return SessionResult(sf.session_id, records, list(tasks.values()), file_events, stats, version, checks)


def _counter_checks(checkpoints, groups, records, tool_uses, agent_meta) -> list[CounterCheck]:
    """Compare our per-model sums with Claude Code's own counter.

    Uses the last ``cost-state`` record of the main file and only the calls
    made before it: main-thread calls above that line, sub-agent calls whose
    launching tool call is above it.
    """
    if not checkpoints:
        return []
    pos, usage = checkpoints[-1]

    def line_of(g: _Group) -> int:
        if g.main_line >= 0:
            return g.main_line
        tu = tool_uses.get(agent_meta.get(g.thread, {}).get("toolUseId", ""))
        return groups[tu.group].main_line if tu and tu.group in groups else 1 << 60

    covered = {g.key for g in groups.values() if line_of(g) < pos}
    sums: dict[str, list[int]] = {}
    for r in records:
        if r.turn_id in covered:
            acc = sums.setdefault(r.model, [0, 0, 0, 0, 0])
            acc[0] += r.input_new
            acc[1] += r.input_cache_read
            acc[2] += r.input_cache_write
            acc[3] += r.output_logged if r.output_logged is not None else r.output_total
            acc[4] += r.output_total
    checks = []
    for model, u in usage.items():
        if not isinstance(u, dict):
            continue
        ours = sums.get(model, [0, 0, 0, 0, 0])
        checks.append(CounterCheck(
            model,
            _int(u.get("inputTokens")) or 0, _int(u.get("cacheReadInputTokens")) or 0,
            _int(u.get("cacheCreationInputTokens")) or 0, _int(u.get("outputTokens")) or 0,
            *ours,
        ))
    return checks


def _trigger(g: _Group, nodes: dict[str, _Node], is_sub: bool):
    """What made this call happen: walk up from the call's first record."""
    uuid = g.first.get("parentUuid")
    ids: list[str] = []
    chars = 0
    trig_node: Optional[_Node] = None
    steps = 0
    while isinstance(uuid, str) and steps < _MAX_CHAIN:
        steps += 1
        n = nodes.get(uuid)
        if n is None:
            break
        if n.kind == "tool_result":
            trig_node = trig_node or n
            ids.extend(n.tool_result_ids)
            chars += n.chars
        elif n.kind == "meta":
            # injected context (skill body, command caveat): transparent, but it is input
            chars += n.chars
        elif n.kind == "prompt":
            if ids:
                break
            return "user_turn", [], chars + n.chars, n
        elif n.kind == "assistant":
            if n.group != g.key:
                break
        uuid = n.parent
    if ids:
        return "tool_call", ids, chars, trig_node
    return ("tool_call" if is_sub else "user_turn"), [], 0, trig_node


def _follows_compaction(g: _Group, nodes: dict[str, _Node], boundaries: set[str]) -> bool:
    uuid = g.first.get("parentUuid")
    steps = 0
    while isinstance(uuid, str) and steps < 50:
        if uuid in boundaries:
            return True
        n = nodes.get(uuid)
        if n is None or n.kind == "assistant":
            return False
        uuid = n.parent
        steps += 1
    return False


def collect(root: Path, sessions: Optional[Iterable[SessionFiles]] = None) -> Iterator[SessionResult]:
    for sf in sessions if sessions is not None else discover(root):
        yield parse_session(sf)
