"""Invented Claude Code transcripts for tests.

Shaped like Claude Code 2.1.288's JSONL (one record per content block, usage
repeated on each, promptId on user records, sub-agents in their own files),
but every project, path and message here is made up.
"""

from __future__ import annotations

import itertools
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

VERSION = "2.1.288"
_ids = itertools.count(1)


def _id(prefix: str) -> str:
    return f"{prefix}{next(_ids):024d}"


class FakeSession:
    def __init__(
        self,
        projects_root: Path,
        project: str = "acme-webshop",
        session_id: Optional[str] = None,
        start: datetime = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc),
        model: str = "claude-opus-5-5",
    ):
        self.root = projects_root
        self.project = project
        self.cwd = f"/srv/demo/{project}"
        self.session_id = session_id or str(uuid.uuid4())
        self.dir = projects_root / ("-srv-demo-" + project)
        self.now = start
        self.model = model
        self.lines: list[str] = []
        self.parent: Optional[str] = None
        self.prompt_id: Optional[str] = None
        self.subagents: dict[str, tuple[list[str], dict]] = {}
        self.last_call_at = start

    # -------------------------------------------------------------- basics
    def tick(self, seconds: float = 5) -> None:
        self.now += timedelta(seconds=seconds)

    def ts(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{self.now.microsecond // 1000:03d}Z"

    def _common(self, **kw) -> dict:
        d = {
            "parentUuid": self.parent,
            "isSidechain": False,
            "uuid": _id("u-"),
            "timestamp": self.ts(),
            "userType": "external",
            "cwd": self.cwd,
            "sessionId": self.session_id,
            "version": VERSION,
            "gitBranch": "main",
        }
        d.update(kw)
        return d

    def raw(self, line: str) -> None:
        self.lines.append(line)

    def emit(self, rec: dict, lines: Optional[list[str]] = None) -> dict:
        (self.lines if lines is None else lines).append(json.dumps(rec))
        return rec

    def attachment(self) -> None:
        rec = self._common(type="attachment", attachment={"type": "date", "date": "2026-09-28"})
        self.emit(rec)
        self.parent = rec["uuid"]

    # -------------------------------------------------------------- turns
    def prompt(self, text: str, command: Optional[str] = None) -> str:
        self.tick(30)
        self.prompt_id = str(uuid.uuid4())
        self.emit({"type": "queue-operation", "operation": "enqueue", "timestamp": self.ts(),
                   "sessionId": self.session_id, "content": text})
        content = f"<command-name>/{command}</command-name> {text}" if command else text
        rec = self._common(
            type="user", promptId=self.prompt_id, message={"role": "user", "content": content},
            origin={"kind": "human"}, turnOrigin="human",
        )
        self.emit(rec)
        self.parent = rec["uuid"]
        self.attachment()
        return self.prompt_id

    def call(
        self,
        *,
        new: int = 2,
        read: int = 0,
        write: int = 0,
        out: int = 100,
        thinking: int = 40,
        tools: tuple[tuple[str, dict], ...] = (),
        result_chars: int = 500,
        ttl: str = "5m",
        model: Optional[str] = None,
        seconds: float = 4,
        lines: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        stop_reason: Optional[str] = "auto",
    ) -> list[str]:
        """One API call: thinking block + one record per tool_use, same usage on each."""
        self.tick(seconds)
        msg_id = _id("msg_")
        usage = {
            "input_tokens": new,
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "output_tokens": out,
            "output_tokens_details": {"thinking_tokens": thinking},
            "cache_creation": {
                "ephemeral_1h_input_tokens": write if ttl == "1h" else 0,
                "ephemeral_5m_input_tokens": write if ttl == "5m" else 0,
            },
            "service_tier": "standard",
        }
        if stop_reason == "auto":
            stop_reason = "tool_use" if tools else "end_turn"
        blocks: list[dict] = [{"type": "thinking", "thinking": "", "signature": "x"}]
        tool_ids = []
        for name, inp in tools:
            tid = _id("toolu_")
            tool_ids.append(tid)
            blocks.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
        if not tools:
            blocks.append({"type": "text", "text": "Done."})
        for b in blocks:
            extra = {"isSidechain": True, "agentId": agent_id} if agent_id else {}
            rec = self._common(
                type="assistant", requestId=_id("req_"),
                message={"model": model or self.model, "id": msg_id, "type": "message",
                         "role": "assistant", "content": [b], "stop_reason": stop_reason,
                         "usage": usage},
                **extra,
            )
            self.emit(rec, lines)
            self.parent = rec["uuid"]
        if lines is None:
            self.last_call_at = self.now
        self.tick(1)
        for tid in tool_ids:
            extra = {"isSidechain": True, "agentId": agent_id} if agent_id else {}
            rec = self._common(
                type="user", promptId=self.prompt_id,
                message={"role": "user", "content": [
                    {"tool_use_id": tid, "type": "tool_result", "content": "x" * result_chars}]},
                toolUseResult={"stdout": "x" * result_chars}, **extra,
            )
            self.emit(rec, lines)
            self.parent = rec["uuid"]
        return tool_ids

    def subagent(self, calls: list[dict], final_out: int) -> str:
        """An Agent tool call whose sub-agent makes `calls`; logged mid-stream (stop_reason null)."""
        tid = self.call(tools=(("Agent", {"description": "look around", "prompt": "..."}),),
                        new=2, read=1000, write=200, out=80, result_chars=0)[0]
        # the call() above already wrote a placeholder tool_result; drop it, the real one comes last
        self.lines.pop()
        agent_id = _id("a")[-17:]
        lines: list[str] = []
        saved_parent, saved_now = self.parent, self.now
        self.parent = None
        first = self._common(type="user", isSidechain=True, agentId=agent_id, promptId=self.prompt_id,
                             message={"role": "user", "content": "look around"})
        self.emit(first, lines)
        self.parent = first["uuid"]
        last_usage = None
        for c in calls:
            self.call(lines=lines, agent_id=agent_id, stop_reason=None, **c)
            last_usage = {"input_tokens": c.get("new", 2), "cache_read_input_tokens": c.get("read", 0),
                          "cache_creation_input_tokens": c.get("write", 0), "output_tokens": final_out}
        self.subagents[agent_id] = (lines, {"agentType": "general-purpose", "toolUseId": tid})
        self.parent = saved_parent
        self.now = max(saved_now, self.now)
        rec = self._common(
            type="user", promptId=self.prompt_id,
            message={"role": "user", "content": [
                {"tool_use_id": tid, "type": "tool_result", "content": [{"type": "text", "text": "found it"}]}]},
            toolUseResult={"status": "completed", "agentId": agent_id, "usage": last_usage},
        )
        self.emit(rec)
        self.parent = rec["uuid"]
        return agent_id

    def cost_state(self, model_usage: dict) -> None:
        """Claude Code's own running counter, as it writes it at the end of a turn."""
        self.emit({"type": "cost-state", "sessionId": self.session_id, "totalCostUSD": 0.0,
                   "modelUsage": model_usage})

    # -------------------------------------------------------------- output
    def write(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        main = self.dir / f"{self.session_id}.jsonl"
        main.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        for agent_id, (lines, meta) in self.subagents.items():
            d = self.dir / self.session_id / "subagents"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"agent-{agent_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (d / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return main


def read(path: str) -> tuple[tuple[str, dict], ...]:
    return (("Read", {"file_path": path}),)


def edit(path: str) -> tuple[tuple[str, dict], ...]:
    return (("Edit", {"file_path": path, "old_string": "a", "new_string": "b"}),)


def bash(cmd: str) -> tuple[tuple[str, dict], ...]:
    return (("Bash", {"command": cmd}),)
