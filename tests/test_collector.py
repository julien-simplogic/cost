from __future__ import annotations

import json

import pytest

from fake_transcripts import FakeSession, bash, edit, read
from tokentrail.collectors import claude_code as cc
from tokentrail.record import UsageRecord


def _parse(projects):
    [sf] = cc.discover(projects)
    return cc.parse_session(sf)


def test_split_blocks_are_counted_once(env):
    s = FakeSession(env.projects)
    s.prompt("hello")
    # one API call written as 1 thinking + 3 tool_use records, all carrying the same usage
    s.call(new=5, read=1000, write=2000, out=300, thinking=120,
           tools=read("a.py") + read("b.py") + read("c.py"))
    s.call(read=3000, write=100, out=50)
    s.write()
    res = _parse(env.projects)
    assert len(res.records) == 2
    first = res.records[0]
    assert (first.input_new, first.input_cache_read, first.input_cache_write) == (5, 1000, 2000)
    assert first.input_total == 3005
    assert first.output_total == 300 and first.output_reasoning == 120
    assert first.extra["tools"] == ["Read", "Read", "Read"]


def test_task_id_turns_and_triggers(env):
    s = FakeSession(env.projects)
    p1 = s.prompt("first")
    s.call(tools=bash("ls"))
    s.call(tools=bash("pwd"))
    s.call()
    p2 = s.prompt("second")
    s.call()
    s.write()
    res = _parse(env.projects)
    assert [r.task_id for r in res.records] == [p1, p1, p1, p2]
    assert [r.trigger for r in res.records] == ["user_turn", "tool_call", "tool_call", "user_turn"]
    assert res.records[1].extra["trigger_tools"] == ["Bash"]
    assert {t.task_id for t in res.tasks} == {p1, p2}
    assert all(r.duration_ms and r.duration_ms > 0 for r in res.records)


def test_subagents_belong_to_the_parent_task_and_output_is_patched(env):
    s = FakeSession(env.projects)
    p = s.prompt("look into it")
    s.subagent([{"new": 4, "write": 8000, "out": 3, "tools": bash("ls")},
                {"new": 2, "read": 8000, "write": 500, "out": 3}], final_out=420)
    s.call(read=5000, write=100)
    s.write()
    res = _parse(env.projects)
    subs = [r for r in res.records if r.trigger == "subagent"]
    assert len(subs) == 2
    assert all(r.task_id == p and r.agent_id for r in subs)
    first, last = subs
    assert first.output_exact is False and first.output_total == 3  # logged mid-stream
    assert last.output_exact is True and last.output_total == 420  # patched from the parent's result
    assert last.input_total == 8502  # input untouched


def test_inline_sidechain_records_count_as_subagent(env):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    lines = []
    s.call(lines=lines, agent_id="old0000000000000a", stop_reason="end_turn")
    for line in lines:  # older versions kept sub-agent records in the main file
        s.raw(line)
    s.write()
    res = _parse(env.projects)
    assert [r.trigger for r in res.records] == ["user_turn", "subagent"]


def test_tolerant_parsing_counts_what_it_cannot_read(env):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call()
    s.raw("{not json")
    s.raw("[1, 2, 3]")
    s.raw(json.dumps({"no": "type"}))
    s.raw(json.dumps({"type": "hologram", "uuid": "h1"}))
    s.raw(json.dumps({"type": "assistant", "message": {"id": "m", "model": "claude-opus-5-5",
                                                       "usage": {"input_tokens": "lots"}}}))
    s.raw(json.dumps({"type": "assistant", "uuid": "s1", "message": {
        "id": "m2", "model": "<synthetic>", "usage": {"input_tokens": 0, "output_tokens": 0}}}))
    s.call()
    s.write()
    res = _parse(env.projects)
    assert len(res.records) == 2
    st = res.stats
    assert st.unreadable == {"bad_json": 1, "not_an_object": 1, "no_type": 1, "assistant_bad_usage": 1}
    assert st.unknown_types == {"hologram": 1}
    assert st.not_understood == 5
    assert st.ignored["synthetic_message"] == 1
    assert st.versions == {"2.1.288": st.versions["2.1.288"]}


def test_empty_and_missing_files_are_not_fatal(env):
    (env.projects / "-srv-demo-empty").mkdir()
    (env.projects / "-srv-demo-empty" / "deadbeef.jsonl").write_text("")
    [sf] = cc.discover(env.projects)
    res = cc.parse_session(sf)
    assert res.records == [] and res.stats.not_understood == 0
    assert cc.discover(env.projects / "nope") == []


def test_rereads_are_flagged(env):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(tools=read("src/a.py"))
    s.call(tools=read("src/b.py"))
    s.call(tools=read("src/a.py"))  # again
    s.call()
    s.write()
    res = _parse(env.projects)
    assert [r.extra["reread"] for r in res.records] == [False, False, False, True]
    assert [e.action for e in res.file_events] == ["read", "read", "read"]


@pytest.mark.parametrize("command,tools,family", [
    ("code-review", (), "review"),
    (None, edit("a.py"), "refactor"),
    (None, read("a") + read("b") + read("c"), "review"),
    (None, bash("hyperfine ./run"), "measure"),
    (None, bash("ls"), "question"),
    (None, (), "question"),
])
def test_family_guess(env, command, tools, family):
    s = FakeSession(env.projects)
    s.prompt("go", command=command)
    s.call(tools=tools)
    s.call()
    s.write()
    [task] = _parse(env.projects).tasks
    assert task.family == family
    assert task.command == command


def test_ttl_and_record_contract(env):
    s = FakeSession(env.projects)
    s.prompt("x")
    s.call(new=1, write=5000, ttl="1h")
    s.write()
    [r] = _parse(env.projects).records
    assert r.input_cache_write_1h == 5000 and r.source == "claude-code"
    with pytest.raises(ValueError):
        UsageRecord("t", "s", "m", 10, 1, 1, 1, 0, None, "turn", "task", "user_turn", None)
    with pytest.raises(ValueError):
        UsageRecord("t", "s", "m", 3, 1, 1, 1, 0, None, "turn", "task", "whim", None)  # type: ignore[arg-type]
