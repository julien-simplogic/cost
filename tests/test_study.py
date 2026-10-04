"""The read-only study script runs on invented transcripts and says what it should."""

from __future__ import annotations

import importlib.util
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fake_transcripts import FakeSession, bash, edit, read
from tokentrail.cli import main as cli

T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "study.py"


def _study():
    spec = importlib.util.spec_from_file_location("study", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["study"] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


PROMPTS = [
    ("fix the failing test in cart.py", 6),
    ("Traceback (most recent call last):\n  File \"x.py\"\nValueError: bad", 9),
    ("what does utils.py do?", 2),
    ("refactor payments.py and orders.py into a package", 14),
    ("continue", 4),
    ("ok", 3),
]


def history(projects: Path, sessions: int = 12, seed: int = 7) -> None:
    rng = random.Random(seed)
    for k in range(sessions):
        s = FakeSession(projects, project=f"proj{k % 3}", start=T0 + timedelta(days=k))
        ctx = 20_000
        for i in range(rng.randint(5, 9)):
            text, base = PROMPTS[0 if i == 0 and k % 2 else rng.randrange(len(PROMPTS))]
            s.prompt(text)
            n = max(1, base + rng.randint(-2, 3))
            s.call(new=2, read=ctx, write=800, tools=edit("x.py") if n > 1 else ())
            for _ in range(n - 1):
                s.call(read=ctx, write=600, out=200, tools=rng.choice([bash("pytest"), read("a.py"), edit("b.py")]))
                ctx += 600
            s.tick(rng.choice([40, 200, 900]))
        s.write()


@pytest.fixture
def study(env):
    history(env.projects)
    cli(["ingest"])
    return _study()


def test_step1_counts_and_merges_follow_ups(study, capsys):
    tasks = study.load()
    fu = [t for t in tasks if t.prompt and t.prompt.followup]
    assert fu, "the fixture has follow-ups"
    merged, folded = study.merge_followups(tasks)
    firsts = {ts[0].task_id for ts in study.by_session(tasks).values()}
    assert folded == sum(1 for t in fu if t.task_id not in firsts)
    assert len(merged) == len(tasks) - folded
    assert sum(t.turns for t in merged) == sum(t.turns for t in tasks)
    study.step1(tasks)
    out = capsys.readouterr().out
    assert "opened by a short follow-up" in out and "before" in out and "after" in out
    assert "continue" not in out.split("opened by")[1].split("\n")[0].replace("follow-up", "")


def test_prompt_text_is_never_printed(study, capsys):
    study.main([])
    out = capsys.readouterr().out
    for text, _ in PROMPTS:
        if len(text) > 8:
            assert text not in out
    assert "proj0" not in out


def test_leak_free_never_uses_the_task_itself_or_later_ones(study):
    tasks = study.load()
    rows = study.leak_free(tasks, lambda t: t.turns, quantiles=(0, 100))
    for i, (t, (lo, hi)) in enumerate(rows):
        idx = tasks.index(t)
        earlier = [x.turns for x in tasks[:idx]]
        assert (lo, hi) == (min(earlier), max(earlier))


def test_step3_chooses_on_one_half_and_judges_on_the_other(study, capsys):
    study.step3(study.load())
    out = capsys.readouterr().out
    assert "chosen on the first half" in out and "judged on the second half" in out
    assert "p10-p90 (today)" in out and "(recalibrated)" in out
