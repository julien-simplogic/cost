# tokentrail

**Where your Claude Code tokens go, and what a prompt will cost before you send it.**

A local command-line tool. It reads the session transcripts Claude Code already
writes on your machine. No API key, no instrumentation, no network: nothing
leaves the machine, and [that is tested](#privacy-tested-not-promised).

## First, the trap: summing the transcript counts every call two to three times

Claude Code writes each session as JSONL. One API call is written as **one
record per content block**: a thinking block, a text block, each tool call.
**Every one of those records carries the call's full `usage`.** A call that
thinks and then makes two tool calls appears three times, with three identical
token counts.

```jsonl
{"type":"assistant","message":{"id":"msg_A","content":[{"type":"thinking",...}],"usage":{"input_tokens":2,"cache_read_input_tokens":27925,"cache_creation_input_tokens":19280,"output_tokens":210}}}
{"type":"assistant","message":{"id":"msg_A","content":[{"type":"tool_use","name":"Bash",...}],"usage":{"input_tokens":2,"cache_read_input_tokens":27925,"cache_creation_input_tokens":19280,"output_tokens":210}}}
```

Any tool that sums `usage` over the lines overcounts, typically by 2 to 3
times, and anyone redoing this work will fall into the same trap.

**The overcount is not a fixed factor.** A call written as *k* lines is
counted *k* times. The overall ratio is therefore the average number of lines
per call, weighted by tokens, and it depends on how you work:

- a plain answer (thinking + text) is 2 lines;
- a step of agentic work (thinking + text + tool call) is 3 lines;
- parallel tool calls make 4 lines or more.

On the session this tool was built in, the cache-read ratio was 2.00 over the
first 10 calls (questions and answers), 2.54 after 40 calls (coding had
started), and 2.62 after 152 calls. It does not drift steadily: it jumps when
the kind of work changes, then plateaus. The ten sub-agent calls barely move
it. Each column (input, cache, output) weights calls differently, which is why
each has its own ratio. **The fix:** group
records by `message.id` (falling back to `requestId`) and count each call once.

**How we know the fix is right.** Claude Code also keeps its own running
counter. It is fed by the API's final usage, which is a different code path
from the per-block lines, and it is written into the same transcript as
`cost-state` records. On a real 2.1.288 session (main thread, Opus):

| | input | cache read | cache write | output |
|---|---:|---:|---:|---:|
| Claude Code's own counter | 152 | 15,141,977 | 263,220 | 125,933 |
| sum over transcript lines | 416 | 43,831,083 | 660,362 | 399,139 |
| ratio | **2.7x** | **2.9x** | **2.5x** | **3.2x** |
| tokentrail (grouped by `message.id`) | 152 | 15,141,977 | 263,220 | 125,933 |

tokentrail matches the counter **to the token**, at each of the three
checkpoints the session wrote. It runs this comparison on every session that
has a counter: the first line of every report says whether the totals are
verified, partly verified, or not verified at all. `tokentrail check` gives it
per Claude Code version.

### Check it yourself, in 30 seconds

You don't have to believe this README, or even tokentrail's code.
`scripts/verify_dedup.py` is one file of standard-library Python, about 100
lines, and it does not import tokentrail. It reads your transcripts and writes
nothing. For every session that carries Claude Code's counter, it compares
three totals over the same calls: the counter, the naive sum of every line,
and one line per `message.id`.

```
git clone <this repository> tokentrail
python3 tokentrail/scripts/verify_dedup.py            # or: ... verify_dedup.py <projects folder>
```

On the session this tool was built in (Claude Code 2.1.288), it prints:

```
1 sessions under ~/.claude/projects; 1 carry Claude Code's counter (cost-state).
                                     input      cache read     cache write        output
Claude Code's counter                  230      25,290,661         388,268       170,375
naive sum of every line                576      65,286,318         922,826       491,227
one line per message.id                230      25,290,661         388,268       170,072
naive / counter                      2.50x           2.58x           2.38x         2.88x
Sessions where 'one line per message.id' equals the counter for input and cache: 1 of 1.
Lines per call: mean 2.54, distribution 1 line: 8, 2 lines: 45, 3 lines: 47, 4 lines: 6, 5 lines: 1, 6 lines: 1.
Each ratio above is this lines-per-call figure, weighted by that column's tokens.
```

**What this proves.** Summing the lines overcounts by the ratio shown, on your
data. Keeping one line per `message.id` gives exactly Claude Code's own input
and cache counts.

**What it does not prove.** That Claude Code's counter equals what you are
billed: it is Claude Code's number, not Anthropic's invoice. Output also
differs slightly (here 170,072 vs 170,375): sub-agent output is under-logged,
[see below](#how-sure-are-these-numbers). If your Claude Code version writes
no counter, the script says so, and nothing can be checked.

## Why

Over one real week of development with Claude Code, the author used 14 million
tokens. Only afterwards did it become clear that 44% of them had gone into
sub-agents launched in parallel and full re-reads of the code.

Existing tools measure API spend. None of them says *where* the budget of
assisted development goes (your messages, tool loops, sub-agents, re-reads),
or what a request will cost *before* you launch it.

tokentrail does both, from the history you already have.

## What it does

### `tokentrail report`: where it went

Over a period, by consumer and by session, plus the most expensive tasks and
how many model calls each one took. A *task* is everything one message of yours
set off, sub-agents included: "this request cost 34 calls".

The sample below is generated by `scripts/demo.py` from **invented** transcripts
(the project names are made up).

```
$ tokentrail report --since 2026-09-21

tokentrail report: since 2026-09-21
14 sessions, 40 tasks, 506 model calls, 23.1M tokens, $17.04 at API prices
  input: 21.5M cache read, 1.3M cache write, 1k new; output: 312k (of which reasoning 19k)

Where it went
consumer            calls  tokens  share    cost  share
------------------  -----  ------  -----  ------  -----
tool turns            273   14.0M    60%   $9.22    54%
sub-agents            135   4.5M‡    19%   $5.01    29%
reviews & re-reads     65    3.3M    14%   $1.97    12%
your messages          33    1.3M     6%  $0.838     5%

Most expensive tasks
task      project       started           family         calls  tokens    cost  sub-agents
--------  ------------  ----------------  --------  ----------  ------  ------  ----------
3df37995  harbor-api    2026-09-25 08:46  refactor  34 (7 sub)    1.4M   $1.13         15%
2c0869f8  lumen-docs    2026-09-26 09:11  refactor  33 (6 sub)    1.4M   $1.12         12%
9767987c  acme-webshop  2026-09-21 08:35  refactor  31 (8 sub)    1.7M   $1.10         14%
a50f17c2  harbor-api    2026-09-27 13:08  refactor  26 (0 sub)    1.9M   $1.03          0%
fbc4ed4c  acme-webshop  2026-09-26 12:34  refactor  25 (6 sub)    1.3M  $0.901         13%
  family: guessed from the first tool calls; * = declared with `tokentrail tag`

By session
session   project       started           tasks  calls  tokens   cost  sub-agents
--------  ------------  ----------------  -----  -----  ------  -----  ----------
8babbc22  lumen-docs    2026-09-26 08:30      4     68    3.6M  $2.45         33%
f3be8698  harbor-api    2026-09-27 12:30      4     62    3.1M  $2.12         23%
...

How sure are these numbers
  Read 33 transcript files, 1621 records: 1541 used, 80 skipped by design, 0 not understood
  ‡ Output: 275k as logged on each call's own line, plus 37k recovered for 19 sub-agent calls ($0.744) from the sub-agent's result in the parent transcript, because their own line was written mid-stream. Rows marked ‡ include them; --logged-only leaves them out.
  116 sub-agent calls were logged mid-stream with nothing to recover from: their output (348 as logged) is a lower bound. Input is exact.
  No session in this period carries Claude Code's own counter: totals are unchecked.
  Prices: (packaged default), verified 2026-09-25
```

Look at both share columns. Token share and cost share diverge because a cache
read is cheap: on Opus 5.5 it costs 1/25 of a 5-minute cache write. A sub-agent starts with an empty
cache and writes its whole context, so it can account for a small share of
tokens and a large share of the bill.

Use `--anonymize` to replace project and session names before you share a report.

### `tokentrail estimate`: before you send

```
$ tokentrail estimate --family refactor "split the cart service in two"

tokentrail estimate: session f3be8698 (harbor-api), claude-opus-5-5

Computed (from the session's last real call)
  last call input                  91,002
  + its answer                        900
  + your text                          12
  = next call input                91,914
    already cached                 91,002   (TTL 5m, last call 4 min ago)
    written to cache                  912
  Your text is 12 tokens; the call sends 91,914: system prompt, tools, files and history ride along.

Predicted (p10-p90 of 15 past 'refactor' tasks; family 'refactor', declared)
  model calls                13 - 32   (main thread 8 - 26)
  input over the task        1.1M - 3.8M
  output over the task       8k - 24k
  cost                       $0.918 - $3.18

Frame
  floor (exact)        $0.023   the first call's input, paid whatever happens
  ceiling             $306.17   27 turns x 128,000 max_tokens, context re-sent each turn
                                tool results and sub-agents have no fixed cap and are not in it

Warnings
  ! The 5m cache expires in 45 s: send now or pay to re-write 92k tokens.
  ! CLAUDE.md is part of the prompt prefix: changing it breaks the cache prefix, and the next call
    that reloads it re-writes the whole context (~92k tokens, $0.460).
```

It keeps two kinds of numbers apart:

- **Computed.** The next call's input is *not* the text you type. It is the
  whole assembled context: system prompt, tools, files, history. Estimating
  from your text alone would say 12 tokens when the call sends 92,000. So the
  estimate starts from the **real input count of the session's last call**,
  read from the transcript, then adds its answer and what you are adding. The
  cached part comes from comparing with that previous call: same model, and
  still inside the cache TTL Claude Code used (5 min or 1 h).
- **Predicted.** Model calls and output come from **your own history**, as the
  10th-90th percentile of comparable past tasks. They are never shown as a
  mean, because the distribution has a long tail. Tasks fall into four families
  (question, refactor, review, measure). A family is guessed from a task's first
  tool calls, or from your text for the estimate. You can always declare it
  (`--family`, or `tokentrail tag <task> <family>` for past tasks).

The **frame** gives the full range. The floor is exact: the first call's input
is paid whatever happens. The ceiling counts `max_tokens` on every turn up to
the turn limit. It also re-sends the growing context on every turn, because
every call re-reads it. (Input + `max_tokens` × turns would understate the
ceiling.) Tool results and sub-agents have no fixed size, so they are left out
of the ceiling, and the output says so.

The most useful part is often the warnings:

| Warning | When |
|---|---|
| cache expired / about to expire | time since the last call vs. the TTL Claude Code used |
| prefix break | `--model` differs from the session's, or `--edits` names a file that is part of the prefix (`CLAUDE.md`, `.mcp.json`, settings) |
| past prefix breaks | this session already lost its cache mid-way, with the cost |
| files loaded, never used | ≥ 10 files read into the context, ≥ 70% never edited or opened again, plus your historical edit rate for that family |
| large context | every further turn re-reads it; `/compact` or a fresh session resets it |
| sub-agent heavy | ≥ 30% of this family's tokens historically went to sub-agents |

## Live display, while you work

```
tokentrail setup        # prints the lines to add to ~/.claude/settings.json; edits nothing
```

**Status line.** Claude Code runs it after each reply. It shows what changes
and what you can act on; anything that is merely reassuring stays in
`tokentrail check`.

```
ctx 43% | 5h 24% | 7d 41% | $17.44 at API rates
```

- `ctx`: context used.
- `5h` and `7d`: how much of your plan's 5-hour and weekly windows you have
  used. These are Claude Code's own figures, documented as `rate_limits`, and
  only Pro and Max subscribers get them; tokentrail does not know your limits
  and shows nothing when Claude Code doesn't pass them.
- `$… at API rates`: the session valued at API prices. On a subscription this
  is not what you pay; it is a common unit for comparing sessions.

Problems appear only when there are some:

```
ctx 43% | $17.44 at API rates | ! cache cold: next message re-caches 431k | ! 2 cache misses (last: tools_changed)
```

The cache alerts use Claude Code's own `prompt_cache` diagnostics when they are
present; on older versions tokentrail spots a miss itself. The other alerts are
a mismatch with Claude Code's counter, transcript lines tokentrail could not
read, and a model missing from the price file. The status line reads only the
session's transcript, so it needs no history; on a 1.3 MB session it takes
about 0.15 s.

**Prompt hook (`UserPromptSubmit`).** When you send a prompt, it shows the
estimate and the warnings:

```
tokentrail: next call 430,797 tokens in (429,141 cached), floor $0.099; this task p10-p90 $0.230-$7.11 (all 10 past tasks, too few 'review' ones)
  ! Large context (431k): every further turn re-reads it (~$0.086 per turn from cache, 27 turns at p90). /compact or a fresh session resets it.
```

Built on Claude Code's documented interface (code.claude.com/docs/en/hooks
and /statusline), read before writing it.

> **If you write a `UserPromptSubmit` hook, read this.** On most hook events,
> plain stdout goes to a debug log. On `UserPromptSubmit` (and
> `SessionStart`), Claude Code adds plain stdout **to the model's context**.
> A hook that prints a cost estimate as plain text therefore makes every
> prompt cost more: a cost-measuring tool that adds cost to each message. To
> show something to the user only, print JSON with `systemMessage`, which
> Claude does not see. tokentrail's hook does only that, and a test fails if
> it prints anything else.

- **The hook only answers with `systemMessage`.**
- **It never blocks or slows a prompt.** It always exits 0. On any error it
  prints nothing. The suggested timeout is 10 s, against a default of 30. It
  reads only the current session, and predictions use whatever history
  `tokentrail ingest` stored before.
- **The status line always prints one line**, even on garbage input.

Limits:

- **Cloud sessions.** In a cloud session the status line and the hook see
  that session's transcript only, not your local history. Heavy work done
  locally is fully visible. If you move that work to cloud sessions, the
  display becomes partial. Whether the status line is shown in the web and
  mobile apps is not documented. The hooks page says hooks run in cloud
  sessions, but I could not confirm it for `UserPromptSubmit` specifically.
- **Not yet watched in a real terminal.** Both commands are tested on
  invented transcripts and were run on a real 2.1.288 transcript with the
  documented JSON. They have not yet been observed inside an interactive
  Claude Code terminal.

## Install

Python 3.11 or newer, no dependencies.

```
git clone <this repository> tokentrail
cd tokentrail
pipx install .        # or: python -m pip install .
```

## When something is missing or broken

tokentrail never shows a traceback. Each case below gets a message that says
what to do next, and each has a test with its own fixture
(`tests/test_robustness.py`):

| Situation | What you get |
|---|---|
| no `~/.claude/projects`, or nothing in it | where it looked; `CLAUDE_CONFIG_DIR` / `--source-dir`; a pointer to the cloud-session limit |
| sessions with no model call yet, or an empty period | what was read, and the dates your history covers |
| a Claude Code version that writes no counter | the report's **first line** says `NOT VERIFIED` (or `Partly verified` with counts) |
| a cut-off last line (session killed or still running) | everything before it is read; the line is counted as `truncated_last_line` |
| bad bytes, unreadable files | counted under "not understood", by reason and by Claude Code version |
| Windows paths and consoles | project names from `C:\...` working directories; characters a console can't show are replaced, not fatal |
| bad `--since`, broken price file, corrupt database, data folder not writable | what is wrong and the one thing to do |
| an open stdin nobody writes to (a pipe, a CI runner) | never read unless you ask (`--file -`); the live commands don't wait when typed in a terminal |
| anything else | a one-line message; `--debug` shows the details |

CI runs the tests on Linux, macOS and Windows. Each test has a 60-second limit
and each CI job a 10-minute limit, so a hang fails with a stack dump instead of
running for hours.

## Use

```
tokentrail report                       # last 7 days
tokentrail report --since 2w --project shop --top 20
tokentrail report --since all --json
tokentrail report --logged-only         # output exactly as each call's own line logged it
tokentrail check                        # per Claude Code version and per month: what to trust
tokentrail setup                        # the settings.json lines for the live display
tokentrail estimate "add retries to the payment client"
tokentrail estimate --file prompt.md --add src/payments.py --family refactor
tokentrail tag 3df37995 review          # declare a past task's family
tokentrail prices                       # show the price file and its date
tokentrail prices --init                # copy it to your data dir to edit
tokentrail where                        # what it reads, where it writes
```

`report` and `estimate` read new or changed transcripts first (incremental).
`estimate` uses the latest session started in the current directory, or
`--session <id prefix>`.

## How it counts

Claude Code writes each session as JSONL under `~/.claude/projects/` (or
`$CLAUDE_CONFIG_DIR/projects`).

- **One call, several records:** de-duplicated by `message.id`, as shown
  [at the top](#first-the-trap-summing-the-transcript-counts-every-call-two-to-three-times).
- **The task id exists.** Every user-side record carries a `promptId` that
  ties the call back to the message you typed. Sub-agents share their parent's.
- **Resumed sessions repeat history.** A call is counted once even if it shows
  up in two session files.
- **Compaction** resets the context legitimately and is not reported as a cache
  break.

### How sure are these numbers

Every report ends with a section of that name, and it is never omitted.

**Sub-agent output is undercounted in the transcripts, and tokentrail says so.**
Sub-agent records are written as soon as streaming starts. Their
`output_tokens` is then a placeholder (typically 1 to 3), and their
`stop_reason` is null.

*How we know.* This is not a deduction from the format alone. On the session
above, the sub-agent (Haiku) made two calls, logged with 3 output tokens each.
Claude Code's own counter says that model produced **309**. Its input, cache
read and cache write match the lines to the token, so the gap is output only.
We have no external reference: Anthropic does not document this. The evidence
is one Claude Code number contradicting another, on calls whose lines also
contain a tool call that cannot fit in 3 tokens.

*What tokentrail does about it, visibly.*

- The parent transcript holds the sub-agent's result, with the **final call's**
  usage (219 output tokens here). tokentrail attributes it to the sub-agent's
  last call **only if its input counts match exactly**. It keeps the logged
  value next to it (`output_logged`) and records where the figure came from
  (`output_source = "subagent_result"`).
- Rows that include such figures are marked **‡**. The report says how many
  tokens, on how many calls, and at what cost, are not from a call's own line.
  `--logged-only` gives the report without them.
- Earlier sub-agent calls have nothing to recover from. They stay as logged
  and are reported as **lower bounds**.
- The remaining gap is measured, not estimated. Where a session has Claude
  Code's counter, the report prints its output next to ours. Here: 222 vs 309,
  so 87 tokens are not visible in the per-call lines. No number is invented to
  fill that gap.

Input and cache counts of sub-agents are exact; the counter confirms them.

### Tolerant parsing

The transcript format is not a public contract. tokentrail only uses what it
recognizes. Anything else is counted and skipped, never fatal, and every report
ends with the counts: records read, used, skipped by design (attachments, queue
events…), and **not understood** (bad JSON, unexpected shapes, unknown record
types), broken down by reason.

**Verified against Claude Code 2.1.288**, including sub-agents. When it sees
transcripts from a newer version, it says so.

### Categories

Each model call is charged to one consumer, first match wins:

1. **sub-agents**: any call made inside a sub-agent;
2. **reviews & re-reads**: calls of a task in the *review* family, or calls
   triggered by re-reading a file already read in that thread;
3. **tool turns**: calls triggered by a tool result;
4. **your messages**: calls triggered directly by what you typed.

Families and categories are heuristics. The code is short and readable:
`src/tokentrail/classify.py`.

## Architecture

```
collectors/ (one per source)  ->  UsageRecord  ->  SQLite (local)  ->  report / estimate
```

Each source has a collector, and every collector emits the same record. That
record is the contract. v0 ships one collector (Claude Code, local). Other
agent harnesses or API logs can be added as collectors without changing
anything downstream.

| Field | Meaning |
|---|---|
| `timestamp`, `source`, `model` | when, which collector, which model |
| `input_total` = `input_cache_read` + `input_cache_write` + `input_new` | input tokens, split by cache status (`input_cache_write_1h` = part written with the 1-hour TTL) |
| `output_total`, `output_reasoning` | output tokens, of which reasoning when the source reports it |
| `output_logged`, `output_source` | what the call's own line said, and where `output_total` comes from (`logged` or `subagent_result`, marked ‡) |
| `turn_id` | one model call |
| `task_id` | everything one request of yours caused, sub-agents included |
| `trigger` | `user_turn`, `tool_call` or `subagent` |
| `duration_ms` | from the triggering record to the end of the call |
| `output_exact` | false when logged mid-stream with nothing to recover: a lower bound |
| `session_id`, `project`, `agent_id`, `extra` | optional context |

See `src/tokentrail/record.py`.

## Prices are data

`src/tokentrail/data/prices.toml` holds USD per million tokens per model,
plus the cache-write multipliers and a `verified_on` date. `tokentrail prices
--init` copies it into your data directory. Your copy wins from then on, so you
can fix a price without waiting for a release. Reports say which file and date
they used, and warn when it is more than 60 days old.

On a Pro or Max subscription you are not billed per token. API prices are
still the best common unit to compare where your limits go.

## Privacy: tested, not promised

The transcripts contain your code and your clients' code. So the guarantees are
tests (`tests/test_privacy.py`), run in CI on every push:

- **No network.** A Python audit hook ([PEP 578](https://peps.python.org/pep-0578/))
  fails the test on any socket, DNS lookup, HTTP connection or subprocess
  while every command runs against invented transcripts. A static check also
  rejects any network or subprocess import in the package.
- **Writes only to its own directory.** The same hook fails on any file
  opened for writing, created, renamed or deleted, and on any SQLite database
  opened outside tokentrail's data directory. Every command is covered.
- **Transcripts untouched.** Hashes of every file under the fake home
  directory are identical before and after.
- **The guards are tested too.** Each one is first shown to catch a deliberate
  violation, including tokentrail's own writes when its directory is not
  allowed. A passing test therefore means something.
- **Test data is invented.** `tests/fake_transcripts.py` generates
  transcripts shaped like the real format. No real transcript is used.

tokentrail writes to `$TOKENTRAIL_HOME`, or by default to
`~/.local/share/tokentrail` (`%LOCALAPPDATA%\tokentrail` on Windows). Prompt
text is never stored, only its length.

## What v0 does not see: cloud sessions

Claude Code sessions run on Anthropic's machines (claude.ai/code, the mobile
app) write their transcripts **there**, not on your computer, so a local
tokentrail cannot see them. If you move heavy work (parallel sub-agents, full
reviews) to cloud sessions, tokentrail goes blind exactly where you spend.

Honest options, none of them complete:

- **Run tokentrail inside the cloud session.** The transcript exists in that
  session's container while it lives. You can ask for `tokentrail report` in
  the session itself. The container is reclaimed afterwards, so this gives no
  history.
- **Per-session totals from Anthropic.** The cloud session record exposes
  totals (tokens by type, API-price cost), equal to Claude Code's own counter.
  They are readable from a cloud session's tools, not from the local CLI, and
  carry no per-turn detail: no categories, no tasks.
- Sessions you drive remotely but that run on your machine (for example from
  VS Code) are local, and fully visible.

## What v0 does not do

- **No brake.** v0 measures and advises. A brake will come next. When it does,
  it will refuse before sending, or stop between two turns. It will never work
  by lowering `max_tokens`: a reply cut off mid-sentence is paid for and
  useless.
- **No ChatGPT, no Claude web or desktop chat.** They publish no token counts,
  and tokentrail will not pretend to know them.
- **No exact tokenizer offline.** Your text and added files are converted with
  a characters-per-token ratio calibrated on your own history (the output
  shows the ratio and its basis). That ratio sits in the input *delta*. The
  base, the last call's real input, comes straight from the transcript.
- Windows is not covered by CI yet.

## Development

```
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/demo.py           # the README samples, on invented data
python scripts/leak_check.py     # before every push: tree + every commit
```

`scripts/leak_check.py` looks for personal paths, e-mail addresses, unknown
domains, and the terms in a private denylist (`.leak-denylist`, git-ignored)
in the working tree and in every commit, metadata included.

## License

MIT
