# tokentrail

**Where your Claude Code tokens go, and what your next message is sure to cost before you send it.**

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
Claude Code's counter              145,076      50,078,149         604,102       251,777
naive sum of every line              1,066     129,236,142       1,554,591       684,869
one line per message.id                406      50,078,149         604,102       238,988
naive / one per message.id           2.63x           2.58x           2.57x         2.87x
Sessions where 'one line per message.id' equals the counter for input and cache: 0 of 1; above it (would mean overcounting): 0.
Input and cache by model (counter vs one line per message.id):
  claude-haiku-4-5-20251001                  598,157         453,487        -144,670
  claude-opus-5-5                         50,229,170      50,229,170              +0
A model the counter has more of usually ran inside a tool (WebFetch reading a page), which writes no call line.
Lines per call: mean 2.58, distribution 1 line: 15, 2 lines: 58, 3 lines: 83, 4 lines: 12, 5 lines: 1, 6 lines: 1.
Each ratio above is this lines-per-call figure, weighted by that column's tokens.
```

**What this proves.** Summing the lines overcounts by the "naive / one per
message.id" ratio, on your data. For the model that wrote the conversation,
one line per `message.id` gives exactly Claude Code's own input and cache
counts.

## Second trap: the transcript is a floor, not a total

**Any tool that calls a model internally creates usage that Claude Code
counts and the transcript never records.** The session above shows it with
`WebFetch`, and the same holds for every tool built this way.

Here, Haiku is
144,670 tokens short. That is the session's two `WebFetch` calls. The tool has
a small model read each fetched page, Claude Code counts that call, and no
call line records it (the tool's result carries only bytes and a duration). So
the call lines are a lower bound on what a session used. Counting *more* than
the counter would mean a counting error. Counting *less* means usage that no
call line records. tokentrail tells these two cases apart in every report and
shows the unattributed part separately, instead of hiding it or spreading it
over categories it can't be assigned to. `tokentrail diagnose <session>` gives
the per-model detail.

Read every total built from the transcript as a **minimum**. Before this
check, you could read a total without knowing in which direction it was
wrong. Known mechanisms only remove usage from the lines: tools' own model
calls, and sub-agent output written mid-stream. Counting *more* than the
counter is never explained by them, so tokentrail treats it as an error.

How tokentrail turns this into a verdict, on the first line of every report:

- **Verified**: every checked session matches Claude Code's counter to the
  token, *and* every model appears on both sides. If a model appears on one
  side only, the verdict is at best "Partly verified", and the model is named.
  (On one machine the counter said `claude-opus-5[1m]` where the call lines
  said `claude-opus-5`. An earlier version of the check skipped that model
  silently and could have reported a match while ignoring most of the volume.
  Names are now compared without bracketed qualifiers; `diagnose` shows them
  as written.)
- The counter side is always the session's **last** `cost-state` record.
  Those records are cumulative snapshots, sometimes written twice in a row,
  so adding them up would count the same usage several times. A test fails if
  the comparison ever sums them. `tokentrail diagnose <session> --raw` prints
  every record as written.
- **Could not verify** when tokentrail counts more than the counter between
  two snapshots of one run, or when the gap grows interval after interval.
  This says the totals are unverified, not which side is wrong: what the
  counter means across Claude Code versions is not documented.
- The counter belongs to a run of Claude Code, not to a file. It can start
  with usage carried in from outside the file, or restart at zero when a
  session is resumed. So what is compared is what both sides *added* between
  two snapshots of the same run. That difference cancels both effects.
  When a session has a single snapshot, tokentrail checks whether the counter
  equals exactly the last *n* calls before it. If it does, on input, cache read
  and cache write at once, the counter covers only a final run, and that run
  is verified to the token. On one machine, a file written by three Claude Code
  versions had a counter equal to exactly its last 4 of 154 calls. Claude Code writes its counter at the end of
  each turn. A hidden call makes the gap jump between one or two snapshots; a
  counting error makes it grow at every one. Each disagreeing session shows
  its gap with sign, size and shape. When the counter goes *down* between two
  snapshots, Claude Code restarted counting, and the report says so.
- Otherwise "Partly verified", with the totals flagged as a minimum.

**On the author's machine**, said with its sample size: on Claude Code
2.1.283, 15 of the 21 sessions that ran on that version alone match its
counter to the token, and none contradicts it. Where both sides cover the same
calls (26 sessions), the call lines are a floor short by +0.23% on input and
cache, and by +12.7% on output (about 1% of the cost). 15 of 21 is a small
sample. It is the reason to trust the rest only as far as `tokentrail check`
confirms it on your own sessions.

**What it does not prove.** That Claude Code's counter equals what you are
billed: it is Claude Code's number, not Anthropic's invoice. Sub-agent output
is under-logged, [see below](#how-sure-are-these-numbers). If a session carries
no counter (many on one machine did not), the script says so, and nothing can
be checked.

## Why

The author's plan limits melted without an obvious reason. The first guess,
made after a week of work, was that 44% of the tokens went into sub-agents
launched in parallel and full re-reads of the code. Measured with this tool
over the whole local history (145 sessions, 3,892 tasks, 64,409 model calls),
the guess was wrong:

| where it went | tokens | cost at API prices |
|---|---|---|
| tool turns (the loop of calls between two of your messages) | 79% | 68% |
| your messages | 8% | 20% |
| sub-agents | 12% | 7% |
| reviews and re-reads | 1% | 4% |

Sub-agents and re-reads together: 13% of tokens, not 44%. Cache lost to idle
time or to a changed prompt prefix: 21% of the cost. Input is 99.8% of tokens;
output, 0.24%. These are one person's figures on one machine; yours will
differ, which is the point of measuring them.

Existing tools measure API spend. None of them says *where* the budget of
assisted development goes, or what the next message will cost *before* you
send it. tokentrail does both, from the history you already have.

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

Cache warm (5m TTL, expires in 45 s): your next message reads 91,002 of its 91,914 tokens
from cache, $0.023. Send now or it re-writes them all.

Computed (from the session's last real call)
  last call input                  91,002
  + its answer                        900
  + your text                          12
  = next call input                91,914
    already cached                 91,002   (TTL 5m, last call 4 min ago)
    written to cache                  912
  Your text is 12 tokens; the call sends 91,914: system prompt, tools, files and history ride along.
  (text -> tokens at 2.5 chars/token, default, too little history to calibrate; no network tokenizer)

Unvalidated estimate: turns from 15 past 'refactor' tasks (family 'refactor', declared)
  model calls (main thread)  8 - 26, median 17
  cost of those turns        $0.159 - $0.479, median $0.317
                             (turns x context; output and sub-agents not included)
  Not yet compared with outcomes: `tokentrail score` says how past estimates fared.

Frame
  floor (exact)        $0.023   the first call's input, paid whatever happens
  ceiling             $306.17   27 turns (your max for this basis) x 128,000 max_tokens,
                                context re-sent each turn
                                tool results and sub-agents have no fixed cap and are not in it

Warnings
  ! CLAUDE.md: whether Claude Code re-reads it mid-session, and where it sits in the prompt, is
    not in the transcripts, and your history has no such edit to measure; if it does, the whole
    92k-token context is written again ($0.460).
```

Three kinds of numbers, never mixed:

- **Certain, stated first.** Arithmetic on a state the transcript shows. The
  next call's input is *not* the text you type: it is the whole assembled
  context (system prompt, tools, files, history). Estimating from your text
  alone would say 12 tokens when the call sends 92,000. So it starts from the
  **real input count of the session's last call**, read from the transcript,
  plus its answer and what you are adding. Whether that context is still
  cached follows from the idle time, the cache TTL Claude Code used (5 min or
  1 h) and the model. After a long pause the first line reads, for example:
  `Idle for 48 min: the 5m cache has expired. Your next message re-writes the
  context, up to 91,915 tokens, $0.460 ($0.437 more than with a warm cache).`
  `tokentrail score` backtests this on your history. On the author's: the
  next-call input was within 5% in 3,659 of 3,724 prompts (median error
  0.0%), and "warm" held in 3,265 of 3,280. "Cold" held in 351 of 444. In
  91 of the other 93, the session had been idle past a 1-hour TTL (median
  3 hours), and a median 15% of the context was still read from cache. So
  "up to": after an expiry, part of the prompt can survive. When your history
  has enough expiries, a second line gives what survived after them and the
  cost that implies. On the author's history that part is nearly fixed in
  size: after 445 expiries, p10 22,075 tokens, median 23,981, p90 30,857,
  even after 22 hours idle. One explanation was tested and ruled out: another
  session on the same model, active just before, keeping a shared prefix
  warm. The median is the same with one (23,985) and without (23,870). Where
  those tokens come from is not in the transcripts ([open question](#open-questions)).
- **Unvalidated estimate.** The one real unknown is how many turns the task
  will take. tokentrail does not try to predict output (on the author's
  history, input is 99.8% of tokens and is computed; output is 0.24%). It takes
  the 10th, 50th and 90th percentiles of main-thread turns over your past
  tasks of the same family, never a mean, and multiplies by the context size.
  When a family's turns spread over two orders of magnitude (p90 ≥ 100 × p10),
  it says so and gives no interval: one that wide covers everything and says
  nothing. **This part has not been checked against outcomes yet**, which is
  why it is called an estimate, not a prediction. Every estimate is recorded
  so that `tokentrail score` can check it (see below).
- **The frame.** The floor is exact: the first call's input is paid whatever
  happens. The ceiling counts `max_tokens` on every turn up to the turn
  limit and re-sends the growing context on every turn. Tool results and
  sub-agents have no fixed size, so they are left out of the ceiling, and the
  output says so.

**Families do not separate turns, so a guessed family is not used.** Tasks fall
into four families (question, refactor, review, measure). On the author's
3,649 tasks with a prompt, the family guessed from the prompt text, the only
one known before sending, gave nearly the same distribution everywhere:

| family guessed from your text | tasks | p10 | median | p90 |
|---|---|---|---|---|
| question | 3,087 | 1 | 5 | 28 |
| refactor | 227 | 1 | 5 | 42 |
| review | 245 | 2 | 11 | 57 |
| measure | 89 | 1 | 4 | 34 |

Only "review" stands apart, and its interval still spans 2 to 57. The family
guessed afterwards from a task's tool calls looks more telling, but partly by
construction: "question" means few or no tool calls. It agreed with the text
guess on 1,847 of 3,649 tasks. So `estimate` takes its turns from all your
past tasks, and records the guessed family so `score` can keep checking. A
family you declare (`--family`, or `tokentrail tag <task> <family>`) is used.

**What else was measured** (`scripts/study.py`, read-only, on the author's
3,916 finished tasks; every backtest estimates each task from earlier tasks
only, with 95% intervals from resampling sessions):

- *The p10-p90 interval is too wide:* it holds 89% of outcomes, not 80%.
  p12.5-p87.5, chosen on the older half, narrows it from 29 to 25 turns on the
  newer half, but holds 89% there too and leaves the interval score unchanged
  (57.2 against 57.4).
- *The longer a task has run, the longer it still runs.* Median turns still
  ahead: 4 after 1 turn, 5 after 3, 7 after 5, 11 after 10 (p90 55).
- *The first task of a session is the long one:* median 22 turns, p90 108
  (146 sessions), against 4 to 7 for later ones. The previous task's turns
  and the rank in the session improve the interval score (by 2.7 and 2.9, both
  intervals below 0) but leave coverage at 86-87%, outside the 75-85% asked
  for. Prompt length, files mentioned, slash command, project and model do
  not improve it. Short follow-ups (19 tasks) and pasted errors (17) are too
  rare to tell.
- *Cost:* turns × context gives the best interval score of four methods
  ($17.97, against $18.70 for quantiles of the cost itself and $32.46 for
  cost scaled by the first input), with 78.5% coverage, but its median runs
  low: about 0.76 times the actual cost, since output and sub-agents are left
  out. Adding the context's growth per turn (median 277 tokens) changes
  nothing.

**What a CLAUDE.md edit costs is measured, not asserted.** Where Claude Code
puts `CLAUDE.md` in the prompt, and whether it re-reads it mid-session, is not
written in the transcripts, so tokentrail cannot say "the prefix breaks at
character 3,200". What it can say is what happened on your history: each time
a tool wrote `CLAUDE.md`, a settings file or something under `.claude/`, did
the next call find its cache? `tokentrail check` counts it, and attributes
every cache break to what the transcript shows just before it: a model switch,
such an edit, or nothing visible. Edits made in your own editor are not in the
transcripts and land in "nothing visible".

Other warnings:

| Warning | When |
|---|---|
| prefix file edited | `--edits` names `CLAUDE.md`, `.mcp.json`, a settings file or a `.claude/` path, with what such edits did on your history |
| past prefix breaks | this session already lost its cache mid-way, with the cost |
| files loaded, never used | ≥ 10 files read into the context, ≥ 70% never edited or opened again, plus your historical edit rate for that family |
| large context | every further turn re-reads it; `/compact` or a fresh session resets it |
| sub-agent heavy | ≥ 30% of this family's tokens historically went to sub-agents |

### `tokentrail turns`: how many turns a task takes

```
$ tokentrail turns

tokentrail turns: how many model calls a task takes, by family

family    tasks  p10  median  p90  p90/p10  all calls p50/p90
--------  -----  ---  ------  ---  -------  -----------------  --------------------------
question      6    1       1    1       x1              1 / 1  too few (estimate needs 8)
refactor     16    8      18   26       x3            23 / 32  tight
review        7    7       8   12       x2            17 / 22  too few (estimate needs 8)
measure      11    2       2    2       x1              2 / 2  tight
all          40    1       8   23      x23            14 / 26  wide

  p10 / median / p90: main-thread model calls per task, the ones that each re-read the whole
  context, so the cost of a task is roughly turns x context size. All calls adds sub-agents.
  A family whose p90 is 100x its p10 or more (two orders of magnitude) gets no
  interval from `estimate`: one that wide would cover everything and say nothing.
  Families: 0 of 40 declared with `tokentrail tag`, the rest guessed from each
  task's first tool calls. `estimate` guesses from your text instead; `score` says how often
  the two agree.
```

(Invented demo data, as everywhere in this README.) Main-thread turns are the
calls that each re-read the whole context, so the cost of a task is roughly
turns × context size. Read the p90/p10 column before trusting any interval.
Families here are guessed from each task's tool calls, which `estimate`
cannot know before you send; see above why it does not use them.

### `tokentrail score`: the estimate grades itself

A prediction that is never scored is an opinion. Every `tokentrail estimate`
(unless `--no-record`) and every prompt seen by the hook records what it said:
numbers only, never your text. Each record is linked to the next task that
starts in the same session within 30 minutes. Once that task is over (a later
task started, or an hour without activity), its outcome is written next to the
estimate and kept, even after Claude Code deletes old transcripts.

`tokentrail score` then reports, over the last N scored estimates (`--last`,
default 50):

- how many outcomes fell inside the p10-p90 turn interval (a calibrated
  interval holds about 80%), and the median error in turns;
- the same for the cost band, with the median ratio actual / estimated (the
  band leaves out output and sub-agents, so expect it to run low);
- the computed part: next-call input error, and whether the cache was warm or
  cold as stated;
- how often the family guessed from your text matched the one guessed later
  from the task's tool calls.

Each interval is also judged by measures that cannot be gamed by widening it:
coverage against the nominal 80%, the interval score (Winkler: width plus a
penalty of 10 per unit by which the outcome falls outside; lower is better),
the median absolute error, and the median log(estimate / actual), which
treats twice too high and half too low alike. Each comes with a 95% interval
from resampling whole sessions, not tasks, because tasks of one session are
alike and resampling tasks would claim more certainty than the data holds.

It also backtests on your whole history what needs no recorded estimate: the
computed next-call input and cache state before each of your prompts, and the
turn interval each task would have got from the tasks before it. The
per-family backtest uses the family guessed from the task's own tool calls,
which the estimate cannot know, and is labelled optimistic.

## Live display, while you work

```
tokentrail setup        # prints the lines to add to ~/.claude/settings.json; edits nothing
```

**Status line.** Claude Code runs it after each reply. It shows what changes
and what you can act on; anything that is merely reassuring stays in
`tokentrail check`.

```
ctx 43% | 5h 24% | 7d 41% | cache 3m left | $17.44 at API rates
```

- `ctx`: context used.
- `5h` and `7d`: how much of your plan's 5-hour and weekly windows you have
  used. These are Claude Code's own figures, documented as `rate_limits`, and
  only Pro and Max subscribers get them; tokentrail does not know your limits
  and shows nothing when Claude Code doesn't pass them.
- `cache 3m left`: time before the cache written by the last call expires
  (5 minutes or 1 hour, read from that call), as of the last refresh of the
  line. Computed, not estimated.
- `$… at API rates`: the session valued at API prices. On a subscription this
  is not what you pay; it is a common unit for comparing sessions.

Problems appear only when there are some:

```
ctx 43% | $17.44 at API rates | ! cache expired: next message re-writes up to 431k, $2.16
ctx 43% | $17.44 at API rates | ! cache cold: next message re-caches 431k, $2.16 | ! 2 cache misses (last: tools_changed)
```

After an idle spell the line says what re-writing the context will cost: the
last call's input and answer at the cache-write price, "up to" because part of
the prompt can survive (see `score`). Claude Code's own `prompt_cache`
diagnostics, when present, take precedence.

The cache alerts use Claude Code's own `prompt_cache` diagnostics when they are
present; on older versions tokentrail spots a miss itself. The other alerts are
a mismatch with Claude Code's counter, transcript lines tokentrail could not
read, and a model missing from the price file. The status line reads only the
session's transcript, so it needs no history; on a 1.3 MB session it takes
about 0.15 s.

**Prompt hook (`UserPromptSubmit`).** When you send a prompt, it shows first
what is certain (idle time, cache state, what re-writing costs), then the
unvalidated turn estimate and the warnings:

```
tokentrail: Idle for 48 min: the 5m cache has expired. Your next message re-writes the
context, up to 91,915 tokens, $0.460 ($0.437 more than with a warm cache).
  unvalidated estimate: 1-20 turns, median 7 (all 39 past tasks, too few 'question' ones)
  x context = $0.460-$0.805, output and sub-agents not included
```

Each of these is recorded for `tokentrail score`, numbers only.

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
  reads only the current session, and the turn estimate uses whatever history
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
tokentrail diagnose <session>           # one session vs Claude Code's counter, numbers only
tokentrail setup                        # the settings.json lines for the live display
tokentrail estimate "add retries to the payment client"
tokentrail estimate --file prompt.md --add src/payments.py --family refactor
tokentrail turns                        # turns per task by family: p10, median, p90
tokentrail score --last 30              # past estimates vs what happened, plus a backtest
tokentrail tag 3df37995 review          # declare a past task's family
tokentrail prices                       # show the price file and its date
tokentrail prices --init                # copy it to your data dir to edit
tokentrail where                        # what it reads, where it writes
```

`report`, `check`, `estimate`, `turns` and `score` read new or changed
transcripts first (incremental), then record the outcome of any finished task
an estimate was waiting for.
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
collectors/ (one per source)  ->  UsageRecord  ->  SQLite (local)  ->  report / estimate / score
```

Everything in the database is rebuilt from the transcripts, except two tables
nothing else can rebuild: families you declared (`tag`) and recorded
estimates with their outcomes (`score`). Those survive upgrades.

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
text is never stored, only its length; recorded estimates hold numbers only,
and a test checks that a prompt's words are not in the database.

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
- **No validated turn estimate yet.** The scoring loop exists; the scores
  do not, until enough estimates have met their outcome. Until `tokentrail
  score` says otherwise on your own history, read the turn interval as a
  guess with a stated basis.

## Open questions

**The block that survives a cache expiry.** About 24,000 tokens are still read
from cache after an expiry (author's history, 445 expiries: p10 22,075,
median 23,981, p90 30,857), even after 22 hours idle. The obvious candidate
is the system prompt plus tool definitions. A test on the transcripts
supports it without settling it:

- The same exact values recur across sessions and projects (63 × 23,870,
  46 × 22,799, 34 × 30,857 tokens…): a fixed piece of content, not something
  that ages.
- The value depends on the Claude Code version (median 30,857 on 2.1.276,
  23,470 on 2.1.283).
- A new session's very first call already reads it from cache (median 25,467
  tokens; zero in 2 of 145 sessions).
- But within one session, its expiries read the same value to the token in
  only 46 of 82 sessions. Tools added or dropped mid-session (MCP servers,
  skills) could explain the rest; the transcripts do not record the tool
  list, so this cannot be checked from them.

Why it outlives the cache TTL even with no other session active is not
explained either. **Unverified hypothesis:** the prefix is kept warm by other
requests from the same account that leave no transcript on this machine (a
session on another computer, a cloud session, Claude Code's own background
calls). The test above only saw local sessions, so it cannot rule this out.
Between two expiries of one session the block stayed identical in 252 of
316 pairs. When it changed (median 2,685 tokens), the transcript more often
showed, in between, a change of the loaded tool list (`deferred_tools_delta`
in 23 of 64 against 19 of 252), of instructions (20/64 against 14/252), of
MCP instructions (13/64 against 9/252), of Claude Code version (15/64 against
7/252). That fits the system prompt plus tools; it does not prove it. tokentrail uses the measured median and does not depend on
the answer.

**The cache breaks with no visible cause.** On the author's history, 169
breaks cost $1,283.61. They are not spread out: one session holds 138 of them
($1,062.76), three hold 155. 160 come from Claude Code 2.1.250; since 2.1.276
there were 9. They share a shape: very late in long sessions (call index
median 924), mid-task, 22 seconds after the previous call on a 1-hour cache,
the previous call used `Read` (149 of 169, against 6% of calls overall),
exactly the ~29,000-token block at the start of the prompt was still read
(168 of 169), and in 82 the context had shrunk (median 1,789 tokens). Something
early in the conversation was rewritten. What, the transcripts do not say.

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
