# Agent delegation cost tracking

Use the local ledger to compare the total usage required to deliver independently
validated work on nntrainer. It runs with Python 3.9 or newer and writes to
`.cache/agent-cost/ledger.jsonl` by default. Run the examples from the repository
root. In linked worktrees, use `--ledger` with the same local path if you want a
shared ledger. Keep the ledger and rollout files out of Git.

## Record a delegation

1. Before delegation, assign a task ID and a repeatable task class, such as
   `cpp-runtime-integration`, `offline-model-mapping`, or `documentation`.
   Record what a reviewer must verify. Register the implementation attempt with
   the requested model and reasoning effort. Record the resolved configuration
   when the runtime exposes it. Use a stable actor ID so a reviewer cannot
   accidentally appear under the implementer's name.
2. Register each retry, review, rework, and orchestration attempt under the same
   task. Give retries a new ID with `--retry-of`. Use separate rows for parent
   and reviewer work even when their usage is unavailable. A task-specific
   dispatch and its review can share one orchestration attempt if they share one
   usage measurement.
3. Record how every attempt ended. For a deliverable, identify the exact commit
   or use a SHA-256 checksum of the reviewed patch or artifact. A parent, reviewer,
   or separate check runner records the validation verdict, revision, and
   reproducible evidence. A newer delivered revision requires a new validation.
   To end an unsuccessful task, close its attempts and record why the task ended.
4. Enter usage from the relevant meter or import Codex rollout counters.
   Omit usage when unavailable. Mark it complete only if it covers the entire
   attempt. Keep a documented rate, speed, date, and source for any converted
   credit or API estimate. Use the rollout's actual token breakdown if a rate
   distinguishes input, cached input, and output.

```sh
python tools/agent_cost/ledger.py task cache-api cpp-runtime-integration --acceptance "Build the CPU target and pass the cache round-trip test"
python tools/agent_cost/ledger.py attempt cache-api-1 cache-api worker-1 implementation --model-requested gpt-6.1-sol --effort-requested medium
python tools/agent_cost/ledger.py outcome cache-api-1 failed
python tools/agent_cost/ledger.py attempt cache-api-2 cache-api worker-2 rework --retry-of cache-api-1
python tools/agent_cost/ledger.py attempt cache-api-review cache-api reviewer-1 review
python tools/agent_cost/ledger.py outcome cache-api-2 completed --revision REVISION_SHA
python tools/agent_cost/ledger.py outcome cache-api-review completed
python tools/agent_cost/ledger.py validation cache-api cache-api-2 reviewer-1 REVISION_SHA pass --evidence "CPU build passed and cache round-trip test passed"
python tools/agent_cost/ledger.py report
```

For a single run, `usage ATTEMPT BASIS UNIT AMOUNT --source METER_ID [--complete]`
accepts `measured`, `estimated`, and `counterfactual` data in `tokens`,
`credits`, or `usd`. For example, `usage cache-api-1 measured tokens 9000
--source metered-run-1 --complete` records an observed count. Repeating the same
attempt, basis, unit, and source supersedes the previous sample while preserving
it in the ledger. Supply cumulative usage if you enter a preliminary sample and
later update it. Multiple meters for the same attempt, basis, and unit require
reconciling their overlap outside the ledger.

A review may close a task with `closure TASK failed --evidence "CHECKS"` or
`closure TASK abandoned --evidence "REASON"`. A failed test during an open
attempt can be recorded with `note ATTEMPT --evidence "TEST AND RESULT"`.

## Import local Codex usage

Import is opt-in and reads one local rollout. Use its exact recorded checkout as
`--checkout` to guard against accidentally importing another project's session.
Use `--scope attempt` only when you can associate the entire rollout with the
registered attempt. Add `--final` for a finished attempt with complete counters.
For a shared, resumed, or historical session with uncertain attribution, use
`--scope session` and omit `--final`.

```sh
python tools/agent_cost/ledger.py import-rollout cache-api-1 LOCAL_ROLLOUT_JSONL --checkout CHECKOUT_PATH --scope attempt --final
```

The importer extracts cumulative token counters and observed model and effort.
It saves a digest of the session identifier. It ignores prompts, responses, tool
output, usernames, and checkout paths. Reimports replace the previous cumulative
sample in reports. The importer rejects decreasing counters and reuse of a
rollout for another attempt. A live sample remains partial until you confirm
completion. If a resumed rollout adds usage, confirm completion again. A rollout with different model or effort values is reported as
`mixed` for the corresponding field. For a historical session, use a
`historical-unclassified` task and leave the outcome unrecorded unless there
is independent evidence for that task and revision. Session-scope token totals
appear as unallocated observations in reports.

OpenAI distinguishes subscription limits, purchased credits, and API pricing.
Rollout tokens are usage counters. Use account meter data to record measured
credits or actual API spend. Keep rate-based conversions under `estimated`
and hypothetical model swaps under `counterfactual`.
[Official Codex pricing](https://learn.chatgpt.com/docs/pricing)

## Compare comparable assignments

`report` groups attempts by task class, model, reasoning effort, and role.
It also groups each task by its initial implementation configuration so the
total includes failed attempts, retries, review, rework, and parent overhead.
Requested and observed configurations appear separately. If the observed
configuration is missing, it remains unknown.

Compare completed cohorts within the same task class and acceptance criteria.
Review validated successes, failed and abandoned tasks, and usage coverage
alongside cost per success. The cost per success uses all attempts in the
cohort and is available only when all tasks are terminal, every attempt is
closed, the selected measure is complete for each attempt, and at least one
task passed independent validation. Unknown usage appears as `null` or as
an unknown attempt count alongside the known subtotal. Partial usage has its
own count. Every basis and unit has a separate total. Counterfactual usage
has no validated success rate.

When choosing a routing default, prefer the least costly measured configuration
with enough comparable completed tasks to meet the team's validation standard.
Review both the case count and the success rate. If coverage or sample size
is insufficient, keep logging the next comparable tasks and record any cheaper
model or effort as a trial. Use a stronger model or additional review when a
trial fails its acceptance checks, and charge that work to the original task.

Run the tests with `python -m unittest discover -s tools/agent_cost -v`.
