# Delegation ledger validation, 2026-11-17

The ledger tool was tested on nntrainer base `5e97254e7`. The parent ran
`python3 -m unittest discover -s tools/agent_cost -v` in WSL and
`git diff --check` with Windows Git. The original nine tests passed, and the
parent independently reviewed that revision. Patch export then detected extra
blank lines at the ends of three files. The parent matched the normalized
source digest and reran the nine tests after those lines were removed.

Review found that a resumed rollout could retain its final status after its
token counters increased. The patched importer requires reconfirmation after
new usage and retains final status for unchanged counts. The parent passed
all ten tests and independently validated the revised artifact. The patch
also passed the whitespace check and `git apply --cached --check`.

## Current delegation measurements

Token counters were captured at 05:17 UTC.

| Task class | Requested and observed model | Requested and observed effort | Input tokens | Cached input tokens | Output tokens | Total tokens | Measurement scope |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Developer tooling | GPT-6.1 Sol | Medium | 1,378,198 | 1,327,104 | 23,176 | 1,401,374 | Partial session spanning implementation, rework, and reporting |
| C++ runtime integration | GPT-6.1 Terra | High | 13,405,385 | 13,292,928 | 29,624 | 13,435,009 | Final dedicated attempt |
| Offline model mapping | GPT-6.1 Sol | Medium | 4,110,821 | 4,029,440 | 31,718 | 4,142,539 | Final dedicated attempt |

For each subagent log, the first cumulative sample matched the first request's
usage across all recorded counters. The agent identity, model, and reasoning
effort also matched. Both final attempt logs ended with task-completion events.
Cached input is included in input tokens. Total tokens comprise input and
output. The developer-tooling session total is an unallocated observation.
Its earlier partial implementation snapshot overlaps this total and is omitted
from this table.

The parent orchestration and review usage remains unknown. The first C++
runtime build failed on an undeclared `core_access` reference and triggered a
correction during the same attempt. The final counters include that work.

| Task class | Latest validation | Validated artifact SHA-256 |
| --- | --- | --- |
| Developer tooling | Passed | `976f6a04718fa3ebc4dba9ea56263de52eb32fb79954eae62b24eb848c8e0500` |
| C++ runtime integration | Passed | `94c6ea0012569b527bf5e98a7f4cc4195ef29bb479033a38498567f4aac4011a` |
| Offline model mapping | Mapper tool passed its verification. The smoke test showed different generated tokens; it did not measure downstream benchmark quality. | `cb105c515d44e1c090441b3a34cc1c2f9c80aa2520005b07cbe519a766d098df` |

The final mapper export normalized trailing whitespace. A separate reviewer
verified the new digest above and reran all six mapper tests in WSL. The
normalization and its review have unknown usage.

A separate local import captured 14 historical session-scope observations
with unknown task outcomes. Five had mixed observed reasoning effort. Their
session totals remain unallocated.

The three validated deliverables belong to different task classes. Usage
coverage is incomplete, so cost per validated success is unavailable.
The figures are token counts. Account charges and included subscription
usage were unavailable.
[Official Codex pricing](https://learn.chatgpt.com/docs/pricing)
