# AGENTS.md

## Goal
Improve x86 CPU kernel performance in nntrainer without changing numerics or public behavior.

## Primary scope
- nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_avx.cpp
- nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl.h
- perf/*
- tools/*
- meson files only when needed to wire the benchmark

## Constraints
- Prefer small, reviewable changes.
- One optimization idea per task.
- Do not assume AVX-512. Treat AVX2/FMA as the baseline unless runtime detection is added.
- Do not refactor unrelated code.
- Do not claim a speedup unless the benchmark shows it.

## Validation steps
1. Build the project.
2. Run unit tests.
3. Run x86 kernel benchmark.
4. Compare against baseline.
5. Keep the change only if:
   - tests pass
   - benchmark median improves by >= 3% on at least one target case
   - no important case regresses by > 1%
   - numerical checks pass

## Benchmark rules
- Default to single-thread kernel benchmarking first.
- Use fixed shapes and repeat enough times to reduce noise.
- Report median, min, max, and relative delta vs baseline.
- If results are noisy or inconclusive, say so and do not update history.

## Recording
- Append accepted results to perf/perf_history.jsonl
- Summarize accepted results in perf/perf_history.md


If a candidate fails the benchmark gate:
- revert the code
- save a short note to perf/rejected_last.md
- include:
  - target function
  - benchmark regressions
  - likely reason for regression
  - what not to try again


 ## Iteration policy

### Roles
- Human or a single orchestrator owns `perf/backlog.yaml`
- Worker agents must read `perf/backlog.yaml` but must not modify it unless explicitly told to act as orchestrator
- Worker agents may append to:
  - `perf/rejected_log.md`
  - `perf/perf_history.jsonl`
  - `perf/perf_history.md`

### Worker loop
1. Read `perf/backlog.yaml`
2. Pick exactly one item that is:
   - assigned to this worker
   - `status: ready`
3. If the item is marked high-risk or blocked, do analysis only and do not edit code
4. Implement only one idea
5. Build the project
6. Run unit tests
7. Run `tools/bench_x86.sh`
8. Compare against `perf/baseline.json`
9. If benchmark gate fails:
   - revert the patch immediately
   - append a short entry to `perf/rejected_log.md`
   - stop
10. If benchmark gate passes:
   - update `perf/perf_history.jsonl`
   - update `perf/perf_history.md`
   - stop

### Constraints
- One idea per iteration
- Small patch only
- No public behavior changes
- No unrelated refactor
- Prefer exact hot-shape wins first
- Do not claim a speedup unless the benchmark shows it
- After two rejected attempts on the same function, stop trying new edits on that function until new profile data is collected

### Failure logging
If a candidate fails the benchmark gate, append this information to `perf/rejected_log.md`:
- date/time
- backlog item id
- function
- owner
- benchmark regressions
- likely reason for regression
- what should not be retried in the same form
- suggested next action