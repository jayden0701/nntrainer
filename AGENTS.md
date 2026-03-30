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