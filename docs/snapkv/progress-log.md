# SnapKV implementation progress log

This file is the compact handoff state for long-running work. Update it when a
decision changes or a verification pass completes.

## 2026-08-03 02:49-03:30 KST

- Confirmed branch `main` and preserved all pre-existing untracked files.
- Located local `SnapKV.pdf` (17 pages, June 2024 preprint).
- Rendered and visually inspected all 17 pages; extracted 110,783 characters
  with pdfplumber for cross-checking.
- Identified arXiv ID `2404.14469` and the final NeurIPS version.
- Pinned the author repository at
  `e216ddc84c5bd210378cbdbbba12ba02102aa640`.
- Confirmed the author implementation performs causal softmax first, sums the
  observation-window attention targeting only the prefix, pools, performs
  per-head top-k, gathers K/V, and appends the observation window.
- Confirmed an important GQA detail: author code calls `repeat_kv` before
  compression and therefore expands cache head width.
- Mapped nntrainer's external cache shape to
  `[batch, 1, max_seq_len, Hkv * head_dim]` and confirmed decode kernels depend
  on fixed GQA sharing.
- Confirmed `mha_core` already materializes softmax-normalized prefill attention
  in a triangular CPU tensor, so SnapKV scoring can reuse it without a second
  QK pass.
- Identified required position split: cache rows become physical positions,
  while RoPE and model generation indices must remain logical positions.
- Chosen v1 GQA policy: pool each query-head score vector, then aggregate within
  each KV group and select one position set per KV head. This preserves native
  GQA cache width and follows NVIDIA KVPress's operation ordering.
- Chosen v1 scope: first full causal prompt only, no attention sink, skip local
  sliding-window layers, no decode-time re-eviction, no physical slab shrink.

## Remaining external evidence

- Full Meson/model runtime testing remains deferred because this environment
  cannot build the complete target; see `implementation-report.md` for the
  exact follow-up commands and evidence boundary.

## 2026-08-03 03:30-04:40 KST

- Added a pure CPU `SnapKVPolicy` for causal observation-score aggregation,
  max/average pooling, deterministic per-KV-head top-k, byte-exact K/V
  compaction, and logical/physical cursor transitions.
- Added overflow-safe size arithmetic, deterministic NaN handling, overlap-safe
  temporary buffers, and explicit invalid-state/configuration failures.
- Integrated one-shot post-prefill compaction after the full attention output is
  computed. Full-attention layers compact; finite sliding-window layers retain
  their original cache behavior.
- Kept RoPE on absolute logical positions while cache reads/writes use the
  per-layer physical cursor. Added pre-compute logical bounds for both
  `max_timestep` and `max_position_embeddings`.
- Added strict `nntr_config.json` parsing and post-compile property propagation
  to all `mha_core` nodes. Blocked skip-prefill, sinks, non-causal/internal
  cache paths, and pre-computed cache save/load.
- Updated Meson and both Android source lists and documented the user-facing
  configuration and v1 memory limitation.
- Refactored the production aggregation path to call the same typed policy
  implementation exercised by tests.
- Built and ran the focused GTest binary directly with MinGW: 20/20 tests
  passed. Tests cover a scalar
  oracle, GQA, padding, ties, NaN, large kernels, cursor state, FP32/two-byte
  cache copying, transactionality, and overflow.
- Compiled `snapkv_policy.cpp` with Clang (`-Wall -Wextra -Werror -pedantic`
  and stricter conversion flags). Replayed existing MinGW compile commands for
  `mha_core.cpp`, `transformer.cpp`, and `causal_lm.cpp`; all passed after
  applying documented workarounds for pre-existing `M_PI` and `WIN_EXPORT`
  warnings in this Windows shell.
- Completed independent policy, MHA integration, verification-plan, and final
  audit passes. Review findings led to fixes for signed overflow, byte-offset
  overflow, GQA validation timing, strict JSON numeric conversion, logical
  RoPE bounds, save/load guards, Windows test linking, state coverage, and an
  invalid `LayerNode` downcast caught before handoff.

## 2026-08-03 04:40-05:10 KST

- Reconciled a cross-document GQA ambiguity found by independent audit.
  Refactored the policy and production path to score and pool per query head,
  then group-sum pooled scores per KV head. This matches NVIDIA KVPress and is
  observably different from reduce-first under max pooling.
- Added a non-commutative max-pooling regression vector and invalid GQA
  geometry tests. Hardened forged cursor-state validation and replaced the
  reserved include guard.
- Made v1 caller boundaries explicit: initial prefill must be a single complete
  unpadded/equal-length batch; chunked prefill and variable-length padded or
  packed batches are unsupported; all-sliding models have no eligible layer.

## 2026-08-03 05:10-05:40 KST

- Added a committed 128-geometry scalar-oracle loop and reran the focused
  suite: 23/23 tests passed.
- Ran targeted Clang-Tidy 14 `bugprone`, `performance`, and `portability`
  checks; after excluding the API-style-only swappable-parameter diagnostic,
  no user-code diagnostic remained.
- Updated the normative spec status, pinned NVIDIA GQA precedent, intentional
  departures, precise non-finite ranking, exception types, runtime eligibility,
  and physical `C+t` terminology.
- Expanded the README with complete-prompt/equal-length restrictions,
  configuration constraints, decode growth, and compaction scratch memory.
- Hardened CausalLM construction so all rejecting configuration parsing occurs
  before checked `ids_history` allocation, with allocation failure and repeated
  setup handled safely.
- Completed and validated the reusable `implement-research-paper` skill. A
  blind Group Normalization forward test caught an intentionally wrong arXiv
  ID and produced a full 31-requirement analysis/spec/plan/verification package.

## 2026-08-03 05:40-06:15 KST

- A fresh release audit reproduced an Android-only semantic failure: the
  existing `std::isnan` defense was optimized away by the repository's
  translation-unit-wide `-O3 -ffast-math` flags.
- Replaced that dependency with an explicitly checked 32-bit IEEE-754 object
  representation classifier. The normal and `-O3 -ffast-math` standalone
  binaries now both pass 23/23 tests, and strict Clang compilation still
  passes.
- Hardened the reusable skill scaffold against concurrent partial publication
  with a bounded exclusive lock around identity validation and all document
  creation. Sequential, mismatch, and two-process concurrent tests passed;
  the official skill validator reports `Skill is valid!`, with no generated
  cache files left in the package.
- Fresh-context re-audits approved both deliverables: the SnapKV diff has no
  remaining actionable P0/P1/P2 issue and is release-ready within its stated
  CPU v1 scope; the skill package has no remaining actionable issue.
