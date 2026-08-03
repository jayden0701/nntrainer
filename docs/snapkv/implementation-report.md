# SnapKV CPU implementation report

Date: 2026-08-03 KST
Branch: `codex/snapkv-cpu-eviction` (created from `main`)
Status: implementation and focused static verification complete; no commit
created

## Delivered scope

This change adds opt-in, one-shot SnapKV eviction to the CausalLM CPU external
KV-cache path. A complete initial causal/global prefill is evaluated normally.
After its output is produced, each eligible MHA layer compacts its prompt cache
to a configured physical capacity. Later decode calls append to that compacted
cache while RoPE and model bookkeeping keep the original logical positions.

The implementation is disabled unless `nntr_config.json` contains:

```json
{
  "snapkv": {
    "cache_capacity": 2048,
    "observation_window": 32,
    "pooling_kernel": 5,
    "pooling": "max"
  }
}
```

`cache_capacity` is required. The remaining values above are defaults.
`pooling` also accepts `avg` and `average`.

## Evidence boundary and selected semantics

- The paper and author code define the MHA one-shot algorithm: score prefix
  positions with the last observation queries' causal-softmax probabilities,
  pool, select per head, gather K/V, and append the recent window.
- The author implementation expands GQA K/V to query-head width. nntrainer's
  compiled cache instead has fixed native KV-head width.
- The selected nntrainer adaptation preserves that width and uses:

  ```text
  per-query observation sum
    -> per-query stride-1 same-length pool
    -> group sum for query heads sharing a KV head
    -> deterministic per-KV-head top-k
  ```

  Group sum and mean have identical rankings for equal-sized groups. Pooling
  before group reduction follows NVIDIA KVPress. The order matters for max
  pooling and is frozen by a non-commutative regression test.
- Prefix selections remain in descending score order, with lower original
  index as the deterministic tie break. The observation window remains
  chronological. K and V use identical indices and overlap-safe temporary
  buffers.

## Production changes

| Area | Result |
| --- | --- |
| Pure policy | Added checked observation scoring, pooling, GQA reduction, top-k, K/V compaction, and cursor-state helpers in `snapkv_policy.{h,cpp}` |
| MHA lifecycle | Compacts only after the full prefill output; splits logical RoPE position from physical cache cursor; global layers only |
| Configuration | Strict positive-integer parsing, defaults, validation, and post-compile propagation to `mha_core` layers |
| Unsupported combinations | Rejects internal cache, non-causal attention, sinks, skip-prefill, invalid GQA, and precomputed cache save/load |
| Build wiring | Added policy source to Meson and both Android source lists; added a focused Meson GTest target |
| User documentation | Added configuration, behavior, and limitations to the CausalLM README |

## Verification completed

### Focused policy tests

The standalone MinGW GTest binary compiled and passed **23/23** tests both with
ordinary flags and with Android-equivalent `-O3 -ffast-math`. Coverage
includes:

- hand-computed and 128-geometry independent scalar oracles for MHA/GQA and
  max/average pooling;
- the GQA max-pool operation-order counterexample;
- exact pooling edges, kernel one, a kernel of `UINT_MAX`, ties, NaN, and
  malformed geometry;
- logical/physical cursor transitions, reset, unsupported reposition, forged
  state, and overflow;
- per-head selection, FP32 and two-byte byte-exact cache copying, source/dest
  overlap, unchanged tail canaries, independent batches, transactionality, and
  byte-geometry overflow.

Command used:

```text
g++ -std=c++17 -Wall -Wextra -Werror \
  test/unittest/layers/unittest_snapkv_policy.cpp \
  Applications/CausalLM/layers/snapkv_policy.cpp \
  subprojects/googletest/googletest/src/gtest-all.cc \
  subprojects/googletest/googletest/src/gtest_main.cc \
  -IApplications/CausalLM/layers \
  -Isubprojects/googletest/googletest/include \
  -Isubprojects/googletest/googletest \
  -o tmp/unittest_snapkv_policy.exe
```

### Static compilation and formatting

- `clang-format-14 --dry-run --Werror`: passed for every changed C/C++ file.
- The focused suite also passed **23/23** with `-O3 -ffast-math`. The policy
  classifies NaN through its checked 32-bit IEEE-754 object representation, so
  Android's finite-math assumptions cannot optimize away the defensive rule.
- Policy and test syntax passed Clang C++17 with
  `-Wall -Wextra -Werror -pedantic -Wconversion -Wsign-conversion -Wshadow`.
- A targeted Clang-Tidy 14 pass over the pure policy and tests using
  `bugprone-*`, `performance-*`, and `portability-*` produced no user-code
  diagnostics after excluding the advisory easily-swappable-parameters check
  for the intentionally geometry-heavy API.
- Existing MinGW compile commands were replayed successfully for
  `mha_core.cpp`, `transformer.cpp`, and `causal_lm.cpp` after adding
  `-DM_PI=3.14159265358979323846` and `-Wno-error=attributes` for two
  pre-existing Windows-environment issues. Only the pre-existing `WIN_EXPORT`
  placement warnings remained.
- Build-source lists were inspected for Meson and both Android targets.
- `git diff --check` passed and `git status --short subprojects` was empty.
- Independent policy, integration, specification-to-code, oracle, and final
  safety reviews found no open P0/P1 defect in the supported path after their
  findings were applied. A final release audit reproduced a `-ffast-math` NaN
  defect, which was fixed and covered by the second focused build above. Its
  fresh-context re-audit found no remaining actionable P0/P1/P2 issue and gave
  a release-ready verdict within the documented v1 scope.

## Deliberate v1 limits and deferred evidence

- The first invocation must contain the complete initial prompt. There is no
  end-of-prompt marker, so chunked prefill is unsupported.
- All batch items must have the same fully valid prompt length. Padded, packed,
  and variable-length prompt batches are unsupported because the score seam
  has no per-sample valid length or padding mask.
- Finite sliding-window layers remain unchanged. An all-sliding model accepts
  the shared configuration but has no eligible layer and therefore performs
  no compaction.
- The external backing slab remains `max_seq_len` sized. SnapKV reduces the
  active physical/read length and decode attention work, not peak allocated
  cache memory or peak prefill memory.
- Arbitrary rewind/forward-skip after compaction is rejected; only sequential
  append or reset to logical zero is supported.
- Precomputed cache save/load is rejected because v1 has no serialized
  selected-position/cursor representation.
- Configuration parsing now completes before `ids_history` allocation; the
  allocation uses checked byte geometry, checks `malloc`, and safely replaces a
  prior construction-time buffer. This closes the review-found rejection-path
  leak associated with the new SnapKV validation.
- A full Meson/Ninja build, real model generation comparison, sanitizer run,
  and FP16 runtime test were unavailable in this environment. Run later:

  ```text
  meson setup build-snapkv -Denable-transformer=true -Denable-test=true
  ninja -C build-snapkv
  meson test -C build-snapkv unittest_snapkv_policy --print-errorlogs
  meson test -C build-snapkv --print-errorlogs
  ```

## Persistent review artifacts

- `paper-analysis.md`: complete primary-source reading and ambiguity ledger.
- `reference-implementations.md`: pinned author and independent ports.
- `nntrainer-integration-analysis.md`: target execution/data ownership map.
- `implementation-spec.md`: normative target contract.
- `static-verification-plan.md`: risk-driven evidence plan.
- `progress-log.md`: chronological decisions and checks.
- `policy-review.md`, `mha-integration-review.md`, `oracle-audit.md`,
  `spec-code-audit.md`, and `final-code-audit.md`: independent review records.
