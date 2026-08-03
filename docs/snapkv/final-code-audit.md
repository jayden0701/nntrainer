# SnapKV final independent code audit

Audit date: 2026-08-03 (Asia/Seoul)

Scope:

- `Applications/CausalLM/layers/mha_core.{h,cpp}`
- `Applications/CausalLM/layers/snapkv_policy.{h,cpp}`
- Transformer/CausalLM configuration and cache-position propagation
- Meson and Android source wiring
- `test/unittest/layers/unittest_snapkv_policy.cpp`
- the normative implementation specification and adjacent cache-manager APIs

This audit did not modify production code. Defects and semantic conflicts found
while the audit was running were reported to the implementing agent and
corrected before the latest re-read; they are recorded in the closed-findings
section.

## Verdict

No open P0/P1 defect was found in the supported CausalLM path after those
corrections. For a dense, common-length batch whose initial prompt is supplied
in one call, the implementation is internally coherent:

- full-prefill output is computed before any cache bytes are moved;
- observation scores use the final `W` causal-softmax rows and prefix columns;
- each query head is pooled independently, then pooled vectors are reduced into
  their native shared KV head for GQA/MQA;
- K and V use the same per-head indices through overlap-safe temporary buffers;
- every batch is compacted at its own byte offset;
- global layers use a physical cursor after eviction while RoPE retains the
  absolute logical cursor;
- finite sliding-window layers keep their original logical/physical cursor;
- the legacy internal-cache path keeps logical and physical positions equal and
  clears any SnapKV-only offset state;
- reset to logical position zero clears the one-shot compaction state;
- FP32, FP16, and the non-FP16-build two-byte cache representation are copied
  byte-exactly rather than numerically converted.

The remaining findings are unsupported-caller boundaries, an adjacent manager
API limitation, and verification gaps. They do not make the
currently documented CausalLM dense single-call path incorrect, but they must
remain explicit until runtime integration tests are available.

Post-fix release re-audit: after the Android fast-math correction, a fresh
review reran both focused binaries, strict Clang diagnostics, clang-format 14,
and diff checks, and rechecked cursor/GQA/build wiring. It found no remaining
actionable P0/P1/P2 issue and judged the change release-ready within the
documented v1 boundary.

Severity convention: P0 is immediate memory corruption/security impact; P1 is
a supported-path correctness/build blocker; P2 is a material unsupported
boundary or missing verification; P3 is defensive hardening or usability.

## Open findings

### [P2 boundary] A layer cannot identify the final chunk of a chunked prefill

Evidence:

- `MHACoreLayer::shouldCompactSnapKV()` infers a complete prompt solely from
  `logical_from == 0`, physical row zero, and `step_size > C`
  (`Applications/CausalLM/layers/mha_core.cpp`, around lines 166-171).
- No prompt-final flag or explicit finalize/compress call reaches the layer.

If an external caller presents a prompt in chunks and the first chunk is longer
than `C`, that first chunk is compacted prematurely. Later prompt chunks are
then appended as continuation tokens and cannot influence the original SnapKV
selection. If the first chunk is at most `C`, the one-shot `logical_from == 0`
gate means a later chunk never triggers compression.

The current CausalLM call supplies the whole prefill range in one
`incremental_inference()` call (`models/causal_lm.cpp`, around lines 598-601),
so its supported path is safe. V1 should continue to state that chunked prefill
is unsupported. Generic support needs an explicit final-prefill signal.

### [P2 boundary] Variable-length padded or packed batches have no valid-length input

Evidence:

- `compactSnapKV()` receives one common `prompt_length` for every batch
  (`mha_core.cpp`, around lines 883-958).
- `observationScoresTyped()` accepts no attention/padding mask or per-sample
  valid length (`snapkv_policy.h`, around lines 96-157).

The byte offset is correctly batch-specific, but the final `W` tensor rows are
treated as real observation tokens for every sample. A batch containing
different valid prompt lengths would score padding and could retain padded
prefix positions. Current CausalLM uses a common dense length, so this is an
API-scope limitation rather than a demonstrated production-path error. General
batched support requires per-sample lengths and the effective attention mask.

### [P2 integration] `KVCacheManager` positional views cannot represent mixed layer cursors

Evidence:

- `KVCacheManager` owns one `cache_pos_`, exposed as a current write position,
  and its write/read view APIs are based on that one position
  (`Applications/CausalLM/kv_cache_manager.h:93-108,129-174`).
- CausalLM sets that manager position to the absolute logical position and also
  forwards the same logical position to every MHA
  (`models/causal_lm.cpp:214-227`).
- After SnapKV, a global layer can be at physical `C+t`, while a finite-window
  layer remains at logical/physical `L+t` (`mha_core.cpp`,
  `shouldCompactSnapKV()` and `advanceCachePosition()` application around
  lines 166-171 and 478-486).

No production SnapKV call uses the manager's positional view methods; CausalLM
binds each complete backing tensor and lets each MHA choose its row. The current
path is therefore safe. The view API must not be introduced into this path
without a per-layer physical cursor/offset.

### [P2 verification] The real MHA/Tensor seam remains unexecuted

The pure policy tests cover aggregation, both pooling modes, deterministic
top-k, GQA grouping, cursor transitions, batch-separated raw buffers, byte
compaction, and overflow rejection. They do not instantiate a compiled
`MHACoreLayer` with external `Tensor` inputs. Consequently the following still
depend on static reasoning:

- `shouldCompactSnapKV()` gating;
- full-output-before-compaction ordering;
- actual `Tensor::getData()` batch offsets;
- FP16 production instantiation;
- a two-batch global layer followed by decode;
- a mixed global/sliding model proving physical `C` versus logical `L`.

When a runnable environment is available, the minimum integration fixture
should use `B=2, Hq=4, Hkv=2, L=6, W=2, C=4`, check the unchanged prefill
output/cache bytes, and execute one decode token (`physical 4 -> 5`, logical
`6 -> 7`, offset 2). Add a finite-window companion layer.

### [P3 error-path hygiene] New incompatibility checks occur after a raw allocation

`CausalLM::setupParameters()` allocates `ids_history` with `malloc` around
lines 82-84, then rejects SnapKV with `skip_prefill` or precomputed-cache mode
around lines 112-117. If construction throws there, the derived destructor does
not run and the allocation can leak. Older exception paths already share this
pattern, but the new validation need not add to it. Prefer validation before the
allocation or an RAII container.

Post-audit resolution: configuration parsing and SnapKV incompatibility checks
now complete before allocation. The final allocation checks multiplication,
handles `malloc` failure, and safely replaces a prior setup-time buffer.

## Findings corrected during this audit

### [Closed P2] Android `-ffast-math` defeated NaN classification

A later release audit compiled the focused policy suite with the same
`-O3 -ffast-math` flags applied by the Android makefile. The former
`std::isnan` checks were optimized under finite-math assumptions, causing both
NaN tests to fail. The policy now classifies NaN through the checked 32-bit
IEEE-754 object representation. Both the ordinary and fast-math binaries pass
23/23 tests; the strict Clang compile remains clean.

### [Closed P1] Invalid downcast from `LayerNode` to `MHACoreLayer`

The first reviewed snapshot used
`static_cast<MHACoreLayer &>(layer).isSnapKVCompactionTarget()` inside
`Transformer::initialize()`. `NeuralNetwork::forEachLayer()` actually passes a
`LayerNode` as `ml::train::Layer &` (`nntrainer/models/neuralnet.cpp:2267-2273`);
only its `getType()` and `setProperty()` delegate to the contained layer. The
downcast therefore had undefined behavior. The latest code removed the cast
and target getter and uses delegated `getType()`/`setProperty()` only.

### [Closed P2] Arbitrary logical reposition after compaction

The first reviewed cursor mapper accepted any logical position at or above the
offset. That is invalid because physical prefix rows are score-ranked per head,
not a chronological logical interval. The latest
`SnapKVPolicy::mapLogicalPosition()` accepts only the current logical append
cursor or zero once compacted (`snapkv_policy.cpp`, around lines 82-105), and a
unit test rejects both rewind and forward-skip positions.

### [Closed P1 semantic conflict] GQA max-pooling operation order

An intermediate implementation reduced query-head scores into their shared KV
head before pooling. Reduction and max pooling do not commute, and that order
conflicted with the paper-analysis decision and the native-GQA precedent. The
latest policy and production path now perform:

```text
per-query observation sum -> per-query pool -> GQA group sum -> per-KV top-k
```

`observationScores{Typed}()` returns `[Hq,P]`, `poolScores()` retains `[Hq,P]`,
and `aggregateGQAScores()` validates divisibility and returns `[Hkv,P]`
(`snapkv_policy.h:80-182`; `mha_core.cpp`, around lines 905-937). Group sum is
ranking-equivalent to group mean because all groups have equal size. A max-pool
non-commutativity regression now distinguishes the two orders, and the
normative specification uses the same sequence.

### [Closed P3] Forged position-state flag/offset combinations

The shared `validatePositionState()` now requires both
`logical - physical == offset` and
`has_compacted == (offset != 0)` (`snapkv_policy.cpp:45-52`). Tests reject both
false-with-nonzero-offset and true-with-zero-offset states.

### [Closed P3] Reserved include guard

The policy header now uses the project-scoped
`CAUSALLM_SNAPKV_POLICY_H_` guard (`snapkv_policy.h:11-12,212`).

### [Closed P3 documentation] All-sliding models deliberately accept a no-op

`Transformer::initialize()` cannot determine through its `LayerNode` wrapper
whether any `mha_core` layer is global, and finite sliding-window layers
deliberately skip compaction. The product choice is now explicit: enabling
SnapKV on an all-sliding model is accepted as a no-op
(`Applications/CausalLM/README.md`; `docs/snapkv/implementation-spec.md`).

## Memory, arithmetic, and state audit

No unchecked source span was found in the raw compaction helper. It uses
checked multiplication/addition for token widths, full/prompt/compacted spans,
per-head source/destination endpoints, and the observation block before either
`memcpy` (`snapkv_policy.cpp`, around lines 260-362). Temporary destination
buffers prevent overlap corruption. K and V are gathered from identical
indices, and observation rows are copied only after all selected per-head
vectors have been staged.

The external MHA path checks logical end against both `max_timestep` and the
RoPE position table, checks physical addition overflow, and checks the physical
write against both cache heights before constructing step views
(`mha_core.cpp`, around lines 365-405). State advances only after all batches
finish. A thrown later-batch compaction can leave earlier cache bytes modified,
but inference has failed and the documented recovery is a logical-zero reset
and new prefill; no continuing success path observes a half-committed state.

For a successful long prefill the transition is:

```text
(logical=0, physical=0, offset=0, compacted=false)
  -- full prefill L and compact -->
(logical=L, physical=C, offset=L-C, compacted=true)
```

Every later successful step advances logical and physical positions by the
same amount and keeps the offset constant. A zero position returns all four
fields to their default state.

## Build and platform audit

- Meson compiles `snapkv_policy.cpp` into `mha_core_layer`; the policy unit test
  compiles its own copy and therefore does not require unexported policy symbols
  from the Windows DLL (`layers/meson.build` and
  `Applications/CausalLM/meson.build`, SnapKV additions).
- Both Android `LOCAL_SRC_FILES` lists containing `mha_core.cpp` also contain
  `snapkv_policy.cpp` (`jni/Android.mk:88-89,206-207`).
- Repository search found no other non-vendored explicit `mha_core.cpp` source
  list requiring an update.
- The new policy uses portable C++17/STL only. FP16-specific code remains under
  `ENABLE_FP16`; raw cache compaction is element-size based.
- No file under `subprojects/` was changed.

## Static validation record

Performed against the moving working tree during this audit:

| Check | Result |
|---|---|
| `snapkv_policy.cpp`, Clang C++17 with `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror` | Pass |
| `unittest_snapkv_policy.cpp`, the same strict Clang syntax-only flags | Pass |
| Standalone MinGW C++17 policy test executable | **23/23 tests pass** |
| New policy header/source/test, `clang-format-14 --dry-run --Werror` | Pass |
| Existing C/C++ files, clang-format 14 on changed lines | Pass |
| `mha_core.cpp`, `transformer.cpp`, `causal_lm.cpp`, existing MinGW compile-database commands | Pass after defining `M_PI` and demoting the repository's pre-existing misplaced `WIN_EXPORT` attribute warning |
| `git diff --check` | Pass |
| Meson/Android explicit-source audit | Pass |
| Full model build and layer-level runtime test | Unavailable/deferred |

The `M_PI` and `WIN_EXPORT` workarounds are baseline Windows issues visible in
unchanged headers/source and are not introduced by SnapKV.

The standalone test was linked directly from the policy, its test source, and
the repository's GoogleTest sources; it does not claim to exercise the real
`MHACoreLayer`/`Tensor` integration seam. A UBSan rerun was attempted, but the
installed MinGW toolchain has no `libubsan`, so no sanitizer result is claimed.
