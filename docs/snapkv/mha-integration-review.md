# SnapKV MHA integration independent static review

Review date: 2026-08-03
Scope: `mha_core.{h,cpp}`, `snapkv_policy.{h,cpp}`, CPU cache layout,
Meson/Android source wiring, and the directly adjacent Transformer/CausalLM KV
cursor lifecycle.
Method: source inspection, invariant tracing, strict standalone compilation,
and build-file/source-list audit. The full repository build and model runtime
were unavailable, as stated in the task.

## Verdict

There is no remaining P0/P1 defect in the reviewed snapshot. The core path is
internally coherent:

- a long first global causal prefill computes its full output before eviction;
- the last observation queries score only the non-observation prefix;
- GQA query heads are aggregated into their shared KV head;
- K and V use the same per-head gather indices through overlap-safe temporary
  buffers;
- subsequent cache access uses a physical cursor while RoPE keeps the absolute
  logical cursor;
- finite sliding-window layers remain uncompressed;
- malformed configuration, unsupported sink/skip-prefill combinations, and
  precomputed-cache save/load are rejected.

One P2 integration boundary and one P2 verification gap remain. Neither is on
the currently exercised CausalLM full-tensor binding path, but both should stay
visible in the handoff.

## Open findings

### [P2] `KVCacheManager` positional view API cannot represent per-layer SnapKV cursors

Evidence:

- `KVCacheManager::getPosition()` is documented as the cache's current write
  position, and its write/read-view APIs use a single manager-wide physical
  row (`Applications/CausalLM/kv_cache_manager.h:93-108,129-174`;
  implementations at `kv_cache_manager.cpp:106-187`).
- CausalLM passes the absolute logical position to both the manager and every
  MHA (`models/causal_lm.cpp:214-227`).
- Each MHA now owns its own logical-to-physical offset
  (`layers/mha_core.cpp:138-146,481-488`). A global-attention layer can be at
  physical `C+t`, while a finite sliding-window layer deliberately remains at
  logical/physical `L+t` (`mha_core.cpp:166-171`).

No production call site currently uses `get*WriteView()` or `get*ReadView()`;
CausalLM binds each full backing tensor and lets `MHACoreLayer` calculate its
per-layer row. Thus the implemented path is safe. However, the manager's
single `cache_pos_` cannot be reused as a physical cursor when SnapKV is on,
especially in a hybrid global/sliding model.

Recommendation for v1: document that the manager's positional view methods
are incompatible with SnapKV and keep them out of this path. A future API
should either accept a layer-specific physical cursor/offset or make manager
position explicitly logical and require the caller to map it. The direct
save/load methods are already guarded while SnapKV is enabled
(`models/causal_lm.cpp:290-309`).

### [P2 verification gap] The pure policy is covered, but the real layer/Tensor seam is not executable here

The unit suite exercises pooling, deterministic selection, GQA aggregation,
byte-exact K/V compaction, overflow guards, and the same cursor mapper used by
MHA (`test/unittest/layers/unittest_snapkv_policy.cpp`). Production FP32 and
FP16 score aggregation now route through the tested policy implementation
(`mha_core.cpp:908-928`).

Still unexecuted in a layer-level test are:

- the `shouldCompactSnapKV()` gating decision (`mha_core.cpp:166-171`);
- full-prefill output-before-compaction ordering (`mha_core.cpp:869-879`);
- `Tensor::getData()` plus batch-byte offsets used for an actual external
  cache (`mha_core.cpp:941-956`);
- two-batch independence through `MHACoreLayer`, rather than two isolated raw
  buffers;
- a hybrid model proving a global layer maps to `C` while a sliding layer stays
  at `L`.

Recommendation: when a runnable nntrainer environment is available, add one
small external-cache `MHACoreLayer` test with `B=2`, `Hq=4`, `Hkv=2`, `L=6`,
`W=2`, `C=4`, followed by one decode step. Assert unchanged prefill output,
per-batch cache contents, physical cursor `4 -> 5`, offset `2`, and logical
RoPE position `6`. Add a second finite-window layer to prove no compaction.

## Findings corrected during review

These were present in an earlier reviewed snapshot and were corrected before
this report was finalized.

| Severity | Finding | Resolution evidence |
|---|---|---|
| P1 | After eviction, physical bounds could pass while logical RoPE position exceeded its table, permitting an out-of-range `from+h` lookup. | Logical end is now checked before compute against both `max_timestep` and `max_position_embeddings` (`mha_core.cpp:367-380`). |
| P1 build | The policy-only test linked against a DLL whose policy symbols had no Windows export decoration. | The test now compiles `snapkv_policy.cpp` directly and links only GTest (`Applications/CausalLM/meson.build:248-261`). |
| P2 | Pooling used signed `int` deltas; `UINT_MAX` odd kernel could overflow at loop termination. | Pool bounds now use checked `size_t` interval endpoints; a very-large-kernel test was added (`snapkv_policy.cpp:171-210`; unit test around lines 172-184 after formatting). |
| P2 | GQA divisibility was checked only after K/V kernels had already consumed a truncated group size. | SnapKV-enabled layers now reject `Hq % Hkv != 0` at finalize and post-compile property application (`mha_core.cpp:270-284,1743-1760`). |
| P2 | The actual MHA duplicated score aggregation, so the tested policy was not the production FP32/FP16 path. | MHA now calls the shared `observationScores()` / `observationScoresTyped()`, `poolScores()`, and `aggregateGQAScores()` stages directly. |
| P2 | JSON numeric fields could be lossy-cast from negative/floating values to unsigned. | A positive-integer/range parser now validates all SnapKV numeric fields (`models/transformer.cpp:153-218`). |
| P2 | Rejecting JSON precomputed-cache mode did not protect direct later `save_kvcache()` / `load_kvcache()` calls. | Both methods now throw while SnapKV is enabled (`models/causal_lm.cpp:290-309`). |
| P3 | Raw compaction byte spans and additions were only partially overflow-checked. | Full allocation/prompt/compacted spans and all vector/observation endpoints now use checked multiply/add (`snapkv_policy.cpp:281-354`). |
| P3 | Public `setCacheIndex()` documentation still described a physical argument. | Header now calls the argument an absolute logical position and labels `getCacheIndex()` physical (`mha_core.h:364-375`). |
| P3 build | New `std::numeric_limits` use depended on transitive headers, and strict iterator signedness warned. | Direct `<limits>` includes were added and pointer destinations remove the iterator difference conversion. |

## Invariant trace

### First long prefill

For the eligible first step, `logical_from == physical_from == 0` and
`step_size=L>C` (`mha_core.cpp:166-171`). K/V writes and attention use the full
physical range `[0,L)`. `softmax_triangle()` finishes before the V-weighted
output, and compaction is called only after that output is complete
(`mha_core.cpp:869-879`). Therefore the first generated logits still reflect
the uncompressed prompt.

The policy reads causal triangular row
`q*(q+1)/2 + key`, for `q in [L-W,L)` and `key in [0,L-W)`, groups contiguous
query heads by `query_head / (Hq/Hkv)`, pools, and top-k selects. These layouts
match the CPU kernel's score layout `[attention_row][query_head]` and cache
layout `[token][kv_head][head_dim]`.

### Compaction and decode

`compactCache()` gathers each KV head independently into the first `C-W` rows
and copies the final `W` whole-token rows afterward. K and V are copied through
separate temporary buffers, so neither cross-head source order nor overlapping
destination rows can clobber a later source (`snapkv_policy.cpp:252-354`).

After success, the shared state helper transitions from `(logical=L,
physical=L, offset=0)` conceptually to `(L,C,L-C)` and marks the one-shot
eviction complete (`snapkv_policy.cpp:101-135`; MHA application at
`mha_core.cpp:481-488`). On decode, `setCacheIndex(L)` maps to physical `C`, K/Q
RoPE receives logical `_from=L` (`mha_core.cpp:818-840`), while cache read
length, causal softmax, and V multiplication receive physical `cache_from=C`
(`mha_core.cpp:844-873`). This is the required split.

### GQA and per-head mixed rows

The CPU kernels lay query heads out as `kv_head * group_size + group_member`.
The policy uses the same contiguous grouping. Summing instead of averaging is
ranking-equivalent because every KV head has the same group size. A physical
row may consequently contain different original token positions for different
KV heads; this remains correct because each head's K/V pair is permuted
together and future global attention consumes all retained rows.

## Static verification performed

- `snapkv_policy.cpp`: Clang C++17 syntax check with
  `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror`:
  passed.
- `unittest_snapkv_policy.cpp`: the same strict syntax check with bundled
  GTest headers: passed.
- `snapkv_policy.cpp`: Clang static analyzer: no diagnostics.
- `mha_core.cpp`, `transformer.cpp`, and `causal_lm.cpp`: compiled separately
  with the repository's MinGW command from `build-api-fix/compile_commands.json`:
  passed after supplying `M_PI` and demoting the repository's pre-existing
  misplaced `WIN_EXPORT` attribute warning. Those two workarounds are baseline
  issues, not introduced by SnapKV.
- Meson source audit: policy is in the MHA shared library; the test owns its
  policy translation unit; no Windows-only export is required.
- Android audit: policy source appears in both relevant `LOCAL_SRC_FILES`
  lists (`Applications/CausalLM/jni/Android.mk:89,207`).
- No edits were made under `subprojects/`.

This is strong static evidence, but it is not a substitute for the layer-level
runtime test described above or a full cross-platform link/build.
