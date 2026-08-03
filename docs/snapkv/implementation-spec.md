# SnapKV CPU implementation specification

Status: implemented CPU v1 baseline, 2026-08-03

## 1. Goal

Add opt-in, one-shot prompt KV eviction to the CPU CausalLM attention path.
The implementation must preserve the existing behavior byte-for-byte when the
feature is disabled. It must compress a long initial prompt after its prefill
attention output has been computed, then make later decode steps attend to the
compressed physical cache while retaining the original logical RoPE positions.

This first version targets the external KV-cache path used by
`Applications/CausalLM`. It is not a generic training feature and does not
change model weights.

## 2. Sources and precedence

1. Local source artifact: `SnapKV.pdf`, especially pp. 4-8 and Listing 1.
2. Final NeurIPS paper: <https://papers.nips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf>.
3. Author implementation, pinned at commit
   `e216ddc84c5bd210378cbdbbba12ba02102aa640`:
   <https://github.com/FasterDecoding/SnapKV/tree/e216ddc84c5bd210378cbdbbba12ba02102aa640>.
4. NVIDIA KVPress, pinned at commit
   `8bb3315aa552d2d0b33f38ef0835e68cfa49a11a`, for its native-KV-head GQA
   ordering:
   <https://github.com/NVIDIA/kvpress/tree/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a>.

The paper defines the algorithm. The author code resolves tensor-layout and
masking details. Any nntrainer-specific adaptation must be called out below and
covered by tests.

## 3. Terminology

- `L`: logical prompt length.
- `W`: observation-window length.
- `C`: configured post-prefill cache capacity.
- `P = L - W`: prefix length.
- `K = C - W`: number of retained prefix positions per KV head.
- `Hq`: query-head count.
- `Hkv`: key/value-head count.
- `G = Hq / Hkv`: GQA group size.
- logical position: original token position used for RoPE and generation
  bookkeeping.
- physical position: row in the compacted KV tensor.

## 4. Normative algorithm

After the configuration constraints in section 7 have been validated, apply
eviction only when all of the following runtime eligibility conditions hold:

- SnapKV is configured with `C > 0`;
- this is the first prompt prefill (`logical_from == 0`);
- `L > C`;
- attention is global (`local_window_size == UINT_MAX`);
- prefill attention was actually computed.

For a valid configuration, a short prompt, a later call, or a finite
sliding-window layer leaves the cache and positions unchanged. Non-causal,
internal-cache, skip-prefill, and attention-sink combinations are invalid
configurations and fail rather than silently skipping.

For every batch and KV head:

1. Use the already softmax-normalized causal attention probabilities produced
   for the last `W` prompt queries.
2. Ignore attention to the observation window itself when scoring. Sum only
   probabilities targeting prefix positions `[0, P)`, independently for every
   query head.
3. Apply stride-1, same-length 1-D pooling to each query-head score vector:
   - `max`: pad by negative infinity;
   - `avg`: pad by zero and divide by the full kernel width, matching PyTorch
     `avg_pool1d(..., count_include_pad=True)`.
   For defensive handling of corrupted/non-finite inputs, treat NaN as below
   every finite score: max pooling ignores NaN samples, while an average-pool
   window containing NaN receives negative infinity. This extension does not
   change finite-input parity with the reference implementation. An all-NaN
   max window therefore yields negative infinity. If group summation later
   produces NaN (for example from mixed infinities), top-k ranks it as negative
   infinity; equal negative-infinity ranks use the normal lower-index tie rule.
   The CPU implementation requires 32-bit IEC 559/IEEE-754 `float` and
   classifies NaN from its object representation so this rule remains valid
   under Android's translation-unit-wide `-ffast-math` flags.
4. Under MHA (`Hq == Hkv`), each pooled query-head vector directly scores its
   matching KV head. Under GQA/MQA, sum the pooled vectors of all `G` query
   heads sharing the KV head. Division by `G` is unnecessary because it does
   not change the per-head ordering.
5. Select the `K` greatest pooled scores independently for every KV head.
   Make ties deterministic by preferring the smaller original prefix index.
6. Gather matching K and V vectors using the same indices. Preserve the
   selected positions in descending score order, then append the original
   observation-window K/V rows in chronological order.
7. Write the result through a temporary buffer before copying it back. Direct
   in-place gather is forbidden because one head's destination can overwrite a
   later source.

The resulting physical cache length is exactly `C` for every compressed layer.

## 5. GQA adaptation

The author implementation calls `repeat_kv` before selection. It therefore
converts a GQA cache to query-head-width storage and can retain a different
position set for every query head.

nntrainer stores `[token][Hkv * head_dim]` and its CPU kernels intentionally
share each K/V head across `G` query heads. Dynamically changing that width is
not compatible with the compiled graph. This implementation therefore uses a
shared selection per KV head. It first pools every query head independently,
then aggregates the pooled scores within each KV group, matching the ordering
used by NVIDIA KVPress. This order is normative: group reduction and max
pooling do not commute.

Consequences:

- MHA behavior follows SnapKV head-for-head.
- GQA/MQA preserves nntrainer's cache layout and memory advantage.
- GQA/MQA selection is an explicit adaptation, not bit-equivalent to the
  author's `repeat_kv` implementation.
- Tests must include different preferred positions in query heads from the
  same group and prove that pooling happens before group reduction. The
  regression vector `q0=[10,0,0]`, `q1=[0,0,10]`, max kernel 3 must produce
  `[10,20,10]` and select position 1; reducing first would tie all positions.

## 6. Pooling and defaults

The local paper and final paper evaluate max pooling, while the current author
repository exposes both modes and defaults its helper to average pooling.
Support both values. Use these nntrainer defaults when a `snapkv` object is
present:

- `observation_window`: 32
- `pooling_kernel`: 5
- `pooling`: `max`

Require an odd, positive pooling kernel so same-length output and valid gather
indices are guaranteed. Require `C > W > 0`.

### 6.1 Intentional reference departures

- Native GQA retains `Hkv` storage and pools each query head before group
  reduction instead of storing the author's repeated `Hq` K/V cache.
- Equal scores prefer the lower original position; PyTorch `topk` does not
  promise this tie order.
- NaN handling follows the deterministic rule in section 4 rather than
  backend-specific propagation; both ordinary and `-O3 -ffast-math` focused
  test binaries exercise that rule.
- `L == C` is a byte-exact no-op rather than a gather/reorder of every prefix
  row.
- FP16 attention values are converted to FP32 before observation accumulation;
  the author helper casts softmax back to query dtype before its reduction, so
  near-tied FP16 boundaries are not claimed bit-identical.

## 7. Configuration contract

SnapKV is disabled when `nntr_config.json` has no `snapkv` object.

Example:

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

Reject configuration when:

- `cache_capacity == 0`;
- `observation_window == 0`;
- `cache_capacity <= observation_window`;
- `pooling_kernel == 0` or is even;
- pooling is none of `max`, `avg`, or `average`;
- `cache_capacity` exceeds `max_seq_len` or an MHA layer's `max_timestep`;
- a configured MHA layer is non-causal, uses internal cache, skip-prefill, or
  an attention sink;
- `Hq` is not divisible by `Hkv`;
- the compiled model contains no `mha_core` layer.

All numeric fields must be positive JSON integers representable by
`unsigned int`; negative numbers, fractional values, and oversized integers
are rejected before model compilation. `average` is accepted as an alias for
`avg`.

Layers with finite sliding-window attention are intentionally left unchanged.

## 8. Logical versus physical positions

Eviction must not renumber RoPE positions. The keys selected from the prompt
already contain rotations for their original positions. New K/Q tensors must
continue using the original logical position.

After compression:

```text
cache_position_offset = logical_prompt_end - C
physical_from          = logical_from - cache_position_offset
rope_position          = logical_from
```

All cache writes, cache read lengths, causal-softmax row lengths, and V-cache
matmuls use physical positions. Rotary embedding uses logical positions.

After compaction, the layer accepts only its current logical append position or
zero. Arbitrary rewind/reposition is invalid because selected physical prefix
rows no longer correspond to a contiguous logical prefix. Setting zero resets
the one-shot compression state and offset.

## 9. Cache ordering invariant

For physical rows `[0, K)`, each KV head may refer to a different original
prefix position. This is valid because:

- K and V are gathered with identical per-head indices;
- RoPE is already embedded in K;
- a decode query can attend to every retained past row;
- attention and the weighted V sum are invariant to a paired K/V permutation.

This invariant does not hold for finite sliding-window attention, which is why
such layers are excluded.

## 10. Lifecycle and scope limits

- Compress at most once after the first full prompt prefill.
- Decode-time eviction is out of scope; newly generated tokens append after
  the compressed cache.
- A later multi-turn prefill appends to the already compressed cache and is not
  recompressed in v1.
- The first invocation is treated as the complete prompt. Chunked prefill is
  unsupported because the current API has no explicit end-of-prompt signal;
  the caller must not split the initial prompt across calls.
- Every batch item must have the same fully valid prompt length. Padded,
  packed, or variable-length batches are unsupported because scoring receives
  neither per-sample valid lengths nor a padding mask.
- An all-sliding-window model accepts the common configuration but performs no
  compaction because no eligible global-attention layer exists.
- Precomputed cache save/load combined with SnapKV is out of scope in v1.
- `KVCacheManager` positional read/write-view APIs are not used while SnapKV is
  active. A single manager cursor cannot represent different physical lengths
  for full-attention and sliding-window layers; production binds each complete
  backing tensor and lets each `mha_core` layer map its own physical cursor.
- The allocated external tensor slab is not shrunk in v1. Eviction reduces the
  active physical/read cache length and attention work while logical positions
  remain `L+t`, but peak allocated KV memory remains the configured
  `max_seq_len` slab. Physical shrinking/rebinding is a separate follow-up
  because compiled placeholder dimensions currently assume the maximum size.
- Non-CPU backends are out of scope, but code must remain portable C++ and must
  not add platform-specific APIs.

## 11. Error handling

Use `std::invalid_argument` for invalid configuration or tensor/head geometry.
Use `std::out_of_range` when a mapped physical cache access would exceed the
external tensor or when compacted-cache reposition is unsupported. Use
`std::logic_error` for an internally inconsistent cursor state and direct
save/load calls that cannot represent SnapKV state, and `std::overflow_error`
for checked size/cursor arithmetic overflow. Allocation can additionally
propagate `std::bad_alloc`. Do not silently fall back after the user explicitly
enables an unsupported combination.

## 12. Required tests

Pure policy tests:

- max-pool and average-pool edge padding;
- kernel size one;
- deterministic top-k ties;
- independent selections for multiple KV heads;
- GQA group aggregation;
- non-commutative GQA max-pooling order (pool per query head, then reduce);
- invalid geometry/configuration;
- deterministic NaN pooling and ranking behavior.

Compaction tests:

- K/V pairing is preserved;
- each KV head can gather a different source position;
- observation window is copied exactly and chronologically;
- temporary storage prevents overlap corruption;
- batches remain independent;
- FP32 and two-byte cache element layouts are copied byte-exactly.

Integration/static tests:

- disabled configuration adds no properties/behavior;
- short prompts remain unmodified;
- first long prefill yields physical length `C` and offset `L-C`;
- first decode token writes at physical `C` but uses logical RoPE position `L`;
- finite sliding-window layers do not compact;
- reset to logical position zero clears offset and one-shot state.

## 13. Acceptance criteria

- Feature is opt-in and disabled behavior is unchanged.
- Selection and pooling match a small independent scalar oracle for MHA and
  the documented GQA aggregation; PyTorch differential testing remains a
  deferred environment test when PyTorch is available.
- No edits occur under `subprojects/`.
- All changed C/C++ is formatted with clang-format 14 when available.
- In the unavailable full-build environment, validation includes focused
  standalone compilation, deterministic oracle vectors, include/dependency
  audit, build-file audit, and independent code review.
