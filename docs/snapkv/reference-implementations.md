# SnapKV reference implementations and CPU-porting notes

Research cut-off: 2026-08-03 KST. This note records the exact sources inspected,
the semantics that can safely be treated as a reference, and the places where a
literal port would be wrong for nntrainer. It is intentionally commit-pinned so
that later upstream changes do not silently change our specification.

## 1. Result in one paragraph

The final NeurIPS paper and the authors' `snapkv_utils.py` agree on the central
algorithm: during the initial prompt prefill, use the last `W` queries as an
observation window, compute their causal softmax attention over the full prompt,
sum the probabilities that target the older prefix, smooth those per-position
scores with stride-1 pooling, retain the highest-scoring `C-W` prefix positions
per head, and append the last `W` K/V rows unchanged. Compression affects the
cache stored for later decoding, not the attention output of the current
prefill. The official implementation is the best MHA reference, but it expands
GQA K/V tensors to query-head width before caching. For nntrainer's fixed
KV-head layout, the best-supported adaptation is the one independently used by
NVIDIA KVPress and now validated by KVCache-Factory: transiently score every
query head, reduce the scores of query heads that share a KV head, and gather
the unrepeated K/V tensors. Sum and mean reduction produce the same ranking for
a fixed group size; nntrainer can use sum and avoid a division.

## 2. Source ledger and trust level

| Priority | Source | Revision inspected | Why it matters |
| --- | --- | --- | --- |
| 1 | [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf), [proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html), DOI [10.52202/079017-0722](https://doi.org/10.52202/079017-0722) | Final conference version; local `SnapKV.pdf` is the 17-page artifact inspected | Normative intent, equations, Listing 1, and experimental hyperparameters |
| 1 | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV/tree/e216ddc84c5bd210378cbdbbba12ba02102aa640) | `e216ddc84c5bd210378cbdbbba12ba02102aa640`; algorithm file last changed at [`557aff1`](https://github.com/FasterDecoding/SnapKV/commit/557aff1c6d018d8d3021cb511a6080c1164083da) | Authors' Apache-2.0 implementation; tensor and mask details |
| 2 | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress/tree/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a) | `8bb3315aa552d2d0b33f38ef0835e68cfa49a11a` | Maintained Apache-2.0 Hugging Face integration; preserves KV-head GQA layout by group averaging |
| 2 | [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory/tree/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0) (formerly PyramidKV) | `94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0`; GQA path introduced in [`ad51c4e`](https://github.com/Zefan-Cai/KVCache-Factory/commit/ad51c4e0f9896dea791e6c95a654b52662cde010) | MIT implementation with both legacy query-head and memory-efficient KV-head modes, CPU unit tests, and recorded GPU validation |
| 3 | [microsoft/MInference](https://github.com/microsoft/MInference/tree/a4eb395f949ea39e871f9bc586d683390692c6be) | `a4eb395f949ea39e871f9bc586d683390692c6be` | MIT integration that explicitly credits and closely copies the author code; useful integration corroboration, not an independent numerical oracle |

License/derivation decision: the author and NVIDIA references are Apache-2.0
and KVCache-Factory/MInference are MIT at the recorded pins. The nntrainer code
is nevertheless a clean-room C++ implementation from the paper's semantics,
observed behavior, and independently written fixtures; no reference source was
copied or translated. New source files use nntrainer's Apache-2.0 SPDX header.
| Negative check | [huggingface/transformers](https://github.com/huggingface/transformers/tree/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722) | `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; complete recursive tree inspected | No SnapKV-named implementation path was present. The authors' 4.37 monkey patch, KVPress, MInference, and KVCache-Factory are external integrations, not an upstream Transformers feature. |
| Negative check | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/bb4e0e1b3f6bb38960769a1c9bcd2081016154cd) | `bb4e0e1b3f6bb38960769a1c9bcd2081016154cd` | No upstream SnapKV path or documented integration was found at this revision. `snap-llama.cpp` found by web search is a Linux **Snap package** of llama.cpp, not the SnapKV algorithm. Do not use it as a reference. |

The author repository has no unit-test suite. Consequently, conformance should
not mean “copied the Python lines”; it should mean agreement with the paper,
commit-pinned algorithm on well-conditioned MHA vectors, and explicit tests for
the documented nntrainer adaptations.

## 3. Exact algorithmic semantics

Let:

- `L` be the logical prompt length;
- `W` be the observation-window length;
- `P = L - W` be the older prefix length;
- `C` be the target prompt-cache capacity;
- `K = C - W` be the number of prefix rows retained;
- `D` be head dimension;
- `Hq` and `Hkv` be query- and KV-head counts;
- `G = Hq / Hkv` for GQA/MQA.

For one batch and query head `h`, the reference score is

```text
logit[h, r, j] = dot(Q[h, P+r, :], K[h, j, :]) / sqrt(D)
prob[h, r, :]  = causal_softmax(logit[h, r, :])
raw[h, j]      = sum(r=0..W-1, prob[h, r, j]),  j in [0, P)
pooled[h, :]   = same_length_pool(raw[h, :])
selected[h]    = top_K(pooled[h, :])
cache[h]       = gather(old_KV_prefix, selected[h]) ++ original_KV[P:L]
```

The implementation details below are observable in the authors'
[`SnapKVCluster.update_kv`](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L38-L70).

### 3.1 Causal mask and normalization

Only the trailing `W x W` observation-to-observation block needs an explicit
triangular mask: observation row `r` may see all `P` prefix positions and
observation columns `0..r`, but not later observation columns. Softmax is over
**all visible prompt positions**, including visible observation-window keys.
Computing softmax over the prefix alone is not equivalent: it removes the
competition between the prefix and recent tokens and changes the votes.

The author code computes softmax in FP32, then casts probabilities back to the
query dtype before summing and pooling. A CPU FP32 implementation should use a
stable subtract-max softmax and FP32 accumulation. A future FP16 path must
decide whether exact author-style post-softmax rounding is required.

The `attention_mask` argument to the author helper is overwritten by its local
causal mask. It therefore does not honor caller padding masks. nntrainer should
either restrict v1 to unpadded prompt rows or explicitly exclude padded keys
and observation queries; silently voting for padding is incorrect.

### 3.2 Voting and pooling

The paper describes voting as a sum over observation queries. NVIDIA KVPress
uses a mean. With a fixed number of valid observation queries these differ only
by a positive constant and produce the same top-k ordering. The distinction
matters if different batch elements have different valid-window lengths.

“Clustering” in the released algorithm does **not** build clusters or merge
K/V vectors. It applies 1-D pooling to the score vector, then gathers the K/V
row at each selected pooled-score center. A large score can therefore promote
neighboring positions, but no neighborhood union is formed.

PyTorch edge behavior is part of the de-facto reference:

- `avg_pool1d(kernel=k, padding=k//2, stride=1)` uses zero padding and includes
  padding in the divisor (`count_include_pad=True`);
- `max_pool1d` pads as negative infinity;
- an odd, positive kernel preserves length;
- an even kernel produces `P+1` outputs under this call and may later create an
  out-of-range gather index. Enforce an odd kernel rather than copying this bug.

### 3.3 Capacity semantics

Equation 3 defines a ratio-derived `k = floor(p * Lprefix)`, while the released
code and the paper's experiments use a fixed maximum cache size. The code keeps
`C-W` prefix positions plus `W` recent positions. Thus the effective
compression ratio varies with prompt length.

The author helper returns unchanged tensors when `L < C`; equality enters the
compression branch, selects every prefix position, and may reorder it despite
not evicting anything. An implementation may safely use `L <= C` as a no-op to
avoid this accidental reorder, but this boundary choice must be in the spec and
tested because it is not byte-identical to the Python helper at `L == C`.

Paper settings are task-specific, not universal defaults:

- LWM Needle-in-a-Haystack: `C=1024`, `W=16`, max kernel size 5;
- LongBench: `C` in `{1024, 2048, 4096}`, max pooling, `W=32`, kernel 7;
- Command-R: `C=4096`, `W=64`, kernel 13;
- the author class constructor defaults to `W=64, C=320, kernel=5, avg`, while
  `init_snapkv` defaults to `W=32, C=2048, kernel=5, avg`.

This inconsistency is a reason to expose and validate all parameters rather
than treating any one upstream default as normative.

### 3.4 Prefill and decode lifecycle

The authors' Llama integration repeats K/V as needed, writes compressed K/V to
the cache, but leaves the local full-length K/V variables intact for the
current prefill attention; see
[`llama_hijack_4_37.py` lines 51-90](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L51-L90).
Compressing before computing the prompt's own attention output changes every
prefill token and is not SnapKV.

Later decode tokens are appended without another SnapKV selection. The
physical cache therefore grows as `C + generated_tokens`; “constant” in the
paper means constant **prompt** cache, not a globally bounded online cache.
SnapKV is not a drop-in implementation of decode-time H2O-style eviction.

Logical sequence length must remain the original prompt length plus generated
tokens even though physical K/V length shrinks. The authors track a separate
`kv_seq_len`, and their generation-input preparation consults it rather than
the compressed tensor length; see
[`prepare_inputs_for_generation_llama`](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L138-L180).
For RoPE, surviving keys keep their original rotations and new Q/K rows use
their original logical positions. Renumbering survivors by compacted row is a
severe correctness bug.

### 3.5 Top-k order and ties

`torch.topk` is not stable on ties and returns rows in score order, not original
chronological order. The authors then append the observation window in
chronological order. For ordinary one-token global causal decoding, a paired
K/V permutation is mathematically harmless because positions are already
encoded in K. It is unsafe to assume this for finite sliding-window attention,
relative-position biases, or a multi-token attention kernel whose causal mask
is derived only from compacted row order.

For deterministic CPU behavior, define a total ordering: greater finite score
first, then smaller original prefix index. Treat NaN as below every finite
score. This is deliberately stronger than PyTorch's unspecified tie behavior.

## 4. Implementation-by-implementation analysis

### 4.1 Authors' FasterDecoding/SnapKV

Relevant pinned files:

- [algorithm helper](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L23-L87);
- [Llama forward integration](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L18-L136);
- analogous Mistral and Mixtral monkey patches in `snapkv/monkeypatch/`.

Strengths:

- closest executable interpretation of Listing 1;
- makes the causal-window and pooling boundary behavior concrete;
- correctly preserves full K/V for prefill output and only compresses stored
  cache state;
- keeps logical length separate from physical cache length.

Limitations relevant to a C++ port:

- tied to Transformers 4.37-era monkey patches and FlashAttention;
- no unit tests or deterministic fixtures;
- ignores the supplied padding mask;
- accepts invalid `W=0` and even pooling kernels until downstream failure;
- `num_key_value_groups` is passed into `update_kv` but unused there;
- expands GQA K/V with `repeat_kv` **before** compression and stores the
  expanded query-head-width cache. This permits a different selected set for
  each query head but can make a Llama-3-style `G=4` compressed cache about
  four times larger than a KV-head cache.

The GQA ambiguity remains documented in the authors' open
[`#22`](https://github.com/FasterDecoding/SnapKV/issues/22): one KV head serves
multiple query heads, so a memory-efficient shared cache needs a score
reduction that the paper and original code do not specify.

### 4.2 Microsoft MInference

MInference's
[`minference/modules/snapkv.py`](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/snapkv.py#L40-L107)
explicitly credits SnapKV and is nearly a line-for-line copy: causal mask,
FP32 softmax, sum, avg/max pooling, top-k, gather, and recent concatenation all
match. Its
[`BaseKVCache.update`](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/kvcompression.py#L209-L281)
compresses on cache initialization and appends later K/V. It also repeats GQA
K/V to query-head width before compression.

This is useful evidence that the lifecycle interpretation is not accidental,
but it is not an independent oracle. It also contains GPU-specific behavior
such as `torch.cuda.empty_cache()`, so it should not be mechanically ported to
a CPU library.

### 4.3 NVIDIA KVPress

Relevant pinned files:

- [`SnapKVPress`](https://github.com/NVIDIA/kvpress/blob/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a/kvpress/presses/snapkv_press.py#L17-L104);
- generic [`ScorerPress.compress`](https://github.com/NVIDIA/kvpress/blob/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a/kvpress/presses/scorer_press.py#L76-L102);
- prefill-only [`BasePress.forward_hook`](https://github.com/NVIDIA/kvpress/blob/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a/kvpress/presses/base_press.py#L101-L162).

KVPress can consume attention returned by the model or reconstruct the last
`W` pre-RoPE queries, apply RoPE, repeat K transiently for QK scoring, and apply
a causal mask. It then:

1. averages votes over the observation-query dimension;
2. applies average pooling only;
3. reshapes query heads as `[Hkv, G]` and averages each GQA group;
4. pads the last `W` scores above the global prefix maximum;
5. lets the generic scorer retain a ratio-derived number of rows from the
   unrepeated KV-head cache.

This is strong independent support for **mean/sum group reduction before
top-k** in a memory-efficient GQA port. It also demonstrates a historical bug:
padding the recent-window score with exactly `scores.max()` allowed tied prefix
tokens to evict recent rows, fixed in
[`42175e7`](https://github.com/NVIDIA/kvpress/commit/42175e729984f08532bb070df37abd81ceefb494)
by using `max + 1`.

KVPress is not bit-identical to the author helper:

- capacity is a ratio of total length rather than fixed `C`;
- observation votes use mean rather than sum (ranking-equivalent in the usual
  fixed-window case);
- only average pooling is exposed;
- top-k runs over prefix plus sentinel-scored recent positions, so tied recent
  rows need not emerge in chronological order;
- if the chosen `n_kept` is less than `W`, only part of the observation window
  can survive. A fixed-capacity port must reject `C <= W` and append the window
  explicitly instead of relying on a sentinel.

### 4.4 KVCache-Factory / PyramidKV

KVCache-Factory retains the official query-head path but added an opt-in
KV-head path after a detailed GQA audit. The important pinned code is:

- [`_gqa_groups`, `_reduce_group_scores`, `_grouped_window_attn_cache`, and
  `_select_topk_kv`](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/pyramidkv/pyramidkv_utils.py#L236-L320);
- [`SnapKVCluster`](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/pyramidkv/pyramidkv_utils.py#L472-L559);
- [GQA cache-layout audit and validation record](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/docs/gqa_cache_layout.md);
- [independent naive-reference tests](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/tests/test_gqa_kv_head.py#L30-L73), [legacy bit-identity tests](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/tests/test_query_head_bitident.py), and [tiny CPU model integration tests](https://github.com/Zefan-Cai/KVCache-Factory/blob/94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0/tests/test_gqa_model_integration.py).

Its KV-head path transiently repeats K for score computation, reduces the
query-head score tensor to KV heads by configurable `mean`, `max`, or `sum`,
then pools and gathers the original unrepeated K/V. The audit correctly calls
this a semantic adaptation rather than a mechanical optimization. It records
that mean-reduced KV-head mode preserved accuracy closely on its tested
LongBench/RULER/NIAH subsets while reducing the compressed GQA cache itself by
the expected group factor. These downstream benchmark numbers are useful risk
evidence, not a substitute for nntrainer validation.

For nntrainer, this repository supplies the best test architecture: an
independent scalar/reference expression, exact selected-index checks, shape and
recent-window invariants, deterministic reruns, short-prompt pass-through, and
tiny end-to-end CPU generation checks.

## 5. Semantic comparison

| Dimension | Paper / authors | MInference | NVIDIA KVPress | KVCache-Factory KV-head mode | Recommended nntrainer CPU v1 |
| --- | --- | --- | --- | --- | --- |
| Compression time | One-shot initial prompt prefill | Cache initialization | Prefill hook | Initial prompt | One-shot initial prompt, after full prefill attention output exists |
| Decode growth | Appends generated K/V | Appends | Base mode compresses prefill; decoding wrapper is separate | Appends | Append; document `C + generated` physical length |
| Vote input | Causal softmax probabilities | Same | Returned or recomputed causal probabilities | Same | Reuse already-computed causal prefill probabilities |
| Observation reduction | Sum | Sum | Mean | Sum before configurable GQA reduction | Sum |
| Pooling | Avg or max | Avg or max | Avg | Avg or max | Support both; exact edge padding |
| Capacity | Fixed `C`; paper also defines ratio | Fixed `C` | Total-length ratio | Fixed `C` | Fixed `C`, require `C > W > 0` |
| MHA selection | Per query/KV head | Same | Per head | Same | Same |
| GQA storage | Repeat to `Hq`; select per query head | Same | Keep `Hkv`; group mean | Configurable; keep `Hkv` with mean/max/sum | Keep `Hkv`; group sum (same ranking as mean) |
| GQA operation order | Per-query-head only | Per-query-head only | Pool query heads, then group mean | Group reduction, then pooling | Pool query heads, then group sum |
| Recent window | Explicit append | Explicit append | High-score sentinel | Explicit append | Explicit append, chronological and byte-exact |
| Selected prefix order | Descending `topk` score, unstable ties | Same | Descending global score | Same | Descending score, smaller original index breaks ties |
| Padding mask | Ignored by helper | Ignored by helper | Depends on supplied model attention/recompute path | Ignored by cluster helper | Explicitly unsupported or correctly masked; never silently ignored |
| Tests | None in repository | No independent SnapKV oracle | Framework tests, but not a fixed numerical vector | Strong unit and CPU integration tests | Hand vectors + independent scalar oracle + static integration invariants |

## 6. Frequent porting bugs and required defenses

1. **Evicting before the prefill attention output is calculated.** SnapKV
   compresses stored state for future decode; it must not make the prompt attend
   to its own already-compressed history.
2. **Using raw logits as votes.** The method sums causal softmax probabilities,
   not QK products. Logit ranking and probability-mass ranking can differ
   across observation rows.
3. **Softmaxing over prefix only.** Visible recent keys must participate in the
   denominator before recent columns are discarded from the vote.
4. **Masking all observation rows identically.** Observation row `r` may see
   only recent columns `0..r`; later recent columns must be masked.
5. **Ignoring padding.** The official helper does this, but a general batched
   port cannot. Padded prefix keys must receive zero probability and padded
   observation queries must not vote.
6. **Expanding and storing GQA heads by accident.** `repeat_kv` is acceptable
   as a transient score-computation view, not as nntrainer's stored cache
   layout. Otherwise memory grows by `G` and compiled tensor shapes change.
7. **Reducing GQA after top-k.** Query-head scores must be reduced first; taking
   separate top-k sets and trying to merge them creates an undefined and often
   oversized budget.
8. **Wrong pooling edges.** Average pooling divides edge sums by the full
   kernel; max pooling uses negative-infinity padding. A truncated-window
   average that divides by the number of real elements changes rankings.
9. **Allowing even or zero kernels.** The PyTorch call can return the wrong
   length or fail later. Validate early.
10. **Losing the observation-window guarantee.** Enforce `C > W` and append the
    last `W` rows explicitly. A finite sentinel plus global top-k is fragile,
    as KVPress's fixed bug demonstrates.
11. **Unspecified ties or NaNs.** Parallel/STL top-k choices can vary by build.
    Define finite-before-NaN and smaller-index tie behavior.
12. **Gathering K and V with different indices.** Store/compute one index list
    per batch and KV head, then apply it identically to both tensors.
13. **Unsafe in-place compaction.** A destination row may overwrite a source
    needed by a later gather, especially because selected indices are not
    monotonic. Use temporary storage or a proven cycle algorithm.
14. **Renumbering RoPE positions.** Physical cache row and logical token
    position diverge after eviction. New tokens still use original logical
    positions, and selected keys retain their original rotation.
15. **Assuming compacted order is chronological.** This is not true for author
    `topk`. Exclude sliding-window/relative-position modes or carry explicit
    logical positions and define their mask semantics.
16. **Promising a permanently bounded cache.** The paper method bounds prompt
    K/V only. Decode-time re-eviction is another algorithm and test surface.
17. **Sharing state across batches or requests.** Selection is per batch, layer,
    and head. Reset compression state and logical/physical offsets on a new
    sequence.
18. **Treating a smaller logical length as released memory.** With a fixed
    preallocated nntrainer slab, v1 reduces decode attention work but does not
    necessarily reduce peak allocation. Report this distinction honestly.

## 7. Deterministic reference vectors and test oracles

### 7.1 Hand-checkable MHA vector

This vector tests causal masking, full-denominator softmax, vote summation,
top-k ordering, paired K/V gather, and recent-window append without depending
on a random-number generator.

```text
B=1, Hq=Hkv=1, D=1, L=6, W=2, C=4, kernel=1
K = [0, 1, 2, 3, 4, 5]
V = [10, 11, 12, 13, 14, 15]
Q = [unused, unused, unused, unused, 1, 2]
```

The first observation query (logical position 4) sees keys 0..4; the second
sees keys 0..5. With `sqrt(D)=1`, their probabilities are approximately:

```text
q=1: [0.011656230956, 0.031684920796, 0.086128544436,
      0.234121657253, 0.636408646559, 0]
q=2: [0.000039255959, 0.000290064480, 0.002143302718,
      0.015836984018, 0.117020363346, 0.864670029480]
```

Prefix vote scores are therefore:

```text
[0.011695486915, 0.031974985276, 0.088271847154, 0.249958641271]
```

`K=C-W=2`, so author-order selected indices are `[3, 2]`; expected compacted
K/V are `[3, 2, 4, 5]` and `[13, 12, 14, 15]`. If an implementation chooses
chronological prefix storage instead, its expected indices must be explicitly
specified as `[2, 3]`; do not let sorting happen accidentally.

### 7.2 Pooling edge vectors

For input `[1, 2, 3, 4, 5]`, kernel 3, stride 1, same-length padding:

```text
avg (zero pad, include pad): [1, 2, 3, 4, 3]
max (negative-inf pad):      [2, 3, 4, 5, 5]
```

Also test kernel 1 as identity. Reject kernel 0 and every even kernel before
accessing tensors.

### 7.3 GQA reduction vector

Use a score-level unit test independent of softmax. Four query heads share two
KV heads with `G=2`; each query-head score vector is:

```text
q0=[1,2], q1=[3,6], q2=[10,0], q3=[20,4]
```

Contiguous grouping (`kv_head = query_head / G`) gives:

```text
mean = [[2,4], [15,2]]
sum  = [[4,8], [30,4]]
max  = [[3,6], [20,4]]
```

Mean and sum must select identical indices. A separate constructed end-to-end
case should make the two query heads in one group prefer different prefix
tokens and verify that reduction occurs before top-k.

### 7.4 Commit-pinned Python oracle

Generate golden fixtures from the authors' helper for **MHA only**, using the
pinned `e216ddc...` source. Use explicit decimal/rational tensor contents rather
than `torch.manual_seed` alone because RNG streams can change across PyTorch
versions. Serialize:

- dimensions and parameters;
- Q/K/V inputs in a simple row-major FP32 format;
- causal attention probabilities and raw/pooled scores;
- selected original indices;
- compacted K and V.

Require exact indices and K/V bytes; compare probabilities/scores with a tight
FP32 tolerance. Avoid ties in conformance vectors. Test the chosen deterministic
tie rule separately because the official helper does not define one.

For GQA, do **not** label the authors' repeated-query-head output as the
nntrainer oracle. Use an independent scalar implementation of “score every
query head -> pool every query head -> group sum -> stable top-k -> gather KV
heads.” Cross-check the pool-then-reduce ordering against NVIDIA KVPress. Treat
KVCache-Factory's reduce-then-pool mode as a useful alternative adaptation,
not as a max-pooling oracle, because the operations do not commute.

### 7.5 Metamorphic and invariant tests

- Short prompt `L <= C` is a byte-exact no-op under the nntrainer boundary
  decision.
- Output length is exactly `C`; the final `W` rows equal the input final `W`
  rows byte-for-byte.
- Every output K/V pair comes from the same original `(batch, kv_head, token)`.
- Each KV head may select a different prefix set; batches remain independent.
- Adding a constant to every visible logit of one observation row leaves that
  row's probabilities and selection unchanged.
- Query-group sum and mean select the same indices for fixed `G` and finite
  scores.
- Kernel 1 pooled scores equal raw votes.
- A paired permutation of prefix K/V and the corresponding score positions
  produces the correspondingly permuted selected-token identities.
- Re-running the same input produces identical indices, including tied-score
  fixtures.
- The first decode token is written at physical row `C` but uses logical RoPE
  position `L`.
- Reset/new-sequence behavior clears one-shot state and logical/physical
  offsets.

## 8. Recommended source hierarchy for code review

When implementation choices conflict, review them in this order:

1. Use the final paper for the method's intent and lifecycle.
2. Use the authors' pinned helper for MHA mask, FP32 softmax, pooling, top-k,
   and gather semantics.
3. Use NVIDIA KVPress and KVCache-Factory as independent support for the
   documented KV-head GQA adaptation.
4. Prefer nntrainer's explicit deterministic, cross-platform invariants over
   accidental PyTorch behavior (unstable ties, even-kernel output, ignored
   padding, or query-head cache expansion).
5. Treat MInference as corroboration only, and do not infer SnapKV behavior
   from llama.cpp or the unrelated `snap-llama.cpp` package.

This hierarchy supports the current `implementation-spec.md`: one-shot CPU
prompt eviction, reuse of normalized prefill attention, fixed KV-head cache
layout, per-query-head pooling followed by group-summed GQA scores, exact
pooling edge semantics, explicit recent append, deterministic top-k, and
separate logical versus physical positions.
