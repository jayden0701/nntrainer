# SnapKV pure-policy oracle audit

Date: 2026-08-03

## Post-audit semantic update

The SHA-pinned 2,000-trial audit below predates the final GQA ordering decision
and therefore validated the former reduce-before-pool variant. It remains a
historical record for MHA, average pooling, top-k, and compaction, but is not the
final GQA+max oracle.

After an independent specification audit exposed the non-commutativity, the
implementation was changed to observation scoring per query head, pooling per
query head, then group-summing pooled scores per KV head. The committed GTest
`matches_scalar_oracle_across_gqa_geometries` now checks this final pipeline
over 128 deterministic geometries in both pooling modes. The dedicated
`pools_each_query_head_before_gqa_reduction` counterexample freezes the max-pool
ordering. Both tests pass in the final focused 23-test run.

## Scope and snapshot

This audit independently reviewed the pure CPU policy and its unit tests. It
did not modify production code or test sources.

Reviewed files and SHA-256 values:

- `snapkv_policy.h`:
  `CD76DAC4558CC8C82504608A74534D65A3D8D7B05CAF7C0C744F54C769644122`
- `snapkv_policy.cpp`:
  `8B41C1629305C46D0F12E1441C47DD06936F68B922473670E7E4330033E1D86F`
- `unittest_snapkv_policy.cpp`:
  `4DCD3A7B590E793F5B10E4C0054B3694AAB25EA30645C85B6BA12C70820DD103`

The author reference was
`FasterDecoding/SnapKV@e216ddc84c5bd210378cbdbbba12ba02102aa640`, in
particular `snapkv/monkeypatch/snapkv_utils.py`.

## Result

No blocking correctness defect was found in the pure policy. A temporary
standalone C++ oracle harness passed **2,000 deterministic randomized trials**
for both max and average pooling.

The harness generated causal, softmax-normalized triangular attention rows and
randomized all of the following:

- prompt length `L` from 3 through 30;
- observation window `W` from 1 through `L - 1`;
- KV-head count from 1 through 4;
- GQA group size from 1 through 4;
- positive odd pooling kernels, including kernels wider than the prefix;
- retained-prefix count from 1 through the complete prefix;
- different selections for every KV head.

For each case it independently computed:

1. observation-window score aggregation using a dense logical indexing loop;
2. same-length scalar pooling with explicit padding;
3. deterministic full-sort top-k selection;
4. byte-position expectations for K/V compaction with `head_dim = 3`.

The production policy output matched the scalar oracle. The strict compile and
execution command was:

```text
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  <temporary-harness> Applications/CausalLM/layers/snapkv_policy.cpp \
  -I Applications/CausalLM/layers -o <temporary-executable>
```

Observed result:

```text
PASS: 2000 randomized trials, both pooling modes
```

Both the policy and the temporary harness also passed Clang 14
`-fsyntax-only -Wall -Wextra -Werror -pedantic`. An ASan/UBSan run could not be
performed because this Windows MinGW installation does not contain `libasan`
or `libubsan`; the failure occurred at link time, before execution.

## Equivalence to the author implementation

For ordinary finite MHA inputs, the policy stages match the author code once
the nntrainer integration supplies its already-softmax-normalized attention:

- the final `W` causal query rows are summed only over prefix columns;
- max pooling is stride one with negative-infinity padding;
- average pooling is stride one with zero padding and a full-kernel divisor,
  equivalent to PyTorch `avg_pool1d` with its default
  `count_include_pad=True`;
- top-k is per head and gathered K/V vectors use identical indices;
- selected prefix vectors precede the chronological observation window.

The pure policy intentionally does **not** implement QK scaling, the causal
mask, or softmax. It consumes the result of those operations from MHACore.
Therefore equivalence depends on the integration invariant that the input is
the complete global-prefill triangular tensor after softmax and before it is
reused or destroyed.

## Intentional divergences

These are documented design decisions, not defects:

1. **GQA/MQA.** The author implementation repeats KV heads and selects per
   query head. The final nntrainer path pools per-query-head scores, sums the
   pooled vectors within each KV group, and makes one selection per shared KV
   head. The committed 128-geometry oracle exercises group sizes 1-4 and both
   pooling modes for this nntrainer-specific rule.
2. **Ties.** PyTorch does not promise stable `topk` indices for equal values.
   The CPU policy deliberately chooses the smaller original position.
3. **NaNs.** The policy ranks NaNs below finite values and prevents a NaN from
   winning max pooling. This is deterministic hardening rather than a promise
   to reproduce backend-specific PyTorch NaN propagation.
4. **`L == C`.** The author helper enters its compression branch when the
   prompt length equals capacity, although this only gathers all prefix
   entries in score order. The nntrainer specification correctly treats this
   as a no-op boundary and compacts only when `L > C`.
5. **FP16 accumulation.** The typed policy converts each attention value to
   float and accumulates in float. The author code converts the softmax back to
   the query dtype before summing. FP16 selections may differ near a top-k
   boundary; the nntrainer behavior is the more numerically stable one but is
   not bit-equivalent.

## Verification gaps to close

Priority is relative to the feature, not an assertion that the current code is
wrong.

### High value

- Add an MHACore seam test proving the policy receives post-softmax
  probabilities in the full global triangular layout. Current pure tests begin
  after that contract and therefore cannot detect passing logits, a windowed
  tensor, or the wrong `from` offset.
- Add a multi-batch integration test. `compactCache` deliberately handles one
  batch, while batch pointer arithmetic lives in `MHACoreLayer::compactSnapKV`.
  Existing unit tests cannot catch an incorrect batch stride.
- Execute the `observationScoresTyped<_FP16>` path in an FP16-enabled
  build. The current test suite exercises only the float overload, and static
  parsing cannot validate the platform's `_FP16` conversions.
- Add lifecycle integration coverage for the exact `L == C` boundary, first
  decode at logical `L` / physical `C`, finite sliding-window exclusion, and a
  reset to logical position zero. The pure position-state tests cover only the
  mapper, not its wiring to RoPE and cache tensor slicing.

### Defensive branch coverage

- Null attention, `L <= W`, zero heads, and non-divisible GQA geometry.
- Score-vector size mismatches for pooling and top-k.
- Top-k retained count zero and retained count larger than the prefix.
- Null cache pointers, zero element size, zero head dimensions, prompt outside
  the allocation, and wrong selected-index count.
- Logical and physical cursor overflow at `UINT_MAX`.
- A direct average/max pooling oracle case with a kernel wider than the prefix
  in the committed unit suite (the randomized harness covered it, but the
  temporary harness is not a regression test).

## Conclusion

The selection, pooling, GQA aggregation, deterministic ranking, and
overlap-safe compaction logic agree with an independent scalar oracle over the
tested domain. Residual risk is concentrated at the MHACore integration seam,
multi-batch address calculation, and FP16 execution rather than in the pure
policy implementation.
