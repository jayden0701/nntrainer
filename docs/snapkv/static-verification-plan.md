# SnapKV static verification and independent-oracle plan

Status: verification design baseline, 2026-08-03

This document defines the evidence required for the CPU SnapKV v1 while a
normal nntrainer configure/build/test cycle is unavailable. It is deliberately
stricter than the current unit tests: a test that repeats the implementation's
logic is not an independent oracle, and a successful standalone helper test is
not evidence that the MHA integration calls the helper correctly.

The final handoff must distinguish **verified now** from **deferred to a real
build environment**. Static checks and standalone executable tests can provide
high confidence in the pure policy, but they cannot justify claiming that the
complete CausalLM graph builds or runs on every supported target.

## 1. Claims to prove

Use the following IDs in the eventual verification report so that every claim
has a concrete piece of evidence.

| ID | Claim | Minimum evidence without a full build |
| --- | --- | --- |
| V01 | Disabled SnapKV leaves legacy behavior unchanged | Source data-flow audit plus a property/state test once the MHA test fixture is available |
| V02 | MHA selection matches the authors' pinned SnapKV helper | Commit-pinned PyTorch fixtures, exact selected token identities, and exact gathered K/V bytes |
| V03 | nntrainer's GQA/MQA adaptation is correct | Independent scalar group-reduction oracle and adversarial query-head preferences |
| V04 | Pooling edges, top-k ties, infinities, and NaNs have deterministic documented semantics | Exhaustive small-vector oracle tests, not only hand-authored expectations |
| V05 | K/V compaction is overlap-safe and byte-exact | Exhaustive/adversarial byte fixtures, guard bytes, and ASan/UBSan standalone execution |
| V06 | The prompt output is computed from the full cache before compaction | Call-order/data-flow audit and a future MHA numerical integration test |
| V07 | Logical RoPE position and physical cache row diverge correctly after eviction | State-transition proof, focused state test, and a future decode integration test |
| V08 | Compaction is one-shot, batch-local, and resettable | State-machine audit and batch/reset tests |
| V09 | Unsupported modes fail explicitly and sliding-window layers remain unchanged | Configuration matrix and source branch audit |
| V10 | All build descriptions contain the new source and test exactly where needed | Meson/Android source-set audit and, later, configure checks |
| V11 | The implementation is portable C++17 and free of obvious memory/overflow errors | Strict standalone compilation, sanitizers, format/diff checks, and manual 32-bit arithmetic review |
| V12 | Scope limitations are accurately reported | Documentation audit: no claim of backing-allocation reduction or decode-time re-eviction |

No item may be marked complete from source inspection alone if the table asks
for executable evidence. In particular, V06 and V07 remain partially deferred
until an nntrainer-linked integration test can run.

## 2. Freeze the evidence inputs

Record the following at the start and end of verification:

```powershell
git rev-parse HEAD
git status --short --branch
git diff -- Applications/CausalLM/layers/mha_core.h `
  Applications/CausalLM/layers/mha_core.cpp `
  Applications/CausalLM/layers/snapkv_policy.h `
  Applications/CausalLM/layers/snapkv_policy.cpp `
  Applications/CausalLM/layers/meson.build `
  Applications/CausalLM/jni/Android.mk `
  Applications/CausalLM/meson.build `
  test/unittest/layers/unittest_snapkv_policy.cpp
git -C tmp/references/snapkv-official rev-parse HEAD
Get-FileHash SnapKV.pdf -Algorithm SHA256
```

The author implementation used for conformance must remain pinned to
`e216ddc84c5bd210378cbdbbba12ba02102aa640`. Never generate a golden from an
unrecorded moving branch. Store generated output, compiler versions, commands,
and exit codes under an ignored directory such as
`tmp/snapkv-verification/<timestamp>/`. If a small golden fixture is checked
in, also check in a human-readable provenance file with the upstream commit,
parameters, dtype, and generator hash.

Use explicit decimal or hexadecimal input tensors. A seed by itself is not
sufficient provenance because framework RNG algorithms can change. Any fuzz
test should use a tiny locally specified PRNG (for example, a 32-bit LCG whose
recurrence and seed are written in the test output).

## 3. Three independent verification paths

### 3.1 Path A: authors' helper, MHA only

Run the pinned
`snapkv/monkeypatch/snapkv_utils.py::SnapKVCluster.update_kv` directly with
explicit FP32 `[B, H, L, D]` tensors. This path establishes conformance to the
released SnapKV behavior for MHA (`Hq == Hkv`). It must cover both `avgpool`
and `maxpool`, odd kernels 1, 3, and 5, multiple heads, and at least two prompt
lengths greater than capacity.

For these fixtures:

- use no ties, NaNs, or nearly equal selection-boundary scores;
- make every source K/V vector encode `(batch, head, token, component)` so the
  selected original token can be recovered from the output;
- record raw causal probabilities, unpooled votes, pooled votes, selected
  indices, and compacted K/V;
- require exact selected token identities and exact K/V values;
- compare probabilities and scores with a declared FP32 tolerance, for example
  `abs <= 2e-6 + 2e-5 * abs(reference)`;
- require the score gap between the last retained and first evicted position
  to exceed ten times the comparison tolerance.

The author helper's `torch.topk` order is not stable on ties. Therefore tie
fixtures must not be used to demand exact official parity. The nntrainer
smaller-index tie break is a deliberate deterministic extension verified in
Path B.

The author helper compresses when `L == C` and may reorder all prefix rows,
whereas the nntrainer specification makes `L <= C` a no-op. Test equality as
an nntrainer boundary rule, not as an upstream-conformance vector.

The full scalar example in `reference-implementations.md` is the first smoke
fixture:

```text
B=1, Hq=Hkv=1, D=1, L=6, W=2, C=4, kernel=1
selected prefix indices = [3, 2]
compacted K token IDs    = [3, 2, 4, 5]
compacted V token IDs    = [3, 2, 4, 5]
```

This detects prefix-only softmax, use of raw logits, chronological sorting of
selected rows, compaction before prompt output, and loss of the recent window.

### 3.2 Path B: independent scalar oracle

Create a small Python oracle that uses only `math`, `struct`, and ordinary
lists (NumPy and PyTorch are intentionally unnecessary). It must be written
from `implementation-spec.md`, not translated line-by-line from
`snapkv_policy.cpp`. Its stages should be separately callable:

```text
causal_softmax(Q, K, logical_query, visible_keys)
observation_votes(probabilities, P, W)
reduce_query_heads(votes, Hq, Hkv)       # contiguous groups, sum
same_pool(scores, kernel, mode)          # explicit virtual padding
stable_topk(scores, K)                   # (-score, original_index)
gather_token_major(KV, selected, tail)
```

The oracle should optionally round intermediates through
`struct.pack('<f', value)`/`struct.unpack('<f', ...)` to model FP32 storage.
Compare both rounded and real-valued results; selected indices must agree for
fixtures with a safe score margin.

This is the normative oracle for GQA/MQA, deterministic ties, NaNs, the
`L <= C` boundary, and byte compaction. It must not import the production
helper, invoke its functions, or copy its comparator. To prevent false
agreement, have it emit a simple JSON fixture that a separate C++ harness
consumes or compare with JSON emitted by that harness.

### 3.3 Path C: actual C++ policy and integration source

Compile the real `snapkv_policy.cpp` with a minimal test executable that has no
nntrainer dependency. Do not paste the implementation into the harness. The
harness should emit raw scores, pooled scores, indices, and cache bytes in a
machine-readable form for comparison with Path B.

Example strict compilation commands, adjusted for installed compiler paths:

```powershell
clang++ -std=c++17 -Wall -Wextra -Werror -pedantic `
  -I Applications/CausalLM/layers `
  Applications/CausalLM/layers/snapkv_policy.cpp `
  tmp/snapkv-verification/snapkv_policy_harness.cpp `
  -o tmp/snapkv-verification/snapkv_policy_harness.exe

clang++ -std=c++17 -Wall -Wextra -Werror -pedantic `
  -fsanitize=address,undefined -fno-omit-frame-pointer `
  -I Applications/CausalLM/layers `
  Applications/CausalLM/layers/snapkv_policy.cpp `
  tmp/snapkv-verification/snapkv_policy_harness.cpp `
  -o tmp/snapkv-verification/snapkv_policy_harness_asan.exe
```

Also compile a translation unit containing only
`#include <snapkv_policy.h>` to prove header self-containment. If MSVC is
available, compile the same harness with `/std:c++17 /W4 /WX /permissive-`.
Record when a sanitizer or compiler is unavailable rather than silently
skipping it.

The MHA integration cannot be independently validated by the policy harness.
It needs the static audits in section 8 now and an nntrainer-linked fixture
later.

## 4. Exact test matrix for observation scoring

### 4.1 Triangular layout and observation range

Test `observationScores` and the later `aggregateGQAScores` stage with these
independent cases:

1. `L=6, W=2, Hq=Hkv=1`: the hand-checkable full-softmax vector above.
2. `L=5, W=2, Hq=4, Hkv=2`: each query head writes a distinct decimal
   fingerprint; expected groups are heads `[0,1]` and `[2,3]`.
3. `L=4, W=1, Hq=8, Hkv=1`: MQA, proving all query heads contribute once.
4. `L=7, W=3, Hq=Hkv=3`: different preferred prefix token per head.
5. Fill all non-observation triangular rows with a huge sentinel. Scores must
   not change, proving only queries `[P,L)` vote.
6. Give visible recent keys most of the probability mass. Their columns must
   not appear in prefix scores, while their participation in the original
   softmax denominator is preserved by the supplied probabilities.

For every case, compare all raw score elements, not only final top-k indices.
Then test null attention, zero heads, non-divisible head counts, `W=0`,
`L<=W`, and attention element counts one below and one above the exact
`L*(L+1)/2*Hq` requirement.

The original production MHA path performed its own FP32/FP16 aggregation loop.
That duplication was removed: `compactSnapKV` now calls the shared typed
observation, pooling, and GQA-reduction stages. For FP16, separately verify the
intended post-softmax rounding and FP32 accumulation behavior in an
FP16-enabled runtime.

### 4.2 Full-softmax conformance

The C++ policy begins after softmax, so Path B must additionally test the
end-to-end mathematical pipeline that produces its input. Each observation
query `q` must normalize over keys `[0,q]`, including visible recent keys. The
following negative controls should produce a different golden and therefore
must be detected:

- normalize over `[0,P)` only;
- normalize every observation row over `[0,L)` without a causal mask;
- sum raw QK logits instead of normalized probabilities;
- vote with queries before `P`;
- include observation columns in the candidate prefix scores.

This protects the integration seam even though softmax itself is existing
nntrainer code.

## 5. GQA/MQA adaptation verification

The official helper repeats K/V to query-head width and is not the oracle for
nntrainer's fixed `Hkv` cache. Verify the documented adaptation independently:

```text
G = Hq / Hkv
kv_head(hq) = floor(hq / G)
group_score[hkv, p] = sum(score[hq, p] for hq in that contiguous group)
```

Required adversarial vectors:

- `Hq=4, Hkv=2`, where heads 0 and 1 prefer different tokens and their
  **combined** score chooses a third ordering; repeat independently for heads
  2 and 3.
- `Hq=8, Hkv=1`, where the winning MQA token is not the winner of head 0.
- `Hq=Hkv`, proving group reduction is exactly the MHA identity.
- two KV heads with deliberately different selected sets, proving there is no
  accidental selection broadcast.

Metamorphic checks:

- permuting query heads within one contiguous group leaves its result
  unchanged;
- group mean and group sum select the same indices for finite values and fixed
  `G`;
- multiplying every query-head score in one group by the same positive scalar
  leaves that group's selection unchanged;
- moving a query head across a group boundary changes only the two affected KV
  heads;
- the gathered cache width remains `Hkv * D`, never `Hq * D`.

Cross-check one finite, no-tie average-reduction vector against NVIDIA
KVPress's KV-head scoring behavior. Treat that comparison as corroboration,
not as the normative oracle. Record explicitly that exact parity with the
authors' repeated-query-head GQA output is neither expected nor claimed.

## 6. Pooling, ranking, and pathological values

### 6.1 Pooling borders

Use all of the following vectors for both one and multiple KV heads:

| Input | Kernel | Mode | Expected output |
| --- | ---: | --- | --- |
| `[1,2,3,4,5]` | 3 | avg | `[1,2,3,4,3]` |
| `[1,2,3,4,5]` | 3 | max | `[2,3,4,5,5]` |
| `[-9,-8,-7]` | 3 | max | `[-8,-7,-7]` |
| `[1,-2]` | 5 | avg | `[-0.2,-0.2]` |
| `[1,-2]` | 5 | max | `[1,1]` |
| any finite vector | 1 | avg/max | exact identity |

The all-negative max vector catches an incorrect zero-padding
implementation. The kernel-wider-than-prefix vectors exercise virtual padding
on both sides. Reject zero and every even kernel before indexing.

For average pooling, simulate the specified left-to-right FP32 accumulation
in the scalar oracle and also compare within tolerance to PyTorch
`avg_pool1d(..., count_include_pad=True)`. Verify that edge positions divide
by the full kernel width, not by the count of in-range elements.

### 6.2 Stable top-k

Exhaust all vectors of prefix length up to five over a small alphabet such as
`{-inf, -2, -0.0, +0.0, 1, +inf, NaN}` and every retained count `1..P`.
Compare against a total-order oracle:

1. non-NaN scores descending;
2. NaN ranks below every finite score;
3. equal rankable scores prefer smaller original index.

The production comparator maps NaN to negative infinity. The specification
does not currently say whether NaN is equal to, or strictly below, an actual
`-inf`; settle and document that corner before freezing the exhaustive oracle.
Whichever rule is selected must form a strict weak ordering on every supported
compiler.

Assert additionally that selected indices are unique, in range, independently
ranked per head, and returned in score order rather than chronological order.

### 6.3 NaN propagation through pooling

There is a contract gap that must be resolved explicitly. Current max pooling
treats a NaN element as `-inf`, effectively ignoring it when a finite neighbor
exists. Current average pooling turns every output window containing any NaN
into `-inf`. The main specification only states that a NaN **score** ranks
below finite scores; it does not define whether NaN poisons an average window.

Before acceptance, choose one of these behaviors, add it to
`implementation-spec.md`, and test isolated NaN, all-NaN, NaN next to `+inf`,
and NaN at each border. Do not use the official helper as authority here:
PyTorch NaN/top-k behavior is not the deterministic cross-platform contract.

## 7. Byte-exact, overlap-safe compaction

The reference compactor should operate on `bytearray` and use a frozen copy of
the entire input as its source. Generate each element from a unique fingerprint
of `(batch, token, kv_head, dim, byte_within_element, K_or_V)`. Compare the
complete first `C * Hkv * D * element_size` bytes with `memcmp` semantics.

Required dimensions and layouts:

- element sizes 1, 2, 4, and 8; the 2-byte case must include arbitrary FP16
  bit patterns, signed zero, infinities, and NaN payloads;
- `Hkv` in 1, 2, and 4; `D` in 1, 2, and 7;
- a two-batch backing allocation, compacting each batch through its correct
  offset and proving the other batch is unchanged;
- a different non-monotonic selection per KV head;
- `W=1` and `W>1`; `K=1` and `K=P` where allowed by the direct helper;
- source index `0` and `P-1`, plus invalid `P`.

The mandatory overlap adversary is a selection where destination row 0 reads
source row 2 and a later destination reads source row 0. A naive direct gather
would overwrite row 0 before it is read. Run this case for both K and V, with
different bytes, and for more than one head.

Check these invariants:

- K and V use identical source indices but retain their distinct payloads;
- output rows `[K,C)` equal original rows `[P,L)` exactly and chronologically;
- head slices within one destination row may come from different source rows;
- no byte before the passed pointer or after the declared allocation changes;
- input bytes beyond the compacted prefix are never read out of bounds under
  ASan;
- invalid selected indices throw before the destination cache is committed.

The last point deserves a dedicated test: place an invalid index after several
valid ones and prove both K and V remain byte-identical after the exception.
The temporary output design should make this transactional behavior hold.

Exercise invalid/null pointers, zero element size, zero heads/dimensions,
`L>Cmax`, `L<=C`, mismatched selection count, and arithmetic products near
`size_t` overflow. Repeat the standalone sanitizer run on a 32-bit target when
one is available; a 64-bit-only result does not validate 32-bit overflow paths.

## 8. Static integration audit

Review the final source, not an intermediate diff, and record line numbers for
each invariant.

### 8.1 Call order and tensor layout

Trace one full prefill batch through
`MHACoreLayer::one_batch_incremental_forwarding` and prove:

1. Q/K logits use the complete prompt cache.
2. causal softmax finishes before votes are inspected;
3. `compute_fp16vcache_transposed` computes the prompt output from full K/V;
4. only then does `compactSnapKV` mutate external K/V;
5. `out_` has exactly the triangular
   `[(q*(q+1)/2+k)*Hq+hq]` layout expected by the aggregation loop;
6. cache rows are token-major `[token][kv_head][head_dim]` for both dtypes;
7. the batch byte offset is based on the entire tensor feature length and is
   applied exactly once.

A textual call-order grep is useful but insufficient. Follow the data objects
and aliases to ensure an asynchronous worker cannot still be reading the cache
when compaction begins.

### 8.2 State-machine proof

Write the state transitions in the verification report and compare every
assignment in source with this table:

| Event | logical next position | physical next row | offset | compacted |
| --- | ---: | ---: | ---: | --- |
| reset/new request | `0` | `0` | `0` | false |
| short prefill `L<=C` | `L` | `L` | `0` | false |
| long prefill, before mutation | `0` | `0` | `0` | false |
| long prefill completes | `L` | `C` | `L-C` | true |
| decode token `t` begins | `L+t` | `C+t` | `L-C` | true |
| decode token `t` completes | `L+t+1` | `C+t+1` | `L-C` | true |

At decode, K/Q RoPE must use the logical position `L+t`, while cache slicing,
write row, attention length, and V matmul must use physical positions through
`C+t+1`. Resetting to logical zero must clear both offset and one-shot state.
Setting a logical position below a nonzero offset must throw.

Check that compaction is performed once for **every batch** before the shared
layer state advances, and that failure during one batch cannot leave a
plausibly valid half-compacted state. The future integration test should cover
batch two with different attention preferences, not only duplicated data.

### 8.3 Branch/configuration matrix

Audit and later execute every row:

| Configuration/event | Required outcome |
| --- | --- |
| SnapKV object absent/capacity zero | legacy path, no compaction |
| `L<C` or `L==C` | byte-exact no-op |
| first full causal prefill, `L>C` | compact once |
| later decode or multi-turn append | append, never reselect in v1 |
| finite local/sliding window | leave layer unchanged |
| internal cache | reject when enabled |
| noncausal attention | reject when enabled |
| attention sink | reject when enabled |
| skip-prefill | reject when enabled |
| precomputed cache save/load | reject at the model/config boundary |
| `C>max_timestep` | reject before execution |
| invalid `C,W,kernel,pooling` | deterministic `invalid_argument` |

Verify that Transformer JSON values reach every eligible `mha_core` instance,
including architecture-specific constructors, without silently configuring
only one model family. Conversely, local-window layers may receive the
properties but must not compact. Confirm that direct layer-property users get
the same validation as JSON users.

### 8.4 Duplicate implementation risk

Confirm `mha_core.cpp::compactSnapKV` calls the shared
`SnapKVPolicy::observationScoresTyped`, `poolScores`, and
`aggregateGQAScores` stages. They must share:

- the same triangular offset formula;
- the same observation query and prefix bounds;
- the same contiguous GQA grouping;
- FP32 accumulation and supported dtype behavior;
- identical geometry and overflow checks.

Any future correction must update both paths unless they are unified. Treat a
helper-only green test with an untested MHA loop as insufficient for V02/V03.

## 9. Build-description and portability audit

Run and record:

```powershell
rg -n "snapkv_policy" Applications/CausalLM/layers/meson.build `
  Applications/CausalLM/jni/Android.mk Applications/CausalLM/meson.build
rg -n "unittest_snapkv_policy" Applications/CausalLM/meson.build
git diff --check
git diff --name-only -- subprojects
clang-format-14 --dry-run --Werror `
  Applications/CausalLM/layers/snapkv_policy.h `
  Applications/CausalLM/layers/snapkv_policy.cpp `
  Applications/CausalLM/layers/mha_core.h `
  Applications/CausalLM/layers/mha_core.cpp `
  test/unittest/layers/unittest_snapkv_policy.cpp
```

Manually establish that:

- `snapkv_policy.cpp` appears exactly once in each relevant Meson/Android
  source set and is linked into the target that owns `mha_core`;
- the gtest target links `gtest_main` and the library containing both policy
  and MHA symbols;
- include directories make `snapkv_policy.h` available on Windows, Android,
  Tizen, Yocto, and Ubuntu builds;
- no new POSIX-only API, VLA, compiler-specific half type outside guards, or
  unguarded architecture intrinsic was introduced;
- all byte offsets use `size_t`, all narrowing conversions are justified, and
  every multiplication used for allocation/offsets is overflow-checked;
- no file under `subprojects/` changed.

When a full environment becomes available, the deferred matrix is:

```text
Ubuntu gcc, transformer on, FP16 off
Ubuntu clang, transformer on, FP16 on
Windows Meson/MSVC CPU
Android NDK arm64 compile
Tizen/GBS compile
Yocto compile
all existing CausalLM tests plus unittest_snapkv_policy
ASan/UBSan host integration fixture
```

Adding a new gtest may affect the repository's test-count checker; inspect its
expected counts before submission.

## 10. Metamorphic and stress suite

After fixed vectors pass, run deterministic small-state enumeration:

- `L=2..12`, `W=1..min(4,L-1)`, every valid `C` with `W<C<L`;
- odd kernel `1,3,5,7`, including kernel greater than prefix length;
- `Hkv=1..3`, `G=1..4`, `D=1..5`, batch 1 and 2;
- both pooling modes;
- at least 10,000 finite score tensors from the recorded LCG.

For each case assert:

- output length is exactly `C`;
- selected indices are unique, in prefix range, and independent per KV head;
- final `W` rows are byte-identical and chronological;
- paired K/V identity is preserved;
- rerunning yields identical bytes and indices;
- group mean and sum choose identical indices;
- kernel one is identity;
- adding a constant to all visible logits in one observation row leaves that
  row's probabilities and final selection unchanged;
- a paired permutation of source prefix positions and their score columns
  preserves selected token identities under the inverse permutation, except
  where the deterministic original-index tie break intentionally resolves a
  tie differently.

Run the exhaustive/fuzz suite under ASan/UBSan and once in optimized mode
(`-O2 -DNDEBUG`) to catch comparator or undefined-behavior differences.
Compare selected-index hashes across clang, gcc, and MSVC when available.

## 11. Current coverage gaps to close

At the time this plan was written, the existing
`unittest_snapkv_policy.cpp` covers configuration parsing, one GQA aggregation
vector, triangular-size rejection, basic max/average borders, kernel one,
per-head ties, one NaN ranking case, FP32/two-byte overlap compaction, and one
invalid source index. That is a useful foundation, but it does not yet prove:

- numerical parity with the pinned official helper;
- the complete softmax-to-cache path;
- the duplicate aggregation loop actually used by `mha_core`;
- multiple batches or correct integration batch offsets;
- short/equal prompt no-op behavior;
- logical versus physical position transitions and RoPE position;
- reset, one-shot, sliding-window, or unsupported configuration behavior;
- canary/transactional behavior and arithmetic overflow;
- deterministic NaN pooling semantics;
- cross-compiler or sanitizer cleanliness.

These are evidence gaps, not necessarily implementation defects. Keep them
visible until each has a recorded test or an explicit deferred-build marker.

## 12. Verification report template and stop rule

Create a separate `docs/snapkv/verification-report.md` when executing this
plan. For each V01-V12 include:

```text
Claim:
Status: PASS / PARTIAL / FAIL / DEFERRED
Files and lines reviewed:
Exact command:
Tool/compiler version:
Exit code:
Artifact or output hash:
Observed result:
Residual risk:
```

Stop and correct the implementation or specification if any of these occur:

- official MHA selected identities disagree on a no-tie fixture;
- the independent oracle and C++ policy disagree on raw scores, pooling, or
  deterministic top-k;
- K/V bytes or recent rows disagree even once;
- sanitizer, strict compiler, format, or diff checks fail;
- production and helper aggregation semantics differ;
- the state-machine trace cannot establish which position is logical versus
  physical;
- an unsupported mode silently falls back instead of failing or skipping as
  specified.

If all executable standalone checks pass but the nntrainer build remains
unavailable, the honest final status is **pure policy verified; MHA integration
statically reviewed; full build/runtime verification deferred**. Do not call
the complete feature build-tested or production-validated until the deferred
matrix runs.
