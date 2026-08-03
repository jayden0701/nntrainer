# Independent review of the SnapKV CPU policy

Review date: 2026-08-03 (Asia/Seoul)

Verdict: **the normal finite-value algorithm is semantically sound, but the
reviewed snapshot is not ready to merge.** Two arithmetic-safety defects and a
mandatory formatting failure should be fixed first. The current tests also do
not yet establish all acceptance criteria claimed by the specification.

No source file was changed as part of this review.

## 1. Scope and snapshot

Reviewed files and Git blob hashes:

| File | Blob hash |
|---|---|
| `Applications/CausalLM/layers/snapkv_policy.h` | `c2c3bcdafd7252f5feb743f4f6cc0ddfd2b32be9` |
| `Applications/CausalLM/layers/snapkv_policy.cpp` | `822f386eabdc4e20a3926f63b1ba7ade4fe8d737` |
| `test/unittest/layers/unittest_snapkv_policy.cpp` | `de00660d5b82d1ed3026697ea970b68b486500ca` |
| `docs/snapkv/implementation-spec.md` | `19fbe3b7e3c1f74038d71ae7e25869b56b73f985` |

The production call site in `mha_core.cpp` and the Meson/Android source lists
were read only far enough to determine whether the reviewed helpers and tests
exercise the actual path. They were not otherwise audited here.

Severity meaning:

- **High**: memory/undefined-behavior risk, or a deterministic merge/CI blocker.
- **Medium**: meaningful correctness or verification gap that should be fixed
  before relying on the feature.
- **Low**: defensive API, portability, or specification-clarity issue.

## 2. Reference semantics used for the review

The following sources were independently compared:

- the local `SnapKV.pdf`, visually reviewed on PDF pages 4 through 8, including
  equations 2-3 and Listing 1;
- the [final NeurIPS paper](https://papers.nips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf);
- the author implementation pinned at
  [`e216ddc84c5bd210378cbdbbba12ba02102aa640`](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L38-L70),
  plus its
  [`repeat_kv` integration](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L73-L87);
- the official PyTorch documentation for
  [`avg_pool1d`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.avg_pool1d.html),
  [`MaxPool1d`](https://docs.pytorch.org/docs/main/generated/torch.nn.MaxPool1d.html),
  and [`topk`](https://docs.pytorch.org/docs/main/generated/torch.topk.html).

The review treats shared selection per native KV head as an intentional
nntrainer adaptation. The author code instead expands K/V to query-head width
before selection and can retain a different set for each query head.

## 3. High-severity findings

### H1. `poolScores` has signed-overflow and narrowing hazards for accepted inputs

Location: `snapkv_policy.cpp:145-176`, especially lines 146 and 152-166.

`pooling_kernel`, `position`, and `prefix_length` are `unsigned int`, but the
implementation converts the radius and positions to `int` and computes:

```cpp
const int source = static_cast<int>(position) + delta;
```

`validateConfig` accepts every positive odd `unsigned int` kernel. Consequently:

- a position greater than `INT_MAX` undergoes implementation-defined conversion;
- `position + delta` can overflow signed `int`, which is undefined behavior;
- with `pooling_kernel == UINT_MAX`, incrementing `delta` after `INT_MAX` is
  itself signed overflow;
- a huge kernel performs billions of iterations even when the prefix contains
  only a few elements, providing a configuration-driven denial-of-service path.

The ordinary 32/5 settings are unaffected, but the public validation contract
currently declares the pathological inputs valid.

Concrete fix:

1. Perform all positions and window bounds in `size_t`.
2. Compute a clipped half-open interval without overflowing, for example from
   `max(position - radius, 0)` to
   `position + min(radius, prefix_length - 1 - position) + 1`.
3. Iterate only over real source elements. For average pooling, continue to
   divide by the full configured kernel width so zero padding remains included.
4. Optionally impose a documented practical upper bound on the kernel to reject
   configurations that cannot be useful. The clipped implementation is still
   needed even if such a bound is added.
5. Add boundary tests for a kernel larger than the prefix. An overflow-only
   test may call a factored window-bound helper so it does not require an
   impossibly large vector or multi-billion-iteration test.

The strict-warning compile also exposes the signed/unsigned indexing at lines
156 and 166.

### H2. `compactCache` does unchecked source-span byte arithmetic before `memcpy`

Location: `snapkv_policy.cpp:248-297`, especially lines 273-294.

The code checks `cache_capacity * token_bytes`, but it never checks the claimed
full cache span `max_seq_len * token_bytes`. Source offsets are then formed with
unchecked multiplication and addition:

```cpp
source * token_bytes + head * head_bytes
prefix_length * token_bytes
```

The destination side is indirectly bounded by the checked compacted-buffer
size. The source side is not: selected prefix positions can be much later than
`cache_capacity`, and `prefix_length` can be much larger than capacity. With
malformed or unrepresentable geometry these expressions wrap as unsigned
integers and the resulting pointer arithmetic/`memcpy` is out of bounds, which
is undefined behavior. This is especially relevant to 32-bit targets, which
the repository explicitly supports.

Concrete fix:

1. Check `max_seq_len * token_bytes` before any source pointer arithmetic.
2. Add a `checkedAdd` companion to `checkedMultiply` and use both for source,
   head, observation, and final-end offsets.
3. Verify each `[offset, offset + bytes)` lies within the representable claimed
   cache span before calling `memcpy`.
4. Keep the documented caller precondition that the two pointers actually
   reference buffers of the claimed size; a raw pointer cannot prove allocation
   extent by itself.
5. Add focused overflow tests that fail before allocation or pointer access.

### H3. All three C++ files fail the repository's required clang-format 14 check

Locations: `snapkv_policy.h`, `snapkv_policy.cpp`, and
`unittest_snapkv_policy.cpp` throughout.

The repository's `AGENTS.md` makes clang-format 14 a non-negotiable CI gate.
This exact command fails on the reviewed snapshot:

```text
clang-format --dry-run --Werror \
  Applications/CausalLM/layers/snapkv_policy.h \
  Applications/CausalLM/layers/snapkv_policy.cpp \
  test/unittest/layers/unittest_snapkv_policy.cpp
```

It reports formatting violations in all three files. This is a deterministic
merge blocker, independent of the unavailable full build.

Concrete fix: run `clang-format-14 -i` on the three files, then rerun the dry
check. Recheck line references in this report after formatting.

## 4. Medium-severity findings

### M1. The max-padding unit test cannot detect zero padding

Location: `unittest_snapkv_policy.cpp:104-110`.

The test is named `max_pool_uses_negative_infinity_padding`, but every input is
positive. Replacing negative-infinity edge padding with zero would produce the
same expected result, so the test passes under the precise bug it claims to
guard against.

Concrete fix: use an all-negative edge vector, for example:

```text
input    = [-4, -2, -10, -7]
kernel   = 3
expected = [-2, -2, -2, -7]
```

An incorrect zero-padded implementation would return zero at the two edges.

### M2. The tests do not satisfy the specification's own acceptance criteria

Locations: `implementation-spec.md:202-240` and the complete policy test file.

At this snapshot, `test/` contains no other SnapKV test. Missing required
coverage includes:

- batch independence;
- an end-to-end aggregate -> pool -> top-k -> K/V compact oracle for both MHA
  and the documented GQA reduction;
- the independent Python/PyTorch oracle required by section 13;
- the integration/static cases listed in section 12 (disabled behavior, short
  prompt, physical/logical positions, sliding-window exclusion, and reset);
- overflow boundaries and a kernel larger than the prefix.

The individual hand-written vectors are useful, but they do not prove the
feature-level claim in section 13.

Concrete fix: check in small golden vectors generated by a pinned PyTorch
script and test them from C++ without requiring PyTorch at C++ test runtime.
Include two batches, MHA, GQA with disagreeing query-head preferences, both pool
modes, and a full cache result. Add the listed layer integration/static tests
separately.

### M3. The tested aggregation helper is not the production aggregation path

Location: `unittest_snapkv_policy.cpp:51-102`; cross-check observation in the
current `mha_core.cpp:919-949`.

The GQA and triangular-layout tests call
`SnapKVPolicy::aggregateObservationScores`, but the production call site
duplicates the aggregation loop in `MHACoreLayer::compactSnapKV` to support
both FP32 and FP16 attention. No production call to the reviewed aggregation
helper exists. Thus the strongest semantic test can remain green while the
actual indexing or GQA mapping drifts.

The duplicated loop matched the reviewed helper at inspection time, so this is
a verification defect rather than a demonstrated output defect.

Resolution: production now calls the shared typed policy stages directly. The
final ordering is `observationScores[Hq,P] -> poolScores[Hq,P] ->
aggregateGQAScores[Hkv,P]`, so the same FP32/FP16 implementation is exercised
by the focused policy tests.

Concrete fix: share one typed/template implementation between the float helper
and production, or add direct integration tests for both production attention
dtypes. Avoid retaining a test-only algorithm clone as the semantic oracle.

### M4. NaN handling is deliberate but is not fully specified as a PyTorch departure

Locations: `snapkv_policy.cpp:41-43`, 150-175, and 204-209;
`snapkv_policy.h:65-69`; `implementation-spec.md:68-73` and 202-212.

For finite inputs, pooling matches PyTorch. For NaN inputs it intentionally does
not:

- max pooling maps each NaN input to negative infinity before taking a max;
- average pooling maps an entire output window containing NaN to negative
  infinity;
- top-k maps NaN to negative infinity for ranking.

Current PyTorch CPU max-pooling kernels propagate an encountered NaN, and
ordinary average arithmetic also produces NaN. The specification requires NaN
to rank below finite scores, so a defensive departure is reasonable, but the
header currently says the pooling behavior matches PyTorch without a finite-
input qualification. No test locks the policy for a NaN inside either pooling
window, or for infinities.

Concrete fix: explicitly state that PyTorch parity is defined for finite
scores, define the chosen NaN/`+inf`/`-inf` policy, and add max/average/top-k
tests for those values. If exact PyTorch behavior is desired instead, remove
the sanitization and accept PyTorch's resulting top-k behavior; do not leave it
implicit.

### M5. `poolScores` silently interprets an invalid enum value as average pooling

Location: `snapkv_policy.cpp:150-176`.

Every value other than `SnapKVPooling::MAX` takes the average branch. Normal
configuration passes through the parser, but this is a public C++ helper and an
invalid cast, ABI mismatch, or future enum member is silently accepted despite
the specification's reject-invalid-configuration rule.

Concrete fix: use an explicit `switch` with `MAX`, `AVERAGE`, and a throwing
default. Add a test using `static_cast<SnapKVPooling>(invalid_value)`.

## 5. Low-severity findings

### L1. The configuration specification disagrees with the parser about `average`

Locations: `snapkv_policy.h:26-31`, `snapkv_policy.cpp:47-57`, and
`implementation-spec.md:103-140`.

The parser and its test accept `average`, while the normative configuration
section says only `max` and `avg` are accepted. Either document `average` as an
alias in sections 6-7 or remove it from the parser. The former is backward-
compatible and matches the current API comment.

### L2. Two intentional reference departures should be labeled explicitly

Locations: `implementation-spec.md:43-81`.

The specification correctly makes `L <= C` a no-op and chooses the lower index
for score ties. The pinned author helper enters compression at `L == C`, and
PyTorch does not guarantee stable tied `topk` indices. These are sensible
choices, but section 2 says nntrainer-specific adaptations must be called out.
Add short notes identifying both as deliberate determinism/no-op departures,
as was already done well for GQA.

### L3. The include guard uses an implementation-reserved identifier

Location: `snapkv_policy.h:11` and 104.

Identifiers containing a double underscore are reserved to the implementation
in C++. `__SNAPKV_POLICY_H__` can therefore collide with a compiler or platform
header. Use a project-scoped guard such as
`NNTRAINER_APPLICATIONS_CAUSALLM_SNAPKV_POLICY_H_` (or `#pragma once` if that is
the accepted repository convention).

## 6. Semantics confirmed correct for ordinary finite inputs

The following parts were independently checked and no defect was found:

1. `triangularElements(q) == q * (q + 1) / 2` is the correct start offset for
   query row `q` in nntrainer's full causal `[key][query-head]` row packing.
   Iterating queries `[P, L)` and keys `[0, P)` exactly sums observation-window
   probability mass directed at the prefix.
2. The attention element-count check uses checked multiplication and rejects a
   non-triangular buffer claim.
3. `query_head / (Hq / Hkv)` is the same contiguous grouping induced by the
   author's `repeat_kv`. Summing all query heads in a group before selection is
   the documented native-GQA adaptation. Dividing by the common group size
   would not change per-KV-head ordering.
4. Odd-kernel, stride-one average pooling skips implicit zeros but divides by
   the full kernel width, matching `count_include_pad=True`.
5. Odd-kernel max pooling skips out-of-range elements, which is equivalent to
   negative-infinity padding for finite inputs.
6. `partial_sort` uses a strict deterministic ordering after NaN normalization:
   descending score, then ascending original index. This is a valid deliberate
   stabilization of PyTorch's unspecified tie order.
7. K and V use identical per-head selected indices. Temporary byte buffers
   prevent in-place overlap corruption, and the full observation window is
   appended chronologically.

## 7. Static validation record

The full repository build and gtest suite were unavailable, as expected. The
following focused checks were performed:

| Check | Result |
|---|---|
| Clang 14, C++17, `-Wall -Wextra -Wpedantic`, compile policy object | Pass |
| Clang static analyzer on `snapkv_policy.cpp` | No diagnostic |
| GCC C++17 standalone compile | Pass |
| GCC/Clang with conversion/sign-conversion warnings | Signed/unsigned index warnings at pooling lines 156/166; Clang also reports iterator difference conversion at line 214 |
| Minimal standalone finite-value harness, including negative-edge max pooling and deterministic NaN top-k | Pass |
| clang-format 14 dry check | **Fail**, all three C++ files |
| Full gtest/PyTorch oracle | Not runnable in this environment; required coverage remains to be added |

## 8. Recommended repair order

1. Replace signed pooling-window arithmetic and add large-kernel boundary tests.
2. Check the complete cache byte span and every source/end offset before
   pointer arithmetic.
3. Run clang-format 14.
4. Make the production aggregation path share the tested implementation or add
   direct production-dtype tests.
5. Add the negative max-padding vector, checked PyTorch golden fixtures, batch
   independence, and the specification's integration/static cases.
6. Resolve and document NaN, enum, alias, equality, tie, and include-guard
   details.
