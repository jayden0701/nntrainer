# SnapKV specification-to-code audit

Audit snapshot: 2026-08-03 03:49 KST

Scope:

- `docs/snapkv/implementation-spec.md`
- `docs/snapkv/paper-analysis.md`
- `docs/snapkv/reference-implementations.md`
- the current SnapKV policy and `mha_core` integration
- Transformer/CausalLM configuration, focused tests, and the user README

This review did not modify production or test code. Several findings were
reported while the implementation was still moving and were corrected before
this final re-read; those are recorded separately from the open findings.

## Verdict

No open P0/P1 algorithm defect was found in the documented v1 path: a dense,
common-length, unpadded prompt supplied in one CausalLM prefill call to global
causal attention using the external cache.

For that path, code and the three primary design documents now agree on the
important sequence:

```text
full prefill attention/output
  -> sum final-W attention rows per query head over prefix columns
  -> pool each query-head vector
  -> sum pooled vectors within each native KV-head GQA group
  -> deterministic per-KV-head top-k
  -> gather paired K/V through temporary storage
  -> append the final W prompt rows chronologically
```

The implementation also keeps logical RoPE position `L+t` separate from the
active physical cache length `C+t`, never re-evicts generated entries, and
does not compact finite sliding-window layers.

The remaining material work is defensive specification precision and
executable integration evidence. In particular:

- the focused pure-policy suite passes, but no checked-in test executes a real
  `MHACoreLayer` with external `Tensor` objects;
- non-finite ordering and deliberate upstream departures could be specified
  more completely;
- the adjacent single-cursor manager view API remains incompatible by design.

Severity convention in this report: P2 is a material contract, integration, or
verification gap; P3 is defensive hardening or documentation precision.

## Open findings

### [P2 verification] Required real-layer cases are still supported only by static reasoning

The current GTest is strong for the pure policy. At this snapshot,
`tmp/unittest_snapkv_policy.exe` ran 23/23 tests successfully. The policy and
test translation units also pass Clang 14 C++17 syntax checks with:

```text
-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
```

Coverage includes an independent scalar expression, 128 deterministic GQA
geometries in both pooling modes, the non-commutative max-pooling regression,
pooling edges, ties, NaNs, cursor state, overlap-safe byte compaction, two raw
batch buffers, two-byte elements, and arithmetic rejection.

It does not instantiate a compiled `MHACoreLayer`. Therefore these items from
`implementation-spec.md` section 12 remain unexecuted:

- `shouldCompactSnapKV()` gating for disabled, short, global, and finite-window
  cases;
- full-prefill output completion before cache mutation;
- the real `Tensor::getData()` batch byte offset;
- property propagation from `nntr_config.json` through Transformer to every
  MHA;
- FP16 production instantiation;
- first decode at logical `L` and physical `C`;
- reset and a hybrid global/sliding graph.

The raw-buffer batch test cannot detect a wrong batch stride in
`MHACoreLayer::compactSnapKV()`, because that pointer arithmetic is above the
tested helper. Likewise, the pure cursor test cannot prove that RoPE receives
the logical cursor while cache views receive the physical cursor.

Minimum follow-up when an nntrainer runtime is available:

1. Run a layer fixture with `B=2, Hq=4, Hkv=2, L=6, W=2, C=4`.
2. Assert that prefill output is unchanged and both batch caches contain the
   expected per-head K/V bytes.
3. Decode one token and assert logical `6 -> 7`, physical `4 -> 5`, and
   offset 2.
4. Add a finite-window companion and JSON configuration matrix.

Until then, the correct claim is “pure policy executed; MHA/Tensor integration
statically audited and deferred,” not full feature runtime verification.

### [P2 adjacent integration] The manager-view restriction is documented but not enforced at its API

`implementation-spec.md` now correctly says that SnapKV must not use
`KVCacheManager`'s position-based read/write views. A manager owns one cursor,
while a hybrid model can have:

```text
global SnapKV layer:      physical C+t
finite-window layer:      physical/logical L+t
manager position:         logical L+t
```

The current production path is safe because it binds each complete slab and
lets each MHA choose its own row. The risk is future reuse: the manager header
does not encode or warn about this restriction. Add adjacent API documentation
or a guard before introducing positional manager views into this path. A
future view API needs a layer-specific physical cursor/offset.

### [P3 non-finite contract] NaN handling is deterministic but not a complete documented order

The code currently defines:

- max pooling ignores NaN samples; an all-NaN real window yields `-inf`;
- average pooling yields `-inf` when any real input in the window is NaN;
- top-k maps NaN to `-inf`, so NaN ties an actual `-inf` and the smaller
  source index wins;
- `+inf` and `-inf` otherwise use ordinary floating-point comparison.

`implementation-spec.md` now documents the first two rules and limits
reference parity to finite inputs, but it does not define NaN versus actual
`-inf`, all-NaN max windows, or mixed infinities. The policy header still says
pooling matches PyTorch without the finite-input qualification.

Complete the normative total order and add all-NaN, NaN-versus-`-inf`, and
mixed-infinity tests. This is defensive behavior rather than a paper-defined
SnapKV rule.

Post-audit resolution: the normative spec now defines all-NaN max output,
NaN-versus-`-inf` ranking/ties, and NaN produced by mixed-infinity group sums.
The focused test now covers each of those pathological cases as well as the
ordinary finite causal-softmax domain.

### [P3 specification hygiene] Implementation status and deliberate departures are not collected

The normative document still says `Status: design baseline` although the
feature and focused tests now exist. Replace it with an implementation status
that distinguishes completed policy work from deferred real-layer testing.

Section 2 also says every nntrainer-specific adaptation must be called out.
GQA and defensive NaN behavior now are, but the following decisions are spread
across reference/review documents rather than collected in the normative spec:

- `L == C` is a no-op, while the author helper enters its gather path;
- equal top-k scores prefer the smaller source position, unlike unspecified
  PyTorch tie order;
- FP16 attention values are converted to float and accumulated in float in the
  policy, which is not promised to be bit-identical to author/backend FP16
  accumulation.

Add a short “deliberate departures” subsection. The error contract should also
mention the implementation's `overflow_error` for checked arithmetic and
`logic_error` for forged cursor state; section 11 currently names only
`invalid_argument` and `out_of_range`.

Post-audit resolution: the spec is now marked as the implemented CPU v1
baseline and contains a collected deliberate-departures subsection plus the
complete exception taxonomy.

### [P3 error-path hygiene] New CausalLM incompatibility checks follow an owning raw allocation

`CausalLM::setupParameters()` allocates `ids_history` with `malloc` before
rejecting SnapKV with `skip_prefill` or precomputed-cache mode. If construction
throws at those checks, the derived destructor does not run and the allocation
can leak.

This is not a supported inference-path algorithm defect, and older setup
exceptions share the same ownership pattern. The new checks should still be
moved before allocation, or the buffer should use RAII.

Post-audit resolution: all rejecting configuration parsing now precedes the
allocation. Byte geometry is checked, `malloc` failure throws `bad_alloc`, and
a successful replacement frees any prior construction-time buffer.

## Caller-enforced v1 boundaries

These are not current spec/code mismatches, but they are important because the
layer cannot enforce them by itself:

- **Chunked prefill is unsupported.** `shouldCompactSnapKV()` interprets a
  first call from zero with `step_size > C` as the complete prompt. There is
  no prompt-final signal. The CausalLM caller supplies one full prefill call;
  a generic chunked caller could compact the first chunk too early.
- **Padded, packed, or variable-length batches are unsupported.** Scoring has
  one common `prompt_length` and no valid-length vector or padding mask. The
  current CausalLM path duplicates the same dense prompt across batch items.
- **All-sliding models are an accepted no-op.** Properties are applied, but no
  finite-window layer is eligible. This is now an explicit product decision,
  not an accidental silent fallback.

`implementation-spec.md` section 10 and the README now state all three
boundaries. General support would require an explicit prompt-final signal and
per-sample valid lengths/effective masks, not a change only inside the
selection helper.

## Findings corrected during the audit

### [Closed P1 semantic conflict] GQA plus max-pooling order

An intermediate implementation and two documents reduced query-head votes into
KV-head scores before pooling, while `paper-analysis.md` selected NVIDIA's
pool-first ordering. The difference is observable for max pooling:

```text
q0 = [10,0,0]
q1 = [0,0,10]

reduce then max-pool -> [10,10,10] -> stable top-1 position 0
max-pool then reduce -> [10,20,10] -> top-1 position 1
```

The final code now uses
`observationScores{Typed} -> poolScores(Hq) -> aggregateGQAScores -> selectTopK`.
The normative spec, paper analysis, and reference note use the same sequence.
The regression test asserts position 1 and includes the old-order negative
control. MHA is unchanged; average pooling is algebraically equivalent, though
near-tie FP32 reassociation should not be called bit-identical.

### [Closed P1 boundary documentation] Chunked and variable-length inputs

The first snapshot exposed the caller assumptions only in the paper analysis.
The specification and README now explicitly restrict v1 to one complete,
unpadded, common-length first prefill.

### [Closed P2 configuration contract] Enabled-mode rejection constraints

The specification now lists the same initialization constraints as the code:
cache/model bounds, causal external-cache MHA, no sink/skip-prefill,
divisible GQA geometry, and at least one `mha_core`. Section 4 now explicitly
evaluates runtime no-op eligibility only after those constraints validate.

### [Closed P2 terminology] Logical versus active physical cache length

The lifecycle section now correctly says that logical positions remain `L+t`
while eviction reduces the active physical/read length. This agrees with
section 8 and the cursor implementation.

### [Closed P2 user documentation] Decode growth, rejection constraints, and scratch memory

The README now states `C + t` active growth with no decode-time eviction,
documents capacity/head/internal-cache constraints, and records the temporary
K/V buffers as `2 * C * Hkv * head_dim * element_size` bytes per processed
batch plus score vectors.

### [Closed P2 configuration documentation] All-sliding model behavior

The chosen behavior is now explicit in both the specification and README:
configuration is accepted and compaction is a no-op when no global layer
exists.

### [Closed P2 integration documentation] Mixed manager cursors

The specification now records why manager positional views are not used and
why full-slab binding is safe. Only adjacent API hardening remains.

### [Closed P3 parser mismatch] Average-pooling alias

Code, tests, spec, and README now agree that `max`, `avg`, and `average`
are accepted. The actual parser is case-insensitive.

### [Closed P3 state validation] Forged compacted-state combinations

`validatePositionState()` now enforces
`has_compacted == (logical_to_physical_offset != 0)`, and tests reject both a
false flag with a nonzero offset and a true flag with a zero offset.
`mapLogicalPosition()` also rejects arbitrary rewind or forward-skip after
compaction; only the current append position or zero is valid.

### [Closed build record] Strict test warning

An intermediate test revision implicitly converted a checked signed index to
`size_t`, contradicting an early strict-compile claim. The test now performs
an explicit conversion after the bounds check. Re-running the exact strict
Clang command for both `snapkv_policy.cpp` and
`unittest_snapkv_policy.cpp` passes with no diagnostics.

## Final consistency checklist

| Contract | Code | Primary docs | Evidence |
| --- | --- | --- | --- |
| Full output before compaction | Yes | Yes | Static call-order audit; real-layer run deferred |
| Prefix-only votes from final `W` rows | Yes | Yes | Pure-policy oracle tests |
| Pool per query head before GQA reduction | Yes | Yes | Non-commutative regression |
| One selected set per native KV head | Yes | Yes | Policy/GQA tests |
| Same K/V indices, overlap-safe | Yes | Yes | FP32/two-byte compaction tests |
| Deterministic tie order | Yes | Yes | Focused test; adaptation label still recommended |
| Logical `L+t`, physical `C+t` | Yes | Yes | Cursor tests plus static MHA trace |
| No decode-time re-eviction | Yes | Yes | Static state trace |
| One complete dense prefill only | Caller contract | Yes | CausalLM call-site audit |
| Finite-window layers unchanged | Yes | Yes | Static gate audit; runtime test deferred |
| JSON/property configuration | Yes | Yes | Static parser/property audit |
| Full model/runtime validation | Deferred | Deferred in review docs | Not available in this environment |

## Verification record for this snapshot

- Focused executable: 23/23 tests passed.
- `snapkv_policy.cpp`: strict Clang 14 C++17 syntax check passed.
- `unittest_snapkv_policy.cpp`: the same strict syntax check passed.
- `git diff --check`: no whitespace error; Git emitted only the existing
  README LF-to-CRLF working-copy warning.
- No file under `subprojects/` was changed.
- Full nntrainer configure/build, cross-platform link, and real-layer runtime
  tests remain unavailable/deferred.
