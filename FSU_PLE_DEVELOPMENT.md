# FSU / PLE development context

## Goal and time limit

This document preserves the implementation context for investigating and
extending Flash Storage Utilization (FSU) and the Gemma4 per-layer embedding
(PLE) data path. The working time limit is **2026-08-24 05:30 KST**. It is a
living hand-off note, not a specification of behavior that has not yet been
implemented.

## Initial baseline

- Repository: `C:\nntrainer`
- Initial branch state: `main`, commit `cf2b653b`, clean worktree (checked at
  the start of this investigation).
- Repository rules: preserve existing user files, do not edit `subprojects/`,
  and do not create a commit for this documentation task.

## Five-stage execution plan

1. Establish the baseline and record the exact model/configuration entry
   points.
2. Trace the existing MoE FSU path from model construction through weight
   loading, virtual-tensor activation, and deactivation.
3. Trace Gemma4 PLE from configuration/API path through the per-layer
   embedding and subsequent decoder blocks; compare it with MoE FSU.
4. Decide the smallest compatible design, document alternatives and required
   invariants, then implement only after the ownership and data-lifetime
   boundaries are agreed.
5. Build/run focused validation, inspect the shared worktree diff, and update
   this note with results, risks, and the final hand-off.

## Findings: MoE FSU data flow

The existing reference implementation is
`Applications/CausalLM/models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp`.
`SlimMoELayer::finalize()` requests the router weight normally and each
expert's gate/up/down projection with `is_virtual=true`. It also creates
router-logit and expert-mask forward-lifetime tensors. During `forwarding()`:

1. Input `[B,1,S,H]` is reshaped to `[B*S,1,1,H]`.
2. The router projection is softmaxed and top-k indices/weights are computed.
3. Token assignments are grouped by expert.
4. For each selected expert, the three virtual weights are `activate()`d,
   expert computation is performed, and the weights are `deactivate()`d.
5. The accumulated result is reshaped back to `[B,1,S,H]`.

`activate()`/`deactivate()` are mmap/munmap operations in
`nntrainer/tensor/tensor.cpp`; activation requires a backing file descriptor
and is rejected on Windows. The inference load path in
`nntrainer/models/neuralnet.cpp` keeps a long-lived descriptor specifically so
virtual weights can be activated after the per-thread stream is closed.

The model-level FSU path is enabled by `Transformer::setupParameters()` reading
`fsu` and `fsu_lookahead`, then `Transformer::initialize()` setting those model
properties. `Manager::requestWeight()` adds lookahead execution orders for
inference FSU weights. `Manager::LoadTensors()` and `UnloadTensors()` schedule
pool cache work; `flushCacheExcept()` has a special asynchronous path for
lookahead 1 and a pool-level path for larger values.

## Findings: Gemma4 PLE data flow

`Applications/CausalLM/models/gemma4/gemma4_causallm.cpp` constructs two
embedding paths. The normal `embedding0` uses `EMBEDDING_FILE_NAME`; the second
embedding, named `per_layer_input_embedding`, has output width
`NUM_LAYERS * HIDDEN_SIZE_PER_LAYER_INPUT` and receives `PLE_FILE_NAME` as its
`quantized_lut_path`. Its output is projected, scaled, normalized, split per
layer, and fed into each decoder block as the per-layer input. Thus PLE is a
per-layer embedding/LUT file path, not an expert-weight cache.

The executable resolves `ple_file_name` relative to the model directory before
constructing the model (`Applications/CausalLM/main.cpp`). The C API carries
the same field through `ModelRuntimeConfig` and resolves it in
`Applications/CausalLM/api/causal_lm_api.cpp`. The checked-in Gemma4 runtime
config currently has no `ple_file_name`, while its `config.json` has
`enable_moe_block: false`, null expert counts, and standard dense MLP settings.
Consequently the current Gemma4 path does not exercise MoE expert FSU.

## Final design and invariants

The implementation uses two explicit, independent controls:

- `ple_split=true` changes the model artifact layout from one packed
  `per_layer_input_embedding` weight `[vocab, L*P]` to one layer-local weight
  `layer<i>_per_layer_input_embedding` `[vocab, P]` per decoder.
- `fsu=true` makes those split PLE weights virtual on POSIX only. Each lookup
  activates the mapped weight, uses it, and unmaps it through an RAII guard.

The split graph first slices the model-projection contribution with
`per_layer_slice`, then adds the matching layer-local embedding and applies the
same `sqrt(0.5)` scale. This preserves the legacy packed mathematical result.

The streamed/resident boundary in this change is deliberately limited to the
large layer-local PLE lookup tables. The `per_layer_input_projection` weight
and the normalized projection activation `[B, S, L*P]` remain full-width; this
does not yet stream, partition, or reduce those projection-side allocations.

For the checked-in Gemma4 text configuration
(`Applications/CausalLM/models/gemma4/config.json`: `vocab_size_per_layer_input`
262144, `num_hidden_layers` 35, `hidden_size_per_layer_input` 256), Q4_0 uses
32 elements per 18-byte block (`nntrainer/tensor/q4_0_tensor.h`: `QK4_0` and
`block_q4_0`). The legacy packed lookup is therefore
`262144 * 35 * 256 / 32 * 18 = 1,321,205,760` bytes (about 1.23 GiB), while
one split layer table is `37,748,736` bytes (36 MiB). Sequential RAII
map/unmap gives an ideal table-backing mapping peak about 35x smaller. This is
only a lookup-table mapping-scope estimate: page granularity, other weights,
and the full-width projection weight/activation above are excluded.

Legacy behavior is retained when `ple_split=false`: the packed PLE weight and
optional `ple_file_name` sidecar LUT path remain in use. `ple_split=true` with
`ple_file_name` is rejected because a LUT sidecar is not a Tensor-backed weight
that can be split or virtually mapped. Windows constructs the split graph with
resident weights even when `fsu=true`; Tensor virtual mmap is not supported on
Windows.

The converter has a matching `--ple_split` artifact mode. It slices the source
HF PLE columns `[i*P:(i+1)*P]`, emits each layer's embedding after that layer's
PLE gate and before its PLE projection, and uses the same layer-local names for
both BIN positional output and safetensors header keys. `quantize.cpp` maps
both legacy and layer-local PLE names to `fc_dtype`, resolving the earlier P0
risk of Q4/Q6 byte-size and offset drift.

## Test coverage

`test/unittest/models/unittest_causallm_gemma4.cpp` now covers:

- resident split PLE graph names and allocation state with `ple_split=true`;
  on Windows this test also enables `fsu=true` and verifies the required
  resident fallback;
- rejection of `ple_split=true` combined with a PLE sidecar;
- platform-independent algebraic equivalence of the legacy packed and resident
  split PLE graphs with matching layer-column weights;
- FP32 resident split save to POSIX virtual split load, including a separate
  zero-PLE baseline that proves the non-zero PLE rows change prefill logits,
  using an orthogonal feature-1 PLE route to avoid RMSNorm scale cancellation,
  plus decode and virtual-state checks before/after inference; it also verifies
  that FSU BIN save is rejected before either the backing BIN or a separate
  BIN output destination is opened, while the backing bytes and subsequent
  prefill remain valid;
- Q4_0 resident split to virtual split parity, exercising the new dtype-map
  names and quantized PLE bytes;
- RAII cleanup after a PLE-specific invalid-token exception. The test keeps the
  primary vocabulary valid but sets a smaller PLE vocabulary, ensuring the
  virtual PLE lookup is activated before it throws.

## Changed files

- `Applications/CausalLM/models/gemma4/gemma4_causallm.{cpp,h}`: split PLE
  graph and POSIX virtual-residency selection.
- `Applications/CausalLM/layers/embedding_layer.{cpp,h}`: virtual-weight
  property and RAII activation/deactivation.
- `Applications/CausalLM/models/transformer.{cpp,h}`: `ple_split` config and
  embedding property plumbing.
- `nntrainer/layers/layer_devel.h`, `nntrainer/tensor/tensor.cpp`: preserve
  virtual-weight offsets and backing file descriptors during FSU load.
- `nntrainer/models/neuralnet.cpp`: reject FSU BIN, INI-with-BIN, and
  Safetensors saves before opening output, preventing truncation of
  virtual-weight backing storage.
- `Applications/CausalLM/res/gemma4/weight_converter.py`: split BIN and
  safetensors artifact mode.
- `Applications/CausalLM/quantize.cpp`: split PLE dtype entries.
- `test/unittest/models/unittest_causallm_gemma4.cpp`: regression coverage.

## Verification results

- Read `AGENTS.md` and inspected the MoE FSU, Gemma4 PLE, per-layer slice,
  converter, quantizer, and existing Gemma4 test paths.
- Ran clang-format 14.0.6 across all eleven changed C/C++ files, then reran
  `--dry-run --Werror`: passed. `git diff --check` also passed.
- Ran the bundled Python 3.12.13 `py_compile` on
  `Applications/CausalLM/res/gemma4/weight_converter.py`: passed.
- Ran AST plus dependency-stubbed converter walker and actual writer-output
  checks: passed. They verified `--ple_split` propagation, legacy/split names,
  layer-local gate-to-embedding-to-projection order, and exact column slices.
  The split BIN writer emitted 41 tensors / 960 bytes with exact two-slice and
  full-segment bytes; the Safetensors writer emitted 40 unique keys / 880 raw
  bytes with `[5,3]` local shapes, exact offsets, and exact two-slice payloads.
  The temporary output files were removed after the check.
- Root independent validation additionally reported tree-sitter parse errors
  of zero for changed `.cpp` files, no increase in header error nodes versus
  baseline, and a passing Ruff check for the converter.

## Remaining risks

- This Windows workspace lacks Meson, Ninja, MSVC, and a native C++ test
  runtime, so the C++ unit suite was not compiled or executed here.
- POSIX/Linux and Android still need an actual model-file run to validate mmap,
  lookahead scheduling, and Q4/Q6 kernels end to end.
- The converter checks above use a fake tensor/state model; a real HF Gemma4
  conversion followed by BIN and safetensors load remains required.
- Peak projection-side memory is still governed by the full-width
  `per_layer_input_projection` weight and `[B, S, L*P]` normalized activation;
  further partitioning or streaming of those objects is outside this change.
- Saving FSU BIN, INI-with-BIN, or Safetensors weights is deliberately
  unsupported; reload with `fsu=false` before saving a materialized model.
  Config-only INI save remains available.
