# What I changed

## Gemma4 NNTrainer text CausalLM port (initial)

- Added a new Gemma4 NNTrainer model implementation at:
  - `Applications/CausalLM/models/gemma4/gemma4_causallm.h`
  - `Applications/CausalLM/models/gemma4/gemma4_causallm.cpp`
  - `Applications/CausalLM/models/gemma4/meson.build`
- Wired Gemma4 model sources into the CausalLM build graph:
  - `Applications/CausalLM/models/meson.build`
  - `Applications/CausalLM/jni/Android.mk`
- Registered `Gemma4ForCausalLM` in app entry points and factory wiring:
  - `Applications/CausalLM/main.cpp`
  - `Applications/CausalLM/quantize.cpp`
  - `Applications/CausalLM/api/causal_lm_api.cpp`

## Porting details

- Reused Gemma3 NNTrainer transformer block topology (RMSNorm -> attention -> RMSNorm -> residual -> RMSNorm -> GeGLU MLP -> RMSNorm -> residual) as baseline.
- Added Gemma4-specific attention configuration logic:
  - Layer-type aware handling (`sliding_attention` vs `full_attention`)
  - `global_head_dim` usage for full attention
  - `num_global_key_value_heads` when `attention_k_eq_v` is enabled on full-attention layers
  - Per-layer-type RoPE theta from `rope_parameters` (`full_attention` and `sliding_attention`)
  - Existing `attn_logit_softcapping` passthrough to `mha_core`
- Preserved tensor-shape comments in attention path around projection/norm/attention/O projection.

## Limitations tracked

- Added explicit TODOs in code and recorded unsupported/partial Gemma4 features in `docs/validation_checklist.md`.

## Follow-up refinement

- Clarified RMSNorm mapping: Gemma4 uses direct scale RMSNorm, so NNTrainer `rms_norm` is intentionally used for decoder/token norms (not Gemma3 `1+weight` offset style).
- Implemented Gemma4 attention `v_norm` parity by extending existing `reshaped_rms_norm` with optional `use_gamma=false` and wiring it in Gemma4 attention inputs to MHA core.
- Implemented Gemma4 per-layer input embedding path (`hidden_size_per_layer_input`) with packed per-layer embedding/projection precompute, layer-wise slice selection, and per-layer gate/projection residual block inside each decoder layer.
- Added TODO note that exact per-layer scalar factors are not yet applied in-graph.

## Follow-up: always-on per-layer input branch

- Removed optional per-layer-input control flow in Gemma4 and now always constructs/applies the per-layer input branch in model graph and decoder blocks.
- Added explicit config validation in `setupParameters()` to require:
  - `hidden_size_per_layer_input > 0`
  - `vocab_size_per_layer_input > 0`
- This matches Gemma4 expectation that per-layer input is always enabled.

## Added ScalarMultiplyLayer

- Added a new custom layer `ScalarMultiplyLayer` that multiplies input tensor by a scalar value:
  - `Applications/CausalLM/layers/scalar_multiply.h` - Header file with layer definition
  - `Applications/CausalLM/layers/scalar_multiply.cpp` - Implementation file
- The layer accepts a `multiplier` property (float) that is multiplied with all elements of the input tensor.
- Updated `Applications/CausalLM/layers/meson.build` to include the new layer in the build system.
- Usage example in config: `type=scalar_multiply | multiplier=0.5`

### ScalarMultiplyLayer use_weight property

- Added `use_weight` property (bool, default: false) to `ScalarMultiplyLayer`.
- When `use_weight=true`:
  - The layer requests a single-element weight tensor named `scalar_multiplier` from the weight file.
  - The multiplier value is loaded from the weight file instead of the `multiplier` property.
- When `use_weight=false` (default):
  - The layer uses the `multiplier` property value as before.
- Usage example with weight file:
  ```
  type=scalar_multiply | use_weight=true
  ```
  The weight file should contain a weight tensor named `scalar_multiplier` with a single float value.

## Gemma4 shared attention wiring for KV-shared layers

- Added `createSharedAttention()` in Gemma4 model implementation to support KV-shared tail layers when direct KV cache sharing is unavailable in NNTrainer.
- Updated decoder block construction to detect KV-shared layers using `num_kv_shared_layers` and map each shared layer to the last non-shared layer with the same `layer_type`.
- For shared layers, the attention block now:
  - creates only the current layer Q projection + Q RMSNorm, and
  - reuses `layer{shared_kv_layer_id}_k_norm` and `layer{shared_kv_layer_id}_v_norm` as `mha_core` K/V inputs.
- This preserves Gemma4 shared-attention behavior by graph connection to normalized K/V tensors from the source layer.

## Gemma4 double-wide MLP for KV-shared tail layers

- Implemented Gemma4 `use_double_wide_mlp` behavior in NNTrainer Gemma4 text MLP path.
- Added model state for `use_double_wide_mlp` and config parsing in `setupParameters()`.
- Updated `createMlp()` so layers in the KV-shared tail region (`num_kv_shared_layers`) use `2x intermediate_size` when `use_double_wide_mlp=true`, matching HF `Gemma4TextMLP` semantics for shared layers only.

## Gemma4 parity fixes for PyTorch output mismatch (non-RoPE path)

- Added config sanitization for HF Gemma4 multimodal checkpoints: when `text_config` exists in `config.json`, Gemma4 NNTrainer now lifts missing text fields to top-level keys before normal Transformer parameter parsing.
- Fixed attention score scaling mismatch:
  - HF Gemma4 text attention uses `scaling=1.0` after q/k RMSNorm.
  - NNTrainer `mha_core` backend applies an internal `/sqrt(head_dim)` on QK.
  - changed internel so it doesn't divide by `sqrt(head_dim)`
- Added support for Gemma4 `final_logit_softcapping` in NNTrainer output head:
  - apply `logits = tanh(logits / softcap) * softcap` after lm_head.
  - This aligns with HF Gemma4 `Gemma4ForCausalLM` forward logic.

## Fix: Gemma4 final-logit softcapping for decode-step row shape

- Added a dedicated custom layer `logit_softcapping` to avoid decode-time row-range mismatch in chained `scalar_multiply -> activation -> scalar_multiply` logic.
- New layer files:
  - `Applications/CausalLM/layers/logit_softcapping.h`
  - `Applications/CausalLM/layers/logit_softcapping.cpp`
- New layer properties:
  - `activation_type` (activation enum, e.g., `tanh`)
  - `apply_rows` (apply from front rows only)
  - `softcap_value` (divide/apply activation/multiply)
- Layer behavior:
  - computes `y = activation(x / softcap_value) * softcap_value`
  - applies only to the first `apply_rows` rows (front), passes other rows through
  - supports both full forwarding and incremental forwarding path without assuming `to-from` equals requested rows.
- Wired `logit_softcapping` into Gemma4 output head (`output_of_causallm_softcapped`) and removed the previous 3-layer softcap chain in Gemma4 graph construction.
- Registered the new custom layer factory in Gemma4 custom layer registration.
- Updated build wiring:
  - `Applications/CausalLM/layers/meson.build`
  - `Applications/CausalLM/meson.build`
  - `Applications/CausalLM/jni/Android.mk`

## Gemma4 RoPE parity update (proportional + multi-type cache)

- Implemented Gemma4 text RoPE in NNTrainer `mha_core` path and enabled it for Gemma4 attention blocks:
  - Gemma4 attention now sets `use_rope=true` and passes per-layer-type RoPE properties (`rope_theta`, `rope_scaling_type`, `rope_partial_rotary_factor`) into `mha_core`.
  - Added parsing of `partial_rotary_factor` from Gemma4 `rope_parameters` for both `full_attention` and `sliding_attention`.
- Added `proportional` RoPE support in `mha_core` to match Gemma4 full-attention behavior:
  - computes rotated frequencies for the configured rotary fraction and zero-fills remaining non-rotary dimensions.
  - preserves HF behavior where returned RoPE dimension remains `head_dim`.
- Updated `mha_core` RoPE cache behavior to support multiple RoPE types/configurations in one model run:
  - replaced single static cos/sin buffers with keyed caches (FP32/FP16) so `default` and `proportional` caches can coexist safely.
  - cache key includes rope type and core parameters (`head_dim`, `seq_len`, `theta`, scaling factors).
- Restored Gemma4 attention score scaling parity in shared-KV attention path by applying `sqrt(head_dim)` pre-scale to `Q` before `mha_core`, matching non-shared path and HF `scaling=1.0` semantics.

## Gemma4 prefill skip optimization for KV-shared tail + LM head

- Added Gemma4 model-side wiring to apply `skip_prefill=true` only when `nntr_cfg.skip_prefill` is enabled.
- Added helper logic in Gemma4 model builder to detect KV-shared tail layers (`num_kv_shared_layers`) and tag expensive per-layer ops in those layers with `skip_prefill`:
  - attention projections/norm/core/output projection
  - MLP FC projections
  - additional RMSNorm-heavy points in decoder block and final output norm
- Applied `skip_prefill` to Gemma4 `output_of_causallm` head construction as well.
- Extended both LM-head implementations to actually honor `skip_prefill` in incremental prefill path:
  - `lm_head` custom layer now reads `skip_prefill` and early-returns for prefill.
  - `tie_word_embeddings` (lm_head mode) now reads `skip_prefill` and early-returns for prefill.
- Scope/intent:
  - keeps decode/generation semantics unchanged (`skip_prefill` only triggers when `from == 0`),
  - avoids unnecessary prefill-only compute where logits are not consumed and shared-tail layers are not needed for KV cache construction.

## Gemma4 skip-prefill expansion for non-FC layers + final softcap

- Expanded Gemma4 `skip_prefill` tagging in KV-shared tail layers beyond FC/RMSNorm/attention blocks to also skip inexpensive-but-unnecessary ops that are only used inside skipped shared-tail decode paths:
  - residual additions in decoder block (`post_attention`, `decoder_output_base`, `decoder_output`)
  - per-layer-input path ops (`per_layer_slice`, `activation`, `multiply`)
  - layer output scaling (`scalar_multiply`)
  - MLP activation/multiply (`ffn_gate_gelu`, `ffn_geglu`)
- Added `skip_prefill` to Gemma4 final `logit_softcapping` layer so output softcap is skipped together with LM head during prefill.
- Effect:
  - prefill avoids extra graph work in KV-shared tail layers and final logits post-processing,
  - decode behavior remains unchanged because `skip_prefill` applies only on prefill path.
