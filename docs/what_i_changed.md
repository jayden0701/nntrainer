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
