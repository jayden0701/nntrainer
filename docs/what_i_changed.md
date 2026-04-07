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
