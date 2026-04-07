# Gemma4 NNTrainer validation checklist

## Implemented in this pass

- [x] Text-only Gemma4 causal LM class skeleton and factory registration
- [x] Layer-type aware attention path (`sliding_attention` / `full_attention`)
- [x] Full-attention `global_head_dim` projection width selection
- [x] Per-layer-type RoPE theta read from `rope_parameters`
- [x] `attn_logit_softcapping` passthrough to MHA core
- [x] Decoder/token RMSNorm uses NNTrainer `rms_norm` (Gemma4 uses direct scale weight, no `1+weight` offset)
- [x] Attention `v_norm` (`Gemma4RMSNorm(with_scale=False)`) via `reshaped_rms_norm` with `use_gamma=false`
- [x] Per-layer input embedding branch (embedding + projection + per-layer slice + per-layer gate/projection path)

## Known gaps / TODO


- [ ] Exact per-layer input scalar factors (`hidden_size^-0.5` and `2^-0.5`) are not yet applied in-graph.
- [ ] Double-wide MLP behavior for KV-shared tail layers is not implemented.
- [ ] Final LM-head logit softcapping (`final_logit_softcapping`) is not implemented in NNTrainer graph yet.
