# Gemma4 NNTrainer validation checklist

## Implemented in this pass

- [x] Text-only Gemma4 causal LM class skeleton and factory registration
- [x] Layer-type aware attention path (`sliding_attention` / `full_attention`)
- [x] Full-attention `global_head_dim` projection width selection
- [x] Per-layer-type RoPE theta read from `rope_parameters`
- [x] `attn_logit_softcapping` passthrough to MHA core

## Known gaps / TODO

- [ ] Exact `attention_k_eq_v` semantics (shared K/V projection weights for full-attention layers) are not yet implemented; current path keeps separate V projection.
- [ ] KV shared-layer cache behavior (`num_kv_shared_layers`) is not implemented.
- [ ] Per-layer input branch (`hidden_size_per_layer_input` and related projections) is not implemented.
- [ ] MoE branch (`enable_moe_block`, router, experts) is not implemented.
- [ ] Double-wide MLP behavior for KV-shared tail layers is not implemented.
- [ ] Final LM-head logit softcapping (`final_logit_softcapping`) is not implemented in NNTrainer graph yet.
