# AGENTS.md

## Project goal
Port Hugging Face `modeling_gemma4.py` into an NNTrainer-native implementation by learning the mapping from:
1. Hugging Face Gemma3 implementation
2. Existing NNTrainer Gemma3 implementation
3. Upstream NNTrainer architecture and coding conventions

Do not invent architecture changes unless they are required by Gemma4 semantics.

## Source-of-truth priority
When there is a conflict, use this priority:
1. `nntrainer/Applications/CausalLM/models/gemma4/modeling_gemma4.py`
2. `nntrainer/Applications/CausalLM/models/gemma3/modeling_gemma3.py`
3. `nntrainer/Applications/CausalLM/models/gemma3/*`

## Required workflow
For any non-trivial change:
1. always summarize it on docs/what_i_changed.md

## Porting rules
- Preserve Gemma4 semantics exactly unless NNTrainer limitations make exact parity impossible.
- Reuse NNTrainer Gemma3 patterns wherever possible.
- Keep naming close to NNTrainer conventions, not Hugging Face conventions.
- Preserve tensor shape comments near all attention / projection / RoPE / cache-related code.
- Try to reuse custom layers.
- Do not silently drop features. If a feature is unsupported, leave a clear TODO and document it in `docs/validation_checklist.md`.
- Do not refactor unrelated NNTrainer code.
