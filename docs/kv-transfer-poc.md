# CPU KV transfer proof of concept

NNTR_KV_CORE_CPU=1 selects synchronous contiguous core-managed cache for batch-one CPU inference with uniform, full-context, unshared KV layers. The legacy path remains the default. This proof of concept supports Qwen3 with unscaled split-half RoPE. It does not implement paged model inference or prefix sharing. The core manager does not publish imported tokens into a prefix cache.

With the existing system-prompt KV save mode, set NNTR_KV_TRANSFER_EXPORT to a new or empty directory, NNTR_KV_TRANSFER_MODEL_ID to the model identifier, and NNTR_KV_TRANSFER_ROPE_THETA to its configured theta. For import, use the existing system-prompt KV load mode and set NNTR_KV_TRANSFER_IMPORT to the snapshot directory, with the same model ID and theta parameters for the target model. Set sys_prompt_token_size to the snapshot token count and provide only the continuation as the prompt. The caller is responsible for ensuring its continuation is paired with the snapshot token IDs. Model IDs are caller assertions, not hashes of model weights.

The directory contains metadata.json, keys.f32, and values.f32. Each plane is little-endian float32 in layer, token, KV head, head dimension order. Keys are after RoPE, values are unrotated, and positions and token_ids list the corresponding prefix. The metadata specifies the model, geometry and RoPE interpretation. Import checks these against the target contract and rejects truncated, extra, non-finite or FP16-overflowing data. The target metadata must name the target model after translation.

An FP32 model here uses FP32 weights and activations but stores KV in FP16. Export widens those FP16 cache values to float32; import rounds them to FP16. Consequently a snapshot does not recover precision lost at the original cache write. Snapshot restore atomically replaces committed cache rows and clears known prefix-token bookkeeping; it never runs a fabricated inference step to read history.

## Local validation

With the locally converted Qwen3 0.6B FP32 model and a seven-token prefix, CPU full prefill on the legacy cache, CPU full prefill on the core cache, F32 snapshot import into the core cache, and native-file import into the core cache produced identical 17 emitted greedy token IDs for the same continuation. The common response began: The answer is Paris. The generated ID files and isolated configurations are under artifacts/corrected-native in the local experiment workspace. The ordinary greedy arithmetic prompt also returned 2 + 2 = 4. These checks establish an identity round trip for this model and prompt; they do not establish cross-model transfer quality.

For a standalone export, create a model configuration whose system_prompt is:

~~~json
{
  "head_prompt": "The capital of France is Paris.\n\n",
  "tail_prompt": "",
  "kvcache": {"pre_computed_cache_path": "/tmp/new-native-cache"}
}
~~~

The native cache path must not yet exist. Then run:

~~~sh
NNTR_ENGINE=cpu NNTR_KV_CORE_CPU=1 NNTR_KV_TRANSFER_MODEL_ID=Qwen/Qwen3-0.6B NNTR_KV_TRANSFER_ROPE_THETA=1000000 NNTR_KV_TRANSFER_EXPORT=/tmp/new-snapshot build-cpu/Applications/CausalLM/nntr_causallm /path/to/model ignored
~~~

For a continuation, retain the system prompt and set sys_prompt_token_size in the kvcache object to the token count in metadata.json. Set NNTR_KV_TRANSFER_IMPORT to the snapshot directory in place of NNTR_KV_TRANSFER_EXPORT and pass the continuation as the final argument. The native cache path must exist to select load mode; its data are not read when snapshot import is enabled. The native cache file is read when the import environment variable is absent.
