# Offline affine KV transfer

Use Python 3 with NumPy to learn a per-target-layer affine map from aligned
source and target CPU KV snapshots.

```sh
python tools/kv_transfer/kv_transfer.py fit dataset.json mapping --layout headwise --k 1
python tools/kv_transfer/kv_transfer.py apply mapping source-held-out predicted
python tools/kv_transfer/kv_transfer.py compare predicted target-held-out
python -m unittest discover -s tools/kv_transfer -v
```

The dataset manifest pairs snapshots for the same token IDs and positions.
Paths can be absolute or relative to the manifest. Give prompt variants from
the same document or template the same optional group label.

```json
{
  "version": 1,
  "pairs": [
    {"source": "snapshots/source-a", "target": "snapshots/target-a", "group": "document-a"},
    {"source": "snapshots/source-b", "target": "snapshots/target-b"}
  ]
}
```

For held-out diagnostics, collect at least two independent groups. The hash
of each group and the seed determine the held-out split before fitting. The manifest rejects duplicate
token sequences and requires complete prompt prefixes and their continuations
to share a group. Fit reports the prompt digests and token counts per split
and post-RoPE K/V RMSE, relative L2 and cosine on the held-out prompts.

Headwise fitting uses the corresponding source head from each selected layer
to predict each target head independently. It requires the same KV head count
in both models. The all-head layout concatenates all source heads from each
selected layer. Target heads have independent coefficients and share one
source covariance matrix. Give each target layer
the same number of source layers with `--k`, or supply explicit indices with
`--layer-map files.json` containing `{"source_layers": [[0, 1], [1, 2]]}`.
The default chooses the source layers nearest each target layer's relative
depth. Decreasing `--block-tokens` bounds the token batch. The memory budget
checks estimated working arrays, including the quadratic normal equations for
fit. Mapping and snapshot files are memory mapped.

The key fit removes the source and target split-half RoPE at their recorded
positions. The applied map rotates predicted keys with the target's RoPE.
Values are mapped directly. `--alpha` is the L2 weight penalty in the summed
least-squares objective. The intercept is unpenalized. This estimates cache
similarity on held-out prompts. Evaluate generated tokens separately through
the native target model.

## Fit with calibration-ranked source layers

This example uses the Ridge mapper from [Heo et al., §3](https://arxiv.org/html/2608.03893v1#S3)
with a fixed source-layer count of four for the Qwen3 CPU pilot. Pass a
manifest that contains only calibration snapshots.

```sh
OPENBLAS_NUM_THREADS=4 python tools/kv_transfer/kv_transfer.py fit calibration.json mapping \
  --layout all-head --layer-selection ols-r2 --k 4 --alpha 0.01 \
  --token-stride 4 --validation-fraction 0 --block-tokens 1024 --memory-mib 1024
```

The layer ranking fits one affine OLS probe per source layer, target layer,
cache plane and corresponding head on the calibration rows. Within a head,
the coefficient of determination uses the total centered variance over
all coordinates. The ranking averages over heads with nonzero target
variance and then over the available planes. Ties favor lower source layer
indices. Ridge uses the selected layers for both planes, with all source
heads as features and an independent output for each target head.

Stride four takes rows 0, 4, 8, and so on from each prompt. With 50 prompts
of 1,024 tokens, both selection and Ridge fitting use 12,800 observations.
`--validation-fraction 0` uses the full manifest for calibration. With a
positive validation fraction, held-out prompts only enter the reconstruction
diagnostics and are scored at full token resolution. Mapping metadata records
the stride, selected layers, ranking scores, split digests and observation
counts.

## Snapshot interchange

Each snapshot directory contains `metadata.json`, `keys.f32`, and
`values.f32`. The planes are headerless little-endian float32 arrays with
shape `[layer, token, kv_head, head_dim]` in C order. Metadata provides the
version, model ID, tensor geometry, positional encoding and token alignment.

```json
{
  "version": 1,
  "model_id": "Qwen/Qwen3-0.6B",
  "layers": 28,
  "tokens": 2,
  "kv_heads": 8,
  "head_dim": 128,
  "key_space": "post_rope",
  "rope": {"layout": "split_half", "theta": 1000000, "rotary_dim": 128},
  "token_ids": [100, 200],
  "positions": [0, 1]
}
```

The CPU bridge exports FP16 cache values widened to float32 and rounds
imported values to FP16. Its prefix path requires consecutive positions
beginning at zero. The Python tools also preserve arbitrary strictly
increasing positions for offline snapshots.

For the current CausalLM CPU bridge, set `NNTR_KV_CORE_CPU=1`,
`NNTR_KV_TRANSFER_MODEL_ID` to the intended model ID, and
`NNTR_KV_TRANSFER_ROPE_THETA` to the model's RoPE theta. Configure
`system_prompt.head_prompt` and `system_prompt.tail_prompt` in
`nntr_config.json`, and set
`system_prompt.kvcache.pre_computed_cache_path` to a new path for export. Run
`nntr_causallm` with `NNTR_KV_TRANSFER_EXPORT` set to the snapshot directory.
The export captures the head prompt. On import, set
`NNTR_KV_TRANSFER_IMPORT` to the mapped directory, set the same model's
`system_prompt.kvcache.sys_prompt_token_size` to the snapshot's token count,
and use an existing `pre_computed_cache_path` marker to enter CausalLM's
cache-load path. Pass the continuation text as the CLI input.

The model ID is an explicit operator-supplied identity. Use distinct IDs when
mixing revisions or local weight builds.

## Reproduce the Qwen3 CPU pilot

Run the pilot with a CPU bridge build and both native FP32 model directories.

```sh
python tools/kv_transfer/pilot.py \
  build-cpu/Applications/CausalLM/nntr_causallm \
  artifacts/models/qwen3-0.6b-fp32 artifacts/models/qwen3-1.7b-fp32 \
  artifacts/native-pilot --layout headwise --k 1 --alpha 1 --threads 8
```

The runner uses the eight passages in `pilot_prompts.json`. It collects pairs
under a new output path, reports held-out error against native target caches,
and checks one held-out continuation with the cold target, a native target
snapshot import, the learned mapping and a direct source-plane transplant.
The direct-copy result is an experimental control and is labeled as such in
its metadata. The runner records generated token IDs in `result.json` and
checks that the cold run and native target snapshot import generate the same
IDs. Pass `--generation-tokens 0` to measure cache errors alone. The generated
logs, snapshots and learned weights stay under the output path.

See [the recorded pilot results](pilot-results.md).
