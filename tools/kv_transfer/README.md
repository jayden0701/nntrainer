# CPU Ridge and CacheBridge

The CausalLM CLI transfers a shared prompt prefix from one Qwen3 model to
another through the CPU KV cache manager. Ridge maps all source heads from
the selected layers. CacheBridge uses the corresponding source head and
optionally calibrates the value map against target attention readouts.
See the [experiment report](../../docs/kv-transfer-results.md) for accuracy,
latency and generation results.
[Closed-form Ridge](https://arxiv.org/abs/2608.03893) · [CacheBridge v2](https://arxiv.org/html/2609.00891v2)

The CLI requires batch-one, full-context Qwen3 models, identical tokenizer
files, matching KV head geometry, FP16 KV storage and unscaled split-half
RoPE. The tested model pair is Qwen3-0.6B to Qwen3-1.7B with FP32 weights.
This entry point runs two resident models in one CPU process.

## Build and prepare models

Follow the [build guide](../../docs/getting-started.md) for system dependencies
and initialize the repository's submodules. If the tokenizer archive is
missing, build the in-tree tokenizer and pass
`-Dcausallm-tokenizer-lib=/path/to/libtokenizers_c.a` to Meson.

```sh
meson setup build -Denable-transformer=true -Denable-opencl=false \
  -Denable-cuda=false -Denable-fp16=false
ninja -C build Applications/CausalLM/nntr_causallm
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r tools/kv_transfer/requirements-reference.txt pyarrow==26.0.1
python -m unittest discover -s tools/kv_transfer -v
```

Use the [Qwen3 layer-graph converter](../../Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py)
with `--data_type float32`. Place each model's converted weights, tokenizer and
CausalLM configuration in its own directory. `model_file_name` in
`nntr_config.json` points to the converted weights. Both models need a
`max_seq_len` of at least 4,352 to run the 4,096-token prefix examples.
`BINARY` is the built CLI. `SOURCE` and `TARGET` identify the native model
directories. `HF_TARGET` is the matching target checkpoint before conversion.

## Fit both layouts on the same data

The capture command runs each model on identical token IDs. Its output
`dataset.json` pairs aligned K/V snapshots. `prepare_paper_data.py` pins the
dataset revisions and records the selected FineWeb-Edu rows and token IDs.

```sh
export OPENBLAS_NUM_THREADS=4
python tools/kv_transfer/prepare_paper_data.py "$TARGET/tokenizer.json" \
  artifacts/data --calibration-count 120 --seed 20261117
python tools/kv_transfer/collect_calibration.py "$BINARY" "$SOURCE" "$TARGET" \
  artifacts/data/calibration.json artifacts/snapshots --threads 4

python tools/kv_transfer/kv_transfer.py fit artifacts/snapshots/dataset.json \
  artifacts/ridge --layout all-head --layer-selection ols-r2 --k 4 \
  --alpha 0.01 --token-stride 4 --validation-fraction 0 \
  --block-tokens 1024 --memory-mib 1024
python -c 'import json; from pathlib import Path; p = Path("artifacts/ridge/metadata.json"); print(json.dumps({"source_layers": json.loads(p.read_text())["source_layers"]}))' \
  > artifacts/layers.json
python tools/kv_transfer/kv_transfer.py fit artifacts/snapshots/dataset.json \
  artifacts/compact --layout headwise --layer-map artifacts/layers.json \
  --alpha 0.01 --token-stride 4 --validation-fraction 0 \
  --block-tokens 1024 --memory-mib 1024
```

This ranks layers on the supplied calibration data. To reproduce the
50-to-120 sequence ablation, select the layers with the 50-sequence fit first
and pass that same `--layer-map` to both 120-sequence fits. Positions
`0, 4, 8, …` enter the fit, so 120 sequences of 1,024 tokens provide 30,720
observations. Keys are fitted before RoPE and rotated at the target positions.
Values are mapped directly. The intercept is unpenalized.

## Calibrate and evaluate CacheBridge

Point `QUALITY_TRAIN` and `QUALITY_DEV` to the QuALITY v1.0.1 JSONL files.
The latency and accuracy drivers record the binary and input hashes and
preserve raw per-request metrics. [QuALITY v1.0.1 dataset](https://github.com/nyu-mll/quality/tree/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1)

```sh
python tools/kv_transfer/cachebridge_data.py prepare "$QUALITY_TRAIN" "$TARGET" \
  artifacts/quality-cal.json --documents 24 --max-tokens 4096
python tools/kv_transfer/cachebridge_data.py prepare "$QUALITY_DEV" "$TARGET" \
  artifacts/quality-dev.json --documents 24 --max-tokens 4096
python tools/kv_transfer/cachebridge_data.py capture artifacts/quality-cal.json \
  "$BINARY" "$SOURCE" "$TARGET" artifacts/quality-caches --threads 4
python tools/kv_transfer/cachebridge_calibrate.py \
  --initial artifacts/compact --initial-data artifacts/snapshots/dataset.json \
  --manifest artifacts/quality-cal.json --caches artifacts/quality-caches \
  --hf-target "$HF_TARGET" --output artifacts/cachebridge

python tools/kv_transfer/cachebridge_benchmark.py quality \
  --binary "$BINARY" --source "$SOURCE" --target "$TARGET" \
  --data artifacts/quality-dev.json --calibration artifacts/quality-cal.json \
  --output artifacts/quality-results --reuse-history \
  --mapper ridge=artifacts/ridge --mapper compact=artifacts/compact \
  --mapper cachebridge=artifacts/cachebridge
```

The calibrator uses the same query states and newly computed query K/V for
reference and mapped readouts. It folds a constrained V correction into the
FP32 affine arrays and forwards the corrected readout to the next layer.
Metadata records the calibration manifest and per-layer readout error and
update energy.

For handoff timing, use the `latency` task with the same model directories and
mapper arguments. For HellaSwag, use the `hellaswag` task with `--data` (the
`hellaswag.jsonl` from preparation), `--public-report` and `--hf-cases`.
The [pinned public Ridge evaluation](https://github.com/alesha-pro/vllm-cross-model-kv/tree/2e27dba)
provides the case ordering. `hf_public_benchmark.py` writes the reference case
manifest, and `native_public_benchmark.py` evaluates the imported public mapper.
Those scripts' `--help` lists the required inputs. Use `--start` and `--limit`
for independent shards. The public-order evaluation normalizes by candidate
token count. The separate `benchmark.py` paper-style evaluation normalizes
ARC-Challenge and HellaSwag by candidate character count and reports
WikiText-2 perplexity.

## CLI request format

Run `NNTR_ENGINE=cpu "$BINARY" "$TARGET" --cachebridge requests.json`.
Supply real token IDs from the shared tokenizer. Mapper paths point to
directories containing `metadata.json`, `weights.f32` and `bias.f32`.

```json
{
  "source_model": "artifacts/models/qwen3-0.6b-fp32",
  "source_id": "Qwen/Qwen3-0.6B",
  "target_id": "Qwen/Qwen3-1.7B",
  "block_tokens": 128,
  "mappers": {
    "ridge": "artifacts/ridge",
    "cachebridge": "artifacts/cachebridge"
  },
  "requests": [{
    "id": "example",
    "history": [100, 200],
    "query": [300],
    "modes": ["cold", "exact", "ridge", "cachebridge"],
    "max_new_tokens": 32,
    "stop_ids": [151643, 151645],
    "seed": 42
  }]
}
```

`cold` pre-fills the complete target prompt. `exact` pre-fills history and
query separately on the target. The named modes apply their affine map to the
source's committed history. The target publishes the mapped prefix only
after all cache layers are written. A failed install emits a `fallback`
reason and recomputes the complete target prompt.

Generation uses the native sampler at temperature 0.6, top-k 20 and top-p 0.95.
The RNG starts from the same seed for each mode. Choose the stop IDs from the
target's generation configuration. The output includes token IDs, EOS or
length termination, TTFT and E2E in milliseconds. For choice scoring, omit
generation controls and provide `options` as an array of nonempty token
arrays. `reuse_history: true` is available for one scoring mode per request.

TTFT starts with the source KV ready and covers transfer or target prefill
through the first token. The request reports source preparation separately.
An empty `mappers` object runs target-only controls with one loaded model.

## Snapshot interchange

Offline tools use `metadata.json`, `keys.f32` and `values.f32` snapshots.
Each plane is little-endian FP32 in layer, token, KV head, head dimension
order. Metadata records model identity, geometry, RoPE, token IDs and
positions. Native export widens FP16 rows and native import restores FP16.
The native bridge requires contiguous positions starting at zero. Model IDs
are supplied by the caller. Use distinct IDs for different weight revisions.

Run `kv_transfer.py apply MAPPER SOURCE_SNAPSHOT OUTPUT` to produce a
target-compatible snapshot. Use `kv_transfer.py compare OUTPUT EXACT_TARGET`
to compare cache reconstruction. The `pilot.py` runner exercises native export,
exact import, mapped import and direct copy on eight small prompts.
