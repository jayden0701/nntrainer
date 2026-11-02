# Resident FP32 tied LM head on CUDA

This builds on the host-head optimization in `4d4e2ac2`. In inference with
`NNTR_ENGINE=cuda`, a tied LM head with FP32 weights and FP16 activations
uploads its `[vocab, hidden]` matrix once to device memory. A CUDA GEMV reads
FP16 input, accumulates in FP32, and produces FP16 or FP32 logits. Only the
input row and output logits are copied each step; the weight stays resident.

The layer owns its weight and staging buffers. Destruction, finalization and
both weight-read interfaces invalidate the cache. Training also invalidates
it. Weight contents must remain immutable between these lifecycle events;
out-of-band weight mutation requires reloading/refinalizing the layer.

The GPU path defaults on. Set `NNTR_CUDA_FP32_HEAD=0` to use the previous
optimized host path. Allocation/kernel failures fall through to existing
dispatch. CPU and OpenCL engine selection is preserved. Stream capture is
excluded from this new path: this change does not add CUDA graph support.
Input/output staging accepts host-addressable and device-only buffers.

## Validation

Ubuntu WSL2, GCC 14, CUDA 13.0, RTX 5070; Qwen3 0.6B with FP16 body and
activations and FP32 tied embedding/head. Native Windows/Android/Tizen/Yocto
builds were not executed.

- Full Meson/Ninja CUDA build passed.
- CPU-engine Meson suites in the CUDA build: 54/54 passed.
- CUDA-engine tied-head tests: 16/16 passed, including direct kernel/reference
  comparisons for both output dtypes, non-warp-aligned dimensions, changing
  inputs with a persistent weight, device-only I/O, invalid arguments,
  allocation lifetime and capture refusal.
- Existing CUDA LayerNorm/GELU/quantized-FC suites: 3/3 passed.
- The changed layer compiles with CPU/CUDA and FP16 enabled/disabled.
- clang-format 14 and `git diff --check` passed.

For the geography and arithmetic prompts, greedy token IDs equal the host
path on all 11 observed logits steps. All values are finite. Maximum absolute
logit difference is 0.0078125; GPU summation order differs from CPU BLAS.
GPU sync/async outputs are bitwise equal on these prompts. These observations
do not guarantee token equality for every prompt/model.

## Measured performance and memory

Both profiles use the same binary/model, selected with
`NNTR_CUDA_FP32_HEAD=0/1`. Workload: 32 context tokens, 12 decode tokens, fixed
non-EOS IDs, four host threads, one BLAS thread, eager execution and
host-addressable activations. One warmup and three measured trials per
profile (36 decode intervals). Warm measurements reuse uploaded weights;
the first projection additionally pays the upload cost.

| Median measurement | Optimized host | Resident CUDA |
| --- | ---: | ---: |
| Decode interval | 135.581 ms | 103.584 ms |
| Run start to first logits | 299.539 ms | 291.220 ms |
| Complete fixed workload | 1940.516 ms | 1590.651 ms |

Decode speedup is 1.31x in this workload, relative to the already optimized
host path. Additional device storage for this model is 622,939,648 bytes
(about 594 MiB): FP32 weights plus reusable input/output staging.

Validation commands, with the CUDA build configured at `build-cuda`:

```bash
NNTR_ENGINE=cpu OPENBLAS_NUM_THREADS=1 meson test -C build-cuda --no-rebuild --num-processes 4 --print-errorlogs
NNTR_ENGINE=cuda OPENBLAS_NUM_THREADS=1 build-cuda/test/unittest/layers/unittest_layers --gtest_filter='TieWordEmbeddingLmHead.*:OutputTypes/TieWordEmbeddingFp32HostHead.*'
NNTR_ENGINE=cuda OPENBLAS_NUM_THREADS=1 meson test -C build-cuda --no-rebuild --num-processes 1 --print-errorlogs unittest_cuda_kernels_layernorm unittest_cuda_kernels_gelu unittest_cuda_kernels_fc_qs4cx
```

Local raw reports, requests, probes, logs and benchmark samples are retained in
`validation/gpu-head/` in the validation worktree. They are not build inputs.
