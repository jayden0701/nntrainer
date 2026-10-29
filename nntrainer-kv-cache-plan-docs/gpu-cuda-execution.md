# CUDA KV 구현·실행 기록 — PC/WSL (2026-10-29)

> 이후 G6 decode replay 구현은 [별도 실행 기록](gpu-cuda-replay-execution.md)에 있다. 아래는 eager 단계 당시의 기록이다.

PR #4460 head `25230aafe1a230e8dc497921d1149d52f6eabdad` 위에서
[GPU 계획](gpu-extension-plan.md)의 G0–G4와 G5의 첫 Qwen 검증을 구현·실행했다.
지원 profile은 **CUDA eager / FP16 native KV / batch=1 / linear / Qwen3**다.
실제 attention의 manager backing 직접 읽기(A)와 최종 K/V producer 직접 쓰기(B)를
확인했다. 새 경로는 `NNTR_CUDA_KV_CORE=1`로 선택한다.

Android 기기가 없는 조건에 맞춰 현재 PC에서 실행했다.
G6 async publication/capture/replay, OpenCL, Android, NPU 및 실제 모델의
CPU↔GPU backend 전환은 후속이다.

## 1. 환경과 기준 분리

| 항목 | 실제 값 |
|---|---|
| 위치 / branch | WSL Ubuntu `/home/jayden/nntrainer` / `kv-manager-gpu-plan-pr4460` |
| OS / kernel | Ubuntu 26.04.1 / WSL 6.18.54.1 |
| GPU | RTX 5070, sm_120, 12,820,480,000 bytes |
| Windows driver / CUDA driver API | 610.88 / 13030 |
| Toolkit / runtime / NVRTC | `/usr/local/cuda-13.0` / 13000 / 13.0 |
| packages | CUDA 13.0.96, NVRTC 13.0.88, cuBLAS 13.1.1.3 |
| compiler / formatter | GCC/G++14 / clang-format 14.0.6 |
| model Python | torch 2.15.0+cpu, transformers 5.21.0, numpy 2.5.4 |

NVIDIA WSL 저장소의 `cuda-cudart-dev-13-0`, `cuda-nvrtc-dev-13-0`,
`libcublas-dev-13-0`, `cuda-nvcc-13-0`를 설치했다. Linux GPU driver는 설치하지 않았다.
설치 경로의 근거는 [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)와
[WSL package repository](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/)다.
ripgrep, gdb, python3-venv도 준비했다.

- `/home/jayden/.cache/nntr-tools`: clang-format14, ruff.
- `/home/jayden/.cache/nntr-gpu-model`: torch CPU, transformers, numpy,
  huggingface_hub, safetensors. HF 기준 모델은 CPU FP32로 실행한다.
- 필요한 xgrammar submodule을 recursive init했다. vendored source는 편집하지 않았다.

기존 `build/`와 `/usr/local` NNTrainer 설치물은 다른 revision이므로 사용하지 않았다.
Windows `C:\nntrainer`, 기존 branch, untracked `subprojects/CLBlast/`는 보존했다.
commit/push/원격 PR 변경은 하지 않았다.

| build | 용도 |
|---|---|
| `build-kv-gpu-cuda-baseline` | 변경 전 HEAD의 CUDA FP16 runtime 및 old-header observer |
| `build-kv-gpu-cuda` | 새 CUDA KV runtime / graph / 공개 모델 검증 |
| `build-kv-gpu-cpu-regression` | CUDA/OpenCL/FP16 비활성 CPU transformer/model 회귀 |
| `build-kv-gpu-core-asan` | app/transformer 비활성 core ASan/UBSan/LSan |

각 build의 `validation/`에 로그·JSON·observer·ABI metadata를 보존했다.
build와 모델은 ignored local artifact다. baseline은 현재 source로 덮어쓰지 않도록
Ninja rebuild 없이 `--no-rebuild`로 실행한다.

## 2. G0 probe

새 `tools/kv_cache/cuda_capability_probe.cpp`를 빌드해 실행했다.

```bash
g++-14 -std=c++17 tools/kv_cache/cuda_capability_probe.cpp \
  -I/usr/local/cuda-13.0/include -L/usr/local/cuda-13.0/lib64 \
  -Wl,-rpath,/usr/local/cuda-13.0/lib64 -L/usr/lib/wsl/lib \
  -lcudart -lnvrtc -lcuda \
  -o build-kv-gpu-cuda-baseline/validation/cuda_capability_probe
build-kv-gpu-cuda-baseline/validation/cuda_capability_probe
```

NVRTC sm_120 CUBIN compile/load/launch, device allocation의 offset pattern/guard,
host-mapped CPU→GPU 변경→checked wait→CPU 검증 **8회 PASS**다.
managedMemory=1, concurrentManagedAccess=0, canMapHostMemory=1, UVA=1,
hostNativeAtomics=0이며 mapped host/device 주소가 같았다.
별도 storage 테스트도 mapped 부분 영역 변경과 인접 영역을 확인한다.

이는 **pattern/storage handoff**다. 작은 attention/공개 모델 backend 전환,
physical migration 0을 증명한 것은 아니다.
strict operations는 Device/HostMapped만 허용하고 Managed는 coherence 별도 검증까지 거절한다.
원본: `build-kv-gpu-cuda-baseline/validation/capability.json`.

## 3. 구현과 연결

| 영역 | 결과 |
|---|---|
| allocation | 실제 kind/base/extent/device/context 보고. CUDA fallback과 interior pool slice도 실제 결과로 기술 |
| ABI | nonvirtual `MemAllocator::describe()` + optional `AllocationReporter`. 기존 MemAllocator vtable 및 ResidencyClass 값 유지 |
| pool/owner | 실제 descriptor로 host-addressable/SVM metadata 결정. CUDA mapped를 OpenCL SVM으로 표시하지 않음. Tensor alias가 storage/allocator/ops 수명 유지 |
| storage | backend zero/read/write/checked wait. 요청한 kind/context/range/alignment가 다르면 정리 후 거절 |
| step | native address와 host dereference API 분리. producer ready와 commit 전에 실제 완료 확인. 실패 시 position 유지, abort/reset 처리 |
| K/attention | GPU RoPE가 final cache slice에 씀. 실제 attention의 backing/read extent 검사와 history mirror 거절 |
| V | Qwen3가 `layerN_wv` final writer 선언. planner allocation 이후 실제 FC 호출 직전에 output+MHA input2를 manager slice에 bind. 실행/주소/범위 확인 후 append identity 연산 생략 |
| cleanup | 성공·예외 시 activation binding 복구. backing은 마지막 alias보다 오래 유지 |
| trace | `NNTR_KV_GPU_TRACE=1`: device attention 호출, direct V submission, append copy bytes, backing API bytes |

`hostAddressable()`은 주소 종류의 속성이다. coherence/동시 접근 허가는 completion 규칙이 결정한다.
기존 vendor allocator를 변경 전 헤더로 별도 컴파일한 후 새 runtime/client에 연결한
ABI probe도 PASS다: `validation/allocator-abi/result.log`.
이는 MemAllocator vtable 검사다. 내부 CausalLM/Transformer ABI는 변경되므로
observer는 runtime과 맞는 전체 header tree로 빌드한다.

private `cpu_cache_`/`allocateCpu()` 이름은 유지했다.
`usesCoreStorage()`는 CPU/CUDA 공통, `usesCpuStorage()`는 실제 CPU만 뜻한다.

## 4. 실행 profile과 build

공개 모델에서 검증한 구성은 FP16 FC/norm/activation + FP32 embedding/tied head,
host-mapped activation plane + device-only KV다.
일반 device activation pool까지 켠 조합의 성공을 주장하지 않는다.

```bash
export NNTR_ENGINE=cuda
export NNTR_CUDA_KV_CORE=1 NNTR_CUDA_KV_DIRECT=1
export NNTR_CUDA_KV_DEV=1 NNTR_CUDA_HOST_MAPPED=1 NNTR_CUDA_DEV_ACT=0
export NNTR_CUDA_ATTN=1 NNTR_CUDA_ROPE=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_ELTWISE=1
export NNTR_CUDA_ASYNC=0 NNTR_CUDA_DECODE_GRAPH=0
export NNTR_CUDA_GRAPH=0 NNTR_CUDA_M2B=0 NNTR_CUDA_PREFILL_GRAPH=0
```

`DIRECT` 미지정 시 core CUDA에서 직접 쓰기가 기본이다.
`DIRECT=0`은 V append-copy 비교 profile, `CORE=0`은 기존 manager다.
세 profile은 같은 arithmetic flag로 비교했다.
capture 기본 주입을 피하려면 graph 관련 opt-out을 모두 명시한다.

batch>1, 비-FP16 KV, heads 누락, alias/hybrid, INT8/OHWI,
memory-swap direct producer, capture/replay, 미선언 V producer는 지원하지 않는다.
`NNTR_KV_INT8`/`NNTR_KV_OHWI`는 기존 코드가 존재 여부로 해석하므로 unset한다.
geometry/backend/필수 연산 flag/capture/alias/hybrid의 지원 거절 테스트를 추가했다.

```bash
CC=gcc-14 CXX=g++-14 meson setup build-kv-gpu-cuda \
  --buildtype=debugoptimized -Dplatform=none \
  -Denable-app=true -Denable-transformer=true -Denable-test=true \
  -Denable-api=false -Denable-fp16=true -Denable-cuda=true \
  -Dcuda-path=/usr/local/cuda-13.0 -Denable-opencl=false \
  -Denable-npu=false -Denable-htp=false \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Denable-nnstreamer-backbone=false -Dml-api-support=disabled \
  -Denable-capi=disabled -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled -Dopenblas-num-threads=1
ninja -C build-kv-gpu-cuda -j4
meson test -C build-kv-gpu-cuda --no-rebuild --num-processes 1 --print-errorlogs \
  unittest_kv_cache_cuda unittest_kv_cache_cuda_graph \
  unittest_cuda_kernels_layernorm unittest_cuda_kernels_gelu unittest_cuda_kernels_fc_qs4cx
```

CPU regression은 별도 build에서 CUDA/FP16=false다.
model test는 `Applications/CausalLM/nntr_quantize` target도 필요하다.
sanitizer build는 app/transformer/CUDA/FP16=false,
`--buildtype=debug -Db_sanitize=address,undefined -Db_lundef=false`이며
`ASAN_OPTIONS=detect_leaks=1`로 core test를 실행했다.

## 5. 테스트

| 실행 | PASS | SKIP |
|---|---:|---|
| CPU core | 29 | 0 |
| CPU graph / runtime / ring | 16 / 23 / 22 | 0 |
| CPU facade / legacy | 25 / 20 | 0 |
| CPU CausalLM model | 106 | 기존 deprecated QINT4 1건 |
| CUDA allocation/storage | 7 | 0 |
| CUDA 실제 wrapper/FC/MHA graph | 7 | 0 |
| 기존 CUDA kernel | 8 | 0, layernorm2 / gelu2 / FC4 |
| CPU-only core ASan/UBSan/LSan | 29 | 0, 오류/leak 없음 |

CPU 총 **241 PASS + 1 SKIP**, CUDA 실제 장치 테스트 **22 PASS**다.
sanitizer는 별도 실행이며 CPU 합계에 다시 더하지 않는다.
실제 graph에서는 full/chunk prefill/decode, capacity+1, 반복 reset, snapshot,
누락 producer, foreign backing 후 retry, copy/direct K/V 전체 byte 동일성,
binding 복구를 검사했다.
portable fake backend로 producer/consumer completion 실패를 주입했고,
실제 CUDA에서는 비동기 GPU write와 checked wait를 확인했다.

- CPU: `build-kv-gpu-cpu-regression/validation/test-abi-final.log` 및 meson testlog.
- sanitizer: `build-kv-gpu-core-asan/validation/test-abi-final.log`.
- CUDA: `build-kv-gpu-cuda/validation/final-cuda-tests.log`, `admission-tests.log`.
- clang-format14 전체 변경 파일/변경 줄, advanced Doxygen, diff check,
  신규 Python/변환기 ruff E/F/W 및 py_compile PASS.
- 다른 OS/Android/OpenCL/전체 build matrix/CodeQL/원격 CI는 미실행.

## 6. 공개 Qwen3와 copy audit

`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`를 고정했다.

| artifact | SHA-256 |
|---|---|
| HF model.safetensors | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| dense FP16 / FP32 embedding, 1,503,264,768 bytes | `e1dc2fe5d60c690ee7920fc830a58e58c753afdd9f72c73a2e06f20e76ef2ad2` |

converter의 새 `--embedding_fp32`는 기본 동작을 유지한다.
tied head는 embedding weight를 공유하므로 runtime embedding/lmhead 둘 다 FP32로 맞춘다.

```bash
# HF snapshot은 위 revision으로 별도 준비한다. 아래는 다운로드하지 않는다.
python Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py \
  --model_path build-kv-gpu-cuda/models/hf/qwen3-0.6b \
  --output_name build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16/nntr_qwen3_dense_fp16.bin \
  --data_type float16 --embedding_fp32
```

config/tokenizer/generation config도 같은 모델 디렉터리에 둔다.
nntr config는 model_tensor_type=FP16-FP16, fc_layer_dtype=FP16,
embedding_dtype=FP32, lmhead_dtype=FP32, batch_size=1, max_seq_len=256,
init_seq_len=64, skip_prefill=false다.
준비 script `validation/prepare_dense_fp16_direct.py`와
모델의 `gpu-model-manifest.json`에 전체 설정과 변환 hash를 보존했다.

기존 `qwen_cpu_probe.cpp`는 backend 독립 logits observer라 그대로 재사용했다.
baseline은 HEAD 전체 header archive, candidate는 현재 헤더와 링크했다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda --output build-kv-gpu-cuda/validation/qwen_gpu_probe
# baseline에는 --header-tree <HEAD git archive 경로>와
# --header-tree-revision 25230aafe1a230e8dc497921d1149d52f6eabdad도 넘긴다.
/home/jayden/.cache/nntr-gpu-model/bin/python tools/kv_cache/validate_qwen3_cuda.py \
  --hf build-kv-gpu-cuda/models/hf/qwen3-0.6b \
  --model build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16 \
  --baseline-probe build-kv-gpu-cuda-baseline/validation/qwen_gpu_probe \
  --candidate-probe build-kv-gpu-cuda/validation/qwen_gpu_probe \
  --output build-kv-gpu-cuda/validation/qwen-correctness-new --snapshot-resume
```

최종 artifact는 `validation/qwen-correctness-frozen/`다.

- arithmetic/geography × chunk0/8 × legacy/copy/direct = **12/12 PASS**.
- 같은 CUDA arithmetic의 전체 logits·greedy token 동일.
  literal-byte audit가 continuation 포함 **60개 logits 파일 쌍**의 bytes 동일성을 확인했다.
- HF FP32 대비 exact greedy token 일치, logits finite.
  사전 max_abs≤0.15/mean_abs≤0.02 기준 PASS.
  관측 최대 max_abs=0.063340, step별 mean_abs=0.013016.
- chunk0/8은 연산 분할이 달라 수치 gate를 적용했다.
  max_abs=0.099610/mean_abs=0.019078로 통과했다.
- snapshot position=32, 후속 두 turn position=47/62.
  live/resumed, baseline/direct의 token·전체 logits·position 동일.
- snapshot 3,670,144 bytes, 두 profile의 SHA-256은
  `04e9c69ba865af6858541013e71346f408bf706f5a8bac84ae042fe4e93ce72d`로 동일.
- snapshot D2H/H2D staging은 persistence 비용으로 분리한다.

counter 외에 CUDA API/driver launch interposer를 별도 실행했다.
`validation/cuda-call-audit/trace.cpp`, `run.py`, `summary.json`이 원본이다.

- arithmetic chunk8 copy: `scalar_mul_fp16` **308회**, V append **1,835,008 bytes**.
- direct: 해당 identity kernel **0회**, append **0 bytes**.
- attention308/RoPE616 등 나머지 관측 NVRTC dispatch 수 동일.
- 관측 runtime memcpy는 초기 준비 H2D 32,768 bytes × 2뿐이며 KV prefix/mirror/append memcpy 없음.
- 인터포저 실행의 전체 logits SHA-256도 copy/direct 동일.
- strict mirror 거절과 실제 K/V 주소·범위 검사를 함께 확인했다.
  이 trace는 앱의 동적 CUDA API/NVRTC launch 범위이며 vendor library 내부나
  driver migration 전체를 포괄하는 물리 전송 증명은 아니다.

## 7. 1K/4K 성능

정확성 observer와 trace를 끄고 `qwen_cpu_benchmark.cpp` fixed-token workload를 사용했다.
input은 ID264 반복, decode는 32/33/34/35 반복이다. 정답 생성 workload와 구별한다.

공통 조건: max_seq8192, activation plane128, prefill chunk128, decode8,
eager/ASYNC0, threads4, OpenBLAS1, 같은 모델/arithmetic.
각 context에서 **baseline→direct→direct→baseline** 새 process 순서로,
각 warmup1 제외 + 측정3 = profile당 **6 sample**이다.
load/initialize/JIT 첫 실행은 steady-state 집계에서 제외했다.

| context | metric ms | baseline median | direct median | 변화 |
|---|---|---:|---:|---:|
| 1024 | run wall | 12050.64 | 11943.95 | −0.89% |
| 1024 | prefill compute | 8095.00 | 8035.50 | −0.74% |
| 1024 | decode API 8 tokens | 3974.00 | 3894.50 | −2.00% |
| 4096 | run wall | 37128.66 | 36879.58 | −0.67% |
| 4096 | prefill compute | 33202.50 | 33015.50 | −0.56% |
| 4096 | decode API 8 tokens | 3920.50 | 3833.00 | −2.23% |

run-wall nearest-rank p95는 1K baseline/direct=12134.17/11991.37ms,
4K=37380.67/37060.19ms다. sample6개에서 p95는 최댓값이므로 tail 추정은 제한적이다.
모두 **5% 회귀 gate PASS**다. 차이가 작고 clock을 고정하지 않아 속도 향상을 주장하지 않는다.

각 process 전후 GPU timestamp/UUID/temperature/clock/pstate/memory/power를 저장했다.
온도42–54°C, benchmark 중 다른 GPU 검증은 병행하지 않았다.
전후 nvidia-smi는 peak VRAM 계측이 아니다.
runner peak_memory_kb는 기존 CausalLM footprint metric이며 pinned/VRAM 전체가 아니다.

최초 실행은 ignored `validation/run_performance.py`였고 원본 hash를
`validation/performance/summary.json`에 남겼다.
같은 설정을 `tools/kv_cache/benchmark_qwen3_cuda.py` CLI로 정리했다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda --source tools/kv_cache/qwen_cpu_benchmark.cpp \
  --output build-kv-gpu-cuda/validation/qwen_gpu_benchmark
python3 tools/kv_cache/benchmark_qwen3_cuda.py \
  --model build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16 \
  --baseline-probe build-kv-gpu-cuda-baseline/validation/qwen_gpu_benchmark \
  --candidate-probe build-kv-gpu-cuda/validation/qwen_gpu_benchmark \
  --output build-kv-gpu-cuda/validation/performance-new
```

## 8. 시행착오와 다음 작업

실패 로그도 보존했다.

- 초기 CPU model suite는 nntr_quantize 누락으로 7 FAIL/8 SKIP였다.
  target 빌드 후 최종106 PASS/1 기존 SKIP다.
- facade/legacy snapshot tests는 같은 파일명 충돌 때문에 serial로 실행했다.
- 초기 FP16 tied embedding/untied head tiny fixture는 현재 검증 profile과 맞지 않았다.
  공개 모델과 같은 FP32 embedding/tied head fixture로 고정했다.
- dense FP16 quantizer save는 unsupported였다. 별도 quantizer 수정을 하지 않고
  converter의 명시적 embedding dtype 옵션을 추가했다.
- 첫 HF 비교는 prefill1 + decode budget 때문에 8 대9 token이었다.
  validator가 num_to_generate=tokens−1의 별도 runtime config를 만들도록 수정했다.
- numpy 수치 동일성과 byte 동일성을 구분해 마지막 파일 bytes audit도 수행했다.

다음 우선순위는 G6의 submitted/completed 분리·event dependency·stable destination·
replay generation invalidation이다. 현재 per-layer wait 최적화도 여기에 속한다.
host-mapped attention 전환, OpenCL device probe, Android target,
INT8/alias/hybrid/ring/다른 final V writer는 각각 별도 capability gate로 진행한다.
작업 재개 위치는 [최신 인계](gpu-handoff-2026-10-29.md)다.
