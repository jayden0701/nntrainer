# CUDA final V slot 최적화 — PC/WSL (2026-10-30)

사용자 요청에 따라 CUDA 남은 작업을 이어갔다.
OpenCL은 다른 원격 환경의 담당 범위이므로 수정하지 않았다.
기본 branch는 `feat/kv-cache-cuda-replay`, 시작점은 `8be027ee`다.
별도 `fix/cuda-host-head-fp32`의 출력층 최적화는 가져오지 않았다.

이후 CUDA attention 작업은 [선행 읽기·profiling 기록](gpu-cuda-attention-performance.md)을 참고한다.

## 변경

G6 `kvValueSlot`은 입력 FP16 row와 dense FP16 weight[K,N]를 읽어
`base[device_position * N + output]`에 최종 V를 생성한다.
기존에는 출력별 스레드가 K 전체를 순차 FP32 FMA로 누적했다.

- K≥256에서는 64개 단위로 펼친 bulk 루프와 별도 나머지 루프를 사용한다.
  독립 load를 더 일찍 발행할 여지를 주되 FMA 순서는 k=0…K−1로 유지한다.
- K<256은 기존 순차 루프를 유지한다.
  NVRTC compile option으로 두 kernel variant를 구분하며,
  기존 context cache의 name+options key를 사용한다.
- 출력 크기, device position, KV 직접 쓰기/완료 계약은 같다.
  staging·append copy·weight cache를 추가하지 않는다.

실행 block 크기만 바꾸는 실험은 이 PC에서 효과가 작았다.
단일 `#pragma unroll 64`를 전체 K 루프에 적용하면 작은 K에서
나머지 처리 비용이 크게 증가했다. bulk/tail 분리 후에도 K=65 mapped 경우가
느려져 K<256을 기존 루프로 보존했다.
이 실패 측정과 중단한 초기 모델 검증 로그도 ignored artifact에 남겼다.
최종 수치와 구분한다.

## 실제 GPU 검증

새 `CudaKv.valueProjectionPreservesFmaOrderAndReplaySlotGuards`는
CPU의 **순서가 정해진 FP32 std::fma + FP16 반올림**을 기준으로 비교한다.

- N/K: 1/1, 31/63, 129/65, 127/127, 131/129,
  129/256, 129/257, 129/511, 1024/1024.
- 출력 block tail, 입력 bulk/tail 경계, bulk variant와 작은 shape variant의
  같은 process 내 전환.
- slot1 eager 쓰기 후 capture; device position만 slot2로 바꾸어 replay.
- 두 slot의 FP16 bytes 정확히 일치, 앞뒤 guard row 보존.
- 기존 core31/storage8/graph7/replay4에 새 storage test1 추가:
  **core31 + 실제 CUDA storage9/graph7/replay4 = 51 PASS, SKIP 없음**.

CUDA test translation unit은 ENABLE_FP16을 끈 syntax-check도 PASS했다.
CPU 제품 소스·공통 API·OpenCL을 변경하지 않았다.
이 세션에서 기존 전체 CPU/Android/Windows matrix를 재실행한 것은 아니다.
원격 Actions/Android 기기 제약은 [이전 플랫폼 기록](gpu-cuda-replay-performance.md)을 따른다.

## 커널 측정

신규 `tools/kv_cache/cuda_kv_value_benchmark.cpp`는
변경 전 serial source를 고정한 baseline과 실제 production API를 비교한다.
source 내 baseline은 `8be027ee`의 누적 루프다.

RTX5070/sm120, WSL Ubuntu26.04.1/GCC14, CUDA runtime13000,
driver API13030(Windows driver610.88)을 사용했다.
FP16 input/output은 device allocation이다. weight는 device와 host-mapped를 따로 측정했다.

- JIT와 capture는 timed section 밖이다.
- 각 graph는 동일 projection kernel 100개, graph warmup3회.
- ABBA arm마다 event 측정3회, arm별 총6개 batch 표본.
- CUDA event로 graph 실행을 재고 100으로 나눈 us/kernel batch 평균이다.
  표의 median은 batch 평균6개의 중앙값이다.
  JSON p95는 nearest rank이며 표본6개에서 최댓값이다.
- 먼저 모든 출력 row bytes를 baseline과 비교하며 모두 동일해야 측정한다.
- 입력/가중치가 반복되는 warm microbenchmark다.
  여러 layer의 서로 다른 weight를 사용하는 모델 workload와 cache 동작이 다르다.
  아래 값을 모델 전체 속도 개선으로 해석하지 않는다.

| weight | N×K | serial us | production us | 변화 |
|---|---|---:|---:|---:|
| device | 31×63 | 2.482 | 2.490 | +0.35% |
| device | 129×65 | 2.344 | 2.342 | -0.10% |
| device | 128×255 | 6.192 | 6.193 | +0.01% |
| device | 128×256 | 5.788 | 5.039 | -12.93% |
| device | 128×257 | 5.928 | 5.177 | -12.67% |
| device | 128×511 | 10.970 | 9.645 | -12.08% |
| device | 128×1024 | 20.183 | 16.883 | -16.35% |
| device | 1024×1024 | 20.391 | 17.139 | -15.94% |
| device | 2048×4096 | 78.679 | 66.545 | -15.42% |
| mapped | 31×63 | 14.437 | 14.392 | -0.31% |
| mapped | 129×65 | 12.121 | 12.156 | +0.29% |
| mapped | 128×255 | 27.049 | 27.062 | +0.05% |
| mapped | 128×256 | 24.938 | 19.229 | -22.89% |
| mapped | 128×257 | 25.685 | 19.929 | -22.41% |
| mapped | 128×511 | 51.324 | 41.112 | -19.90% |
| mapped | 128×1024 | 98.300 | 75.375 | -23.32% |
| mapped | 1024×1024 | 102.017 | 80.722 | -20.87% |
| mapped | 2048×4096 | 440.177 | 373.937 | -15.05% |

작은 K의 median 차이는 약±0.4% 이내이고,
최종 18개 shape/weight 조합 모두 5% 회귀 예산을 통과했다.
Qwen 크기 N=K=1024에서 device 약16%, mapped 약21% 감소했다.
다른 GPU/driver에서 같은 개선 폭은 미검증이다.
GPU clock을 고정하지 않았으므로 소폭 차이는 측정 변동으로 취급한다.

## 공개 모델 확인

기존과 동일한 공개 Qwen3-0.6B revision/변환 모델을 사용했다.
`qwen-final/manifest.json`은 **PASS**다.

- classic eager / ordered eager / replay fixed44 token과 두 후속 turn,
  초기 position68로64-key 경계를 통과해 recapture.
- ordered/replay 전체 logits bitwise 동일.
- snapshot 파일 bytes 동일, live/resumed logits와 position114/161 일치.
- arithmetic/geography 각8개 실제 greedy token을 frozen HF FP32와 비교:
  토큰 동일, finite와 사전 수치 gate 통과. max_abs=0.0577316284.
- classic 대비 max_abs=0.0390625로 기존 tolerance 안이다.
- 마지막 replay trace captures=3/replays=80, append_copy_bytes=0.
- 기존 `qwen-replay-final`의 ordered/replay와 새 결과를 직접 비교:
  **440쌍 모두 bytes 동일, max_abs=0, token/position 동일**.

이번 비교는 V 커널의 전후 비교다.
[출력층 최적화 당시의440쌍 수치 차이 정정](gpu-cuda-replay-performance.md#정합성-수치-변경과-기록-정정)과
다른 실험이다. 별도 출력층 branch의 변경을 섞지 않았다.
모델 전체 latency ABBA를 재측정하지 않았으며 모델 전체 속도 개선은 주장하지 않는다.


## 재현 및 artifact

artifact prefix: `build-kv-gpu-cuda-g6/validation/value-slot/`.
`manifest.json`에는 source·runtime hash, 테스트 수, microbenchmark gate를 기록했다.
`final-tests-full.log`, `guarded-{device,mapped}.json`,
`qwen-final/manifest.json` 및 기존 실행과의 byte audit가 최종 결과다.
`final-*`/`bulk-*`/`unroll-*`/`prefetch-*`는 후보 탐색 기록이며
`guarded-*`만 최종 성능 수치로 사용한다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda-g6 \
  --source tools/kv_cache/cuda_kv_value_benchmark.cpp \
  --output build-kv-gpu-cuda-g6/validation/value-slot/benchmark
NNTR_ENGINE=cuda NNTR_CUDA_ASYNC=1 NNTR_CUDA_HOST_MAPPED=1 \
  build-kv-gpu-cuda-g6/validation/value-slot/benchmark NEW-device.json
NNTR_ENGINE=cuda NNTR_CUDA_ASYNC=1 NNTR_CUDA_HOST_MAPPED=1 \
  build-kv-gpu-cuda-g6/validation/value-slot/benchmark NEW-mapped.json --mapped
meson test -C build-kv-gpu-cuda-g6 --no-rebuild --num-processes 1 --print-errorlogs \
  unittest_kv_cache_core unittest_kv_cache_cuda \
  unittest_kv_cache_cuda_graph unittest_kv_cache_cuda_replay
```

기존 JSON은 덮어쓰지 않는다. binary/model/log는 ignored 상태로 유지한다.
다음 CUDA 작업은 attention/split-KV kernel의 GPU 시간 분석,
실제 모델의 CPU↔CUDA 전환 또는 prefill capture 중 독립 범위를 정해 진행한다.
다중 stream·Android GPU 검증은 여전히 후속이다.
