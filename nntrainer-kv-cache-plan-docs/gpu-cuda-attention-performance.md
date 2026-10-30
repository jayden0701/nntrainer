# CUDA split-KV decode 선행 읽기 — PC/WSL (2026-10-30)

`feat/kv-cache-cuda-replay`의 `c0493936`에서 이어서 attention을 분석했다.
OpenCL과 `fix/cuda-host-head-fp32`는 수정하지 않았다.

## 측정으로 정한 범위

기존 split-KV는 partial과 reduce 두 kernel이다.
기본 HPW=1 partial은 warp마다 key를 하나씩 읽고 online softmax를 갱신한다.
기존 fused-head partial은 이미 KPI=4 선행 읽기를 사용하고 있었다.
HPW=1에도 같은 key 순서를 유지한 채 네 key를 stage하는 variant를 추가했다.

**실제로 처리할 chunk가4개 이하이고 HPW=1일 때만** `_p4` variant를 선택한다.
기본 chunk64에서는65–256-token 문맥이다. 64 이하에서는 기존 dense 경로를 쓴다.
window-clipped layer는 기존의 고정 clipped stride를 기준으로 판단한다.
기존 fused-head/긴 문맥/legacy fallback 선택은 유지한다.

unclipped layer는 네 번째 chunk 경계를 `kv_regime_gt`로 기록해
kernel 선택이 바뀔 때 G6가 eager prewarm 후 재캡처하도록 한다.
scratch layout·FP32 누적 순서·KV 직접 읽기/쓰기 계약은 바꾸지 않았다.

후보를 넓게 적용하면 fusion off·긴 문맥에서 register 부담으로 약20% 회귀하는
경우가 있었다. 그래서 긴 문맥은 기존 KPI=1로 보존했다.
reduce의 warp max 및32-thread block 실험도 일관된 개선이 없어 제외했다.
해당 후보 로그는 최종 결과와 분리해 남겼다.

## 실제 실행 kernel을 분리해 측정

`tools/kv_cache/cuda_kv_attention_benchmark.cpp`가 실제 production attention을
capture하고 node의 function/arguments/grid/block을 읽는다.
partial-only/reduce-only graph를 각각100개 node로 구성해 GPU 시간을 분리한다.
원래 source graph는 복제 graph보다 오래 유지해 node 소유 arguments 수명을 지킨다.
NVRTC kernel node는 Driver API로 읽고 추가한다.
API 계약은 [NVIDIA Graph Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)를 따른다.

- RTX5070/sm120, Windows driver610.88, CUDA13.0, WSL Ubuntu26.04.1/GCC14.
- Q/K/V/O는 device allocation, max capacity8192, chunk64, 단일 stream.
- shapes HQ/HKV/d:16/8/128,8/1/256,8/1/512.
- lengths65/129/256/257/512/1024/4096/4097/8192.
- fusion on/off를 별도 process로 실행한 총54조합.
- 이전 runtime dependency closure를 복사·hash하여 보존했다.
  동일 observer를 baseline/candidate/candidate/baseline(ABBA) 순서로 실행했다.
- process당9개 event batch, arm별2개 process를 합쳐18개 batch 평균의
  median/p95를 계산한다. p95는 nearest rank이며 이 표본 수에서는 최댓값이다.
- whole sample은100번 attention(partial+reduce), 분리 sample은
  단일 kernel100번이다. 표는 batch 시간을100으로 나눈 us/attention이다.
- 각각 graph warmup3회, JIT/capture/초기화는 timed section 밖이다.
  각 process 전후 GPU clock/온도/pstate/power를 별도 기록했다.
- 고정 operand를 반복하는 microbenchmark다. 같은 trial 내 batch는 상관되며,
  실제 모델 전체 latency나 다른 GPU에서의 개선 폭을 나타내지 않는다.

기본 fusion on의 짧은 문맥 결과:

| HQ/HKV | d | KV length | 이전 us | 이후 us | 변화 |
|---|---:|---:|---:|---:|---:|
| 16/8 | 128 | 65 | 7.976 | 6.444 | -19.20% |
| 16/8 | 128 | 129 | 7.990 | 6.500 | -18.65% |
| 16/8 | 128 | 256 | 8.233 | 6.941 | -15.69% |
| 8/1 | 256 | 65 | 8.492 | 7.158 | -15.71% |
| 8/1 | 256 | 129 | 8.480 | 7.199 | -15.11% |
| 8/1 | 256 | 256 | 8.508 | 7.192 | -15.48% |
| 8/1 | 512 | 65 | 9.499 | 8.262 | -13.03% |
| 8/1 | 512 | 129 | 9.571 | 8.270 | -13.59% |
| 8/1 | 512 | 256 | 9.560 | 8.278 | -13.41% |

긴 문맥은 기존 variant를 선택한다. 최종54조합 모두 whole median 5% 회귀
예산을 통과했고 전체 FP16 출력 bytes도 이전과 같았다.
관측 전체 변화 범위는 약−19.24%…+4.11%다.
node 이름·grid·register 수·local-memory byte 수와 partial/reduce 개별 시간,
전체54조합·p95는 `guarded-paired/` JSON에 있다.
GPU clock을 고정하지 않아 소폭 차이는 변동으로 취급한다.

## 기능 검증

core31 + CUDA storage9/기존 graph7/replay6, **총53 PASS, SKIP 없음**이다.

새 직접 attention 테스트:

- d128/256/512, GQA 및 일반 multi-head, full/window129, softcap.
- moving Q 입력과 KV length65/66/79/127/129/256/257/513/1089.
- 독립 FP64 causal/window/softcap softmax reference와 비교:
  절대0.0003+상대0.002 허용 범위, finite 필수.
- eager/replay 전체 FP16 bytes 동일, 출력 앞뒤 guard 보존.
- GQA fusion bracket이 없는 shape에서256까지 regime 유효,
  257에서 만료됨을 독립 검증.

새 실제 wrapper 테스트:

- head_dim128 모델의2-token prefill 후265까지 단일 token decode.
- staging256→257 경계를 통과해 capture 수 증가·실제 replay 확인.
- eager와 매 step 전체 logits, 마지막 전체 K/V backing bytes 동일.
- position265, append_copy_bytes=0.

CUDA 외 제품 코드는 변경하지 않았다. 이 세션에서 CPU/Android/Windows 전체
build matrix는 재실행하지 않았다. 기존 Actions422/실기기 제약은 이전 인계를 따른다.

## 공개 모델

기존 pinned Qwen3-0.6B dense FP16 모델로 `qwen-final/manifest.json` **PASS**.

- classic/ordered/replay fixed44 token,64-key 경계, 두 continuation turn.
- ordered/replay logits 및 snapshot bytes 동일.
- live/resumed 전체 logits·position114/161 일치.
- arithmetic/geography 각8개 실제 greedy token이 frozen HF FP32와 일치,
  finite·수치 gate 통과. HF 대비 max_abs=0.0577316284.
- 직전 V slot 단계 `value-slot/qwen-final`과 직접 비교한
  **440쌍 logits 모두 byte 동일**, token/position도 동일.
- 성공 replay trace append_copy_bytes=0, captures=3/replays=80.

출력층 최적화 branch의 결과는 섞지 않았다.
이 세션에서 모델 전체 latency ABBA를 다시 측정하지는 않았다.


## 재현 및 artifact

artifact prefix: `build-kv-gpu-cuda-g6/validation/attention/`.
`guarded-paired/{manifest,summary}.json`이 최종 성능 결과다.
`baseline-runtime/manifest.json`은 이전 runtime hash,
`manifest.json`은 최종 source hash와 검증 범위를 기록한다.
wrapper 테스트 추가는 성능 측정 후였으며 production/observer hash는 그대로임을 확인했다.
`final-tests-full.log`, `qwen-final/manifest.json` 및 byte audit를 함께 보존한다.
`kpi4*`/`warpmax*`/`rb32*`/`paired/`는 제외한 후보의 실험 기록이다.
초기 test fixture/observer 실패 로그도 별도로 보존했다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda-g6 \
  --source tools/kv_cache/cuda_kv_attention_benchmark.cpp \
  --output build-kv-gpu-cuda-g6/validation/attention/benchmark
NNTR_ENGINE=cuda NNTR_CUDA_FLASH_DECODE=64 NNTR_CUDA_DECODE_GRAPH=1 \
  NNTR_CUDA_GRAPH=1 NNTR_CUDA_M2B=1 NNTR_CUDA_ASYNC=1 NNTR_CUDA_ATTN_FUSE=1 \
  build-kv-gpu-cuda-g6/validation/attention/benchmark NEW.json
```

fusion off는 새 process에서 `NNTR_CUDA_ATTN_FUSE=0`으로 실행한다.
기존 output을 덮어쓰지 않는다. 모델/binary/log는 ignored artifact다.

후속 CUDA 항목은 실제 kernel 선택이 같은 동안 불필요한 chunk별 재캡처를 줄이는
분석, prefill capture, 더 긴 문맥/모델에서의 profiling이다.
이번에는 기존 재캡처 정책을 넓게 바꾸지 않았다.
