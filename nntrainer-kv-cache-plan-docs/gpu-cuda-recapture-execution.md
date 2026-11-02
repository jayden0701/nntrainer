# CUDA decode graph 재캡처 조건 개선 (2026-11-02)

`feat/kv-cache-cuda-replay`의 `c827456e`에서 이어서 구현했다.
OpenCL과 별도 FP32 host head branch는 수정하지 않았다.

## 문제와 변경

이 조건은 KV manager의 CUDA 연결 이전 upstream에도 있던 CUDA Graph 정책이다.
split-KV head fusion은 실제로 선택되는 HPW(heads per warp)에 따라
kernel 및 grid가 바뀐다. 기존 코드는 HPW가 그대로여도 현재 chunk 수 양옆의
경계를 기록해 기본64 token마다 graph를 만료시켰다.

새 코드는 각 후보 HPW가 실제로 사용 가능해지는 경계만 기록한다.

```text
groups = query_heads / candidate_hpw
needed_chunks = ceil(minimum_blocks / groups)
key-count threshold = (needed_chunks - 1) * chunk_size
```

후보를 원래 우선순위로 검사하고 첫 eligible HPW를 선택하는 로직은 유지한다.
고정 capacity 안에서 도달 불가능한 후보는 경계에 포함하지 않는다.
window-clipped 경로의 고정 stride, dense→split, 짧은-grid staging, window 전환의
기존 별도 경계는 유지한다.

이 완화는 **live device position + 고정 split scratch stride**가 있는 경우에만
적용한다. device position이 없으면 scalar key count/query position이 graph에
고정되므로 `kv_regime_exact`로 같은 step에서만 유효하도록 보수적으로 제한한다.

GPU kernel 소스·grid 계산·연산 순서는 변경하지 않았다.
reset·snapshot 복원·seek·재할당·owner/network 변경에 따른 기존 무효화와
실패/overflow 시 fail-closed 규칙도 유지한다.
일반 CPU/공통 API/OpenCL 제품 코드는 변경하지 않았다.

예를 들어 기본 Qwen shape(HQ16/HKV8/d128, chunk64, minimum blocks64)의
선택 구간은 dense≤64, staged HPW1=65–256, plain HPW1=257–448,
HPW2≥449다. 같은 선택 구간에서는 단순한 chunk 증가로 재캡처하지 않는다.

## 단위 및 실제 graph 검증

fusion on/off 각각 **core31 + CUDA storage10/graph7/replay7 =55 PASS**,
SKIP 없음이다.

- 기존 wrapper 경계 테스트를641 token까지 확장했다.
  head_dim128/HQ4/HKV2, capacity704의 eager와 매 step 전체 logits 및
  마지막 전체 K/V backing bytes가 같다.
  capture는 초기 decode,65-key dense→split,257-key staging→plain의 **3회**다.
  그 사이 chunk 경계에서는 늘지 않는다. append_copy_bytes=0, position641.
- 새 launch-signature oracle은 실제 캡처한 두 kernel의 CUfunction identity,
  grid, block, shared-memory byte 수를 읽어 validity table과 비교한다.
  선택 공식을 테스트에 복제하지 않는다.
- HQ/HKV/d:16/8/128,12/2/256,8/1/512,8/8/128,4/2/64,
  그리고 window129인16/8/128을 확인했다.
  GQA2/6/8, MHA, legacy fallback, clipped window를 포함한다.
- 각 shape에20개 key-count point(65…8192)를 사용한다.
  20×20×6=2400쌍을 profile마다 검사하며, 같은 관측 launch signature를
  유지하는 구간만 유효하고 실제 선택이 달라지는 경계에서는 무효다.
- fusion off에서도 동일 oracle과 wrapper를 실행했다.
- device position을 사용하지 않는 별도 process profile에서는
  같은65-key step만 유효하고64/66에서는 만료됨을 확인했다.
- CUDA storage test source의 ENABLE_FP16 off syntax-check도 PASS다.

이 세션에서 CPU/Android/Windows 전체 build matrix를 재실행하지는 않았다.
이전 Actions422/Android 실기기 부재 제약은 그대로 별도 기록을 따른다.

## 공개 Qwen의 실제 capture/replay 수

기존 pinned Qwen3-0.6B dense FP16 모델과 같은 observer를 사용했다.
출력층 최적화를 포함하지 않은 기존 runtime dependency closure를 먼저 복사해
hash와 ldd 결과를 보존했다.

context1024 / decode160 / max capacity8192 / chunk·activation plane128 /
단일 stream / mapped activation / device KV / batch1이다.
두 arm 모두 같은 GRAPH/M2B/DECODE_GRAPH/KV_CORE/KV_GRAPH/DIRECT/ASYNC=1이다.
각각 **cold process1회, warmup0, repetition1**로 실행했다.

| 항목 | 변경 전 | 변경 후 |
|---|---:|---:|
| graph captures | 3 | 1 |
| graph replays | 154 | 158 |
| 최종 committed position | 1184 | 1184 |
| KV append copy bytes | 0 | 0 |

동일한 forced token160개를 사용했다.
이 실행은 재캡처 횟수 확인 목적이다. 반복 latency ABBA가 아니므로
모델 전체 속도 개선 수치로 사용하지 않는다.
정합성은 별도 전체 logits/HF/greedy/snapshot gate로 검증한다.

## 공개 모델 정합성

기존 pinned Qwen3-0.6B 모델로 `qwen-final/manifest.json`은 **PASS**다.

- classic/ordered/replay fixed44 token,64-key 경계와 두 continuation turn.
- ordered/replay logits 및 snapshot bytes 동일.
- live/resumed logits와 position114/161 동일.
- arithmetic/geography 각8개 실제 greedy token이 frozen HF FP32와 일치.
  finite·수치 gate 통과, HF max_abs=0.0577316284.
- 직전 `attention/qwen-final`과 직접 비교한 logits **440쌍 모두 byte 동일**,
  token/position도 동일하다.

해당 multi-turn replay의 마지막 trace도 captures3→2, replays80→82로 바뀌었고
append_copy_bytes=0을 유지했다.

snapshot I/O와 reset/load publication 의미는 유지했다.
모델 자산과 weight/config/hash는 이전 공개 모델 실행 기록과 같다.
출력층 최적화 branch는 가져오지 않았다.


## artifact와 재현

prefix는 `build-kv-gpu-cuda-g6/validation/recapture/`다.

- `baseline-runtime/manifest.json`: 이전 c827456e runtime/observer hash.
- `counter-manifest.json`, `counter-{baseline,candidate}.json`,
  `counter-comparison.json`: 공개 모델 capture/replay 비교.
- `tests-full.log`, `tests-no-fuse-full.log`:55 PASS씩.
- `qwen-final/manifest.json`, `before-after-byte-audit.json`: 모델 정합성.
- `non-fp16-syntax.log`, `doxygen.log`: 정적/compile 검사.

```bash
meson test -C build-kv-gpu-cuda-g6 --no-rebuild --num-processes 1 --print-errorlogs \
  unittest_kv_cache_core unittest_kv_cache_cuda \
  unittest_kv_cache_cuda_graph unittest_kv_cache_cuda_replay
NNTR_CUDA_ATTN_FUSE=0 meson test -C build-kv-gpu-cuda-g6 --no-rebuild \
  --num-processes 1 --print-errorlogs unittest_kv_cache_core \
  unittest_kv_cache_cuda unittest_kv_cache_cuda_graph unittest_kv_cache_cuda_replay
```

모델 정합성 runner는 이전과 같은 `validate_qwen3_cuda_replay.py`다.
baseline을 Ninja로 다시 빌드하지 않고 보존한다.
모델/binary/log는 ignored local artifact이며 source와 문서만 push한다.

남은 CUDA 항목은 prefill capture, 더 긴 문맥/다른 모델의 실행시간 측정,
다중 stream 및 실제 장치/backend 전환이다.
