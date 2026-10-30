# G6 CUDA decode capture/replay — PC/WSL (2026-10-29)

> 브랜치 분리 후 현재 적용 범위는 [출력층 최적화 분리 기록](gpu-branch-separation.md)을 따른다.
> KV/G6 기본 branch에서는 출력층 최적화를 제외하고, `fix/cuda-host-head-fp32`에만 추가한다.

[CUDA eager A+B](gpu-cuda-execution.md)에 기존 CUDA Graph 실행기를 연결했다.
작업 브랜치는 `feat/kv-cache-cuda-replay`, 지정 remote는
`jayden=https://github.com/jayden0701/nntrainer`다.
기존 origin/Windows 작업/untracked CLBlast는 보존했다.

## 지원 profile

**Qwen3 / dense native FP16 KV·V weight / batch1 / linear / resident graph /
단일 CUDA stream / decode capture/replay**다.
prefill·multi-token continuation은 eager로 실행하고 다음 decode에서 재캡처한다.
prefill capture, ring/alias/hybrid/INT8, 다중 stream, 비동기 session publication,
다른 모델의 미선언 producer는 지원하지 않는다.

G5의 공개 모델·RTX5070·CUDA13.0·GCC14·mapped activation profile을 사용한다.
새 opt-in은 `NNTR_CUDA_KV_GRAPH=1`이며 기존 eager 기본 동작은 유지한다.

```bash
export NNTR_ENGINE=cuda
export NNTR_CUDA_KV_CORE=1 NNTR_CUDA_KV_GRAPH=1 NNTR_CUDA_KV_DIRECT=1
export NNTR_CUDA_KV_DEV=1 NNTR_CUDA_HOST_MAPPED=1 NNTR_CUDA_DEV_ACT=0
export NNTR_CUDA_ATTN=1 NNTR_CUDA_ROPE=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_ELTWISE=1
export NNTR_CUDA_ASYNC=1 NNTR_CUDA_GRAPH=1 NNTR_CUDA_M2B=1 NNTR_CUDA_DECODE_GRAPH=1
export NNTR_CUDA_PREFILL_GRAPH=0
# 진단 시:
export NNTR_KV_GPU_TRACE=1 NNTR_CUDA_GRAPH_DBG=1
```

KV_GRAPH=1은 CORE=1과 DIRECT=1이 필요하다.
GRAPH/M2B 불일치, prefill capture, DEV_ACT=1은 allocation에서 거절한다.
비교용 ordered eager는 KV_GRAPH=1을 유지하고 GRAPH/M2B/DECODE_GRAPH만 0으로 둔다.

## 연결과 완료 규칙

- producer 상태는 Missing/Ordered/Complete다. CUDA ordered profile은 producer
  event record와 stream wait dependency를 기록하며 capture 중 host wait를 하지 않는다.
- publication은 동기식이다. host tail 실행 전 stream completion, commit 전 checked event
  completion을 확인하며 오류 시 position이 전진하지 않는다.
- `KvReplayBinding`은 immutable backing/context/geometry의 session owner,
  replay generation, final V producer identity를 보존한다.
  일반 step에서는 generation을 유지하고 reset/seek/load/reallocate/abort/destruction에서
  graph를 폐기한다. graph의 retained owner가 KV allocation 수명을 유지한다.
- replay마다 host wrapper가 새 lease를 예약하고 binding/network identity를 검증한다.
  실제 graph launch 성공 후 submission을 기록하고 checked completion 뒤 commit한다.
- K RoPE/attention은 기존 device-position 경로를 사용한다.
  V는 `kv_value_slot`에서 dense FP16 dot을 FP32 FMA로 계산해
  `base[d_pos[0] * width]`에 직접 쓴다. append copy나 별도 V staging은 없다.
  bias/fused activation/다른 V weight dtype은 거절한다.
- classic cuBLAS V reduction과 slot reduction은 연산 순서가 다를 수 있다.
  classic 대비 max_abs≤0.15/mean_abs≤0.02, ordered eager/replay끼리는
  같은 kernel이므로 전체 logits/KV bytes 동일성을 요구한다.
- Qwen3는 feed `embedding0`와 host tail `output_of_causallm`을 선언한다.
  현재 FP32 tied head는 capture에서 제외하고 GPU 완료 후 실행한다.
  recording 때 아직 실행되지 않은 GPU 결과를 host math가 읽지 않게 한다.
- C++ 예외 시 RAII guard가 capture를 종료·폐기한 후 lease를 abort한다.
  feed/tail 실행 모드도 성공·예외 모두 복원한다.

event capture 규칙은 [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)를 따랐다.
같은 capture 안에서 기록한 event의 wait가 graph dependency가 되는 규칙을 사용한다.
외부 consumer stream을 지원하는 범용 event API는 이번 범위에 포함하지 않는다.

## 경계 문제와 검증

64-key 경계 초기 테스트에서 재캡처 이후 logits 차이를 발견했다.
tiny model이 weight-load prewarm을 거치지 않아 split-KV fixed stride가 준비되지 않았으므로,
KV allocation 시 workspace를 준비하고 regime 변경 후 eager prewarm을 거쳐 재캡처하도록 했다.
72-token position까지 매 step 전체 logits와 전체 K/V bytes가 eager와 일치한다.

처음 추가한 파생 test adapter는 virtual Transformer base를 명시적으로 초기화하도록 수정했다.
초기 fixture 및 경계 실패 로그도 보존했다.

새 build `build-kv-gpu-cuda-g6`를 사용해 기존 G5/HEAD baseline artifact를 보존했다.

| 검증 | 결과 |
|---|---|
| core portable | 31 PASS |
| CUDA storage/event/capture completion | 9 PASS |
| 기존 실제 CUDA wrapper/graph | 7 PASS |
| 신규 decode replay | 기본 branch 6 PASS / 분리 당시 출력층 branch 5 PASS |
| non-FP16 CPU 회귀 | 243 PASS + 기존 QINT4 1 SKIP |
| CPU-only ASan/UBSan/LSan core | 31 PASS |

replay test는 threshold, capacity 초과, snapshot 복원, reset, backing 재할당,
두 live model의 교차 실행·파괴, capture 중 foreign backing 예외·retry,
필수 profile 거절을 포함한다.
Ordered 상태에서도 실제 completion 실패 시 publication을 금지하는 portable 실패 주입을 포함한다.

최초 공개 Qwen artifact `validation/qwen-replay/manifest.json`:

- classic eager / ordered eager / replay의 fixed-token initial44 + 두 후속 turn.
- 초기 position68로 64-key 경계를 통과하고 실제 재캡처 확인.
- ordered/replay의 각 turn 전체 logits bitwise 동일.
- live/resumed logits·position 동일, 두 snapshot 파일 bytes 동일.
- classic 대비 수치 gate 통과.
- 성공 trace의 append_copy_bytes=0 및 capture/replay counter 증가.

방어 조건 추가 후 최종 `validation/qwen-replay-final/manifest.json`도 **PASS**다.
fixed-token 비교와 별도로 arithmetic/geography의 실제 greedy를 ordered/replay 각각 실행했다.

- HF FP32와 greedy token 일치, 전체 logits finite.
  max_abs=0.057732, step별 mean_abs=0.008929로 사전 기준 통과.
- ordered/replay의 실제 greedy 전체 logits도 bitwise 동일.
- 두 후속 turn position=114/161, live/resumed 및 ordered/replay 모두 일치.
- 마지막 성공 trace는 captures=3/replays=80, append_copy_bytes=0이다.
- `validation/cuda-call-audit/`의 인터포저로 실제 cudaGraphLaunch **7회**와
  반환값 전부 success를 확인했다.
  관측 `kv_value_slot` dispatch=56, append identity kernel=0이며,
  인터포저를 붙인 출력도 원래 greedy token/logits와 byte 동일하다.
  이 kernel dispatch 수는 capture 때의 host 호출을 포함하며 replay 내부 실행 횟수 합계는 아니다.

최초 fixed-token 결과와 실제 greedy 검증 결과를 구분해 기록했다.

```bash
meson test -C build-kv-gpu-cuda-g6 --no-rebuild --num-processes 1 --print-errorlogs \
  unittest_kv_cache_core unittest_kv_cache_cuda \
  unittest_kv_cache_cuda_graph unittest_kv_cache_cuda_replay
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda-g6 --output build-kv-gpu-cuda-g6/validation/qwen_gpu_probe
/home/jayden/.cache/nntr-gpu-model/bin/python tools/kv_cache/validate_qwen3_cuda_replay.py \
  --model build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16 \
  --probe build-kv-gpu-cuda-g6/validation/qwen_gpu_probe \
  --requests build-kv-gpu-cuda/validation/qwen-correctness-frozen \
  --output build-kv-gpu-cuda-g6/validation/qwen-replay-new
```

observer의 선택형 forced_ids는 원래 logits를 기록한 후 다음 token만 고정한다.
옵션이 없는 기존 CPU/G5 greedy 동작은 유지한다.

## 제한과 후속

prefill capture/다중 stream/Android/OpenCL은 후속이다.
명시적인 host tail이 있으며 모델 전체를 GPU graph로 실행하는 profile은 아니다.
G5 1K/4K 성능을 G6 수치로 사용하지 않는다.
G6 1K/4K latency와 기존 FP32 host head 병목 수정 결과는
[성능 비교 기록](gpu-cuda-replay-performance.md)에 있다.
출력층 수정 후 ordered/replay·HF·snapshot 검증도 다시 PASS했다.
일반 CPU/legacy CUDA 경로는 유지했다.
이후 [V slot 커널 최적화](gpu-cuda-value-slot-performance.md) 및
[attention 선행 읽기](gpu-cuda-attention-performance.md)와 전후 byte 검증을 완료했다.
Android NDK 부분 compile/link는 PASS했으나 실기기 실행은 없다.
GitHub dispatch는422로 거절되었으며 Windows/전체 원격 CI/CodeQL은 미실행이다.
모델·빌드·로그는 ignored local artifact이며 source/재현 문서만 커밋한다.
