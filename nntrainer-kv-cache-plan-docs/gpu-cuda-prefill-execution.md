# CUDA KV manager의 초기 prefill capture 연결 (2026-11-02)

작업 branch는 `feat/kv-cache-cuda-replay`이며 시작점은 `d9c0347b`다.
기존 CUDA executor의 **from=0, multi-token prefill one-shot capture**를
core KV의 ordered step과 연결했다. OpenCL 구현과 별도 lm head branch는 수정하지 않았다.

## 동작과 지원 조건

- `NNTR_CUDA_PREFILL_GRAPH=1`, `NNTR_CUDA_KV_CORE=1`,
  `NNTR_CUDA_KV_GRAPH=1`, `NNTR_CUDA_ASYNC=1` 조합을 허용한다.
- 나머지 지원 범위는 기존 G6 profile과 같다:
  batch1/session1/linear, native FP16, Qwen3 direct V FC, mapped activations/device KV,
  `ATTN/ROPE/QKNORM/ELTWISE=1`, `DEV_ACT=0`, direct V 켬.
- decode graph를 함께 사용할 때 `GRAPH/M2B/DECODE_GRAPH=1`로 맞춘다.
  prefill만 capture하려면 세 flag를0으로 맞춘 ordered profile을 사용한다.
- KV_GRAPH=0인 synchronous/unordered core profile에서는 prefill capture를 계속 거절한다.
- unset인 prefill flag의 기존 integrated/discrete GPU 기본 정책은 바꾸지 않았다.
  검증한 RTX5070에서는 명시적으로1을 설정했다.

기존 prefill capture는 graph를 저장해서 모든 chunk에 반복 적용하는 구조가 아니다.
`from==0 && span>1`인 첫 step을 기록하고 한 번 launch한 뒤 graph/exec를 파괴한다.
뒤의 `from>0` multi-token chunk는 eager로 실행하고 decode는 기존 G6 replay를 사용한다.
이 작업은 후속 chunk capture, prefill graph 재사용, speculative decoding을 추가하지 않는다.

## 변경한 코드와 이유

1. `KVCacheManager`의 admission에서 ordered core profile의 prefill flag를 허용했다.
2. CUDA MHA finalize에서 기존 host/device RoPE LUT를 사전 준비한다.
   첫 prefill을 eager로 한번 실행해 덮어쓴 뒤 capture하는 방식은 사용하지 않는다.
3. GEMM attention이 선택될 수 있는 경우 score scratch를 capture 전에 준비한다.
   selector의 기존 환경변수 해석을 공용 helper로 옮겼다.
   짧은 discrete-GPU block-Q prefill에는 불필요한 quadratic scratch를 할당하지 않는다.
4. 기존 declared host tail(output projection)은 GPU capture 밖에서 stream 완료 후 실행한다.
   이 경계는 기존 ordered core executor의 동작을 그대로 이용한다.
5. RMSNorm/FC/GeGLU/SwiGLU가 core capture 중 host fallback으로 빠지면 오류를 보고한다.
   capture는 기록만 하므로 host가 GPU 중간 결과를 읽으면 안 된다.
   FP16 Qwen supported profile에 대한 보호이며 임의의 모델/모든 host op에 대한 지원 선언은 아니다.
6. prefill의 graph/exec는 RAII로 정리하고 launch/sync 결과를 검사한다.
   core capture 실패 시 같은 step의 forward를 두 번 실행하지 않고 lease를 abort한다.
   종료 guard가 capture 상태를 정리하며 position은 성공적으로 완료된 step만 갱신한다.
7. `capturedSubmission()`으로 producer/readiness를 확인하고 기존 `graph_captures`를 집계한다.
   이 counter는 이제 prefill one-shot과 decode capture를 합산한다.
   저장된 decode graph의 재사용 횟수는 기존 `graph_replays`다.

## 검증

검증 기록은 WSL `/home/jayden/nntrainer/build-kv-gpu-cuda-g6/validation/prefill/`에 보존한다.
CUDA13.0/RTX5070/sm120, 단일 GPU stream, Android 장치 없음이다.

- 단위 회귀: 기존 core31/storage10/eager-wrapper7/decode-replay7에 더해
  prefill4 test를 block-Q/GEMM × decode off/on의4개 독립 process에서 실행한다.
- cold 첫 capture와 이어지는 nonzero multi-token step/decode의 전체 logits 및 전체 K/V bytes를
  capture/replay를 끈 eager reference와 비교한다.
- 128-token 초기 prefill 및 decode도 전체 logits/K/V bytes를 비교한다.
- reset 후 초기 capture 반복, snapshot 복원 뒤 continuation 출력 일치를 확인한다.
- gamma를 FP32로 바꿔 GPU RMSNorm을 거절하게 만든 후, capture 종료/position0/원상복구 후 재시도를 확인한다.
- 실제 pinned Qwen3-0.6B dense FP16 모델의 fixed-token/HF greedy/snapshot/2-turn 검증은
  `validate_qwen3_cuda_replay.py --prefill-capture`로 실행한다.
  classic reference는 prefill0이며 ordered/replay는 prefill1이다.
  첫 trace의 captures=1도 검사하므로 flag만 켜고 eager로 우회한 경우를 통과시키지 않는다.

최종 결과:

| 항목 | 결과 / artifact |
|---|---|
| CUDA 단위 회귀 | **71 PASS, SKIP0**. 기존55 + prefill4×4. `verified-tests.log` |
| CUDA-off CPU 회귀 | **205 PASS + 기존1 SKIP**. core/runtime/facade/legacy/model의5 target, `cpu-tests.log` |
| 공개 Qwen correctness | **PASS**. `qwen-capture/manifest.json` |
| prefill off→on 파일 비교 | **504쌍 전체 logits 파일 bytes 동일**, ordered/replay snapshot도 동일. `prefill-toggle-comparison.json` |
| 실제 capture 여부 | ordered/replay 첫 GPU trace의 captures=1. decode threshold 재캡처 및 replay도 확인 |
| 코드 검사 | clang-format14, advanced Doxygen, ruff, py_compile, `git diff --check` PASS |

off reference는 `validation/recapture/qwen-final`의 보존된 실행이다.
같은 pinned weights, request/config, token sequence, classic/ordered/replay 및 greedy 조합을 비교했다.
504쌍은 initial뿐 아니라 live/restored continuation도 포함하며,
이전 문서의440쌍 집계 범위와 다르다.
새 실행의 linked library hash도 실행 후 다시 확인해 manifest와 모두 일치했다.

초기 cold 테스트는 capture 내부의 첫 RoPE LUT 할당으로 실패했으며
`initial-tests.log`에 보존했다. LUT 사전 준비 후 cold capture와 재시도 검증이 통과했다.
GEMM 강제 실행은 scratch의 cold 준비 및 128-row 확장을 함께 검증했다.
이 결과는 integrated GPU 실행이나 모든 CUDA 모델의 capture 지원을 뜻하지 않는다.

## 재현

기존 G6 build와 모델이 준비된 환경에서:

```sh
ninja -C build-kv-gpu-cuda-g6 -j4 \
  test/unittest/unittest_kv_cache_core \
  test/unittest/unittest_kv_cache_cuda \
  Applications/CausalLM/unittest_kv_cache_cuda_graph
meson test -C build-kv-gpu-cuda-g6 --no-rebuild --num-processes 1 \
  unittest_kv_cache_core unittest_kv_cache_cuda \
  unittest_kv_cache_cuda_graph unittest_kv_cache_cuda_replay \
  unittest_kv_cache_cuda_prefill unittest_kv_cache_cuda_prefill_decode \
  unittest_kv_cache_cuda_prefill_gemm unittest_kv_cache_cuda_prefill_decode_gemm
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda-g6 \
  --output build-kv-gpu-cuda-g6/validation/prefill/qwen_gpu_probe
/home/jayden/.cache/nntr-gpu-model/bin/python \
  tools/kv_cache/validate_qwen3_cuda_replay.py \
  --model build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16 \
  --probe build-kv-gpu-cuda-g6/validation/prefill/qwen_gpu_probe \
  --requests build-kv-gpu-cuda/validation/qwen-correctness-frozen \
  --output build-kv-gpu-cuda-g6/validation/prefill/qwen-capture \
  --prefill-capture
```

output directory는 새 경로를 사용한다. 기존 baseline build는 재빌드하지 않는다.
이 검증을 처리량 향상 측정으로 사용하지 않는다.
Android GPU, integrated GPU, Windows 전체 build, 원격 CI는 이번 실행 범위 밖이다.
KV와 독립적인 기존 CUDA 개선은 [변경 분류](gpu-cuda-change-audit.md)를 참고한다.
