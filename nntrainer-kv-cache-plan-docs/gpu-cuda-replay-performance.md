# G6 성능 비교와 출력층 병목 — PC/WSL (2026-10-30)

> 브랜치 분리 후 현재 적용 범위는 [출력층 최적화 분리 기록](gpu-branch-separation.md)을 따른다.
> KV/G6 기본 branch에서는 출력층 최적화를 제외하고, `fix/cuda-host-head-fp32`에만 추가한다.

[G6 실행 기록](gpu-cuda-replay-execution.md)의 correctness 검증 후,
ordered eager와 capture/replay를 같은 연산·설정으로 비교했다.
측정 중 발견한 **기존 FP32 출력층의 전체 가중치 변환 비용**을 별도 커밋
`911eb7e3`으로 수정하고 같은 조건으로 다시 측정했다.
작업 branch는 `feat/kv-cache-cuda-replay`, remote는
`https://github.com/jayden0701/nntrainer`다.

## 비교 조건

- WSL Ubuntu26.04.1/x86_64, GCC14, RTX5070/sm120, driver610.88,
  CUDA runtime13.0/NVRTC13.0/cuBLAS13.1.1.3.
- Qwen3-0.6B dense FP16 FC/norm + FP32 embedding/tied head, batch1.
  device KV + mapped activation, single stream, max_seq_len8192,
  prefill chunk/activation plane128.
- 고정 입력 context1024/4096, decode32, process별 warmup1+측정3회.
  ordered→replay→replay→ordered(ABBA), profile/context별 측정6회.
  NNTR_NUM_THREADS=4, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1.
- 두 arm 모두 CORE/KV_GRAPH/DIRECT/ASYNC=1이다.
  GRAPH/M2B/DECODE_GRAPH만 ordered=0, replay=1로 바꾼다.
  같은 V slot 연산과 같은 출력층을 사용하므로 그래프 효과를 분리한다.
  prefill은 양쪽 모두 eager, layer/debug/copy trace는 끈다.
- timed run은 prefill+decode API를 포함하고 model 초기화·weight load는 제외한다.
  decode API는 첫 prewarm/capture 비용도 포함한다.
  steady inter-token 통계는 trial마다 처음 두 interval을 제외한
  30개×6회=180개를 pooling한다.
- trial 통계 p95는 nearest rank이며 표본6개에서는 최댓값이다.
  pooled interval들은 같은 trial 내부에서 상관되므로 독립180회 실험으로 해석하지 않는다.
  GPU clock을 고정하지 않았고 다른 PC/driver/모델에 대한 일반화는 하지 않는다.
  각 process 전후 nvidia-smi 상태도 artifact에 기록했다.

model revision:
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
변환 model SHA256:
`e1dc2fe5d60c690ee7920fc830a58e58c753afdd9f72c73a2e06f20e76ef2ad2`.

## 1. 수정 전: 그래프 효과가 작고 host head가 지배적

단위 ms, trial median(괄호 안 p95), 변화율은 replay/ordered−1이다.

| context | metric | ordered eager | replay | 변화 |
|---|---|---:|---:|---:|
| 1024 | run wall | 21645.975 (21818.327) | 21378.311 (21467.965) | −1.24% |
| 1024 | prefill compute | 7297.5 (7360) | 7327 (7352) | +0.40% |
| 1024 | decode32 API | 14335.5 (14456) | 14059 (14145) | −1.93% |
| 4096 | run wall | 44890.623 (46169.747) | 44523.611 (44657.306) | −0.82% |
| 4096 | prefill compute | 30598 (31588) | 30462 (30549) | −0.44% |
| 4096 | decode32 API | 14245 (14596) | 14038 (14160) | −1.45% |

| context | steady pooled interval | ordered median / p95 | replay median / p95 |
|---|---|---:|---:|
| 1024 | ms/token | 444.526 / 472.574 | 437.144 / 454.187 |
| 4096 | ms/token | 431.778 / 531.757 | 428.020 / 505.476 |

최초 별도 layer profile(context128/decode16)에서
`output_of_causallm` median365.807ms, trial steady423.171ms였다.
약86%가 host head에 해당한다. layer profile 수치는 계측이 없는 위 표와 분리한다.

### 기존 코드에서 발견한 문제와 수정 범위

이 현상은 KV storage 소유권 문제와 별개다.
PR #4460 시작점 `25230aaf`에도 있던
`HalfTensor::dot(FP32)`는 호출마다 FP32 vocab matrix 전체를 임시 FP16으로 좁힌다.
x86 FP16 GEMV fallback은 이 matrix를 다시 FP32로 넓혀 CPU BLAS에 넘긴다.
소스 주석은 이를 “one-time”이라고 설명하지만 실제로는 매 호출 발생한다.
정밀도도 원래 FP32 가중치에서 한 번 좁아진다.

`911eb7e3`은 명시적으로 선언된 G6 host tail만
기존 `shgemv`로 실행한다. 원래 FP32 가중치를 직접 읽고 입력 FP16 row만 넓힌 뒤
마지막 logits를 activation dtype으로 반올림한다.
가중치 전체 변환·복제나 영속 변환 cache를 만들지 않는다.
공통 HalfTensor/x86 backend, 일반 CPU 및 legacy CUDA 경로는 변경하지 않았다.
따라서 다른 CPU 모델의 같은 비용까지 해결했다고 주장하지 않는다.

이 host tail의 필수 조건인 DEV_ACT=0도 allocation 단계에서 검사하도록 했다.
기존 split-KV workspace 준비 및 threshold 재캡처 보완은
[G6 연결 기록](gpu-cuda-replay-execution.md#경계-문제와-검증)에 있는 별도 통합 수정이다.

## 2. 수정 후: 같은 workload 재측정

`911eb7e3`의 clean tracked source로 실행했다.

| context | metric | ordered eager | replay | 변화 |
|---|---|---:|---:|---:|
| 1024 | run wall | 6905.292 (6991.257) | 6936.576 (7006.553) | +0.45% |
| 1024 | prefill compute | 4411 (4491) | 4510.5 (4539) | +2.26% |
| 1024 | decode32 API | 2495 (2521) | 2442 (2512) | −2.12% |
| 4096 | run wall | 21267.677 (21427.491) | 21510.626 (21646.770) | +1.14% |
| 4096 | prefill compute | 18799.5 (18946) | 19042.5 (19140) | +1.29% |
| 4096 | decode32 API | 2466 (2477) | 2462.5 (2505) | −0.14% |

| context | steady pooled interval | ordered median / p95 | replay median / p95 |
|---|---|---:|---:|
| 1024 | ms/token | 77.995 / 79.304 | 76.264 / 78.437 |
| 4096 | ms/token | 77.083 / 77.819 | 76.589 / 78.212 |

수정 전후 replay steady median은 1K 437.144→76.264ms,
4K 428.020→76.589ms다. 약5.6–5.7배 개선은 주로 **출력층 수정의 효과**다.
현재 profile에서 그래프 자체의 decode 이득은 작고,
run wall은 eager보다 소폭 느리므로 전체 추론 가속을 주장하지 않는다.
양쪽 context 모두 사전 5% median 회귀 gate(run wall/prefill/decode API)는 PASS다.

모든 측정 trial에서 ordered captures/replays=0/0,
replay captures/replays=1/30, KV append_copy_bytes=0이었다.
KV backing은 양쪽 모두939524096bytes(896MiB)다.
driver 내부 migration이나 물리 copy0을 입증하는 통계는 아니다.

### 별도 profile 및 memory 진단

동일한 context128/decode16/warmup1/repetitions1 요청을
동결한 수정 전 runtime과 수정 후 runtime으로 순서대로 실행했다.
`NNTR_LAYER_PROFILE=1`, GNU `/usr/bin/time -v`를 사용한 **단일 진단 실행**이다.

| 항목 | 수정 전 | 수정 후 |
|---|---:|---:|
| host head decode median, warmup 포함32개 | 369.442ms | 22.933ms |
| measured trial steady median | 433.076ms | 89.637ms |
| process 전체 maximum RSS | 3056884KiB | 2144980KiB |

RSS는 초기화·weight load·warmup도 포함한 전체 프로세스 high-water mark다.
GPU VRAM peak가 아니며, 내부 `peak_memory_kb`의 일부 시점 표본과도 구분한다.
profile/메모리 진단은 위 ABBA latency 표의 표본에 포함하지 않았다.

## 정합성, 수치 변경과 기록 정정

출력층 수정 후 다음 검증은 PASS다.

- core31, CUDA storage8, 기존 CUDA graph7, replay5(실제 GPU 총20).
- non-FP16 CPU 회귀243 PASS+기존 deprecated QINT4 1 SKIP.
  기존 core ASan/UBSan/LSan31 PASS는 앞 G6 단계 결과다.
- FP32 head 가중치를 FP16으로 좁히면 구분되는 cancellation fixture로
  새 host tail이 원래 FP32 값을 사용하는지 검사한다.
- 공개 모델 fixed44 token, 64-key 경계 통과, 두 continuation turn,
  snapshot save/load, live/resume 비교.
- arithmetic/geography 각8개 실제 greedy token을 HF eager FP32와 비교.
  HF 대비 max_abs=0.0577316284, 기존 수치 gate PASS.
- **수정 후 ordered eager/replay끼리** 전체 logits가 bitwise 동일하고,
  continuation position114/161과 snapshot bytes도 일치한다.

`911eb7e3` 커밋 설명의 “440 before/after file pairs byte-identical”은
잘못된 기록이므로 이 문서와 후속 문서 커밋으로 정정한다.
**출력층 수정 전후** 440개 file pair는 byte 동일하지 않다.
동일한 것은 수정 후 ordered/replay 간 비교다.

수정 전후 실제 재감사 결과:

| 항목 | 결과 |
|---|---:|
| 비교 pair | 440 |
| byte 동일 pair | 14 |
| 전체 logits max_abs | 0.0078125 |
| step별 mean_abs의 최댓값 | 8.8653749243e−08 |
| 전체 element pooled mean_abs | 8.2403360060e−09 |
| 값이 다른 element 수 | 1960 |
| 생성/강제 token ID와 before/after position | 전부 동일 |
| finite | 전부 통과 |

이는 확인한 모델·입력 범위의 결과다. FP32 가중치 직접 사용으로 수치 경로가
달라지므로 모든 모델/입력에서 이전 출력과 동일한 것을 보장하지 않는다.
push한 commit history는 덮어쓰지 않았다.

## Android/Windows compile 확인과 미완료 범위

GitHub의 `causallm_android.yml`과 `windows_msvc_fp16.yml`을
`d15e3e6d` 시점 branch에 dispatch했으나 둘 다 HTTP422,
`Actions has been disabled for this repository.`로 거절되었다.
repository Actions permission API는 enabled=true를 반환했지만 실제 workflow는
실행되지 않았다. 설정을 바꾸지 않았으며 이를 CI 통과로 집계하지 않는다.

로컬에는 공식 Android NDK r26d(26.3.11579264)를 준비했다.
archive SHA1은 `fcdad75a765a46a9cf6560353f480db251d14765`다.

- Android28/AArch64로 실제 production core/memory sources를 compile하고
  최소 KV smoke ELF를 link했다.
- 기존 portable core31 test 코드의 translation unit도 compile했다.
  전체 test binary의 최소 closure link는 Tensor/BasicPlanner 의존성 부족으로
  실패했으며 이를 Android test 통과로 집계하지 않는다.
- CausalLM `causal_lm.cpp`/`transformer.cpp`/`kv_cache_manager.cpp` 및
  `nntrainer/models/neuralnet.cpp`를 FP16 on/off로 syntax-check, 8/8 PASS.
- Android 기기가 없으므로 실행 테스트는 없다.
  전체 CausalLM Android cross-link 및 Android GPU 검증, Windows native build,
  전체 원격 CI/CodeQL은 여전히 미검증이다.

NDK 출처:
[Android NDK 다운로드 목록](https://github.com/android/ndk/wiki/Unsupported-Downloads).
Actions API:
[workflow dispatch](https://docs.github.com/en/rest/actions/workflows),
[repository permissions](https://docs.github.com/en/rest/actions/permissions).

## artifact와 재현

artifact prefix는 `build-kv-gpu-cuda-g6/validation/perf-session/`이다.

| 경로 | 내용 |
|---|---|
| `paired/summary.json` | 출력층 수정 전 ABBA |
| `optimized-paired/summary.json` | 수정 후 ABBA, clean source911eb7e3 |
| `baseline-runtime/manifest.json` | 수정 전 observer 및 실제 ldd dependency closure 동결 |
| `layer-profile.*` | 최초 병목 분석 |
| `{baseline,optimized}-layer-memory.*`, `layer-memory-summary.json` | 후속 layer/RSS 진단 |
| `optimized-correctness/manifest.json` | fixed/greedy/HF/snapshot 결과 |
| `before-after-byte-audit.json` | 정정한440쌍 수치 재감사 |
| `optimized-tests.log` | G6 네 test target |
| `android/summary.json`, `android/syntax-summary.json` | NDK compile/link 범위 |
| `ci-status.json` | workflow dispatch422 원문 요약 |

CPU 회귀 로그는
`build-kv-gpu-cpu-regression/validation/perf-session-tests.log`다.
처음 기록한 paired manifest의 HEAD는 a91a6f2d이며 source diff hash가 별도 있다.
측정 도구 변경은 d15e3e6d에 커밋했다.
출력층 수정 후 correctness manifest도 d15e3e6d+변경 source hash를 기록하며
해당 변경은911eb7e3이다. optimized ABBA는911eb7e3의 clean source다.
동결 baseline은 runtime library hash를 기준으로 재현하고 Ninja 재빌드하지 않는다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py \
  --build build-kv-gpu-cuda-g6 --kv-statistics \
  --output build-kv-gpu-cuda-g6/validation/perf-session/qwen_gpu_benchmark
python3 tools/kv_cache/benchmark_qwen3_cuda.py --mode replay \
  --model build-kv-gpu-cuda/models/qwen3-0.6b-dense-fp16 \
  --candidate-probe build-kv-gpu-cuda-g6/validation/perf-session/qwen_gpu_benchmark \
  --contexts 1024 4096 --decode-tokens 32 --warmup 1 --repetitions 3 \
  --output build-kv-gpu-cuda-g6/validation/perf-session/new-paired
```

기존 output은 덮어쓰지 않는다. 구형 PR baseline header로 observer를 만들 때는
새 통계를 참조하지 않도록 `--kv-statistics`를 생략한다.
모델·binary·로그는 ignored local artifact이고 source/문서만 push한다.

남은 우선순위는 원격 CI 실행 환경 복구 후 Android/Windows 전체 build,
그 후 GPU attention/V-slot kernel profile에 따른 다음 병목 분석이다.
prefill capture/다중 stream/OpenCL/Android GPU는 별도 확장 범위다.
