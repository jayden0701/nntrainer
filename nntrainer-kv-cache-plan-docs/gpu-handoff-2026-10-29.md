# GPU 작업 인계 — PC/WSL 우선 (2026-10-29)

> 브랜치 분리 후 현재 적용 범위는 [출력층 최적화 분리 기록](gpu-branch-separation.md)을 따른다.
> KV/G6 기본 branch에서는 출력층 최적화를 제외하고, `fix/cuda-host-head-fp32`에만 추가한다.

## 현재 상태

처음에는 계획만 작성했으나 후속 구현·테스트 승인에 따라
**CUDA eager Qwen3 A+B 및 G6 decode capture/replay**까지 구현하고 현재 PC에서 검증했다.
G6 성능 비교와 기존 host 출력층의 가중치 변환 비용 개선도 완료했다.
[성능·플랫폼 확인 기록](gpu-cuda-replay-performance.md)을 먼저 확인한다.
Android 기기는 없다. 최신 결과는 [G6 실행 기록](gpu-cuda-replay-execution.md)과 [eager 실행 기록](gpu-cuda-execution.md)에 있다.

읽는 순서:

1. [G6 decode replay 실행 기록](gpu-cuda-replay-execution.md)
2. [eager 실행 기록](gpu-cuda-execution.md)과 [단계별 계획](gpu-extension-plan.md):
   G0–G5 및 scoped G6 decode 검증 완료, prefill capture/OpenCL/Android 후속
3. 초기 [분석](gpu-extension-analysis.md)은 구현 전 기록
4. 필요 시 [기존 CPU 인계](handoff-2026-10-29.md)

## 위치와 보존할 상태

- WSL `/home/jayden/nntrainer`,
  Windows `\\wsl.localhost\Ubuntu\home\jayden\nntrainer`.
- branch `feat/kv-cache-cuda-replay`,
  시작점 `25230aafe1a230e8dc497921d1149d52f6eabdad` (PR #4460).
  G6 성능 도구 d15e3e6d, host head 최적화911eb7e3; 최신 HEAD는 git log로 확인한다.
- base `141871a58ff439e3caeac40c1632ff8ac3543d20`.
- 사용자 요청으로 기존 작업을 새 branch로 옮겼다.
- push 대상은 `jayden=https://github.com/jayden0701/nntrainer`다.
  현재 commit/push 상태는 git log/status와 remote ref로 확인한다.
- origin은 `/mnt/c/nntrainer`. GitHub PR ref는 upstream URL로 fetch한다.
- Windows `C:\nntrainer`, 기존 다른 branch, `subprojects/CLBlast/` 보존.
- 기존 `build/`, `/usr/local` NNTrainer와 새 runtime을 섞지 않는다.
- WSL idle 종료 때 `/tmp`가 사라졌으므로 재개 자료는 build와 `/home/jayden/.cache`에 둔다.

## 환경과 지원 profile

RTX5070/driver610.88/sm120, CUDA13.0 runtime/NVRTC/cuBLAS, GCC14를 준비했다.
Linux GPU driver 추가 설치 없음. xgrammar recursive submodule init 완료.

- `/home/jayden/.cache/nntr-tools/bin/clang-format`:14.0.6, ruff도 같은 venv.
- `/home/jayden/.cache/nntr-gpu-model/bin/python`: HF CPU FP32 reference용.
- NVRTC device pattern 및 mapped handoff8회 PASS.
- concurrentManagedAccess=0, strict managed KV는 거절.

`NNTR_CUDA_KV_CORE=1`, native FP16 KV, NCHW token-major,
batch1/session1/linear, Qwen3 final V FC, resident eager graph가 지원 profile이다.
direct 쓰기가 기본이며 `NNTR_CUDA_KV_DIRECT=0`은 append-copy 비교 경로다.
실제 모델은 mapped activation + device KV, FP16 FC/norm + FP32 embedding/tied head다.
필수 CUDA flag와 graph opt-out은 실행 기록의 전체 env를 사용한다.

MemAllocator vtable은 optional AllocationReporter로 보존했다.
`usesCoreStorage()`는 CPU/CUDA 공통, `usesCpuStorage()`는 CPU만 뜻한다.
private cpu_cache_ 이름은 유지했다.

prefill capture, 다중 stream/async publication, managed coherence,
OpenCL/Android/NPU, 실제 모델 CPU↔GPU 전환, INT8/alias/hybrid는 후속이다.
pattern handoff 결과를 이들 기능의 증거로 사용하지 않는다.

## 검증과 artifact

G6 최신 build는 `build-kv-gpu-cuda-g6`다.
core31/storage8/eager graph7/replay4 PASS(출력층 branch는 replay5 PASS), CPU243 PASS+기존1 SKIP,
sanitizer31 PASS다. 최신 Qwen fixed-token/실제 greedy/HF/snapshot/2-turn 검증과
실제 graph launch audit가 모두 PASS다.
출력층 수정 후에는 `validation/perf-session/optimized-correctness/manifest.json`도 PASS다.
수정 후 ordered/replay byte 동일성과 수정 전후 수치 차이를 구분한다.
911eb7e3의440쌍 byte 동일 주장은 오류이며 성능 기록에 정정 수치를 남겼다.
원본은 G6 build `validation/qwen-replay-final/manifest.json`,
`final-tests.log`, `cuda-call-audit/summary.json`에 있다.
아래 표는 보존한 **이전 eager 단계** artifact다.

| 항목 | 결과 / 경로 |
|---|---|
| CPU | 241 PASS + 1 기존 QINT4 SKIP, `build-kv-gpu-cpu-regression/validation/test-abi-final.log` |
| CUDA storage / graph / kernel | 7/7/8 PASS, `build-kv-gpu-cuda/validation/final-cuda-tests.log` + `admission-tests.log` |
| sanitizer core | 29 PASS, `build-kv-gpu-core-asan/validation/test-abi-final.log` |
| old vendor/new MemAllocator ABI | PASS, `build-kv-gpu-cuda/validation/allocator-abi/result.log` |
| Qwen3 12조합 | PASS, `build-kv-gpu-cuda/validation/qwen-correctness-frozen/manifest.json` |
| 2-turn snapshot / logits bytes | PASS, 같은 디렉터리 `sessions.json` / `literal-byte-audit.json` |
| 실제 CUDA audit | identity kernel308→0, logits 동일, `validation/cuda-call-audit/summary.json` |
| 1K/4K 성능 | 5% 회귀 gate PASS, `validation/performance/summary.json` |
| 정적 검사 | clang-format14, changed-line format, advanced Doxygen, diff check, ruff, py_compile PASS |

공개 모델은 `build-kv-gpu-cuda/models/` 아래 ignored 상태다.
revision/hash는 실행 기록과 gpu-model-manifest.json에 있다.
초기 실패 로그도 보존했다. Android NDK r26d 부분 compile/link 및8개 syntax-check는 PASS다.
Android runtime/전체 CausalLM cross-link/Windows/전체 CI/CodeQL은 미실행이다.
원격 Android/Windows workflow dispatch는 HTTP422(Actions disabled)로 거절되었다.

**변경 전 baseline은 `build-kv-gpu-cuda-baseline`**다.
Ninja를 다시 실행하면 현재 source로 덮어쓸 수 있다. `--no-rebuild`로만 실행한다.
baseline observer는 validation/headers의 HEAD archive와 연결했다.
candidate observer는 현재 전체 header/runtime과 맞춰 빌드한다.

## 재개 순서

1. git status/HEAD와 최신 실행 기록을 확인하고 기존 uncommitted 변경을 유지한다.
2. 필요 시 matching G6 build의 serial CUDA tests를 실행한다. 이전 eager/baseline runtime은 보존한다.
3. G6 decode 구현·검증은 별도 실행 기록에서 확인한다.
   1K/4K ABBA 및 head 병목 개선은 성능 기록을 확인한다.
   원격 CI 복구 후 전체 Android/Windows build, 이후 kernel 병목 profile을 진행한다.
   prefill capture/다중 stream/slot kernel 최적화는 후속이다.
4. 최적화 시 현재 eager output/copy/completion 결과를 유지한다.
5. host-mapped attention 전환/OpenCL은 독립 capability gate로 진행한다.
6. commit 요청이 생기면 AGENTS.md의 DCO `git commit -s`, `[component]`,
   `Co-authored-by: Codex (OpenAI)`와 PR template을 따른다.
