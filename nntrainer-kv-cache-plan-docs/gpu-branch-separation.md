# 출력층 최적화 브랜치 분리 (2026-10-30)

사용자 요청에 따라 KV/G6 확장과 기존 FP32 출력층 변환 최적화를 분리했다.

| branch | 포함 범위 |
|---|---|
| `feat/kv-cache-cuda-replay` | KV/G6 구현, 성능 측정 도구·기록, 지원 조건 검사; 출력층 최적화 제외 |
| `fix/cuda-host-head-fp32` | 위 KV/G6 branch를 기반으로 출력층 최적화와 전용 정밀도 테스트만 추가 |

별도 branch는 G6 host tail 계약에 의존하는 작은 추가 변경이다.
현재 구현을 PR #4460이나 upstream main에 단독 적용할 수 있다는 의미는 아니다.
공통 HalfTensor/CPU backend 또는 legacy CUDA 경로까지 확장하지 않았다.

이미 push한 `911eb7e3` 및 성능 기록은 history에 보존하고,
KV/G6 branch의 후속 commit에서 출력층 변경과 전용 precision test만 제거했다.
DEV_ACT=1 거절, 그 admission test, KV header 주석 수정은 KV/G6 branch에 유지했다.
새 branch의 추가 diff는 다음 두 파일뿐이다.

- `nntrainer/layers/llm/tie_word_embedding.cpp`
- `test/unittest/models/unittest_kv_cache_cuda_graph.cpp`

`gpu-cuda-replay-performance.md`의 수정 전후 수치와 수치 차이 정정은
당시 측정 artifact의 기록이다. 기본 KV/G6 branch에는 **수정 전** 수치를 적용하고,
출력층 최적화 branch에는 **수정 후** 수치를 적용한다.
이 분리 자체로 성능 측정을 다시 수행한 것은 아니다.

분리 후 각 source 상태에서 G6 build와 네 Meson target을 다시 검증한다.
기본 branch는 core31 + CUDA storage8/graph7/replay4,
출력층 branch는 core31 + CUDA storage8/graph7/replay5다.
FP32 head 전용 precision test 1개의 차이다.
기존 공개 모델·CPU·NDK 검증 범위와 원격 Actions422 제약은
[성능 및 플랫폼 기록](gpu-cuda-replay-performance.md)에 남아 있다.

분리 검증 로그는 ignored local
`build-kv-gpu-cuda-g6/validation/branch-split/` 아래
`kv-targeted-build.log`/`kv-tests.log` 및
`head-build.log`/`head-tests.log`에 보존한다.
기존 출력층 최적화 runtime은 같은 디렉터리의 `optimized-runtime/`에 동결했다.

비교는 다음 명령으로 할 수 있다.

```bash
git diff feat/kv-cache-cuda-replay...fix/cuda-host-head-fp32
```

추가 개발 위치는 기존 WSL `/home/jayden/nntrainer`이며
기본 checkout은 `feat/kv-cache-cuda-replay`로 유지한다.
두 branch 모두 `https://github.com/jayden0701/nntrainer`에 push한다.
