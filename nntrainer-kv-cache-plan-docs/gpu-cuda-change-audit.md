# CUDA 변경 분류 — KV manager와 독립적인 개선

사용자 요청에 따라 lm head 변경은 이 목록에서 제외했다.
기준은 기존 upstream `141871a5` / PR4460 `25230aaf`와
현재 `feat/kv-cache-cuda-replay`의 변경 이력이다.

## 기존 CUDA 구현에도 적용되는 변경

| 변경 | 기존에 있던 상태와 현재 동작 | 최초 commit |
|---|---|---|
| 짧은 grid attention 선행 읽기 | 기존 split-KV kernel의 다음 K 읽기 지연을 줄이는 성능 개선. 정확도 버그 수정은 아니다. kernel pair microbenchmark에서 13–19% 개선했으며 전체 모델 향상률과 구분한다. | [c827456e](https://github.com/jayden0701/nntrainer/commit/c827456e) |
| 과도한 decode 재캡처 감소 | 기존 CUDA가 live chunk 수 변경마다 graph를 폐기했다. 실제 kernel/grid/공유 메모리/HPW 선택이 유지되는 구간에는 graph를 재사용한다. 공개 Qwen context1024/decode160에서 capture3→1, replay154→158. 이 비효율은 KV manager 이전에도 존재했다. | [d9c0347b](https://github.com/jayden0701/nntrainer/commit/d9c0347bfda0dcc54b27525d29057f9fdf789050) |
| scalar position graph의 재사용 범위 제한 | live device position이 없으면 capture에 기록된 position과 key 수가 고정된다. 이때 다른 위치로 graph가 재사용되지 않도록 exact guard를 적용했다. 일반 CUDA graph 안전장치다. | [d9c0347b](https://github.com/jayden0701/nntrainer/commit/d9c0347bfda0dcc54b27525d29057f9fdf789050) |
| device-only FC의 CPU fallback 차단 | GPU FC가 실행되지 않았을 때 device-only operand를 CPU가 역참조하지 않고 오류를 보고한다. KV 개발 중 추가했지만 기존 CUDA FC에도 적용된다. | [608001dd](https://github.com/jayden0701/nntrainer/commit/608001dd) |
| capture 예외 정리와 launch 오류 확인 | KV 개발 중 공통 CUDA executor에 capture 종료 guard 및 decode launch/sync 반환값 검사를 추가했다. 실패 후 capture 상태가 남거나 오류가 누락될 가능성을 줄인다. | [608001dd](https://github.com/jayden0701/nntrainer/commit/608001dd) |

위 항목이 모두 KV와 독립된 별도 commit으로 분리되어 있다는 뜻은 아니다.
특히 `608001dd`는 CUDA KV 도입과 공통 안전장치가 함께 있는 commit이다.
별도 branch `fix/cuda-host-head-fp32`는 사용자 요청대로 lm head 최적화만 분리했다.

## KV/G6 도입으로 필요해진 변경

- 신규 direct V slot projection 및 [c0493936](https://github.com/jayden0701/nntrainer/commit/c0493936)의 load 최적화:
  새 G6 경로의 구현/개선으로, 기존 CUDA에 같은 V slot kernel이 있었던 것은 아니다.
- `allocateAndBindKVCache()`에서도 split-KV scratch를 준비:
  기존 weight-load 시점의 prewarm만으로는 새 KV 연결/초기화 순서에 충분하지 않았다.
- allocation kind, host accessibility, native backing 정보 구분:
  manager가 device pointer를 host pointer처럼 사용하지 않도록 하기 위한 공통 메모리 계약 확장이다.
- 초기 prefill capture 지원:
  기존 CUDA의 one-shot capture를 manager의 ordered step/직접 V producer/host tail과 연결한다.
  RoPE LUT와 GEMM scratch 사전 준비, capture 안의 host fallback 거절, 실패 시 position 미갱신을 포함한다.

## Prefill에 관해 바로잡을 점

Prefill capture 자체는 KV manager 이전부터 있었다.
`NNTR_CUDA_PREFILL_GRAPH=1`이면 discrete GPU에서도 시도했고,
unset이면 integrated GPU에서 기본 활성화했다. NVIDIA 노트북 GPU도 보통 discrete GPU다.

기존 조건은 `from==0`인 multi-token step이다.
capture → 한 번 실행 → graph 파괴 순서이며, 모든 prefill chunk를 저장해 반복 replay하는 기능은 아니다.
새 KV profile이 처음에는 이 조합을 거절했던 것을 연결하는 작업이다.

더 자세한 근거/검증:
[재캡처](gpu-cuda-recapture-execution.md),
[attention](gpu-cuda-attention-performance.md),
[브랜치 분리](gpu-branch-separation.md),
[prefill 연결](gpu-cuda-prefill-execution.md).
