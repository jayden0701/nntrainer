# 최적화별 sparse gather 성능 파이프라인

fd5fbb3e 리뷰(정상 성공 경로 비용 검토)의 최적화를 production에 적용한 뒤, 각 최적화의
기여도를 따로 측정하는 호스트 파이프라인이다. S26U 없이도 checkout에서 실행할 수 있고,
같은 절차를 기기에서 재실행해 비교할 수 있다.

## 측정 대상 (누적 변형)

| 변형 | 적용 최적화 |
|---|---|
| `baseline` | 기존 구현 (73f8feec4 본문과 동일한 경로) |
| `1_pages_reserve` | + destination pages reserve |
| `2_skip_discarded_rows` | + sliding 범위 밖 행 반복 제거 |
| `3_no_key_precopy` | + 회전할 K의 사전 memcpy 제거 (source에서 직접 회전) |
| `4_share_exact_pages` | + 변경 없는 선두 exact 전체 page 공유 |
| `5_rotary_tail_bitcopy` | + active rotary index만 회전, 나머지 차원 비트 복사 |

모든 변형은 한 템플릿의 플래그로만 다르다. all-true 변형은 production
`gatherSparsePrefix`와 바이트 단위로 같아야 하며, run.py가 이를 강제한다.

## 실행

Linux/WSL의 C++17 컴파일러와 Python 3가 필요하다. checkout의 헤더를 사용하므로
repo 안에서 실행한다.

```sh
python3 tools/kv_cache/sparse_prefix_opt_bench/run.py . --reps 12
python3 tools/kv_cache/sparse_prefix_opt_bench/run.py . --sanitize \
  --output tools/kv_cache/sparse_prefix_opt_bench/result-asan.json
```

## 무엇을 빌드하는가

- `bench.cpp`: 플래그별 gather 변형, fixture(작은 대조 행렬 + 24K→20K 큰 입력),
  `operator new` 계측, 체크섬 불변성 확인, 반복 시간 측정.
- `bench_support.h`: memcpy/resize/행 반복 계측 훅.
- `golden.cpp`: run.py가 **현재 checkout의 실제 gather 본문**을 그대로 추출해 생성.
  이 함수와 다른 바이트가 나오면 변형 중단이다.
- `nntrainer/utils/fp16.cpp`: FP16 변환 대조군.

## 결과 해석

- **카운터**: 요청된 할당 수/크기, resize 값 초기화 바이트, 지정한 memcpy 바이트,
  행 루프 실행(방문+스킵), 공유 page 수다. DRAM 트래픽이나 RSS가 아니다.
- **시간**: 큰 입력(31.25MiB payload, 축소된 D64)에서 반복 측정한 gather 시간이다.
  실제 Gemma4 geometry(600MiB)와 HTP 지연을 대변하지 않는다.
- delta는 누적 변형 기준이므로 각 행을 독립 개선율처럼 합산하면 안 된다.
- 작은 대조 행렬: FP16/FP32 × split/interleaved/no-RoPE × 여섯 run 배치 × 여섯 변형,
  총 216회 바이트 비교와 source 체크섬 불변성을 확인한다.

## 기기(S26U)에서 확인할 지표

호스트 카운터로 확인한 뒤, 기기에서는 같은 source/request에 대해 external TTFT,
gather/restore/bind 각 시간(`decode_prefix_restore_stats_`), high-water memory,
allocation 수를 함께 기록한다. 이 파이프라인의 변형 플래그는 기기 빌드에서 조건부
컴파일로 동일하게 적용해 비교할 수 있다.
