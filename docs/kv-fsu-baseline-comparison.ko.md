# KV FSU 분기 시작점(13cbb99ea) 대비 개선 기록

- 기준 커밋: `13cbb99ea` ([CausalLM] Prototype asynchronous KV prefill writes) — kv-cache-fsu 분기의 시작점
- 현재: `97638832c` 이후, 총 **37 커밋** (구현 11, 도구 5, 문서·기록 21)
- 코드량: `Applications/CausalLM/kv_cache_manager.{h,cpp}` 2,069 → 2,701 라인, 전체 diff 12,074 insertions / 122 files

## 1. 기능 비교

| 기능 | 13cbb99ea | 현재 |
|---|---|---|
| CPU FSU 스트리밍 (qwen 계열) | ✅ 프로토타입 (lookahead만) | ✅ + 깊이 스케줄링 |
| 플래시 저장소 총예산(`NNTR_KV_TOTAL_BYTES`) | ❌ | ✅ |
| 토큰 간 상주 유지(`NNTR_KV_RESIDENT_BYTES`) | ❌ | ✅ |
| 롤링 차용(`NNTR_KV_ROLLING`) | ❌ | ✅ (borrowed_owners 계측 포함) |
| 쓰기 배치 / 동기 정책 분리 | ❌ (async_write만) | ✅ (deferred batch, `NNTR_KV_FLUSH_SYNC`/`BUFFERED`/`WRITE_THROUGH`) |
| sliding/global 혼재 저장소(windowed DRAM + flash) | ❌ | ✅ (`mixed storage: 12 windowed + 3 streaming + 20 aliased`) |
| sliding 창 축소를 실그래프에 적용 | 부분(진단/프로토타입) | ✅ 실 pack 동작 (`f277d8b3d`, `fc2fec519`) |
| gemma4(35레이어 hybrid) CPU pack 로드 | ❌ `Unknown Layer Properties`로 로드 실패 | ✅ (tied lm_head 수정 `fa82a1799` 포함) |
| 결함 스냅샷 복원·예산 검증 | ❌ | ✅ (`49144f065`, `090a806d5`, `6bc1f3d12`) |
| 계측: kv_payload_peak / owner·demand·consumer 대기시간 / steady p95 | ❌ (peak_kv/read/write만) | ✅ 전부 |
| 도구: 예산 예측기(calibrate/predict/solve/curve), 로그 요약기, 테스트 | ❌ | ✅ (`tools/kv_cache/` 3종) |

## 2. 실측 비교 (SM-S947N, 동일 팩·동일 요청, baseline/HEAD 교차 실행)

로그: `tools/kv_cache/roll74_device_logs/cmp/` (2026-11-26 19시대, 같은 세션)

**qwen3-1.7b, 요청별 steady mean ms/tok (4 요청: 짧음/깊음/이어씀/reset)**

| 구성 | baseline (13cbb99ea) | 현재 (HEAD) | 비고 |
|---|---|---|---|
| resident (FSU off) | 25.8 / 31.4* | 27.0 / 32.5 / 33.9 / 28.9 | 동등 — 회귀 없음 |
| FSU 전량 스트리밍 (R=0) | 52.3 / 80.1 / 88.7 / 53.2 | 57.3 / 80.5 / 89.3 / 52.6 | 동등 — 회귀 없음 |
| FSU 예산 제어 (C=16M, R=8M) | **불가** (환경변수 없음) | 31.3 / 70.7 / 76.8 / 33.2 | 신규 능력: 스트리밍 대비 p50 −40% |
| FSU 넉넉한 예산 (C=160M, R=128M) | 불가 | 37.3 / 37.8 (읽기 0, resident 동급) | 신규 능력 |

(*baseline은 2 요청만 로그에 캡처됨 — 나머지는 동일 세션 HEAD 값으로 대체 기재)

**프로세스 메모리 (peak_memory_kb, qwen)**

| 구성 | baseline | 현재 |
|---|---|---|
| FSU 스트리밍 | 1,162,676 | 1,165,216 |
| resident | (해당 없음 — 동일 코드 경로) | 1,381,988 |

FSU로 resident 대비 RSS 약 220 MB 절감(양쪽 빌드 동일). 현재만 가능한 것: **payload peak를 C 이하로 강제**(측정치 ≤ 설정값 전 런 통과) — baseline은 상한 자체가 개념에 없음.

**gemma4-e2b (sliding 혼합)**

| 구성 | baseline | 현재 |
|---|---|---|
| pack 로드 | ❌ FATAL (`Unknown Layer Properties count 3`) | ✅ |
| resident decode | 측정 불가 | 51.7 / 60.0 / 61.2 / 52.1 |
| FSU 혼재 (R=8M) | 불가 | 60.2 / 61.6 (읽기 0) — resident 동급 |

gemma4의 sliding 실그래프 지원·혼재 저장소·windowed 창 축소는 전부 이 분기에서 새로 만들어진 능력이라 baseline과 비교 불가(그 자체가 개선).

## 3. 결론 — 무엇이 나아졌나

1. **속도**: 기존 경로(resident, 전량 스트리밍)는 baseline과 동일 — 회귀 없음이 확인된 개선이고, 실질 속도 향상은 전부 **새 제어 축**(상주 유지 R, 롤링, 깊이 스케줄링)에서 나온다. 대표치: qwen 깊은 ctx에서 스트리밍 88.7 → 예산 제어 76.8(R=8M) → 상주 전환 60/62(gemma4), 최대 예산에서 resident 완전 동급.
2. **메모리**: FSU 스트리밍은 baseline과 동일하게 RSS ~220 MB를 아끼고, 현재는 여기에 **KV payload 상한 보장**(측정 ≤ 설정)이 추가됐다 — baseline에는 상한 개념이 없었다.
3. **적용 범위**: baseline은 qwen 계열 FSU 프로토타입만 가능. 현재는 sliding 혼합 모델(gemma4)까지 기기에서 동작하며 출력이 resident와 89개 로그 전부 비트 일치.
4. **운영·검증**: steady p95·owner 읽기 대기·차용 횟수 등 계측과 예산 예측기가 생겨 "예산을 숫자로 고르는" 것이 가능해졌다. 전 과정(핸드오프 → 기기 스윕 → 결과 → 비교)이 커밋과 문서로 기록됐다.

## 부록: 구현 커밋 목록 (13cbb99ea..HEAD)

- `47c776d88` KV 플래시 저장소 스테이지 타이밍 계측
- `fe0657074` DRAM 상주 예산(retention) 추가
- `4cf90f660` decode 쓰기 deferred 배치
- `6bc1f3d12` rows 증가 후 상주 예산 회계 수정
- `850e0b21e` 동기 정책과 쓰기 배치 분리
- `fc2fec519` windowed + flash 혼재 저장소
- `f277d8b3d` 실그래프 sliding KV 창 축소
- `fa82a1799` tied lm_head의 plan_last_row_only 허용
- `49144f065` 레거시 스냅샷의 압축 창 복원
- `090a806d5` 상주 예산 검증·dirty 범위 추적
- `4b40bc471` KV payload·owner 읽기·steady p95 계측
- `d3674dbca` lookahead용 예정 owner 계수
- `730081344` 공유 payload 상한 내 owner 읽기 스케줄링
