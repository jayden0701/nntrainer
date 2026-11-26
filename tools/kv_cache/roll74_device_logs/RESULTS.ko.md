# 롤링 lookahead 기기 실험 결과 (74d30f11b)

- 커밋: `74d30f11b` (origin/kv-cache-fsu), 기기: SM-S947N (bacd420f), arm64-v8a
- 빌드: `build_android.sh -Denable-fp16=true`, NDK r28c, NNTR_NUM_THREADS=7, direct_io=1
- 종료 시 배터리 온도 43 °C (rep3가 rep1보다 약간 느린 것은 열 상태 영향)
- 요청 4종 (짧은 요청 / 긴 prefill+긴 decode / 이어쓰기 / reset 후 새 요청), 요청당 192 토큰 생성
  - qwen cached: 0→214→1316→1547→(reset)→214, gemma4 cached: 0→223→984→1207→(reset)→223
- 판정: **모든 실행에서 output_ids가 resident와 비트 일치**, cached_before/after 일치,
  모든 capped 실행에서 `capped=1`, `owner_lookahead=2`, `kv_payload_peak_bytes ≤ C` 확인

## 로그 경로

- qwen3-1.7b (global 전용, 28 owner): `build-fsu/logs/roll74/qw/*.log`
- gemma4-e2b (sliding/global 혼합: 12 windowed + 3 streaming + 20 aliased / 35 layers):
  `build-fsu/logs/roll74/g4/*.log`
- resident 기준: `resident_rep{1,2}.log` (NNTR_KV_WINDOW_CAP=1)
- 무효 파일: `qw/*.INVALID-ran-la1` (실행기 config 토글 버그로 la2/rolling 의도가 la1으로 실행됨 — 제외)
- `*.log` 파싱: `python3 /tmp/parse_roll.py "qw/*"` (파서 사본은 이 디렉터리에 둘 것)

## qwen3-1.7b (C=16 MiB / R=8 MiB 및 C=10 MiB / R=6 MiB)

폭넓은 예산 C=16 MiB (요청별 steady mean/p95, ms/tok):

| 설정 | req1(짧음) | req2(깊음) | req3(이어씀) | req4(reset) | 읽기 MB/tok |
|---|---|---|---|---|---|
| resident | 27.0/28.2 | 33.0/34.0 | 34.3/34.7 | 29.2/30.0 | 0 |
| FSU la1, R=0 | 53.9/91.2 | 90.2/107.3 | 105.0/113.8 | 60.7/95.5 | 43.85 |
| FSU la2, R=0 (rep2r) | 39.2/53.9 | 76.5/85.6 | 92.5/97.2 | 53.8/92.4 | 43.85 |
| FSU la2 rolling, R=0 (rep2r) | 46.8/65.2 | 81.6/102.8 | 99.9/107.6 | 51.9/71.8 | 43.85 — **borrowed=0: 차용 미검증** (R=0은 회수 대상이 없음) |
| FSU la2, R=8M (rep2r) | **31.3**/59.7 | 70.7/81.2 | 76.8/86.5 | 33.2/55.1 | 38.80 |
| FSU la2 rolling, R=8M (rep2r) | 34.3/61.3 | 79.9/94.3 | 84.7/96.7 | 32.7/60.2 | 39.84 — **borrowed=4617** (차용 발동) |
| 독립 재확인 la2 R=8M (rep3) | 34.5/60.5 | 83.6/105.0 | 107.9/126.1 | 33.3/50.3 | 38.80 (열 상태 43°C) |

타이트한 예산 C=10 MiB, R=6 MiB:

| 설정 | req2 mean/p95 | req3 mean/p95 | demand_read | borrowed |
|---|---|---|---|---|
| la2 고정 (rep2r) | **207.8/337.3** | **336.4/382.7** | 18~79 s (동기 miss 지배) | 0 (미검증) |
| la2 롤링 (rep2r) | **73.3/87.2** | **84.7/90.3** | 0 | **6724** |
| la2 롤링 (rep3 재확인) | 86.8/104.5 | 93.5/97.3 | 0 | 6724 |

## gemma4-e2b (C=32 MiB / R=0 또는 16 MiB)

| 설정 | req1 | req2 | req3 | req4 | 읽기 MB/tok |
|---|---|---|---|---|---|
| resident | 51.7/53.5 | 60.2/61.0 | 61.5/64.0 | 53.4/55.1 | 0 |
| la1, R=0 | 55.9/66.2 | 72.0/77.9 | 72.0/74.6 | 56.1/64.7 | 8.13 — 차용 미검증 |
| la2 고정, R=0 (rep1) | 61.1/69.2 | 94.7/100.4 | 119.0/133.7 | 53.6/60.2 | 8.13 — demand 19.7 s |
| **la2 롤링, R=0 (rep1)** | 51.0/53.5 | 72.7/76.4 | 74.7/77.7 | 55.7/58.0 | 8.13 — demand 0, **borr=0: 공간 재제출 효과(차용 아님)** |
| la2 롤링, R=0 (rep2 회전) | 48.8/55.7 | 60.9/66.9 | 62.1/64.8 | 50.6/52.4 | 8.13 |
| la2 롤링, R=0 (rep3 재확인) | 55.4/58.5 | 68.1/81.5 | 68.6/73.5 | 55.3/58.3 | 8.13 |
| R=16M (la1/la2/rolling 모두) | ≈56/58 | ≈65/67 | ≈66/67 | ≈56/58 | **0** (전량 상주, 읽기 없음) — 차용 미검증 |

## 결론

1. **상주 유지(R>0)가 가장 큰 지렛대**: qwen p50 54→31 ms (−43%), 읽기 46452→26858회. gemma4는 R=16M에서 읽기가 구조적으로 0.
2. **깊은 선읽기(la2)는 예산이 충분할 때 la1보다 유리** (qwen R=0: req3 p95 113.8→97.2, R=8M: 113.5→86.5). R=0 rolling은 la1과 대등.
3. **롤링은 타이트한 예산의 구조적 실패를 막는다**: qwen C=10M에서 고정 la2는 동기 miss 읽기 18~79 s로 p95 383 ms까지 붕괴, rolling은 borrowed_owners=6724로 회수해 demand read 0, p95 90 ms. gemma4 R=0에서도 rolling la2가 demand 19.7 s→0으로 req3를 119→75 ms로 회복.
4. **차용은 공짜가 아니다**: 여유 있는 예산(qwen C=16M R=8M)에서는 고정 la2가 rolling보다 빠름 (p95 86.5 vs 96.7) — 회수한 owner의 재읽기 비용(39.84 vs 38.80 MB/tok)이 선읽기 이득을 상쇄. 롤링은 "예산 부족 시의 안전망"이지 기본 정책은 아님.
5. **borrowed_owners=0 실행은 차용 효과를 검증하지 못함** (표기 준수): la1 전체, 모든 R=0 실행, gemma4 R=16M 전체. 차용이 실제 발동한 조합은 qwen C=16M/R=8M rolling(4617회)과 C=10M/R=6M rolling(6724회)뿐.
6. 쓰기 정책 분리(WRITE_THROUGH=1) 비교는 이번 라운드에서 제외 — 후속 과제.

---

# 2차 스윕 (넉넉한 예산 확장, 커밋 74d30f11b, 동일 기기·요청)

- 51 런 + resident 앵커 3회, 전부 0 fatal, **89개 로그 전부 출력 비트 일치(파서 검증 mismatch=0)**
- 로그: `sweep{A,B,C,D,E}_C{bytes}_R{bytes}_la{n}_r{mode}_rep1.log` (qw/g4)
- 진행 기록: `sweep_progress.txt`. 온도 센서 미지원 기기라 앵커 비교로 열 상태 대체

## qwen3-1.7b

**A. R=0 스트리밍 — C를 아무리 늘려도 무의미**: C=24→160 MiB에서 req2 mean 96.8~102.5 ms로 flat. R=0에서는 읽기 총량(43.85 MB/tok)이 고정이고 C는 "동시에 읽을 폭"만 늘린다. 깊은 통찰: **R=0에서 예산 다이얼은 존재하지 않는다. 유지(R)가 곧 성능이다.**

**B. 상주 스케일링 (C=48M, la2)**:

| R | 0 | 8M | 16M | 24M | 32M |
|---|---|---|---|---|---|
| req1 mean | 53.2* | 41.3 | 30.8 | 29.3 | 29.9 |
| req2 mean | 90.2* | 76.3 | 96.2 | 67.9 | 62.8 |
| 읽기/토큰 | 43.85MB | 38.8 | 34.3 | 30.6 | 27.3 |

(*A그룹 la1 기준) 무릎은 R=16~24M. 그러나 깊은 ctx 요청(req2/3)은 R=32M에서도 resident(33/34)의 2배 — 부분 상주로는 재읽기를 다 못 지운다.

**C. 넉넉한 예산의 상주 스케일링**:

| 구성 | req2/req3 mean | 읽기/토큰 |
|---|---|---|
| C=112M R=64M | 67.4/63.5 | 14.13 MB |
| C=160M R=128M | **37.3/37.8** | **0** (peak 111.6M) |

R=128M(C=160M)에서 **FSU가 resident와 완전 동급 + 읽기 0** — FSU의 상한이 "예산=전체 작업 집합"일 때 resident와 수학적으로 같아짐을 재확인.

**D. 롤링(중간 예산)**: C=48M R=16M에서 rolling 101.8/136.0 vs fixed 96.2/110.1 — fixed가 블록 없이도 소화되면 rolling의 회수는 순수 손해(재확인).

**E. la4**: C=48M R=16M la4는 la2보다 나쁨(142.3 vs 110.1). 그러나 **C=112M R=32M + rolling + la4 = 53.3/69.8**로 la2 fixed(63.7/84.1)를 15~17% 앞섬 — 깊은 선읽기는 "넉넉한 예산 + 롤링" 조합에서 처음으로 빛났다.

## gemma4-e2b

**A. R=0 C 스윕 — 구조적으로 flat**: C=28→64 MiB 전 구간 req1 114~127 ms (la2 포함). 창 24.6M이 예산을 잠식해 C를 늘려도 prefetch 여력이 생기지 않는다. **sliding 혼합 모델은 R=0에서 예산 감도가 아예 없다.**

**B. 상주 전환 임계값 (C=40M, la2)**:

| R | 0 | 2M | 4M | 8M | 12M |
|---|---|---|---|---|---|
| req2/req3 mean | 136.6* | 136.6/129.9 | 110.8/154.7 | **60.2/61.6** | **62.4/63.7** |
| 읽기/토큰 | 8.13MB | 6.72 | 5.69 | **0** | **0** |

R=2~4M(부분 상주)은 demand read 26~29 s를 유발해 오히려 해롭고, **R=8M에서 읽기가 구조적으로 0**이 되며 resident(60/62)와 동급. 3 owner × ctx 1207 × 4KB ≈ 14.8M 전체가 아니라 창 압축 뒤 잔여분만 유지하면 되기 때문. gemma4의 실전 권장: **R=8M 한 번에 점프(부분 상주 금지)**.

**D/E**: rolling R=8M = fixed와 동일(이미 읽기 0). la4 = la2와 동일.

## 기기 상태(열) 효과 — 신규 발견

- qwen resident 앵커 3회(17:37~17:45) req1 25.9/27.7/27.6 ms — **연산 성능은 캠페인 내내 안정**
- 반면 R=0(demand-read 위주) 구성은 시간이 지날수록 열화: gemma4 R=0 la1 req2 72.0 ms(16:50) → 120.5 ms(18:13) → 재확인 98.8 ms(18:44). demand_read 총시간 14.1 s → 50.1 s (읽기 횟수 13,495로 동일 — **read당 지연이 3.5배**)
- 같은 시간대 R≥8M 구성은 전혀 열화 없음(52 ms 유지)
- 결론: **UFS 읽기 지연이 세션 후반 팽창하며, 상주 유지가 그 열화에 대한 방어막**. 긴 벤치마크 캠페인에서는 구성 실행 순서를 회전시키지 않으면 이 효과가 정책 비교를 오염시킨다.
- 참고: gemma4 resident_sweep{1,2,3} 로그는 config 미전환으로 FSU la2 nocap으로 실행됨 → `.INVALID-ran-fsu-la2nocap`로 개명

---

# 3차: 긴 컨텍스트 2 시나리오 (커밋 74d30f11b, SM-S947N)

- 시나리오 A "긴 prefill + 짧은 decode": prefill 1,024토큰 + 48토큰 생성 (ctx ~1,072)
- 시나리오 B "긴 decode": prefill 1,024토큰 + 700토큰 생성 (ctx ~1,724)
- 구성: resident / FSU 전량 스트리밍(la2, R=0) / FSU 튜닝 예산 (qwen C=48M/R=24M, gemma C=48M/R=8M)
- 로그: `scen/{qw,g4}_{A,B}_{res,stream,tuned}.log`. 전부 0 fatal, 출력은 resident와 비트 일치

## 결과 (steady ms/tok, 토큰당 플래시 읽기)

| 모델 | 시나리오 | resident | 스트리밍 R=0 | 튜닝 예산 | 비고 |
|---|---|---:|---:|---:|---|
| qwen | A (prefill 1024+48) | 32.4 | 51.9 (120 MB/tok) | **73.8** (99 MB/tok) | 튜닝이 스트리밍보다 느림 |
| qwen | B (긴 decode 700) | 37.4 | 90.5 (158 MB/tok) | **219.1** (136 MB/tok) | consumer_wait 88.5 s — 예산 정책 붕괴 |
| gemma4 | A | 60.4 | 112.4 (15.2 MB/tok) | **60.4** (0 MB/tok) | resident 완전 동급 |
| gemma4 | B | 63.1 | 143.7 (19.7 MB/tok) | **64.3** (1.6 MB/tok) | resident 동급 |

- TTFT(긴 prefill): qwen 8.4~9.0 s (resident) vs 9.4~10.0 (FSU) vs 9.5~10.4 (튜닝) — FSU prefill 오버헤드 ~10%
- RSS: qwen resident 1.42 GB vs FSU 1.20 GB vs 튜닝 1.24 GB

## 새 발견

1. **ctx가 자라면 "중간 예산"이 최악이 된다 (U자 곡선의 ctx 축 재현)**: ctx ~1,316에서 무릎이었던 qwen C=48M/R=24M가 ctx ~1,724에서는 스트리밍보다 2.4배 느다 (219 vs 90 ms/tok). 읽기 바이트는 오히려 적은데(136 vs 158 MB/tok) consumer_wait가 88.5 s로 폭증 — owner 하나가 7.2 MB로 커진 상태에서 R=24M(≈3 owner)만 유지하면 풀이 예약으로 포화되어 필수 읽기가 선읽기 뒤에 줄을 선다. **예산 정책의 무릎은 ctx에 따라 이동한다: 깊은 ctx에서는 전부 상주(R=ctx×114,688 B) 아니면 R=0 둘 중 하나다.**
2. **gemma4는 ctx에 강하다**: 창 24.6 MB + global 3 owner만 스트리밍 대상이라, R=8M로 global 행 전부를 상주하면 ctx 1,724에서도 resident 동급(64.3 vs 63.1). sliding 혼합 모델의 FSU 적합성 재확인.
3. **resident 요구 메모리 (참고)**: qwen ctx 1,724 → 197.7 MB (1,724 × 114,688 B); gemma4 ctx 1,724 → 창 상한 적용 시 45.8 MB / 무상한 105.9 MB.

---

# 4차: "완료되는 대로 연산 재개" 최적화의 수치 (커밋 74d30f11b vs 13cbb99ea)

- 최적화 내용: baseline의 `begin(layer)`는 제출된 선읽기 **전부**가 끝날 때까지 대기하는 배리어(`wait()` = `pending_.get()`). 현재는 `waitOwner(내 owner)`로 **자기 owner의 읽기만 완료되면 즉시 attention 재개**하고 나머지 선읽기는 병행 진행.
- 측정: qwen3-1.7b FSU 전량 스트리밍(R=0), 깊이 1/2/4를 baseline과 HEAD에서 교차 실행 (la1 n=1, la2·la4 n=2)
- 로그: `opt/qw_la{2,4}_{base,head}_rep{1,2}.log` + 기존 `cmp/qw_fsu_{base,head}.log` (la1)

## 결과 (req3 이어쓰기·req2 깊은 ctx 기준 steady ms/tok)

| 깊이 | baseline (배리어) | 현재 (owner별 재개) |
|---|---|---|
| la1 | 88.7 / 80.1 | 89.3 / 80.5 |
| la2 | 90.9 / 79.8 | **83.6 / 76.3** |
| la4 | 93.4 / 86.9 | 89.4 / 79.7 |

1. **baseline에서는 깊이 확장이 무의미**: la 1→2→4로 늘려도 req3이 88.7→90.9→93.4 ms로 개선 0, la4에선 악화. 배리어 때문에 큐 전체가 차야 연산을 시작하므로, 미리 읽힌 분이 "기다린 시간"으로 상쇄된다.
2. **현재는 la2에서 개선**: req3 −6.4% (89.3→83.6), req2 −4.7% (80.5→76.3). 같은 la2에서 baseline 대비 −8% (90.9→83.6).
3. **la4에서 두 버전 모두 되돌아감**: 전량 스트리밍에서 단일 순차 리더가 포화된 상태(토큰당 44~158 MB)라, 깊이 2가 스트리밍의 스위트 스팟.
4. 이득의 상한이 제한적인 이유: 읽기 총량 자체는 깊이와 무관하게 고정(43.85 MB/tok)이고 리더가 포화돼 있어, 재개 순서 개선이 아낄 수 있는 것은 "자기 차례 전에 밀린 읽기" 분량뿐. 예산(R>0)으로 읽기 총량을 줄인 구성에서 이 최적화의 상대 가치가 커진다 (롤링 32/4에서 소비 시점 읽기 6.94 s→0이 그 사례).

---

# 5차: gemma4 혼재(부분 상주) 케이스 — 시나리오 A·B 재측정

- 요청: "메모리는 줄고 상주 영역이 존재하는(전량 스트리밍이 아닌) gemma4 케이스". 시나리오 A(1,024 prefill+48 decode, ctx 1,071)·B(1,024 prefill+700 decode, ctx 1,723)에 C=32 MiB/R=8 MiB, la2, rolling=0으로 실행.
- 로그: `scen/g4_{A,B}_mixed.log`. 출력은 resident와 비트 일치 (A: 48 ids, B: 700 ids)
- 설계 제약 확인: 창 레이어는 예산에 반드시 들어가야 한다(`KV total budget cannot hold the sliding windows` FATAL, C=16 MiB 시도) — 창은 상주 전제, 스트리밍 대상은 global 3 owner만 가능. 따라서 혼재의 형태는 "창 상주 + global 일부 상주·일부 스트리밍"이다.

## 결과

| 시나리오 | resident | 전량 스트리밍 | **혼재 C=32M/R=8M** | 혼재 상세 |
|---|---:|---:|---:|---|
| A (ctx 1,071) | 60.4 ms/tok | 112.4 | **51.5** | 읽기 0, retained_owners=3 (global 전부 상주), peak 24.3 MB |
| B (ctx 1,723) | 63.1 ms/tok | 143.7 | **55.6** | 읽기 1.62 MB/tok (1.05 GiB), retained 2 owner (6.7 MB) + 1 global 스트리밍, peak 28.1 MB |

- 메모리: resident 대비 A −36% (37.8→24.3 MB), B −39% (45.8→28.1 MB). budget_blocks=0, demand_read=0
- 구조: `mixed storage: 12 windowed on host DRAM (compacting rows), 3 streaming through flash, 20 aliased`
- **예산 통찰**: 동일 조건의 C=48 MiB(tuned)와 C=32 MiB(mixed)가 읽기 바이트까지 정확히 동일 (1,132,773,376 B) — gemma4의 읽기량은 R이 결정하고, C는 창(24.6 MB)+작업집합만 커버하면 충분. 48 MiB는 불필요.
- 참고: 이전 §3차 표의 tuned 읽기 수치는 정확하나, 측정 스크립트가 누적 라인 1개짜리 로그에서 차분 0을 보고하는 버그가 있었다(mixed 최초 파싱). 원본 `[kv-fsu]` 누적 카운터로 재확인한 값이 위 표다.

### 보강: 세션 드리프트 통제 재실측 (교차 A/B/A/B)

§5차의 혼재 속도(51.5/55.6 ms/tok)가 resident보다 빠르게 나온 것은 세션 간 열 상태 차이로 판명. 같은 세션에서 resident↔혼재를 교차 재실행한 결과:

| 반복 | resident | 혼재 32/8 |
|---|---:|---:|
| rep1 | 51.9 ms/tok | 53.1 ms/tok |
| rep2 | 56.4 ms/tok | 59.7 ms/tok |

→ 혼재는 resident 동급(−2~−6%, 노이즈 범위). 혼재의 이득은 속도가 아니라 KV 메모리(−36/−39%)이며, 세 구성(스트리밍·48/8·32/8)의 kv_payload_peak_bytes가 행별로 정확히 동일한 이유는 유지(R)가 별도 영역이 아니라 스트리밍 버퍼 슬롯을 고정하는 방식이기 때문(읽기만 19.7→1.62 MB/tok로 감소). 로그: `scen/g4_A_{res,mixed}_{x1,x2}.log`. 유지분 회계(retained 6.7 MB > 가변부 4.3 MB)의 물리 위치는 후속 확인 과제.
