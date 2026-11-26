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
