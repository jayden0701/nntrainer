# KV FSU CPU 최적화 — 계획·이행·측정· 남은 일 (쉬운 요약)

이 문서는 `kv-cache-fsu-analysis.ko.md`(문제 분석), `kv-cache-fsu-htp-handoff.ko.md`(실험 계획), `kv-cache-fsu-cpu-device-log.ko.md`(자세한 실행 기록)를 한눈에 이해하기 위한 요약이다. 세부 수치의 원 출처는 마지막 문서다.

## 1. 한눈에

- **무엇을 했나**: KV 캐시를 플래시(기기 UFS)에 내리고 쓰는 기능(FSU)이 있어서 메모리는 약 200 MB 아꼈지만, 토큰 하나 만들 때마다 과거 KV를 통째로 다시 읽어 속도가 3배 느려져 있었다. **읽는 양을 줄이고(상주 예산), 쓰는 횟수를 줄이면(Write-back)** 속도를 원래 수준으로 되돌리면서 메모리 절감도 유지함을 기지에서 증명했다.
- **그 다음**: CPU sliding window도 전체 이력을 다 들고 있었다는 게 2번째 병목이었다. **sliding 레이어는 창 K행만 DRAM에, 나머지(global)는 flash 스트리밍**하는 혼재 저장소를 만들어 실제 gemma4 pack에서 동작을 증명했다. 측정 결과 RSS는 작지만 줄고, 출력은 3개 구성 모두 비트 일치.
- **지금 상태**: CPU 추론 단계는 3.1·3.2·3.4 완료 + sliding window 축소(1순위 최적화)까지 실그래프에 연결 완료. compaction×FSU 레이아웃 화해 설계 문서도 작성됨. 남은 것(HTP 단계, 실측 보강)은 장비 대기 / 별도 과제.
- **출력 안전성**: 총 60회 이상의 resident / FSU 비교(직접 I/O·버퍼드·write-through·sync·windowed cap 포함, tiny fixture + 1.7b + 실 gemma4 pack)에서 생성 토큰(`output_ids`)이 100% 일치.

## 2. 무엇이 문제였나 (쉬운 비유)

KV 캐시 = "지금까지 읽은 내용의 메모"다. FSU는 이 메모를 책상(DRAM) 대신 서류함(플래시)에 넣어 책상을 비우는 기능이다.

문제는 메모를 볼 때 **필요한 부분만 꺼내는 게 아니라 서류함 전체를 책상 위에 다시 펼쳐놨다**는 것. 638토큰짜리 대화에서 토큰 하나를 더 만들 때마다 약 65 MB를 다시 읽었다(기기 실측 64.5~67.7 MB/토큰). 그래서 토큰 생성 속도가 13~19 ms에서 42 ms로 처졌다.

또 새 메모를 적을 때마다 4 KiB씩 그날그날 서류함에 써 넣어(Write-through), 이 역시 부담이었다.

## 3. 계획과 이행 상태

계획은 `kv-cache-fsu-htp-handoff.ko.md`의 3~4단계다.

| 단계 | 계획한 것 | 상태 |
|---|---|---|
| 3.1 계측 | 읽기/쓰기/대기가 각각 얼마 드는지 재는 장치 | 완료 (환경변수 `NNTR_KV_TIMING`) |
| 3.2 읽기 억제 | DRAM에 얼마를 남겨둘지 예산을 정하고, 남은 건 다시 읽지 않게 | 완료 (환경변수 `NNTR_KV_RESIDENT_BYTES`) |
| 3.3 워커 큐 | 미리 읽기 요청을 여러 개 걸 수 있게 워커 구조 개선 | **미착수 (우선순위 낮음)** |
| 3.4 쓰기 억제 | 매 스텝 쓰지 않고, 쫓겨날 때 한 번에 쓰게 | 완료 (기본값, `NNTR_KV_WRITE_THROUGH`로 이전 방식 비교 가능) |
| (발생) 회계 수정 | 예산 계산이 처음으로 멈추는 버그 발견·수정 | 완료 |
| 리뷰 반영 | A/B 비교의 **sync 대칭** 확보, `compact()`×FSU 가드, dirty 장부 순서, 방어 코드 | 완료 (`850e0b21e`) |
| (신규) 화해 설계 | compaction×FSU 레이아웃 충돌을 푸는 설계 문서 (4대안 비교, D 선행 + B 계약 권장) | 완료 (`4f935d80d`) |
| (신규) 혼재 저장소 | windowed 레이어는 호스트 DRAM에 창만, 나머지는 flash 스트리밍 — sliding 축소의 기반 | 완료 (`fc2fec519`) |
| (신규) sliding 축소 연결 | 실그래프에 windowed cap 적용 + 리뷰 결함 3건(리셋·무제한 생성·alias base) 수정 | 완료 (`f277d8b3d`) |
| (신규) tied lm_head FATAL 수정 | 기존 버그 — tied pack이 그래프 구성부터 죽던 것. 속성 선언만 추가 | 완료 (`fa82a1799`) |
| (신규) 실 pack 측정 | gemma4 runtime pack에서 3-way 비트 일치 + RSS 절감 확인 | 완료 (`c6d2bf6e4`) |
| 4 HTP | DSP(HTP) 쪽으로 같은 소유권 설계 이식 | **장비(S26U) 필요** |

## 4. 만든 것

| 커밋 | 내용 |
|---|---|
| `47c776d88` | 단계별 계측 추가 (`NNTR_KV_TIMING`) |
| `fe0657074` | 상주 예산 `NNTR_KV_RESIDENT_BYTES` — 예산 안에서 owner(KV 평면 쌍)를 DRAM에 유지. 다음 스텝에 전체 이력 읽기를 건너뛴다 |
| `4cf90f660` | Write-back — `store()`는 dirty 범위(dirty range)만 표시하고 owner가 풀에서 나갈 때 합쳐서 쓴다. 스냅샷 복원·reset은 즉시 기록 유지 |
| `850e0b21e` | 리뷰 반영 — `NNTR_KV_BUFFERED`(buffered 강제)·`NNTR_KV_FLUSH_SYNC`(flush 시 sync)로 쓰기 모음과 sync 생략을 분리 측정, `flushDirty()`는 기록 성공 후에만 장부 삭제, flash 캐시에서 `compact()` 행 이동 거부, 풀 크기 언더플로 가드, 테스트 env를 RAII로 정리 |
| `6bc1f3d12` | 회계 버그 수정 — 예산을 채운 뒤 카운트를 깎는 순서 오류로 `retained_bytes`가 프리필 값에 고정(=디코드 중 예산 무력화)됐다. owner별 기여량을 따로 추적해 수정하고 유닛테스트를 정확값 검증으로 강화 |

이후 커밋:

| 커밋 | 내용 |
|---|---|
| `134a850d8` | 호스트 3-way 측정 — 쓰기 모음 기여와 sync 생략 기여를 교차 반복으로 분리 (아래 5.6) |
| `4f935d80d` | compaction×FSU 레이아웃 화해 설계 문서 (`docs/kv-fsu-compaction-reconciliation-design.ko.md`) — 4대안 비교, **혼재 구성(D) 선행 + 세대 기반 주소(B) 계약** 권장 |
| `fc2fec519` | **혼재 저장소** — windowed 레이어는 호스트 DRAM 텐서로 창만 상주, 나머지는 flash 스트리밍. 스냅샷 포맷 바이트 호환 유지, windowed-flash alias 거부 |
| `f277d8b3d` | **sliding window 축소를 실그래프에 연결** — HTP와 같은 행 규칙(min(max_seq, roundUp64(창+여유))), attention 행 오프셋을 plane-relative로. 리뷰 결함 3건(리셋 후 FATAL, 무제한 생성 중단, alias base stale) 동시 수정 |
| `fa82a1799` | tied lm_head FATAL 수정 — `TieWordEmbedding`/`LmHead`가 `plan_last_row_only` 속성을 선언(동작은 이미 그러했음). 실 pack이 열림 |

변경 범위: `Applications/CausalLM/kv_cache_manager.{h,cpp}`, `layers/mha_core.{h,cpp}`, `models/causal_lm.{h,cpp}`, `models/transformer.h`, `kv_ring.h`, `nntrainer/layers/llm/{tie_word_embedding,lm_head}.{h,cpp}` + 유닛테스트. 현재 유닛테스트 **46/46**.

## 5. 측정 결과

### 5.1 qwen3-0.6b (기기 SM-S947N, 프리필 542 + 디코드 96)

| 구성 | 디코드 p50 (ms/tok) | 피크 RSS (MB) | 읽기 (MB/tok) |
|---|---:|---:|---:|
| resident (플래시 안 씀) | 13.9~18.8 | 680 | 0 |
| FSU, 예산 없음 | 42.7 | 468 | 64.5 |
| FSU, 예산 242 MB(전량) | 29.1 (최선 16.5) | 551 | 0 |
| FSU, 예산 전량 + write-back | **15.0** | **558** | 0 |

- 짧은 컨텍스트에서는 RSS가 753 → 540 MB(−213 MB)로 줄었고 출력은 전 구성 일치.
- 최적 조합(전량 예산 + write-back)은 resident보다 **디코드 15.0 vs 18.8 ms로 오히려 빠르고 RSS는 130 MB 적다**. TTFT도 1620 → 1257 ms(−22%).

### 5.2 qwen3-1.7b (HF 원본에서 직접 변환, 가중치 1.05 GB)

기기(프리필 542 + 디코드 96):

| 구성 | 디코드 (ms/tok) | 피크 RSS (MB) | 읽기 | retained |
|---|---:|---:|---:|---:|
| resident | 26.33 | 1341 | – | – |
| FSU, 예산 없음 | 60.20 | 1119 | 6.50 GB (67.7 MB/tok) | 0 |
| FSU, 예산 74 MiB | 26.69 | 1169 | **0** | 73,170,944 |
| FSU, 예산 24 MiB | 33.77 | 1164 | 4.20 GB | 23,519,232 |
| FSU, 예산 74 MiB + write-through | 37.69 | 1193 | 0 | 73,170,944 |

- **읽기 증폭 계수는 모델 크기와 무관**하다(KV 폭이 0.6b와 같아 0.6b 64.5 ≈ 1.7b 67.7 MB/tok). 예산을 작업 집합 이상 주면 어느 모델에서든 읽기가 0이 되고 디코드가 resident와 동급(+1.4%)이 된다.
- 회계가 정확하다: `retained_bytes` = 638행 × 114,688 B = 73,170,944 B와 1바이트까지 일치(버그 수정의 실측 확인).

### 5.3 여러 번 대화하면? (요청 3개, 컨텍스트 638→1276→1914)

| 구성 | 디코드 요청별 (ms/tok) | 총 읽기 |
|---|---|---:|
| resident | 26.2 / 29.3 / 40.7 | – |
| FSU, 예산 74 MiB | 34.5 / 48.3 / **64.8** | 19.75 GB |
| FSU, 예산 250 MiB | 26.5 / 36.5 / **40.9** | **0** |

**중요한 교훈**: 예산은 "한 번에 들고 있을 최대 토큰 수 × 114,688 B"로 줘야 한다. 대화가 길어져 컨텍스트가 예산을 넘으면 그 구간이 다시 읽혀 속도가 처진다. `max_seq_len 2048`를 다 유지하려면 224 MiB가 필요하고, 그렇게 하면 RSS가 resident에 근접하므로 **메모리 절감과 속도는 같은 예산 다이얼의 양쪽**이다.

### 5.4 예산 다이얼 요약

| 예산 | 읽기 | 디코드 | DRAM |
|---|---|---|---|
| 0 (기본) | 최대(전체 이력 반복) | resident의 2~3배 | 최소 |
| 일부 | 비례 감소 | 그 사이 | 그 사이 |
| 작업 집합 이상 | 0 | resident와 동급 | resident에 근접 |


### 5.5 호스트 3-way — 쓰기 모음 vs sync 생략 vs 예산 상주 (qwen3 1.7b, 5회 교차)

| 구성 | 디코드 (ms/tok) | 읽기 (MB/tok) | 비고 |
|---|---:|---:|---|
| 예산 0 + write-through (기존 FSU) | 124.1 | 66.8 | 기존 동작의 재현 |
| 예산 24 MiB + write-through | 104.5 | 42.7 | 상주 효과만 |
| 예산 24 MiB + write-back + sync | 101.5 | 42.7 | **+쓰기 모음 −3.0 ms (2.9%)** |
| 예산 24 MiB + write-back | 103.0 | 42.7 | sync는 direct I/O에서 구조적 0 |

- 호스트(NVMe direct I/O)에서는 **예산 상주가 유일한 큰 축(−17.0%)**이고 쓰기 정책은 ±3 ms 수준. 기기(UFS 버퍼드)에서는 반대로 쓰기 정책 두 축이 −41 ms. **병목 축은 저장매체 특성에 따라 다르므로 측정 없이 일반화 금지**.
- 읽기 증폭은 플랫폼 무관 (호스트 66.75 vs 기기 67.67 MB/tok — 차이는 컨텍스트 길이 비례분).
- 20/20 실행 출력 일치, I/O 바이트 1바이트 오차 없이 동일.

### 5.6 실 pack — sliding window 축소 (gemma4 runtime pack, 호스트)

sliding 레이어의 KV 높이를 `min(max_seq, roundUp64(창+여유))`로 줄이면:

| 구성 | KV rows | 피크 RSS | 디코드 (ms/tok) |
|---|---:|---:|---:|
| A: 기존 (전체 높이) | 107,520 | 3,470 MB | 49.9 |
| B: 창 상한 상주 (NNTR_KV_WINDOW_CAP=1) | 64,512 | 3,452 MB (−17.6 MiB) | 50.0 |
| C: B + flash 스트리밍 (혼재) | 64,512 | 3,439 MB (−29.9 MiB) | 63.8 |

- **3개 구성의 출력이 11회 전부 비트 일치** — “창 밖 행을 버려도 sliding 출력은 같다”는 것을 실팩 규모에서 처음 확인. 물리 평면 회계(예상 18.0/33.9 MiB)와 절감량이 부합.
- C의 디코드 +27.8%는 예산 0 스트리밍의 알려진 비용(읽기 증폭 + lookahead 겹침 손실). **예산을 주면 사라지는 축** — §5.4의 다이얼이 sliding 축소에도 그대로 적용된다.
- 이 기능이 발동하는 프로파일: “짧은 프롬프트 + 긴 생성” (기본 설정이 full height인 pack에서는 효과 없음 — 설정 복제 필요).

### 5.7 무제한 생성·리셋·alias — 리뷰가 잡은 3건의 결함

실그래프 연결 과정에서 리뷰어가 잡아 수정한 것들(모두 회귀 테스트 추가):

1. **리셋 후 재요청 FATAL** — 첫 요청이 compaction으로 base를 옮긴 뒤 새 프롬프트가 windowed 상태를 안 지워 죽었다. → 리셋 시 원점 복원.
2. **무제한 생성이 중간에 FATAL** — `num_to_generate`가 0으로 정규화되면 창+여유 계산이 유한 평면을 만들어 생성 도중 넘침. → 전체 높이로 fall-back.
3. **alias된 sliding 레이어가 오래된 base를 읽음** — 소유자가 compaction으로 움직여도 sharer가 row 0을 절대 0으로 해석(조용한 오답). → 소유자 base로 resolve.

### 5.8 기기에서의 sync 분해 — “쓰기를 모아서”의 이득 중 얼마나 “sync를 안 해서”인가?

버퍼드 경로(직접 I/O 미지원 파일시스템)에서는 write-through가 store마다 `fdatasync`를 하는데 write-back은 sync를 하지 않는다. 그래서 두 이득을 분리해 재측정했다(qwen3 1.7b 기기, 상세는 기록 문서 9장; 호스트 direct 버전은 5.5).

| 구성 | write-through | write-back + sync | write-back (sync 없음) |
|---|---|---|---|
| direct I/O (이 기기 기본) | 37.7 ms/tok (sync 0) | flush 없음 → 아래와 동일 | 26.7 ms/tok (sync 0) |
| 버퍼드, 예산 24 MiB | 139.9 ms/tok (sync가 디코드의 43%) | **120.3 ms/tok (−14%)** | 98.9 ms/tok (−18%) |

- **direct I/O에서는 sync가 애초에 안 돈다** → 기존에 보고한 write-back 이득(+11.0 ms/tok)은 전부 "쓰기 회수 감소"다.
- **버퍼드에서는 페널티의 절반가량이 sync 생략 몫**이다. 버퍼드 기기에서 그 이득을 전부 "쓰기 모음"의 공으로 보면 안 된다.
- 단, 예산을 작업 집합 이상 준 구성은 요청 중 owner가 쫓겨나지 않아 **flush 자체가 0회**다 → 파일에 남는 건 프리필분뿐이고 내구성은 "요청 종료까지 DRAM"이다.

## 6. 남은 것

| 우선순위 | 할 일 | 막힌 이유 / 준비물 |
|---|---|---|
| 장비 대기 | 4단계 HTP 이식 (DSP와 ARM이 KV 소유권을 나눠 갖는 설계) + compaction 경계 행 | S26U 기기 필요. 설계 문서(`docs/kv-fsu-compaction-reconciliation-design.ko.md`)의 3~4단계(B 계약)와 준비 문서(`kv-cache-fsu-htp-handoff.ko.md`) 작성돼 있음 |
| 낮음 | 3.3 워커 큐 심화(선읽기 요청 다수 대기) | owner 단위 선읽기에서도 묶음 완료 대기가 얼마나 남는지 기기에서 측정 |
| 중간 | 긴 컨텍스트 스트레스(2048 전량, 4K급), 요청 여러 개의 eviction 압박, 세션 종료/복구 경로 | 기기·pack 준비됨. windowed cap 발동 프로파일은 “짧은 프롬프트 + 긴 생성” — pack 설정 복제 필요(§5.7) |
| 중간 | 다른 모델 확인 (qwen3 4B 이상, gemma4 26B CPU) | 1.7b 변환 파이프라인 재사용 가능. gemma4_cpu 등 구 pack은 Q4_0 시절 양자화본이라 **재양자화 필요**(별개 기존 문제, 기록 문서 11장) |
| 중간 | 균형 예산(~52 MiB) 구간 재측정, 병렬 리더 프로토타입 | §9의 검증 가능한 예측. 예산만으로 기기 재측정 가능, 병렬 리더는 워커 구조 변경 필요 |
| 정리 | 업스트림 PR 준비 | 커밋 21개(self-contained). `lib/libtokenizers_c.a`는 로컬 재생성물이라 **PR에서 제외**. docs 커밋 스쿼시 여부 판단, 테스트 수 변화(40→46)는 check_count·PR 본문에 반영 |
| 정리 | 세션 중 `HF_TOKEN` 값이 툴 로그에 노출됨 | 토큰 재발급 권장 |

## 7. 다시 실행하려면

```bash
# 기기: SM-S947N, 아티팩트 /data/local/tmp/nntr-cpu (pack: qwen3-17b-resident / qwen3-17b-fsu)
adb shell 'cd /data/local/tmp/nntr-cpu && LD_LIBRARY_PATH=$PWD \
  NNTR_KV_TIMING=1 NNTR_KV_RESIDENT_BYTES=77594624 \
  ./nntr_causallm $PWD/qwen3-17b-fsu --requests $PWD/requests_long.json'

# 호스트: ~/TEST_MODELS/qwen3-1.7b (x86 pack, README에 상세 커맨드·예산 산식)
NNTR_KV_TIMING=1 NNTR_KV_RESIDENT_BYTES=75497472 \
  <repo>/build-fsu/Applications/CausalLM/nntr_causallm \
  ~/TEST_MODELS/qwen3-1.7b-fsu --requests ~/TEST_MODELS/qwen3-1.7b/requests.json
```

- 예산 산식: `1행 = 114,688 B`(28레이어 × K+V × kv_dim 1024 × 2바이트) → `예산 = 보유 최대 토큰 × 114,688`
- 환경변수: `NNTR_KV_TIMING`(계측), `NNTR_KV_RESIDENT_BYTES`(상주 예산), `NNTR_KV_WRITE_THROUGH`(이전 쓰기 방식), `NNTR_KV_FLUSH_SYNC`(flush 시 sync), `NNTR_KV_BUFFERED`(버퍼드 강제)
- 1.7b pack: `~/TEST_MODELS/qwen3-1.7b`(x86·ARM 가중치 + 설정 + README), FSU 변형 `~/TEST_MODELS/qwen3-1.7b-fsu`. 변환 절차는 기록 문서 8장
- 유닛테스트: `ninja -C build-fsu Applications/CausalLM/unittest_kv_cache_manager && ./build-fsu/Applications/CausalLM/unittest_kv_cache_manager` (46/46)
- windowed cap: `NNTR_KV_WINDOW_CAP=1`(resident에도 같은 창 상한을 강제 — FSU 혼재와의 공정 비교용). 실 pack 측정 절차는 기록 문서 12장

## 8. 커밋 (기준 `13cbb99ea` 이후)

```
91ffbc385 [docs] 호스트 기준 재현
47c776d88 [CausalLM] 3.1 단계 계측
42ebf873f [docs] 0.6b 기기 기준·계측
fe0657074 [CausalLM] 3.2 상주 예산
d6abcc67f [docs] 예산 기기 결과
4cf90f660 [CausalLM] 3.4 write-back
21cd2c881 [docs] write-back 기기 A/B
6bc1f3d12 [CausalLM] 회계 버그 수정
2bf8a15ed [docs] 수정 후 재측정
2bd8c26a5 [docs] 1.7b 변환·매트릭스
850e0b21e [CausalLM] 리뷰 반영 (sync 분리·compact 가드·장부 순서)
060a0f24a [docs] 리뷰 라운드 — 버퍼드 sync 분해·compact 가드 기록
134a850d8 [docs] 호스트 3-way 측정 (쓰기 모음/sync/예산 분리)
4f935d80d [docs] compaction×FSU 화해 설계 문서
fc2fec519 [CausalLM] 혼재 저장소 (windowed DRAM + flash 스트리밍)
f277d8b3d [CausalLM] CPU sliding window 축소 실그래프 연결 + 결함 3건 수정
5393403e3 [docs] sliding 축소·결함 수정 기록
fa82a1799 [lmhead] tied lm_head FATAL 수정 (속성 선언)
10979d2b8 [docs] tied lm_head 수정 기록
c6d2bf6e4 [docs] 실 pack sliding 축소 측정
```

(code 커밋 7개 = 계측/예산/write-back/회계 수정/리뷰 반영/혼재 저장소/sliding 축소/lm_head, docs 커밋 11개)

## 9. 롤링 lookahead·병렬 리더·예산 자동화 검토 (설계 대화 기록)

9.1~9.3은 후속 스윕 전의 가설을 보존한 기록이다. 포화·균형점·자동 예산에 대한 해석은 9.4의 측정과 9.5의 계측 정의를 따른다.

### 9.1 검토한 아이디어 세 가지

1. **롤링 창**: 유지 예산만큼의 레이어를 회전 창으로 — 레이어 i를 통과할 때 i+창크기까지 미리 올리고, 창이 링을 한 바퀴 돌아 다음 스텝의 앞 레이어가 미리 채워진 상태로 시작
2. **균형 예산**: 읽기 시간 == 연산 시간이 되는 예산이 최적인가
3. **실측 기반 자동 예산**: read/연산 밸런싱 공식으로 예산을 자동 결정

### 9.2 결론

**(1) 롤링 prefetch는 읽기 총량을 줄이지 못한다.** 창 크기 w<28이면 매 스텝 창이 한 바퀴 돌아 **전 레이어(28회)를 매 스텝 로드**한다. 같은 8슬롯을 정적으로 쓰면(현행 greedy prefix) 8개 레이어는 영구 면제(스텝당 20회). prefetch는 "언제 읽느냐"만 바꾸고 "몇 번 읽느냐"(retention)는 안 바꾼다. 롤링이 슬라이드하지 않게 얼어붙이면 그게 곧 현행 정책이다.

**(2) 깊은 prefetch는 단일 순차 리더에서 효과가 0이다.** 읽기 워커는 스레드 1개·순차 pread라, 출발 시각이 아니라 "워커가 그 읽기를 잡는 시각"이 도착을 결정한다. 기존 로그로 포화 확인:

```
예산 0   : io_read 8.18 s vs decode 벽시계 5.78 s, ready_wait 1.94 s → 워커 포화
예산 24M : io_read 3.43 s vs decode 벽시계 3.24 s, ready_wait 0.28 s → 워커 포화
예산 74M : io_read 28 µs → 숨길 읽기 없음
```

**(3) 병렬의 상한: decode ≥ max(연산, 읽기).** 겹치기는 decode를 min이 아니라 max로 만들어줄 뿐이다. 기기 예산 0에서 읽기(≈85 ms/tok)는 연산(≈26 ms)의 3배라 아무리 겹쳐도 흡수 불가 — 이것이 예산 유지(총량 제거)가 유일한 구조적 해인 이유.

**(4) 병렬 리더는 "균형 구간"에서만 승부를 뒤집는다.** 출발시각이 실제로 의미 있으려면 여러 outstanding I/O(병렬 리더)가 필요하고, 이때 레이어당 읽기는 R/P(P=병렬도)로 줄어든다. 뒤집는 조건은 P ≥ R/C:

| 구성 | 레이어당 읽기 R | 연산 C | 필요 병렬도 P |
|---|---:|---:|---:|
| 예산 0 | 3.04 ms | 0.93 ms | 3.3배 (비현실적) |
| 예산 24 MiB | 1.27 ms | 0.93 ms | 1.4배 (UFS 큐잉으로 현실적) |

즉 **예산 24 MiB + 병렬 리더(2배)로 연산 바운드 복귀 가능성** — 74 MiB 없이 하한에 도달하는 메모리 절감 경로. 단 flush 쓰기와의 UFS 경합·전력·워커 구조 변경이 대가다.

**(5) 균형 예산은 "최적"이 아니라 "하한에 닿는 문턱"이다.** decode = max(연산, 읽기)이므로 읽기 ≤ 연산이면 하한 도달 — 읽기 0까지 갈 필요는 없다. 검산(기기, ctx 638): 읽기 ≤ 20.6 MB/tok ⇔ 유지 owner ≈ 20개 ⇔ **예산 ≈ 52 MiB**. 24 MiB(decode 33.8)와 74 MiB(26.7) 사이의 미측정 구간에 스위트스팟 예측이 선다. 다만 균형점은 ctx에 비례해 이동하고(ctx 1914 → k 26/28 필요), 균형점에서는 읽기·쓰기·연산이 다 바빠 간섭·분산이 커진다 → 실전 기본값은 전량:

```
예산 = min(메모리 상한, ctx_max × 114,688 B)
```

**(6) 예산 자동화는 가능하고 단순하다.** 지연·메모리 트레이드오프가 단조라 내부 최적점이 없기 때문. 3단계: ① 시작 시 권장 예산 로깅(geometry row_bytes는 이미 자동, direct I/O 프로브로 BW 측정 — 코드에 존재), ② `ready_wait_us` 폐쇄루프 감시(계측 이미 스텝별 기록), ③ 다중 요청 세션의 적응형 성장. 남는 공학 과제는 실행 중 풀 리맵.

### 9.3 후속 조치

- 기기 매트릭스에 균형점(~52 MiB) 구간 추가 → decode = max(연산, 읽기) 모델 검증
- 3.3 워커 큐 항목 갱신: "단일 순차 리더에서 깊이 무의미(포화 확인). 병렬 리더로 바꾸면 균형 구간(R/C ≈ 1.4)에서 유효 — 예산 24 MiB + 병렬 2배로 하한 도달 가능성"
- HTP: 연산이 더 빨라 R/C 비율이 커지므로 예산 0은 병렬로도 극복 불가 — committed 공유 메모리 유지(예산 정책)가 여전히 정답

### 9.4 기기 검증 결과 (예산 스윕, SM-S947N) — 균형점 예측은 기각

9.2(5)의 균형점 가설을 기기에서 검증했다. qwen3-1.7b, prefill 534 + decode 96(ctx 630), 신규 안드로이드 빌드(HEAD), 예산 스윕을 회전 교차 2~3회:

| 예산 | k/28 | decode ms/tok(중앙값) | read 1건당 시간 |
|---|---:|---:|---:|
| resident | – | 31.9 | – |
| 74 MiB (전량) | 28 | **33.5** | ~0 |
| 60 MiB | 23 | 37.7 | 3.58 ms |
| 52 MiB (예측 균형점) | 20 | 40.7 | 2.97 ms |
| 48 MiB | 19 | 41.5 | ~3.2 ms |
| 40 MiB | 16 | 41.8 | ~2.9 ms |
| 24 MiB | 9 | 36.2 | 1.4~2.0 ms |

**발견:**

1. **측정한 예산과 지연 사이의 관계는 비단조적이다.** 52 MiB에서 하한에 닿는다는 예측을 이 요청이 반증했다. FSU 중 가장 빠른 설정은 전량을 유지한 74 MiB였고, 가장 느린 설정은 40 MiB였다.
2. 같은 크기의 읽기에서 이전 집계가 보고한 건당 시간은 24 MiB에서 1.4~2.0 ms, 52~60 MiB에서 3.0~3.6 ms였다. 그 집계에는 병렬 K·V의 시간이 합산될 수 있다. UFS 전력 상태 전환은 원인 후보이며, 원인을 확인하려면 읽기 요청 간격과 실제 완료 시각을 같은 실행에서 재야 한다.
3. 24 MiB는 이 요청에서 40~60 MiB보다 적은 메모리로 더 빨랐다. 74 MiB도 resident 기준보다 1.6 ms/token 느렸다. 다른 문맥이나 기기에서 이 예산 순서가 유지되는지는 별도 측정 대상이다.
4. 중간 예산의 지연을 줄일 후보로 더 깊은 prefetch를 실험할 수 있다. 같은 KV 메모리 상한에서 현재 레이어가 실제로 기다린 시간과 토큰 지연을 함께 비교해야 한다.
5. b74가 오늘 resident 대비 +1.6~4 ms(이전 세션 +0.36)로 격차가 크다 — 기기 열 상태·신규 빌드(계획 메모리 +97 MB) 혼입 의심, 별도 추적. 참고로 오늘 빌드에서 resident·FSU(전량) 출력은 비트 일치(60회 비교 누적).

측정 당시 원본 경로는 `build-fsu/logs/dev17b-bal/`이다. 브랜치에는 요약표만 포함되어 있다.

### 9.5 기기별로 두 제약을 비교하는 방법

현재 CPU FSU의 `lookahead=N`은 앞으로 사용할 서로 다른 KV owner 중 파일에서 선읽어 둘 최대 개수다. DRAM의 sliding 창, 유지 예산으로 상주한 owner, 비활성 레이어는 이 개수를 소비하지 않는다. 같은 KV를 공유하는 레이어는 한 번 예약하고, 선읽어 둔 KV는 소비될 때까지 한 자리를 차지한다. 0이면 선읽기를 끈다. §5와 §9.4의 기기 수치는 레이어 번호로 거리를 세던 구현에서 얻었다. 로그의 `owner_lookahead=1`은 owner를 세고 묶음 완료를 기다리던 구현, `owner_lookahead=2`는 owner별로 완료를 기다리는 구현이다. 예산 도구는 서로 다른 스케줄러의 로그를 한 스윕에 합치면 오류를 낸다. 총예산과 동적 롤링의 비교 절차는 [기기 실험 계획](docs/kv-fsu-rolling-device-test.ko.md)에 있다.

§9.4의 표에서 SM-S947N의 630행 요청은 24 MiB에서 36.2 ms/token, 52 MiB에서 40.7 ms/token, 74 MiB에서 33.5 ms/token이었다. 이 순서는 측정한 기기·pack·요청에 한정된다. 표의 가장 느린 중앙값은 40 MiB의 41.8 ms/token이다. 전력 상태 전환과 읽기 지연 사이의 인과관계는 이 로그만으로 식별할 수 없다. §9.2의 단일 워커 포화 판단에 사용한 `io_read_us`에는 동시에 수행한 K·V 읽기 시간이 합산된다. direct I/O probe는 파일시스템 지원 여부를 검사한다.

같은 기기·빌드·pack·전원 조건에서 resident 기준과 여러 FSU 예산의 `--requests` 로그를 수집한다. 정책별로 디렉터리를 분리하고 모든 실행에 `NNTR_KV_TIMING=1`을 준다. resident 파일명은 `resident_r1.log`처럼 시작한다. 각 반복을 별도 파일로 남기고, 실행 순서를 교차해 열 상태와 스토리지 상태가 한 예산에만 몰리지 않게 한다. 후보를 고른 뒤에는 독립된 반복으로 제약을 다시 확인한다. 다음 값들은 명령 형식 예시다.

```sh
python3 tools/kv_cache/kv_budget_predict.py calibrate --logs LOGDIR --rows 630 --device DEVICE --model PACK --out profile.json
python3 tools/kv_cache/kv_budget_predict.py solve --profile profile.json --rows 630 --memory kv-peak --cap 80MiB --metric p95
python3 tools/kv_cache/kv_budget_predict.py solve --profile profile.json --rows 630 --memory kv-peak --metric p95 --slowdown-pct 10
```

- 첫 solve는 지정한 메모리 이내에서 가장 빠른 측정 설정을 고른다. 둘째는 같은 기기의 resident p95 대비 허용 지연을 만족하는 가장 작은 측정 메모리를 고른다. `--target`은 절대 ms/token이며 상한과 함께 쓸 수 있다. 동률이면 더 적은 메모리나 더 낮은 지연을 차례로 고른다.
- `retention`은 설정한 `NNTR_KV_RESIDENT_BYTES`, `kv-peak`는 고유 sliding 평면 + flash 풀의 고수위 + 쓰기 staging의 합이다. 후자는 해당 실행에서 관측한 KV payload의 보수적인 상한이다. 프로세스 RSS, 커널 페이지 캐시, HTP 메모리와 별도로 해석한다.
- p50은 반복별 steady 중앙값의 중앙값이다. p95는 매 요청에서 첫 decode를 제외한 토큰별 시간의 nearest-rank p95를 구한 뒤 반복 중 최댓값을 사용한다. 짧은 요청은 표본이 적으므로 긴 decode와 함께 확인한다. 메모리도 반복 중 최댓값을 사용한다.
- 예전 로그에 설정 예산이 없으면 `--budget-map budgets.json`으로 설정값을 제공한다. 예를 들어 `{"b24":"24MiB"}`는 실제로 그 값을 사용한 경우에만 쓴다. 예전 p50 로그로 p95를 복원할 수 없다.
- 위 표의 내부 예산 4개를 하나씩 제외하고 인접 예산으로 보간했을 때 RMSE는 1.10 ms/token이었다. 가장 큰 오차는 40 MiB로, 예측 39.73 ms와 관측 41.8 ms는 40 ms 제약에 대해 서로 다른 결론을 준다. 같은 요청에서도 SLA 후보는 직접 측정할 이유가 있다.
- 현재 런타임은 기본적으로 예산이 0에서 양수로 바뀔 때 쓰기 정책도 write-through에서 write-back으로 바꾼다. 그대로 스윕하면 두 변경을 합친 설정의 성능을 측정한다. 예산 자체의 효과를 분리하는 실험은 모든 예산에 `NNTR_KV_WRITE_THROUGH=1`을 적용해 별도로 비교한다.
- 추천과 보간을 구분한다. 측정 프로파일의 `solve`는 관측한 설정만 비교하고 `predict`는 같은 문맥의 인접 측정점 사이를 보간할 수 있다. 새 기기나 다른 문맥에서 정책을 고를 때는 그 조건에서 다시 수집한다. `solve --example`은 위 SM-S947N 표를 재생한다. 일반 사용에서는 `--profile`로 대상 프로파일을 명시한다. 출력한 예산의 적용 여부는 호출자가 결정한다.
