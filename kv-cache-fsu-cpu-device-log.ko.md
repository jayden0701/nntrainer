# KV FSU CPU 구현·기기 테스트 기록

`kv-cache-fsu-analysis.ko.md`의 분석과 `kv-cache-fsu-htp-handoff.ko.md`의 순서를 따라 실제 작업을 진행하면서 남기는 기록이다. 기준 커밋은 `13cbb99ea161f636dbefd610c1bd435de641e89a`이며 이 기록 자체도 같은 트리의 `kv-cache-fsu` 브랜치에 커밋으로 남긴다. S26U가 준비되기 전이므로 문서의 3단계(CPU 계측·튜닝)까지를 먼저 진행하고, 기기 실험도 S26U가 아닌 다른 Android 기기로 CPU 부분을 수행한다.

## 0. 진행 상황 요약

| 단계 | 내용 | 상태 |
|---|---|---|
| 기준 재현 | 호스트에서 3단계 문서의 기준 검사(unittest + tiny fixture CLI A/B)를 재현 | 완료 |
| 3.1 계측 | `KVFlashStorage`에 단계별 대기·전송 시간 계측 추가 | 완료 |
| 기기 기준값 | 연결된 Android 기기에서 qwen3 0.6b resident/FSU A/B | 완료 |
| 기기 계측 | 3.1 계측을 켠 기기 실행으로 읽기·대기 비용 확보 | 완료 |
| 3.2 예산 | owner 상주 예산(환경변수) 구현·기기 검증 | 완료 |
| 3.4 쓰기 | 디코드 쓰기를 owner 퇴피까지 지연(write-back)·기기 검증 | 완료 |
| 회계 수정 | `retained_bytes`가 프리필값에 고정한 버그 수정·재측정 | 완료 |
| 규모 확장 | qwen3 1.7b를 HF 원본에서 변환해 같은 매트릭스 재검증 | 완료 |
| 리뷰 반영 | sync 대칭 스위치, `compact()` 가드, dirty 장부 순서, 방어 코드·테스트 정리 | 완료 |

## 1. 작업 환경

| 항목 | 값 |
|---|---|
| 호스트 | Ubuntu 22.04 x86-64, GCC 11.4, 시스템 페이지 4096 B, 32 스레드 |
| 기준 트리 | `kv-cache-fsu` @ `13cbb99ea`, 빌드 디렉터리 `build-fsu` |
| 빌드 설정 | `-Denable-transformer=true -Denable-fp16=true -Denable-test=true -Dwerror=false` |
| 기기 | SM-S947N (serial `R3KYC06DR6R`), Android 16, erd9965, arm64-v8a, 페이지 4096 B, /data 여유 136 GB |
| 기기용 모델 | `~/TEST_MODELS/qwen3-0.6b` (Q4_0/FP32, embdq6k, 375 MB). gemma4 26B CPU 기기 실행은 현 기기 예산을 넘으므로 제외 |
| 비고 | qwen3 1.7b nntr pack은 원래 없어 0.6b로 추세를 먼저 확정했고, 이후 HF 원본(`~/turboquant-exp/models/Qwen3-1.7B`)에서 직접 변환해 8장에서 재검증했다. 검증된 pack(x86·ARM 가중치, 호스트/기기 설정, 사용법 README)은 `~/TEST_MODELS/qwen3-1.7b`와 `~/TEST_MODELS/qwen3-1.7b-fsu`에, 중간 산출물은 `~/NNTrainer_models/qwen3-1.7b*`에 둔다 — 둘 다 커밋하지 않는다 |

## 2. 호스트 기준 재현 (커밋 `13cbb99ea` 그대로)

문서의 절차를 그대로 실행했다. fixture pack의 심볼릭 링크를 만들 때는 절대 경로를 써야 한다(상대 경로로 만들면 `nntr_causallm`가 `config.json`을 찾지 못한다).

| 검사 | 결과 |
|---|---|
| `unittest_kv_cache_manager` | 7 suites, 37/37 통과 (async write env unset / `=1` 모두) |
| tiny fixture CLI A/B | qwen3_tiny, gemma4_tiny × resident/0/1/3, 요청 2개씩 총 16개 전부 기준 `output_ids`·위치·pending 일치 |
| 기준 출력 | gemma4 첫 요청 `[26,16,30,26]`/cached_after 7, 둘째 `[6,14,3,4]`/6. qwen3는 `[3,3,3,3]`/7, `[5,5,5,5]`/6 — 분석 문서 기록과 동일 |
| `[kv-fsu]` I/O | gemma4 lookahead 1: 요청당 read 49,152/98,304 B, write 65,536/131,072 B, `direct_io=1`, `peak_kv_bytes=16384`. qwen3 lookahead 1: read 24,576/49,152 B, write 32,768/65,536 B, `peak_kv_bytes=8192`. async env unset이라 `async_writes=0`, `staging_bytes=0` |

분석 문서 4장의 재현 결과를 그대로 확인했으므로, 이 트리에서 기능이 의도대로 동작한다는 전제는 유지된다. 다음 단계는 문서 3.1의 계측 추가다.


## 3. 3.1 계측 — 단계별 I/O 시간 (커밋 대상)

`KVFlashStorage`에 환경변수 `NNTR_KV_TIMING`이 있을 때만 켜지는 단계별 시간 측정을 추가했다. 꺼져 있으면 기존 wait·copy·sync 경로 코드는 그대로다.

| 항목 | 잰 곧 | 의미 |
|---|---|---|
| `ready_wait_us` | `wait()`의 `future.get()` | begin/finish에서 선읽기가 끝나길 기다린 시간 |
| `write_wait_us` | `waitWrite()`의 `future.get()` | 다음 store나 step 종료가 앞선 비동기 쓰기를 기다린 시간 |
| `copy_us` | store의 staging `memcpy` 두 평면 | 비동기 쓰기를 위해 겹쳐 복사한 시간 |
| `fdatasync_us` | store의 buffered `fdatasync` | buffered 경로의 내구성 동기화 시간 |
| `io_read_us` / `io_write_us` | `transferRaw()`의 pread/pwrite 루프 | 실제 시스템 콜에 든 시간 (워커 스레드 포함) |
| `drain_us` | `finishStep()`의 wait+waitWrite | 요청 종료 시 남은 I/O를 비우는 시간 |

`[kv-fsu-timing]` 줄은 요청 단위 delta(이전 report 이후)와 누적 total을 같이 출력한다. write·value 워커 스레드에서도 더해지므로 카운터는 atomic(relaxed)이다. 쓰레드당 `job_` 한 칸 구조라 상호 배제는 이미 serialize돼 있어, 카운터만으로 합산이 맞다.

호스트 스모크(gemma4_tiny, lookahead 1, async write off): 계측을 켠 실행도 기준 출력과 동일했고, 첫 요청에서 `ready_wait_us=1687.8 io_read_us=2820.5 io_write_us=2334.2`로 요청 1.17 ms decode 중 상당 부분이 실제 I/O와 선읽기 대기임을 확인했다. 유닛테스트 37/37은 여전히 통과한다(계측 OFF 기본 경로).

## 4. 기기 기준값과 3.1 계측 결과 — qwen3 0.6b, CPU

### 배포

Android CPU 빌드(`build_android.sh -Denable-fp16=true`, NDK r28c)의 산출물을 `/data/local/tmp/nntr-cpu`에 배치했다. 모델 bin과 tokenizer는 기기의 기존 배포(`/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b`)를 symlink로 재사용했다. FSU 임시 파일 디렉터리는 `/data/local/tmp/nntr-fsu-work`다.

| 산출물 | sha256(앞 16) |
|---|---|
| `nntr_causallm` | `d1406fe0ebd7f569` |
| `libcausallm.so` | `f291394ce65d10bb` |
| `libnntrainer.so` | `d1eadc1ae408f371` |
| 모델 bin (`qwen3_0.6b_q40_embdq6k_arm.bin`) | `4e5dd65f4d378249` |
| pack `nntr_config.json` | `f72191326b2ba183` |

pack 구성: resident, fsu0/1/3 (`kv_flash_fsu.directory=/data/local/tmp/nntr-fsu-work`), `init_seq_len=1024`, `max_seq_len=2048`, `num_to_generate` 64 또는 96, batch 1, `NNTR_ENGINE=cpu`, 스레드 4. 기기 페이지 4096 B, `/data`는 UFS.

### 짧은 컨텍스트 정확성 A/B (prefill 10, decode 64, cached_after 74)

resident와 fsu0/1/3의 `output_ids`가 전부 일치했다. `[kv-fsu]`: `direct_io=1` — 기기 UFS에서 direct 경로가 자연 선택됐다(호스트에서 예상한 buffered fallback은 아님). 요청당 read 308,281,344 B, write 15,826,944 B, `peak_kv_bytes`는 lookahead 0/1/3에서 각각 303,104/606,208/1,212,416 B. peak RSS는 resident 753 MB에서 FSU 540 MB로 약 213 MB 줄었다(resident KV 배분 약 235 MB와 부합).

### 긴 컨텍스트 계측 (prefill 542, decode 96, cached_after 638, `NNTR_KV_TIMING=1`, 3회씩 교차 실행)

| 모드 | decode p50 ms/token | TTFT p50 ms | peak RSS p50 MB | read MB/token | ready wait p50 | io(r+w) 합계 p50 |
|---|---:|---:|---:|---:|---:|---:|
| resident | 13.91 | 1,329 | 680 | 0 | — | — |
| fsu0 (동기) | 424.18 | 1,562 | 468 | 64.53 | 0 µs | 22.4 s |
| fsu1 | 42.67 | 1,846 | 468 | 64.53 | 1.47 s | 12.0 s |
| fsu3 | 40.75 | 1,610 | 471 | 64.53 | 1.59 s | 7.2 s |

요청당 총 read 6,495,928,320 B(6.05 GiB, 토큰당 평균 64.53 MB), write 84,180,992 B(80.3 MB). fsu1/3은 `parallel_reads=2688`로 매 토큰 V/K 평면 읽기가 겹쳤고, 요청당 요청 총 io 시간은 lookahead 3에서 7.2 s까지 내려갔다(같은 바이트를 더 겹쳐 읽음). `copy_us`, `fdatasync_us`, `drain_us`는 direct I/O·비동기 쓰기 OFF 조합에서 0에 수렴했다. 세 FSU 모드 모두 resident와 `output_ids`가 3회 전부 일치했다.

### 해석

1. **메모리 절감은 목표대로다.** 638 토큰 기준 peak RSS 680→468 MB, 약 212 MB를 아꼈다. FSU 풀(`peak_kv_bytes` 최대 2.6 MB)과 staging이 이득을 대부분 삼키지 않는다.
2. **읽기 증폭이 지배 비용이다.** 매 miss마다 평면의 `saved`(지금까지 저장한 전체)를 다시 읽는 현재 정책 때문에 토큰당 64.53 MB를 읽는다. 분석 문서 2장이 예측한 `2 × A(T × r)` 읽기가 기기에서 그대로 확인됐다. 이것이 문서 3.2(owner별 byte 예산)의 출발점이다.
3. **lookahead는 대기를 겹치는 데만 유효하다.** 바이트는 그대로인데 대기·전송 시간이 22.4→12.0→7.2 s로 줄었다. 그러나 decode 42 ms 중 순수 I/O 겹침이 대부분이므로, 읽는 범위를 줄이는 3.2를 먼저 해야 한다. fsu0의 424 ms/token은 동기 전체 이력 읽기가 그대로 노출된 결과다.
4. **direct I/O가 기본 경로다.** 기기 UFS에서 `direct_io=1`이므로 buffered fallback(staging 복사·fdatasync)은 이 실험 대상이 아니다. 실패 주입 실험은 별도로 한다.

### 다음 단계

- 문서 3.2: owner별 상한(예: 활성 tail 몇 행 + 마지막 N 페이지)으로 읽는 범위를 줄이고 bytes/token과 decode ms가 함께 내려가는지 같은 매트릭스로 재측정한다.
- 문서 3.3: 워커 `job_` 1칸 구조를 슬롯별 완료 상태로 바꿔 첫 레이어 선읽기와 owner 중복을 정리한다.
- 1.7b는 `~/TEST_MODELS`에 nntr pack이 없어 미수행. 읽기 증폭 정책이 0.6b에서 검증되면 변환 후 같은 매트릭스를 돌린다.

## 5. 3.2 구현과 기기 재측정 — owner 상주 예산

### 구현

환경변수 `NNTR_KV_RESIDENT_BYTES`(바이트)를 주면 `finishFlashLayer`의 evict가 예산 안에서 owner의 두 평면을 풀에 **유지**한다. 유지된 owner는 다음 단계에서 전체 이력 읽기를 건너뛰고, binding은 기존 계약대로 매 finish마다 원래 매핑 주소로 돌아간다(그래프가 만든 상태 유지). 예산이 부족하면 기존처럼 방출하고 다음 단계에서 파일에서 다시 읽는다. `reset`(새 프롬프트)은 유지된 owner를 전부 풀에서 내리고 페이지를 0으로 되돌린다. 풀은 필요한 슬롯 수(owner + 현재 + lookahead)로 생성 시 늘어나고, `[kv-fsu]` 줄에 `retained_bytes`가 추가됐다. 유닛테스트 1개(`KVCacheManagerFlash.resident_budget_keeps_owners_across_steps`)가 38/38에 포함됐다.

주의: 예산 판정은 store 이후의 `buffer->resident` 기준이라 한 스텝 늦게 적용될 수 있고, 레이어 인덱스 순서로 채운다(결정적 prefix 정책).

### 기기 재측정 (qwen3 0.6b, prefill 542 / decode 96, 교차 5회)

| 모드 | decode p50 ms | decode 개별 | peak RSS p50 MB | read MB/token |
|---|---:|---|---:|---:|
| resident | 18.84 | 13.9–19.2 | 681 | 0 |
| fsu1 예산 없음 | 41.73 | 35.0–86.8 | 468 | 64.53 |
| fsu1 예산 242 MB(전부 유지) | 29.05 | 16.5–29.5 | 551 | 0.00 |

15회 전부 resident와 `output_ids`가 일치했다. 예산 매트릭스(fsus × 예산, 3회씩)에서는 예산 64 MB 이상이면 retained가 총 KV(~59 MB)를 전부 담아 `read_bytes=0`이 됐고, 8 MB에서는 retained 6.35 MB·read 57.62 MB/token으로 예산 없음과 같았다. 단, 이 수치의 retained는 뒤에 드러난 회계 버그(7장)로 프리필 기준 미관측 값이었고, 읽기량과 성능 결론은 수정 후 재측정에서도 그대로 유효하다.

### 해석

1. **읽기 증폭 제거.** 예산이 작업 집합(이 모델·이 컨텍스트에서 ~59–70 MB)을 담으면 토큰당 읽기가 64.53 MB → 0이 됐다. 분석 문서 2장의 1순위 후보("읽는 데이터 줄이기")가 CPU에서 동작함이 기기에서 확인됐다.
2. **메모리–지연 교환.** 전부 유지 시 DRAM은 예산 없음 대비 +83 MB(풀)이고 resident 대비 −130 MB, decode는 p50 41.7→29.1 ms(−30%), 최선 16.5 ms로 resident(18.8 ms)에 근접한다. 예산을 줄이면 그만큼 읽기가 돌아온다(8 MB → 예산 없음과 동일).
3. **남은 격차.** 전부 유지해도 resident보다 ~10 ms/token 느리다. 이제 읽기가 아니라 **쓰기 경로**(decode 행 단위 동기 `pwrite`)와 관리 비용이 지배하므로 다음 후보는 문서 3.4의 쓰기 모음이다.
4. **분산 유의.** 기기 실행은 냉·온 상태에 따라 이중 양상(≈16 ms vs ≈29–48 ms)을 보였다. 교차 실행으로 완화했고, 정밀한 p95는 별도 워밍업·고정 주파수 실행이 필요하다.

### 다음 단계

- 문서 3.4: decode 행 쓰기를 페이지 경계까지 모아 쓰거나 비동기화해 남은 ~10 ms를 줄인다.
- 문서 3.3: 워커 `job_` 1칸 구조 개선(첫 레이어 선읽기와 예산 부족 시의 동기 처리 겹침).
- qwen3 1.7b는 nntr pack 변환이 필요해 미수행. 0.6b에서 확정한 정책(예산 ≥ 작업 집합이면 읽기 0)을 긴 컨텍스트·더 큰 모델에서 반복 검증한다.

## 6. 3.4 구현과 기기 재측정 — 쓰기 모음(write-back)

### 구현 (커밋 `4cf90f660`)

예산 모드에서 풀 버퍼가 이미 작업 복사본이므로 `store()`가 더 이상 매번 `pwrite`하지 않고 dirty 범위만 표시하고, owner가 풀에서 쫓겨날 때(evict 방출 경로) 합쳐서 쓴다. 스냅샷 복원(`replace=true`)은 여전히 즉시 기록해 오류를 반환 전에 보고하고, `reset`은 새 프롬프트 데이터이므로 미기록 범위를 함께 버린다. `NNTR_KV_WRITE_THROUGH`로 구 경로(매 store 기록)를 A/B할 수 있다. `store`의 논리적 `write_bytes` 계산은 그대로 두고 실제 기록만 늦춘다. 유닛테스트 1개(`resident_budget_flushes_dirty_on_eviction`, 예산을 owner 1개 크기로 제한해 쫓겨나는 레이어의 flush 후 재읽기를 검증)를 추가해 39/39다.

### 기기 측정 — qwen3 0.6b, fsu1, 예산 242 MB(전부 유지)

같은 빌드에서 교차 5회씩(write-through vs write-back), prefill 542 / decode 96, requests_long:

| 모드 | decode p50 (ms) | decode min–max | TTFT p50 (ms) | 요청당 flash write IO | peak RSS |
|---|---:|---:|---:|---:|---:|
| write-through (기존) | 16.98 | 16.6–74.0 | 1620 | io_write 414,913 us | 561 MB |
| write-back | **15.01** | 14.8–15.4 | **1257** | **0 us** | 558 MB |

10회 전부 `output_ids` 일치(단일 결과), cached_after 638, `read_bytes=0` 유지. `io_write_us` 합계 0은 이 요청에서 owner가 전혀 쫓겨나지 않아 flush가 한 번도 없었음을 뜻한다(논리적 write_bytes 84,180,992는 그대로 기록).

### 해석

1. **쓰기 제거 효과.** decode p50 17.0→15.0 ms(−12%), 분산 상한이 74 ms → 15.4 ms로 사라졌다. 토큰당 56회 4 KiB `pwrite`가 우발 부하에서 빠졌다.
2. **TTFT 개발.** prefill의 큰 청크 쓰기도 지연 기록되어 TTFT p50 1620→1257 ms(−360 ms, −22%).
3. **트레이드오프.** 전부 유지 + write-back에서는 요청 내 파일 기록이 전혀 없다 — 데이터는 풀 버퍼(DRAM)에만 있고, owner가 예산 밖으로 쫓겨날 때만 파일에 도달한다. 프로세스 비정상 종료 시 미기록 데이터가 사라진다(재시작 정책은 기존과 동일: 실패 세션을 잇지 않고 새 프롬프트로 재시작). 예산을 줄이면 쫓겨나는 시점마다 flush되므로 기존 의미가 보존된다.
4. **잔여 격차.** resident 대비 decode ~15 vs ~18.8 ms 수준에서 역전(이 컨텍스트에서는 FSU 쪽이 오히려 빠르다 — 쓰기·읽기 IO가 0이면 resident의 KV 전역 접근과 동일해지기 때문). 남는 차이는 관리 비용과 풀 슬롯 복사이다.

### 결론

- 예산 ≥ 작업 집합 + write-back 조합이 qwen3 0.6b CPU에서 resident보다 RSS −130 MB, decode p50 ~15 ms로 최적이며 출력은 전부 일치한다.
- 문서 3단계(CPU)의 계측·읽기 억제·쓰기 억제가 모두 기기에서 검증됐다. 남은 문서 작업은 3.3(워커 `job_` 1칸 구조 개선)과 4단계(HTP, S26U 확보 후)다.
- qwen3 1.7b는 nntr pack 변환이 필요해 미수행(HF 원본만 존재).
## 7. 회계 버그 추적과 수정 — retained_bytes가 첫 스텝에 고정된 문제

### 발견

3.4 검증 중 `[kv-fsu]`의 `retained_bytes`가 매번 프리필 스텝 값에 고정되는 현상을 발견했다(decode 96에서 62,160,896 = 542행 상당, decode 512에서 1,146,880 = 10행 상당). 원인은 `evict()`의 유지 경로다: 같은 스텝의 `store()` 안에 있는 `account()`가 `buffer->resident`를 이미 늘려 놓는데, 그 현재값을 그대로 뺐다가 다시 같은 값을 더해줘서 순증가가 항상 0이 됐다. 그래서 예산 판정(`retained_bytes_ + 2×resident ≤ 예산`)이 프리필 시점 기준으로 고정되고, 예산이 부족한 구간에서는 디코드 중 유지 갱신이 제때 반영되지 못할 수 있다.

### 수정 (커밋 `6bc1f3d12`)

owner가 유지될 때 기여한 값을 `Layer::retained_resident`로 추적해 그 값만 빼게 했고 `reset()`에서 함께 초기화한다. 유닛테스트 2개의 `retained_bytes` 검증을 정확값 비교로 강화했다(링을 도는 owner와 선형으로 진행하는 owner의 페이지 정렬 크기 차이까지 포함) → 39/39 통과.

### 기기 사후 재측정 — qwen3 0.6b (8 MiB는 교차 3회, 전체 예산은 1회씩)

| 구성 | decode p50 | read MB/token | retained_bytes | 요청당 io_read / io_write |
|---|---:|---:|---:|---|
| 예산 8 MiB, write-through | 31.1–39.5 ms | 60.42 | 7.84 MB | 4.85–6.04 s / 0.51–1.40 s |
| 예산 8 MiB, write-back | 30.4–39.1 ms | 60.42 | 7.84 MB | 4.23–6.04 s / 0.69–1.70 s |
| 예산 242 MB, write-through | 16.8 ms | 0 | 73.17 MB | 0 / 0.36 s |
| 예산 242 MB, write-back | 14.2 ms | 0 | 73.17 MB | 0 / 0 |

1. **회계가 실측과 일치했다.** 8 MiB에서는 정확히 owner 3개(3 × 2 평면 × 1,306,624 B = 7,839,744 B)가 디코드 내내 유지되고, 242 MB에서는 28 owner 전체(28 × 2 × 638행 × 2048 B = 73,170,944 B)가 유지된 값이 보고된다. 수정 전에는 프리필 기준값(6.35–6.7 MB / 62.2 MB)에 고정된 채 보고됐다.
2. **성능 결론은 그대로 유효.** 전체 예산 구간은 수정 전과 동일하게 나왔다(decode 14.2–16.8 ms, 읽기 0). 8 MiB 구간은 읽기 지배(io_read 4.2–6.0 s)라 write-back 이득이 없고, 읽기량은 유지되지 않는 owner 수가 결정한다(재읽기 25 owner ≈ 60.4 MB/token — 수정 전후 동일. 이 구성에서는 고정된 회계도 결과가 같았고, 차이는 보고값과 향후 예산 변동 동작에서 난다).
3. **출력 정상.** 8회 전부 `output_ids` 일치(고가치 1개).

### 산출물 노트

체크인된 `lib/libtokenizers_c.a`(구 툴체인 산출물)로는 호스트 링크가 실패해(`std::ios_base_library_init()` 미정의) `tools/build_tokenizers_c.sh --update-prebuilt`로 재생성했다. 로컬 산출물이므로 커밋에 포함하지 않고 dirty로 남긴다.

## 8. qwen3 1.7b 검증 — 모델 규모 확장

0.6b로 확정한 결론(읽기 증폭, 상주 예산으로 제거, write-back 이득)이 모델 규모가 커져도 유지되는지 확인한다. 1.7b는 hidden 2048 / 28레이어 / kv_head 8 × head_dim 128로 **KV 폭이 0.6b와 같다**(kv_dim 1024 → KV 114,688 B/token). 절대 KV 크기는 같은데 가중치가 375 MB → 1.0 GB로 늘어나 FSU의 상대 이점과 오버헤드 비중이 어떻게 바뀌는지가 관측 목표다.

### 변환 파이프라인 (재현 절차)

| 단계 | 내용 | 산출 |
|---|---|---|
| HF 샤드받기 | `HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-1.7B model-00001-of-00002.safetensors model-00002-of-00002.safetensors` (+ `tokenizer.json`). xet 백엔드는 사내 프록시에서 `.incomplete` 0 B로 멈추므로 껐고, 파일명을 positional로 주면 `--include`는 무시된다 | bf16 2샤드 4.06 GB |
| 샤드 병합 | `weight_converter_layergraph.py`는 단일 `model.safetensors`만 읽는다. `model.safetensors.index.json` 순서로 페이로드를 이어붙이고 병합 헤더를 앞에 붙여 한 파일로 작성 | 311 텐서 / 4,063,479,808 B |
| FP32 bin | `python3 Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py --model_path <병합 dir> --output_name nntr_qwen3_1.7b_fp32.bin --data_type float32` (tied라 lm_head 미기록) | 6.88 GB |
| 양자화 | `nntr_quantize <dir> --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q6_K --isa ARM`(기기용) / `--isa X86`(호스트용). `--isa`를 주면 한 호스트에서 아치별로 재포장된다(README의 Q4_0 크로스플랫폼 금지는 `DEFAULT` 기준 설명). 임베딩은 bin 안에 유지되고 sidecar는 안 나온다 | 1.05 GB × 2 |

함정 두 개를 확인했다.

1. `nntr_config.json`의 `tokenizer_file`은 CWD 기준으로 해석되므로 절대 경로를 써야 한다.
2. pack의 `config.json`은 **반드시 그 모델 것**이어야 한다. 0.6bpack처럼 shared `config.json` 심링크를 재사용하면(hidden 1024) 1.7b는 차원 불일치로 조용히 깨진다 — 에러 없이 로드되고 출력만 id 1–19 잡음 96개가 나온다. pack마다 자기 `config.json`/`generation_config.json`을 복사해 해결했다.

### 호스트(x86) 결과 — prefill 512 + decode 32 (ctx 544)

| 구성 | TTFT (s) | decode (ms/tok) | 피크 RSS (MB) | read (MB) | retained (B) |
|---|---:|---:|---:|---:|---:|
| resident | 2.79 | 32.06 | 1225 | – | – |
| FSU, 예산 0 | 3.06 | 83.15 | 1184 | 1938 | 0 |
| FSU, 예산 64 MiB | 2.81 | 32.04 | 1167 | 0 | 62,390,272 |
| FSU, 예산 8 MiB | 2.95 | 74.99 | 1122 | 1728 | 6,684,672 |
| FSU, 예산 64 MiB + write-through | 3.00 | 33.51 | 1171 | 0 | 62,390,272 |

전체 예산(64 MiB ≥ 544행 × 114,688 B = 62,390,272 B)에서 resident와 동급(−0.1%)이고 읽기가 0이며, 회계값은 working set과 1바이트까지 일치한다.

### 기기 결과 — prefill 542 + decode 96 (ctx 638)

| 구성 | TTFT (s) | decode (ms/tok) | 피크 RSS (MB) | read (MB) | read MB/token | retained (B) | 요청당 io_read / io_write |
|---|---:|---:|---:|---:|---:|---:|---|
| resident | 2.58 | 26.33 | 1341 | – | – | – | – |
| FSU, 예산 0 | 3.03 | 60.20 | 1119 | 6496 | 67.67 | 0 | 8.18 s / 0.53 s |
| FSU, 예산 74 MiB | 2.65 | 26.69 | 1169 | 0 | 0 | 73,170,944 | 28 µs / 0 |
| FSU, 예산 24 MiB | 3.07 | 33.77 | 1164 | 4197 | 43.72 | 23,519,232 | 3.43 s / 0.40 s |
| FSU, 예산 74 MiB + write-through | 3.68 | 37.69 | 1193 | 0 | 0 | 73,170,944 | 0 / 0.55 s |

`retained_bytes` 73,170,944 B는 638행 × 114,688 B와 정확히 같고, 24 MiB 구성은 예산(25,165,824 B) 아래로 유지된다 — 회계 수정이 1.7b에서도 의도대로 동작한다.

### 기기 결과 — 요청 3개, 컨텍스트 성장 (ctx 638 → 1276 → 1914)

`requests_x3.json`(같은 롱 프롬프트 3회, 2·3회차는 `reset: false`)으로 프리필 캐시가 앞 요청까지 포함해 자라게 했다. 요청별 값이고 read/retained는 누계/최종이다.

| 구성 | decode 요청별 (ms/tok) | TTFT 요청별 (s) | 총 read | 최종 retained | 최종 RSS |
|---|---|---|---:|---:|---:|
| resident | 26.2 / 29.3 / 40.7 | 2.69 / 3.20 / 4.43 | – | – | 1342 MB |
| FSU, 예산 74 MiB | 34.5 / 48.3 / 64.8 | 4.58 / 5.49 / 5.85 | 19.75 GB | 70,557,696 | 1285 MB |
| FSU, 예산 24 MiB | 43.8 / 58.3 / 85.6 | 3.89 / 5.02 / 5.78 | 34.06 GB | 23,519,232 | 1182 MB |
| FSU, 예산 250 MiB | 26.5 / 36.5 / 40.9 | 2.75 / 3.57 / 6.54 | 0 | 219,512,832 | 1313 MB |

컨텍스트(1,914행 = 219,512,832 B)가 예산(74 MiB)을 넘으면 그 구간이 다시 읽혀 디코드가 요청 3에서 resident 40.7 ms → 64.8 ms로 나빠진다. 예산을 250 MiB(1,914행 전량)로 올리면 세 요청 모두 읽기 0, 디코드 26.5/36.5/40.9 ms로 resident에 붙고 회계값도 219,512,832 B(=1,914행)로 정확하다.

### 확인한 것

1. **읽기 증폭 계수는 규모와 무관하다.** KV/token이 같아 예산 0에서 기기 67.67 MB/token(0.6b 64.5 MB/token), 호스트 60.6 MB/token으로 같은 자리이고, 디코드 페널티도 resident 대비 +129%로 0.6b와 같다.
2. **전체 예산이면 resident와 동급이다.** 기기 26.69 vs 26.33 ms(+1.4%), 읽기 0. RSS는 1341 → 1169 MB(−172 MB)로 줄지만 감소율은 0.6b(−28%)보다 작다(−13%) — 가중치가 커져 KV 비중이 낮아졌기 때문이다. KV 자체의 절감량(풀 235 MB − 상주 73 MB)은 0.6b와 같다.
3. **write-through는 여전히 손해다.** 같은 예산에서 +11.0 ms/tok, TTFT +1.03 s → write-back 기본 유지가 타당하다.
4. **예산은 보유할 최대 컨텍스트 기준으로 산정한다.** 요청이 이어져 컨텍스트가 자르면 그 시점 working set이 예산을 넘어 읽기가 재개된다. 산식은 `예산 ≈ 보유 최대 토큰 × 114,688 B`이고, `max_seq_len 2048` 전량을 상주시키려면 234,881,024 B(224 MiB)가 필요하다. 그렇게 하면 RSS가 resident에 근접해(1313 MB) 메모리 절감 목적이 퇴색하므로, 메모리가 급하면 예산을 의도적으로 줄이고 디코드 지연을 감수하는 절충이 된다.
5. **출력 정상.** 단일 요청 5구성, 3요청 4구성 모두 resident와 `output_ids`가 완전 일치한다(요청 27개).
6. **3.3 워커 큐 심화는 우선순위 낮음 유지.** 예산 0(읽기 지배)에서도 요청당 io_read(8.18 s)가 디코드 총합(5.8 s)을 넘고, 예산을 준 구성은 읽기가 0이라 큐가 이득을 낼 구간이 없다.
7. **pack 재배치 재검증.** 검증 후 pack을 `~/TEST_MODELS/qwen3-1.7b`(x86·ARM 가중치, 호스트·기기 설정, README)로 옮겨 같은 요청으로 다시 돌렸다. resident / FSU 64 MiB / FSU 72 MiB 3회의 `output_ids`가 일치했고, 64 MiB는 630행(=534+96 → 72,253,440 B)에 못 미쳐 읽기 170 MB가 남은 반면 72 MiB에서는 읽기 0, `retained_bytes` 72,253,440 B(행 수 × 114,688과 정확 일치), decode 32.37 ms(resident 32.43 ms)로 돌아왔다 — 4항 산식의 실측 확인이다.

## 9. 코드 리뷰 반영 — sync 대칭, compact/FSU 정합성, 방어 코드

HEAD `2bd8c26a5` 기준 독립 리뷰가 세 가지를 지적했다. (1) write-through/write-back A/B이 buffered 모드에서만 `fdatasync` 대칭이 맞아 "쓰기 모음 이득"과 "sync 생략 이득"이 섞일 수 있다, (2) `compact()`의 행 이동이 FSU의 물리 레이아웃과 어긋난다, (3) `flushDirty()`가 쓰기에 앞서 dirty 장부를 지운다. 아래는 이를 확인·분해하고 코드에 반영한 기록이다. 반영 후 유닛테스트는 40/40이고, 기기 실행을 위해 Android CPU 빌드를 다시 배포했다.

### 9.1 sync 조건을 맞춘 재비교 — qwen3 1.7b, prefill 542 / decode 96

먼저 비대칭의 위치를 특정했다. `store()`의 `fdatasync`는 `!direct_`일 때만 실행된다. 이 기기의 `/data`(UFS)는 direct I/O가 자연 선택되므로 **8장의 direct 모드 A/B(write-back 26.69 vs write-through 37.69 ms/token)에는 sync 비용이 들어 있지 않다** — 두 로그 모두 `fdatasync_us=0`이고 write-through 쪽 `io_write_us=549,800`만 잡힌다. 즉 그 +11.0 ms/token은 순수 쓰기 회수 감소다. buffered fallback으로 내려가면 비대칭이 실체가 된다.

비교를 맞추는 스위치 두 개를 추가했다. `NNTR_KV_BUFFERED`(direct I/O 프로브를 건너뛰고 buffered 경로를 강제)와 `NNTR_KV_FLUSH_SYNC`(`flushDirty()` 마지막에 `fdatasync`)이다. 산출물은 `build-fsu/logs/17b-buf_*.log`, `build-fsu/logs/17b-b24_*.log` (기기 원본 `/data/local/tmp/nntr-cpu/`).

| 예산 | I/O 경로 | write-through (store마다 sync) | write-back + flush sync | write-back (sync 없음) |
|---|---|---|---|---|
| 74 MiB (전부 유지) | direct | 37.7 ms/tok · sync 0 | flush 0회 → 좌항과 동일 | 26.7 ms/tok · sync 0 |
| 74 MiB (전부 유지) | buffered | 37.9 / 132.2 ms/tok · sync 15.1 / 64.7 ms/tok (디코드의 40–49%) | flush 0회 → sync 0, 측정 33.4 / 34.6 | 26.6 / 32.5 ms/tok · sync 0 |
| 24 MiB (퇴피 발생) | buffered | 142.7 / 137.1 ms/tok · sync 58.5–61.0 ms/tok (43%) | 126.4 / 114.2 ms/tok · sync 33.3–34.5 ms/tok (27–29%) | 132.1 / 65.7 ms/tok · sync 0 |

분해 결과:

1. **direct 모드는 애초에 대칭이었다.** sync 항이 0이므로 8장의 write-back 이득(+11.0 ms/tok, TTFT +1.03 s)은 전부 쓰기 회수 감소다. 결론은 그대로 유효하다.
2. **buffered에서는 sync가 페널티의 대부분이다.** 읽기가 0인 예산 전량 구성에서 write-through의 `fdatasync` 합계가 1.45–6.21 s로 디코드의 40–49%를 차지하고, 같은 실행의 실제 전송 시간(`io_write_us`)은 0.05–0.22 s에 불과하다. 24 MiB에서도 비율은 같다(5.6–5.9 s, 43%; 전송 0.72–1.0 s).
3. **sync를 맞춰도 쓰기 모음은 이득이다.** 24 MiB(buffered)에서 write-through 평균 139.9 ms/tok vs write-back+flush-sync 평균 120.3 ms/tok → **−19.6 ms/tok(−14%)**. 여기서 sync까지 생략하면 평균 98.9 ms/tok로 추가로 −21.4 ms(−18%). **buffered 구간에서 관측되는 합산 이득의 절반가량이 sync 생략 몫**이므로, buffered 기기에서 그 이득을 "쓰기 모음"의 효과로 라벨링하면 안 된다.
4. **전부 유지 구성은 이 비교가 성립하지 않는다.** 요청 중 owner가 퇴피하지 않아 flush가 0회이므로 write-back+sync는 write-back과 같은 코드를 돈다(실측 33.4 / 34.6 vs 26.6 / 32.5의 차이는 이 라운드 실행 순서 편차다. 같은 배치의 TTFT가 2.78 → 4.64 s로 올라간 것과 함께 봐야 한다). 파일에 남는 기록은 프리필·스냅샷 경로뿐이다. 6장 해석 3의 "요청 내 데이터는 DRAM에만 있다"와 같은 뿌리다.
5. **24 MiB 구간은 읽기가 지배한다**(요청당 read 4,002 MB). run 간 편차가 65.7–142.7 ms/tok라 위 비율도 ±30%로 봐야 하며, 정밀 분리에는 워밍업·주파수 고정 실행이 별도로 필요하다.

### 9.2 compact() × FSU 정합성

`compact()`는 windowed 레이어(= `allocate()`의 rows/keep으로 만든 레이어, `windowed()` ⇔ `keep_ > 0`)만 이동시키고, `allocate()`는 flash 저장소를 켠 캐시에서 rows/keep을 **이미 거부한다**(`KV FSU requires batch-one CPU attention with FP16 externally bound KV`). 따라서 flash 모드에서 `compact()`의 이동 대상은 존재하지 않으며, 4–8장 실행에 조용한 데이터 손실은 없었다. 리뷰가 남긴 위험은 조합이 열리는 미래(CPU ring 축소, HTP 이식)다 — `finishFlashLayer`의 `start = from % cap`이 base 이동을 따라가지 못하는 지점 그대로 남는다.

방어 두 개를 넣었다.

- `compact()`은 flash가 켜진 캐시에서 windowed 레이어의 이동을 만나면 `std::logic_error`로 거부한다(`allocate()` 가드가 있어 지금은 도달 불가지만, 레이아웃이 reconciled되기 전까지 조용한 memmove를 막는 belt).
- 회귀 테스트 `flash_storage_and_windowed_rows_are_exclusive`: flash + rows/keep `allocate()`는 `std::invalid_argument`, flash 캐시의 `compact()`는 `false` 반환 + base 0 유지 + droppedAmax 비움.

장기 과제는 `kv-cache-fsu-htp-handoff.ko.md` 4단계 0항으로 옮겼다.

### 9.3 나머지 리뷰 항목

| 항목 | 반영 |
|---|---|
| `flushDirty()`가 쓰기 전에 `dirty.clear()` | merged 범위를 전부 기록한 **뒤에** 지운다. 쓰기 예외 시 장부가 남아 "DRAM에만 있는 행"의 기록이 사라지지 않는다(세션 재시작 정책은 기존 유지) |
| sync 경로 중복 | `syncBackingFile()`로 통합 — `store()`(buffered write-through)과 `flushDirty()`(옵션)가 같은 `fdatasync_us` 계측을 쓴다 |
| 생성자 `layers_.size() - 1` | `layers_`가 비면 slack 0. 언더플로로 거대한 resize가 되는 경로를 차단 |
| budget 테스트의 env 누수 | `ScopedEnv` RAII 헬퍼로 교체 — 실패한 `ASSERT_*`가 `NNTR_KV_RESIDENT_BYTES`를 후속 테스트에 남기지 않는다 |
| 저장소 루트 `resident`(0 B) | 실행 부산물이라 삭제 |

### 9.4 회귀 확인

- 유닛테스트 40/40 (`unittest_kv_cache_manager`, 신규 1개 포함).
- 새 바이너리로 resident 26.5 / FSU(write-back) 27.4 ms/tok — 수정 전 26.33 / 26.69 대비 무회귀.
- 이번 라운드 buffered/direct 8회의 `output_ids` 해시가 전부 `17fb61…`로 resident와 일치.

## 10. 호스트 3-way 분리 측정 — 쓰기 정책 기여 분리

§9.1의 기기 3-way(24 MiB, buffered)는 run 간 편차가 ±30%라 "쓰기 모음"과 "sync 생략"을 정밀하게 분리하지 못했다(§9.1 분해 결과 5항). 같은 매트릭스를 호스트에서 **교차 반복**(4구성 × 5회, 매 반복 A→B→B2→C 순서)으로 돌려 분해 정밀도를 확보한다. 호스트는 NVMe ext4에서 direct I/O가 열린다는 점이 기기(UFS, buffered 강제)와 다르다. 즉 이번 측정은 "direct I/O 환경의 3-way"이고, §9.1의 buffered 3-way와 나란히 놓으면 어느 축이 환경 의존인지가 분리된다.

### 측정 환경

- 바이너리: `build-fsu/Applications/CausalLM/nntr_causallm`(kv-cache-fsu 브랜치, §9 이후 무변경).
- pack: `~/TEST_MODELS/qwen3-1.7b-fsu` — `kv_flash_fsu` → `~/TEST_MODELS/qwen3-1.7b-fsu-work/kv`(lookahead 1), 가중치·토크나이저는 원본 pack 심링크. 요청은 원본 pack의 `~/TEST_MODELS/qwen3-1.7b/requests.json`(단일 요청, prefill 534 + decode 96, ctx 630).
- 환경변수: `NNTR_ENGINE=cpu NNTR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NNTR_KV_TIMING=1`.
- 저장매체: 백킹 파일은 위 config 경로(ext4, `/dev/nvme0n1p2`)에 만들어지고 곧바로 unlink된다. **direct_io=1**(O_DIRECT 프로브 성공, 20/20 로그). 스크립트가 준비한 `/var/tmp/nntr-fsu-matrix`는 실제로는 쓰이지 않는다(같은 ext4).
- 스크립트 `/tmp/run_fsu_matrix.sh`, 로그 `build-fsu/logs/{a0-budget0-wt,b24-wt,b24-wb-sync,c24-wb}-r{1..5}.log`.

| 구성 | 예산 | 쓰기 정책 |
|---|---|---|
| A `a0-budget0-wt` | 0 | write-through(기존 FSU 동작) |
| B `b24-wt` | 24 MiB | write-through(`NNTR_KV_WRITE_THROUGH=1`) |
| B2 `b24-wb-sync` | 24 MiB | write-back + flush sync(`NNTR_KV_FLUSH_SYNC=1`) |
| C `c24-wb` | 24 MiB | write-back(기본) |

### 결과 — 반복 5회, p50(min–max)

decode ms/token은 `[PERF-SPLIT]` steady 구간(첫 토큰 제외 95토큰)이며 `[REQUEST]`의 `decode_ms`와 같다. read/write/retained/deferred **바이트는 반복 5회 전부 1바이트 오차 없이 같았다**(정책이 결정적이라 변하는 것은 시간뿐) — 편차가 시간 축에만 있다는 뜻이다.

| 구성 | decode (ms/tok) | CV | TTFT (s) | read MB/token | io_read (s) | io_write (s) | fdatasync (s) | retained (B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A 예산 0 + WT | 124.12 (121.77–126.65) | 1.5% | 10.82 | 66.75 | 8.81 | 4.56 | 0 | 0 |
| B 24 MiB + WT | 104.48 (102.48–106.91) | 1.7% | 10.88 | 42.73 | 5.31 | 3.06 | 0 | 23,224,320 |
| B2 24 MiB + WB+sync | 101.46 (98.20–105.11) | 2.9% | 10.74 | 42.73 | 5.31 | 2.67 | 0 | 23,224,320 |
| C 24 MiB + WB | 103.03 (97.03–109.50) | 5.1% | 10.69 | 42.73 | 5.34 | 2.68 | 0 | 23,224,320 |

I/O 회계(네 구성·모든 반복 공통 확인): read A 6,407,847,936 B, B·B2·C 4,101,619,712 B(A의 0.640×) · write 네 구성 모두 83,263,488 B(총 기록량은 정책과 무관) · deferred_writes는 B2·C만 2,716(= 28레이어 × 97 store) · retained 23,224,320 B = 소유자 9개 × 2,580,480 B로 예산 25,165,824 B 이내 · peak_kv A 5,160,960 B(lookahead 1의 2소유자), 24 MiB 구성 33,185,792 B(예산 풀 + 작업 버퍼). 전체 KV는 630행 × 114,688 B = 72,253,440 B라 24 MiB 예산은 그 35%로 실제 constrained 구간이다.

### 분해 (median 기준)

1. **쓰기 모음(batching) 기여 = B−B2 = +3.0 ms/tok(2.9%).** io_write 합계도 3.06 s → 2.67 s(−0.39 s ≈ 95토큰당 4.1 ms)로 같은 방향이라 교차 확인된다. write-through는 store마다 평면당 4 KB짜리 pwrite를, write-back은 퇴피 시 모아서 큰 pwrite를 한다 — 총 기록 바이트가 같으므로 차이는 전부 호출 크기·시점이다.
2. **sync 생략 기여 = B2−C = 구조적으로 0.** flush sync는 `flushDirty()`에서 `flush_sync_ && !direct_`일 때만 걸리고, store()의 sync도 마찬가지다. 호스트는 direct I/O라 B2도 fdatasync를 한 번도 호출하지 않는다(fdatasync_us = 0.0) — B2와 C는 같은 코드를 돈다. 측정된 −1.6 ms/tok(B2가 빠른 방향)은 CV 2.9%·5.1% 노이즈 안이다.
3. **기존 FSU 대비 전체 개선 = A−C = +21.1 ms/tok(−17.0%).** 읽기 6,407.8 → 4,101.6 MB(−36%), io_read 8.81 → 5.34 s. 이 매트릭스에서 유일하게 큰 축은 예산 상주다.

### §9.1 기기 수치와의 정합성

| 축 | 기기 24 MiB buffered(§9.1, UFS) | 호스트 24 MiB direct(이번) |
|---|---|---|
| 쓰기 모음 (WT → WB+sync) | −19.6 ms/tok(−14%) | −3.0 ms/tok(−2.9%) |
| sync 생략 (WB+sync → WB) | −21.4 ms/tok(−18%) | 구조적 0(direct라 sync 경로 없음) |
| decode ms/tok — WT / WB+sync / WB | 139.9 / 120.3 / 98.9(2회 평균) | 104.5 / 101.5 / 103.0(5회 중앙값) |
| read MB/token — 예산 0 | 67.67 | 66.75 |
| read MB/token — 예산 24 MiB | 43.72 | 42.73 |
| 편차 | ±30%(2회) | CV 1.5–5.1%(5회 교차) |

1. **read 증폭은 플랫폼 무관이다.** 예산 0에서 66.75 vs 67.67 MB/token, 24 MiB에서 42.73 vs 43.72 — 차이는 ctx(630 vs 638) 비례분이다. §8의 "읽기 증폭 계수는 규모와 무관"에 플랫폼 축이 추가된다.
2. **§9.1의 큰 쓰기 정책 이득은 buffered 한정이다.** 기기에서는 WT의 store마다 fdatasync(58.5–61.0 ms/tok, 디코드의 43%)가 지배했고 sync를 맞춰도 모음 이득 −19.6 ms가 남았다. 호스트 direct에서는 애초 sync가 경로에 없어 모음 이득만 남는데, 그 값이 −3.0 ms(2.9%) — 호스트 NVMe는 소형 pwrite도 비싸지 않다. 즉 write-back의 우위 폭은 "sync 비용 × 소형 쓰기 비용"이 큰 저장매체에서 커지고, direct I/O가 열리는 환경에서는 크게 줄어든다. write-back 기본 유지(C) 결론은 어느 쪽에서도 뒤집히지 않는다.
3. **분해의 우선순위가 뒤집힌다.** 기기(buffered)에서는 쓰기 정책 두 축이 합계 −41 ms였다면, 호스트(direct)에서는 예산 상주 −21.1 ms가 지배이고 쓰기 정책은 ±3 ms 수준이다. constrained-regime의 병목 축은 저장매체 특성(sync 비용, 소형 쓰기 비용)에 따라 달라지므로, 측정 없이 한 축의 이득을 다른 환경에 일반화하면 안 된다.

### 유의점

1. 절대 시간은 기기와 비교하지 않는다. 이 호스트는 단일 스레드가 기기보다 느려(같은 요청 prefill 49.8 tok/s vs 기기 142) 구성 간 델타와 I/O 바이트만 본다.
2. `requests.json`의 프롬프트는 `sample_input`의 앞부분 절단본이라 prefill이 542가 아니라 **534 토큰**이다. read MB/token의 기기·호스트 차이 대부분이 이 ctx 차이다.
3. 호스트 decode 쓰기(평면당 4 KB)는 64 KB 비동기 경로 미달이라 전부 동기 실행된다(async_writes = 0) — 쓰기 모음 이득이 과소 계산될 여지는 있으나 네 구성 모두 같은 조건이다.
4. 첫 시도에서 스크립트가 FSU 설정이 없는 pack(`~/TEST_MODELS/qwen3-1.7b`)을 가리켜 FSU가 꺼진 채 돌았다. pack을 `~/TEST_MODELS/qwen3-1.7b-fsu`로 고쳐 재실행했고(요청 파일은 원본 pack 것을 사용), 위 20개 로그가 그 결과다. 코드 변경은 없다.
5. `output_ids`는 20개 실행 모두 완전 일치한다(요청당 97 ids — `gen_tok` 96 표기와 1 차이 나는 보고 필드 특성은 §9.1 기기 로그와 같은 기존 현상이며 전 실행 동일하다).

## 11. 혼재 KV 저장소를 CPU 실그래프에 연결 — sliding window 행 축소 활성화 (호스트)

§9.2가 기록한 대로 `compact()`의 이동 대상은 "allocate()가 rows/keep을 거부하는 동안에는 존재하지 않는다"가 정확했다. fc2fec519가 저장소 레이아웃(혼재 구성)을 연 뒤에도 CPU 실그래프의 `planKVCache()`는 sliding 레이어에 여전히 full height(`MAX_SEQ_LEN`)를 내려주고 있어서, windowed 레이어라는 물리 실체가 그래프 위에는 만들어지지 않았다. 이번 변경으로 CPU 경로가 HTP(`gemma4_htp_causallm.cpp`의 행 산식)와 **동일한 windowed 상한**을 쓰도록 활성화했다 — 즉 1단계(fc2fec519)와 6단계(CPU 반쪽)가 이어졌다. 레이아웃 수식 자체는 바꾸지 않았다(§3.1의 D 정책 유지, B는 여전히 미래 과제).

### 구현

| 위치 | 내용 |
|---|---|
| `transformer.h` | `getKVCacheRows()`에 windowed 분기 추가: sliding 레이어가 `rows = min(max_seq, roundUp64(window + max(init_seq_len, num_to_generate)))` 행으로 줄어든다(HTP 산식과 동일). `kvFsuWindowCap()`이 참일 때만 적용되고, sink/int8 레이어는 구조적으로 제외(`kvRingLayerEligible` 재사용), window ≥ max_seq면 full height(기존 동작) |
| `causal_lm.{h,cpp}` | `KVCacheLayout`에 rows/keep 벡터 추가. `planKVCache()`가 windowed rows를 계산하고 `allocateKVCacheStorage()`가 `kv_cache.allocate(..., rows, keep)`로 전달. `CausalLM::kvFsuWindowCap()`: `kv_flash_fsu` 구성 + cpu 엔진에서 자동, 상주 실행에서는 `NNTR_KV_WINDOW_CAP=1`로 강제(동일 상한의 A/B 기준선용) |
| `causal_lm.cpp` | `compactKVCache()`가 `cache_base`를 CPU `MHACoreLayer`에도 push(기존엔 HTP 전용). prefill 경로(windowed가 켜지면)에 compaction 지점 정리: 청크 루프는 청크 앞, 단일 블록은 블록 앞, 행 단위 재개 프리필은 행 앞 |
| `mha_core.{h,cpp}` | legacy host 경로에 `kv_cache_base` 추가 — 모든 캐시 행 오프셋(`cacheRow()`)이 평면 상대가 되고 RoPE/마스크는 절대 위치 유지. 창을 벗어나는 step 쓰기는 "compaction을 건너뛰었다"는 원인을 밝히는 예외로 거부 |
| `kv_cache_manager.cpp` | allocate 완료 시 혼재 분해를 한 줄로 증언: `[kv-fsu] mixed storage: N windowed layer(s) on host DRAM (compacting rows), M streaming through flash, ...` |

### 의미 변화 (기존 CPU 상주 실행 대비, 명시)

**CPU sliding 레이어의 KV 캐시 저장 높이가 "전체 이력(max_seq행)"에서 "창 K행(window + headroom)"으로 바뀐다.** 기존 CPU 상주 실행은 sliding 레이어도 max_seq 전체를 유지했지만, 참조하는 마스크는 마지막 W행뿐이었다(HTP와 gemma4 모델 의미와 같은 근거). 이 변경은 그 **버려지는 접두 행의 저장을 실제로 줄이는 것**이고, 정확도 기준선은 같은 상한을 강제한 상주 실행(`NNTR_KV_WINDOW_CAP=1`)이다. 전체평면 상주 실행(구 의미)도 함께 비교했고 결과는 아래 표와 같다. full-attention 레이어와 sliding 없는 모델(qwen3)은 이 변경과 무관하다 — cap 수식이 full height를 그대로 내려주므로 회귀면이 아니다.

### 검증 — gemma4_tiny 3-way (sliding_window=4, layer 0 sliding + layer 1 full, max_seq_len 128 확장)

원본 fixture(max_seq 8)는 `roundUp64(4+4)=64 ≥ 8`이라 full height 그대로라(회귀 확인용으로 남김), `max_seq_len=128`·`max_position_embeddings=128`로 확장한 pack 3종으로 비교했다. 프롬프트 80토큰, `NNTR_PREFILL_CHUNK=4`(청크마다 compaction), num_to_generate 1.

| 구성 | windowed 상한 | output_ids | prefill / cached_after | 비고 |
|---|---|---|---|---|
| 상주, 전체평면 (구 의미) | 없음(2×128행) | `[0, 0]` | 80 / 81 | 기준선 A |
| 상주 + `NNTR_KV_WINDOW_CAP=1` | sliding 64행(`KV rows 192 (vs 256 full)` 로그) | `[0, 0]` | 80 / 81 | 기준선 B — A와 일치 |
| FSU (`kv_flash_fsu`) | 자동 | `[0, 0]` | 80 / 81 | `[kv-fsu] mixed storage: 1 windowed layer(s) on host DRAM (compacting rows), 1 streaming through flash, 0 aliased, of 2 layers` |

FSU 실행의 I/O 회계: `peak_kv_bytes=16384 read_bytes=196608 write_bytes=172032` — flash 스트리밍은 full 레이어(128행)만으로 windowed 레이어는 flash에 없다. 이어지는 요청(reset 없음, cached_after 81 → 89)도 세 구성 모두 `[0, 0]`으로 일치 — 재개 프리필 구간에서도 compaction·base 전파가 유지된다. 위치 4 이후 매 청크에서 창 이동(memmove)이 일어나므로 이 비교는 "창 밖 행을 버린 후의 출력"이 같음을 검증한다.

### 회귀

- `unittest_kv_cache_manager` **42/42**(build-fsu, fp16 빌드) — fc2fec519의 혼재 3건 포함.
- `unittest_causallm_models` **124/125**(builddir_x86_test, fp32 빌드; `NNTR_QUANTIZE_BIN` 지정, `ASAN_OPTIONS=detect_leaks=0` — leak은 별개 기존 항목) — Gemma4/Qwen3/GptOss 등 CPU 실그래프 전 경로. 1건 스킵은 QINT4의 KAI int4 패킹이 NEON 전용이라는 기존 플랫폼 게이트.
- 원본 fixture 그대로: qwen3_tiny 상주=FSU, gemma4_tiny(max_seq 8, windowing 불가) 상주=FSU — `output_ids`·FSU 바이트 회계 모두 변경 전과 동일, mixed 라인 없음.

### 리뷰 결함 3건 수정 (동일 변경 내 재검증 라운드)

첫 리뷰에서 세 가지 결함이 지적되어 모두 수정하고 재검증했다.

**1. 리셋 후 재요청 FATAL (치명) — 수정.** 첫 요청의 compact가 남긴 `bases_ > 0`을 `setSessionTokenPosition(0)`이 지우지 않아, `reset: true` 두 번째 요청의 `compact(0)`이 `validateRange`에서 "layer 0 does not hold rows [0, 0)"로 죽었다. `setPosition(0)`은 쓰기 커서만 되감을 뿐 windowed 메타데이터(base·decode offset·dropped amax)는 그대로 둔다. 수정: 새 프롬프트 진입점(`setSessionTokenPosition`의 `pos == 0` 분기, 기기측 세션 상태를 지우는 곳)에서 `kv_cache.reset()`을 호출해 원점으로 복원하고 — flash 소유권 해제와 prefix view freeze가 `reset()` 안에 이미 있다 — 바로 이어 `compactKVCache(0, /*force=*/true)`로 `cache_base=0`을 레이어에 재푸시한다. manager reset과 base 재푸시는 한 쌍이다: `compactKVCache`의 no-op 조기 리턴(`!compact && !force`)은 "레이어가 이미 관리자의 base를 알고 있는" 경우에만 지름길이 되어야 하는데, 리셋 직후는 그 반대이므로 force로 지나간다.

**2. 무제한 생성이 mid-generation FATAL (중간) — 수정.** `NUM_TO_GENERATE`는 `setupParameters`가 없는/음수 값을 0으로 정규화하는데("컨텍스트 끝까지 생성"), headroom 계산은 `max(INIT_SEQ_LEN, 0)` = INIT_SEQ_LEN만 남겨 유한 평면을 만들었다. g4t_512 팩(max_seq 512, window 4, num_to_generate 없음)에서 60토큰쯤 생성 후 `mha_core kv-window: step write [140, 141) does not fit a 64-row windowed plane based at 76`로 죽었다(전체평면 컨트롤은 cached_after 511까지 정상 — 회귀). 수정: 행 산식을 `windowedKVRows()`(kv_ring.h, 유닛테스트 가능한 순수 함수)로 뽑아내고, `num_to_generate <= 0`이면 headroom을 `max_seq - window`로 잡아 수식이 전체 높이로 퇴화시킨다 — planKVCache의 keep은 full height 레이어에 0이므로 그 레이어는 애초 windowed가 아니게 된다(사실상 전체 높이 fall-back). 기존 주석의 "unsigned cast가 headroom을 balloon시킨다"는 서술은 정규화를 놓친 것이어서 실제 근거(무제한 꼬리는 유한 평면에 못 담는다)로 고쳤다. 참고로 HTP 산식(`Gemma4HtpTransformer::getKVCacheRows`)에는 이 무제한 분기가 없는데, HTP 팩은 항상 유한 `num_to_generate`를 실어 보내므로 그대로 뒀다(다른 저장소 경로라 여기서 넓히지 않음).

**3. windowed→windowed alias의 base stale (중간) — 수정, 대안 (a) 채택.** gemma4 `num_kv_shared_layers > 0`에서 공유 trailing 레이어가 sliding 타입이면 windowed 레이어가 windowed 소유자를 alias한다. compact는 aliased 레이어를 스키핑하므로 sharer의 `getBase()`가 영원히 0이고, 소유자가 compact로 움직인 뒤 sharer의 CPU attention는 평면 row 0을 절대 0으로 읽는다(조용한 오답), save 오프셋도 어긋난다. 택일: (a) base 푸시 시 alias 소스의 base로 resolve / (b) windowed alias 자체를 거부. **(a)를 채택했다** — sharer의 mha_core는 그래프가 소유자의 k/v norm으로 연결되므로 같은 절대 행에 byte-identical 값을 쓰는 redundant write가 되고, 소유자 사상 위에서는 무해하다. 기능 보존(gemma4 kv-share + 혼재 공존)이 (b)보다 낫다. 구현: `getBase()`/`getDecodeOffset()`가 alias 소스를 재귀 resolve(isFlashOwned와 같은 모양)해서 `cache_base` 푸시·`validateRange`·save의 행 오프셋이 모두 소유자 base를 쓰게 하고, `validateKVSources`가 alias 가족의 창(keep) 동일성을 강제한다(창이 다르면 소유자 base가 sharer의 liveFrom과 어긋나므로 allocate 시 거부). sharer의 `bases_`에 소유자 base를 복사하는 방식은 base 변경 지점 4곳(compact·prepareDecodeView·restore·reset)마다 복사를 맞춰야 해서 기각 — resolve는 읽는 곳이 한 곳이다.

**회귀 테스트 4건 추가**(`unittest_kv_cache_manager.cpp`, 42 → 46):

- `KVCacheManagerWindowed.reset_returns_windowed_state_to_origin` — compact로 base 4까지 진행한 뒤 reset → base/position/dropped amax 원점 단언, `compact(0)` 성공, 이어쓰기 정합(재충전 후 같은 compaction 결과). FSU 변형은 flash 소유권 해제(`flashRetainedBytes() == 0`)까지 단언. CLI 레벨 확인은 아래 재검증 (i).
- `KVCacheManagerWindowed.rows_formula_bounds_the_decode_tail` — `windowedKVRows` 수식: 유한 num_to_generate(창+headroom 반올림, max_seq 캡), 무제한(0/음수)은 전체 높이.
- `KVCacheManagerFlash.windowed_alias_tracks_the_owner_base` — windowed 소유자 + windowed alias + full flash + full alias 4층 구성에서 compaction마다 `getBase(aliased) == getBase(owner)` 단언(수정 전에는 0으로 머무는 것을 확인했다 — stash로 구 수정을 걷어내고 동일 테스트가 이 단언에서 실패하는 것을 검증), 스냅샷 페이로드가 "sharer가 자기 평면을 소유하는 DRAM 빌드"와 byte-identical(헤더의 geom digest는 alias 맵 자체를 인코딩하므로 페이로드만 비교).
- `KVCacheManagerShare.rejects_windowed_alias_with_mismatched_window` — keep 불일치·windowed/full 방향 불일치 거부, 동일 keep 수용.

### 재검증 (수정 후, /tmp/kvwin-verif 팩)

| 확인 | 결과 |
|---|---|
| `unittest_kv_cache_manager` (build-fsu, fp16) | **46/46** (기존 42 + 신규 4) |
| `unittest_kv_cache_manager` (builddir_x86_test, fp32) | **36/36** (fp32 빌드는 fp16 전용 블록이 컴파일 제외 — 기존과 동일 기준, 신규 테스트는 fp16 게이트 안에 있어 빌드 자체가 정상) |
| `unittest_causallm_models` (builddir_x86_test, fp32, `NNTR_QUANTIZE_BIN`, `ASAN_OPTIONS=detect_leaks=0`) | **124/125**(1건 스킵 — QINT4 KAI int4 패킹 NEON 전용 기존 게이트, 변경 전과 동일) |
| `unittest_causallm_mha_core_dispatch` (build-fsu) | 11/11 |
| (i) 리셋 2요청 — 상주 windowed(`NNTR_KV_WINDOW_CAP=1`)·FSU 모두 | FATAL 0, exit 0. 요청 1은 수정 전 기록(cached_after 8, `[0, 0]`)과 동일, 요청 2는 두 구성 모두 cached_after 4·`[0, 0]`으로 서로 일치(수정 전에는 둘 다 FATAL로 요청 2 미완료) |
| (ii) 상주 windowed vs FSU 혼재 (연속 요청 포함) | `output_ids`·`cached_after`(81→89) 전 필드 일치 |
| (iii) 3-way(전체평면 A = windowed B = FSU C, 단일+연속) | 세 구성의 `output_ids`/`cached_after`/`prefill_tokens` 일치, 수정 전 기록(A_full/B2_win/C2_fsu)과도 일치 |
| (iv) qwen3_tiny·원본 gemma4_tiny(max_seq 8) 상주=FSU | `output_ids` 수정 전과 동일(회귀 없음, mixed 라인 없음) |
| 무제한 생성(g4t_512, num_to_generate 없음, reset 요청 1개) | 전체평면·windowed·FSU 세 구성 모두 FATAL 없이 **cached_after 511까지 완주**, `output_ids`가 전체평면 컨트롤과 비트 일치(수정 전: windowed·FSU는 140에서 FATAL) |
| 포맷 | `git clang-format-14 HEAD`로 변경 라인 검사 — 추가 수정 없음 |

### 실험실 실팩(gemma4_cpu / gemma4_it_x86) — 선결과제로 보류

로컬 gemma4 실팩 전체가 `tie_word_embeddings=true` + `skip_prefill=true`인데, HEAD의 `buildLmHeadOutput()`이 tied lm_head에 `plan_last_row_only=true`를 넣고 `TieWordEmbedding`이 이 속성을 받지 못해 그래프 구성 단계에서 FATAL난다(변경 코드가 전부 dormant인 구 의미 상주 경로에서도 동일 재현 — 본 변경과 무관한 기존 불일치). **이 FATAL은 이후 [lmhead] 커밋(`fa82a1799`)으로 수정됐다** — 두 레이어가 이미 height-1 플래닝과 마지막 행만 계산을 무조건 하고 있었으므로 속성 선언만 추가했고, tied pack 두 경로(skip_prefill on/off) 완주를 확인했다. 또한 실팩의 기본 설정은 windowed 상한이 어차피 full height다: gemma4_cpu는 `roundUp64(512+2560)=3072 = max_seq`, gemma4_it_x86은 `min(roundUp64(512+4500), 5000) = 5000 = max_seq`. 행 축소를 관측하려면 `init_seq_len`을 줄인 pack이 필요하다(위 FATAL은 해제됐다). 실팩 측정(메모리 절감량, RSS, 창 크기별 디코드)은 별도 기록으로 남긴다. 참고: gemma4_cpu·gemma4_it_arm_precomputed_q40 pack은 별개의 기존 문제(Q4_0 시절 양자화본과 HEAD의 Q6_K `per_layer_input_embedding` 기대 불일치, 재양자화 필요)로 가중치 로드 단계에서 실패한다.

### 남은 것

1. HTP 기기 매트릭스의 compaction 경계 행(6단계의 기기 반쪽) — 이 기록은 호스트 반쪽만이다.
2. 실팩 규모의 메모리 절감 실측(위 보류 사유). tiny fixture에서는 절감량이 64행×32채널 수준이라 측정 의미가 없다. → §12에서 수행.
3. lookahead 겹침 손실(fc2fec519 기록)의 실팩 규모 정량화. → §12에서 decode ms로 1차 확인(개별 요인 분해는 미수행).

## 12. 실팩 측정 — gemma4 runtime pack에서 CPU sliding window 축소의 효과 (호스트)

§11의 남은 것 2번이다. tiny fixture에서는 절감량이 측정 의미가 없다고 판단했던 "실팩 규모의 메모리 절감 실측"을 실제 pack으로 수행한다. 측정 시점 HEAD는 `10979d2b8`(§11의 행 축소 + tied lm_head FATAL 수정 `fa82a1799`까지 반영)이며, 이 절의 작업에 코드 변경은 없다(측정·기록만).

### pack과 설정 복제

원본 pack `~/TEST_MODELS/gemma4_it_x86_runtime_pleq6k`는 tied lm_head + `skip_prefill=true`로 동작이 확인된 pack이다(§11의 보류 사유는 `fa82a1799`로 해제됐다). 구조는 35개 레이어가 4 sliding + 1 full 패턴 7조, `num_kv_shared_layers=20`(KV 오너 15개 = sliding 12 + full 3, 나머지 20개는 alias), `sliding_window` 512, `max_seq_len` 3072, `init_seq_len` 2560, `num_to_generate` 60.

기본 설정의 windowed 상한은 full height다: `roundUp64(512 + max(2560, 60)) = 3072 = max_seq`. 원본 pack을 그대로 `NNTR_KV_WINDOW_CAP=1`로 돌려 확인했다 — `[kvcache] window-ring: 35 layers, KV rows 107520 (vs 107520 full), 126.0 MB`. §11이 든 gemma4_it_x86(전체 높이 5000)과 같은 "설정값 산식이 전체 높이" 케이스이며, **이 pack의 전체 높이는 3072**다. 즉 기본 설정에서는 축소가 관측되지 않는다.

축소가 발동하는 비교 pack을 tmpdir에 복제한다(가중치는 심링크, 설정만 복제 — 원본 디렉터리는 읽기만 한다):

| 경로 | 내용 |
|---|---|
| `/tmp/kvwin-g4-real` | A·B용. `nntr_config.json`만 수정 — `init_seq_len` 2560→512, `num_to_generate` 60→1024, `tokenizer_file`을 복제본으로 재지향, `chat_input`을 짧은 프롬프트로 교체(실행에는 미사용, requests 모드 사용). `config.json`·`generation_config.json`·`tokenizer.json`·`tokenizer_config.json` 복사, 가중치 bin 심링크. windowed 상한 = roundUp64(512+1024) = **1536 < 3072** → sliding 레이어 1536행 vs 3072행 |
| `/tmp/kvwin-g4-real-fsu` | C용. 위와 동일 구성 + `nntr_config.json`에 `kv_flash_fsu {type: "directory", directory: "/tmp/kvwin-g4-real/kv", lookahead: 1}` 추가. 나머지 파일은 전부 `/tmp/kvwin-g4-real` 심링크 |
| `/tmp/kvwin-g4-real/kv` | FSU 백킹 파일이 만들어지는 실제 경로. tmpfs가 아닌 ext4(`/dev/nvme0n1p2`)라 `direct_io=1` 프로브가 성공한다 |

pack에 `requests.json`이 없어 새로 만들었다. gemma4 템플릿을 직접 적용한 40토큰 프롬프트 1건(`--requests` 모드, `init_seq_len` 512 이하). `generation_config.json`은 원본 그대로(do_sample=true, temperature 1.0, top_k 64, top_p 0.95 — 매 실행 고정 시드 재사용으로 결정적, §10과 같은 전제). 세 구성 모두 `cached_after` 1064(프리필 40 + 생성 1024, EOS 조기 종료 없음)로 끝까지 생성했다.

### 매트릭스

| 구성 | 설정 | 기대 저장소 |
|---|---|---|
| A | resident, 기본 | 전 레이어 3072행(기존 의미 — §11 이전 기준선) |
| B | resident + `NNTR_KV_WINDOW_CAP=1` | sliding 오너 12개 1536행(windowed·compacting), full 3072행, flash 없음 |
| C | `kv_flash_fsu`(lookahead 1) + 자동 상한 | B와 같은 상한 + full 오너 3개는 flash 스트리밍(혼재 저장소) |

환경: `NNTR_ENGINE=cpu`(스레드 기본값, 32), 피크 RSS는 `/usr/bin/time -v`의 Maximum resident set size, 로그는 `/tmp/kvwin-g4-real/logs/{a,b,c}-r*.log`. A 5회(첫 회 이상치 확인을 위해 2회 추가), B·C 3회, A→B→C 교차.

### 결과

**output_ids는 전 회차 일치다.** 완주 실행 11회(A×5, B×3, C×3)가 모두 1,024 ids로 동일한 해시(`42ff968760ff`), 아래 구조분해 프루브 3회(gen=1)도 첫 토큰 1408로 일치.

| 구성 | peak RSS KB, 중앙값 (min–max) | 저장소 증거 (로그 그대로) | decode ms/tok | TTFT ms |
|---|---|---|---|---|
| A resident 풀높이 (5회) | 3,469,980 (3,468,804–3,650,560)* | `[kvcache] window-ring: 35 layers, KV rows 107520 (vs 107520 full), 126.0 MB` | 49.93 (49.83–50.16) | 233.5 (232.0–285.5) |
| B resident windowed (3회) | 3,451,912 (3,451,268–3,452,004) | `[kvcache] window-ring: 35 layers, KV rows 64512 (vs 107520 full), 84.0 MB` | 50.04 (49.90–50.16) | 231.1 (229.0–231.2) |
| C FSU 혼재 (3회) | 3,439,392 (3,434,804–3,443,316) | `[kv-fsu] mixed storage: 12 windowed layer(s) on host DRAM (compacting rows), 3 streaming through flash, 20 aliased, of 35 layers` | 63.79 (62.96–63.82) | 242.9 (241.6–245.2) |

\* A의 첫 회(3,650,560)만 이후 4회와 프루브 대비 +176 MiB 높다. 출력·시간은 동일하고 r2–r5는 모두 3.47 GB대라 첫 실행 한정 할당 히스토리로 보는데, 중앙값에 영향이 없어(min–max에 그대로 남김) 별표로만 기록한다.

window-ring 라인의 행 수는 논리합(35레이어, alias 포함)이다. B의 64,512 = 28 sliding × 1,536 + 7 full × 3,072로 상한 산식과 정확히 일치하고, MB 표기는 MiB다. C에는 window-ring 라인이 없는데, flash 분기가 `allocate()`에서 링 진단 블록 전에 리턴하기 때문이다 — 그 자리의 증인이 mixed 라인이며(12 windowed / 3 streaming / 20 aliased는 예상 오너 분해와 정확히 같다), kv-share 라인(`20 of 35 layers alias …, 72.0 MB not allocated`)은 세 구성 모두 동일했다. window-ring 라인은 `ml_logi`라 stderr가 아니라 `~/.cache/nntrainer/logs/log_nntrainer_*.out`에 남는다.

### 절감 수치

| 비교 | time -v 중앙값 차이 | `[REQUEST]` peak_memory_kb 중앙값 차이 | 물리 평면 회계(오너 15개, K+V fp16) |
|---|---|---|---|
| A−B | 18,068 KB = 17.6 MiB (전체 RSS의 0.52%) | 17.1 MiB | sliding 오너 12 × 1,536행 절감 = 18.0 MiB |
| A−C | 30,588 KB = 29.9 MiB (0.88%) | 33.7 MiB | 18.0 + full 오너 18.0 − 플래시 스테이징 2.1 = 33.9 MiB |
| B−C | 12,480 KB = 12.2 MiB | 16.5 MiB | 18.0 − 2.1 = 15.9 MiB |

전체 RSS의 대부분(≈3.3 GB)은 가중치 매핑이라 %로는 작지만, 절감량이 물리 평면 회계와 같은 자리에서 나온다는 두 갈래 확인(절감량이 상한 수식으로 계산한 평면 차이와 일치, `peak_memory_kb` 독립 계측과 방향·크기 일치)이 이 측정의 핵심 산출이다. 절감은 로드+프리필 시점의 구조적 차이로 거의 정해지고, 디코드 구간의 추가 성장은 세 구성 모두 작다(아래 프루브).

FSU 회계(C, 3회 모두 바이트 단위 동일): `lookahead=1 direct_io=1 peak_kv_bytes=2179072 read_bytes=8126791680 write_bytes=25411584 reused_buffers=7172 parallel_reads=305 async_writes=0 staging_bytes=0 retained_bytes=0 deferred_writes=0`. `retained_bytes=0`은 예산을 설정하지 않은 순수 스트리밍 구성 때문이다(§10의 24 MiB 예산 구성과 다른 설정). write 24.2 MB는 4 KB pwrite 그레인의 예상치(디코드 스텝당 레이어당 K·V 2×4 KB × 3레이어 × 1,024스텝 ≈ 24 MB)와 일치하고, read 7.57 GiB는 스텝별 가시행 최소 합계(≈3.2 GiB)의 2.35×로 §8의 읽기 증폭과 같은 성격이다.

### 구조분해 프루브 (num_to_generate=1)

RSS 차이가 언제 생기는지 보려고 같은 프롬프트에 생성 1토큰만 받는 복제 구성 세 개를 돌렸다:

| 구성 | gen=1 KB | gen=1024 중앙값 KB | 디코드 구간 증가 |
|---|---|---|---|
| A | 3,461,128 | 3,469,980 | +8.6 MiB |
| B | 3,439,008 | 3,451,912 | +12.6 MiB |
| C | 3,434,052 | 3,439,392 | +5.2 MiB |

프루브 시점의 A−B 격차 21.6 MiB ≈ 평면 절감 18.0 MiB + 프리필 스크래치 차이라는 계산과 부합한다. 즉 RSS 절감의 대부분은 할당 시점에 확정되는 평면 높이 차이고, 디코드가 길어져도 세 구성 모두 13 MiB 이내로 수렴한다.

### 의미 변화 확인 — A = B

§11의 의미 변화 단락이 주장한 "창 밖 접두 행을 버려도 sliding 레이어의 출력이 같다"를 실팩 규모에서 처음 확인했다. 풀높이 상주(A)와 창 상한 상주(B)가 1,024개 생성 토큰 전체에서 비트 일치한다 — do_sample=true(temperature 1.0) 조건의 일치라 확률 벡터까지 실질적으로 동일하다는 뜻이다. 여기에 C(혼재)까지 3-way 일치이므로 compact 이동과 flash 스트리밍이 끼어든 구경도 같은 결과를 낸다. sliding 의미 보존의 증거는 이제 tiny fixture(§11)와 실팩 양쪽에 있다.

### 유의점

1. C의 decode +13.9 ms/tok(+27.8%)는 알려진 비용의 실팩 확인이다. full 오너 3개를 매 스텝 ext4에서 읽는 비용과, fc2fec519가 기록한 "flash 레이어 뒤에 windowed 레이어가 오면 lookahead 1겹침을 잃는다"는 구조(gemma4는 4 sliding 뒤 full 패턴이라 매 조에서 겹침이 끊긴다)가 겹친다. B는 A와 decode 차이가 없다(50.04 vs 49.93 ms/tok) — windowed 상한 자체는 속도를 바꾸지 않는다.
2. 상한 산식은 설정값(`init_seq_len`, `num_to_generate`) 기준이지 실제 프롬프트 길이가 아니다. 이 pack의 본래 용도(2,560토큰 프롬프트)로는 상한이 full height라 절감이 없다. 이 기능이 발동하는 프로파일은 "짧은 프롬프트 예산 + 긴 생성" — 즉 `max(init_seq_len, num_to_generate) < max_seq − window`일 때다.
3. FSU 스트리밍의 절감이 full 레이어 3개분(18.0 MiB)으로 작은 건 이 모델의 KV 오너 분해(sliding 12 : full 3) 탓이다. sliding 비중이 더 큰 모델이나 sliding_window·컨텍스트가 큰 설정에서는 windowed 절감과 스트리밍 절감이 모두 커진다.
4. 원본 pack 무수정: 프리미스 확인 실행(원본 + `NNTR_KV_WINDOW_CAP=1`)은 읽기만 했고, 이 실행이 내놓은 경고(`chat_input`에 채팅 템플릿이 없어 `sample_input`으로 폴백)는 상한 확인과 무관하다 — 상한은 설정값 산식이라 프롬프트와 무관하게 full height로 확인됐다.
5. 로그·pack은 tmpdir과 사용자 캐시에만 있다(`/tmp/kvwin-g4-real*`, `~/.cache/nntrainer/logs`). 재현 절차는 이 절의 표와 명령(`nntr_causallm <pack> --requests <requests.json>`)으로 충분하다.
