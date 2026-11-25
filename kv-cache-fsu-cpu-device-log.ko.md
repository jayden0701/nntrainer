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
