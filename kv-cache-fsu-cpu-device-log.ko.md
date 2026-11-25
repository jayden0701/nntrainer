# KV FSU CPU 구현·기기 테스트 기록

`kv-cache-fsu-analysis.ko.md`의 분석과 `kv-cache-fsu-htp-handoff.ko.md`의 순서를 따라 실제 작업을 진행하면서 남기는 기록이다. 기준 커밋은 `13cbb99ea161f636dbefd610c1bd435de641e89a`이며 이 기록 자체도 같은 트리의 `kv-cache-fsu` 브랜치에 커밋으로 남긴다. S26U가 준비되기 전이므로 문서의 3단계(CPU 계측·튜닝)까지를 먼저 진행하고, 기기 실험도 S26U가 아닌 다른 Android 기기로 CPU 부분을 수행한다.

## 0. 진행 상황 요약

| 단계 | 내용 | 상태 |
|---|---|---|
| 기준 재현 | 호스트에서 3단계 문서의 기준 검사(unittest + tiny fixture CLI A/B)를 재현 | 완료 |
| 3.1 계측 | `KVFlashStorage`에 단계별 대기·전송 시간 계측 추가 | 완료 |
| 기기 기준값 | 연결된 Android 기기에서 qwen3 0.6b resident/FSU A/B | 완료 |
| 기기 계측 | 3.1 계측을 켠 기기 실행으로 읽기·대기 비용 확보 | 완료 |

## 1. 작업 환경

| 항목 | 값 |
|---|---|
| 호스트 | Ubuntu 22.04 x86-64, GCC 11.4, 시스템 페이지 4096 B, 32 스레드 |
| 기준 트리 | `kv-cache-fsu` @ `13cbb99ea`, 빌드 디렉터리 `build-fsu` |
| 빌드 설정 | `-Denable-transformer=true -Denable-fp16=true -Denable-test=true -Dwerror=false` |
| 기기 | SM-S947N (serial `R3KYC06DR6R`), Android 16, erd9965, arm64-v8a, 페이지 4096 B, /data 여유 136 GB |
| 기기용 모델 | `~/TEST_MODELS/qwen3-0.6b` (Q4_0/FP32, embdq6k, 375 MB). gemma4 26B CPU 기기 실행은 현 기기 예산을 넘으므로 제외 |
| 비고 | `~/TEST_MODELS`에 qwen3 1.7b nntr pack은 없다. HF 원본(`/home/jayden/turboquant-exp/models/Qwen3-1.7B`)만 있어 0.6b의 A/B 결과로 KV 부하 추세를 먼저 확정하고, 필요해지면 변환을 추가한다 |

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
