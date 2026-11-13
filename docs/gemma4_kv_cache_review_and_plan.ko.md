# Gemma4 HTP KV cache 검토와 CLINE 수정 계획

검토 기준은 `jayden0701/nntrainer`의 `pr/4481`,
`a185c0b69b242e7510c1d807a473510e585090b2`이다. 2026-11-13에 fetch와
fast-forward pull을 완료했다. 로컬 검토 브랜치는 `codex/gemma4-kv-review`다.
최근 jayden0701의 KV16 도입, 비동기화, encode 최적화를 중심으로 기존
Gemma4 prefix capture → DRAM/Flash → restore → seed 연결을 검토했다.
다른 로컬 브랜치의 sparse reuse/TurboQuant 작업은 이 브랜치에 합치지 않았다.

**결론: codec의 비트 보존 결과는 양호하지만, 비동기 구현과 장치 검증에는
먼저 고쳐야 할 결함이 있다. 추가 SIMD 최적화보다 경합 수정과 실제 restore
검증을 먼저 수행한다.** 호스트에서 데이터 경합을 재현했고, 기존 full-scope
비교기가 검사하지 않은 결과까지 PASS로 보이게 하는 공백을 확인했다.

이 문서는 수정 계획이다. production 코드와 기존 테스트는 수정하지 않았다.
NPU 추론, Android/Hexagon 빌드, 기기 성능은 이번 PC 검토에서 실행하지 않았다.
아래의 기기 결과 분석은 저장소에 이미 있는 증거를 다시 읽은 결과다.

TTFT/decode의 구조적 개선 후보와 성능 지표의 단위 오류는
[Gemma4 HTP 속도 검토와 실행 계획](gemma4_htp_latency_review.ko.md)에 추가했다.
CLINE은 두 문서를 함께 읽고 정확성 수정부터 진행한다.

## 1. 유지해야 할 설계와 검토 범위

현재 경로를 다음처럼 구분한다.

1. 활성 DSP attention KV: 기존 paged INT8와 Q2/U16 경로.
2. host checkpoint: 원래 prefill call 경계의 FP16 K/V와 calibration.
3. KV16: 2번 중 eligible global layer의 보관 형식만 압축.
4. restore: 원래 FP16 bits를 복원한 뒤 기존 seed/attention 경로 사용.

`ATTENTION_K_EQ_V`가 있어도 K와 V 전체가 같은 것은 아니다. V 원본,
rotary K 원본, non-rotary K의 predictor XOR residual을 보관해야 한다.
predictor 오차가 남아도 residual이 원본을 복원한다. residual 삭제, float
연산으로 predictor 교체, BF16 재해석은 이 최적화 범위에 포함하지 않는다.

압축은 token prefix의 재사용 가능성을 확대하지 않는다. whole-call calibration
때문에 원래 call 내부를 잘라 재사용하면 계산이 달라질 수 있다. 64-row codec
page와 4096-row prefill 경계는 다른 단위다. 기존 call plan, sliding base,
dropped maxima, runtime gamma, rotary map, pending token 계약을 유지한다.

좋은 점도 보존한다.

- optimized encode와 원래 `5ededcb00`의 container가 이번 A/B에서 일치했다.
- decoder가 CRC, extent, index, sparse bitmap을 검증한다.
- 이득 없는 페이지와 encode 실패는 RAW를 유지한다.
- `materializePrefix()`는 전체 페이지를 확인한 후 live cache를 쓴다.
- raw v1과 mixed v2 Flash를 함께 읽고, 모델 gamma/map 불일치를 거부한다.

주요 파일:
[codec](../Applications/CausalLM/kv16_lossless.cpp),
[KV manager](../Applications/CausalLM/kv_cache_manager.cpp),
[page/API](../Applications/CausalLM/kv_cache_manager.h),
[CausalLM](../Applications/CausalLM/models/causal_lm.cpp),
[Gemma4 binding](../Applications/CausalLM/models/gemma4/gemma4_htp_causallm.cpp),
[Flash](../Applications/CausalLM/prefix_flash_store.cpp),
[device evidence collector](../test/htp/host/kv16_evidence_check.cpp),
[evidence comparator](../tools/kv16/compare_device_evidence.py).
이하 줄 번호는 검토 기준 SHA의 번호다.

## 2. 이번 PC 검증 결과

원본 로그와 재현 스크립트는
[검토 증거 디렉터리](validation/gemma4-kv-review-20261113/)에 있다.
큰 실행 파일과 중간 산출물은 `build/gemma4-kv-review/`에 두었다.

| 검사 | 관측 결과 | 의미 |
| --- | --- | --- |
| Python oracle 대 C++ codec, ASan/UBSan | PASS: predictor 871,968쌍, wire 41개, malformed 745개 | codec 원본 복원 및 wire 호환 |
| `5ededcb00` 대 HEAD encode A/B | PASS: 2,048 cases, mismatch 0 | 검사한 입력의 container byte exact |
| predictor A/B | PASS: 65,536 FP16 × 4,096 gamma = 268,435,456쌍 | FP32 gamma 전체 공간의 전수 검사는 아님 |
| prefix/Flash/baked 통합 | 5 binaries PASS, ASan/UBSan 진단 없음 | 새 storage 코드와 checker를 instrument; engine 전체 sanitizer 빌드는 아님 |
| 기존 evidence Python 테스트 | 15개 PASS | 아래 검증 공백을 기존 테스트가 놓침 |
| HTP device-KV host 검사, ASan/UBSan | 4개 실행 대상 PASS | manager, KV 변환, snapshot, paged attention의 호스트 emulation |
| capture 직후 reset, ThreadSanitizer | **FAIL: data race, exit 66** | R1 실제 재현 |
| 동일 재현 + 격리된 최소 수정 대조군 | exit 0, TSan 진단 없음 | 수정 방향 확인; production 적용 완료를 뜻하지 않음 |
| 저장된 최신 full evidence 재검사 | 원본 PASS; final query 결과 변조 후에도 PASS | R2 비교 범위 결함 재현 |

ASan/UBSan 통합 대상은 `kv16_prefix_check`, `prefill_prefix_check`,
`prefix_flash_check`, `kv16_flash_check`, `baked_prefix_check`다. 실제 검토 소스를
컴파일하고 기존 matching host engine을 연결했다. runner의 source/ABI 일치
검사를 통과했다. 엔진 경로는
`/home/jayden/nntrainer-kv16-npu-core/build/kv16-core`다.

부수적인 encode microbenchmark는 old 43.64ms, new 35.09ms였다. 다른 검사를
함께 실행하던 x86 측정이므로 기기의 2.6배 개선을 재확인한 수치나 성능 gate로
사용하지 않는다. 정확성 A/B 결과와 시간 측정 결과를 분리한다.

## 3. 발견 사항

### R1 P1 비동기 encoder와 reset 사이의 실제 데이터 경합

**확정, TSan 재현.** `kv_cache_manager.cpp:1016`에서 shared-view page를
`prefix_views_`에 등록한 뒤, 1027–1029에서 freeze하고 worker에 넘긴다.
등록 목록에는 이미 owned RAW가 된 page가 그대로 남는다.

워커는 861에서 `page.key_view = page.value_view = nullptr`를 쓴다.
동시에 `reset():684–685` → `freezePrefixViews():1125–1131` →
`Page::freeze()`의 `kv_cache_manager.h:246`은 `key_view`를 읽는다.
이 두 접근 사이에 lock이나 drain이 없다. 이미 null인 값을 다시 쓰더라도
C++ data race다. 모델 오출력이나 실제 기기 crash까지 재현했다는 뜻은 아니다.

재현은 fake shared allocator를 사용해 4096-token prefix를 capture하고,
drain 없이 바로 `reset()`한 후 snapshot을 읽는 정상적인 manager 호출 순서다.
기존 owner-release 검사는 capture 다음에 drain하므로 이 interleaving을 가린다.
`compact()`도 view freezer를 쓰지만 codec은 global non-windowed layer만
지원하므로 현재 sliding compaction에서 같은 경합이 발생한다고 확대하지 않는다.

**권장 최소 수정:** codec으로 즉시 owned가 될 page를 `prefix_views_`에 넣지
않는다. 비압축 borrowed page의 등록과 freeze는 그대로 유지한다. 격리 복사본에
`if (!codec)` 조건만 넣은 대조 실험은 동일 TSan 재현에서 exit 0이었다.
최종 구현은 `will_encode = codec && element_bytes == 2u`를 한 번 계산하고
등록과 enqueue 양쪽에서 같은 조건을 사용하면 조건 불일치를 피할 수 있다.

모든 read에 global drain을 추가하는 방법은 간단하지만 reset/요청 전환을
불필요하게 막을 수 있다. `key_view` 하나만 atomic으로 바꿔도 encoded/vector/
owner 필드의 publication 문제는 해결되지 않는다.

### R2 P1 full evidence가 실제 prefix restore와 마지막 query를 검증하지 않음

**확정, 저장된 artifact와 mutation으로 재현.** 최신
`encode-opt/{baseline,codec}-trace-clz3.jsonl.gz`에는 각각 capture 3건과
restore **0건**이 있다. endpoint는 2079, 4096, 4607이다. 대응 evidence의
events는 `run-1`, `run-2` seed 두 건뿐이다.

collector `kv16_evidence_check.cpp:369–385`는 calibration trace가 있는 run만
`events`에 넣는다. 마지막 query는 408에서 `reset=false`인 continuation으로
실행한다. comparator는 `runs`를 비교하지 않는다. candidate의 마지막
`runs[-1].ids`를 `[999999]`, position을 원래 값+123으로 바꿔도 full PASS다.

이 결과는 fresh capture와 그 뒤 resident decode의 회귀 증거로는 유용하다.
하지만 **compressed host prefix를 선택·복원한 뒤 suffix를 계산하는 경로의
full 검증은 아니다.** 별도 sessions/checkpoint 자료에는 restore 증거가 있다.
그 자료까지 무효라는 뜻도, 저장소의 모든 NPU 검증이 실패했다는 뜻도 아니다.

수정 방향은 모든 요청의 token/position/logits 관측을 보존하고, 시나리오가
요구하는 restore event와 실제 reused token 수를 필수로 검사하는 것이다.
`compressed_pages > 0`만으로 restore 사용을 증명하지 않는다.

### R3 P1 evidence 필드의 관측 지점과 build identity가 부정확함

**확정, 코드/manifest 확인.** 다음 세 문제를 R2와 별도 작업으로 고친다.

- `kv16_evidence_check.cpp:361`의 `call_plan_sha256`은 실제 call plan이 아니라
  prompt 문자열과 총 prefill token 수를 hash한다. call의 분할, from/to,
  activation plane이 달라져도 같은 값이 될 수 있다.
- 380–384는 run 종료 시점의 DSP dump hash 하나를 K, V, seed KV, seed scales,
  decode state 다섯 필드에 넣는다. 완료 dump의 equality는 유효하지만 **seed
  직후** state나 **host FP16 checkpoint**의 직접 관측을 대신하지 못한다.
  원래 [인계서](kv16_npu_handoff.ko.md)의 관측 지점 구분과도 다르다.
- 최신 양쪽 manifest의 `code_revision`은 실제로 `unknown`이며 comparator는
  이를 PASS로 인정한다. `arm_sha256`은 `libnntrainer.so`만 hash한다.
  app/codec/collector가 들어가는 실행 파일과 `libcausallm.so` 등의 실제 빌드
  identity를 포괄하지 못한다. static linking이면 실행 파일 hash가 특히 중요하다.

관측하지 않은 필드에 다른 시점의 hash를 복사하지 않는다. 현재 dump는
`post_run_state`라는 명시적 범위로 보존한다. seed 관측을 구현하기 전에는 seed
scope를 INCOMPLETE로 남긴다. 같은 전체 dump hash를 여러 필드에 쓴 행위 자체보다
**다른 시점·다른 데이터처럼 표시한 것**이 문제다.

### R4 P2 비동기화가 모든 경로에서 압축 지연을 숨기지는 않음

**코드상 확정, 기기 지연 크기는 미측정.** `capturePrefix():942`는 이전 큐
전체를 drain한다. `captureDevicePrefill():1971–1972`는 Flash archive와 baked
writer에 넘기기 전에 현재 압축도 drain한다. 따라서 긴 요청의 다음 checkpoint나
Flash 저장에서는 encode가 여전히 foreground 대기 비용이 될 수 있다.
`getDeviceKVPrefixBytes():1154`도 조회 전에 drain한다.

TRACE는 1847에서 drain하고 decodeCopy까지 하므로 TRACE=1인 기능 검사는
실제 비동기 interleaving과 성능을 대표하지 않는다. 요청 직후 manager를 교체하는
`beginIndependentHtpPrompt()`에서는 pool 소멸자의 join이 남은 작업을 기다린다.

우선 `capture_copy_ms`, `encode_cpu_ms`, `codec_wait_ms`와 wait reason을 측정한다.
기존 drain을 근거 없이 제거하면 Flash worker와 encoder가 같은 page를 만진다.
Flash 비동기 pipeline 변경은 R1의 최소 수정과 별도 단계로 다룬다.

### R5 P2 queue와 임시 메모리의 한도가 없고 실제 절감량이 불명확함

**설계상 위험, 이 PC에서 OOM을 재현하지는 않음.** pool의 `queue_`에는 byte
budget, peak counter, cancellation이 없다. encoder는 최대 4개다. 한 capture가
긴 global prefix 전체를 freeze/enqueue할 수 있다. 다음 capture가 drain하므로
무한히 여러 capture가 누적된다고 말할 수는 없지만, 한 capture의 RAW staging과
4-worker scratch만으로도 메모리 peak가 커질 수 있다. prefix clear 후에도 task가
shared_ptr를 잡고 있으면 작업과 memory retention이 끝날 때까지 남는다.

live RPC slab, owned RAW, block 후보/최종 encoded, VERIFY scratch가 겹친다.
codec이 실패/no-gain이면 owned RAW가 계속 남는다. `failures_`는 no-gain과 예외를
같이 세고 외부에 노출되지 않아 VERIFY mismatch 여부를 checker가 확실히 요구할
수도 없다. 다른 page가 압축되면 `compressed_pages > 0`은 여전히 만족한다.

현재 trace의 `raw_bytes/stored_bytes`는 이벤트별 합계다. 동일 공유 페이지가
capture/restore 여러 이벤트에 반복 포함될 수 있다. 기존 문서의 109,298,882-byte
절감은 6개 이벤트 합산의 payload 차이이며, 단일 시점의 unique retained memory나
RSS 감소로 사용하면 안 된다. `getDeviceKVPrefixBytes()`는 unique page/slab을
세지만 staging/scratch/allocator overhead 전체는 포함하지 않는다.

### R6 P2 restore가 같은 압축 page를 여러 번 검증하고 해제함

**코드상 확정, 성능 효과는 프로파일 필요.** codec ON 정상 경로에서:

1. `validateDevicePrefixCodec()`가 매 page를 inspect하고 같은 layer gamma/map도
   반복 구성한다.
2. `restoreDevicePrefill():2047`에서 decodeCopy하고 즉시 버린다.
3. host cache가 이미 있으면 2064에서 materialize하고 즉시 버린다.
4. 실제 bind 단계 2305의 restorePrefix가 다시 materialize한다.

즉 compressed page의 full decode가 최소 2회, cache가 있으면 3회다. inspect와
decode 모두 parse/CRC를 하므로 pass도 중복된다. Flash load 및 TRACE는 추가다.
이 변경은 안전성 검증을 제거해서 고치지 않는다. 요청에만 살아 있는 prepared
snapshot을 만들어 validation 결과와 decoded RAW를 실제 commit까지 전달한다.

capture에서도 기존 full page를 재사용할 때마다 `capturePrefix():993`의 inspect가
전체 encoded bytes를 다시 읽는다. 긴 prefix를 여러 checkpoint로 확장하면 이
검사 비용도 누적된다. 신뢰할 수 있는 in-memory finalized page와 파일에서 새로
읽은 page를 구분한 뒤에만 검증 결과 재사용을 고려한다. Flash 입력 검증을
생략하는 최적화로 바꾸지 않는다.

## 4. CLINE 실행 순서

아래 작업은 **한 카드씩, 하나의 주제로 commit**한다. A→B→C를 완료한 뒤
NPU correctness gate를 실행한다. D와 E는 그 다음 성능/메모리 개선이다.
F는 별도 실험이다. 한 번에 비동기·codec·Flash format을 재설계하지 않는다.

### 작업 A view 경합만 수정

대상: `kv_cache_manager.cpp::capturePrefix`, `kv16_prefix_check.cpp`.

1. 위 R1 재현을 원래 SHA에서 실행하고 exit 66/TSan stack을 보존한다.
2. codec 대상 page는 shared-view registry에 등록하지 않도록 수정한다.
3. 기존 non-codec borrowed-view freeze와 owner retention 동작을 유지한다.
4. capture→즉시 reset을 반복하는 회귀 검사를 추가한다. 테스트는 압축 완료를
   기다린 다음 원래 K/V bytes와 전체 snapshot을 비교해야 한다.
5. capture→move→reset, capture→manager 파괴 후 snapshot 복원,
   codec 없는 borrowed view reset도 검사한다. manager 자체의 모든 API를
   여러 foreground thread에서 동시에 호출하는 지원 범위를 새로 만들지는 않는다.

완료 기준: 새 재현이 수정 전 실패/수정 후 TSan 진단 없음, 5개 기존 storage
검사 PASS, 비압축 snapshot bytes 보존. worker를 동기 실행으로 바꿔 통과시키거나
테스트에서 reset 전에 drain을 넣어 경합을 숨기지 않는다.

권장 commit: `[CausalLM] Keep codec pages out of borrowed prefix views`.

### 작업 B evidence가 검사할 요청을 빠뜨리지 않게 수정

대상: `kv16_evidence_check.cpp`, `compare_device_evidence.py`,
`test_device_evidence.py`. inference 수식은 수정하지 않는다.

1. 먼저 기존 최신 JSON을 읽어 final run IDs/position만 바꾸는 regression test를
   추가한다. 수정 후 비교 결과는 FAIL이어야 한다.
2. `request_id`, `phase`, token IDs, position, 첫 full-logit row를 모든 run에서
   저장한다. calibration 유무로 요청 자체를 events에서 제외하지 않는다.
3. fixture에 요구 시나리오와 expected prefix endpoint를 적는다. actual reused
   tokens 및 restore source(DRAM/Flash/baked)를 관측하고 필수 phase로 검사한다.
4. 예상 restore가 없으면 INCOMPLETE, 관측한 내용이 다르면 FAIL로 분리한다.
5. 실제 original call의 `from/to/plane/chunk`와 순서를 수집해 hash한다.
   restore된 call과 새로 실행한 suffix call은 phase를 구분해 비교한다.
6. capture-only 회귀는 별도 scope로 유지한다. 이를 restore-full이라고 부르지 않는다.

완료 기준: query 변조→FAIL, restore event 삭제→INCOMPLETE,
동일 prompt/총길이에서 call 분할만 변경→FAIL, 사용되지 않은 codec→INCOMPLETE.
기존 15개 Python test도 PASS. `hardware_validated`나 hash 필드를 하드코딩해서
누락을 메우지 않는다.

### 작업 C 관측 지점과 빌드 identity를 고정

대상: evidence collector/comparator, 필요한 디버그 export hook만.

1. manifest schema를 명시적으로 갱신한다. 기존 v1 결과는 historical 범위로
   남기고 새 full gate에 자동 승격하지 않는다.
2. source SHA, dirty patch SHA256, checker 실행 파일, 실제 linked CausalLM/engine
   library, DSP skeleton, config/model/tokenizer/fixture hash를 기록한다.
   `unknown`, 빈 source SHA는 새 gate에서 INCOMPLETE다.
3. host FP16 checkpoint hash와 post-run DSP state hash의 이름·dtype·shape·시점을
   분리한다. SHA256으로 FNV 문자열을 감싸 원본 SHA256처럼 쓰지 않는다.
4. seed 직후 검증은 기존 `hvx_attn_m1_snapshot_bytes/export()`를 출발점으로
   현재 경로에서 모든 layer의 seed 완료 후, decode가 state를 바꾸기 전 경계를
   먼저 찾는다. RPC hook 위치를 설명한 뒤 구현한다. host seed-input trace와
   seed 이후 DSP INT8 readback을 별도 항목으로 둔다.
5. run-unique cookie를 제외할 때는 snapshot version/extent를 검증한 후 정확한
   framing 필드만 제외한다. 다른 모든 numerical bytes는 비교한다.
6. logits 위치를 pre/post softcap 및 processor 적용 순서와 함께 고정한다.
   full logits 전송은 correctness 전용이며 timing run에서 끈다.

완료 기준: 각 필드에 관측 시점·dtype·shape·position과 실제 artifact가 있고,
변조/누락 negative test가 각각 FAIL/INCOMPLETE다. 실제 seed readback이 없으면
post-run 검증 성공과 seed 미검증을 함께 기록한다.

### 작업 D counters와 queue byte budget

대상: PrefixCodecPool, CausalLM prefix memory/statistics reporting.

먼저 counters만 추가한 commit을 만든다. `queued_raw_bytes`,
`inflight_raw_bytes`, `peak_codec_bytes`의 정의를 정하고, `encoded_pages`,
`no_gain_pages`, `encode_errors`, `verify_mismatches`, `budget_skips`,
`cancelled_pages`를 분리한다. snapshot trace event 합산과 unique retained bytes를
다른 이름으로 출력한다. `codec_wait_ms`는 capture/restore/archive/stats/shutdown
이유를 구분한다. 정확한 allocator peak가 아니면 추정치로 표시한다.

그 다음 budget commit을 만든다.

1. enqueue 직전이 아니라 **owned copy를 만들기 전** RAW staging bytes를 reserve.
2. reserve 실패 시 원래 RAW/shared-view capture 경로로 돌아간다. snapshot
   자체를 깨뜨리거나 worker 완료를 foreground에서 기다리지 않는다.
3. 모든 성공/no-gain/예외/취소 경로에서 reservation을 정확히 한 번 반환한다.
4. worker scratch 상한은 별도 항목으로 계산한다. queue 예산 하나로 전체 RSS를
   제한한다고 주장하지 않는다. worker 수 1/2/4를 선택할 수 있게 한다.
5. request 중 `clearDeviceKVPrefixes()`와 queued task 정리를 검사한다.
   실행 중 task의 page를 강제로 free하지 않는다. cancellation을 넣을 경우
   page/task 상태와 reservation 소유자를 먼저 정의한다.

완료 기준: 작은 budget에서도 원본 bytes와 restore 결과가 같고, queue 예약이
정한 한도를 넘지 않으며 종료 후 0. VERIFY fault injection 시 RAW fallback은
정상이어도 correctness checker 결과는 FAIL이어야 한다. no-gain은 실패가 아니다.

### 작업 E restore의 중복 decode 제거

대상: `restoreDevicePrefill`, `materializePrefix`, 실제 bind/restore 경계.

1. 먼저 page별 decode 호출 수와 decode/CRC/binding 시간을 측정한다.
2. `PreparedPrefixRestore`처럼 **한 요청만 소유하는** 결과에 validated geometry,
   raw snapshot, calibration 및 목표 position을 함께 보관한다.
3. 모든 검증과 allocation을 끝낸 뒤 live KV와 session metadata를 commit한다.
   페이지를 한 장씩 decode해 곧바로 live cache에 쓰면 후반 손상 시 부분 restore다.
4. 앞에서 만든 RAW를 bind에 넘겨 재사용한다. 영구 prefix index에 decoded RAW를
   캐시하지 않는다. 그렇게 하면 압축 memory 이득을 잃는다.
5. 실패/다른 prefix 선택/요청 종료/reset에서 prepared state를 해제한다.
6. 같은 layer의 expected gamma/map은 page loop 밖에서 한 번 계산한다.

완료 기준: codec ON 선택된 page의 준비 과정 full decode 1회, 실제 copy 단계
추가 decode 0회. 마지막 page CRC 손상, gamma mismatch, allocation 실패에서
live KV/position/calibration은 그대로. 정상 suffix logits/token/seed gate PASS.
중간 OOM이나 allocation 실패에 대한 전체 추론 복구 가능성은 기존 계약과 구분한다.

### 작업 F Flash encode 대기 축소는 별도 실험

R4 측정에서 archive wait가 실제 병목일 때만 시작한다. 기존 drain만 삭제하지
않는다. 현재 FlashStore API가 immutable finalized snapshot을 받는 계약을 먼저
명시한다. 단순한 첫 후보는 **owned RAW snapshot을 archive worker가 전담해서
encode→serialize**하는 방식이다. foreground와 동일 page를 동시에 갱신하지
않도록 in-flight load가 읽는 RAW와 encoder output을 분리한다.

staging 예산은 압축 완료 전 RAW와 scratch, file 예산은 실제 최종 bytes/framing을
구분한다. `canStage(prefixPayloadBytes)`의 보수적 사전 거부와 최종 submit의 실제
encoded bytes 차이를 검사한다. file byte 이득으로 RAW staging 한도를 무시하면
안 된다. budget full이면 기존 정책대로 그 경계 백업을 skip한다.

완료 기준: in-flight/ready load, CRC 오류, failed write, LRU/clear/destruction의
기존 계약 PASS. NPU 8K/12K 이상 요청에서 archive foreground wait 감소와
memory peak를 함께 측정. 이 작업은 A–E와 독립 commit/실험으로 유지한다.

## 5. NPU에서 반드시 실행할 correctness matrix

같은 binary/model/config에서 별도 process로 codec OFF와 ON을 실행한다.
TRACE/VERIFY=1 검사는 numerical correctness 용도다. A의 비동기 수명 검사는
TRACE=0에서도 별도 stress해야 한다. TRACE가 매 capture를 drain하기 때문이다.
기존 config를 덮어쓰지 말고 별도 pack/config copy를 사용한다. 기존 기록은 원래
config를 백업하지 못했고 크기만 같은 pf2k 파일로 복원했다. 같은 byte 수는
같은 config의 증거가 아니다.

| 시나리오 | 반드시 관측할 것 | 비교 기준 |
| --- | --- | --- |
| cold capture 후 decode | eligible compressed page > 0 | checkpoint bits/calibration, logits, IDs, position |
| 동일 prompt를 독립 요청으로 재실행 | DRAM restore endpoint와 reused tokens | codec OFF 동일 재사용 경로와 bit exact |
| 4608-token A에서 4096 공유 B로 분기 | 4096 restore + suffix 원래 call plan | OFF/ON, 그리고 가능한 uncached reference |
| call 내부에서만 공통인 prefix | 기존 계약대로 miss | 재사용을 억지로 허용하지 않음 |
| 4096 경계 주변 prompt | prefill row 기준 4095/4096/4097 등, M1 tail | seed 횟수/position/pending token 포함 |
| 8704-token source의 과거 4096 Flash 복원 | 실제 file hit, in-flight hit 별도 | FP16/calibration, seed readback, logits/IDs |
| 새 process의 baked v1/v2, codec OFF/ON | 모델 identity 검증 후 load/restore | 동일 원본 결과 |
| 테스트 전용 late-page 손상 | live state 변경 전 거부 | 이전 compatible boundary 또는 recompute |
| session select/fork/reset/LRU/clear, 짧은 decode 뒤 즉시 다음 요청 | queued task 생존/정리, source snapshot 보존 | crash/race 없이 원본 bytes/결과 유지 |
| codec budget full/no-gain/encode 실패 | RAW fallback와 각 counter | 성능 기능 실패가 numerical 결과를 변경하지 않음 |

token 길이는 사람이 입력한 문자열 길이가 아니라 실제 tokenizer 결과로 확인한다.
`skip_prefill`의 마지막 token 처리 때문에 prompt length와 prefill endpoint가
하나 다를 수 있다. fixture에는 두 값을 모두 기록한다. call 내부 crop을 miss로
기대하는 test와 compression page tail test를 혼동하지 않는다.

host checkpoint/seed/logit 비교는 dtype별 원본 bits 기준이다. greedy 결과 몇 개가
같다는 이유로 seed나 logits equality를 추정하지 않는다. 수정 전부터 uncached
reference와 다른 numerical route가 있으면 먼저 이유를 확인하고 baseline 비교를
분리한다. 새로 tolerance를 크게 만들어 통과시키지 않는다.

## 6. 이후 최적화의 순위와 판단 기준

**1순위는 실제 retained memory와 wait 비용 측정이다.** 기본 OFF 유지 여부는
저장 이벤트 합계가 아니라 long-resident prefix의 unique bytes와 RPC slab,
heap/RSS peak, request latency로 판단한다. cold capture, steady decode,
idle/drain 완료, restore peak, prefix clear 완료의 다섯 시점을 기록한다.

**2순위는 중복 decode 제거와 bounded scheduling이다.** x86 encode loop가 빨라져도
HTP 모델 TTFT 개선으로 직결되지 않는다. nice 19 작업이 다른 부하에 밀리면 drain
지연으로 다시 나타날 수 있다. worker 1/2/4와 예산을 같은 fixture에서 비교한다.

**3순위는 page metadata와 allocation 비용이다.** D=512, heads=2,
rotary=128, rows=64, block_rows=16이면 page마다 header 52 + gamma 2048 +
rotary 512 + index 128 = **2740 bytes**다. RAW K+V 262144 bytes의 약 1.05%다.
gamma/map 공유는 이 정도의 직접 wire 절감과 parse/allocation 절감을 가져올 후보다.
새 format과 binding ID가 필요하므로 v1 byte-exact optimization과 별도 작업으로
둔다. 64-row ownership을 유지한 채 worker scratch/vector 재사용부터 측정한다.

**4순위는 SIMD predictor다.** R1–E와 NPU gate 이후에도 predictor가 병목일 때
NEON을 검토한다. 정수 double rounding, canonical NaN, zero/subnormal/overflow,
mode tie-break, CRC를 유지한다. ARM/x86 fallback을 두고 기존 A/B와 oracle을
전부 통과시킨다. 4096 gamma 표본은 전체 FP32 전수 검사가 아니므로 경계값
fixture도 유지한다. DSP kernel이나 activation quantization 변경과 묶지 않는다.

**장기 후보는 Flash global-page 중복 저장 제거다.** call별 calibration/sliding
window를 독립 metadata로 유지하면서 global page만 content-addressing하는 방식은
별도 설계가 필요하다. checksum 충돌, refcount/LRU, 원자적 파일 publication,
재시작 cleanup까지 다뤄야 하므로 낮은 모델의 첫 수정 과제로 주지 않는다.

성능 측정은 correctness PASS 뒤 TRACE/VERIFY/full-logits hook을 끈다. OFF/ON
순서를 교대하고 최소 3 paired 반복, 가능하면 5쌍을 확보한다. 각 표본의 thermal,
메모리, worker 수, 실제 hit endpoint를 함께 보존한다. sample median/range를 보고
작은 n으로 p95를 주장하지 않는다. TTFT뿐 아니라 다음 요청 시작·clear·destructor의
대기도 포함한 end-to-end 시간을 기록한다. 기본 ON 전환은 새 evidence와 실측
memory 이득이 확인된 이후의 별도 결정이다.

## 7. 재현 명령과 전달 규칙

Linux/WSL에서 저장소 root 기준:

```bash
python3 test/htp/host/run_kv16_codec_check.py --sanitize \
  --output build/kv16-review-new/codec
python3 test/htp/host/run_kv16_integration_checks.py \
  --engine-build /ABS/MATCHING_ENGINE_BUILD --sanitize \
  --json-include /ABS/DIR_WITH_JSON_HPP \
  --output build/kv16-review-new/integration
python3 -m unittest discover -s tools/kv16 -p 'test_*.py' -v
python3 docs/validation/gemma4-kv-review-20261113/reproduce.py ab \
  --output build/kv16-review-new/probes
python3 docs/validation/gemma4-kv-review-20261113/reproduce.py race \
  --engine-build /ABS/MATCHING_ENGINE_BUILD \
  --output build/kv16-review-new/probes
python3 docs/validation/gemma4-kv-review-20261113/audit_evidence.py \
  --output build/kv16-review-new/evidence-audit.json
SANITIZE=1 sh test/htp/host/run_device_kv_checks.sh
```

`race`의 기준 SHA 예상 결과는 TSan data race exit 66이다. 수정 후에는 exit 0과
진단 없음이 기대 결과다. 런타임이 TSan을 지원하지 않아 시작 실패한 것은 data
race 재현도 PASS도 아니다. `race-control`은 production을 수정하지 않고 temporary
복사본에 R1 최소 조건을 적용하는 대조 실험이다. 다음 담당자의 회귀 test를
대신하지 않는다. ASan과 TSan은 별도 binary로 실행한다.

Windows CRLF checkout의 shell script는 WSL `sh`에서 그대로 실행하면 실패할 수
있다. 이번 DSP host-check 실행은 build 폴더의 LF 복사본과 명시적 repo root를
사용했다. 원본 source는 바꾸지 않았다. HVX emulation/host stand-in 결과를 실제
HMX, FastRPC, DSP 기기 PASS로 기록하지 않는다.

기존 [HTP radix 계약](gemma4-htp-radix-prefix.md),
[NPU 인계서](kv16_npu_handoff.ko.md),
[encode A/B 가이드](kv16_encode_review_guide.ko.md)는 경로 설명에 활용하되,
기존 `full-scope PASS`는 이 문서의 R2/R3에 따라 범위를 제한한다.

commit은 AGENTS.md대로 component prefix, DCO `git commit -s`, 실제 작업 agent의
`Co-authored-by:` trailer를 사용한다. 변경 C/C++는 clang-format 14를 적용한다.
`subprojects/`는 편집하지 않는다. PR을 만들 때는 repository template의 commit별
details/Self evaluation/Sign-off와 Summary를 따른다.

CLINE에 첫 요청으로 전달할 문구:

> `docs/gemma4_kv_cache_review_and_plan.ko.md`와 AGENTS.md를 읽고 작업 A만
> 수행하라. 원래 SHA와 작업 tree 상태를 확인하고, R1 TSan 재현을 먼저 남겨라.
> codec 대상 page의 view 등록만 최소 수정하고, non-codec borrowed view 회귀를
> 검사하라. codec 수식, format, attention, calibration, call plan은 변경하지
> 마라. 수정 전/후 로그, 실행 명령, source SHA, 미실행 항목을 기록하라.
> 이 gate가 끝나기 전 작업 B 이후의 변경을 섞지 마라.
