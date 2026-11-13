# Gemma4 HTP TTFT·decode 속도 검토와 실행 계획

기준: `jayden0701/nntrainer pr/4481`, commit
`a185c0b69b242e7510c1d807a473510e585090b2`, 검토일 2026-11-13.
[KV cache 정확성 검토](gemma4_kv_cache_review_and_plan.ko.md)의 후속 문서다.
이번에는 KV16 codec 내부에 한정하지 않고 continuation, DSP token 실행,
MoE cache, prefix restore, 요청 초기화까지 살폈다. production 코드는 변경하지 않았다.

**TTFT에는 불필요한 LM head 실행과 prefix 중복 복원이, steady decode에는
MoE expert miss가 우선 검토 대상이다. 먼저 기존 TPOT 보고서의 단위 오류를
고쳐야 한다.** 아래 숫자는 저장된 기기 로그의 재분석이며, 이 PC에서 새로운
NPU 성능을 측정한 결과가 아니다. 예상 개선율을 실측 결과처럼 사용하지 않는다.

## 1. 측정부터 바로잡기 — P0

`Applications/CausalLM/models/causal_lm.cpp:3359–3368`에서
`decode_steady_ms`는 첫 token을 제외한 **token별 latency의 중앙값**이다.
`decode_steady_tokens`는 그 중앙값에 사용한 표본 수다. 둘을 다시 나누면 안 된다.

기존 `docs/validation/kv16-npu-device-20261111.json`의
`performance_no_trace.rows[].tpot_ms` 6개 행 전부에서 이중 나눗셈이 재현된다.
예를 들어 첫 baseline은 `57.547552 ms/token`, 표본 수 31인데 보고서에는
`57.547552 / 31 = 1.86 ms`로 기록되어 있다. 실제 값과 31배 차이가 난다.
기존 반복 측정 요약 등 다른 파생 표도 원본으로 다시 계산해야 한다.
이번 자동 검증에서 직접 대조한 범위는 위 6개 행이다.

최신 encode-opt 저장 자료의 올바른 값은 다음과 같다. OFF/ON은 KV16 설정이며
정확한 의미는 원본 설정과 함께 보관한다. 이 표 하나로 유의한 개선/회귀를
판정하지 않는다. paired 반복과 온도·클록 통제가 부족하다.

| Session / turn | OFF 중앙값 ms/token | ON 중앙값 ms/token |
|---|---:|---:|
| 0 / 0 | 61.870 | 64.730 |
| 1 / 0 | 62.945 | 62.139 |
| 0 / 1 | 57.211 | 57.121 |
| 1 / 1 | 59.306 | 59.429 |
| 0 / 2 | 56.224 | 56.155 |
| 1 / 2 | 58.133 | 58.620 |

`1000 / 중앙값(ms)`는 중앙값의 역수인 rate이며 전체 처리량과 같지 않다.
처리량은 실제 출력 token 수와 대응되는 실제 시간 구간으로 따로 계산한다.

재현 스크립트와 결과:
[recompute_metrics.py](validation/gemma4-latency-review-20261113/recompute_metrics.py),
[metrics-audit.json](validation/gemma4-latency-review-20261113/metrics-audit.json).
저장소 루트에서 다음 명령을 실행한다. Python 표준 라이브러리만 사용한다.

```sh
python docs/validation/gemma4-latency-review-20261113/recompute_metrics.py --output build/gemma4-latency-metrics.json
```

### TTFT 계측 경계도 고칠 것

- `Gemma4HtpMoECausalLM::run()`의 `beginIndependentHtpPrompt()`는
  `CausalLM::run()`의 `start_total`보다 앞선다. 반복 요청의 pool join,
  DSP KV 해제, host manager 재설정이 기존 TTFT에서 빠질 수 있다.
- session 선택/복원은 호출자에서 먼저 수행될 수 있다. benchmark의
  `switch_ms`를 숨기거나 TTFT에 중복 합산하지 않도록 API 경계를 정의한다.
- 현재 첫 출력 timestamp는 출력 처리와 prefix publication의 배치에 영향을
  받는다. 첫 ID 준비, 실제 callback 전달, 이후 capture 완료를 각각 기록한다.
  codec 작업을 다음 token으로 미룬 것만으로 개선이라고 보고하지 않는다.
- cold independent prefill, DRAM prefix hit, Flash prefix hit, 같은 DSP state를
  유지하는 resident continuation을 별도 시나리오로 측정한다.

권장 timeline은 `request_enter → reset/select → lookup/read → decode/validate →
bind/seed → prompt_consume → first_id_ready → first_callback → request_done`이다.
동시에 진행한 작업은 span으로 기록하고 중첩 시간을 합계에 중복 가산하지 않는다.
기존 필드를 유지한 채 `request_to_first_callback_ms` 등을 추가해 호환성을 지킨다.

## 2. 개선 후보의 우선순위

| 순서 | 후보 | 주 효과 | 근거 수준 | 첫 변경 규모 |
|---|---|---|---|---|
| P0 | 성능 지표·계측 경계 수정 | 올바른 판단 | 이중 나눗셈 6/6 재현 | 보고서/계측 |
| P1 | continuation의 출력 불필요 행에서 LM head 생략 | TTFT | 실행 경로 확인, 비용은 미측정 | 명시적 output mode + DSP 계약 |
| P2 | prefix restore 1회 복원 후 결과 재사용 | prefix-hit TTFT/메모리 | 중복 decode 경로 확인 | 기존 계획 E |
| P3 | expert cache 용량·정책·prefetch 튜닝 | steady decode 및 continuation | 저장 profile에 miss wait 9–10 ms/token | 먼저 설정 실험 |
| P4 | 매 token의 host 입력 구성/graph walk 비용 축소 | steady decode | 경로 확인, 비용 분해 필요 | binding metadata cache부터 |
| P5 | 요청 간 buffer 재사용, seed 호출 묶기 | 반복 요청·prefix-hit TTFT | 반복 해제/seed 경로 확인 | 수명·프로토콜 변경, 후순위 |

정확성 문서의 **A: 데이터 경합 수정, B/C: restore 증거와 비교기 보강**을
성능 변경의 선행 조건으로 한다. 계측 P0은 먼저 진행해도 된다.
여러 최적화를 한꺼번에 합쳐 원인 추적을 어렵게 만들지 않는다.

## 3. P1 — 입력 continuation에 불필요한 LM head가 실행됨

### 확인한 실행 경로

1. `nntrainer/htp_context.cpp:45`의 `HtpContext::runDecode()`는 resident
   continuation의 여러 입력 token을 한 token씩 `incremental_forwarding()`한다.
   중간 행은 `logits.set(false)`, 마지막 행은 `true`다. 중간에 받은 ID는 버린다.
2. `nntrainer/tensor/htp_backend/dsp/nntr_hvx_dspq.c:dspq_token()`의
   `HTP_DSPQ_TOKEN_LOGITS`는 반환 buffer 유무를 제어한다.
3. `nntrainer/tensor/htp_backend/hmx/hexkl_token.c:315`의
   `hexkl_token_main()`은 logits 포인터가 NULL이면 `g->logits`로 바꾸고
   `[0, g->n_ops)` 전체를 실행한다.
4. 마지막 `hexkl_graph.c:806 graph_op_lm_head()`는 vocabulary projection,
   필요시 softcap, argmax를 수행한다. 따라서 `logits=false`여도 head 계산은 남는다.
5. `CausalLM::run()`의 `SKIP_PREFILL` 경로는 `do_prefill(init_len - 1)` 후
   마지막 입력 token을 generation loop에서 처리한다. 이 경로에서는 앞선
   prefill의 최종 vocabulary 결과까지 사용하지 않는다.

이미 정해진 prompt token을 소비하는 중간 행은 다음 token을 고르지 않으므로
head 결과가 필요 없다. 이는 여러 번 실행될 수 있는 불필요한 계산이다.
저장된 최신 continuation 사례에서는 prefill token 11개에 TTFT 약 716 ms,
steady token 약 55.6 ms다. serial row 처리가 TTFT를 키울 수 있다는 맥락은
되지만, 그중 LM head가 몇 ms인지는 stage profile을 추가해야 안다.

### 구현 시 가장 위험한 함정

`nntrainer/layers/llm/htp_tie_word_embedding.cpp`의 마지막
`htpDecodeLmHead()` hook이 `invokeForward()`를 거쳐 **전체 DSP token 실행을
시작한다**. CPU의 LM-head layer를 그냥 return시키면 transformer와 KV 갱신까지
실행되지 않을 수 있다. 이 수정은 “layer 건너뛰기”가 아니다.

`test/htp/host/token_host_check.c:433–454`도 NULL logits 요청에서
`g->logits`와 reference를 비교한다. 기존 boolean을 “계산 생략”으로 재해석하면
기존 token-ID decode 계약을 깨뜨린다.

### CLINE 작업 P1a — 계약을 먼저 추가

1. `OutputKind { None, TokenId, Logits }`와 같은 명시적인 내부 표현을 도입한다.
   이름은 예시다. 기존 boolean 호출의 의미는 그대로 매핑한다.
2. `None`은 알려진 prompt token을 소비하고 반환 출력의 소비자가 없는 경우만
   선택한다. 첫 단계에서는 multi-row continuation의 중간 행에만 적용한다.
   마지막 행, 일반 single-token generation, sampling/logits processor는 기존 경로다.
3. capability 협상 후에만 새 DSP flag를 보낸다. 기존 skeleton은 unknown flag를
   거부한다. 미지원이면 **dispatch 전에** 기존 경로로 fallback한다.
   상태를 일부 갱신한 실패 요청을 같은 position으로 재시도하지 않는다.
4. `None` 완료 시 “행 실행 완료”와 “출력 ID 있음”을 분리한다. 가짜 ID를
   만들거나 이전 ID를 남기지 않는다. `have_id_`, position, pending inputs,
   pool ticket의 정상 완료/예외 처리를 함께 명시한다.
5. 아직 DSP 연산을 생략하지 않고 기존 결과와 동일한 채로 plumbing을 검증한다.

### CLINE 작업 P1b — DSP에서 마지막 head만 생략

1. supported graph의 최종 op가 LM_HEAD인지 검증한다. `None`에서는 transformer,
   attention KV, MoE 동작과 완료 처리를 모두 실행하고 최종 head만 생략한다.
2. 임의로 `n_ops - 1`을 넘기는 것으로 끝내지 않는다. graph stretch/resume 조건,
   `hexkl_graph_forward()`의 출력 buffer 계약, worker park, profile counter,
   response의 ID-valid 상태를 함께 수정한다. 오류는 상태 손상을 숨기지 않아야 한다.
3. tail norm 생략이나 batch attention 변경은 이 단계에 포함하지 않는다.
4. `token_host_check`에 동일 known-token sequence를 기존 모드와 None 모드로
   각각 실행하는 검사를 추가한다. 모든 layer의 KV 내용·길이·base, position을
   비교하고 **다음 정상 Logits 요청의 전체 logits와 ID**까지 비교한다.
   None 응답 자체에 logits가 없는 것은 성공 증거가 아니다.
5. 길이 1/2/여러 행, sliding window 경계, prefill 직후, restore 직후,
   session 전환, unsupported capability, 중간 실패를 검증한다.
6. NPU에서 `test/htp/htp_continuation_device_check.cpp` 기반 검증을 확장한다.
   동일 call plan의 baseline 대비 host checkpoint와 DSP KV/seed, 최종 continuation
   결과가 일치해야 한다. 기존 증거 수집기의 공백은 먼저 고친다.
7. 통과 후 별도 변경으로 SKIP_PREFILL의 마지막 미사용 결과도 None으로 지정한다.
   다른 caller가 반환 logits를 사용하는 경우에는 활성화하지 않는다.

측정: prompt token 수별 TTFT와 `lm_head_calls`, head cycles, DSP row 수를 기록한다.
None 대상만큼 head 호출이 줄고 transformer row 수는 같아야 한다. 전체 요청 시간,
첫 token 이후 latency, peak RSS/RPC도 확인한다.

장기 후보는 known-token 여러 개를 한 DSP 요청 안에서 순서대로 실행하는 것이다.
현재도 generated token 하나의 전체 graph는 한 요청이다. batching은 prompt의
host 왕복을 줄일 뿐 serial transformer 계산을 없애지 않는다. 진짜 M>1 prefill
kernel로 바꾸는 것은 quantization/calibration/연산 순서가 달라질 수 있는 별도 연구다.

## 4. P2 — prefix hit에서 같은 데이터를 여러 번 복원

`CausalLM::restoreDevicePrefill()`은 압축 page의 `inspect()` 후
`decodeCopy()` 결과를 버리고, 조건에 따라 `materializePrefix()`도 수행한다.
이후 실제 cache bind/restore에서 다시 materialize한다. codec OFF로 압축 baked
asset을 읽는 경우에는 raw 변환도 있다. 경로에 따라 2–3회 전체 복원과 반복 CRC,
gamma 준비, 임시 RAW allocation이 발생할 수 있다.

[기존 계획 E](gemma4_kv_cache_review_and_plan.ko.md)에 따라
검증과 복원이 완료된 `PreparedPrefix`를 만들고 실제 bind/seed까지 재사용한다.
CRC 검사를 없애는 식으로 해결하지 않는다. checksum/geometry/model binding을
검증한 immutable 결과만 재사용하며 실패하면 기존 session state를 유지한다.

측정 항목은 hit당 `decode_calls`, `decoded_bytes`, `CRC_bytes`, `raw_alloc_bytes`,
`prepare_ms`, `seed_ms`, peak live bytes다. 일반적인 한 번의 restore에서 page당
실제 decode가 1회가 되는지 검증한다. Flash read, RAW fallback, codec OFF baked
asset, 손상된 입력의 transaction 실패도 각각 검사한다.

추가로 capture 시 이미 압축된 과거 page를 반복 inspect하는 비용도 계측한다.
안전하게 검증된 immutable page에만 검증 결과를 memoize할 수 있다. 외부 Flash
파일이나 변경 가능한 encoded bytes에 검증 생략을 일반화하지 않는다.

## 5. P3 — steady decode는 expert miss부터 측정·조정

저장된 `evidence-full/{baseline,codec}/run.log`의 profile 재분석:

| 항목 | baseline | codec |
|---|---:|---:|
| DSP wall / token | 55.65 ms | 55.85 ms |
| host 관측 round trip / token | 56.62 ms | 57.12 ms |
| ARM tokenForward / token | 58.64 ms | 59.06 ms |
| expert miss wait / token | 9.27 ms | 9.86 ms |
| expert misses / token | 37.37 | 37.37 |
| predictor drain / token | 0.4 µs | 0.4 µs |
| prefetch drain / token | 0.1 µs | 0.1 µs |

miss wait는 DSP wall의 약 16.7–17.7%다. 나머지 병목이 그대로이고 이 대기만
줄어든다는 가정 아래의 기회 규모이며, 실제 개선율을 보장하지 않는다.
위 시간들은 서로 포함 관계다. 합산하면 안 된다. 이 로그는 과거 diagnostic run이며
현재 HEAD의 성능 baseline을 대신하지 않는다. `code_revision=unknown` 문제도 있다.

현재 `tokenForward()`는 token당 하나의 DSP queue 요청과 응답을 사용한다.
greedy 경로는 이미 token ID만 받는다. 전체 decode를 새로 “한 번의 RPC로 합치기”나
greedy vocabulary 반환 제거는 이 경로에서 이미 이루어진 최적화다.
반면 resident graph라고 모든 MoE expert weight까지 상주하는 것은 아니다.

저장 설정은 `moe_cache_experts=16`, decode prefetch 사용, cache grow=0이다.
codec 로그의 prefetch는 `submitted=6897`, `completed=5802`, `used=2766`,
`late=1303`, `discarded=4131`을 기록한다. 별도 hints counter의 0만 보고
prefetch 전체가 꺼졌다고 결론 내리면 안 된다. counter 의미와 중복 집계를 먼저 확인한다.

### CLINE 작업 P3a — 코드 변경 전 설정 실험

1. 현재 설정으로 corrected baseline을 새로 수집한다. layer별 miss, demanded
   expert, useful/late/discarded prefetch, 실제 I/O bytes, 읽기 대기/복사/등록,
   RPC pinned bytes, 전체 live memory를 기록한다.
2. 허용된 실제 기기 메모리 내에서 cache 용량 2–3개를 비교한다. OOM 직전까지
   무작정 키우지 않는다. prefix RAW/encoded/RPC/scratch와 같은 시점의 peak를 본다.
3. 동일 cache 용량에서 prefetch ON/OFF, 기존의 지원되는 정책을 독립 비교한다.
   useful hit가 늘면서 demand miss wait가 줄었는지 확인한다. 예측 제출 수만
   증가하거나 스토리지 읽기 경쟁이 늘면 개선이 아니다.
4. 이어서 trace replay로 layer별 cache 배분 정책을 비교하고, 효과가 있는 경우에만
   실제 replacement 정책을 변경한다. replay는 기기 I/O overlap을 증명하지 못한다.
5. router top-k, expert 선택, weight bits, expert 합산 순서는 변경하지 않는다.
   동일 입력의 IDs뿐 아니라 전체 logits와 KV 상태도 일치해야 한다.

KV16의 메모리 절감이 expert cache를 더 확보하게 한다면 **간접적인 decode 개선**은
가능하다. 다만 encoded 누적 bytes를 RSS 절감으로 간주하면 안 되며, 실제 반환된
host 메모리가 RPC/pinned allocation 여유로 이어지는지도 확인해야 한다.

`poolDisarm()`의 drain은 token/메모리 수명을 보호한다. 위 로그에서는 합계
0.5 µs/token이므로 제거를 우선하지 않는다. KV codec의 global drain과 expert
prefetch drain은 서로 다른 문제다. 전자는 기존 검토 R4의 요청/restore 대기 문제다.

## 6. P4 — DSP는 한 요청인데 host는 매번 graph를 순회

`CausalLM::incrementalInference()`는 매번 input dimension을 얻고 cache input
이름을 해석하며 Tensor wrapper와 입력 컨테이너를 구성한다.
`NetworkGraph::incremental_forwarding()`은 각 node의 forwarding hook을 순회한다.
resident hook이 실제 CPU 연산을 생략해도 이 준비 비용은 남는다.

다만 현재 logits output buffer는 재사용하고, `allocateAndBindKVCache()`도 이미
bound이면 빠르게 반환한다. “매 token 전체 KV를 새로 할당”한다고 해석하면 틀린다.
저장 profile의 `arm_us`는 prompt/capture 등 요청 간 공백까지 포함하므로
steady host overhead로 사용할 수 없다. 먼저 steady token 구간에만
`input_prepare_us`, `graph_walk_us`, `callback_us`를 계측한다.

CLINE의 첫 변경은 input slot/name → cache layer 매핑을 model topology별로
캐시하는 것이다. allocation/rebind/grow/model 변경 시 generation을 올려 invalidation한다.
해제된 Tensor data 포인터 자체를 장기 캐시하지 않는다. baseline과 input mapping,
실제 data owner, length/base가 동일한지 검증한다.

직접 resident row를 실행해 CPU graph walk 전체를 제거하는 것은 그다음이다.
embedding/PLE 입력, seed, layer state hook, exception 복구를 모두 보존해야 하므로
단순히 `tokenForward()`를 호출하는 shortcut을 만들지 않는다. host overhead가
측정상 작으면 이 구조 변경은 중단한다.

## 7. P5 — 요청 수명과 seed 비용

### 반복 초기화/해제

`beginIndependentHtpPrompt()`는 반복 요청에서 DSP prompt reset과 layer reset을
수행하고 `kv_cache = KVCacheManager{}`로 교체한다. 이전 codec pool의 join과
buffer 수명 정리가 포함될 수 있다. seed 뒤 CPU cache owner가 해제되는 경로도
있으므로, bind flag만 유지하여 이전 pointer를 재사용하면 안 된다.

먼저 reset, join, rpc allocation/free를 분리 계측한다. 유의하면 bounded buffer
pool로 실제 소유권을 유지하고 shape/capacity가 맞는 할당만 재사용한다.
session 종료/오류/모델 교체에서 확실하게 unpin/free하며, 사용 중 작업이 없는 것을
확인한 buffer만 반환한다. 독립 요청 간 KV 내용이 섞이지 않는 검증이 필수다.

prefill cleanup도 주의한다. `release_prefill_scratch()`에는 해제뿐 아니라
필요한 결과 materialization이 섞여 있다. 이름만 보고 호출을 삭제하지 않는다.
첫 token 내부의 staging release, heap purge, cache grow, footprint sample은
계측 후 선택적 지연을 검토하되, 첫 callback 개선과 2번째 token/다음 요청의
회귀를 함께 보고한다. peak memory를 숨기는 변경도 허용하지 않는다.

### seed RPC

기존 profile에는 prompt당 seed 30회, seed 합계 약 17/105/142 ms(codec)가 있다.
`htp_compute_ops.cpp:3007 seedRows()`는 FP16 입력에 직접 RPC buffer를 사용하는
경로가 이미 있다. “FP16을 FP32로 넓혔다가 다시 줄이는 작업 제거”가 모든 seed에
새로운 개선인 것은 아니다.

host 중복 복원을 제거한 뒤에도 seed가 중요하면 여러 layer의 seed descriptor를
한 요청에 담는 후보를 검토한다. 위 합계는 quantization과 데이터 처리까지 포함하며
전부 호출 overhead가 아니다. per-layer calibration, row 범위, sliding base,
buffer lifetime과 전체 성공/실패 계약을 보존해야 한다. protocol capability와
최대 packet 크기를 검증하고 partial mutation 뒤 무조건 fallback하지 않는다.

`kv_seed_q=true`는 DSP V 재사용 등 다른 tradeoff를 제공하지만 현재 host prefix
capture를 비활성화하는 경로와 연결된다. prefix 재사용 시나리오에서 전역으로
켜는 “한 줄 최적화”로 취급하지 않는다. cold workload와 prefix-hit workload를
나누고 수치 검증과 손익분기점을 확인한다.

## 8. 성능 변경마다 적용할 검증 절차

1. **고정 입력:** 모델/weight hash, ARM checker·engine·CausalLM library와 DSP
   skeleton hash, source revision, 설정, prompt IDs, call plan, sampling seed를 저장한다.
   `unknown == unknown`은 동일 빌드 검증이 아니다.
2. **정확성:** A/B를 같은 prefill call 경계로 실행한다. host checkpoint, seed 입력,
   DSP KV/scale/position, 모든 비교 대상 출력(마지막 continuation 포함)을 수집한다.
   단순 생략/재배치 최적화는 byte equality를 기본 gate로 삼는다.
   quantization/kernel 변경의 tolerance 정책은 별도 작업으로 승인·정의한다.
3. **계측 분리:** correctness trace ON과 performance trace OFF를 분리한다.
   trace가 codec queue를 drain해 race나 대기를 숨기는 기존 효과를 피한다.
4. **반복:** 같은 길이·cache hit 상태·storage warm/cold 조건으로 OFF/ON 순서를
   교차하여 최소 5 paired 반복을 시작점으로 삼는다. 가능하면 각 요청에서 첫 token
   이후 128개 이상을 측정한다. EOS로 일찍 끝나면 표본 수를 그대로 보고하고
   latency 통계를 억지로 채우지 않는다. 온도/클록/전력 모드를 함께 기록한다.
5. **보고:** API-entry TTFT median/p95, steady token median/p95, 실제 처리량,
   전체 request wall, 다음 요청 reset 시간, peak RSS/RPC, queue high-water를 보고한다.
   표본이 적은 p95에는 그 제한을 표시한다. 내부 span의 이동을 전체 개선으로 오인하지 않는다.
6. **채택:** 정확성 gate 실패 시 중단한다. 성능은 사전 합의한 noise floor보다
   큰 반복 가능한 개선이 있어야 채택한다. 안정적 차이가 없으면 default 활성화하지 않는다.

## 9. CLINE 인계 순서와 중단 조건

권장 순서: 기존 A → 기존 B/C → P0 → 기존 E/P2 → P1a → P1b → P3a → P4/P5 판단.
P2는 protocol 변경 없이 TTFT 중복 작업을 줄일 수 있어 구현 순서상 P1보다 먼저다.
각 단계는 독립 commit으로 만들고 DCO/agent trailer와 저장소 형식 규칙을 따른다.

각 변경의 결과에는 다음을 반드시 적는다.

- 변경한 invariant와 해당 함수; 변경하지 않은 수치 연산 계약.
- 실행한 host/NPU 검사, 실제 명령, exit code, artifact 경로와 빌드 hash.
- 대상 workload의 baseline/변경값, 반복 수, 실패/회귀 사례.
- capability fallback 및 새 설정을 끄는 방법.

새 구현이 출력 경로를 우회하거나 수명 문제를 해결하지 못하면 다음 최적화로
넘어가지 않는다. NPU가 없는 환경에서는 host 검증과 설계까지 완료하고
“기기 속도 개선 확인”으로 표기하지 않는다. 이 문서 역시 소스 분석과 기존 로그
재계산에 근거하며 새로운 NPU speedup을 주장하지 않는다.
