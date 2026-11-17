# KV transfer evaluation: 위임 작업 비용 기록

2026-11-17. 세 작업 모두 독립 검증을 통과한 뒤 완료 처리했다. 세션 전체가 해당 attempt에 속하는 전용 subagent 기록의 최종 누적 token counter를 가져왔다.

| 작업 | 모델 / effort | uncached input | cached input | output | total |
| --- | --- | ---: | ---: | ---: | ---: |
| C++ native scorer | gpt-6.1-terra / high | 70,032 | 6,712,064 | 24,389 | 6,806,485 |
| OLS/Ridge mapper | gpt-6.1-sol / medium | 61,757 | 1,477,888 | 12,945 | 1,552,590 |
| Benchmark harness/run/audit | gpt-6.1-sol / medium | 230,410 | 26,232,576 | 66,658 | 26,529,644 |

cached input은 input의 부분집합이며 total에 다시 더하지 않는다. output에는 reasoning이 포함되므로 별도 가산하지 않는다. 대화 문맥을 반복 참조한 cached input이 대부분이다. 이 숫자는 실제 ChatGPT 요금/크레딧 또는 unique tokens를 뜻하지 않는다.

C++ 수명·메모리·수치 정렬 작업은 이전 CPU bridge 성공 경험을 근거로 Terra/high, 범위가 정해진 NumPy 및 평가 도구는 Sol/medium에 배정했다. 이는 초기 라우팅 가설이며 최저 비용 모델임이 입증된 것은 아니다.

부모의 검토·수정·실행관리 사용량은 다른 작업과 섞여 있어 unknown으로 남겼다. 동일 과제의 대조 모델 실행 및 실제 요금 데이터도 없으므로 전체 성공당 비용, 모델 간 비용 우위, 절감 토큰/요금 비율은 계산하지 않는다. 과거 작업의 session-scope 기록은 이번 task 비용에 배분하지 않았다.

검증: native CPU build, KV core 64 tests, Python 17 tests, 260개 benchmark 독립 재집계, exact-cache 6개 control 일치, 좁은 후보 간격 11개 scalar 재채점에서 winner 변화 없음. 실험에서 downstream 품질이 저하된 것은 도구 구현 실패로 기록하지 않는다.

개선할 위임 방식: 긴 실행은 shell 작업으로 묶고 체크포인트에서만 결과를 회수한다. benchmark 구현과 장시간 실행 감시를 분리해 큰 대화 문맥을 반복 처리하는 비용을 줄일 후보로 기록한다. 현재 기록만으로 절감률을 주장하지 않는다.

기계 판독 원장: `.cache/agent-cost/ledger.jsonl` (로컬, ignored). 각 attempt의 model/effort, 최종 사용량, source digest, 검증 근거가 별도 event로 저장된다. 일반화된 도구는 `tools/agent_cost/ledger.py`를 사용한다.

## Transformers 공개 벤치와 cross-family 실험의 병렬 위임

두 작업은 사용자 요청에 따라 병렬로 실행했다. 각 전용 subagent를 시작하기 전에 acceptance와 요청 모델·effort를 로컬 원장에 등록했다. 이전에 검증된 Python 평가·매핑 작업 경험을 근거로 두 작업 모두 `gpt-6.1-sol / medium`, 새 문맥(`fork_turns=none`)으로 배정했다. 서로 다른 모델을 같은 과제로 비교한 근거는 없어 최저 비용 선택이 입증됐다고 주장하지 않는다.

| 작업 성격 | 요청·관측 모델 / effort | uncached input | cached input | output | total |
| --- | --- | ---: | ---: | ---: | ---: |
| Cross-family 학습·정렬·대조군·HF 평가 | gpt-6.1-sol / medium | 87,816 | 4,077,568 | 24,044 | 4,189,428 |
| 공개 조건 이식·Transformers 1,000문항·재채점 | gpt-6.1-sol / medium | 115,169 | 9,939,712 | 28,287 | 10,083,168 |

Transformers attempt는 전체 1,000문항, 실제 전송 968문항, 짧은 문항 32개, 문항별 원시 점수의 독립 재집계와 13개 artifact hash를 검증했다. 경계 문항 41건의 직렬 재채점에서 2개 선택이 바뀌었으나 정답 수의 순변화는 0이었다. 본 배치 점수를 보존하고 해당 민감도를 보고했다.

Cross-family attempt는 독립 raw-logit FP64 NLL 재계산, 집계 결과, train/validation 분리, 19개 artifact hash를 확인한 뒤 성공 처리했다. 학습된 mapper의 품질 이득이 없다는 실험 결과는 구현 실패가 아니다. FP32 cross-entropy와 독립 FP64 재계산 간 작은 차이를 발견해 체크포인트·KV·logit을 그대로 유지한 채 평가 정밀도를 수정했다. 동일 seed 재실행과 수정 비용도 같은 attempt에 포함했다.

부모는 별도로 native 모델 변환, Llama snapshot 허용 범위 확장, 64개 native 문항 및 6개 exact control, HF/native 동일 캐시 비교, 코드 검토와 게시를 수행했다. 부모의 작업별 사용량은 분리할 수 없어 unknown으로 남긴다. 위 표는 subagent의 완료된 전용 attempt counter이며 전체 작업 비용이나 ChatGPT 요금으로 환산하지 않는다.

과거 Ridge 벤치 위임은 새로운 native 채점 경로 개발을 포함했고 이번 작업은 기존 도구와 모델을 재사용한다. 과제 범위·기준선·표본 수가 달라 토큰 총량의 차이를 동일 작업에서의 절감률로 표시하지 않는다. 장시간 실행의 반복 문맥 조회를 줄이기 위해 shell 실행과 checkpoint를 사용하고, 완료 후에 집중 검증했다.
