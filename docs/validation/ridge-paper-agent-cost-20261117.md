# Ridge evaluation: 위임 작업 비용 기록

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
