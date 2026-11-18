# Qwen3 Ridge 실행 결과를 같은 HellaSwag 문항에서 비교한 기록

nntrainer는 공개 fit 120 매퍼로 HellaSwag 1,000문항 중 **551개**를 맞혔다. 같은 문항에서 로컬 Transformers는 **555개**, 공개 vLLM 보고서는 **553개**다. 전송 전에는 nntrainer와 Transformers가 각각 582개를 맞혔다. 전송 후 두 로컬 런타임은 985문항에서 같은 후보를 골랐다. 아래 정확도는 후보의 로그확률 합을 후보 토큰 수로 나눈 점수로 계산했다.

| 동일한 공개 1,000문항 | 일반 실행 | Ridge 전송 | 전송 후 정확도 유지율 |
| --- | ---: | ---: | ---: |
| nntrainer CPU FP32 모델, FP16 KV, FP32 매퍼 | 58.2% | 55.1% | 94.67% |
| Transformers CPU BF16 | 58.2% | 55.5% | 95.36% |
| 공개 vLLM | 58.0% | 55.3% | 95.34% |

두 로컬 런타임의 전송 정답 차이는 4문항이다. nntrainer만 맞힌 문항이 3개, Transformers만 맞힌 문항이 7개였다. 나머지 선택 불일치 5개는 둘 다 오답이었다. 실제 KV를 전송한 968문항에서는 nntrainer가 536개, Transformers가 540개를 맞혔다. 짧은 문맥 32문항은 두 방식 모두 target 일반 실행으로 처리했고 각각 15개를 맞혔다. [nntrainer 문항별 결과](validation/kv-transfer/native-parity/cases.json) · [문항별 비교와 점수 오차](validation/kv-transfer/native-parity/native-public-paired-audit.json) · [공개 vLLM 결과](https://github.com/alesha-pro/vllm-cross-model-kv/blob/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/results/quality-report-n200.json)

## 고친 코드

기존 CPU 채점기는 캐시를 불러오는 모든 요청에서 마지막 문맥 토큰 한 개만 target에 넘겼다. 이전 실험은 항상 앞의 `L−1`개를 전송했으므로 그 실험에서 기록한 점수는 유효하다. 공개 방식은 `16 × floor((L−1)/16)`개를 전송하고 남은 1~16개를 target에서 계산한다. 이전 채점기에 이 경계를 지정하면 전송도 계산도 되지 않는 문맥 토큰이 생긴다.

`NativeScorer.score`가 snapshot의 모델·토큰 ID·위치를 실제 문맥 앞부분과 확인한 뒤 `context[snapshot.tokens:]` 전체를 넘기도록 수정했다. 마지막에 채점할 문맥 토큰이 하나도 남지 않은 요청은 거절한다. 단일 요청 경로의 모델 설정도 여러 토큰의 tail을 수용하도록 수정했다. 회귀 테스트에서 2토큰을 가져와 3토큰을 다시 계산하는 경우, 잘못된 prefix, 남은 토큰이 없는 경우를 검증했다. 기존 및 추가 Python 테스트 18개가 통과했다. [채점기](../tools/kv_transfer/benchmark.py) · [회귀 테스트](../tools/kv_transfer/test_kv_benchmark.py)

공개 비교용 실행기를 추가했다. Transformers 실행에 사용한 seed 0 문항 순서, 후보·문맥의 토큰 ID 해시, 정답, 전송 경계를 **1,000문항 전체에서 대조**한다. 모델의 일반 실행과 전송 실행은 nntrainer CPU CausalLM과 PR #4481 기반 KV manager에서 수행하고, 공개 k4-n200 계수 적용은 기존 호스트 FP32 매퍼에서 수행한다. 매퍼 재학습 없이 기존 checkpoint를 사용했다. [공개 조건 실행기](../tools/kv_transfer/native_public_benchmark.py) · [입력 및 모델 파일 해시](validation/kv-transfer/native-parity/inputs.json)

## 기존 128문항에서 무엇이 달라졌나

같은 128문항과 fit 120 매퍼를 고정했다. 각 행의 세 숫자는 일반 target, 마지막 1토큰을 다시 계산한 전송, 공개 블록 경계로 다시 계산한 전송의 정답 수다.

| 후보 텍스트와 정규화 | 일반 target | 전송 후 1토큰 재계산 | 전송 후 1~16토큰 재계산 |
| --- | ---: | ---: | ---: |
| 기존 텍스트 / 문자 수 | 71 | 61 | 64 |
| 기존 텍스트 / 토큰 수 | 73 | 54 | 58 |
| 공개 텍스트 / 문자 수 | 70 | 63 | 68 |
| 공개 텍스트 / 토큰 수 | 72 | 56 | 58 |

- 기존 텍스트와 토큰 수 정규화를 유지하면 재계산 경계만 바꿔 전송 정답이 54개에서 58개가 된다. 새로 맞힌 문항이 6개, 잃은 문항이 2개다. 실제 target KV를 가져온 6개 대조군에서 일반 실행과의 최대 토큰 로그확률 차이는 0.00483 nat였다.
- 128개 문맥의 토큰과 정답 라벨은 공개 전처리 결과와 전부 일치했다. 후보 512개 중 155개(68문항)에는 기존 전처리에서 앞에 공백 토큰 220이 하나 더 들어갔다. 해당 후보만 다시 계산했다. 공개 블록 경계와 토큰 수 정규화에서는 공백을 고친 뒤에도 전송 정답은 58개였다.
- **공개 방식으로 기존 128문항을 Transformers에서도 채점**했다. 일반 실행은 69개, 전송은 59개를 맞혔다. 같은 문항의 nntrainer 결과는 72개와 58개였다. 두 런타임은 일반 및 전송에서 각각 125/128문항의 후보 선택이 일치했다. 공개 1,000문항과 기존 128문항은 12개만 겹친다. 평가 방식을 맞춰도 기존 표본에서는 양쪽 모두 전송 정확도가 낮다. 원래 61/128과 공개 555/1,000의 차이를 런타임만의 손실로 환산할 수 없다. [128문항 조건별 집계](validation/kv-transfer/native-parity/native-ridge-ablation-summary.json) · [같은 128문항의 Transformers 점수](validation/kv-transfer/native-parity/hf-legacy-sample-public-protocol.json)

## 모델 변환과 남은 수치 차이

양쪽 native 설정이 실제로 여는 FP32 가중치 파일을 사용했다. 0.6B와 1.7B에서 각각 **310개 텐서**를 graph 순서와 전치 규칙에 따라 원본 Safetensors와 비교했다. 정확히 FP32로 확장한 값과 모든 비트가 일치했다. 공개 Ridge 계수와 source layer 순서의 보존도 앞선 독립 검증에서 확인했다. [가중치 검증 코드](../tools/kv_transfer/verify_qwen3_weights.py) · [모델별 바이트 검증](validation/kv-transfer/native-parity/ridge-weight-parity.json) · [매퍼 변환 검증](validation/kv-transfer/public-mapper-conversion-check.json)

공개 1,000문항에서 후보 선택이 다른 사례를 전부 모았다. 일반 실행 11개, 전송 15개로 중복을 합치면 25문항이다. 이를 로컬 Transformers에서 배치 크기 1로 고정하고 FP32와 BF16으로 다시 계산했다.

| 비교 | 일반 실행 불일치 11건 | 전송 불일치 15건 |
| --- | ---: | ---: |
| Transformers FP32가 nntrainer와 같은 답을 선택 | 11 | 14 |
| Transformers BF16 배치 1이 기존 BF16 배치 4와 같은 답을 선택 | 11 | 13 |
| Transformers에서 배치 1을 유지하고 FP32와 BF16만 바꿨을 때 답이 달라짐 | 11 | 12 |

전송 불일치 15건에서 기존 BF16의 1·2위 차이는 최대 0.03788 nat/token이었다. 이 문항들의 후보 점수에서 nntrainer와 배치 1 Transformers FP32의 최대 차이는 0.00398 nat/token이었다. 따라서 남은 순위 차이 대부분은 수치 정밀도와 근소한 후보 점수차로 설명할 수 있다. 배치 구성도 일부 선택에 영향을 주었다. 이 실험이 보인 정확도는 고정된 HellaSwag 표본과 해당 모델 조합에 해당한다. [정밀도별 후보 점수](validation/kv-transfer/native-parity/ridge-precision-final.json)

FP32에서도 선택이 달랐던 전송 문항 5444는 native 스칼라 채점과 블록 채점에서 모두 정답인 네 번째 후보를 골랐다. **같은 native 매핑 캐시**를 Transformers FP32에 넣자 이쪽도 네 번째 후보를 골랐다. FP16 import 반올림을 맞춘 두 consumer의 후보 평균 점수 차이는 최대 0.000119 nat/token이었다. 이 사례에서는 채점 블록 크기보다 모델이 받아들인 KV 값이 선택 차이를 설명한다. [동일 캐시 대조군](validation/kv-transfer/native-parity/tie-5444.json)

추가로 두 모델의 nntrainer 실행을 유지하고 호스트 매퍼 연산만 공개 BF16 방식으로 바꿨다. 공개 순서의 첫 32문항에서 Transformers와 후보 선택이 일치한 수는 31개에서 32개가 됐다. 원래 선택이 달랐던 15문항에 한정하면 5개가 Transformers와 같아졌다. 두 집합을 합쳐 46문항을 검사했다. 전체 1,000문항 점수와 기본 매퍼는 FP32 실행 결과를 사용한다. [매퍼 정밀도 대조](validation/kv-transfer/native-parity/mapper-precision-pilot.json)

## 재현

WSL x86에서 기존 CPU 빌드, 변환된 두 모델, 공개 매퍼, 로컬 벤치 입력을 준비한 뒤 저장소 루트에서 실행한다. 결과 디렉터리는 중단 후 같은 입력으로 재개할 수 있다. [모델·매퍼 준비 과정](public-ridge-evaluation.ko.md) · [Transformers 참조 환경 및 입력](hf-public-benchmark.ko.md)

~~~sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv-reference/bin/python tools/kv_transfer/native_public_benchmark.py \
  --binary build-cpu/Applications/CausalLM/nntr_causallm \
  --source artifacts/models/qwen3-0.6b-fp32 --target artifacts/models/qwen3-1.7b-fp32 \
  --mapper artifacts/public-ridge-k4-n200 --data artifacts/ridge-paper-data/hellaswag.jsonl \
  --public-report artifacts/hf-public-benchmark-upstream.json \
  --hf-cases docs/validation/kv-transfer/hf-public-cases.json \
  --out artifacts/native-public-1000 --scratch /dev/shm --threads 4 --batch 8
.venv/bin/python tools/kv_transfer/audit_native_public.py \
  artifacts/native-public-1000/cases.json docs/validation/kv-transfer/hf-public-cases.json \
  artifacts/native-public-paired-audit.json
~~~

기존 모델과 저장된 참조 점수를 재사용했다. 1,000문항 native 실행은 체크포인트 기준 약 2시간 41분이 걸렸고 그중 일부 시간에 128문항 대조 실험도 같은 CPU에서 수행했다. 별도 subagent를 호출하지 않았다. 부모 대화의 이번 작업만 분리한 토큰 수와 실제 구독 크레딧 차감액은 미측정이다. [향후 작업용 비용 기록](validation/agent-cost-rework-20261118.md)
