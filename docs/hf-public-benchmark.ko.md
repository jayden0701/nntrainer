# 공개 k4/n200 매퍼의 Transformers HellaSwag 평가

공개 vLLM 평가에 기록된 Qwen3-0.6B → Qwen3-1.7B 매퍼를 Transformers에서 실행했다. 공개 보고서는 HellaSwag validation 1,000문항에서 길이 정규화 정확도 58.0% → 55.3%, 정확도 유지율 95.34%를 기록했다. 이번 평가는 해당 문항, 토큰화, prefix 경계, 매퍼와 채점 방식을 맞추고 CPU BF16 eager attention과 메모리 내 Transformers KV 캐시를 사용한다. [공개 보고서](https://github.com/alesha-pro/vllm-cross-model-kv/blob/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/results/quality-report-n200.json)

## 평가 조건

- 공개 보고서의 seed 0 인덱스 1,000개를 순서대로 사용했다. 로컬 데이터는 Rowan/hellaswag validation의 `218ec52e09a7e7462a5400043bb9a69a41d06b76` revision에서 추출했다. 공개 스크립트는 데이터셋 revision을 지정하지 않았다. 로컬 validation 행 순서로 seed 0 셔플을 재실행했을 때 보고서 인덱스와 일치했고, 토큰화 결과에서 짧은 문항 32개의 인덱스도 일치했다. [데이터셋](https://huggingface.co/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76)
- context에는 활동명, `ctx_a`, 첫 글자를 대문자로 처리한 `ctx_b`를 사용했다. 공개 스크립트와 같은 텍스트 정리 과정을 적용한 뒤 context와 공백이 앞에 붙은 각 ending을 특수 토큰 없이 따로 토큰화했다. ending 토큰의 teacher-forced log-probability 합으로 `acc`를, 합을 ending 토큰 수로 나눈 점수로 `acc_norm`을 구했다. [공개 채점 코드](https://github.com/alesha-pro/vllm-cross-model-kv/blob/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/tools/quality_eval.py)
- context 길이를 `L`이라고 하면 source 캐시에서 `16 × floor((L − 1) / 16)`개 토큰을 매핑했다. target이 나머지 1–16 context 토큰을 계산한 뒤 ending을 채점했다. 16토큰 이하 문항은 공개 보고서와 같이 두 방식 모두 target의 일반 점수를 사용했다. [공개 prefix 스케줄러](https://github.com/alesha-pro/vllm-cross-model-kv/blob/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/cross_model_kv/scheduler.py)
- 매퍼 Hub revision은 `4653cc51c5867ffec5eb36ef1628239453d054c8`이다. FP32 파일에서 계수 행렬을 BF16으로 내린 파일의 SHA-256은 Hub에 게시된 BF16 파일과 일치했다. 실제 매핑 시 bias도 BF16으로 변환해 `addmm`에 입력했다. K의 역·정방향 RoPE는 FP32에서 계산하고 각각 BF16으로 변환했다. [매퍼 파일](https://huggingface.co/alesha-pro/vllm-cross-model-kv/tree/4653cc51c5867ffec5eb36ef1628239453d054c8/ridge-qwen3-0.6b-to-1.7b-k4-n200) · [공개 매핑 코드](https://github.com/alesha-pro/vllm-cross-model-kv/blob/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/cross_model_kv/mapper.py)
- Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`의 모델 파일 SHA-256은 각 Hub revision과 일치했다. 모델 config와 토크나이저 파일의 SHA-256은 공개 매퍼 manifest와 일치했다.
- Windows WSL x86 CPU, Python 3.12.16, PyTorch 2.9.1+cpu, Transformers 4.57.3, 스레드 4개, 배치 4문항을 사용했다. 공유 context를 한 번 처리한 뒤 KV 캐시를 후보 4개에 복제했다. 유효 토큰의 RoPE 위치는 패딩과 무관하게 context의 실제 위치를 사용했다.

## 파일럿 검증

첫 16문항으로 전체 채점 경로를 실행했다. 그중 import 토큰 수가 각각 16개와 80개인 문항 232와 9707에서 후보를 하나씩 처리하는 별도 teacher forcing과 배치 경로를 비교했다.

| 비교 | 문항 232 최대 차이 | 문항 9707 최대 차이 |
| --- | ---: | ---: |
| 일반 배치 ↔ 일반 직렬 | 0 | 0 |
| target exact-cache 배치 ↔ 일반 직렬 | 0 | 0.007681 |
| 매핑 배치 ↔ 매핑 직렬 | 0 | 0.007140 |

차이는 후보당 log-probability 합의 절대 오차를 해당 후보의 토큰 수로 나눈 값이다. 정규화 점수의 1위 후보는 여섯 비교에서 일치했다. 서로 다른 캐시 분할과 패딩에서 BF16 결과에 차이가 생겼으며, 이 검증으로 개별 커널의 기여도까지 확인할 수는 없다.

## 전체 평가

1,000문항을 모두 완료했다. 채점에는 문항당 평균 2.900초, 합계 약 48분 20초가 걸렸다.

| 환경과 표본 | 문항 수 | Standalone acc | Mapped acc | Standalone acc_norm | Mapped acc_norm | acc_norm 유지율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 공개 vLLM 보고서 | 1,000 | 46.7% | 43.3% | 58.0% | 55.3% | 95.34% |
| Transformers CPU 전체 | 1,000 | 468/1,000 (46.8%) | 433/1,000 (43.3%) | 582/1,000 (58.2%) | 555/1,000 (55.5%) | 95.36% |
| Transformers CPU 실제 전송 | 968 | 452/968 (46.69%) | 417/968 (43.08%) | 567/968 (58.57%) | 540/968 (55.79%) | 95.24% |

전체 표본의 Transformers 정규화 정확도는 두 방식 모두 공개 보고서보다 0.2%p 높았고, 매핑에 따른 정확도 차이는 두 환경에서 모두 −2.7%p였다. 짧은 문항 32개에서는 양쪽이 같은 target 점수를 사용했고, `acc_norm` 기준 15개를 맞혔다. 실제 전송 968개에서는 정규화 기준으로 34개를 새로 맞혔고 61개를 잃었다. 정답 여부가 그대로인 경우를 포함해 127개에서 선택한 후보가 바뀌었다.

문항을 짝지어 bootstrap 10,000회(seed 0)를 수행했다. 95% 구간은 전체 `acc_norm` 차이에 대해 −4.6에서 −0.8%p, 유지율에 대해 92.19%에서 98.56%다. 실제 전송 문항의 구간은 각각 −4.75에서 −0.83%p, 91.97%에서 98.55%다. 구간은 고정된 모델과 매퍼에서 문항을 재표집한 변이를 나타낸다.

공개 수치와 근접한 결과를 다른 런타임에서 얻었으며, 공개 보고서에 문항별 점수가 없어 예측 단위 일치 여부는 비교할 수 없다. 기존 nntrainer 128문항 평가와는 문항, prefix 범위 및 채점 방식이 다르다. 이번 결과만으로 기존 격차의 각 요인이 기여한 정도를 구분할 수 없다. [기존 nntrainer 평가](public-ridge-evaluation.ko.md)

## 경계 문항 직렬 검증

각 방식에서 정규화 점수 1·2위 차이가 0.016 nat/token 이하인 모든 문항과 차이가 가장 작은 최소 4문항을 직렬로 다시 채점했다. 일반 15건, 매핑 26건으로 후보 164개를 검사했다. 일반 채점의 선택은 모두 같았고 매핑에서 2문항의 선택이 바뀌었다. 348번은 정답에서 오답으로, 6415번은 오답에서 정답으로 바뀌었으므로 두 문항의 직렬 점수를 반영해도 정답 개수는 같다. 위 표는 전체 배치 실행의 원본 점수를 사용한다.

관측한 최대 후보 평균 log-probability 차이는 0.044426 nat/token이었다. 이는 문항 선정 기준인 0.016보다 크며, 이 검증 범위 밖의 문항에서도 선택 변화가 발생할 수 있다. [경계 문항 직렬 점수](validation/kv-transfer/hf-public-near-ties.json)

## 실행

`.venv-reference`에 위에 명시된 Transformers/PyTorch 환경을 준비하고 repo 루트에서 실행한다. `prepare_paper_data.py`로 준비한 HellaSwag JSONL 및 공개 매퍼 원본을 사용한다.

~~~sh
curl -fsSL \
  https://raw.githubusercontent.com/alesha-pro/vllm-cross-model-kv/2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc/results/quality-report-n200.json \
  -o artifacts/hf-public-benchmark-upstream.json

.venv-reference/bin/python tools/kv_transfer/hf_public_benchmark.py \
  --source /home/jayden/nntrainer/build-kv-gpu-cuda/models/hf/qwen3-0.6b \
  --target artifacts/models/hf-qwen3-1.7b \
  --mapper artifacts/public-ridge-original/ridge-qwen3-0.6b-to-1.7b-k4-n200 \
  --data artifacts/ridge-paper-data/hellaswag.jsonl \
  --public-report artifacts/hf-public-benchmark-upstream.json \
  --out artifacts/hf-public-benchmark --minutes 75
~~~

`metadata.json`에 전체 문항 인덱스 및 모델·데이터·매퍼·토큰화·러너의 해시와 라이브러리 버전을 기록한다. `cases.json`은 각 문항의 채점 기록이고, `summary.json`에는 전체와 실제 전송 문항의 지표가 들어 있다. 중단 후 같은 인수와 출력 디렉터리로 실행하면 마지막 16문항 단위 checkpoint에서 이어서 진행한다. 파일럿은 별도 출력 디렉터리에 `--limit 16 --pilot-check`를 붙여 실행한다.

경계 문항 직렬 검증은 다음 명령으로 다시 실행한다.

~~~sh
.venv-reference/bin/python tools/kv_transfer/hf_public_benchmark_tie_audit.py \
  /home/jayden/nntrainer/build-kv-gpu-cuda/models/hf/qwen3-0.6b \
  artifacts/models/hf-qwen3-1.7b \
  artifacts/public-ridge-original/ridge-qwen3-0.6b-to-1.7b-k4-n200 \
  artifacts/ridge-paper-data/hellaswag.jsonl \
  artifacts/hf-public-benchmark-upstream.json \
  artifacts/hf-public-benchmark
~~~

[실행 메타데이터](validation/kv-transfer/hf-public-metadata.json) · [문항별 배치 점수](validation/kv-transfer/hf-public-cases.json) · [요약 지표](validation/kv-transfer/hf-public-summary.json) · [입력 파일 검증](validation/kv-transfer/hf-public-input-verification.json) · [독립 집계](validation/kv-transfer/hf-public-root-audit.json)
