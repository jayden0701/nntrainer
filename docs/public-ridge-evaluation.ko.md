# 공개 Qwen3 0.6B → 1.7B Ridge mapper의 nntrainer 평가

## 목적과 고정 조건

`alesha-pro/vllm-cross-model-kv`의 공개 `k4-n200` mapper를 재학습 없이 가져와, 기존 nntrainer 실험과 동일한 문항 및 채점 조건에서 비교한다. 이번 변경 요소는 mapper artifact이다. 공개 프로젝트의 vLLM 실행이나 headline 수치 자체를 재현하는 실험은 아니다.

- nntrainer PR #4481 기반: `5e97254e764aa30db4223c33d0dc3b0fded091b8`, WSL x86 CPU.
- source / target: Qwen3-0.6B / Qwen3-1.7B. FP32 모델, FP16 native KV, FP32 mapping 연산.
- 고정 manifest: ARC-Challenge 128문항, HellaSwag 128문항, WikiText-2 4구간(4,096 scored tokens).
- manifest SHA-256: `b2b3d3dbdc6fab83704fff1fe28a5e2450dc3c2f1beddc1607a789775bb1f3c8`.
- ARC/Hella는 문자 수 정규화, context 마지막 한 토큰만 target에서 재계산, seed 202608, threads 4, score block rows 64.
- 기존 run의 실행 파일·라이브러리·모델 설정·manifest fingerprint가 모두 동일함을 확인한 뒤 cold target 점수를 재사용했다. 첫 ARC/Hella 두 사례는 새로 cold 평가하여 기존 점수와 bitwise 일치함을 확인했다. 총 6개 exact-cache control은 다시 실행했다.

## 외부 artifact와 호환성

[공개 코드](https://github.com/alesha-pro/vllm-cross-model-kv), [공개 mapper](https://huggingface.co/alesha-pro/vllm-cross-model-kv).

- Hub revision: `4653cc51c5867ffec5eb36ef1628239453d054c8`.
- 파일: `ridge-qwen3-0.6b-to-1.7b-k4-n200/mapper-fp32.safetensors`.
- 원본 SHA-256: `6db1051f8a47e931b95ab315729c569b483e34fab64ebfdd593005f4884808d2`. Hub LFS hash와 일치.
- 공개 manifest의 source revision `c1899de289a04d12100db370d81485cdf75e47ca`, target revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`는 로컬 변환에 사용한 모델과 동일하다. 양쪽 config.json 및 tokenizer.json 해시도 원본 manifest와 일치한다.
- geometry는 28 layers × 8 KV heads × 128 head dimension, unscaled split-half RoPE(theta=1e6)이다.
- 원본 `source_layers`, `W_K`, `W_V`, `b_K`, `b_V`를 native mapper layout으로 재배치했다. 모든 coefficient와 source layer 순서의 bitwise 보존을 검증했다. 외부 Python 코드를 실행하지 않고 safetensors 데이터만 읽었다.
- 실제 native cache의 위치 997–1003에서 28개 전체 layer에 대해 독립적인 complex-number RoPE 참조와 비교했다. 최대 절대 차이는 K 9.1553e-5, V 0이었다. 참조는 complex128, native 도구는 trig를 FP32로 반올림하므로 연산 순서 차이가 있다.

이름의 `n200`은 fit에 200개를 모두 썼다는 의미가 아니다. 배포 manifest에는 **fit 120개, validation 40개**가 기록되어 있다. 이전 응답의 'calibration 200개'는 공개 bundle 이름을 따른 표현이었다. 우리 기존 mapper는 50개 전체를 fit에 사용했다. 두 mapper 모두 k=4, lambda=0.01이며 source layer 선택·calibration 문서·fit activation 정밀도가 다르므로 결과 차이를 학습 문서 수 하나의 효과로 해석할 수 없다.

## 공개 headline과 비교할 때의 차이

공개 HellaSwag 수치는 1,000문항, token-length normalization이다. 우리의 character-length normalization과 다르다. 공개 경로는 block 정렬에 따라 마지막 1–16개 context 토큰을 재계산하고, 공개 결과 JSON의 short-prompt 128 entries는 중복을 제거하면 32문항으로 이 문항들은 full target prefill 점수를 포함한다. 우리는 모든 문항에서 prefix를 mapping하고 마지막 한 토큰만 재계산한다.

공개 WikiText은 1,024-token chunk / 512-token prefix이며 우리와 원 논문은 2,048 / 1,024 구성이다. 원본 공개 연산은 BF16이고 이번에는 FP32 mapper artifact를 사용한다. 따라서 95.3% headline 대비 수치 차이는 동일 프로토콜에서의 구현 오차를 뜻하지 않는다.

## 재현

`requirements-eval.txt`의 의존성에 `safetensors==0.8.0`이 추가로 필요하다.

```sh
.venv/bin/python tools/kv_transfer/import_public_mapper.py \
  artifacts/public-ridge-original/ridge-qwen3-0.6b-to-1.7b-k4-n200/manifest.json \
  artifacts/public-ridge-original/ridge-qwen3-0.6b-to-1.7b-k4-n200/mapper-fp32.safetensors \
  artifacts/models/qwen3-0.6b-fp32 artifacts/models/qwen3-1.7b-fp32 \
  artifacts/public-ridge-k4-n200 --revision 4653cc51c5867ffec5eb36ef1628239453d054c8

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python tools/kv_transfer/benchmark.py run \
  build-cpu/Applications/CausalLM/nntr_causallm \
  artifacts/models/qwen3-0.6b-fp32 artifacts/models/qwen3-1.7b-fp32 \
  artifacts/public-ridge-k4-n200 artifacts/ridge-paper-data/bench-manifest-128.json \
  artifacts/public-ridge-benchmark --threads 4 --score-block-rows 64 --batch-cases 16
```

새 output directory에서 실행하면 cold 점수도 다시 생성하므로 baseline 복사는 재현에 필수 사항이 아니다. 원본 다운로드 및 변환 결과는 `artifacts/public-ridge-*`에 있으며 모델/mapper 대용량 파일은 코드 패치에 포함하지 않는다.

## 결과

전체 260개를 완료했다. 새 mapper의 점추정은 세 지표 모두 기존 mapper보다 좋지만, 원래 target 대비 품질 손실은 남았다.

| 평가 | 원래 target | 기존 local k4 / fit50 | 공개 k4-n200 / fit120 |
| --- | ---: | ---: | ---: |
| ARC-Challenge acc_norm | 59/128, 46.09% | 42/128, 32.81% | 44/128, 34.38% |
| ARC 정확도 유지율 | 100% | 71.19% | 74.58% |
| HellaSwag acc_norm | 71/128, 55.47% | 54/128, 42.19% | 61/128, 47.66% |
| HellaSwag 정확도 유지율 | 100% | 76.06% | 85.92% |
| WikiText-2 perplexity (낮을수록 좋음) | 13.0364 | 14.7955 | 14.0856 |

공개 mapper의 chance-floor 보정 유지율은 ARC 44.44%, HellaSwag 74.36%다. WikiText-2 PPL은 원래 target보다 약 8.05% 높고, 기존 mapper보다 약 4.80% 낮다. 평균 NLL은 target 2.567746, 기존 2.694326, 공개 2.645154 nat/token이다. 네 구간만으로 모집단 PPL 신뢰구간을 주장하지 않는다.

Paired bootstrap 10,000회, seed 202608으로 얻은 정확도 차이의 95% 구간:

| 비교 | ARC 차이 [95% 구간], %p | HellaSwag 차이 [95% 구간], %p |
| --- | ---: | ---: |
| 공개 − 기존 mapper | +1.56 [-4.69, +7.81] | +5.47 [-1.56, +12.50] |
| 공개 − 원래 target | -11.72 [-20.31, -3.13] | -7.81 [-14.84, -1.56] |

기존 mapper 대비 향상의 구간은 0을 포함한다. 따라서 점추정 개선은 관측했지만, 이번 128문항 표본만으로 일반적인 향상을 확정하지 않는다. target 대비 두 정확도 손실의 구간은 0을 포함하지 않는다. 구간은 고정 mapper에서 문항만 재표집한 것이며 calibration seed의 불확실성을 포함하지 않는다.

문항별로 기존 대비 ARC는 10개를 새로 맞히고 8개를 새로 틀렸으며, HellaSwag는 각각 14개와 7개였다. 원래 target 대비 ARC는 8개를 새로 맞히고 23개를 잃었으며, HellaSwag는 각각 5개와 15개였다.

전체 문항의 원시 token log-probability에서 normalization, answer choice, 합계, PPL을 별도 스크립트로 재계산했다. `artifacts/public-ridge-benchmark/independent-comparison.json`에 기록했고 재계산 코드는 `artifacts/verify_public_ridge.py`이다. 기존과 새 cold 점수는 전부 같고, 새로 실행한 exact-cache 6개 control도 bitwise 일치했다. 후보 간격이 0.002 nat/character 미만인 11개 case/mode를 scalar로 다시 채점했으며 winner 변화는 없었다. 최대 token log-probability 차이는 0.002382 nat였다. 기존 Python 테스트 17개도 통과했고, 모델 쌍을 뒤집으면 config hash 검사에서 출력 생성 전에 거부하는 것을 확인했다.

### 공개 수치와의 격차 진단

같은 저장 점수에서 정규화만 token count로 바꾸는 사후 진단도 수행했다. HellaSwag는 target 73/128, 기존 55/128, 공개 54/128로 공개 유지율이 73.97%였다. ARC는 target 55/128, 기존 44/128, 공개 48/128로 87.27%였다. 이는 원래 character-normalized 평가를 대체하지 않는다. 원본 프로젝트의 HellaSwag 95.3%와의 격차를 정규화 방식 하나로 설명할 수 없다는 증거다.

이번에 공개 coefficient 자체를 가져왔으므로 '우리 fit50 학습만 잘못되어 격차가 생겼다'는 설명은 충분하지 않다. 그렇다고 nntrainer 런타임이 잘못됐다고 확정할 수도 없다. 남은 차이는 모델 forward/KV 수치(FP32·FP16 대 BF16), prefix 재계산 범위, 평가 문항, source/target KV 표현의 런타임 차이다. 다음 원인 분리는 동일 token IDs에 대해 Transformers/vLLM과 nntrainer의 레이어별 원래 KV 및 logits를 먼저 대조하고, 동일 mapper·동일 채점 조건으로 비교하는 것이다.

## 작업 비용 기록

이번 작업은 별도 subagent를 띄우지 않고 부모가 처리했다. 공개 학습 artifact와 검증된 cold 점수를 재사용했다. 추가 subagent 사용량은 없지만 부모 사용량은 작업별로 분리되지 않아 unknown이며, 이를 비용 0 또는 ChatGPT 요금 절감률로 표시하지 않는다.
