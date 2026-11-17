# Ridge KV transfer: 논문 기준 평가

## 판정 기준과 이전 결과 정정

원 논문은 Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*, arXiv:2608.03893v1이다.
[원문 §3–4 및 Appendix D–E](https://arxiv.org/html/2608.03893v1).

논문의 보존 대상은 downstream task metric이다. 생성 토큰 ID가 완전히 같은지나 KV의 L2 오차만으로 판정하지 않는다. 따라서 이전 작은 실험의 첫 생성 토큰 불일치는 **품질 저하를 입증한 벤치마크 결과가 아니다**. 이전 비용 검증 보고서에서 이를 quality loss라고 표현한 부분도 이 기준으로 정정한다.

논문의 주 평가는 ARC-Challenge와 HellaSwag의 acc_norm, WinoGrande와 MMLU 5-shot의 acc, GSM8K 8-shot CoT의 정답률이며, WikiText-2 prefix-conditioned perplexity와 CoQA multi-turn도 사용한다. Accuracy retention은 transfer/target 정확도 비율이며 chance floor를 뺀 유지율도 제시한다. 모든 모델 쌍에서 높은 보존을 보장하는 주장은 아니다.

## 이번 검증의 범위

| 항목 | 이전 smoke test | 이번 CPU 평가 |
| --- | --- | --- |
| source→target | Qwen3 0.6B→1.7B | 동일; 논문의 본 실험에 없는 모델 쌍 |
| 학습 데이터 | 합성 입력 6개, 909토큰 | 고정 revision의 FineWeb-Edu 50개 × 1,024토큰 |
| token sampling | 전체 | 입력마다 stride 4, 12,800 observations |
| source feature | 같은 번호의 head만 | 선택 layer들의 모든 KV head를 연결 |
| layer 선택 | 같은 깊이의 1개 | calibration-only OLS R²로 top-4 |
| Ridge penalty | 1 | 0.01, centered summed SSE, bias 비정규화 |
| K 처리 | inverse RoPE→mapping→target RoPE | 동일 |
| 평가 | 짧은 생성 한 건 | ARC 128, HellaSwag 128, WikiText-2 4구간 |

논문 본 실험의 500개 calibration, 모델별 k sweep, 대형 모델, BF16 forward와 동일한 재현은 아니다. 이번에는 작은 모델에서 논문 구조를 적용한 별도 검증이며 k=4를 평가 결과 확인 전에 고정했다. 모든 출력 head를 한 번에 푸는 행렬 Ridge는 동일 source features와 penalty를 쓰는 독립적인 target-head 회귀와 수학적으로 동등하다.

평가용 샘플은 seed 202608로 사전에 고정했다. 128/128/4 manifest SHA-256은 `b2b3d3dbdc6fab83704fff1fe28a5e2450dc3c2f1beddc1607a789775bb1f3c8`이다. calibration과 benchmark는 서로 다른 데이터셋으로 구성하며 benchmark 점수로 layer나 k를 선택하지 않는다.

## 채점 검증

채점은 completion mode이며 chat template을 넣지 않는다. ARC/HellaSwag의 prompt와 character-length normalization은 lm-evaluation-harness의 정의에 맞췄다. 정답 후보 전체를 teacher forcing하고 softmax 전 원래 logits에서 로그확률을 계산한다. EOS도 일반 gold token으로 채점하며 조기 종료하지 않는다.

동일 스레드 설정에서 cold→정확한 cache import→cold의 점수가 일치하는 control을 수행했다. tied FP32 Qwen head의 retained hidden states를 묶어서 projection하는 별도 평가 경로도 구현했다. 1.7B의 70-token cross-block control에서 scalar 대비 최대 차이는 0.00710 nat/token, sum 차이는 약 -0.00142였다. 이 경로는 기본 추론 동작을 바꾸지 않는다. 좁은 정답 후보 간격은 별도로 표시한다.

WikiText-2는 고정된 non-overlapping 2,048-token 구간에서 뒤 1,024토큰을 채점한다. 첫 suffix logit을 구하기 위해 prefix 마지막 한 토큰을 target에서 재계산하므로 실제 import되는 prefix는 1,023토큰이다. 이 한 토큰 경계 처리는 논문에 상세히 명시된 동일 구현이라고 주장하지 않는다.

가중치·activation은 FP32이고 native KV는 FP16이다. F32 snapshot은 FP16 값을 확장한 값이다. calibration native thread 설정은 8, benchmark는 4이며 각 cold/exact/transfer 비교 안에서는 thread 설정을 고정한다. 동시 실행과 일시 정지 구간이 있으므로 수집된 wall time으로 가속률을 주장하지 않는다.

## 재현

소스 및 실행 방법은 [benchmark 안내](../tools/kv_transfer/benchmark.md)와 [mapper 안내](../tools/kv_transfer/README.md)를 참조한다. 데이터 준비는 `tools/kv_transfer/prepare_paper_data.py`, 정확한 raw-token calibration 수집은 `tools/kv_transfer/collect_calibration.py`를 사용한다. 데이터 원본 revision과 SHA-256은 `artifacts/ridge-paper-data/provenance.json`에 저장했다.

실행에 사용한 Python dependency 버전은 `tools/kv_transfer/requirements-eval.txt`에 고정했다. 데이터 준비는 Hugging Face의 고정 revision parquet 파일을 사용한다. FineWeb-Edu 전체가 아닌 명시된 한 shard에서 seed 20261117로 1,024토큰 이상인 문서 50개를 선택했다.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python tools/kv_transfer/kv_transfer.py fit \
  artifacts/ridge-paper-calibration/dataset.json artifacts/ridge-paper-mapping \
  --layout all-head --layer-selection ols-r2 --k 4 --alpha 0.01 \
  --token-stride 4 --validation-fraction 0 --block-tokens 1024 --memory-mib 1024
```

실험 산출물은 `artifacts/ridge-paper-*`, `artifacts/ridge-benchmark`, `artifacts/score-control`에 보관한다. 원본 모델·데이터·cache는 코드 패치에 포함하지 않는다.

## 측정 결과

이번 설정에서는 task 성능이 충분히 보존됐다고 보기 어렵다. 생성 토큰의 동일성 대신 원 논문과 같은 종류의 downstream metric으로 확인한 결과다.

| 평가 | 원래 target | Ridge transfer | 변화 | 정확도 유지율 | chance 보정 유지율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge acc_norm, 128문항 | 59/128, 46.09% | 42/128, 32.81% | -13.28%p | 71.19% | 37.04% |
| HellaSwag acc_norm, 128문항 | 71/128, 55.47% | 54/128, 42.19% | -13.28%p | 76.06% | 56.41% |
| WikiText-2 perplexity, 4,096토큰 | 13.0364 | 14.7955 | +13.49% (악화) | — | — |

정확도 차이의 paired bootstrap 95% 구간은 ARC [-21.88, -4.69]%p, HellaSwag [-21.09, -5.47]%p였다. 고정된 mapper에서 문항을 재표집한 구간이며 calibration/학습 seed의 불확실성까지 포함하지 않는다. 두 평가 모두 0을 포함하지 않았다. ARC에서 target만 맞힌 문항은 25개, transfer만 맞힌 문항은 8개였으며 HellaSwag에서는 각각 23개, 6개였다. WikiText-2 평균 NLL은 2.567746에서 2.694326으로 0.126580 nat/token 증가했다. 네 구간만의 PPL 결과에 모집단 신뢰구간을 부여하지 않는다.

원래 target cache를 export/import한 6개 control은 모두 cold target과 토큰별 점수가 완전히 같았다. 이 control은 task별 두 사례의 경로 검증이며 전체 benchmark의 exact 정확도/PPL로 해석하지 않는다. 정답 후보 간격이 0.002 nat/character 미만인 11개 case/mode를 scalar 경로로 다시 채점했으며 정답 선택 변화는 없었다(최대 token log-probability 차이 0.003749). 따라서 확인한 export/import 경로 오류나 가까운 후보에 대한 block 채점 오차가 이번 정확도 하락을 설명하지는 않는다.

전체 260개 결과를 원시 token log-probability와 고정 manifest에서 부모 agent가 별도로 재집계했고, counts·정규화·정답 선택·PPL·6개 exact control을 확인했다. CPU build, KV core 64개 및 Python 17개 테스트가 통과했다. 결과는 `artifacts/ridge-benchmark/summary.json`, 독립 검증은 `parent-independent-check.json`, scalar 검증은 `audit-scalar/summary.json`에 있다.

이번 결과는 **Qwen3 0.6B→1.7B, N=50, k=4**에 대한 부정적 결과다. 원 논문에서 평가한 대형 모델 쌍의 재현 실패라고 해석해서는 안 된다. 원인을 분리하려면 동일 평가 manifest를 유지하면서 calibration 500개, 별도 validation에서 선택한 k, source-only 및 직접 복사 baseline을 비교해야 한다. mapper 학습 시간은 약 288.65초, 파일 총량은 약 939.8 MB였다. CPU 동시 실행 영향을 배제한 시간 측정이 아니므로 prefill 가속이나 비용 회수 여부는 아직 판정하지 않는다.

WinoGrande·MMLU·GSM8K·CoQA와 원 논문의 대형 모델 쌍은 이번 평가 범위에 포함하지 않는다.
