# 다른 모델 계열의 KV 캐시 융합 실험

Qwen3-0.6B의 KV를 Llama 계열 구조의 SmolLM2-135M KV에 더하는 소규모 C2C 기반 실험이다. 두 모델을 고정하고 융합기만 CPU에서 학습했다. [cross_family](../tools/kv_transfer/cross_family)의 스냅샷 적용 도구는 nntrainer의 v1 FP32 캐시 포맷을 사용한다.

## 구현

- Qwen3-0.6B는 28층 × 8 KV 헤드 × 128차원, SmolLM2-135M은 30층 × 3 KV 헤드 × 64차원이다. 마지막 층부터 짝지어 Qwen3 0–27층을 SmolLM2 2–29층과 연결한다. SmolLM2의 첫 두 층은 원래 캐시를 사용한다.
- 수신 토큰 하나를 디코드한 뒤 송신 모델 토크나이저로 재인코딩한다. 후보가 여럿이면 디코드 문자열이 가장 긴 후보를 고르고, 동률이면 앞선 후보를 사용한다. 결과로 만든 송신 ID 열을 수신 ID 열과 같은 위치에서 prefill한다. 일반 텍스트를 사용해 채팅 템플릿 패딩은 생략했다. 짧은 문자열 단위의 정렬은 문맥 전체를 Qwen 토크나이저로 인코딩한 것과 달라질 수 있다.
- 연결된 각 층의 K와 V에 독립적인 융합기를 둔다. 정규화한 송신·수신 KV를 이어 붙여 폭 32의 SiLU 투영을 적용한다. 이 특징으로 수신 KV와 같은 형상의 residual과 토큰별·헤드별 sigmoid 가중치를 계산한다. 층 및 K/V별 sigmoid gate와 수신 KV 헤드의 RMS로 residual을 스케일한 뒤 원래 수신 KV에 더한다. 공식 융합기는 더 깊은 SwiGLU 블록, 온도를 낮추는 Gumbel-sigmoid, 추론용 이산 gate를 사용한다. 여기서는 학습과 추론에서 같은 연속 sigmoid를 사용하고 각 모델의 post-RoPE K를 직접 입력한다. [C2C 논문 §3.3, 부록 A.1](https://arxiv.org/html/2510.03215) · [공식 정렬 코드](https://github.com/thu-nics/C2C/blob/5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf/rosetta/model/aligner.py) · [공식 융합 코드](https://github.com/thu-nics/C2C/blob/5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf/rosetta/model/projector.py)
- 양쪽 모델로 컨텍스트를 각각 prefill한다. 수신 모델의 마지막 컨텍스트 토큰은 캐시 밖에 두고 융합 캐시와 함께 입력해 첫 후속 토큰부터 채점한다. 후속 텍스트의 teacher-forced next-token loss를 수신 모델의 연산을 거쳐 역전파한다. 추론에서도 송신 prefill, 수신 prefill, 융합, 수신 후속 처리가 모두 필요하다.

## 실험 설정

| 항목 | 설정 |
| --- | --- |
| 송신 모델 | Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca |
| 수신 모델 | HuggingFaceTB/SmolLM2-135M 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 |
| 하드웨어 | x86_64 Intel Core Ultra 5 250K Plus, PyTorch CPU 스레드 4개 |
| 학습 | FineWeb-Edu 문서 32개, 95 캐시 토큰 + 앵커 1개 + 후속 정답 16개 |
| 검증 | 같은 구성의 별도 FineWeb-Edu 문서 8개, 검증 NLL로 학습 체크포인트 선택 |
| 최적화 | seed 4217, 배치 4, AdamW, 학습률 0.0003, 8 epochs |
| 평가 | WikiText-2 test 문단 32개와 HellaSwag validation 문제 32개 |
| 샘플링 | 태스크·seed·원본 행 번호의 SHA-256 순서로 조건을 충족하는 평가 행 선택 |

HellaSwag는 보기별 평균 토큰 NLL이 가장 낮은 보기를 선택한다. 문맥과 보기 경계에서 토큰화가 바뀌는 행, 문맥이 96 토큰을 넘는 행, 보기가 64 토큰을 넘는 행은 건너뛴다. WikiText-2는 최소 112개 토큰이 있는 문단에서 연속된 첫 112개 토큰을 사용한다. 평가 행의 선정 기준은 학습 전에 고정했다.

`native`는 SmolLM2 자체 KV를 사용한 HF 수신 기준선이다. `zero_source`는 학습된 융합기를 유지하고 송신 KV를 0으로 채운다. `wrong_source`는 WikiText-2에서 같은 길이의 이전 평가 행 KV를 송신 입력으로 사용한다. 첫 행에는 마지막 행의 KV를 넣는다. 초기 residual이 0일 때 수신 캐시가 정확히 보존되는지, 첫 WikiText-2 행에서 수신 전체 순전파와 분할 순전파의 logit 오차가 0.0003 이하인지 검사한다.

## HF 평가 결과

| 수신 캐시 조건 | WikiText-2 평균 NLL | HellaSwag 정답 보기 평균 NLL | HellaSwag 정답 수 |
| --- | ---: | ---: | ---: |
| SmolLM2 자체 KV | 2.981126 | 2.502158 | 15/32 |
| Qwen3 + SmolLM2 융합 | 2.980801 | 2.502666 | 15/32 |
| 송신 KV 0 | 2.980986 | 2.502247 | 15/32 |
| 다른 문단의 송신 KV | 2.980720 | — | — |

WikiText-2의 융합-기준선 차이는 −0.000325 nats/token이었다. 문단 단위 쌍체 부트스트랩 95% 구간은 [−0.001006, +0.000341]이었다. 다른 문단의 송신 KV를 넣어도 비슷한 변화가 나왔으며, HellaSwag 정답 수는 그대로였다. 이 표본에서 일치하는 송신 문맥의 품질 이득은 확인되지 않았다.

기준 캐시 대비 상대 L2 변화는 WikiText-2 평균 0.143%, HellaSwag 평균 0.144%였다. 첫 WikiText-2 행의 수신 전체·분할 logit 최대 오차는 5.82×10⁻⁵였다. 저장한 HF 캐시와 체크포인트에 별도로 적용 CLI를 실행해 캐시 두 평면과 메타데이터가 원래 융합 출력과 바이트 단위로 일치함을 확인했다.

학습 가능한 파라미터는 2,541,280개였다. 검증 NLL은 초기 2.812473, 선택된 1 epoch 2.813111, 8 epochs 3.035206이었다. 검증 NLL은 학습 전이 가장 낮았으며, 학습된 변환의 효과를 측정하기 위해 학습 완료 체크포인트 중 최저 NLL인 1 epoch를 평가했다. 학습용 컨텍스트 캐시 생성에는 송신 11.67초와 수신 3.03초가 들었고, 64 스텝과 검증에는 20.53초가 들었다.

평가 컨텍스트 64개에서 CPU 단계 시간 합계는 송신 prefill 18.59초, 수신 prefill 5.06초, 융합 0.65초였다. 전체 160개 후속 시퀀스의 수신 처리는 자체 KV 8.55초, 융합 KV 8.98초였다. 이 실행은 다른 CPU 작업과 병행하면서 단계 시간을 합산했으며, 모델 로드·파일 입출력이나 별도의 텍스트 전달 비교를 포함한 종단간 지연은 측정 범위 밖이었다.

## 재현

레포 루트의 기존 CPU 참조 가상환경과 FineWeb-Edu·WikiText-2·HellaSwag 자료를 사용한다. 데이터 디렉터리가 없으면 `requirements-eval.txt`를 설치한 환경에서 다음 명령으로 생성한다. [고정 데이터 다운로드 및 샘플 추출 코드](../tools/kv_transfer/prepare_paper_data.py)

~~~bash
python tools/kv_transfer/prepare_paper_data.py \
  /path/to/hf-qwen3-0.6b/tokenizer.json artifacts/ridge-paper-data
~~~

~~~bash
.venv-reference/bin/hf download HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --local-dir artifacts/models/hf-smollm2-135m \
  --include '*.json' 'merges.txt' 'model.safetensors'

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-reference/bin/python -m tools.kv_transfer.cross_family.train \
  --source /path/to/hf-qwen3-0.6b \
  --receiver artifacts/models/hf-smollm2-135m \
  --data artifacts/ridge-paper-data \
  --output artifacts/cross-family-c2c
~~~

`artifacts/cross-family-c2c/results.json`에 학습·검증 이력, 평가 행별 NLL, 모델·데이터 SHA-256, 단계별 시간이 있다. `bridge/case.json`은 첫 WikiText-2 평가 행의 양쪽 prefix ID, 후속 입력 및 label ID, HF 기준 logit의 argmax와 토큰별 NLL을 보관한다. `bridge/{source_hf,receiver_hf,fused_hf}`에는 해당 캐시가 v1 FP32 `[layer, token, kv_head, head_dim]` 형식으로 저장된다.

다른 실행기에서 캐시를 만들 때 `case.json`의 송신·수신 ID를 각각 prefill하고, 0부터 시작하는 동일한 위치 범위로 내보낸다. 두 캐시의 모델 ID는 표에 명시한 Hugging Face 모델명과 일치해야 한다. 융합 결과는 수신 ID와 메타데이터를 사용한다.

~~~bash
.venv-reference/bin/python -m tools.kv_transfer.cross_family.fuse \
  --checkpoint artifacts/cross-family-c2c/fuser.pt \
  --source /path/to/source_snapshot \
  --receiver /path/to/receiver_snapshot \
  --output /path/to/fused_snapshot
~~~

평가 NLL은 FP32 logit의 어휘 축을 FP64로 합산한다. 첫 평가 행은 저장한 원시 logit에서 토큰별 NLL을 독립적으로 재계산할 수 있다. 각 WikiText-2 행의 정답 NLL과 HellaSwag 보기별 NLL을 집계해 표의 값을 확인할 수 있다.

~~~bash
.venv-reference/bin/python - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path("artifacts/cross-family-c2c")
case = json.loads((root / "bridge/case.json").read_text())
labels = np.array(case["labels"])
for mode in ("native", "fused"):
    logits = np.fromfile(root / f"bridge/{mode}_logits.f32", dtype="<f4")
    logits = logits.reshape(len(labels), 49152).astype(np.float64)
    log_partition = np.logaddexp.reduce(logits, axis=1)
    nll = log_partition - logits[np.arange(len(labels)), labels]
    np.testing.assert_allclose(nll, case[mode]["loss"], rtol=0, atol=1e-10)
    print(mode, "bridge mean nll", nll.mean())
for task, rows in json.loads((root / "results.json").read_text())["examples"].items():
    for mode in rows[0]["nll"]:
        nll = np.mean([row["nll"][mode][row["gold"]] for row in rows])
        correct = sum(np.argmin(row["nll"][mode]) == row["gold"] for row in rows)
        print(task, mode, nll, (correct if task == "hellaswag" else None))
PY
~~~

## nntrainer KV manager에서의 실행

PR #4481 기반 CPU 경로에서 같은 평가 문항을 추가로 실행했다. SmolLM2의 FP32 가중치를 변환하고, 기존 snapshot 입출력 검증이 unscaled, full-dimension, split-half RoPE의 Llama 구조도 허용하도록 확장했다. Qwen3와 SmolLM2 모두 **nntrainer가 실제 계산한** KV를 내보내어 학습된 융합기를 적용한 뒤, 결과를 SmolLM2의 core KV manager에 다시 넣었다. 융합기 자체는 외부 PyTorch CPU에서 실행한다.

| 실행기 / 조건 | WikiText-2 평균 NLL | HellaSwag 정답 보기 평균 NLL | HellaSwag 정답 수 |
| --- | ---: | ---: | ---: |
| nntrainer 자체 KV | 2.981197 | 2.502140 | 15/32 |
| nntrainer 양쪽 KV → 융합 → 재주입 | 2.980820 | 2.502680 | 15/32 |

32개 WikiText 문단에서 융합-기준선 차이는 −0.000377, 쌍체 bootstrap 95% 구간은 [−0.001047, +0.000276]이었다. HF와 같이 품질 향상을 입증하지 못했다. 두 실행기의 보기별 평균 NLL 차이는 자체 KV에서 최대 0.000898, 융합에서 최대 0.001018이었다. 모델 가중치와 활성값은 FP32이고 nntrainer의 물리적 KV 저장은 FP16이다.

- 양쪽 토크나이저와 정렬된 송신 ID를 HF 실험과 64개 문항 모두 대조하여 일치함을 확인했다.
- WikiText와 HellaSwag 각 3개 문항의 정확한 자체 KV 재주입은 모든 보기에서 cold 점수와 비트 단위로 일치했다.
- 첫 문단의 같은 FP16 반올림 융합 KV를 HF와 nntrainer에 넣었을 때 정답 토큰 로그확률 차이는 최대 0.000857이었다. HF에서 생성한 융합 KV를 넣은 대조군은 최대 0.000777이었다.
- CPU 실행 파일 빌드, core KV 테스트 64개, 기존 Python KV/벤치 테스트 17개가 통과했다. C++ 변경은 clang-format 14.0.6을 적용했다. scaled, interleaved, partial-rotary 설정을 별도로 주입한 export 거부 검사 3개도 통과했다.

64개 문항에서 프로세스 시작을 포함한 source KV export는 56.53초, receiver export는 17.27초, snapshot 파일 I/O를 포함한 융합은 2.28초였다. cold·fused 및 6개 exact 대조군 전체 채점은 174.46초였다. 공개 Ridge 벤치와 병행한 측정이므로 순수 모델 속도나 가속비로 해석하지 않는다.

[HF 평가 원시 점수](validation/kv-transfer/cross-family-hf-results.json), [nntrainer 평가 원시 점수](validation/kv-transfer/cross-family-native-results.json), [동일 캐시 런타임 대조](validation/kv-transfer/cross-family-runtime-bridge.json)에 수치와 대조군을 보관했다.

SmolLM2 native 모델 디렉터리는 다음과 같이 준비할 수 있다. 원본 HF 디렉터리의 config/tokenizer 파일은 수정하지 않는다.

~~~bash
mkdir -p artifacts/models/smollm2-135m-fp32
.venv/bin/python tools/kv_transfer/convert_llama_fp32.py \
  --source artifacts/models/hf-smollm2-135m \
  --output artifacts/models/smollm2-135m-fp32/smollm2-135m-fp32.bin
.venv/bin/python - <<'PY'
import json, shutil
from pathlib import Path
src = Path('artifacts/models/hf-smollm2-135m')
dst = Path('artifacts/models/smollm2-135m-fp32')
for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
    shutil.copyfile(src / name, dst / name)
config = dict(model_type='CausalLM', model_tensor_type='FP32-FP32',
              model_file_name='smollm2-135m-fp32.bin', fc_layer_dtype='FP32',
              embedding_dtype='FP32', lmhead_dtype='FP32', lora_rank=0,
              lora_alpha=0, lora_target=[], bad_word_ids=[], fsu=False,
              num_to_generate=1, init_seq_len=128, max_seq_len=2048,
              batch_size=1, tokenizer_file='tokenizer.json', skip_prefill=False)
(dst / 'nntr_config.json').write_text(json.dumps(config, indent=2))
PY
.venv-reference/bin/python -m tools.kv_transfer.cross_family_native \
  --executable build-cpu/Applications/CausalLM/nntr_causallm \
  --source artifacts/models/qwen3-0.6b-fp32 \
  --receiver artifacts/models/smollm2-135m-fp32 \
  --checkpoint artifacts/cross-family-c2c/fuser.pt \
  --data artifacts/ridge-paper-data \
  --hf-results artifacts/cross-family-c2c/results.json \
  --output artifacts/cross-family-native-benchmark
~~~

출력 경로는 새 디렉터리여야 한다. snapshot 및 모델 파일은 크기 때문에 Git에 넣지 않으며, 위 명령과 고정 revision으로 다시 만들 수 있다.
