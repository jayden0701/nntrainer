# 작은 CPU cross-runtime 오차 진단

## 결론

제한된 입력에서 **nntrainer와 Transformers 사이의 큰 수치 불일치는 발견하지 못했다**. 기본 Qwen3 forward, KV export의 layer/head 배치, 공개 mapper cache의 target import를 의심할 뚜렷한 증거가 없었다. 이것이 모든 입력·긴 context·BF16 vLLM 경로의 정확성을 보증하지는 않는다.

## 범위

Native 소스는 push된 `3717dec94a3edeee86397dfff3d6cfe66e000385`의 CPU build를 사용했다.

- 이미 내려받은 동일 checkpoint의 Qwen3-0.6B와 Qwen3-1.7B를 재사용했다. 두 런타임의 config.json 바이트가 같음을 확인했다.
- Transformers 4.57.3 / PyTorch 2.9.1+cpu, CPU FP32, eager attention, threads 4. nntrainer도 FP32 모델과 기존 FP16 KV 경로를 사용했다.
- 동일 token IDs의 17토큰 HellaSwag, 22토큰 ARC, 128토큰 WikiText 기반 입력 세 개를 각 모델에서 비교했다. tokenizer/chat template을 새로 적용하지 않았다.
- 모든 28개 layer의 post-RoPE K와 V를 `[token, head, dim]`으로 맞춰 비교했다. 각 입력의 최대 8개 고정 target token을 teacher forcing해 로그확률을 비교했다. short 입력에서는 HF top-1 token에 대한 native 점수도 추가 확인했다.
- 128토큰 진단의 target IDs는 기존 WikiText benchmark의 suffix에서 가져왔다. 입력을 128토큰으로 줄였으므로 자연스러운 연속 문장의 PPL 평가가 아니라 동일 입력에 대한 수치 정합성 control이다.
- 추가로 공개 k4/n200 mapper로 127토큰 cache를 만들고, native import와 동일하게 FP16으로 반올림한 **같은 cache**를 Transformers DynamicCache와 nntrainer 양쪽에 넣었다. 마지막 context 1토큰과 고정 target을 양쪽에서 계산했다.

새 모델 다운로드, 재학습, 전체 벤치마크 재실행, C++ 변경은 하지 않았다. 참조 환경만 별도 `.venv-reference`에 설치했다.

## 결과

다음 값은 세 입력과 모든 layer에서의 최댓값이다. relative L2는 `||native-reference|| / ||reference||`이다.

| 모델 | K relative L2 최대 | V relative L2 최대 | target token 로그확률 차이 최대 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 0.0597% | 0.1093% | 0.00417 nat |
| Qwen3-1.7B | 0.0494% | 0.0951% | 0.00647 nat |

전체 layer의 K/V cosine은 0.9999994 이상이었다. 특정 초기 layer에서 크게 벌어지는 패턴은 없었다. 첫 layer의 HF KV를 FP16으로 반올림해서 비교하면 오차가 더 작아져, 그 지점에서는 native FP16 cache 반올림이 차이의 대부분을 설명했다. 이후 layer의 누적 차이까지 모두 같은 원인이라고 단정하지는 않는다.

**같은 mapped cache를 target에 넣은 control의 로그확률 차이는 최대 0.001249 nat**였다. 반면 Transformers 자체에서 원래 cache와 mapped cache를 바꾸면 해당 control의 토큰 로그확률은 최대 약 2.35 nat 달라졌다. 이 값은 방향이 섞인 점수 변화이며 정확도 저하량이 아니다. 이 사례에서는 mapper로 cache를 바꾸는 효과가 두 consumer 런타임 간 차이보다 훨씬 컸다.

## 해석과 한계

공개 프로젝트의 HellaSwag 95.3%와 nntrainer 동일조건 비교 85.9% 사이 격차를, 이번에 확인한 FP32 기본 forward나 cache import의 큰 오류로 설명할 증거는 얻지 못했다. 큰 코드를 고치거나 재학습을 시작할 이유도 현재는 없다.

아직 원인을 확인한 것은 아니다. 공개 실험의 BF16 forward/mapper, 마지막 최대 16토큰 재계산, 서로 다른 문항과 평가 코드 등은 남아 있다. 이번 control은 source와 target native KV가 HF FP32와 가깝고, **같은 mapped cache를 소비하는 동작도 가깝다**는 데 범위가 한정된다. GPU vLLM의 BF16 경로를 직접 실행하지 않았고 전체 vocabulary logit tensor를 비교한 것도 아니다.

요청한 가벼운 진단은 여기서 마친다. 더 조사할 경우에는 많은 모델/문항을 추가하기보다 동일 소수 입력의 BF16 참조와 prefix tail 재계산 길이를 따로 바꾸는 것이 다음 단계다.

## 재현과 증거

- [17·22토큰 원시 비교](validation/kv-transfer/runtime-short-summary.json)
- [128토큰 및 mapped-cache 비교](validation/kv-transfer/runtime-128-summary.json)
- [진단 스크립트](../tools/kv_transfer/diagnose_runtime.py)
- [참조 환경 버전](../tools/kv_transfer/requirements-reference.txt)

```sh
.venv-reference/bin/python tools/kv_transfer/diagnose_runtime.py \
  --source-hf /path/to/hf-qwen3-0.6b \
  --target-hf artifacts/models/hf-qwen3-1.7b \
  --case-set short --output artifacts/runtime-check-short-new

.venv-reference/bin/python tools/kv_transfer/diagnose_runtime.py \
  --source-hf /path/to/hf-qwen3-0.6b \
  --target-hf artifacts/models/hf-qwen3-1.7b \
  --case-set 128 --output artifacts/runtime-check-128-new
```

출력은 새 directory를 지정한다. 기존 native 모델, benchmark manifest, 공개 mapper artifact가 필요하다. 실제 측정은 `artifacts/runtime_diagnostic.py` 및 `runtime_diagnostic_128.py`로 실행했고, 같은 로직에 CLI 경로/입력 선택을 붙인 재현 스크립트를 저장했다. 재현 스크립트의 CLI와 Python compilation을 확인했다.

이번에도 별도 subagent는 사용하지 않았다. 부모의 task별 token/요금은 분리 계측할 수 없어 unknown으로 남긴다.
