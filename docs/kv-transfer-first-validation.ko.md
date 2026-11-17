# KV transfer PoC 1차 검증 — 2026-11-17

PR #4481 head `5e97254e764aa30db4223c33d0dc3b0fded091b8`에서 CPU 경로로 실험했다. NPU 경로는 실험하지 않았다. 작업 브랜치는 `codex/kv-transfer`이며 기존 Windows 작업 디렉터리는 수정하지 않았다.

## 구현과 검증

- PR의 contiguous KV manager에 CPU graph를 연결했다. 기본 경로는 기존대로이며 `NNTR_KV_CORE_CPU=1`로 선택한다.
- committed cache를 snapshot으로 내보내고 복원한다. import가 실패하면 기존 cache가 유지되며 graph의 메모리 binding을 보존한다.
- head별 또는 모든 head를 이용하는 affine Ridge 학습, layer 대응, K의 inverse RoPE/target RoPE, 입력 단위 train/holdout 분리를 구현했다.
- Qwen3 sharded Safetensors 변환을 지원하도록 기존 converter를 확장했다.
- 최종 core 테스트 64개, mapper 테스트 6개 통과. 부모 agent가 실행 및 결과를 별도 확인했다.
- 정상 변환한 0.6B에서 legacy full prefill, core full prefill, snapshot import, 기존 native cache load의 생성 토큰 ID가 모두 일치했다. 동일 프로세스의 reset 요청 두 번도 일치했다.

FP32 가중치·activation 설정이지만 이 경로의 실제 KV 저장은 FP16이다. F32 snapshot은 FP16을 확장한 값이며 손실된 정밀도를 복구하지 않는다. 모델 ID는 호출자가 지정하는 계약이고 자동 weights hash 검증은 아직 없다. paged model inference, 다른 family, C2C/MoT 학습 및 DroidSpeak 부분 재계산은 이번 단계에 포함하지 않았다.

## 첫 교차 모델 실험

Qwen3 0.6B → 1.7B, headwise k=1, Ridge alpha=1, 8개 합성 입력을 사용했다. 6개 입력/909토큰으로 학습하고 2개 입력/244토큰을 holdout으로 분리했다.

| 측정 | K 상대 L2 | V 상대 L2 |
| --- | ---: | ---: |
| Ridge, holdout 전체 | 0.207512 | 0.726171 |
| 단순 복사, holdout 첫 입력 | 0.473239 | 1.049721 |
| 단순 복사, holdout 둘째 입력 | 0.471740 | 1.046490 |

1.7B 전체 prefill과 정확한 native cache 복원의 생성 ID는 일치했다. Ridge와 단순 복사는 첫 생성 ID부터 달랐다. tensor 오차는 줄었지만 생성 품질 보존을 확인한 결과가 아니다. 생성 비교는 holdout 중 한 입력의 짧은 continuation에 한정되며, 정확도 평가나 성공률 추정으로 사용할 수 없다.

짧은 입력에서 단 한 번 측정한 시간으로 속도 향상을 주장하지 않는다. 실제 이득 판단에는 source prefill, mapper, import, target suffix와 첫 토큰까지를 모두 포함하고, source cache가 이미 있는 경우와 새로 만드는 경우를 분리해야 한다. 다음 비교는 calibration 확대, all-head/layer 조합, 일부 target 재계산 순으로 진행하는 것이 타당하다.

## 모델 이력

| 모델 | Hugging Face revision | 변환된 FP32 bin SHA-256 |
| --- | --- | --- |
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | 88ffa34cf3c2d5900b62730416aac3f1912df9e8015c6c749f3fd37814bf70ef |
| Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 175b1ec912111c9b835b8fa8b95de95f30039a5c419c7de0c5f8a9d2830703bb |

처음 재사용한 0.6B bin은 현재 graph의 MLP weight 순서와 맞지 않았다. byte 비교에서 첫 차이를 `model.layers.0.mlp.gate_proj.weight`에서 확인했다. 현재 converter로 별도 파일을 생성한 뒤 `2 + 2 = 4` 정상 출력을 확인하고 실험을 다시 했다. 이전 `artifacts/native-identity` 결과는 모델 품질 근거에서 제외한다.

## 재현 환경과 명령

WSL2 Ubuntu 26.04, x86 Intel Core Ultra 5 250K Plus, RAM 약 35 GiB, GCC 15.2, Meson 1.10.1, Ninja 1.13.2. native inference의 `NNTR_NUM_THREADS=8`, mapper OpenBLAS threads=1을 사용했다. Python은 NumPy 기반이며 GPU 학습은 사용하지 않았다.

```sh
git submodule update --init --recursive --depth 1 Applications/CausalLM/xgrammar
meson setup build-cpu -Denable-transformer=true -Denable-fp16=false -Denable-opencl=false -Denable-cuda=false -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-test=true -Dthread-backend=omp
meson configure build-cpu -Dcpp_args=-Wno-error=overloaded-virtual
ninja -C build-cpu -j8 Applications/CausalLM/nntr_causallm test/unittest/unittest_kv_cache_core
build-cpu/test/unittest/unittest_kv_cache_core
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tools/kv_transfer -v
```

GCC 15의 기존 `overloaded-virtual` 경고 때문에 해당 경고만 오류 승격을 해제했다. C++ 수정 파일은 clang-format 14로 정리했다. 다른 플랫폼의 전체 CI는 실행하지 않았다.

[CPU snapshot 계약](kv-transfer-poc.md), [mapper 및 재현 harness](../tools/kv_transfer/README.md)를 참고한다. 로컬 결과는 `artifacts/corrected-native`, `artifacts/native-pilot`, `artifacts/results`에 보관한다. 모델·snapshot·개인 agent 로그는 패치에 포함하지 않는다.
