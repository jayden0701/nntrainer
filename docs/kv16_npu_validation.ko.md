# KV16 NPU 통합 구현·검증 기록

기준: PR #4481 `5e97254e764aa30db4223c33d0dc3b0fded091b8`.
작업: `C:\nntrainer-kv16-npu`, `codex/kv16-npu-integration`.
검사일: 2026-11-10. 실제 NPU와 CPU/GPU 전체 모델 추론은 실행하지 않았다.

## 구현된 경로

| 경계 | 실제 연결 |
| --- | --- |
| C++ codec | `Applications/CausalLM/kv16_lossless.{h,cpp}`; Python wire v1과 동일한 RAW/fixed/sparse, 정수 predictor, CRC/길이/offset 검증 |
| 모델 eligibility | `Gemma4HtpCausalLM::allocateAndBindKVCache()`; global, unshared, K/V projection 공유, HTP 경로만 configure |
| runtime parameter | `HtpModelBinding::f32(layerN_qkv:k_norm_gamma)`에서 FP32 bits 복사, 실제 head geometry와 split-half rotary map 사용 |
| prefix capture | `KVCacheManager::capturePrefix()`; 작은 전체 backing만 채택, 원본 vector/borrowed owner release, immutable page 재사용 |
| 검증·복원 | `materializePrefix()`가 모든 compressed page를 별도 owned raw scratch에 완성한 뒤 live cache 복원 |
| cache 선택 | `restoreDevicePrefill()`이 model shape/gamma/map, page CRC, calibration pointer를 session publication 전에 검사하고 실패를 `PrefixUnavailable`로 처리 |
| Flash | raw-only 파일 v1 유지; compressed/raw 혼합 파일 v2, 실제 payload+record framing으로 budget 계산 |
| baked asset | Writer와 manifest loader가 v2를 지원; 등록 시 header/extent만 검사하고 payload/CRC 읽기는 hit 시점에 수행 |
| opt-out restore | 같은 모델의 v2 baked asset을 codec OFF에서 읽으면 검증 후 raw page로 보관·복원 |
| 기기 증거 | checkpoint JSONL, 전체 seed 입력 JSONL, checkpoint/full scope를 분리한 비교 도구 |

기본 DSP paged INT8, quantization/calibration 수식, RoPE/projection, HMX/HVX kernel과
attention 누산 순서는 변경하지 않았다. C++ codec의 predictor도 hardware FP mode에
의존하지 않는다. 새 모델 파일을 다운로드하거나 CPU/GPU 모델 추론을 시도하지 않았다.

## 실행 결과

### C++ / Python 교차 검사: PASS

`build/kv16-codec-sanitize/report.json`, 현재 codec/helper/reference source hash와 일치.

- predictor **871,968쌍**: 전체 FP16 bit pattern × gamma 13종 + 임의 20,000쌍.
- **41개 wire fixture**: C++ decode가 원본 K/V를 복원하고 re-encode가 Python blob과
  바이트 단위로 일치. 모든 FP16 원본 패턴, NaN payload, sparse, per-head gamma,
  비연속 rotary map, 세 mode 혼합, 빈 입력과 partial tail 포함.
- **745개 잘못된 입력**: checksum 재작성, header/index/length/mode/bitmap/특수 경계.
- 실제 변경 codec/checker를 ASan/UBSan으로 compile/run. CPU/GPU 모델 추론 및 DSP
  실행은 이 검사에 포함되지 않음.

최종 container 크기를 미리 reserve해 vector growth capacity가 저장 이득을
상쇄하지 않도록 했다. `storedBytes()`는 wire payload 크기이며 allocator/object
overhead를 뜻하지 않는다. prefix 보관량 API는 encoded vector capacity를 반영한다.
runtime gamma/map의 layer별 configuration과 raw fallback의 shared attempt hint,
Page object와 allocator overhead는 별도다. 원본 RPC slab을 다른 view가 pin하면
압축 payload 감소가 곧 시스템 memory 감소는 아니다.

### Matching-source prefix/Flash/baked 회귀

최종 **다섯 binary 모두 PASS**, ASan/UBSan 오류 없음.
결과와 명령은 `build/kv16-integration-verified/report.json`에 기록했다.
현재 source와 다른 engine을 연결한 실행은 성공으로 인정하지 않는다.
runner는 compile source/ABI header hash와 stale library를 검사하며 실패 보고서도
보존한다. 포맷 정리 후 engine source가 달라진 첫 실행은 의도대로 FAIL했고,
최종 source를 격리 Linux checkout에 복사해 코어를 다시 빌드했다.

matching host engine은 `/home/jayden/nntrainer-kv16-npu-core/build/kv16-core`다.
actual source에서 코어와 `libnntrainer.so`를 빌드했다. 기존 사용자의
engine/build와 PR baseline checkout을 덮어쓰지 않았다. vendor source는 수정하지
않았으며 OS package/SDK/model 설치도 하지 않았다.

새 storage code와 checker에는 ASan/UBSan을 적용했다. matching core library
자체는 별도의 debugoptimized host build이므로 전체 engine이 sanitizer로
instrument되었다고 해석하지 않는다.

실행 대상은 다음 다섯 개다.

- `kv16_prefix_check`: opt-out, mixed/tail, gamma mismatch, 늦은 page 손상 시
  live bytes/position 보존, shared page, RPC owner release, FP32/window/alias 거부.
- `prefill_prefix_check`: 기존 prefix/window/borrowed view/freeze/restore 회귀.
- `prefix_flash_check`: 기존 raw v1/LRU/budget/blob/worker 실패 회귀.
- `kv16_flash_check`: mixed raw/v2, physical bytes budget, record/footer/CRC 손상과
  truncated/trailing 입력의 cache miss 처리.
- `baked_prefix_check`: 실제 Writer→manifest→lazy payload load, compressed v2와
  raw/sliding page, calibration/identity, 잘못된 extent와 payload checksum 거부.

### 기기 증거 비교 도구: PASS

`build/kv16-tools-tests.log`: **15개 Python 검사 통과**. malformed/missing/empty
증거, codec 미실행, 실제 CLI 종료 코드, checkpoint/seed-input/session ID mismatch,
기본 full scope의 INCOMPLETE 처리를 확인했다.

FNV checkpoint/seed-input scope의 PASS는 해당 진단 관측값의 일치다. DSP INT8
readback/scales/state와 logits, 실행한 binary/장치의 provenance 없이 full PASS로
올리지 않는다. RPC 직전 입력 hash는 DSP가 실제 저장한 INT8 bytes의 증거가 아니다.

### 모델 연결 문법 검사

`build/model-syntax.json`에 저장한 native `-DENABLE_HEXKL` syntax 검사에서
`causal_lm.cpp`, `gemma4_htp_causallm.cpp`, `htp_attention_state.cpp`가 모두 통과했다.
이는 Android/Hexagon cross build나 실제 linkage 검증을 대신하지 않는다.
변경한 C/C++에는 repo 설정의 clang-format **14.0.6**을 적용했다.

## 재현

```bash
python3 test/htp/host/run_kv16_codec_check.py --sanitize \
  --output build/kv16-codec-sanitize
python3 -m unittest discover -s tools/kv16 -p 'test_*.py' -v
python3 test/htp/host/run_kv16_integration_checks.py \
  --engine-build /absolute/path/to/matching-meson-core-build --sanitize \
  --json-include /absolute/path/containing/json.hpp \
  --output build/kv16-integration-verified
```

새로운 output directory를 사용한다. runner는 기존 결과를 덮어쓰지 않는다.
`--json-include`를 생략하면 baked manifest 검사는 미실행으로 기록된다.
dependency는 자동 다운로드하지 않는다. 실제 Android build/deploy, 환경 변수,
원문 fixture와 ADB 비교 명령은 [NPU handoff](kv16_npu_handoff.ko.md)를 따른다.

## 아직 입증하지 못한 것

Android/Hexagon SDK compile/link, ARM/DSP native 실행, 실제 Gemma4 gamma binding과
checkpoint 압축률, seed 후 INT8 bytes/scales/state, logits/token 동등성,
전체 memory peak와 TTFT/TPOT/TPS/전력은 미검증이다. 실제 inference capture에서
압축 이득이 없으면 RAW로 남아야 하며 compressed page가 하나도 없으면 해당
장치 실행을 codec 검증 성공으로 기록하지 않는다.
