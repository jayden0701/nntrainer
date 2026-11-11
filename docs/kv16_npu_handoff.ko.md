# KV16 NPU 통합 인계서

작성일 2026-11-10. 작업 위치는 `C:\nntrainer-kv16-npu`, branch는
`codex/kv16-npu-integration`, 기준은 PR #4481의
`5e97254e764aa30db4223c33d0dc3b0fded091b8`이다. 앞선 Python reference는
`C:\nntrainer-kv16-lossless`에 별도로 보존되어 있다.

## 이 변경이 다루는 경계

첫 대상은 Gemma4 global layer의 **host FP16 prefix/checkpoint 보관**이다.
복원한 FP16 K/V는 기존 restore 및 INT8 seed 경로로 넘긴다. NPU가 읽는 기본
paged active KV는 여전히 INT8이다. Q2/U16 attention, whole-call calibration,
prefill call plan, RoPE, projection, DSP page layout와 누산 순서는 바꾸지 않는다.

V 전체와 rotary K 원본, non-rotary K의 XOR residual을 저장한다. 정수 predictor는
FP32 multiply RNE 후 FP16 RNE를 재현한다. RAW/fixed-width/sparse 후보 가운데
전체 저장 크기가 작아지는 것만 채택한다. FP16 특별값도 비트 단위로 보존한다.
runtime gamma와 실제 global geometry를 쓰며 BF16을 FP16으로 재해석하지 않는다.

장치가 없는 상태에서 구현한 코드와 호스트 검사만으로 실제 HTP 동등성이나
성능을 PASS로 판정하지 않는다. 새 binary를 실제 장치에 빌드·배포하고 아래
증거를 수집하는 것이 다음 담당자의 첫 임무다. CPU/GPU 전체 모델 검증은
이번 작업의 진행 조건이 아니다.

## 런타임 옵션

| 환경 변수 | 의미 |
| --- | --- |
| `NNTR_HTP_PREFIX_KV16=1` | Gemma4 HTP host prefix codec opt-in. 첫 cache allocation에서 읽음 |
| `NNTR_HTP_PREFIX_KV16_VERIFY=1` | 원본 K/V와 encode/decode 결과를 publication 전에 바이트 비교하는 진단 모드 |
| `NNTR_HTP_PREFIX_KV16_TRACE=/absolute/path/trace.jsonl` | checkpoint capture/restore 진단 JSONL |
| `NNTR_HTP_PREFIX_KV16_SEED_TRACE=/absolute/path/seed.jsonl` | 실제 engine의 seed RPC 직전 K/V·floor 입력 진단 JSONL |

`1`만 enable 값이다. 환경을 바꾸고 같은 model instance를 재사용하지 않는다.
baseline과 codec은 **같은 ARM binary·DSP skeleton·model pack·numerical config**를
사용하는 별도 process에서 실행한다. baseline은 codec 변수 없이 실행하고,
codec은 위 세 변수를 지정한다. VERIFY/TRACE는 timing 실험과 분리한다.

작은 page에서 metadata가 압축 이득을 상쇄하면 RAW로 남는 것이 정상이다.
`compressed_pages=0`이면 codec 경로를 실제로 검증한 것으로 기록하지 않는다.
source page의 view owner나 다른 snapshot이 원본 RPC slab을 pin하고 있으면
압축 payload 감소와 시스템 resident memory 감소는 다를 수 있다.

## 장치 연결 전에 준비할 것

지원하는 v81 Android 장치와 권한 있는 ADB, Android NDK, Hexagon SDK/toolchain,
HexKL이 필요하다. 이미 검증된 fully resident Gemma4 26B-A4B `htp_fused` pack을
사용한다. repository의 `res/gemma4/htp_chat/nntr_config.json`은 LM head가 CPU인
예제이고, model 기본 config도 다른 variant일 수 있으므로 그대로 사용하지 않는다.

실제 기준은 [paged sessions](gemma4-htp-paged-sessions.md)와
[radix contract](gemma4-htp-radix-prefix.md)다. pack의 최소 조건은 다음과 같다.

```json
{
  "attention_kv_dtype": "q8",
  "skip_prefill": true,
  "init_seq_len": 4096,
  "htp": {
    "attn_q2": true,
    "oproj_u16": true,
    "kv_seed_q": false,
    "kv_prefix_cache": true,
    "kv_session": true,
    "kv_paged": true,
    "kv_request_reserve": true
  }
}
```

attention·O projection·dense FFN·MoE·LM head는 HTP여야 한다.
`kv_seed_q=true`는 별도 직접 quantized seed 경로이며 host prefix retention을
끄므로 이 기능의 검증 설정으로 사용하지 않는다. full-attention D=512,
KV heads=2, `attention_k_eq_v=true`인 실제 pack을 확인한다. rotary map은
`[0,64) ∪ [256,320)`이며 K norm은 runtime gamma 직접 곱이다.

다음 항목을 결과 directory에 보존한다: source SHA와 변경 patch의 SHA256,
device fingerprint/SoC, SDK/toolchain 버전, ARM libraries와 DSP skeleton의 SHA256,
model/weight/tokenizer/config/fixture SHA256, 실제 환경 변수, 전체 stdout/stderr,
원본 checker JSON과 trace JSONL. weight가 여러 파일이면 정렬한 파일명→SHA256
목록을 보존하고 그 목록 자체의 hash도 기록한다.

## 실제 Android build와 배포

아래는 저장소의 [build_android.sh](../Applications/CausalLM/build_android.sh)가
실제로 처리하는 옵션과 환경 변수다. 경로와 serial은 해당 검증 환경 값으로
설정한다. 이 인계 작성 환경에서 Android/Hexagon build는 실행하지 않았다.

```bash
export ANDROID_NDK=/path/to/android-ndk
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
export DEFAULT_HEXAGON_TOOLS_ROOT=/path/to/hexagon-tools
export HEXKL_ROOT=/path/to/hexkl
export HEX_ARCH=v81
export ANDROID_SERIAL=actual-device-serial
adb -s "$ANDROID_SERIAL" get-state

# 저장소 루트에서 ARM engine/app와 matching DSP skeleton을 함께 빌드·배포
Applications/CausalLM/build_android.sh --htp --install

# native checker 옵션은 app_build의 Meson 옵션이다.
meson configure builddir_app/htp -Denable-htp-device-check=true
ninja -C builddir_app/htp htp_sessions_check htp_calibration_check \
  htp_prefill_check htp_prefix_flash_check htp_decode_kv_convert_check \
  htp_kv16_codec_check htp_kv16_prefix_check htp_kv16_flash_check
adb -s "$ANDROID_SERIAL" push builddir_app/htp/htp_sessions_check \
  builddir_app/htp/htp_calibration_check builddir_app/htp/htp_prefill_check \
  builddir_app/htp/htp_prefix_flash_check \
  builddir_app/htp/htp_decode_kv_convert_check builddir_app/htp/htp_kv16_codec_check \
  builddir_app/htp/htp_kv16_prefix_check builddir_app/htp/htp_kv16_flash_check \
  /data/local/tmp/Quick.AI/
```

`HEXKL_SDK_VER` 기본은 `6.4.0.2`, `HEXKL_ARCH` 기본은 `armv8_android26`다.
실제 SDK library 위치가 다르면 명시한다. ARM engine 결과는
`builddir/android_build_result/lib/arm64-v8a`, app/checker는 `builddir_app/htp`,
DSP skeleton은 `nntrainer/tensor/htp_backend/dsp/build/libnntr_hvx_skel.so`다.
`--install`은 `/data/local/tmp/Quick.AI`에 native libraries, `libsdkl.so`, DSP
skeleton과 실행 wrapper를 배포한다. 해당 장치의 이전 검증 binary를 보존하고
matching ARM/DSP 세트를 확인한다. 이전 skeleton과 새 ARM library를 섞지 않는다.

기존 checker들의 인자는 각 source의 `Usage`/`main`을 확인한다. 특히 모델이
필요 없는 `htp_calibration_check`, `htp_prefill_check`, `htp_prefix_flash_check`,
`htp_decode_kv_convert_check`를 먼저 실행해 좁은 회귀부터 확인한다.

## 모델 검사와 checkpoint trace

ARM 정수 predictor도 PC의 전체 oracle 입력으로 먼저 확인할 수 있다. PC의
`run_kv16_codec_check.py --sanitize` 실행에서 생성한 `predictor.in`과 Python으로
검증한 `predictor.out`을 보존한다. 장치의 checker는 SDK ARM binary로 다시 빌드한다.

```bash
adb -s "$ANDROID_SERIAL" push build/kv16-codec-sanitize/predictor.in \
  /data/local/tmp/Quick.AI/predictor.in
adb -s "$ANDROID_SERIAL" shell 'cd /data/local/tmp/Quick.AI && \
  LD_LIBRARY_PATH=.:/vendor/lib64 ./htp_kv16_codec_check predict \
  predictor.in predictor-arm.out'
adb -s "$ANDROID_SERIAL" pull /data/local/tmp/Quick.AI/predictor-arm.out .
python3 -c "from pathlib import Path; assert Path('predictor-arm.out').read_bytes() == Path('build/kv16-codec-sanitize/predictor.out').read_bytes()"
```

이는 ARM에서 실행한 codec 검사다. DSP seed와 전체 NPU inference 검사는 아래에서
별도로 수행한다. wire `roundtrip` 인자는 `kv16_codec_check.cpp`의 실제 Usage를 따른다.

기존 [htp_sessions_check](../tools/kv_cache/check_htp_sessions.cpp)는 실제
tokenizer를 사용하며 출력 token IDs와 position을 independent references와 비교한다.
fixture는 `{"seeds":[{"prompt":"..."}],"query":"..."}` 형식이다. 원래 4096-token
call boundary를 넘는 prompt와 정상적인 generation cap을 사용한다. 문자의 수를
token 수로 가정하지 말고 checker의 실제 tokenization과 prefix hit/reused counters를
확인한다. 동일 fixture를 baseline과 codec에 사용한다.

다음은 저장소에 남아 있는 과거 native 검증의 **원문 prompt**를 재사용하는
fixture 준비 예다. 당시 2080/4608-token 샘플이었으나 현재 pack/tokenizer에서도
같은 길이라고 가정하지 않는다. 원문이 저장소에 있기 때문에 외부 prompt 파일을
따로 구할 필요는 없다.

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
source = Path('docs/validation/gemma4-htp-native-radix-20261105.json')
data = source.read_bytes()
prompts = json.loads(data)['cold_same_model_raw']['fixture']['prompts']
fixture = {
    'seeds': [{'prompt': text} for text in prompts],
    'query': '<|turn>user\nSay READY.<turn|>\n<|turn>model\n',
    'origin_sha256': hashlib.sha256(data).hexdigest(),
    'note': 'Historical prompt text; recheck actual tokenizer lengths.'
}
Path('kv16-fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
PY
adb -s "$ANDROID_SERIAL" push kv16-fixture.json /data/local/tmp/Quick.AI/
```

```bash
# PACK과 FIXTURE는 이미 장치에 배치한 실제 절대 경로다.
adb -s "$ANDROID_SERIAL" shell 'mkdir -p /data/local/tmp/Quick.AI/kv16-results'

# baseline: codec 변수 없이 동일 binary, prefix retention 자체는 켠다.
adb -s "$ANDROID_SERIAL" shell 'cd /data/local/tmp/Quick.AI && \
  LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
  NNTR_HTP_PREFIX_KV16=0 NNTR_HTP_PREFIX_KV16_VERIFY=0 \
  NNTR_HTP_PREFIX_KV16_TRACE=/data/local/tmp/Quick.AI/kv16-results/baseline.jsonl \
  NNTR_HTP_PREFIX_KV16_SEED_TRACE=/data/local/tmp/Quick.AI/kv16-results/baseline-seed.jsonl \
  ./htp_sessions_check /absolute/PACK /absolute/FIXTURE \
  /data/local/tmp/Quick.AI/kv16-results/baseline-sessions.json' \
  > baseline.log 2>&1

adb -s "$ANDROID_SERIAL" shell 'cd /data/local/tmp/Quick.AI && \
  LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
  NNTR_HTP_PREFIX_KV16=1 NNTR_HTP_PREFIX_KV16_VERIFY=1 \
  NNTR_HTP_PREFIX_KV16_TRACE=/data/local/tmp/Quick.AI/kv16-results/codec.jsonl \
  NNTR_HTP_PREFIX_KV16_SEED_TRACE=/data/local/tmp/Quick.AI/kv16-results/codec-seed.jsonl \
  ./htp_sessions_check /absolute/PACK /absolute/FIXTURE \
  /data/local/tmp/Quick.AI/kv16-results/codec-sessions.json' \
  > codec.log 2>&1

adb -s "$ANDROID_SERIAL" pull /data/local/tmp/Quick.AI/kv16-results results
```

위 명령은 model pack과 fixture 경로를 자동 추측하지 않는다. placeholder를 실제
절대 경로로 바꾸고 shell 종료 코드와 checker의 `passed` 결과를 보존한다.
모델 답변의 문장 유사성이나 process 종료만으로 동등성을 판정하지 않는다.

TRACE records describe checkpoint K/V FNV1a64, geometry, original-call boundaries,
calibration bits and storage accounting. They are lower-cost diagnostics, not
cryptographic integrity proofs. VERIFY performs the original-vs-restored byte
comparison before publication. Keep failure logs even if a later rerun passes.

TRACE는 append 방식이다. 실행마다 비어 있는 새 결과 directory를 사용하고, 이전
JSONL에 새 실행을 이어 쓰지 않는다. trace 쓰기 실패는 stderr의 `[KV16 trace]`
메시지로 남으므로 이 메시지가 있으면 해당 trace 증거는 미수집으로 판정한다.

실제 진단과 sessions checker 결과는 다음으로 비교한다.

```bash
python3 tools/kv16/compare_checkpoint_traces.py results/baseline.jsonl \
  results/codec.jsonl --scope checkpoint \
  --baseline-results results/baseline-sessions.json \
  --candidate-results results/codec-sessions.json \
  --baseline-seed-trace results/baseline-seed.jsonl \
  --candidate-seed-trace results/codec-seed.jsonl --output checkpoint-comparison.json
```

이 도구는 실제 trace schema와 모든 layer의 K/V FNV, dropped maxima, calibration의
FP32 bit 배열, 원래 call boundary를 비교한다. capture/restore와 compressed page가
실제로 관측돼야 한다. optional sessions JSON의 exact token IDs와 position도
비교하며 timing 필드는 동등성 기준으로 사용하지 않는다. 결과의
`byte_comparison_proven=false`와 `hardware_validated=false`는 의도적이다.
FNV 진단 비교만으로 원본 바이트 완전 일치나 실제 장치 실행 provenance를
증명할 수 없기 때문이다. `--scope`를 생략하면 full이므로 seed/logits가 없어
INCOMPLETE가 정상이다.

seed trace는 engine의 실제 `htpDecodeKvSeedF16()` 호출 경계에서 전체 seed rows의
K/V bytes와 dropped floor 입력을 관측한다. layer·position·base·rows·width와
각 element size도 비교한다. 한쪽 파일만 있거나 비어 있으면 INCOMPLETE다.
이 결과는 **RPC 이전 입력** 비교이며 DSP seed 이후 INT8 bytes·scale/state의
readback이 아니다. 이 옵션을 켜도 `full` 판정은 완료되지 않는다.
`Cannot write KV16 seed input trace` 경고가 있으면 해당 seed 증거도 미수집이다.

## 증거 비교의 판정 기준

`tools/kv16/compare_device_evidence.py`는 explicit evidence manifest 두 개를
비교한다. `--scope full`이 기본이며 seed KV/scale, decode state, logits와 token
IDs까지 요구한다. `--scope checkpoint`는 call plan/calibration/K/V/position까지만
판정하고 seed/inference가 검증되지 않았음을 결과에 남긴다.

```bash
python3 tools/kv16/compare_device_evidence.py baseline-evidence.json \
  codec-evidence.json --scope full --output full-comparison.json
```

exit 0은 선택한 scope의 모든 관측값 일치, 1은 불일치, 2는 미수집/비정상/미실행이다.
`hardware_validated=true`, 같은 binary/device/model identity, 비어 있지 않은 event
목록과 실제 `compressed_pages>0`을 요구한다. hash 필드에는 SHA256 문자열 또는
`{"file":"relative/binary/path","bytes":123}`를 넣을 수 있다. 후자는 원본 파일을
직접 SHA256으로 계산한다. 없는 seed/logits를 token IDs로 추정해 채우지 않는다.

현재 일반 checker는 seed INT8 bytes/scales와 raw logits를 export하지 않는다.
따라서 checkpoint trace와 기존 checker JSON만으로는 `full` PASS를 만들 수 없다.
추가 DSP/engine hook으로 seed와 logits 원본을 수집하기 전에는 해당 항목을
미수집으로 남긴다. `performance_verified`도 자동으로 true가 되지 않는다.

explicit manifest의 필수 구조는 다음과 같다. 값이 없는 항목을 임의로 채우지 않는다.

| 경로 | 필수 정보 |
| --- | --- |
| `schema`, `mode` | `nntrainer.kv16.device-evidence.v1`, baseline 또는 codec |
| `hardware_validated` | 실제 장치 실행을 확인한 경우에만 true |
| `identity` | model/tokenizer/numerical_config/ARM/DSP의 `*_sha256`, `code_revision`, `device_fingerprint` |
| `events[]` | 같은 순서의 고유 `id`, `position`과 관측한 hash/token 목록 |
| checkpoint event fields | `call_plan_sha256`, `calibration_sha256`, `k_sha256`, `v_sha256` |
| full 추가 event fields | `seed_kv_sha256`, `seed_scales_sha256`, `decode_state_sha256`, `logits_sha256`, `token_ids` |
| candidate `codec` | `compressed_pages`, `raw_bytes`, `stored_bytes` |

FP32 calibration은 decimal float가 아닌 원본 u32 bit 배열과 필드 순서를 정해
직렬화한다. seed/logits는 수집 hook·dtype·shape·position·byte order를 별도
sidecar에 남기고 baseline과 같은 지점을 비교한다. FNV 문자열의 SHA256을 계산해
원본 K/V SHA256인 것처럼 넣지 않는다. 빈 파일이나 empty SHA256은 증거로 거부한다.

## full 판정을 위한 후속 hook 위치

현재 input trace를 DSP 결과로 바꾸어 해석하지 않는다. 다음 경계를 실제 장치에서
추가 관측하면 full evidence를 보완할 수 있다.

- `hvx_attn_m1_snapshot_bytes()` / `hvx_attn_m1_snapshot_export()`는 layer의
  live length, scaled flag, six scale arrays와 INT8 K/V를 caller-owned payload로
  export한다. pointer/scratch는 직렬화하지 않는다. seed 직후 같은 position에서
  읽어 numerical payload와 state를 비교하는 디버그 hook의 출발점이다.
- `CausalLM::saveDeviceKVCache(path, false)`는 유효한 resident session의 완료 state를
  기존 `save_decode_kv()`로 저장할 수 있다. 완료 state의 회귀에는 쓸 수 있지만,
  seed 직후 state와 동일한 관측 지점이라고 가정하지 않는다.
- `htp_decode_runtime.h::LogitsRequest`와
  `dsp/nntr_hvx_session.h::htp_session_token(..., logits, logits_len, ...)`는
  logit 요청/반환 경계다. argmax-only route에서 full logits를 요청하면 전송량과
  시간이 바뀌므로 이 진단은 성능 측정과 분리한다. raw/capped logits 중 어느
  지점인지 고정하고 같은 값을 비교한다.

이 후속 hook과 DSP readback은 이번 변경에서 구현·실행했다고 기록하지 않는다.

## 변경 전달과 새 검증 checkout

구현 범위와 실행한 host 검사 결과는 [검증 기록](kv16_npu_validation.ko.md)을
확인한다. host 저장 경로를 재현할 때는 실제 source와 일치하는 Meson engine build와
`json.hpp`를 준비한 뒤 다음을 실행한다. CPU/GPU 모델 inference는 수행하지 않는다.

```bash
python3 test/htp/host/run_kv16_integration_checks.py \
  --engine-build /absolute/path/to/matching-meson-build --sanitize \
  --json-include /absolute/path/containing/json.hpp \
  --output build/kv16-integration-verified
python3 -m unittest discover -s tools/kv16 -p 'test_*.py' -v
```

runner는 컴파일 대상 core source와 ABI header가 다르거나 library가 오래되면
실패한다. `--json-include`가 없으면 baked manifest 검사를 미실행으로 기록한다.
최종 실행 로그는 `build/kv16-integration-verified/report.json`이며, 새 환경에서는
기존 결과 directory를 덮어쓰지 않고 새 이름을 사용한다.

현재 변경은 기준 PR head에 대한 별도 작업이며, 사용자의 원래 checkout과 앞선
Python PoC checkout을 덮어쓰지 않는다. 새 검증 환경에서는 다음을 먼저 확인한다.

```bash
git status --short
git rev-parse HEAD
git diff --stat
```

patch로 전달할 경우 새 파일도 포함해야 한다. 담당자가 관련 새 파일에
`git add -N -- <files>`로 intent-to-add를 설정한 뒤 `git diff --binary HEAD`를
저장하거나, 검토 후 DCO/agent trailer를 갖춘 별도 commit으로 전달한다.
미커밋 patch의 SHA256과 전체 파일 목록을 결과에 보존한다. `subprojects/`의
차이가 섞이지 않았는지 확인한다.

받는 쪽은 기준 `5e97254e764aa30db4223c33d0dc3b0fded091b8`의 깨끗한 별도 checkout에서
`git apply --check <patch>` 후 적용한다. 다른 HEAD의 미완성 변경 위에 무조건
덮어쓰거나 conflict를 버리지 않는다. source만 갱신하고 기존 ARM/DSP binary를
재사용하지 말고 앞의 matching build를 수행한다. commit을 만들 때는 AGENTS.md의
`git commit -s`와 agent `Co-authored-by:` trailer를 따른다.

## 기기에서 순서대로 확인할 시나리오

1. codec OFF reference와 ON+VERIFY의 cold capture, exact repeated input, 원래
   full-call boundary에서 갈라지는 suffix를 비교한다. call 내부 crop은 기존대로 miss.
2. global/sliding 혼합, 64-row page 경계와 tail, prefix reuse/append/fork/reset/LRU,
   기존 shared slab owner release, 최종 host offset→seed를 확인한다.
3. prefix Flash를 켜고 in-flight staging와 완료 file restore를 각각 검사한다.
   실제 compressed format이 저장되고 raw legacy format도 읽는지 확인한다.
4. 별도 테스트 전용 fixture에서 compressed payload를 손상시킨다. 잘못된 page를
   publish하거나 live cache를 부분 갱신하지 않고 이전 compatible prefix/miss로
   재계산하는지 확인한다. 실제 사용자 cache 파일을 손상시키지 않는다.
5. K/V와 metadata 일치 후 실제 seed INT8 bytes/scale/state, logits/token/position을
   비교한다. scalar 호스트 PASS를 실제 HVX/HMX/FastRPC PASS로 기록하지 않는다.
6. 정확성 통과 후 VERIFY/TRACE를 끄고 반복·balanced OFF/ON 측정을 수행한다.
   encode/decode/restore/seed 시간, 저장 bytes, RSS/RPC slab/system peak,
   TTFT/TPOT/TPS와 thermal 상태를 기록한다. 모델 load 비용을 분리한다.

compressed logical payload 감소만으로 전체 model memory/DRAM traffic/NPU TPS
개선을 주장하지 않는다. 현재 기본 INT8 대비와 host FP16 보관 대비를 구분한다.
실제 dump에서 이득이 없으면 기본 OFF를 유지하고 별도 후보로 조사한다.
