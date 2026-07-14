# QuickDotAI API/AAR 단순화 작업 현황

마지막 갱신: 2026-07-15

작업 브랜치: `api_fix`

시작 기준: nntrainer PR #4076 이후 코드

## 현재 진행률

이 호스트에서 수행하기로 한 구현과 정적 검증 작업은 **100% 완료**했다.

핵심 API 재설계, plugin multimodal 보완, 회귀 테스트와 대상 플랫폼 정적
검증을 완료했다. 검증된 변경의 DCO commit/push 이력은 `api_fix` 브랜치에
연속해서 유지한다.

이 수치는 로컬에서 가능한 구현과 정적 검증을 기준으로 한다. 실제 x86
Linux 전체 빌드와 Android 실기기 검증은 `origin/api_fix` 푸시 후 빌드 가능한
원격 환경에서 별도로 진행해야 한다.

## 리뷰용 커밋 구성

기존의 큰 구현 커밋은 아래 순서의 작업 단위로 재구성한다. 각 커밋 메시지에는
문제 배경, 핵심 변경, 호환성 영향, 검증 범위를 기록한다.

1. `[OpenAI] Add strict chat request parsing`
   - OpenAI message/content/tool/response-format 정규화와 parser 단위 테스트
2. `[CausalLM] Isolate per-request generation state`
   - 취소, sampling RNG, EOS, KV/conversation 상태의 model 단위 관리
3. `[xgrammar] Make request constraints handle-local`
   - tool 및 JSON schema grammar 선택, 수명, cache/reset 경계
4. `[QNN] Make generation grammar-aware and cancellable`
   - QNN sampling, grammar mask, 취소와 model-specific 입력 검증
5. `[quick_dot_ai_api] Consolidate generation entry points`
   - `quickAiRunText`, `quickAiRunOpenAI`, handle lifecycle와 image sidecar
6. `[QuickDotAI] Define exact-text and OpenAI contracts`
   - AAR DTO/interface, descriptor catalog와 image preprocessing 계약
7. `[QuickDotAI/Native] Route requests through the native API`
   - handle 기반 JNI, sidecar marshalling, request epoch와 JNI 안전성
8. `[QuickDotAI/LiteRT] Adopt the OpenAI request contract`
   - OpenAI message와 LiteRT media 입력 변환, 동시성 및 취소
9. `[SampleTestAPP] Migrate to descriptor-driven generation`
   - catalog 선택과 통합 generation 경로를 사용하는 샘플
10. `[QuickDotAI] Remove superseded session adapters`
    - backend별 chat session과 숨은 image store 제거
11. `[QuickDotAI] Tighten AAR packaging boundaries`
    - storage permission, runtime dependency와 consumer rule 정리
12. `[docs] Document the unified QuickDotAI API`
    - API/architecture/streaming 및 migration 문서
13. `[docs] Record the QuickDotAI migration handoff`
    - 로컬 검증 결과와 원격 환경 확인 순서
14. `[VJEPA2_QNN] Fix embedding output buffer sizing`
    - UF16 destination 크기, batch stride, overflow와 ownership 수정
15. `[quick_dot_ai_api] Support extension multimodal dispatch`
    - backend-independent composition과 versioned plugin V2 callback
16. `[QuickDotAI] Enforce multimodal catalog capabilities`
    - plugin descriptor 선택, single/multi-image fail-fast와 JVM 회귀 테스트
17. `[docs] Document extension multimodal behavior`
    - plugin ABI, fused/composite dispatch, generic fallback와 제약사항
18. `[docs] Finalize the QuickDotAI validation handoff`
    - 최종 정적 검증, 리뷰 순서와 실제 기기 테스트 matrix

권장 리뷰 순서는 위 번호 순서다. 1~5는 native 실행 기반과 C API, 6~11은
AAR/JNI migration series, 12~13은 기본 API 문서와 초기 handoff, 14~16은
`libqai_ext_model.so`를 고려한 멀티모달 재검토, 17~18은 최종 plugin 계약과
검증 handoff를 설명한다. 최종 tree 전체의 정적 검증 결과는 이 문서의
`최종 로컬 검증` 절에 기록한다.

## 완료한 작업

### 1. Generation API를 두 개의 개념으로 통합

C API의 generation surface를 다음 두 함수로 정리했다.

- `quickAiRunText(...)`: 입력 UTF-8 평문을 변형하지 않고 이미 포맷된 모델
  prompt로 전달한다.
- `quickAiRunOpenAI(...)`: OpenAI Chat Completions 형식 JSON을 파싱하고 chat
  template, tool/structured-output grammar, 선택적 image tensor sidecar를 한
  경로에서 처리한다.

Android AAR도 같은 개념으로 맞췄다.

- `runText(text, sink)`
- `runOpenAI(OpenAIRequest(...), sink)`

기존의 중복된 `runModel*`, message 전용, multimodal 전용, chat-session 전용
generation 변형은 공개 API에서 제거했다. 대화 기록은 일관된 세션 의미가
정의될 때까지 OpenAI JSON의 `messages`로 명시적으로 전달한다.

### 2. OpenAI 요청 파서 추가

`api/openai_request.{h,cpp}`를 추가해 다음을 검증/정규화한다.

- message role과 text/image content part
- tool과 tool choice
- assistant tool call과 tool result의 상관관계
- `response_format` JSON schema
- 지원하지 않는 generation option의 명시적 거부
- JSON의 image occurrence와 native tensor sidecar의 순서/URL 일치

tool 또는 structured output이 강제되는 경우 xgrammar용 schema를 생성한다.
지원하지 않는 필드를 조용히 무시하지 않고 `INVALID_PARAMETER` 또는
`UNSUPPORTED`로 반환하도록 설계했다.

### 3. xgrammar 수명과 동시성 보강

- grammar를 전역 공유 상태가 아니라 handle 소유 상태로 이동했다.
- grammar cache key와 reset 수명을 요청 단위로 정리했다.
- forced tool, required tool, JSON schema response 형식을 구분했다.
- grammar 초기화/실행 오류가 일반 inference 오류로 묻히지 않게 했다.

### 4. Multimodal을 OpenAI 경로에 병합

JSON 본문은 OpenAI `image_url` content part를 유지하고, Android에서 이미
전처리한 pixel 값은 `OpenAIImageTensorSidecar`로 함께 전달한다. C ABI에는
고정 V1 stride의 `QuickAiImageTensorV1`을 정의했다.

native `runOpenAI`는 loaded descriptor의 capability를 기준으로 sidecar 요청을
검증한다. `MULTIMODAL`이 있어야 image sidecar를 받을 수 있고,
`MULTI_IMAGE`가 있으면 JSON의 여러 `image_url` occurrence에 대응하는 여러
sidecar를 같은 호출로 전달할 수 있다. `MULTI_IMAGE`가 없는 모델은 한 장만
허용한다.

실행은 별도 공개 multimodal 함수로 분기하지 않는다. 하나의 `runOpenAI`
경로에서 `libqai_ext_model.so`가 architecture별로 등록한 fused/composite
multimodal hook을 먼저 사용하고, hook이 없으면 호환되는 `[vision encoder,
embedding-input LLM]` handle에 generic composer를 적용한다. 어느 경로도
descriptor capability와 실제 model interface를 충족하지 못하면
`UNSUPPORTED`를 반환한다. 공개 tree가 private model을 직접 포함하지 않는다는
사실은 native image API 자체가 의도적으로 비활성이라는 뜻이 아니다.

LiteRT-LM은 data URL 또는 앱에서 읽을 수 있는 local file을 자체 media 입력으로
변환하며 native tensor sidecar는 받지 않는다.

### 5. 모델 catalog와 engine binding 수정

- `createEngine(descriptor)`가 runtime 종류만 보고 descriptor를 버리던 문제를
  수정했다.
- 생성된 engine은 지정 descriptor에 bind되고 load 시 model ID와 backend를
  검증한다.
- V-JEPA를 text/OpenAI generation 모델이 아닌 standalone vision encoder로
  표시하고 Android generation picker에서 제외했다.
- raw text, OpenAI API, multimodal, embedding 기능을 capability로 확인한 뒤
  native 진입 전에 실패하도록 했다.

### 6. Android AAR 설계 정리

- concrete backend class를 내부 구현으로 숨기고 factory를 공개 진입점으로
  정리했다.
- 빈 model ID, model path/base path, descriptor/backend 불일치를 검증한다.
- 한 engine에서 동시에 두 generation 요청이 실행되지 않게 했다.
- callback exception을 native 경계 밖으로 안전하게 변환한다.
- JNI UTF-8/NUL 처리와 local reference/배열 검증을 보강했다.
- `cancel()`의 요청 epoch와 JNI/native arm/disarm 경계를 추가해 generation
  시작 전 취소를 현재 요청에만 적용하고 idle 취소가 다음 요청으로 누출되지
  않게 했다.
- 사용되지 않던 `ImageStore`, backend별 chat-session wrapper, 중복 streamer
  header를 제거했다.

### 7. Native CausalLM/QNN 실행 안정성 수정

- handle/model 소유권을 RAII 중심으로 정리하고 실패 경로의 leak을 줄였다.
- callback 취소, EOS batch 처리, per-handle RNG와 sampling 경계를 보강했다.
- conversation/KV 상태 reset과 grammar 적용 순서를 명시적으로 정리했다.
- Gemma QNN의 한 token 입력을 잘못 거부하던 조건을 수정했다.
- V-JEPA QNN이 dtype-tagged input을 현재 nntrainer inference API에 직접 넘겨
  Android compile에 실패하던 경로를 공용 QNN adapter로 교체했다.
- image embedding의 할당/해제 경로를 일관되게 정리했다.

### 8. 빌드 및 패키징 정리

- Meson의 QuickDotAI API source 목록에 OpenAI parser를 추가했다.
- 중복된 streamer symbol export를 제거했다.
- native multimodal dispatch를 QNN 전용/experimental API로 설명하던 경계를
  제거하고, plugin hook과 generic model composition이 같은 OpenAI 경로를 쓰는
  계약으로 정리했다.
- `libcausallm.so`와 `libquick_dot_ai_api.so`의 설치 위치를 맞췄다.
- public C headers를 CausalLM include 경로에 설치하도록 했다.
- Android AAR/JNI와 sample app의 제거된 API 참조를 새 API로 교체했다.
- README와 architecture/streaming 문서를 현재 library/API 이름으로 갱신했다.

### 9. Optional model plugin과 multimodal 계약 재검토

`libqai_ext_model.so`를 단순한 catalog 보조 파일이 아니라 실제
descriptor/model/callback provider로 다시 검토했다.

- plugin descriptor의 `MULTIMODAL`/`MULTI_IMAGE` capability가 AAR와 C API의
  image sidecar 허용 범위를 결정한다.
- fused/multi-image model은 architecture별
  `OpenAIMultimodalCallbackRegistry` V2 hook으로 실행하고, 이미
  `[vision, LLM]`으로 구성된 handle은 generic composer로 실행한다.
- V2 hook에는 검증된 OpenAI request, nullable core-formatted prompt와 모든
  submodel의 versioned non-owning view를 전달한다. 따라서 core chat template가
  없는 fused plugin도 자체 `<video>` 등 model-specific template를 적용할 수 있다.
- 기존 plugin의 by-value `ModelCallbacks` ABI는 크기를 바꾸지 않았다. legacy
  `multimodal_streaming`은 metadata/grammar를 표현할 수 없으므로 단일
  unconstrained RGB 512x512 patch 요청에만 호환 경로로 사용한다.
- V2/legacy callback을 실제 호출한 뒤에는 `UNSUPPORTED`도 최종 결과로 처리해
  token, grammar, KV state가 변경된 상태에서 다른 경로로 재실행하지 않는다.
- callback output은 성공할 때만 handle에 반영하고, legacy pair의 KV 상태는
  callback 등록 model이 아니라 실제 text model 기준으로 갱신한다.
- caller는 어느 topology인지 구분하지 않고 기존 `runOpenAI`만 호출한다.
- 여러 이미지는 JSON occurrence 순서와 sidecar 순서를 일치시키며,
  `MULTI_IMAGE` capability가 없는 descriptor에는 fail-fast한다.
- plugin은 `Transformer` virtual interface와 C++ callback registry를 공유하므로
  `libcausallm.so`, `libquick_dot_ai_api.so`, plugin을 같은 source revision으로
  함께 다시 빌드해야 한다.
- V-JEPA QNN encoder output을 LLM의 UF16 embedding quant space로 변환할 때
  source dtype이 8-bit여도 destination byte 수를 기준으로 할당/stride하도록
  수정해 2-byte write의 buffer overflow 가능성을 제거했다.
- multimodal callback 도중의 active cancellation도 text capability gate 없이
  모든 loaded model에 전달하고, vision encoder의 empty output 오류 경로에서
  반환 buffer가 누수되지 않도록 했다.

## 최종 로컬 검증

- Android QuickDotAI JVM 단위 테스트: **32/32 통과**
- SampleTestAPP Kotlin compile: **통과**
- OpenAI parser/chat-template 수동 gtest: **21/21 통과**
- xgrammar manager 수동 gtest: **3/3 통과**
- Android NDK 대상 core API(일반/QNN), JNI, Gemma QNN, V-JEPA QNN syntax
  check: **통과**
- C11/C++17 public header 및 C++17 internal header compile check: **통과**
- clang-format 14 dry-run, `git diff --check`, 기본 doxygen-tag 검사: **통과**
- Meson reconfigure는 변경된 Meson 파일을 파싱한 뒤 Windows 호스트의 기존
  CMake subproject가 Ninja executable을 찾지 못한 지점에서 종료됐다. 이
  Windows 전용 환경 문제는 지원 대상 검증으로 보지 않는다.

Windows native application build는 이 작업의 지원 대상이 아니므로 추가로
추적하지 않는다. 최종 검증은 x86 Linux/Android에 영향을 주는 portable source,
Meson/Gradle 구성과 Android NDK 문법에 집중한다.

이번 plugin/multimodal 재검토 보완은 source/header/문서 정적 검증 범위다.
실제 `libqai_ext_model.so`를 포함한 fused/composite 실행, multi-image token
streaming과 Android lifecycle은 이 호스트에서 실기기로 검증하지 못했다.

## 로컬 작업 완료 및 다음 단계

로컬 구현과 가능한 정적/JVM 검증은 완료했다. 남은 단계는 빌드/기기 접근이
가능한 원격 환경에서 아래 항목을 실행하고, 발견되는 실제 runtime 문제를
후속 수정하는 것이다. 검증된 변경은 `api_fix` 브랜치에 후속 commit으로
publish한다.

## 알려진 제한과 후속 설계 항목

- 이 호스트에는 연결된 Android 기기가 없으므로 실제 token streaming,
  cancellation latency, LiteRT media 처리와 QNN 실행은 검증할 수 없다.
- 전체 x86 Linux build는 Linux build host에서 확인해야 한다.
- optional `libqai_ext_model.so`는 C++ virtual interface를 사용하므로 core,
  API, plugin을 반드시 같은 source revision으로 함께 다시 빌드해야 한다.
- public AAR 산출물은 private plugin을 기본 포함하지 않는다. downstream package가
  같은 revision으로 빌드한 plugin을 `jniLibs`에 포함하면 plugin constructor가
  descriptor/model/callback을 등록하고 같은 `runOpenAI` 경로를 활성화한다.
- SampleTestAPP의 encoded-image 편의 경로는 아직 LLaVA-NeXT 전처리기를 사용한다.
  다른 plugin 모델은 descriptor별 preprocessor registry가 생길 때까지 해당
  모델이 요구하는 `OpenAIImageTensorSidecar`/`MODEL_NATIVE` tensor를 직접
  구성해야 한다.
- legacy `multimodal_streaming`은 opaque handle의 private layout에 의존하는 기존
  C++ ABI다. 신규 plugin은 versioned model view를 받는 V2 hook을 사용하며, 장기적으로
  extension 등록 경계 전체를 version handshake가 있는 C ABI로 바꾸는 것이 좋다.
- legacy `jni/Android.mk`의 오래된 ndk-build 경로에는 이번 작업 이전부터 없는
  source 참조가 남아 있다. 지원하는 AAR 경로는 Meson/`build_android.sh`와
  Gradle JNI build다.
- low-level pair loader는 vision/LLM별 backend와 quantization을 각각 표현하지
  못한다. backend가 다른 조합은 plugin이 완성된 composite descriptor/model을
  제공하거나 향후 per-component load spec을 사용해야 한다.
- sidecar 요청의 `UNSUPPORTED`는 "public pair가 없다"는 전역 정책이 아니라,
  선택 descriptor의 capability, image count, plugin hook 또는 generic pair
  interface가 맞지 않을 때의 요청별 결과다.
- 이번 재검토에서는 실제 plugin binary, x86 Linux 전체 build와 Android 실기기
  fused/composite inference를 검증하지 않았다.

## 원격 환경에서 우선 확인할 항목

푸시 후 다음 순서로 검증하는 것이 좋다.

1. x86 Linux Meson configure/build/test
2. Android AAR와 SampleTestAPP clean build
3. native `runText`가 입력 prompt를 그대로 전달하는지 확인
4. native/LiteRT `runOpenAI` text-only 요청 확인
5. forced/required tool과 JSON schema response 확인
6. generation 시작 전/중/직후 `cancel()` 반복 확인
7. LiteRT data URL/local file multimodal 요청 확인
8. 같은 SHA로 재빌드한 optional native plugin이 catalog descriptor와 callback을
   등록하는지 확인
9. plugin fused model과 generic `[vision, LLM]` composite 각각의 single-image
   `runOpenAI` 확인
10. `MULTI_IMAGE` descriptor의 여러 sidecar 성공과 capability가 없는 descriptor의
    fail-fast 확인
