# QuickDotAI API/AAR 단순화 작업 현황

마지막 갱신: 2026-07-14

작업 브랜치: `api_fix`

시작 기준: nntrainer PR #4076 이후 코드

## 현재 진행률

전체 작업의 약 **95%**를 완료했다.

핵심 API 재설계, 구현, 회귀 테스트와 대상 플랫폼 정적 검증은 끝났다.
현재 남은 작업은 최종 diff/staging 검토, DCO commit과 원격 푸시다. 예상 잔여
작업량은 큰 추가 문제가 발견되지 않는다는 전제에서 약 **30~60분**이다.

이 수치는 로컬에서 가능한 구현과 정적 검증을 기준으로 한다. 실제 x86
Linux 전체 빌드와 Android 실기기 검증은 `origin/api_fix` 푸시 후 빌드 가능한
원격 환경에서 별도로 진행해야 한다.

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

현재 native fused runner는 이미지 한 장만 지원하므로 여러 장은 명시적으로
`UNSUPPORTED`를 반환한다. LiteRT-LM은 data URL 또는 앱에서 읽을 수 있는
local file을 자체 media 입력으로 변환하며 native tensor sidecar는 받지 않는다.

현재 공개 catalog에는 vision encoder와 embedding-input LLM을 조합해 native
sidecar inference를 완주할 수 있는 모델 pair가 없다. 따라서 이 경로는
capability로 fail-fast하며, 호환 composite model/plugin이 추가될 때까지
experimental 성격을 유지한다.

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
- QNN/experimental multimodal compile define을 일치시켰다.
- `libcausallm.so`와 `libquick_dot_ai_api.so`의 설치 위치를 맞췄다.
- public C headers를 CausalLM include 경로에 설치하도록 했다.
- Android AAR/JNI와 sample app의 제거된 API 참조를 새 API로 교체했다.
- README와 architecture/streaming 문서를 현재 library/API 이름으로 갱신했다.

## 최종 로컬 검증

- Android QuickDotAI JVM 단위 테스트: **27/27 통과**
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

## 남은 작업

1. 최종 diff와 staging 범위를 확인한다.
2. DCO sign-off와 agent co-author trailer를 포함한 commit을 만든다.
3. `origin/api_fix`로 푸시하고 원격 branch를 확인한다.

## 알려진 제한과 후속 설계 항목

- 이 호스트에는 연결된 Android 기기가 없으므로 실제 token streaming,
  cancellation latency, LiteRT media 처리와 QNN 실행은 검증할 수 없다.
- 전체 x86 Linux build는 Linux build host에서 확인해야 한다.
- optional `libqai_ext_model.so`는 C++ virtual interface를 사용하므로 core,
  API, plugin을 반드시 같은 source revision으로 함께 다시 빌드해야 한다.
- legacy `jni/Android.mk`의 오래된 ndk-build 경로에는 이번 작업 이전부터 없는
  source 참조가 남아 있다. 지원하는 AAR 경로는 Meson/`build_android.sh`와
  Gradle JNI build다.
- experimental multimodal pair loader는 vision/LLM별 backend와 quantization을
  각각 표현하지 못한다. 실제 public composite model을 추가할 때 per-component
  load spec 또는 catalog-owned composite descriptor로 교체해야 한다.
- public native multimodal 모델 pair가 생기기 전에는 AAR의 native image
  sidecar 요청이 `UNSUPPORTED`인 것이 의도된 동작이다.

## 원격 환경에서 우선 확인할 항목

푸시 후 다음 순서로 검증하는 것이 좋다.

1. x86 Linux Meson configure/build/test
2. Android AAR와 SampleTestAPP clean build
3. native `runText`가 입력 prompt를 그대로 전달하는지 확인
4. native/LiteRT `runOpenAI` text-only 요청 확인
5. forced/required tool과 JSON schema response 확인
6. generation 시작 전/중/직후 `cancel()` 반복 확인
7. LiteRT data URL/local file multimodal 요청 확인
8. 같은 SHA로 재빌드한 optional native plugin load 확인
