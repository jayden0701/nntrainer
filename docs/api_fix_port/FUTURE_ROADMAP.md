# `api_fix` 이식 작업: 향후 PR 계획

## 현재 출발점

- 구현 스택의 최신 로컬 tip:
  `0535256726f34b24720faf8b71059041d45c428c` (PR17)
- PR1~PR15: `origin` push 완료
- PR16~PR17: 로컬 commit만 존재
- 다음 구현 브랜치:
  `agent/api-fix-postreview-qnn-packaging-docs`
- 이 문서 작성 시점에는 PR18 코드를 아직 수정하지 않았다.

아래 번호는 독립 review/merge 단위를 뜻한다. 실제 조사에서 강한
의존성이 발견되면 범위를 더 쪼개되, 서로 다른 실패 상태를 한 PR에
섞지는 않는다.

## 우선순위 로드맵

### PR18 — QNN header packaging과 build/runtime 문서 정정

목표:

- 설치되지만 단독 include할 수 없는 `qnn_context.h`를 public header
  목록에서 제거한다. plugin ABI는 이미 설치되는 `context.h`의
  `ContextPluggable`이다.
- `QNN_BUILD.md`를 실제 generator와 PR17 target에 맞춘다.

필수 정정:

- generator가 QNN headers뿐 아니라 SampleApp/Genie source도 복사함
- 실제 `jni/vendor/`와 `jni/QNNGraph.*`, `jni/qnn_properties.*` 경로
- Engine 객체를 초기화할 때 plugin DSO를 best-effort로 로드한다. 일반
  `Engine::Global()` 경로에서는 singleton의 최초 초기화 때이며, vendor
  QNN runtime은 실제 QNN 사용 시 lazy initialize한다.
- build-tree/install libdir의 `libqnn_context.so`도 loader search
  path에 있어야 함
- 존재하지 않는 QNN 전용 test binary 명령 제거
- native QNN은 shared/both core 구성만 지원하고 static-only는 거부함

분리 유지:

- 전용 plugin registration smoke test는 별도 PR
- Engine plugin handle/context 소유권 재설계는 별도 PR

### PR19 — CausalLM RPC allocator를 공용 loader로 통합

현재 위험:

- `android_memory_allocator.cpp`의 local `handle`이 global을 가린다.
- `dlopen`/symbol lookup 실패 뒤에도 initialized로 표시될 수 있어 다음
  allocate/free가 null function pointer를 호출할 수 있다.
- initialization이 경쟁 가능하고 `fileSize + 140` overflow 및 `int`
  narrowing 검사가 없다.

계획:

- 이미 QNN plugin에서 쓰는 `RpcMem` singleton implementation을
  CausalLM의 두 Meson target에 정확히 한 번씩 포함한다.
- size overflow, `INT_MAX`, null allocation을 fail-closed로 처리한다.
- 실제 caller가 null을 역참조하지 않도록 실패를 명시적으로 전파한다.

### PR20 — QNN graph vendor execution transaction

현재 위험:

- 성공한 `beforeExecute()` 뒤 config 검증/allocation/`graphSetConfig()`
  실패가 `afterExecute()`를 건너뛸 수 있다.
- vendor throw/실패 후 graph lifecycle이 RUNNING으로 남을 수 있다.
- `graphRetrieve`, `graphSetConfig`, `graphExecute`의 ambiguous output을
  일관되게 분류하지 않는다.

계획:

- hook/config/execute/after의 정확한 transaction timeline을 만든다.
- primary execution failure와 secondary cleanup failure를 모두 보존한다.
- 정상 입력 오류와 vendor state ambiguity를 구분해 필요한 경우에만
  runtime을 quarantine한다.

### PR21 — op-package extension hook pairing

현재 위험:

- `registerOpPackages()`가
  `beforeRegisterOpPackages`/`afterRegisterOpPackages`를 호출하지 않는다.
- 일부 package 등록 뒤 실패하면 rollback API가 없다.

계획:

- 모든 문자열/function pointer를 mutation 전에 preflight한다.
- before/vendor/after를 하나의 transaction으로 묶는다.
- 부분 등록, throw, hook 실패처럼 state를 증명할 수 없는 결과는
  quarantine한다.

### PR22 — RPC allocation identity 검증

계획:

- 반환 pointer가 요청 alignment를 만족하는지 확인한다.
- `uintptr_t` overflow와 live allocation interval overlap을 거부한다.
- fd는 모든 음수를 거부하고 live allocation 사이 duplicate fd를
  검출한다.
- non-QNN fallback도 alignment contract를 지키는 allocator를 사용한다.

### PR23 — generated vendor tree의 SDK provenance 결합

현재 위험:

- compatibility stamp는 SDK root/content를 식별하지 않는다.
- `qnn-sdk-root`를 바꿔도 이전 generated tree가 재사용될 수 있다.
- 일부 legacy replacement는 아직 exact-count fail-closed가 아니다.

계획:

- SDK package/root와 safety-critical source fingerprint를 stamp에 기록한다.
- configure 시 provenance가 다르면 재생성하거나 명확히 실패한다.
- 남은 compatibility transform도 unique anchor, exact count,
  postcondition으로 검증한다.

### PR24 — DLC record buffer와 임시 파일 소유권

현재 위험:

- memory-mapped DLC pointer가 no-op deleter로 ResourceManager lifetime
  밖으로 탈출한다.
- record handle을 해제하지 않거나 size/read를 여러 transaction으로
  나눌 수 있다.
- record suffix를 임시 filename으로 직접 사용하면 path traversal을
  통한 overwrite/delete 위험이 있다.

계획:

- 한 record transaction 안에서 size/data를 얻어 host-owned buffer로
  복사하고 `systemDlcFreeRecord()`를 호출한다.
- exclusive-create 임시 파일과 scope cleanup을 사용한다.
- path component를 외부 입력에서 직접 만들지 않는다.

### PR25 — profiling traversal 제한

계획:

- profile pointer/count 조합과 function pointer를 검증한다.
- vendor throw를 C++ boundary 안에서 처리한다.
- cycle, maximum depth, maximum event count 제한을 둔다.

profiling은 현재 기본적으로 꺼져 있으므로 앞선 runtime correctness
작업보다 우선순위가 낮다.

## 별도 후보 PR

아래 항목은 위 작업과 결합하지 않고 근거가 확보될 때 독립 제안한다.

- native `libqnn_context.so` registration smoke test
- Engine의 plugin DSO/context process-lifetime 소유권 명시
- `native_lib_dir` 또는 명시 경로를 QNN plugin lookup에 반영
- Tizen/Yocto/Windows native QNN 지원 여부를 검증한 플랫폼 표
- CausalLM의 중복 QNN context 등록 경로 정리

## 사용자 환경 검증 순서

이 저장소 작업 환경에서는 아래 명령을 실행하지 않는다. PR stack을 실제
merge하기 전 사용자 x86 Linux/Android 환경에서 다음 순서로 확인한다.

1. PR별 tip을 순서대로 build하여 최초 실패 PR을 식별한다.
2. QAIRT 2.47.0.260601 SDK로 `enable-npu=true`,
   `default_library=shared` native Linux configure/compile/link를 확인한다.
3. `libqnn_context.so` 생성·설치·loader search path와
   `ml_train_context_pluggable` 등록을 확인한다.
4. parser, xgrammar, C API, CausalLM unit/integration test를 실행한다.
5. Android NDK/Gradle/AAR/SampleTestAPP build를 별도로 확인한다.
6. Snapdragon device에서 QNN context load, graph forwarding, cancellation,
   multi-handle grammar, RPC allocation/register/free를 확인한다.
7. vendor failure injection으로 create/free, hook, DSO close,
   memRegister/memDeRegister의 success/null/error/non-null/throw 조합과
   resource ledger가 의도대로 rollback 또는 quarantine하는지 확인한다.

실제 결과는 각 PR 문서의 “외부 검증 필요”에 기록하고, 실패가 발견되면
해당 PR을 수정한 뒤 뒤쪽 스택을 다시 rebase한다.
