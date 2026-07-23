# QNN reload 구현 현황

- 최종 갱신: 2026-07-23
- branch: `codex/qnn-reload-lifecycle-hardening`
- source HEAD: `473e8255 [qnn] include required QNN SDK interfaces` (후속 문서 commit 제외)
- 현재 세션의 source 변경: PR 1~11 + Issue #49 build repair 완료, 총 22개 commit
- 현재 세션의 산출물: 분석/계획/현황/요약 Markdown 4개 갱신
- 검증 제한: 부분 compile과 사용자 Linux 환경의 전체 app/QNN SDK build는 통과했지만 실제 device run은 미수행
- SDK 호환 기준: QAIRT/QNN 2.47 — 버전별 workaround가 아니라 lifecycle/ownership 계약이 분석의 중심

이 문서는 실제 구현된 코드와 앞으로의 설계를 구분하는 source of truth다. 체크박스는 코드와 검증이 모두 끝난 경우에만 완료로 바꾼다.

## 1. 현재 상태 한 줄 요약

제공된 Gemma 한 사이클에서는 기존 keep-warm workaround가 reload 실패를 해결했다. PR 4는 process-wide deregistration을 pointer/context별 lossless 정리로 바꿨고, PR 5~6은 allocator/plugin 중복 소유를 정리했다. PR 7~8은 해제 ownership과 앱 오류 전파를 닫았고, PR 9~10은 model-local set과 process-wide allocator ledger를 backing acquisition 전 선확보한다. PR 11은 보조적인 두-QNN-model 조립 실패에도 같은 teardown 오류 계약을 적용한다. `473e8255`는 Issue #49에서 드러난 PR 4의 SDK header self-containment 회귀를 고쳤다. 부분 compile은 통과했으나 실제 QNN device cycle과 bounded context cache는 아직 미검증·미구현이다.

## 2. 분석에 사용한 입력

- `nntrainer_qnn_reload_handoff.md`: 과거 실험과 관찰 참고
- `2026-07-22-qnn-reload-fix.ko.md`: `8cdc4dd3` 초기 workaround의 의도와 변경 설명
- `load_unload_before_fix.txt`: 수정 전 load/run/unload/reload logcat
- `load_unload_after_fix.txt`: 수정 후 동일 시나리오 logcat
- 현재 repository source와 Git history

핸드오프 문서의 계획은 그대로 따르지 않았고, 로그와 현재 코드로 다시 검증한 사실만 현황에 반영했다.

## 3. 이미 구현되어 있는 기반

### 3.1 process-wide Engine identity

- 관련 commit: `3ae89c5e [core] anchor singleton instances to owner library`
- 관련 commit: `dc996d2b fix(engine): make Engine::Global() a singleton and registries idempotent`
- 상태: 구현됨
- 의미: QNN plugin context와 registry가 DSO별로 갈라지는 위험을 줄이고 process 공통 owner 기반을 제공한다.
- 남은 점: process-global이라는 사실만으로 lifecycle 동기화와 정확한 sub-resource ownership이 보장되지는 않는다.

### 3.2 QNN activation pool용 RPC allocator

- 관련 commit: `f28defb0 feat(qnn): tensor-only rpcmem allocator for QNN activation pool`
- 상태: 기반 구현 후 현재 branch에서 ownership/retention 강화 완료
- 의미: NNTrainer tensor pool이 QNN zero-copy registration에 사용할 RPC backing을 가진다.
- 현재 HEAD: allocation pointer가 context별 registration과 datatype/dimension fingerprint를 가지며, deregistration 실패 entry/backing을 보존한다.
- 남은 점: allocation ID/generation과 session `RegistrationScope`, batch prepare/commit release가 없다.

### 3.3 binary path 기반 SDK context registry

- 파일: `nntrainer/qnn/jni/qnn_context_var.h`
- 상태: transactional state와 manager-wide lifecycle guard까지 구현됨
- 현재 구조: `ct_map<binary path, Qnn_Context_Graph_t>`
- 동작: 첫 `QNNGraph::forwarding()`에서 `contextCreateFromBinary()`로 lazy create
- Gemma: prefill/generation graph가 같은 binary path entry와 SDK context를 공유
- 현재 HEAD: `CREATING/ACTIVE/QUARANTINED` 전이, create/free rollback, manager-wide shared/exclusive guard를 적용했다.
- 남은 점: raw path의 artifact/config identity, refcount/lease, per-context guard, bounded cache가 없다.

### 3.4 제공 after 로그를 만든 기존 reload workaround

- commit: `8cdc4dd3`
- 파일: `nntrainer/qnn/jni/QNNGraph.cpp`
- 상태: 구현 및 제공 로그 한 사이클 검증됨

변경 전:

```text
QNNGraph destructor
  → freeContext(binary)
  → SDK rc를 보존하지 않고 host context entry 폐기
  → 다음 NN model pool이 host가 폐기한 이전 context handle로 memDeRegister
  → 실패해도 backing free
  → 두 번째 run에서 새 context create 경로 진입 후 graphExecute 반복 실패
```

변경 후:

```text
QNNGraph destructor
  → process-wide deRegisterQnnTensor()
  → context/graph는 ct_map에 유지
  → backing free
  → 두 번째 run에서 기존 context 재사용
```

효과:

- live context에서 deregistration을 선행한다.
- 두 번째 load/run이 cached context를 재사용한다.
- 제공 로그에서 deregistration과 graph execution 실패가 사라졌다.

workaround 당시 잔여점과 현재 HEAD 처리:

- 개별 graph destructor의 process-wide sweep은 `15873b03`에서 제거했다.
- deregistration 실패 entry/backing은 `15873b03`, `7aa3527d`에서 보존하고 release 오류를 상위로 전파한다.
- context가 process 동안 무제한 누적될 수 있는 문제는 아직 남아 있다.
- manager-wide execution/cleanup guard는 구현했지만 per-context lease는 아직 없다.
- “context recreate가 불가능하다”는 결론은 철회했다. 올바른 strict free/recreate는 별도 device 대조 실험이 필요하다.

## 4. 제공 로그로 확인된 결과

| 관찰 항목 | 수정 전 | 수정 후 | 상태 |
|---|---:|---:|---|
| lazy context-create 경고 | 2 | 1 | 두 번째 run의 context 재사용 확인 |
| `memDeRegister failed` | 62 | 0 | 각 unload 31회 실패 제거 |
| `Execution of Graph: 0 failed!` | 36 | 0 | 두 번째 run 실패 제거 |
| 첫 run token | 172 | 172 | 정상 |
| 두 번째 run token | 깨진 33 | 정상 165 | workaround 효과 확인 |
| tracked direct allocation 정리 | unload마다 155개 | unload마다 155개 | 로그상 수행됨 |

### 소스와 로그로 재구성한 teardown 경로 — 강한 추론

1. `Quick_Dot_AI_QNN`이 `models`의 NNTrainer model handle을 순차 reset한다.
2. `NeuralNetwork::~NeuralNetwork()`는 layer 파괴 전에 tensor pool을 `deallocate()`한다.
3. 먼저 reset되는 model은 자기 pool을 live context에서 정상 deregister한다.
4. 그 model의 `QNNGraph` destructor가 수정 전에는 공유 context에 `contextFree()`를 호출한 뒤 SDK rc와 무관하게 host handle/metadata/entry를 폐기했다.
5. 다음 model의 pool 31개가 host가 이미 폐기해 유효성을 추적할 수 없는 이전 context handle에서 deregister되어 모두 실패했다.
6. 실패 후에도 map entry와 RPC backing이 제거됐다.

### 로그만으로 확인되지 않은 것

- 사용한 QNN/QAIRT SDK version
- 실제 `contextFree`, `memDeRegister`, `graphExecute` numeric rc
- contextFree가 SDK 내부에서 성공했는지, host만 성공으로 간주했는지
- 올바른 순서의 strict free/recreate 성공 여부
- 20~100 cycle 뒤 host/RPC/DSP/HTP memory 안정성
- multi-handle 및 two-binary 동시성 안전성

## 5. 최초 결함 목록의 최신 상태

### P0 — reload correctness 핵심 상태

| 항목 | 상태 | 위험 |
|---|---|---|
| `graphExecute` 실패 상위 전파 | 구현·commit 완료 — device 미검증 | 깨진 token과 허위 성공/성능 보고 |
| numeric rc/resource identity logging | graph execute 범위 구현·commit 완료 — device 미검증 | 다른 lifecycle API는 후속 PR에서 추가 |
| `makeContext()` zero-init/fail-fast/rollback | 구현·commit 완료 — device 미검증 | `CREATING/ACTIVE/QUARANTINED`와 rollback 적용 |
| `freeContext()` failure preservation | 구현·commit 완료 — device 미검증 | free 실패 entry는 `QUARANTINED`로 보존 |
| deregistration 실패 시 entry 보존 | 구현·commit 완료 — device 미검증 | 실패 entry를 `QUARANTINED`로 유지, 자동 retry는 보류 (`15873b03`) |
| deregistration 실패 시 backing 보존 | 구현·commit 완료 — device 미검증 | dereg 실패 backing을 manager가 보존하고 상위 release 실패를 보고 (`15873b03`, `7aa3527d`) |
| explicit model-session shutdown | 구현·commit 완료 — device 미검증 | terminal/idempotent checked deallocation과 C/JNI/Kotlin 오류 계약 (`38227432`~`10bee850`) |
| `QNNGraph` global cleanup 제거 | 구현·commit 완료 — device 미검증 | 정상 owner의 pointer별 free로 일원화 (`15873b03`) |
| active execution drain/guard | manager-wide guard 구현·commit 완료 | 최종 per-context lease/count/timeout은 후속 (`15873b03`) |
| context-scoped registration | `(pointer, context)` registry와 descriptor fingerprint 구현·commit 완료 | session scope, allocation ID/generation과 batch transaction은 후속 (`15873b03`) |

### P1 — 반복성과 확장성

| 항목 | 상태 | 위험 |
|---|---|---|
| `ContextKey/ContextRegistry` | 부분 구현 | transactional state는 완료; artifact/config identity와 generation은 미구현 |
| `ContextLease/ExecutionGuard` | 부분 구현 | manager-wide guard는 완료; lease와 per-context guard는 미구현 |
| `RegistrationKey/RegistrationScope` | 부분 구현 | pointer/context/fingerprint는 완료; session scope와 batch transaction은 미구현 |
| bounded cache/explicit purge | 미구현 | context/HTP memory 무제한 누적 |
| strict free/recreate feature flag | 미구현 | keep-warm 필수 여부 검증 불가 |
| per-forward tensor descriptor RAII | 구현·commit 완료 — device/ASan 미검증 | token 수에 비례한 host 누수 차단 |
| input/output buffer vector reset | 누적 vector 제거·commit 완료 — device 미검증 | call-local 현재 RPC backing 직접 bind |
| graph handle generation cache | 미구현 | 매 token 불필요한 graphRetrieve |
| duplicate QNN runtime initialization 제거 | 구현·test source commit/compile-check 완료 | test binary 실행과 실제 QNN device 초기화 횟수는 미검증 (`5163e606`, `82a05282`) |
| Kotlin unload/destroy handle 정합성 | 구현·offline compile 완료 — device 미검증 | teardown 실패를 terminal로 보존하고 reload 차단 (`10bee850`) |
| fake QNN lifecycle tests | 부분 구현 | plugin once와 generic release retention test source를 compile-check함; test 실행과 concrete SDK failure injection은 미구현 |

### P2 — reload 안정화 뒤

| 항목 | 상태 | 비고 |
|---|---|---|
| cache budget/eviction tuning | 보류 | 먼저 process-lifetime keep-warm으로 correctness 확보 |
| runtime full reset/recovery | 보류 | 모든 QNN client idle 조건 필요 |
| NPU vision + NPU LLM stress test | 보류 | two-binary 확장성 검증 |
| cross-context zero-copy | 보류 | host-copy를 유지한 채 lifecycle부터 안정화 |

## 6. `8cdc4dd3` workaround와 현재 HEAD의 성능/구조 평가

### 긍정적 효과

- 다음 load의 contextCreateFromBinary와 graph 준비 비용을 피할 수 있다.
- 제공된 after log의 두 번째 prefill은 첫 prefill보다 짧았지만 통제된 benchmark는 아니다.
- 가장 위험했던 context-before-deregister 순서를 피한다.

### 현재 리스크

- binary path가 늘어날 때 `ct_map` context와 HTP 메모리가 process 종료까지 남는다.
- context 내부의 stateful graph/extension 상태가 logical unload 뒤 완전히 fresh한지 보장되지 않는다.
- manager-wide lifecycle guard 때문에 서로 다른 context의 run과 cleanup도 필요 이상으로 직렬화되고, pointer별 unload가 exclusive lock을 반복 획득해 latency/jitter가 커질 수 있다.
- 같은 context의 graph retrieve/handle publication과 execute 병렬성이 SDK에서 안전한지 확인하지 못했다.
- 실패는 이제 상위 API까지 error 7/sticky terminal로 전파된다. 기존 앱에는 외형상 regression이 될 수 있고 실제 device에서 deregistration 실패·retention을 주입 검증하지 못했다.
- 두 번째 run의 latency 개선과 context memory 상주 비용을 함께 계측하지 않았다.

### 현재 배포 판단

- 제공된 before/after 로그는 `8cdc4dd3` workaround의 단 한 cycle만 검증한다. 현재 21-commit HEAD의 device 결과가 아니다.
- 현재 HEAD는 정적/부분 compile을 통과한 **device-validation candidate**이며 production-ready로 선언하지 않는다.
- 여러 QNN handle, 서로 다른 binary, partial unload, background 병렬 실행은 동시성 시험 전 안전하다고 간주하지 않는다.
- 장시간/여러 모델을 load하는 제품은 memory soak와 bounded cache policy 없이 완료로 간주하지 않는다.

## 7. 현재 세션에서 완료한 작업

- [x] before/after log의 핵심 event를 재계수
- [x] lazy-create 경고/진입 2→1, dereg failure 62→0, graph failure 36→0 확인
- [x] Gemma prefill/generation의 shared-context 소유 구조 확인
- [x] `Quick_Dot_AI_QNN` → `NeuralNetwork` → tensor pool/`QNNGraph` 실제 파괴 순서 확인
- [x] `8cdc4dd3` workaround가 ordering과 keep-warm 두 변수를 동시에 바꿨음을 확인
- [x] “context recreate 자체가 불가능” 결론을 미검증으로 교정
- [x] P0/P1 정적 결함 목록 갱신
- [x] keep-warm, strict recreate, runtime reset 정책 비교
- [x] reload 중심 분석 문서 전면 재작성
- [x] 단계별 구현 계획 전면 재작성
- [x] 구현 현황 문서 갱신
- [x] R0 source code 수정 — 실행 실패 전파와 C API 실패 계약
- [x] per-forward descriptor RAII/direct binding 구현 및 commit
- [x] context-create 사전조건 검증 기반 commit
- [x] transactional context create/free 본체 commit
- [x] PR 4 lossless RPC registration/backing ownership 구현과 독립 exact-diff 리뷰
- [x] `QNNGraph` process-wide destructor sweep 제거
- [x] PR 4 commit `15873b03` 완료
- [x] Engine allocator lookup 동기화 commit `4577a696`
- [x] CausalLM/QNN RPC allocator 통합 commit `5e2580b8`
- [x] plugin once 초기화, checked release/retention, explicit model shutdown, C/JNI/Kotlin teardown status 구현
- [x] direct RPC/RoPE/manager allocation의 ownership-first 전환
- [x] 두 QNN model 조립 실패의 temporary-owner checked cleanup
- [x] clang-format/diff check, host compile, Android NDK 부분 compile, JNI object, Kotlin offline compile
- [x] 사용자 Linux 환경의 전체 Android app/QNN SDK build
- [ ] device validation

## 8. 현재 구현 경계와 다음 단위

PR 1과 PR 2 후보는 완료했다. PR 3은 다음 두 commit으로 분리했다.

1. `9c1c717f [qnn] validate context creation prerequisites` — 함수 포인터, binary size/open/mmap, metadata optional 역참조의 사전조건 검증
2. `24e184c6 [qnn] make context lifecycle transactional` — 완료

PR 3 본체의 현재 정책은 다음과 같다.

- registry entry는 `CREATING → ACTIVE`로 최종 성공 시에만 재사용 가능하다.
- create/free/rollback이 불확실하게 실패하면 entry를 `QUARANTINED`로 남겨 handle, copied metadata, binary mmap을 보존한다.
- 한 context가 quarantine되면 기존 `ACTIVE` cache hit은 유지하지만 새 context miss 생성은 차단한다.
- `contextFree()`가 성공한 뒤에만 host metadata와 binary mapping을 제거한다.
- config list는 호출 지역 owner이고 extension config는 borrowed pointer로 취급해 임의 `free()`하지 않는다.
- SystemContext metadata는 handle을 닫기 전에 deep-copy하며 모든 return/exception 경로를 guard가 닫는다.
- normal unload의 keep-warm 동작은 바꾸지 않는다. 이 PR의 `freeAllContexts()`는 QNN runtime destructor 경로다.
- binary mmap은 persistent-binary config 가능성을 보수적으로 포괄하기 위해 cached context가 성공적으로 free될 때까지 유지한다.

세 독립 리뷰에서 나온 필수 수정인 global quarantine gate, `ACTIVE`-only normal free, partial metadata cleanup count, zero-count 방어, destructor catch-all과 실제 process-lifetime retention을 반영했다. PR 3은 `9c1c717f`와 `24e184c6` 두 commit으로 닫았다. 다음 단위는 `QNNRpcManager`의 failed-deregister entry/backing 보존과 `QNNGraph` process-wide destructor sweep 제거다.

R0 진행 체크포인트:

- `QNNGraph::forwarding()`이 실패한 `graphExecute`의 raw 64-bit rc, `QNN_GET_ERROR_CODE()` 공개 오류 코드, binary, graph, context와 graph handle을 기록하고 `std::runtime_error`를 던지도록 수정함.
- QNN Gemma run 시작에서 `has_run_ = false`, performance metric zero-reset을 추가함.
- non-stream `run_on_handle()`에 catch-all을 추가해 동적 backend 예외가 C ABI 밖으로 나가지 않게 함.
- 기존 streaming helper는 이미 예외를 `CAUSAL_LM_ERROR_INFERENCE_FAILED`로 변환함을 확인함.
- clang-format 14, `clang-format-14 --dry-run --Werror`, `git diff --check` 통과.
- QAIRT 2.47 헤더에서 context/graph handle이 `void *` alias이고 `Qnn_ErrorHandle_t`가 `uint64_t`임을 확인해 현재 진단 형식의 타입 호환성을 검증함.
- `QnnGraph_execute()` 비정상 반환 뒤 output 유효성 보장은 없으므로 실패 output을 소비하지 않고 즉시 전파하는 정책을 확정함.
- commit `467eeadd [qnn] propagate graph execution failures` 완료.
- commit `6cc4f924 [CausalLM] preserve QNN inference failure state` 완료.
- QNN SDK/vendor tree가 없어 compile/device test는 미수행.

PR 3 진행 체크포인트:

- commit `9c1c717f [qnn] validate context creation prerequisites` 완료.
- commit `24e184c6 [qnn] make context lifecycle transactional` 완료.
- `Qnn_Context_Graph_t`에 `CREATING/ACTIVE/QUARANTINED` 상태와 binary mapping owner를 추가했다.
- raw binary path가 같은 `ACTIVE` entry의 기존 keep-warm cache hit은 mmap/system hook/core create 없이 즉시 성공한다.
- core create 이전에 `CREATING` map node를 확보하고, core와 extension 후처리가 모두 성공한 경우에만 `ACTIVE`로 전환한다.
- create error가 non-null handle을 남기거나 rollback/free가 실패하면 host가 임의 정리하지 않고 runtime 신규 create를 차단한다.
- `freeContext()`는 `ACTIVE`만 정상 해제하며 SDK success 전에는 handle/metadata/map/mmap을 지우지 않는다.
- QNNContext destructor의 cleanup 예외와 free 실패는 backend/device/dlclose로 진행하지 않고, distinct runtime별 leaked shared-owner에 보존한다.
- context config의 stack/extension pointer를 entry에 저장하거나 직접 free하지 않도록 변경했다.
- metadata helper의 부분 할당을 되돌릴 수 있도록 copy 전 예상 graph count를 기록하고 count 0 dereference를 막았다.
- `ct_map` lifecycle mutex/lease와 context-scoped RPC registration은 아직 이 PR에 포함하지 않았다.
- clang-format 14 dry-run과 `git diff --check` 통과. Android/QNN compile 및 device test는 미수행.

PR 4 진행 체크포인트:

- 정상 registration owner를 `QNNGraph`가 아니라 activation `MemoryPool`/RPC allocation으로 확정했다.
- 현재 unload는 각 `NeuralNetwork`의 pool deallocate가 graph destructor보다 먼저 실행된다. 따라서 global graph-destructor sweep을 제거해도 각 pool이 live cached context에서 자기 pointer를 순차 deregister할 수 있다.
- `QNNRpcManager` registry를 `pointer → context → registration`으로 바꿔 같은 RPC allocation의 context별 memHandle을 분리했다.
- registration은 `REGISTERING/ACTIVE/QUARANTINED` 상태를 가지며, `memRegister` 오류가 non-null output을 남기거나 `memDeRegister`가 실패하면 entry를 지우지 않는다.
- pointer free에서 context별 deregistration이 하나라도 실패하면 성공 entry만 제거하고 RPC backing allocation은 manager가 계속 소유한다.
- registration/allocation map에 mutex를 추가하고, graph forwarding은 shared execution guard, pointer/context cleanup은 exclusive lifecycle guard를 사용한다.
- `contextFree`는 해당 context registration count가 0일 때만 호출하도록 precondition을 추가했다.
- manager가 `libQnnHtp.so`를 다시 dlopen/provider 조회하지 않고 이미 초기화된 QNN interface table을 주입받도록 변경했다.
- cache hit은 같은 pointer/context뿐 아니라 datatype과 dimensions까지 일치해야 재사용한다. 반복 forward에서는 dimension vector를 새로 만들지 않고 저장된 fingerprint와 직접 비교한다.
- lazy context miss는 shared execution guard를 놓고 exclusive cleanup guard에서 double-check/create한 뒤 shared guard를 다시 얻는다. shared→exclusive 중첩 upgrade deadlock과 create/free 경합을 피한다.
- `QNNContext::load()`, pointer free, context free/runtime teardown은 exclusive guard를 사용하고 forwarding의 lookup/retrieve/register/execute 전체는 shared guard를 유지한다.
- runtime shutdown은 admission을 영구 폐쇄하고 context cleanup부터 backend/device release와 backend DSO close까지 exclusive guard를 유지한다. teardown 뒤 대기 forwarding/load가 stale function table로 재진입하지 못한다.
- `QNNGraph` destructor의 global sweep과 사용되지 않던 자체 context-owner 필드를 제거했다. 정상 teardown은 `NeuralNetwork::deallocate()`가 먼저 실행하는 각 `MemoryPool`의 pointer별 allocator free가 담당한다.
- 호출처가 0개가 된 public process-wide deregistration API도 제거해 같은 blast radius가 다시 도입될 경로를 닫았다.
- 직접 QNNRpcManager host test는 gitignored vendor header/Android QNN module 결합 때문에 불가능하다. SDK-free generic ledger를 분리하면 failure-injection test가 가능하지만 이번 commit 범위에 넣을지는 리뷰 후 결정한다.
- shutdown 경합 수정 후 두 차례 독립 lifecycle/ownership exact-diff 재검토에서 compile blocker/P0/P1이 없음을 확인했고 clang-format 14 dry-run 및 `git diff --check`를 통과했다. commit `15873b03`으로 완료했으며 실제 SDK compile/device run은 남았다.

PR 5 진행 체크포인트:

- Engine allocator registry에 mutex-protected `getAllocator(name)`를 추가하고 기존 bulk getter도 동기화했다 — `4577a696`.
- CausalLM의 별도 `libcdsprpc` dlopen/dlsym과 전역 function pointer를 제거하고 Engine의 `qnn` allocator로 모든 앱 QNN buffer를 할당/해제한다 — `5e2580b8`.
- 첫 성공 조회의 allocator `shared_ptr` holder를 process-lifetime으로 유지해 function-static model destructor가 Engine 종료 뒤 registry를 다시 조회하지 않게 했다.
- `deallocate()`는 `noexcept`이고 model graph를 먼저 파괴한 뒤 같은 manager가 context별 registration을 해제하고 backing을 free한다.
- Windows clang++와 MinGW g++에서 allocator translation unit C++17 `-fsyntax-only -Wall -Werror`를 통과했다.
- 변경 4개 파일 clang-format 14 dry-run과 `git diff --check`, 독립 ownership/DSO lifetime 리뷰를 통과했다.
- 실제 Android link/run, QNN device cycle, OOM unwind와 process-exit device test는 미수행이다.

## 9. 검증 현황

### 수행함

- current HEAD/diff와 ancestor commits 확인
- source-level ownership/lifecycle 추적
- before/after log pattern 정량 집계
- forwarding, context create/free, RPC register/free failure path 정적 감사
- Android unload/destroy handle 흐름 정적 감사
- Qualcomm 공식 release note의 ION deregistration/HTP execute known-issue 유형과 비교
- PR 4 변경 8개 파일의 독립 lifecycle/ownership exact-diff 재검토
- PR 4 변경 8개 파일 `clang-format-14 --dry-run --Werror`, `git diff --check`
- PR 5 allocator source Windows clang++/MinGW g++ C++17 syntax-only compile
- PR 5 변경 파일 clang-format 14 dry-run, diff check, 독립 DSO/teardown 리뷰
- PR 7 `MemoryPool`, `TensorPool`, `CachePool`, `MemAllocator` MinGW host object compile
- PR 7 memory-pool unit test source MinGW C++17 syntax-only compile
- PR 7 변경 파일 clang-format 14 dry-run, diff check, 독립 ABI/ownership/lifecycle 리뷰
- PR 8 ccapi Windows object compile/symbol 확인, Android NDK arm64 QNN/C API syntax compile
- PR 8 실제 Quick JNI object build와 Kotlin offline compile
- PR 9 CausalLM direct RPC/RoPE 변경 Android NDK arm64 syntax compile
- PR 10 임시 최소 QNN type stub 기반 `ENABLE_QNN` Android NDK arm64 syntax compile과 non-QNN host compile
- PR 11 experimental multimodal macro OFF/ON Android NDK arm64 syntax compile
- Issue #49 실제 과거 QNN API 2.36/QAIRT 2.47 호환 header에서 수정 전 6개 오류 재현 및 수정 후 syntax compile
- Issue #49 fake header 기반 clang++/g++ QNN-enabled translation-unit compile과 header-alone compile
- 사용자 Linux 환경의 `build_android.sh --enable-qnn --install --clean` 전체 build 성공

### 수행하지 못함

- host QNN unit test — 기본 Meson/공개 CI가 QNN source를 빌드하지 않고 fake seam이 없음
- QNN device cycle test
- strict context purge/recreate
- failure injection
- memory/thermal/HTP profiling
- multi-handle/two-binary test

따라서 현재 판단은 `8cdc4dd3`에서 얻은 제공 로그, 현재 HEAD의 정적 분석과 부분 compile에 근거하며 current HEAD의 device 결과는 아니다.

## 10. 체크포인트 운용 규칙

사용자 요청에 따라 컨텍스트가 길어지기 전에 다음 내용을 이 문서들에 즉시 기록한다.

- 새로 확정된 사실과 근거
- 이전 결론이 바뀐 경우 변경 전/후와 이유
- 구현된 파일과 핵심 동작
- 수행한 테스트와 정확한 결과
- 실패/막힘/미검증 항목
- 다음 구현 경계

세 문서의 역할은 다음과 같다.

- `qnn_reload_analysis.ko.md`: 왜 문제가 생겼는지와 설계 판단
- `qnn_reload_implementation_plan.ko.md`: 앞으로 무엇을 어떤 순서로 바꿀지
- `qnn_reload_implementation_status.ko.md`: 실제로 무엇이 구현·검증됐는지

## 11. 변경 기록

### 2026-07-23 — reload 중심 상태 재작성

- 멀티모달 구현 상태를 주 내용에서 제거했다.
- 현재 workaround의 구현/검증/미해결 경계를 분리했다.
- 31회 deregistration 실패의 실제 destructor 순서를 반영했다.
- execute failure 은폐, failed-deregister backing free, transactional create/free 부재를 P0로 승격했다.
- strict recreate는 미구현/미검증으로 명시했다.
- 현재 세션에는 source 변경과 build/device test가 없음을 명시했다.
- 교차검토 후 SDK context의 실제 해제 성공을 단정하지 않고 host-side 폐기 이후 실패로 교정했다.
- SDK 실제 create 호출 계수가 아니라 lazy-create 경고/진입 경로 계수임을 명시했다.

### 2026-07-23 — 구현 착수

- 대상 SDK version을 QAIRT/QNN 2.47로 확정했다.
- 첫 독립 PR 후보를 R0 실행 실패 전파와 진단 로그로 정했다.
- source 수정 전 call-stack, 2.47 API 계약, host test wiring 감사를 시작했다.
- R0 첫 source 변경 3개 파일을 적용하고 정적 diff 검사를 완료했다.
- QAIRT 2.47 헤더 타입 및 `graphExecute` 오류 계약을 확인하고 raw/public 오류 코드를 함께 기록하도록 보강했다.
- 첫 PR은 현재 3개 파일의 end-to-end failure contract에서 닫고, lifecycle transaction과 fake seam은 다음 PR로 분리하기로 했다.
- 첫 PR 후보의 두 commit을 DCO sign-off와 agent co-author trailer를 포함해 완료했다.

### 2026-07-23 — PR 2 후보: per-forward tensor binding

- 변경 중: `nntrainer/qnn/jni/iotensor_wrapper.hpp`, `QNNGraph.cpp`, `QNNGraph.h`.
- input/output descriptor를 `unique_ptr` owner로 감싸 모든 성공·실패·예외 경로에서 SDK `freeQnnTensors()`로 metadata를 정리하도록 변경했다.
- setup status를 검사해 null/partial descriptor 실행을 차단했다.
- token마다 증가하던 member buffer vector와 typed variant를 제거하고 현재 NNTrainer Tensor의 RPC backing pointer를 직접 QNN mem registration에 bind하도록 변경했다.
- MEMHANDLE descriptor의 null client buffer로 수행되던 NATIVE populate no-op을 제거했다.
- descriptor cleanup은 memHandle deregistration이나 RPC backing free를 수행하지 않는다. 두 수명은 `QNNRpcManager`가 별도로 소유한다.
- context create/free 변경은 이 PR 후보에서 제외했다.
- clang-format 14, dry-run, `git diff --check`, 독립 정적 리뷰를 통과했다.
- commit `facd1111 [qnn] make tensor bindings exception safe` 완료.
- Android/QNN compile, ASan/LSan 및 device memory plateau 검증은 환경 제약으로 미수행이다.

### 2026-07-23 — PR 3 후보: transactional context lifecycle

- 기반 commit `9c1c717f [qnn] validate context creation prerequisites`를 완료했다.
- context entry 상태를 `CREATING/ACTIVE/QUARANTINED`로 분리하고 `ACTIVE`만 lookup/cache reuse 대상으로 제한했다.
- mmap, SystemContext, copied graph metadata, extension hooks, core context create를 하나의 rollback 흐름으로 묶었다.
- `contextFree`/rollback 실패 시 SDK live 상태를 host에서 잊지 않고 metadata와 persistent binary mapping을 보존한다.
- quarantine이 생긴 runtime에서는 다른 binary의 신규 create도 차단하되 이미 `ACTIVE`인 entry의 cache hit은 유지한다.
- normal free와 create rollback의 허용 상태를 각각 `ACTIVE`와 `CREATING`으로 분리했다.
- binary-created context의 extension free hook은 인접 Qualcomm 구현의 HTP extension crash 주의 때문에 임의 추가하지 않았다. strict-free 기기 검증 항목으로 남긴다.
- destructor 실패는 덮어쓸 수 있는 function-static 한 칸이 아니라 의도적으로 파괴되지 않는 per-runtime holder로 보존한다.
- 세 독립 정적 리뷰와 수정 후 재검토를 통과하고 commit `24e184c6`으로 완료했다.

### 2026-07-23 — PR 4 후보: lossless RPC registration teardown

- registration key를 raw pointer 하나에서 `(pointer, QNN context)`로 확장하고 descriptor datatype/dimensions fingerprint를 함께 보존했다.
- graph 실행은 shared lifecycle guard, load/free/runtime teardown은 exclusive guard로 보호한다. lazy create는 lock upgrade 대신 unlock/double-check/relock을 사용한다.
- `memRegister`/`memDeRegister`의 불확실한 실패는 terminal quarantine으로 남기고 RPC backing을 해제하지 않는다. 성공한 context registration만 erase한다.
- allocator의 pointer별 free가 정상 registration owner가 되며, `QNNGraph` destructor의 process-wide sweep과 public global deregistration API는 제거했다.
- manager가 별도 QNN backend DSO/provider를 고르지 않고 QNNContext가 선택한 function table을 주입받는다.
- 반복 cache hit의 dimension vector allocation을 없애 hot path 부하를 줄였다.
- 두 독립 exact-diff lifecycle 검토와 clang-format 14 dry-run, whitespace 검사를 통과했다. SDK compile/device cycle과 failure injection은 환경 제약으로 미수행이다.
- `MemAllocator::free(void)`는 실패 status를 반환하지 않으므로 unload API 실패 전파는 다음 explicit session shutdown PR로 남는다.

### 2026-07-23 — PR 5: CausalLM/QNN allocator ownership 통합

- `4577a696`에서 Engine allocator 단일 조회와 bulk map 복사를 mutex로 보호했다.
- `5e2580b8`에서 CausalLM의 별도 rpcmem loader를 제거하고 QNN plugin이 설치한 `QNNRpcManager`를 재사용한다.
- 앱 buffer와 NNTrainer pool buffer가 같은 allocation/registration ledger를 거치므로 context가 2개여도 `(pointer, context)`별 memHandle을 한 manager가 정리한다.
- 정적 handle/Engine 파괴 순서를 피하기 위해 allocator holder는 현재 process-lifetime plugin 정책에 맞춰 의도적으로 파괴하지 않는다.
- 앱 destructor는 graph를 먼저 정지/파괴한 뒤 app buffer를 manager에 반환한다. deregistration 실패는 backing 보존으로 끝나며 destructor 밖으로 예외를 내보내지 않는다.
- 두 host compiler syntax check와 독립 리뷰를 통과했다. Android/QNN device 검증은 남았다.

### 2026-07-23 — PR 6 진행: context plugin 중복 초기화 제거

- 선행 commit `fc14f16d [core] make dynamic library errors value safe`를 완료했다.
- 공개 `DynamicLibraryLoader::getLastError()`의 `const char *` 계약은 유지하고, 내부 호출부가 즉시 소유하는 `getLastErrorString()`을 추가했다. Windows의 기존 임시 문자열 dangling pointer는 `thread_local std::string` backing으로 제거했다.
- `dlopen`/`LoadLibrary`와 `dlsym`/`GetProcAddress` 전에 플랫폼 오류 상태를 지워, 성공한 호출이 이전 실패의 stale 오류로 거부되지 않게 했다.
- AppContext와 Engine은 라이브러리를 연 직후 RAII guard를 설치한다. 단, 기존 Engine registry가 아직 transactional하지 않으므로 publish 호출 직전에는 handle을 release해 예외 시 이미 공개된 context의 DSO가 닫히는 회귀를 피했다. 다음 commit에서 registry transaction과 함께 이 경계를 완성한다.
- 공개 헤더 ABI/소스 호환성, 예외 경로와 double-close를 독립 재검토했고 P0/P1이 없음을 확인했다.
- `app_context.cpp`와 `engine.cpp`는 MinGW g++ 및 Windows clang++ C++17 syntax-only 검사를 통과했다. 6개 변경 파일은 clang-format 14 dry-run과 `git diff --check`를 통과했다.
- 본체 commit `5163e606 [engine] avoid duplicate context plugin initialization`을 완료했다.
- process-wide plugin identity를 bare loader name과 명시 file path의 두 domain으로 분리했다. 명시 경로는 CWD 기준 absolute/lexical load path를 유지하되 cache key만 `weakly_canonical`로 alias를 합친다.
- path별 `std::once_flag` record가 성공한 Context/DSO를 직접 process lifetime 소유한다. factory 예외는 Context→DSO 순으로 정리되고 once flag가 완료되지 않아 다음 명시 등록이 재시도한다.
- 같은 path를 여러 Engine이 호출해도 같은 Context 포인터를 각 Engine registry에 attach한다. 다른 포인터가 동일 context name을 이미 차지하면 조용한 오등록 대신 collision 오류를 반환한다.
- Engine name/allocator publish는 external allocator 조회를 mutex 밖에서 수행하고, 두 번째 확인 뒤 두 map을 transactionally 삽입한다. allocator 삽입 예외는 name map을 rollback한다.
- 독립 동시성 검토에서 P0/P1 없음이 확인됐다. 남은 P2는 invalid path record의 작은 process-lifetime 누적, 다른 path/same-name collision record의 stranded resource, 동일 identity factory 재귀 시 `call_once` self-deadlock 가능성이다. QNN factory는 재귀 등록하지 않는다.
- `engine.cpp`와 self-contained `engine.h`는 MinGW g++/Windows clang++ C++17 syntax-only 검사, clang-format 14 dry-run, `git diff --check`를 통과했다.
- test commit `82a05282 [test] cover context plugin initialization lifecycle`을 완료했다.
- 동일 fake source를 서로 다른 identity/name으로 5개 shared module로 빌드해 순차 same-path, 16-thread 동시 same-path, explicit path alias, factory 1회 실패 후 재시도, 다른 pointer의 same-name collision을 한 native host test에서 검증한다.
- 별도 `Engine` 인스턴스도 같은 path record의 동일 Context 포인터를 attach하는지 확인하고, probe handle을 닫은 뒤 virtual `getName()`을 호출해 Engine record가 DSO를 계속 보유하는지 검사한다.
- Windows의 기존 generic entry 선언에 `dllexport`를 재부여하지 않도록 `.def` 파일로 entry/probe symbol을 export한다. Meson 0.55 호환을 위해 `range()` 대신 명시 ID 배열을 사용한다.
- test source는 MinGW g++, Windows clang++, clang-cl `/W1 /WX` syntax 검사를 통과했다. Meson source parser는 통과했지만 전체 configure는 로컬 Ninja/GTest 부재로 test subdir 전에 중단되어 실제 test binary build/run은 미수행이다.
- 독립 test 리뷰에서 P0/P1 없음이 확인됐다. Windows native CI의 기본 static-library 설정에서는 이 shared-DSO test가 gate되어 실행되지 않는 커버리지 제한이 남는다.
- 이 지점까지 PR 6으로 독립 제출 가능한 경계다: `fc14f16d`, `5163e606`, `82a05282`.

### 2026-07-23 — PR 7: checked allocator release와 QNN retention 신호

- `d06e0c5f [core] preserve buffers after release failures`를 완료했다.
- 설치되는 `MemAllocator`의 기존 virtual 함수와 객체 layout은 바꾸지 않고, 기존 `free(void *)`를 호출해 예외를 bool로 변환하는 non-virtual inline `tryFree()`를 추가했다.
- allocator는 정상 반환한 경우에만 caller가 pointer ownership을 잊을 수 있다. 해제를 확정하지 못하면 예외를 내고 backing/ownership record를 유지한다.
- `MemoryPool::deallocate()`는 모든 unique buffer를 끝까지 시도하며 성공 pointer만 즉시 owner vector에서 지운다. 실패 pointer는 남기고 tensor layout/pointer는 무효화한 뒤 오류를 올린다.
- `TensorPool`은 부분 해제 뒤 stale tensor를 다시 실행하지 못하도록 성공/실패와 무관하게 모든 tensor data binding을 null로 만든다.
- `MemoryPool` destructor는 no-throw fallback이다. 실패 backing은 allocator의 ledger가 계속 보존하며 destructor 밖으로 예외를 내보내지 않는다.
- FSU resize는 old buffer free가 실패하면 alias/owner를 old buffer에 유지하고 새 buffer를 rollback한다. 새 buffer rollback도 실패하면 두 allocation을 모두 owner list에 보존한다.
- `CachePool`은 active cache를 invalidate하고 swap device를 종료한 뒤 backend pool을 한 번만 해제한다. 부분 backend free 뒤 cache가 dangling pointer를 swap-out하는 UAF와 중복 retry를 피한다.
- fake allocator unit test는 partial release 후 failed-only ownership, destructor containment, FSU old-release 실패 rollback, old/new 양쪽 실패 보존을 검증한다.
- `7aa3527d [qnn] report retained RPC allocations`를 완료했다.
- `QNNRpcManager::free()`는 cleanup admission closed, unknown pointer, memDeRegister failure/quarantine 잔존에서 더 이상 정상 반환하지 않는다. 로그와 lossless retention 후 예외를 내 `tryFree()`가 실패를 정확히 판정한다.
- QNN deregistration 실패는 terminal quarantine이라 generic fake allocator test처럼 재시도 성공을 보장하지 않는다. 보장은 backing의 영구 안전 보존과 상위 unload 실패 신호다.
- Android core/QNN/plugin은 같은 NDK C++ runtime과 exception 설정으로 함께 배포한다는 전제가 있다. 구 QNN plugin과 새 core를 섞으면 semantic ABI 계약이 맞지 않으므로 artifact 동시 배포가 필요하다.
- MinGW host object compile 4개와 test-source syntax compile, clang-format/diff check, 독립 재검토를 통과했다. QNN SDK compile과 실제 failure injection/device run은 미수행이다.
- 이 지점은 PR 7로 독립 제출 가능한 경계다: `d06e0c5f`, `7aa3527d`.
- 다음 PR은 concrete QNN model의 one-phase, terminal, idempotent shutdown과 CausalLM unload/destroy 오류 반환을 연결한다. registration scope 기반 true transaction/batch release는 그 뒤로 분리한다.

### 2026-07-23 — PR 8: 명시적 QNN model shutdown과 앱 오류 전파

- `9d84c730 [core] expose explicit model deallocation`을 완료했다. `Model` virtual 함수나 layout을 바꾸지 않고 기존 ccapi bridge에 `deallocateModel(Model &) noexcept` free function을 추가했다.
- `38227432 [CausalLM] make QNN model shutdown explicit`을 완료했다. `Quick_Dot_AI_QNN::shutdown() noexcept`가 각 NNTrainer model의 tensor/weight pool을 먼저 해제하고 direct RPC allocation을 checked 방식으로 정리한다.
- shutdown은 one-phase terminal 동작이다. 성공한 자원만 잊고, 실패 model은 null map entry, 실패 direct pointer는 set entry로 남겨 반복 호출이 과거 실패를 성공으로 덮지 못하게 했다.
- partial initialization 중에도 만든 graph model을 즉시 `models`에 공개하므로 compile/initialize/load-weight 예외 뒤 explicit shutdown 대상에서 누락되지 않는다.
- 이 shutdown은 QNN SDK context를 free하지 않는다. binary path별 context/graph는 현재 keep-warm 정책에 따라 남고, 이번 PR은 session/model-local buffer 소유권만 닫는다.
- `d6a4c818 [CausalLM] report model teardown failures`를 완료했다. C API에 `CAUSAL_LM_ERROR_RESOURCE_RELEASE_FAILED`를 추가하고 unload/destroy/load-replace 및 partial-load cleanup에서 sticky failure를 보존한다.
- destroy는 오류를 반환하더라도 handle 자체는 삭제한다. release 실패 뒤 같은 handle의 reload는 차단한다.
- `10bee850 [CausalLM] propagate teardown errors to Android`를 완료했다. 기존 JNI load symbol은 ABI 호환용으로 유지하고 status-bearing 결과 API를 추가했다. Kotlin은 teardown 실패를 terminal로 기억하고 reload를 차단한다.
- ccapi Windows object compile/symbol 확인, Android NDK arm64 QNN/C API syntax compile, 실제 QuickDotAI JNI object build, Kotlin offline compile, clang-format/diff check를 통과했다.
- 실제 vendor QNN 실행과 failure injection은 수행하지 못했다. 외부 proprietary `Quick_Dot_AI_QNN` subclass가 derived destructor에서 base buffer를 참조하는지도 별도 호환성 확인이 필요하다.
- 이 지점은 PR 8로 독립 제출 가능한 경계다: `9d84c730`, `38227432`, `d6a4c818`, `10bee850`.
- 다음 PR은 raw RPC allocation 뒤 ownership set 등록 사이의 예외 창을 줄인다. 특히 Gemma RoPE의 두 번째 allocation 실패와 set-node allocation 실패를 분리해 처리한다.

### 2026-07-23 — PR 9: direct RPC allocation의 ownership-first 전환

- `3f8ea209 [CausalLM] track QNN allocations before acquisition`을 완료했다.
- 기존에는 RPC buffer를 먼저 얻고 `std::set` node를 나중에 할당했다. set-node OOM이면 live pointer가 model-local owner set에 들어가지 못했다.
- 새 helper는 임시 key로 set node를 먼저 확보하고 `extract()`한 뒤 RPC buffer를 할당한다. 같은 node의 key를 실제 pointer로 바꿔 재삽입하므로 metadata OOM은 RPC acquisition 전에 발생한다.
- `get_qnn_input_data(TensorInfo, std::set<void *> &)` 전역 심볼과 `Quick_Dot_AI_QNN` layout/vtable은 그대로 유지했다. 외부 파생 QNN 모델의 binary ABI를 불필요하게 바꾸지 않는다.
- `20e8355b [CausalLM] make QNN RoPE allocation exception safe`를 완료했다.
- Gemma full/sliding RoPE는 cos와 sin buffer를 각각 `tracked_allocate()`한 직후 owner set에 넣고, 별도 `fill_cos_sin()`이 caller-owned buffer를 채운다. 두 번째 allocation 또는 fill 실패 시 앞선 buffer는 explicit shutdown 대상에 남는다.
- 기존 `get_cos_sin(...)` 심볼은 compatibility wrapper로 유지하고, 예외 시 두 buffer를 checked rollback한다. rollback도 실패하면 QNN manager가 backing을 보존하지만 model-local sticky status에는 연결되지 않는 legacy-path P1이 남는다.
- 호출자가 없는 `get_zero_memory()` raw allocation API도 ABI 때문에 제거하지 않고 caller-owned 계약을 주석으로 명시했다.
- RoPE dimension/size overflow와 null output을 검증하고, large valid dimension에서 index 계산이 signed overflow하지 않도록 `size_t` offset을 사용한다.
- NDK r27 arm64에서 Quick/generate utility는 `-Werror` syntax compile, Gemma는 기존 missing-override 경고만 non-error 처리한 syntax compile을 통과했다. generate utility object와 기존/new RoPE 심볼도 `llvm-nm`으로 확인했다.
- 정상 경로의 set node 수와 RPC buffer 수는 이전과 같다. load 시 node extract/reinsert tree 연산이 추가되지만 run hot path 비용은 없으며 vendor allocation 비용보다 작다.
- 이 지점은 PR 9로 독립 제출 가능한 경계다: `3f8ea209`, `20e8355b`.
- PR 9 추가 감사에서 `QNNRpcManager::alloc()`도 RPC backing 획득 뒤 `allocations_` map node를 할당함을 확인했다. 이 문제는 다음 PR 10 (`f7f64236`)에서 같은 ownership-first 순서를 적용해 해결했다.

### 2026-07-23 — PR 10: QNNRpcManager allocation ledger 선확보

- `f7f64236 [qnn] reserve allocation ledger before RPC acquisition`을 완료했다.
- `QNNRpcManager::alloc()`은 lifecycle shared guard와 registration mutex를 얻은 뒤 `allocations_` map node를 임시 key로 먼저 확보·extract하고, 그 다음 `rpcmem_alloc`을 호출한다.
- rpcmem이 null/예외를 반환하면 node handle만 파괴되고 map은 원래 상태다. 성공하면 node key를 실제 pointer로 바꾸어 같은 map에 재삽입한 뒤에만 caller의 `*ptr`에 공개한다.
- backing acquisition 뒤에는 map-node allocation이나 두 번째 mutex 획득이 남지 않는다. metadata OOM, runtime shutdown, mutex failure가 모두 backing 생성 전에 끝난다.
- `QNNRpcManager` header, member, vtable, class layout과 public signature는 바꾸지 않았다. `ENABLE_QNN` off의 calloc/free fallback도 그대로다.
- duplicate live address를 rpcmem이 반환하는 allocator 계약 위반은 pointer-key map과 `rpcmem_free(ptr)`만으로 두 allocation을 구분할 수 없다. 기존 정책대로 ambiguous address를 free하지 않고 오류를 낸다.
- registration mutex를 vendor `rpcmem_alloc` 동안 보유하므로 병렬 model load의 allocation은 직렬화된다. 동시에 run 중인 다른 handle의 `registerQnnTensor` fast path도 그 시간만큼 대기할 수 있다. 단독 load→run과 steady-state run에는 새 비용이 없다.
- QNN SDK vendor tree가 없는 환경이므로 임시 최소 QNN type stub으로 Android NDK r27 arm64 `ENABLE_QNN` syntax compile과 Windows host non-QNN syntax compile을 수행했다. clang-format 14와 diff check도 통과했다.
- 이 지점은 PR 10으로 독립 제출 가능한 경계다: `f7f64236`.
- deterministic map-OOM/RpcMem failure injection은 기본 map allocator와 private singleton 때문에 아직 test seam이 없다. 이를 위해 production helper/template와 새 test target을 추가하는 것은 이번 작은 correctness PR과 분리한다.

### 2026-07-23 — PR 11: 멀티모달 임시 handle cleanup status 보존

- `32f6575d [CausalLM] preserve multimodal load cleanup failures`를 완료했다.
- 이 PR은 주 reload 원인의 구조를 바꾸지 않는 보조 확장성 수정이다. experimental `loadMultimodalHandleByName()`이 vision encoder와 LLM을 각각 임시 handle로 load한 뒤 조립하는 실패 경로만 다룬다.
- peer LLM load, 모델 개수 검증, embedding compatibility, combined handle allocation, vector/string move 중 어느 지점에서 실패해도 combined→LLM temp→vision temp를 모두 explicit unload한다.
- 한 cleanup이 실패해도 나머지 cleanup을 short-circuit하지 않는다. 하나라도 실패하면 기존 MODEL_LOAD_FAILED/UNSUPPORTED/UNKNOWN보다 `CAUSAL_LM_ERROR_RESOURCE_RELEASE_FAILED`를 우선 반환한다.
- combined handle은 `std::unique_ptr`로 조립하고 성공 시에만 release한다. 부분 move 예외에도 모델 owner가 정확히 하나라 leak/double-free/C ABI exception escape를 막는다.
- 각 descriptor가 정확히 한 model을 load하는지 검증하고, 성공 handle의 네 parallel vector가 항상 두 entry를 갖도록 보완했다.
- 유효한 out-parameter는 함수 진입 직후 null로 만들어 모든 실패 경로의 header 계약을 지킨다.
- public function signature, `CausalLmModel` layout, error enum은 바꾸지 않았다.
- Android NDK r27 arm64에서 experimental macro OFF/ON 두 구성을 `ENABLE_QNN_MODELS`와 함께 syntax compile했다. 기존 Bert/Gemma missing-override 경고 두 개 외 새 경고는 없다.
- 이 지점은 PR 11로 독립 제출 가능한 경계다: `32f6575d`.
- 자동 failure-injection seam은 없으며, 주 단일-model reload device test보다 우선하지 않는다.

### 2026-07-23 — Issue #49: QNN SDK header self-containment

- commit `473e8255 [qnn] include required QNN SDK interfaces`를 완료했다.
- `QNN_INTERFACE_VER_TYPE`을 사용하는 manager header가 `QnnInterface.h`를 직접 include하고, memory descriptor를 생성하는 `.cpp`가 `QnnMem.h`를 직접 include한다.
- 회귀는 `15873b03`에서 `Utils/DynamicLoadUtil.hpp`를 제거할 때 `QNN.hpp → QnnInterface.h → QnnMem.h` 전이 경로도 함께 사라져 발생했다.
- Meson source, `ENABLE_QNN`, generated `vendor/QNN` include path는 정상이라 build configuration은 바꾸지 않았다.
- 실제 과거 QNN API 2.36/QAIRT 2.47 호환 header와 분리 fake header의 clang++/g++ compile을 통과했다.
- runtime/ABI/성능 변화는 없으며, 사용자가 Linux에서 전체 `build_android.sh --enable-qnn --install --clean` 성공을 확인했다. 실제 device lifecycle 검증은 남아 있다.
- 최종 PR stack에서는 별도 기능 PR보다 PR 4 (`15873b03`)에 함께 넣거나 squash하는 것이 맞다.

### 2026-07-22 — 기존 reload workaround

- commit: `8cdc4dd3`
- 변경: `QNNGraph` destructor에서 context free 대신 process-wide tensor deregistration 수행
- 확인: `load_unload_after_fix.txt`의 두 번째 run 성공
- workaround 당시 잔여 위험: global deregistration, context 무제한 cache, 실패 전파 부재. 현재 HEAD에서는 앞뒤 두 항목을 해결했고 bounded cache는 남아 있다.
