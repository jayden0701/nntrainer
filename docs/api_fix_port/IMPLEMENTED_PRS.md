# `api_fix` 이식 작업: 완료 PR 요약

## 문서의 범위

이 문서에서 PR1~PR17은 GitHub pull request 번호가 아니라, 검토와 병합을
독립적으로 할 수 있도록 나눈 **stacked change unit**을 뜻한다. 현재
GitHub pull request 자체는 생성하지 않았다.

사용자가 제시한 개념적 기준은
`agent/pr4076-after-4053-compat-linear`의 `019abe01`이었지만, 실제 Git
조사에서 두 브랜치의 literal merge base는 `1a80cfa3`으로 확인됐다.
`api_fix`의 마지막 pre-fix commit `43c2ad8f`는 `019abe01`에 대응하는
rebased commit이지만 두 tree는 같지 않았다. 따라서 실제 이식 범위
`43c2ad8f..6b33c8c2`를 raw rebase/cherry-pick하지 않고, 발전한 대상
브랜치의 시작점
`0e5a913c1978c0b72b78598ec1e0464d1513d6af`에서 각 변경을 다시
검토·재설계해 이식했다.

전체 작업에 공통으로 적용한 원칙은 다음과 같다.

- 기존 대상 브랜치의 더 최신 동작과 충돌하면 최신 동작을 보존한다.
- 과거 패치의 ABI, 소유권, 예외, 동시성 가정은 그대로 신뢰하지 않는다.
- C/C++ 변경에는 clang-format 14를 적용한다.
- DCO sign-off와 `Co-authored-by: Codex <noreply@openai.com>`를 남긴다.
- `subprojects/`는 수정하지 않는다.
- 이 환경에서는 configure, build, test, Gradle, Android/device runtime,
  QNN generator를 실행하지 않는다. 문서의 “검증”은 정적 검토만 뜻한다.

## 현재 상태

| 단위 | 브랜치 | tip commit | 핵심 내용 | 원격 상태 |
| --- | --- | --- | --- | --- |
| [PR1](prs/PR01.md) | `agent/api-fix-port-vjepa` | `1ad40bfd` | V-JEPA2 QNN embedding output buffer 안전화 | push 완료 |
| [PR2](prs/PR02.md) | `agent/api-fix-port-native` | `c2f0f55f` | request-scoped generation과 additive C API | push 완료 |
| [PR3](prs/PR03.md) | `agent/api-fix-port-android` | `814c6bbc` | Android/JNI/LiteRT를 통합 API로 이행 | push 완료 |
| [PR4](prs/PR04.md) | `agent/api-fix-port-multimodal` | `db253cd8` | versioned C/POD multimodal extension dispatch | push 완료 |
| [PR5](prs/PR05.md) | `agent/api-fix-port-cleanup` | `706101be` | legacy API 제거와 통합 API 문서화 | push 완료 |
| [PR6](prs/PR06.md) | `agent/api-fix-postreview-build` | `6718c1c0` | cross-platform source/link/install 정합성 | push 완료 |
| [PR7](prs/PR07.md) | `agent/api-fix-postreview-loader-contract` | `72ed9f84` | native loader/backend 계약 명확화 | push 완료 |
| [PR8](prs/PR08.md) | `agent/api-fix-postreview-qnn-lazy-init` | `373af576` | allocator publication 동기화와 QNN lazy init | push 완료 |
| [PR9](prs/PR09.md) | `agent/api-fix-postreview-qnn-context-load` | `8dce69fb` | QNN binary-context transaction과 오류 전파 | push 완료 |
| [PR10](prs/PR10.md) | `agent/api-fix-postreview-qnn-forwarding` | `80a5d253` | graph forwarding descriptor/buffer 예외 안전성 | push 완료 |
| [PR11](prs/PR11.md) | `agent/api-fix-postreview-qnn-rpc` | `4f9d9e13` | RPC allocation/registration ledger 안전화 | push 완료 |
| [PR12](prs/PR12.md) | `agent/api-fix-postreview-qnn-execution-lease` | `e3d64752` | 실행과 teardown 사이 lifetime lease | push 완료 |
| [PR13](prs/PR13.md) | `agent/api-fix-postreview-qnn-teardown` | `d5c2a9e1` | QNN dependency-ordered runtime teardown | push 완료 |
| [PR14](prs/PR14.md) | `agent/api-fix-postreview-qnn-rpc-quarantine` | `1cff60d3` | 모호한 RPC 상태의 sticky quarantine | push 완료 |
| [PR15](prs/PR15.md) | `agent/api-fix-postreview-qnn-extension-dso` | `13688127` | generated BackendExtensions DSO 소유권 | push 완료 |
| [PR16](prs/PR16.md) | `agent/api-fix-postreview-qnn-resource-manager` | `8612527d` | generated ResourceManager/system DSO teardown | 로컬만 존재 |
| [PR17](prs/PR17.md) | `agent/api-fix-postreview-qnn-native-plugin` | `05352567` | native Linux `libqnn_context.so` target 복구 | 로컬만 존재 |

PR1~PR15는 `origin`에 push한 뒤 각 remote tip이 위 commit과 일치하는지
다시 확인했다. PR16과 PR17은 구현·커밋·정적 검토까지 끝났지만 아직
원격에 push하지 않았다.

## 스택을 나눈 기준

### PR1~PR5: 사용자가 보는 API 이식

- PR1은 독립적인 V-JEPA2 memory-safety 수정만 분리했다.
- PR2는 parser, grammar, cancellation, sampling, native C API를 한
  request-scoped runtime 계약으로 묶었다.
- PR3은 그 additive API를 Android/JNI/LiteRT에 채택했다.
- PR4는 과거의 C++ DSO callback ABI를 거부하고 versioned C/POD
  extension ABI로 대체했다.
- PR5는 모든 소비자가 새 경로로 이동한 뒤에만 legacy API와 죽은
  process-global 상태를 제거하고 문서를 갱신했다.

### PR6~PR7: 이식 후 드러난 portability/계약 문제

- PR6은 Windows source manifest, Android `liblog`, 설치 DSO 위치를
  실제 build graph에 맞췄다.
- PR7은 backend descriptor를 실제 구현과 일치시키고 load 인자를
  runtime switch가 아닌 compatibility assertion으로 정의했다.

### PR8~PR17: QNN lifecycle의 단계적 안전화

QNN은 vendor handle, DSO, rpcmem backing, graph execution이 서로 다른
수명을 갖는다. 한 PR에서 모두 바꾸면 실패 상태를 검토하기 어렵기 때문에
다음 순서로 분리했다.

1. first-use initialization과 late allocator publication(PR8)
2. binary context의 create/free transaction(PR9)
3. per-forward descriptor와 현재 buffer binding(PR10)
4. RPC allocation/registration ledger(PR11)
5. forwarding 대 teardown lifetime lease(PR12)
6. dependency 역순 teardown(PR13)
7. ambiguous RPC result quarantine(PR14)
8. generated extension DSO 소유권(PR15)
9. generated system/DLC DSO 소유권(PR16)
10. 실제 native Linux plugin build target 복구(PR17)

## 전체 정적 검증과 한계

각 PR에서 작업 기록의 범위에 맞게 다음을 수행했다.

- 변경 C/C++에 clang-format 14 적용. strict dry-run 수행 여부는 각 PR
  문서의 개별 기록을 따른다.
- `git diff --check`
- 앞 PR tip의 ancestry 확인
- DCO/Codex trailer 확인
- symbol/caller/source-manifest/삭제 참조 검색
- 독립 subagent를 이용한 ABI, 예외, 동시성, 소유권 재검토
- `subprojects/` 무변경 확인

그러나 다음은 **수행하지 않았다**.

- Meson configure와 native compile/link
- unit/integration test 실행
- Gradle, Android NDK build, APK/AAR 실행
- QAIRT generator 실행
- QNN vendor runtime 및 Snapdragon device fault injection

따라서 “완료”는 코드와 정적 검토가 끝났다는 뜻이며, merge 가능성을
확정하는 실제 build/runtime 검증은 사용자 환경에서 필요하다. 구체적인
후속 작업과 검증 순서는 [향후 PR 계획](FUTURE_ROADMAP.md)에 정리했다.
