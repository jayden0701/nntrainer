# PR별 GitHub 코드리뷰 댓글 초안

이 문서는 PR #4103부터 #4152까지의 상세 리뷰에서, **각 PR의 직접 변경분에
남길 만한 댓글 하나만** 골라 짧게 정리한 것이다.

- 아직 upstream PR에는 게시하지 않았다.
- 추천 위치는 각 문서에 적힌 review head 기준이므로, PR이 갱신되면 줄 번호가
  달라질 수 있다.
- P1/P2는 수정 요청 댓글이고, `승인`은 직접 변경분에서 blocker를 찾지 못한
  경우의 review body 초안이다.
- 자세한 근거가 필요하면 각 항목의 상세 리뷰 링크를 보면 된다.

## PR #4103 — P1

- 추천 위치: `nntrainer/opencl/meson.build:14`
- [상세 리뷰](./pr-4103.md)

> **[P1] Tizen devel 패키지에도 새 헤더를 등록해야 합니다.**
>
> Tizen GPU devel RPM을 만들 때 여기서 새로 설치하는 OpenCL 헤더들이
> `packaging/nntrainer.spec`의 `%files devel`에 없어, 빌드가
> `installed but unpackaged files`로 실패하거나 패키지에서 헤더가 빠집니다.
> `%{with gpu}` 파일 목록에도 같은 설치 경로로 추가해 주세요.

## PR #4105 — P1

- 추천 위치: `nntrainer/layers/fc_layer.cpp:263`
- [상세 리뷰](./pr-4105.md)

> **[P1] incremental FC에도 fused activation을 적용해야 합니다.**
>
> 일반 `forwarding()`은 bias 뒤에 activation을 적용하지만
> `incremental_forwarding()`은 matmul과 bias만 수행합니다. 따라서
> `fused_activation`으로 별도 ActivationLayer를 대신한 CausalLM decode는
> 활성화 전 값을 반환합니다. 두 forward 경로에 같은 epilogue를 적용하고
> 결과 동등성을 테스트해 주세요.

## PR #4106 — P1

- 추천 위치: `nntrainer/engine.h:196`
- [상세 리뷰](./pr-4106.md)

> **[P1] non-CPU backend의 factory 등록이 항상 실패합니다.**
>
> `gpu`나 `qnn` engine으로 이 API를 호출하면 concrete context에 override가
> 없어 기본 구현으로 내려가고 `-1`을 반환합니다. 현재 공개한 facade의 핵심
> 기능이 CPU 밖에서는 동작하지 않으므로, 각 context의 실제 registry로
> 위임하는 구현을 추가해 주세요.

## PR #4108 — P1

- 추천 위치: `nntrainer/context.h:185`
- [상세 리뷰](./pr-4108.md)

> **[P1] integer-dot 지원을 따로 확인한 뒤 DP4A를 선택해야 합니다.**
>
> DPAS가 없는 OpenCL 장치를 모두 DP4A로 분류하지만, `dpas == false`가
> integer-dot 지원을 뜻하지는 않습니다. 이 plan이 적용되면 extension이 없는
> 장치에서 kernel compile이 실패합니다. integer-dot capability를 별도로
> probe하고 미지원 장치는 일반 GEMM이나 CPU 경로로 보내 주세요.

## PR #4109 — P1

- 추천 위치: `nntrainer/cl_svm_allocator.cpp:74`
- [상세 리뷰](./pr-4109.md)

> **[P1] host fallback pointer를 SVM으로 표시하면 안 됩니다.**
>
> SVM 미지원이나 용량 부족으로 `clSVMAlloc()`이 실패하면 일반 host memory를
> 반환하지만, 상위 코드는 계속 이 포인터를 SVM으로 취급합니다. 이 값이
> `clSetKernelArgSVMPointer()` 등에 전달되면 `CL_INVALID_*` 또는 잘못된 출력이
> 발생합니다. allocation별 실제 메모리 종류를 기록해 staging으로 보내거나
> SVM 실패를 명시적으로 처리해 주세요.

## PR #4110 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/blas_kernel_interface.cpp:1052`
- [상세 리뷰](./pr-4110.md)

> **[P1] 주소만으로 activation cache hit를 판단하면 이전 token을 재사용합니다.**
>
> 같은 storage 주소·shape·dtype를 유지한 채 내용만 바뀌고 중간에 다른 cache
> miss가 없으면, 새 입력의 upload와 quantization을 건너뜁니다. 메모리 자리는
> 같아도 안의 tensor는 새 값일 수 있으므로 storage generation을 key에 넣거나
> cache 수명을 동일 fan-out 범위로 제한해 주세요.

## PR #4111 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/blas_kernels.cpp:1682`
- [상세 리뷰](./pr-4111.md)

> **[P1] `K=64`는 이 XMX 경로로 보내면 안 됩니다.**
>
> `M > 4`, `N % 64 == 0`, `K == 64`이면 gate를 통과하지만 weight surface
> 폭은 `K/2`, 즉 32바이트입니다. Intel 2D block I/O의 최소 64바이트 조건을
> 어겨 undefined behavior나 device fault가 날 수 있으므로 `K >= 128`로
> 제한하거나 surface를 padding해 주세요.

## PR #4112 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/geglu_cl_op.cpp:55`
- [상세 리뷰](./pr-4112.md)

> **[P1] whole-op GeGLU kernel을 사용 전에 등록해야 합니다.**
>
> 현재 tree에는 `registerGeGLUClKernels()` 호출처가 없어 첫 GeGLU 실행 때
> kernel vector가 비어 있습니다. 이 인덱싱은 fallback 전에 out-of-bounds
> 접근으로 crash할 수 있으므로, context 초기화에 등록을 연결하거나 첫 호출에서
> 안전하게 lazy registration해 주세요.

## PR #4113 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/blas_kernels.cpp:2235`
- [상세 리뷰](./pr-4113.md)

> **[P1] out-of-order queue에서는 upload event를 첫 GEMM에 연결해야 합니다.**
>
> `NNTR_GPU_SVM_POOL=0`이고 weight가 64MiB 미만이면 비동기 upload의 완료
> event가 consumer wait-list에 전달되지 않습니다. GEMM이 먼저 실행되면 덜
> 채워진 weight를 읽을 수 있으므로, backing별 event를 첫 dispatch에 넘기거나
> 그 경로에서는 upload 완료를 먼저 보장해 주세요.

## PR #4114 — P1

- 추천 위치: `nntrainer/tensor/tensor_pool.h:409`
- [상세 리뷰](./pr-4114.md)

> **[P1] graph output은 암묵적인 host consumer로 취급해야 합니다.**
>
> terminal FP16 GPU output은 downstream view가 없어 `all_consumers_gpu`의
> 초기값 `true`가 유지되고 `GPU_CLMEM`으로 분류됩니다. 반환 경로에는 device
> plane을 host로 내리는 과정이 없어 API caller가 stale/zero shadow를 받을 수
> 있습니다. graph output을 host consumer로 등록하거나 반환 전에 lower해 주세요.

## PR #4115 — P1

- 추천 위치: `nntrainer/layers/llm/swiglu_layer.cpp:75`
- [상세 리뷰](./pr-4115.md)

> **[P1] nonzero 위치에서 시작하는 multi-token prefill을 허용해야 합니다.**
>
> system-prompt cache나 이전 대화 뒤에 새 prompt를 붙이면
> `from > 0`, `to - from > 1`인 정상 prefill이 발생합니다. 현재 검사는 이를
> 예외로 처리하므로 전체 구간을 계산하고, 1-token 전용 최적화는 실제 decode에만
> 적용해 주세요.

## PR #4116 — P1

- 추천 위치: `Applications/CausalLM/llm_util.hpp:135`
- [상세 리뷰](./pr-4116.md)

> **[P1] 환경변수가 아니라 실제 graph engine을 기준으로 repack을 결정해야 합니다.**
>
> OpenCL 빌드에서 `NNTR_ENGINE`이 없으면 여기서는 `gpu`를 반환하지만, graph
> node에는 `engine=`이 없어 실제 실행은 CPU입니다. 이 상태에서 QS4CX repack을
> 건너뛰면 ARM CPU가 첫 FC에서 `getPackedData()` 예외를 냅니다. 실제 등록·실행
> context를 확인하거나 graph 전체에 선택한 engine을 적용해 주세요.

## PR #4117 — P1

- 추천 위치: `nntrainer/cuda_context.cpp:85`
- [상세 리뷰](./pr-4117.md)

> **[P1] CUDA graph의 tensor pool도 CUDA allocator에 연결해야 합니다.**
>
> 여기서 allocator를 등록해도 `NeuralNetwork::compile()`은 CUDA node를
> 확인하지 않아 graph에 계속 `engine_name="cpu"`를 넘깁니다. 그 결과
> `engine=cuda` 모델의 weight/activation pool이 일반 host memory로 만들어지므로,
> CUDA node가 있으면 graph allocator도 `cuda`로 선택해 주세요.

## PR #4118 — P1

- 추천 위치: `nntrainer/layers/llm/logit_softcapping.cpp:141`
- [상세 리뷰](./pr-4118.md)

> **[P1] softcap kernel 뒤의 host-read 경계에서 동기화해야 합니다.**
>
> `NNTR_CUDA_ASYNC=1`에서는 이 kernel을 enqueue한 뒤 기다리지 않고, 위의
> `finish()`는 kernel 호출 전에 실행됩니다. incremental inference가 logits를
> CPU에서 읽을 때 softcap이 아직 끝나지 않았을 수 있으므로 kernel 다음에
> terminal synchronization을 넣어 주세요.

## PR #4119 — P1

- 추천 위치: `nntrainer/cuda/cuda_compute_ops.cpp:84`
- [상세 리뷰](./pr-4119.md)

> **[P1] device-only tensor를 CPU fallback에서 직접 읽으면 안 됩니다.**
>
> `NNTR_CUDA_DEV_ACT=1`에서 scale 준비나 CUDA kernel이 실패하면 이
> `input.dot()`으로 내려오지만, input/output은 CPU가 접근할 수 없는
> `cudaMalloc` pointer일 수 있습니다. device-only tensor에서는 오류를
> 반환하거나 명시적인 D2H/H2D staging 뒤에만 CPU fallback을 실행해 주세요.

## PR #4120 — P1

- 추천 위치: `nntrainer/cuda/cuda_fc_qint4.cpp:393`
- [상세 리뷰](./pr-4120.md)

> **[P1] GEMV의 `K % 4` tail을 계산해야 합니다.**
>
> `M == 1`, `K % 4 != 0`인 QS4CX FC에서는 이 loop가 4개 묶음만 처리해
> 마지막 1~3개 곱을 버립니다. `wrowsum`은 K 전체를 포함하므로 함수는
> 성공하면서 결과만 조용히 틀립니다. scalar tail을 추가하거나 이런 K는 일반
> GEMM으로 보내 주세요.

## PR #4121 — P1

- 추천 위치: `nntrainer/cuda/cuda_compute_ops.cpp:111`
- [상세 리뷰](./pr-4121.md)

> **[P1] dtype 변환용 device copy 실패를 무시하면 안 됩니다.**
>
> `cuda::copy_any()`가 `false`를 반환해도 결과를 확인하지 않아, FP32→FP16
> 변환 중 device input read가 실패하면 0으로 초기화된 `xs`를 실제 입력처럼
> 변환해 반환할 수 있습니다. copy 실패를 호출자에게 전달하고 성공한 경우에만
> source pointer를 바꿔 주세요.

## PR #4122 — P1

- 추천 위치: `nntrainer/cuda/cuda_fc_qint4.cpp:1123`
- [상세 리뷰](./pr-4122.md)

> **[P1] CUDA 12.x와 13의 prefetch API를 구분해야 합니다.**
>
> 이 5개 인자 `cudaMemPrefetchAsync()`는 CUDA 13 형식이라 CUDA 12.x
> header로 `enable-cuda` 빌드를 하면 컴파일되지 않습니다.
> `CUDART_VERSION`으로 호출을 나누거나 configure 단계에서 CUDA 13 미만을
> 명확히 거부해 주세요.

## PR #4123 — P2

- 추천 위치: `nntrainer/cuda/cuda_fc_qint4.cpp:1145`
- [상세 리뷰](./pr-4123.md)

> **[P2] JIT int8 weight scratch에도 tail padding이 필요합니다.**
>
> `NNTR_CUDA_I8_JIT=1`일 때 이 buffer는 정확히 `K*N`바이트만 잡지만, 같은
> cuBLAS IMMA operand인 persistent weight에는 wide tail read를 위해
> `FC_I8_TAIL_PAD`를 붙입니다. 마지막 vector read가 allocation을 넘지 않도록
> JIT scratch에도 동일한 padding을 포함해 주세요.

## PR #4124 — P1

- 추천 위치: `nntrainer/layers/llm/tie_word_embedding.cpp:731`
- [상세 리뷰](./pr-4124.md)

> **[P1] CUDA 성공 뒤에도 공통 bias 후처리를 실행해야 합니다.**
>
> 여기서 함수 전체를 `return`하면 Q6_K lm-head의 CUDA matmul 뒤에 있는
> bias add가 생략됩니다. `batch > 1`이면 첫 batch 뒤에 나머지 계산도 끝납니다.
> matmul 완료 여부만 표시하고 각 batch의 공통 후처리까지 도달하도록 제어 흐름을
> 바꿔 주세요.

## PR #4125 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/attention_kernels.cpp:2115`
- [상세 리뷰](./pr-4125.md)

> **[P1] 정적 kernel cache가 clone ring을 우회합니다.**
>
> 이 attention 경로는 `registerClKernel()`의 반환값을 함수 정적 변수에
> 저장해, 첫 호출 뒤에는 ring에서 다음 clone을 빌리지 않습니다. 따라서 이 PR이
> 막으려는 kernel argument 덮어쓰기가 핵심 QK/softmax/SV 경로에 그대로 남습니다.
> 인자 설정과 enqueue마다 ring-aware lease를 얻도록 바꿔 주세요.

## PR #4126 — P1

- 추천 위치: `nntrainer/cuda_context.cpp:332`
- [상세 리뷰](./pr-4126.md)

> **[P1] position buffer를 읽는 consumer 없이 graph를 replay하면 안 됩니다.**
>
> 두 번째 token부터 바뀌는 값은 `cuda_set_pos()`가 쓰는 두 정수뿐인데, 현재
> tree에는 `cuda_pos_buffer()`를 kernel 인자로 사용하는 곳이 없습니다. replay는
> 첫 capture의 position과 KV 길이를 그대로 써서 다음 token부터 잘못 계산합니다.
> consumer를 연결하거나 그 전에는 M2-B를 eager fallback 처리해 주세요.

## PR #4127 — P1

- 추천 위치: `nntrainer/cuda_context.cpp:124`
- [상세 리뷰](./pr-4127.md)

> **[P1] 준비되지 않은 M2-B를 discrete GPU 기본값으로 켜면 안 됩니다.**
>
> `NNTR_CUDA_M2B`는 이미 읽히므로 이 기본값은 `NNTR_ENGINE=cuda`만으로 graph
> replay를 활성화합니다. position buffer consumer가 없어 두 번째 token부터 첫
> capture의 위치와 KV 길이를 재사용하므로, 필요한 consumer가 모두 등록된
> 경우에만 켜고 그렇지 않으면 eager 경로를 유지해 주세요.

## PR #4128 — P1

- 추천 위치: `nntrainer/tensor/cpu_backend/arm/arm_compute_backend_fp16.cpp:455`
- [상세 리뷰](./pr-4128.md)

> **[P1] packed RHS와 kernel의 `nr` layout을 맞춰야 합니다.**
>
> ARM i8mm의 FP16×QS4CX 경로는 `QS4CX_Tensor::pack()`이 만든 `nr=8` buffer를
> variant 확인 없이 `nr=4` kernel에 넘깁니다. 두 layout의 weight와
> scale/bias 위치가 달라 결과 전체가 오염되므로, 같은 layout의 kernel을
> 사용하고 불일치하면 명시적으로 거부해 주세요.

## PR #4129 — P1

- 추천 위치: `api/ccapi/include/half_fp16.h:152`
- [상세 리뷰](./pr-4129.md)

> **[P1] `numeric_limits<Half>`를 binary16 값으로 특수화해야 합니다.**
>
> wrapper FP16 빌드에서 `lowest()`가 `Half{}`, 즉 0을 반환합니다. 그래서
> max/global-max pooling 입력이 모두 음수이면 입력에 없는 0이 최댓값이 되고,
> backward용 위치도 찾지 못할 수 있습니다. IEEE binary16 값으로 정의하고
> 음수만 있는 pooling 회귀 테스트를 추가해 주세요.

## PR #4130 — P1

- 추천 위치: `nntrainer/utils/thread_manager.cpp:45`
- [상세 리뷰](./pr-4130.md)

> **[P1] Windows 기본 static 구성에서는 singleton이 DLL마다 복제됩니다.**
>
> 여러 layer DLL이 정적 nntrainer를 각각 연결하면 이 함수와
> `static instance`도 DLL마다 한 벌씩 생깁니다. 따라서 함수 정의를 cpp로
> 옮겨도 DLL별 thread pool이 남으므로, 모든 DLL이 공유하는 DLL 한 곳에서
> instance를 소유하도록 연결해 주세요.

## PR #4131 — P1

- 추천 위치: `.github/workflows/msvc_gpu_verify.yml:18`
- [상세 리뷰](./pr-4131.md)

> **[P1] 이 Windows GPU 검사를 일반 PR에서도 실행해야 합니다.**
>
> workflow가 `ci/msvc-*` branch의 push와 `workflow_dispatch`만 받아 현재
> `upstream-pr/trackW4-msvc-gpu-ci` head에서도 실행되지 않고,
> `pull_request` 이벤트도 없습니다. 새 검사가 병합 전 변경을 실제로 검증하도록
> `main` 대상 `pull_request` trigger를 추가하고 현재 PR에서 성공을 확인해 주세요.

## PR #4141 — P2

- 추천 위치: `nntrainer/tensor/manager.cpp:930`
- [상세 리뷰](./pr-4141.md)

> **[P2] V2가 감싸는 형태의 gradient 생존 구간도 겹침으로 봐야 합니다.**
>
> `NNTR_MEM_PLANNER=v2` 학습에서 한 gradient의 생존 구간이 다른 구간을
> 감싸면 현재 검사가 충돌을 놓쳐 같은 offset을 줄 수 있습니다. 한 gradient가
> 다른 값을 조용히 덮어쓰므로, 표준식
> `a.start < b.end && b.start < a.end`로 고치기 전에는 학습 pool에서 V2를
> 거부하거나 V1으로 되돌려 주세요.

## PR #4142 — P2

- 추천 위치: `Applications/CausalLM/models/causal_lm.cpp:673`
- [상세 리뷰](./pr-4142.md)

> **[P2] LFM2 embedding 실행 경로에도 같은 메모리 지표를 출력해야 합니다.**
>
> 이번 변경은 `CausalLM::run()`에만 `peak commit`을 추가했지만
> `USE_EMBEDDING=true`인 LFM2는 별도 `run_with_embeddings()`를 사용해 여전히
> 예전 `peak memory`만 출력합니다. 측정과 출력 코드를 공통 helper로 빼 두
> 실행 경로가 같은 정보를 보여주게 해 주세요.

## PR #4143 — P1

- 추천 위치: `nntrainer/layers/cuda_layers/cuda_rmsnorm_layer.cpp:38`
- [상세 리뷰](./pr-4143.md)

> **[P1] 모델 파일의 FP32 gamma를 FP16 weight로 요청하면 안 됩니다.**
>
> activation이 FP16이면 `getWeightDataType()`도 FP16이 되지만 RMSNorm gamma는
> 모델 파일에 FP32로 저장됩니다. 값을 절반 크기로 읽으면 gamma와 다음 weight의
> 파일 위치까지 어긋납니다. gamma를 FP32로 요청하고 mixed-type kernel이나
> 올바르게 변환한 GPU 사본을 사용해 주세요.

## PR #4144 — P1

- 추천 위치: `Applications/CausalLM/models/transformer.cpp:190`
- [상세 리뷰](./pr-4144.md)

> **[P1] runtime 길이는 실제 RoPE 표 길이로 제한해야 합니다.**
>
> Gemma3/Llama처럼 MHA에 `max_position_embeddings`를 넘기지 않는 경로는 실제
> RoPE 표가 기본 40,960칸뿐인데, 여기서는 설정 파일의 더 큰 값만 확인합니다.
> 설정이 131,072이고 100,000을 요청하면 clamp되지 않아 표 밖을 읽으므로,
> model cap을 MHA에 전달하거나 실제 생성된 표 길이로 제한해 주세요.

## PR #4145 — 승인

- 추천 위치: 전체 변경에 대한 review body
- [상세 리뷰](./pr-4145.md)

> **승인 의견**
>
> 같은 offset을 쓰는 token 중 하나라도 `GPU_CLMEM`이면 `cl_mem`을 유지하고,
> 모두 SVM일 때만 생략하는 것을 확인했습니다. allocation 전후의 분류 입력도
> 같아 이번 PR의 직접 변경분에서는 correctness blocker를 찾지 못했습니다.

## PR #4148 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/attention_kernels.cpp:2704`
- [상세 리뷰](./pr-4148.md)

> **[P1] DPAS 확장 존재만으로 SG16 kernel을 선택하면 안 됩니다.**
>
> `caps().dpas`는 확장 존재만 알려 주며, 장치가 요구하는 subgroup 크기가
> 16이라는 뜻은 아닙니다. 최소 크기가 8인 DPAS 장치에서 SG16 경로가 기본으로
> 켜지면 kernel 실패나 정의되지 않은 계산이 발생할 수 있으므로, subgroup
> 요구값을 조회해 16일 때만 켜고 나머지는 block-Q로 되돌려 주세요.

## PR #4149 — P1

- 추천 위치: `nntrainer/tensor/cl_operations/blas_kernels.cpp:1183`
- [상세 리뷰](./pr-4149.md)

> **[P1] `cl_mem` view의 시작 offset을 kernel에 전달해야 합니다.**
>
> 시작 위치가 0이 아닌 Tensor view도 `getClMem()`에서는 전체 buffer handle만
> 전달되어 이 index가 항상 0번부터 읽고 씁니다. `batch > 1`이면 두 번째
> batch도 batch 0 자리를 덮으므로, input/output element offset을 더하거나
> nonzero-offset view는 안전한 경로로 보내 주세요.

## PR #4150 — P1

- 추천 위치: `Applications/CausalLM/jni/Android.mk:87`
- [상세 리뷰](./pr-4150.md)

> **[P1] Android CausalLM 빌드에는 OpenCL GPU 구현이 포함되지 않습니다.**
>
> 새 GPU 소스를 Android.mk에 넣었지만 prebuilt nntrainer가 소비자 compile에
> `ENABLE_OPENCL`을 전달하지 않아 두 파일의 OpenCL 코드가 전처리에서 빠집니다.
> 이 macro를 export하고, OpenCL 경로가 include하는 `blas_kernels.h`도 prebuilt
> 설치 header로 노출해야 Android에서 실제 GPU layer를 사용할 수 있습니다.

## PR #4151 — 승인

- 추천 위치: 전체 변경에 대한 review body
- [상세 리뷰](./pr-4151.md)

> **승인 의견**
>
> 허용되는 NSG/TM 조합에서 각 행이 정확히 한 subgroup에 배정되고, 부분합을
> 기존과 같은 순서로 더하는 것을 확인했습니다. `psum`과 `ssum` 사이 barrier와
> tail 조건도 맞아 XRED 직접 변경분에서는 correctness blocker를 찾지 못했습니다.

## PR #4152 — P2

- 추천 위치: `nntrainer/cuda/cuda_fc_qint4.cpp:473`
- [상세 리뷰](./pr-4152.md)

> **[P2] fused GEMV의 `K % 4` tail을 처리해야 합니다.**
>
> `NNTR_CUDA_FC_FUSED_DECQ=1`, `M == 1`, `K % 4 != 0`인 FP16 QS4CX FC에서는
> 이 loop가 마지막 1~3개 곱을 버립니다. `wrowsum`은 그 weight까지 포함하므로
> 함수는 성공하면서 출력만 조용히 틀립니다. scalar tail을 추가하거나 fused
> 진입을 `K % 4 == 0`으로 제한해 주세요.
