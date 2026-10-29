# GPU KV 확장 분석 — PC/WSL 우선 (2026-10-29)

> 구현 전 계획·분석 기록이다. 후속 실제 결과는 [CUDA 실행 기록](gpu-cuda-execution.md)과 [최신 인계](gpu-handoff-2026-10-29.md)를 참고한다.

이 문서는 PR #4460의 CPU 단계 이후 GPU 확장을 위한 **소스 분석**이다.
제품 코드, probe, 모델 실행은 이번 세션에 구현하지 않았다.
실행 순서와 완료 기준은 [구현 계획](gpu-extension-plan.md),
다음 세션 시작점은 [GPU 인계](gpu-handoff-2026-10-29.md)를 따른다.

## 1. 기준과 이번 세션의 결정

- 저장소: WSL Ubuntu `/home/jayden/nntrainer`.
- PR: [nntrainer/nntrainer#4460](https://github.com/nntrainer/nntrainer/pull/4460),
  draft/open, head `25230aafe1a230e8dc497921d1149d52f6eabdad`.
- PR base: `141871a58ff439e3caeac40c1632ff8ac3543d20`.
- 로컬 브랜치: `kv-manager-gpu-plan-pr4460`.
  upstream의 `refs/pull/4460/head`를 fetch한 뒤 해당 SHA로 switch했다.
- 읽은 인계: [CPU handoff](handoff-2026-10-29.md),
  [리뷰 후속](review-followup.md), [다음 backend 조사](next-backend-review.md),
  [CPU 재현 절차](../docs/kv-cache-cpu-validation.md).
- 사용자 추가 조건: **연결된 Android 기기가 없으므로 현재 PC에서 GPU 검증을 먼저 수행**.
- 권고 순서: **공통 allocation/수명 계약 → WSL CUDA → OpenCL → Android 실기기**.
  CUDA를 먼저 구현·검증하는 순서는 이번 환경에 대한 설계 결정이다.
  OpenCL/SVM의 기존 조사 결과는 후속 adapter의 입력으로 보존한다.

기존 문서의 Adreno 750/SVM 0xB, Android NDK, QNN SDK, 모델·로그 및
`/tmp/` artifact 경로는 이전 개발 환경의 기록이다.
현재 WSL에서 사용할 수 있는 파일이나 연결된 기기라고 해석하지 않는다.
CPU 성능·정확성 통과 기록도 이번 checkout에서 재실행한 결과가 아니다.

## 2. 현재 컴퓨터에서 확인한 범위

| 항목 | 이번 세션의 조회 결과 | 의미 |
|---|---|---|
| Windows GPU | NVIDIA GeForce RTX 5070, Intel(R) Graphics | CUDA 첫 대상은 RTX 5070 |
| NVIDIA 조회 | driver 610.88, memory.total 12227 MiB, compute capability 12.0 | Windows와 WSL에서 같은 값 |
| WSL | Ubuntu 26.04.1 LTS, kernel 6.18.54.1-microsoft-standard-WSL2 | 개발·빌드 위치 |
| CUDA driver 노출 | `/usr/lib/wsl/lib/libcuda.so.1`, `nvidia-smi`, `/dev/dxg` 존재 | CUDA 실행 준비의 일부만 확인 |
| CUDA 개발 구성 | `nvcc` 미발견, `/usr/local/cuda*` 미발견; ldconfig에서 cudart/NVRTC/cuBLAS 미발견 | Toolkit 준비 필요 |
| WSL OpenCL | clinfo, system OpenCL headers/ICD 경로 및 libOpenCL 미확인 | GPU device/platform은 아직 조회하지 않음 |
| 기타 | git/python3/meson/ninja/gcc/g++/cmake/pkg-config 있음; clang-format-14/rustc/cargo/adb 미발견 | CPU 도구 일부는 기존 설치 상태 |
| 모델 | 사용자 조건상 실험 모델 미준비 | synthetic 테스트부터 설계 |
| 기존 build | 이전 gemma4-ple-fsu 소스의 CPU build 기록 존재 | PR #4460용 새 build 디렉터리 필요 |

GPU 조회는 `nvidia-smi`와 파일·도구 존재 확인이다.
CUDA allocation, kernel, NVRTC compile, bandwidth, managed coherence는 **미검증**이다.
Windows의 Intel GPU 존재로 WSL OpenCL 지원을 추론하지 않는다.
`nvidia-smi`의 드라이버 정보로 설치된 Toolkit 버전을 대신하지 않는다.

NVIDIA는 WSL에 Windows 드라이버가 노출되며 개발에는 Linux Toolkit이 필요하다고
설명한다. Toolkit 준비 시 WSL용 driver-free 설치 경로를 사용한다.
또한 WSL의 managed memory 및 pinned memory 제약을 명시하므로,
실제 device attribute와 handoff 실험으로 profile을 결정해야 한다.
[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 3. CPU core에서 그대로 재사용할 부분과 확장할 부분

| 소스 | 현재 계약/근거 | GPU 확장 영향 |
|---|---|---|
| `nntrainer/kv_cache/kv_cache_spec.h` | token-major, encoding/backing 분리, checked sizing, source alias | 최초 GPU도 같은 geometry 유지 |
| `kv_storage.cpp:19` | host-addressable이며 device-visible/SVM/register가 아닌 allocator만 허용; host memset | device allocation/초기화/실제 결과 계약을 먼저 추가 |
| `kv_storage.h` | aliasing shared_ptr로 storage 수명 유지 | Tensor보다 native backing/context가 먼저 해제되지 않도록 확장 |
| `kv_cache_manager.cpp:85` | storage 생성과 alias 연결, position/generation/단일 step | 정책은 재사용하되 GPU 완료 전에 publish 금지 |
| `kv_cache_manager.cpp:194` | markReady는 bool, commit은 ready만 검사 | enqueue 성공과 실행 완료를 구별할 장치 필요 |
| `kv_cache_manager.cpp:257` | thread_local 동기식 KvStepScope, batch 1/linear 제한 | 최초 CUDA eager graph 범위에서 재사용 가능; async escape는 금지 |
| `Applications/CausalLM/kv_cache_manager.cpp:77` | 새 core는 CPU에서만 선택, GPU는 기존 pool | core 선택과 backend 선택을 분리 |
| `models/causal_lm.cpp:630` | usesCpuStorage일 때만 begin/bind/commit | CUDA도 검증된 step 경로로 연결 |
| `layers/mha_core.cpp:2055,3241` | 실제 write slice 검사 후 producerReady | GPU producer와 consumer의 포인터·범위·완료 검증을 연결 |

위 위치는 기준 SHA의 탐색 시작점이며 변경 후 줄 번호는 이동할 수 있다.
소스 링크: [core](../nntrainer/kv_cache/kv_cache_manager.cpp),
[storage](../nntrainer/kv_cache/kv_storage.cpp),
[facade](../Applications/CausalLM/kv_cache_manager.cpp),
[wrapper](../Applications/CausalLM/models/causal_lm.cpp),
[MHA](../Applications/CausalLM/layers/mha_core.cpp).

## 4. 기존 CUDA 경로의 실제 연결점

### 4.1 Allocation은 이미 여러 결과를 낸다

[CudaMemAllocator](../nntrainer/cuda_mem_allocator.cpp)는 요청에 따라
`cudaMalloc`, host-mapped pinned allocation, managed allocation,
일반 host fallback을 사용한다. integrated GPU에 대한 별도 처리도 있다.
`cuda-uvm`이라는 이름만으로 managed allocation이라고 판단할 수 없다.

allocator의 `isDeviceVisible()` 및 `isHostAddressable()`는 allocator 단위 계약이다.
실패 후 host fallback의 실제 상태를 표현하기에는 충분하지 않다.
[MemoryPool::getMemory](../nntrainer/tensor/memory_pool.cpp)는 allocator의 속성으로
MemoryData를 표시한다. 새 strict KV 경로에는 **allocation별 결과**가 필요하다.

CUDA allocator는 `isSVM()=false`를 유지한다.
이 flag를 true로 바꾸면 OpenCL kernel binding 경로가 CUDA pointer를 받을 수 있다.
실제 allocation kind를 추가해도 이 구분을 보존해야 한다.
기존 installed `ResidencyClass` 값을 재번호화하지 않는다.

### 4.2 직접 읽기의 기반은 있다

기존 facade의 `NNTR_CUDA_KV_DEV=1`은 device allocator,
`NNTR_CUDA_KV_UVM`은 CUDA 공용 allocator를 선택할 수 있다.
초기화·snapshot에는 device memset/copy 경로도 있다.

[cuda_attention.cpp:202](../nntrainer/cuda/cuda_attention.cpp)의 `mirror_kv`는
device-accessible pointer이면 그대로 반환한다. 일반 host pointer이면 별도
device mirror를 만들고 history를 H2D copy한다.
따라서 manager 소유 device KV를 실제 attention에 전달하면 직접 읽기의 기반이 있다.
이때 mirror 생성/copy가 0임을 실행에서 확인해야 하며 flag 설정만으로 통과시키지 않는다.

### 4.3 K 직접 쓰기와 V 직접 쓰기의 난이도가 다르다

MHA의 K 경로는 `cuda_rope_fp16` 또는 slot variant가 cache write slice에
RoPE 결과를 쓸 수 있다. 이는 필요한 연산을 최종 저장소에 쓰는 직접 쓰기 후보다.

반면 V는 `cuda_scalar_mul_fp16(..., 1.0f)` 또는 slot variant로
`value_step`에서 cache로 옮긴다(`mha_core.cpp:3150` 부근).
GPU에서 실행해도 **순수 append copy**다. 기존 GPU 경로 전체를 A+B 완료로 볼 수 없다.

[Qwen3 graph](../Applications/CausalLM/models/qwen3/qwen3_causallm.cpp)에서는
V의 최종 producer가 `wv` FC다.
V 직접 쓰기를 위해서는 FC의 실제 출력 destination을 매 step의 cache slice로 지정해야 한다.
K는 q/k norm과 RoPE의 순서를 유지하며 마지막 RoPE destination을 지정한다.

### 4.4 Graph output을 setData 한 번으로 바꾸면 부족하다

[FC incremental](../nntrainer/layers/fc_layer_cl.cpp#L169)은 from을 0으로 정규화하고
현재 activation의 0번 row부터 출력한다.
[TensorPool::fillPlaceholder](../nntrainer/tensor/tensor_pool.cpp#L495)는
UNMANAGED source만 허용하고 dependent를 동기화한다.
현재 cache input placeholder 연결은 일반 FC output을 외부 write slice로 바꾸는 API가 아니다.

새 binding은 producer output, 모든 dependent, memory planner의 재사용 계획,
backend ops, active row 수 및 tensor offset을 함께 다뤄야 한다.
Qwen3의 bias 없는 wv/FP16/단일 consumer profile부터 제한하는 것이 적합하다.
Gemma의 v_norm 또는 다른 epilogue가 있으면 그 연산이 최종 producer다.
이름 문자열을 core에서 파싱해서 임의의 FC output을 교체하지 않는다.

### 4.5 CUDA graph replay는 별도 단계다

[StreamManager](../nntrainer/cuda/cuda_stream_manager.h)는 stream과 capture/replay,
device position buffer를 제공한다.
capture 중에는 host 동기화가 억제되고, replay는 일반 MHA host 코드를 다시 호출하지 않는다.

따라서 MHA 안의 producerReady 호출만 붙이면 replay마다 readiness가 갱신되지 않는다.
동적 slice 포인터를 capture하면 다음 token에서 이전 slot을 덮어쓸 수도 있다.
**첫 core 기반 KV adapter는 eager 실행으로 제한**하고,
capture/replay는 storage generation·base pointer·position·completion을 포함한 별도 gate로 연다.
기존 unmanaged CUDA graph 경로의 회귀도 확인해야 한다.

## 5. 권고 memory profile과 완료 의미

| profile | 우선 목적 | 필수 조건/한계 |
|---|---|---|
| CUDA device KV | RTX 5070에서 A+B, VRAM에서 history 읽기 | host dereference 금지; CPU 접근은 명시적 staging |
| CUDA host-mapped KV | 같은 backing의 CPU↔GPU 순차 handoff | host/device 주소·매핑 확인, 제한된 pinned 용량, PCIe 비용 측정 |
| CUDA managed KV | 지원 속성에 따른 추가 비교 | cMA/host access 규칙·coherence·migration 실험 후 지원 판정 |
| OpenCL SVM buffer | 후속 공통 adapter | 실제 CL device/context, allocation flags, map/unmap 필요 |
| OpenCL OHWI/image | 후속 최적화 | K/V layout과 authoritative backing 재설계, mirror 제거 필요 |

A는 manager 소유 history를 attention이 직접 읽는 것,
B는 최종 K/V producer가 write view에 결과를 쓰는 것,
C는 CPU/GPU가 호환 layout/encoding의 같은 backing을 순차 사용함을 뜻한다.
device KV의 snapshot D2H/H2D는 명시적 I/O다.
이를 C나 zero-copy 전환으로 집계하지 않는다.
host-mapped direct read는 PCIe 읽기 비용이 있으며, 같은 주소가 physical migration 0을 증명하지 않는다.

## 6. OpenCL 후속에 반드시 남길 발견

- `ContextManager::createSVMRegion`은 `CL_MEM_READ_WRITE`만 전달한다.
- `coarseGrainSVMOnly()`는 allocation flags 대신 device의 fine-grain 지원 bit를 본다.
- `ClSVMAllocator`는 실패 시 host fallback을 반환하지만 allocator 속성은 device-visible이다.
- 현재 MHA의 `svm_ok`는 MemoryData flag를 신뢰한다.
- V GPU copy, OHWI mirror/scatter/gather와 non-SVM upload 경로를 copy 종류별로 분리해야 한다.
- context/queue identity, alignment, map 상태는 backing별로 관리해야 한다.
  allocation 직후 host memset을 포함한 host 접근도 coarse-grain 규칙의 대상이다.

Khronos는 fine-grain allocation flag와 동기화 지점의 역할을 명시한다.
device capability만으로 실제 allocation의 coherence 정책을 결정하지 않는다.
[clSVMAlloc](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clSVMAlloc.html),
[clEnqueueSVMMap](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clEnqueueSVMMap.html)

## 7. 위험 우선순위와 판정

| 우선순위 | 위험 | 해소 조건 |
|---|---|---|
| P0 | host fallback을 device KV로 오표시 | allocation별 kind, strict reject, 실패 주입 |
| P0 | device pointer에 memset/Tensor host op/snapshot 직접 접근 | backend 초기화·접근 guard·명시적 I/O |
| P0 | enqueue 후 조기 commit/free/reset | checked completion, owner/context 유지, 실패 시 drain |
| P0 | V destination 교체 후 dependent/planner가 옛 buffer 사용 | 실제 FC→MHA graph 검증, 주소·범위·guard 검사 |
| P0 | replay가 step 검사·producerReady 우회 | 최초 eager 제한, 후속 replay 계약 |
| P1 | GPU kernel 실패 후 CPU fallback·mirror 사용을 통과로 기록 | strict admission와 실제 dispatch/copy 계측 |
| P1 | FP16 GPU model test가 생성되지 않음 | FP16 전용 synthetic 실제 wrapper target 추가 |
| P1 | ABI가 다른 기존 build/library 혼용 | 새 build, source/runtime hash와 header 일치 |
| P1 | GPU 시계·열·JIT 비용이 성능 비교를 오염 | 같은 PC/profile/커널, load·JIT·steady state 분리 |

단일 step의 position rollback은 전체 모델 상태 rollback이 아니다.
append가 실패하면 새 범위를 publish하지 않고 queued work를 정리한다.
context/device 오류나 prefix 손상 가능성이 있으면 session을 invalid로 만들고 복구를 요구한다.
실패 후 raw Tensor pointer가 안전해진다고 주장하지 않는다.

## 8. 외부 확인 자료

- [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html):
  드라이버 노출, Toolkit 설치 방식, managed/pinned 제약.
- [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html):
  runtime compile와 지원 architecture. 설치할 NVRTC가 sm_120을 지원하는지 확인한다.
- [NVIDIA Linux installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html):
  설치 시 OS/compiler/Toolkit 조합을 확인한다. 현재 문서의 지원 목록만으로
  WSL의 특정 조합이 실행 검증됐다고 판단하지 않는다.
- [Khronos SVM pointer binding](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clSetKernelArgSVMPointer.html):
  offset pointer와 argument type alignment 계약.

이번 분석의 구현 제안은 위 계약과 기준 SHA의 소스를 바탕으로 한 설계 판단이다.
새 adapter의 정확성·성능 및 실기기 지원은 다음 세션 이후 gate를 통과해야 확정된다.
