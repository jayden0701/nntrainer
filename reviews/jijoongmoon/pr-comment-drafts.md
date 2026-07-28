# Draft GitHub Review Comments by PR

This document distills the detailed reviews for PRs #4103 through #4152 into
**one comment on the direct changes in each PR**.

- None of these comments have been posted to the upstream PRs.
- Suggested locations refer to the reviewed head. Line numbers may move if a PR
  is updated.
- P1 and P2 entries are change-request comments. `Approval` entries are draft
  review bodies for PRs where no blocker was found in the direct changes.
- Follow the detailed-review link under an entry for the full analysis.

## PR #4103 — P1

- Suggested location: `nntrainer/opencl/meson.build:14`
- [Detailed review](./pr-4103.md)

> **[P1] Add these headers to the Tizen devel package.**
>
> When building the Tizen GPU devel RPM, the newly installed OpenCL headers are
> missing from the `%files devel` section in `packaging/nntrainer.spec`. This
> can fail the build with `installed but unpackaged files`, or omit the headers
> from the package. Please add their installation paths to the `%{with gpu}`
> file list.

## PR #4105 — P1

- Suggested location: `nntrainer/layers/fc_layer.cpp:263`
- [Detailed review](./pr-4105.md)

> **[P1] Apply the fused activation during incremental FC forwarding.**
>
> Regular `forwarding()` applies the activation after the bias, but
> `incremental_forwarding()` performs only the matrix multiplication and bias
> addition. As a result, CausalLM decoding returns the pre-activation value
> when `fused_activation` replaces a separate `ActivationLayer`. Please apply
> the same epilogue in both paths and add an equivalence test.

## PR #4106 — P1

- Suggested location: `nntrainer/engine.h:196`
- [Detailed review](./pr-4106.md)

> **[P1] Make factory registration work for non-CPU backends.**
>
> Calling this API for the `gpu` or `qnn` engine falls through to the base
> implementation because the concrete contexts do not override it, so
> registration always returns `-1`. The new facade therefore works only for
> the CPU backend. Please implement the corresponding overrides and delegate
> to each context's registry.

## PR #4108 — P1

- Suggested location: `nntrainer/context.h:185`
- [Detailed review](./pr-4108.md)

> **[P1] Check integer-dot support before selecting DP4A.**
>
> This classifies every OpenCL device without DPAS as DP4A-capable, but
> `dpas == false` does not imply support for integer-dot operations. Once this
> plan drives dispatch, the kernel will fail to compile on devices without the
> required extension. Please probe integer-dot support separately and fall
> back to a generic GEMM or CPU path when it is unavailable.

## PR #4109 — P1

- Suggested location: `nntrainer/cl_svm_allocator.cpp:74`
- [Detailed review](./pr-4109.md)

> **[P1] Do not expose a host fallback allocation as SVM.**
>
> If SVM is unsupported or exhausted, this returns ordinary host memory, but
> the caller continues to treat the pointer as SVM. Passing it to
> `clSetKernelArgSVMPointer()` can then produce a `CL_INVALID_*` error or
> incorrect output. Please track the actual memory type per allocation and use
> a staging path, or handle SVM allocation failure explicitly.

## PR #4110 — P1

- Suggested location: `nntrainer/tensor/cl_operations/blas_kernel_interface.cpp:1052`
- [Detailed review](./pr-4110.md)

> **[P1] Do not reuse activation-cache entries based only on their address.**
>
> If the same storage address, shape, and dtype are reused with different
> contents and no intervening cache miss, this skips both upload and
> quantization for the new input. The memory planner can reuse one allocation
> for different tensor values, so this may reuse data from the previous token.
> Please include a storage generation in the key or limit the cache lifetime
> to a single fan-out group.

## PR #4111 — P1

- Suggested location: `nntrainer/tensor/cl_operations/blas_kernels.cpp:1682`
- [Detailed review](./pr-4111.md)

> **[P1] Do not route `K == 64` through the XMX path.**
>
> Inputs with `M > 4`, `N % 64 == 0`, and `K == 64` pass this gate, but the
> resulting weight-surface width is `K / 2`, or 32 bytes. That violates the
> Intel 2D block-I/O minimum width of 64 bytes and can cause undefined behavior
> or a device fault. Please require `K >= 128` or pad the surface appropriately.

## PR #4112 — P1

- Suggested location: `nntrainer/tensor/cl_operations/geglu_cl_op.cpp:55`
- [Detailed review](./pr-4112.md)

> **[P1] Register the whole-op GeGLU kernels before use.**
>
> There is no caller of `registerGeGLUClKernels()` in this tree, so the kernel
> vector is empty on the first GeGLU execution. This indexing can therefore
> access the vector out of bounds and crash before any fallback runs. Please
> register the kernels during context initialization or perform safe lazy
> registration on first use.

## PR #4113 — P1

- Suggested location: `nntrainer/tensor/cl_operations/blas_kernels.cpp:2235`
- [Detailed review](./pr-4113.md)

> **[P1] Chain the upload event to the first GEMM on out-of-order queues.**
>
> With `NNTR_GPU_SVM_POOL=0` and a weight smaller than 64 MiB, this asynchronous
> upload's completion event is not added to the consumer's wait list. The GEMM
> may therefore run first and read a partially uploaded weight. Please
> associate the event with the backing allocation and wait for it at the first
> dispatch, or otherwise guarantee completion on this path.

## PR #4114 — P1

- Suggested location: `nntrainer/tensor/tensor_pool.h:409`
- [Detailed review](./pr-4114.md)

> **[P1] Treat graph outputs as implicit host consumers.**
>
> A terminal FP16 GPU output has no downstream views, so
> `all_consumers_gpu` remains `true` and the tensor is classified as
> `GPU_CLMEM`. Because the return path does not lower the device plane to the
> host, the API caller can receive a stale or zero-filled host shadow. Please
> register graph outputs as host consumers or lower them before returning.

## PR #4115 — P1

- Suggested location: `nntrainer/layers/llm/swiglu_layer.cpp:75`
- [Detailed review](./pr-4115.md)

> **[P1] Allow multi-token prefill from a nonzero position.**
>
> Appending a prompt to a cached system prompt legitimately produces a prefill
> with `from > 0` and `to - from > 1`. This check rejects that case, preventing
> the full range from being evaluated. Please support the multi-token range and
> reserve the one-token restriction for actual decoding.

## PR #4116 — P1

- Suggested location: `Applications/CausalLM/llm_util.hpp:135`
- [Detailed review](./pr-4116.md)

> **[P1] Base the repack decision on the graph's actual engine.**
>
> In an OpenCL build without `NNTR_ENGINE`, this helper returns `gpu`, but the
> graph nodes have no `engine=` property and therefore still execute on the
> CPU. Skipping the QS4CX repack in that state causes the ARM CPU path to throw
> from `getPackedData()` on the first FC operation. Please inspect the
> registered execution context, or apply the selected engine consistently to
> the entire graph before skipping the repack.

## PR #4117 — P1

- Suggested location: `nntrainer/cuda_context.cpp:85`
- [Detailed review](./pr-4117.md)

> **[P1] Wire CUDA graphs' tensor pools to the CUDA allocator.**
>
> Registering the allocator here is not enough: `NeuralNetwork::compile()`
> never checks for CUDA nodes, so it still passes `engine_name="cpu"` when
> constructing the graph. When an `engine=cuda` model is compiled, its weight
> and activation pools therefore use ordinary host memory. Please select the
> `cuda` graph allocator whenever the graph contains a CUDA node.

## PR #4118 — P1

- Suggested location: `nntrainer/layers/llm/logit_softcapping.cpp:141`
- [Detailed review](./pr-4118.md)

> **[P1] Synchronize after the softcap kernel before the host reads logits.**
>
> With `NNTR_CUDA_ASYNC=1`, this kernel is enqueued without a wait, while the
> `finish()` above runs before the enqueue. If incremental inference
> immediately reads the logits on the CPU, it can observe stale or partially
> updated values. Please synchronize after this kernel at the host-read
> boundary.

## PR #4119 — P1

- Suggested location: `nntrainer/cuda/cuda_compute_ops.cpp:84`
- [Detailed review](./pr-4119.md)

> **[P1] Do not send device-only tensors through the CPU fallback.**
>
> With `NNTR_CUDA_DEV_ACT=1`, a scale-preparation or CUDA-kernel failure
> reaches this `input.dot()` call even though the input or output may be a
> device-only `cudaMalloc` pointer. The CPU cannot dereference those pointers,
> so this fallback can crash. Please return an error for device-only tensors,
> or perform explicit D2H/H2D staging before invoking the CPU path.

## PR #4120 — P1

- Suggested location: `nntrainer/cuda/cuda_fc_qint4.cpp:393`
- [Detailed review](./pr-4120.md)

> **[P1] Handle the `K % 4` tail in the GEMV kernel.**
>
> For a QS4CX FC with `M == 1` and `K % 4 != 0`, this loop processes only
> groups of four and drops the final one to three products. Because `wrowsum`
> still covers all of K, the kernel reports success but silently returns the
> wrong result. Please add a scalar tail or route these shapes to the general
> GEMM path.

## PR #4121 — P1

- Suggested location: `nntrainer/cuda/cuda_compute_ops.cpp:111`
- [Detailed review](./pr-4121.md)

> **[P1] Propagate failures from device copies used for dtype conversion.**
>
> The return value from `cuda::copy_any()` is ignored here. If reading a device
> input fails during FP32-to-FP16 conversion, the zero-initialized `xs` buffer
> is converted and returned as though it contained valid input. Please
> propagate the copy failure and switch the source pointer only after a
> successful transfer.

## PR #4122 — P1

- Suggested location: `nntrainer/cuda/cuda_fc_qint4.cpp:1123`
- [Detailed review](./pr-4122.md)

> **[P1] Use the correct prefetch API for CUDA 12.x and CUDA 13.**
>
> This five-argument form of `cudaMemPrefetchAsync()` is a CUDA 13 API, so an
> `enable-cuda` build with CUDA 12.x headers does not compile. Please select
> the appropriate call with `CUDART_VERSION`, or reject CUDA versions below 13
> during configuration.

## PR #4123 — P2

- Suggested location: `nntrainer/cuda/cuda_fc_qint4.cpp:1145`
- [Detailed review](./pr-4123.md)

> **[P2] Add tail padding to the JIT int8 weight scratch buffer.**
>
> With `NNTR_CUDA_I8_JIT=1`, this buffer is allocated as exactly `K*N` bytes,
> while the persistent weight used as the same cuBLAS IMMA operand includes
> `FC_I8_TAIL_PAD` for wide tail reads. The final vector load can therefore
> cross the JIT allocation boundary. Please give the JIT scratch buffer the
> same padding.

## PR #4124 — P1

- Suggested location: `nntrainer/layers/llm/tie_word_embedding.cpp:731`
- [Detailed review](./pr-4124.md)

> **[P1] Continue through the common post-processing after a successful CUDA matmul.**
>
> Returning from the function here skips the bias addition after the Q6_K
> lm-head CUDA matmul; with `batch > 1`, it also stops after the first batch.
> Please record that the matmul completed and let control reach the common
> per-batch post-processing instead of returning from the function.

## PR #4125 — P1

- Suggested location: `nntrainer/tensor/cl_operations/attention_kernels.cpp:2115`
- [Detailed review](./pr-4125.md)

> **[P1] Do not let the static kernel cache bypass the clone ring.**
>
> This attention path stores the result of `registerClKernel()` in a
> function-static variable, so after the first call it never borrows another
> clone from the ring. The kernel-argument overwrite hazard this PR is intended
> to prevent therefore remains in the main QK/softmax/SV path. Please acquire a
> ring-aware lease for every argument-binding and enqueue operation.

## PR #4126 — P1

- Suggested location: `nntrainer/cuda_context.cpp:332`
- [Detailed review](./pr-4126.md)

> **[P1] Do not replay the graph without a consumer for the position buffer.**
>
> From the second token onward, the only updated values are the two integers
> written by `cuda_set_pos()`, but nothing in the current tree passes
> `cuda_pos_buffer()` to a kernel. The replay therefore keeps using the
> position and KV length captured for the first token and produces incorrect
> subsequent tokens. Please wire the buffer into its consumers, or keep M2-B
> on the eager path until they are available.

## PR #4127 — P1

- Suggested location: `nntrainer/cuda_context.cpp:124`
- [Detailed review](./pr-4127.md)

> **[P1] Do not enable the incomplete M2-B path by default on discrete GPUs.**
>
> `NNTR_CUDA_M2B` already has an active reader, so this default enables graph
> replay with only `NNTR_ENGINE=cuda`. Because no kernel consumes the position
> buffer, the second token onward reuses the position and KV length from the
> first capture. Please enable this only when all required consumers are
> registered; otherwise retain the eager path.

## PR #4128 — P1

- Suggested location: `nntrainer/tensor/cpu_backend/arm/arm_compute_backend_fp16.cpp:455`
- [Detailed review](./pr-4128.md)

> **[P1] Match the packed RHS layout to the kernel's `nr` layout.**
>
> On the ARM i8mm FP16×QS4CX path, the `nr=8` buffer produced by
> `QS4CX_Tensor::pack()` is passed to an `nr=4` kernel without validating the
> variant. The two layouts place weights and scale/bias metadata at different
> offsets, so the entire result is corrupted. Please use a kernel matching the
> packed layout and explicitly reject mismatches.

## PR #4129 — P1

- Suggested location: `api/ccapi/include/half_fp16.h:152`
- [Detailed review](./pr-4129.md)

> **[P1] Specialize `numeric_limits<Half>` with binary16 limits.**
>
> In wrapper FP16 builds, `lowest()` returns `Half{}`, or zero. When every
> max/global-max pooling input is negative, this selects a zero that was never
> present in the input, and the backward index may remain unset. Please define
> the IEEE binary16 limits and add a regression test for pooling over
> all-negative inputs.

## PR #4130 — P1

- Suggested location: `nntrainer/utils/thread_manager.cpp:45`
- [Detailed review](./pr-4130.md)

> **[P1] The default static Windows configuration still creates one singleton per DLL.**
>
> When multiple layer DLLs each link the static nntrainer library, each DLL
> gets its own copy of this function and its `static instance`. Moving the
> definition into a `.cpp` file therefore still leaves one thread pool per
> DLL. Please make a single shared DLL own the instance and have every layer
> DLL use it.

## PR #4131 — P1

- Suggested location: `.github/workflows/msvc_gpu_verify.yml:18`
- [Detailed review](./pr-4131.md)

> **[P1] Run this Windows GPU check for regular pull requests.**
>
> This workflow only handles pushes to `ci/msvc-*` branches and
> `workflow_dispatch`; it does not run for the current
> `upstream-pr/trackW4-msvc-gpu-ci` head and has no `pull_request` trigger.
> Please add a `pull_request` trigger for changes targeting `main` and confirm
> that the job succeeds on this PR so the new check actually validates changes
> before merge.

## PR #4141 — P2

- Suggested location: `nntrainer/tensor/manager.cpp:930`
- [Detailed review](./pr-4141.md)

> **[P2] V2 must treat nested gradient lifetimes as overlapping.**
>
> With `NNTR_MEM_PLANNER=v2` during training, the current overlap check misses
> the case where one gradient lifetime contains another and may assign both
> gradients the same offset. One gradient can then silently overwrite the
> other. Until this is fixed using the standard test
> `a.start < b.end && b.start < a.end`, please reject V2 for training pools or
> fall back to V1.

## PR #4142 — P2

- Suggested location: `Applications/CausalLM/models/causal_lm.cpp:673`
- [Detailed review](./pr-4142.md)

> **[P2] Report the same memory metrics in the LFM2 embedding path.**
>
> This change adds `peak commit` only to `CausalLM::run()`, but LFM2 uses a
> separate `run_with_embeddings()` path when `USE_EMBEDDING=true` and still
> reports only the old `peak memory` value. Please move the measurement and
> reporting into a shared helper so both execution paths display the same
> information.

## PR #4143 — P1

- Suggested location: `nntrainer/layers/cuda_layers/cuda_rmsnorm_layer.cpp:38`
- [Detailed review](./pr-4143.md)

> **[P1] Do not request the model's FP32 gamma as an FP16 weight.**
>
> When activations are FP16, `getWeightDataType()` also returns FP16, but
> RMSNorm gamma is stored as FP32 in the model file. Reading it at half the
> element size corrupts both gamma and the file position for the next weight.
> Please request gamma as FP32 and use either a mixed-type kernel or a
> correctly converted GPU copy.

## PR #4144 — P1

- Suggested location: `Applications/CausalLM/models/transformer.cpp:190`
- [Detailed review](./pr-4144.md)

> **[P1] Clamp the runtime length to the actual RoPE table length.**
>
> Paths such as Gemma3 and Llama do not pass `max_position_embeddings` to MHA,
> so their actual RoPE table still has the default 40,960 entries, while this
> check only uses the larger value from the model configuration. For example,
> with a configured limit of 131,072 and a requested length of 100,000, no
> clamp occurs and the code reads past the table. Please pass the model limit
> to MHA or clamp against the table length that was actually created.

## PR #4145 — Approval

- Suggested location: overall review body
- [Detailed review](./pr-4145.md)

> **Approval**
>
> I verified that `cl_mem` is retained whenever any token sharing an offset is
> classified as `GPU_CLMEM`, and is skipped only when all such tokens use SVM.
> The classification inputs are also identical before and after allocation, so
> I found no correctness blocker in this PR's direct changes.

## PR #4148 — P1

- Suggested location: `nntrainer/tensor/cl_operations/attention_kernels.cpp:2704`
- [Detailed review](./pr-4148.md)

> **[P1] Do not select the SG16 kernel based only on the presence of the DPAS extension.**
>
> `caps().dpas` only reports that the extension exists; it does not mean the
> device requires subgroup size 16. On a DPAS device whose required minimum
> subgroup size is 8, enabling the SG16 path by default can cause kernel
> failure or undefined results. Please query the subgroup requirement, enable
> this path only when it is 16, and otherwise fall back to block-Q.

## PR #4149 — P1

- Suggested location: `nntrainer/tensor/cl_operations/blas_kernels.cpp:1183`
- [Detailed review](./pr-4149.md)

> **[P1] Pass the starting offset of each `cl_mem` view to the kernel.**
>
> For a Tensor view with a nonzero starting offset, `getClMem()` still passes
> only the handle for the full buffer, so this indexing always reads and writes
> from element zero. With `batch > 1`, the second batch therefore overwrites
> the batch-0 region. Please add the input and output element offsets to the
> kernel indices, or route nonzero-offset views through a safe fallback.

## PR #4150 — P1

- Suggested location: `Applications/CausalLM/jni/Android.mk:87`
- [Detailed review](./pr-4150.md)

> **[P1] The Android CausalLM build does not include the OpenCL GPU implementation.**
>
> Although the new GPU sources were added to `Android.mk`, the prebuilt
> nntrainer package does not propagate `ENABLE_OPENCL` to the consumer build,
> so preprocessing removes the OpenCL code from both files. Please export this
> macro and install `blas_kernels.h`, which the OpenCL path includes, as a
> prebuilt header so Android can actually use these GPU layers.

## PR #4151 — Approval

- Suggested location: overall review body
- [Detailed review](./pr-4151.md)

> **Approval**
>
> I verified that every row is assigned to exactly one subgroup for all
> supported NSG/TM combinations and that partial sums are accumulated in the
> same order as before. The barriers between `psum` and `ssum` and the tail
> conditions are also correct, so I found no correctness blocker in the direct
> XRED changes.

## PR #4152 — P2

- Suggested location: `nntrainer/cuda/cuda_fc_qint4.cpp:473`
- [Detailed review](./pr-4152.md)

> **[P2] Handle the `K % 4` tail in the fused GEMV.**
>
> For an FP16 QS4CX FC with `NNTR_CUDA_FC_FUSED_DECQ=1`, `M == 1`, and
> `K % 4 != 0`, this loop drops the final one to three products. Because
> `wrowsum` still includes those weights, the function reports success while
> silently producing an incorrect output. Please add a scalar tail or restrict
> the fused path to `K % 4 == 0`.
