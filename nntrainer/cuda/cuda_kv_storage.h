// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_kv_storage.h
 * @brief Strict CUDA operations for core-owned synchronous KV allocations.
 * @author Codex (OpenAI)
 * @bug Only one backend stream and explicit dense FP16 replay are supported.
 */
#ifndef NNTRAINER_CUDA_KV_STORAGE_H
#define NNTRAINER_CUDA_KV_STORAGE_H
#include <kv_cache/kv_storage_operations.h>
#include <memory>
namespace nntrainer::cuda {
/** Uses the existing primary context and backend stream. */
std::shared_ptr<kv::KvStorageOperations>
makeKvStorageOperations(AllocationKind kind = AllocationKind::CudaDevice,
                        bool graph_ordered = false);
/** Dense FP16 final V projection, directly into base[d_pos[0] * N].
 * FP32 accumulation, one decode row; no activation/append staging.
 */
bool kvValueSlot(const unsigned short *input, const unsigned short *weight,
                 unsigned short *base, int outputs, int inputs);
} // namespace nntrainer::cuda
#endif
