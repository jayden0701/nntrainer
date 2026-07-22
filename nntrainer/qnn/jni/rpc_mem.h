// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    rpc_mem.h
 * @date    21 May 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  haehun.yang <haehun.yang@samsung.com>
 * @bug     No known bugs
 * except for NYI items
 * @brief   Shared QNN backend loader for Hexagon RPC
 * memory.

 */
#ifndef __QNN_RPC_MEM_H__
#define __QNN_RPC_MEM_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer {

typedef void *(*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void *);
typedef int (*RpcMemToFdFn_t)(void *);

/** @brief default heap id / flags for rpcmem_alloc (Hexagon system heap) */
constexpr int kRpcMemHeapIdSystem = 25;
constexpr int kRpcMemDefaultFlags = 1;

/**
 * @brief Lazy singleton wrapping libcdsprpc.so rpc memory primitives.
 */
class RpcMem {
public:
  /** @brief access the library-local singleton (thread-safe init in C++11) */
  static RpcMem &global();

  /** @brief whether libcdsprpc.so and the rpcmem symbols were resolved */
  bool valid() const {
    return alloc_ != nullptr && free_ != nullptr && to_fd_ != nullptr;
  }

  void *alloc(int heapid, uint32_t flags, int size) const {
    return alloc_ ? alloc_(heapid, flags, size) : nullptr;
  }
  void free(void *ptr) const {
    if (free_)
      free_(ptr);
  }
  int to_fd(void *ptr) const { return to_fd_ ? to_fd_(ptr) : -1; }

private:
  RpcMem();
  RpcMem(const RpcMem &) = delete;
  RpcMem &operator=(const RpcMem &) = delete;

  RpcMemAllocFn_t alloc_{nullptr};
  RpcMemFreeFn_t free_{nullptr};
  RpcMemToFdFn_t to_fd_{nullptr};
};

} // namespace nntrainer

#endif // __QNN_RPC_MEM_H__
