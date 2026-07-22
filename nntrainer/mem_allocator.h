// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.h
 * @date    13 Jan 2025
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */
#ifndef __MEM_ALLOCATOR_H__
#define __MEM_ALLOCATOR_H__

#include <cstddef>
#include <memory>
#include <string>

namespace nntrainer {

/**
 * @brief MemAllocator, Memory allocator class
 *
 * Backend-pluggable allocator for MemoryPool. The default implementation
 * uses std::aligned_alloc (zero-initialized), so MemoryPool no longer
 * embeds calloc/SVM/rpcmem dispatch via macros. Per-vendor Contexts
 * (ClContext, QNNContext) install their own subclass through
 * ContextData::setMemAllocator(). MemoryPool then takes the allocator
 * by shared_ptr at construction and routes allocate/deallocate through
 * it — see ARCHITECTURE.md.
 */
class MemAllocator {
public:
  MemAllocator() = default;
  virtual ~MemAllocator() = default;

  /**
   * @brief Allocate aligned memory.
   * @param[out] ptr       receives the allocated address
   * @param[in]  size      bytes
   * @param[in]  alignment alignment in bytes (must be a power of two);
   *                       caller passes the page size or a smaller value
   *                       depending on the use case
   *
   * The default implementation uses std::aligned_alloc and zero-fills.
   * Subclasses (ClSVMAllocator, QNNRpcManager) override to plumb the
   * vendor allocator instead.
   */
  virtual void alloc(void **ptr, size_t size, size_t alignment);

  /**
   * @brief Free memory previously returned by alloc().
   *
   * Must match the allocator that produced ptr — never mix free() with
   * a vendor allocator's release call.
   *
   * A backend that cannot release @p ptr must retain ownership and throw an
   * exception. Returning normally is the ownership-transfer boundary: callers
   * may immediately forget the pointer and must never pass it to free() again.
   */
  virtual void free(void *ptr);

  /**
   * @brief Attempt a release without allowing backend exceptions to escape.
   * @return true only when free() returned normally and the caller may forget
   *         @p ptr; false means release was not confirmed and the caller must
   *         retain its ownership record.
   *
   * This non-virtual adapter preserves the MemAllocator vtable while allowing
   * lifecycle code to aggregate release failures across multiple buffers.
   */
  bool tryFree(void *ptr) noexcept {
    try {
      free(ptr);
      return true;
    } catch (...) {
      return false;
    }
  }

  /**
   * @brief Backend identifier ("cpu" / "gpu-svm" / "qnn-rpc").
   *
   * MemoryPool uses this in error messages and lets callers reason
   * about pointer ownership (e.g. SVM vs host memory).
   */
  virtual std::string getName() { return "cpu"; };
};
} // namespace nntrainer

#endif
