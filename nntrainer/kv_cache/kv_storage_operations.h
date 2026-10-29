// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_storage_operations.h
 * @brief Backend operations for strictly typed KV storage and ordered device
 * access.
 * @author Codex (OpenAI)
 * @bug Publication is synchronous; multiple execution streams are unsupported.
 */
#ifndef NNTRAINER_KV_STORAGE_OPERATIONS_H
#define NNTRAINER_KV_STORAGE_OPERATIONS_H
#include <mem_allocator.h>
namespace nntrainer::kv {
/** Operations retain their backend lifetime. synchronize() must report errors
 * and really complete submitted work, even when normal dispatch is
 * asynchronous.
 */
class KvStorageOperations {
public:
  virtual ~KvStorageOperations() = default;
  virtual AllocationKind kind() const = 0;
  virtual bool accepts(const AllocationInfo &info) const {
    return info.kind == kind();
  }
  virtual void zero(void *address, size_t bytes) = 0;
  virtual void synchronize() = 0;
  /** True means completed; false means a device dependency orders the consumer.
   * Host readers must not consume a merely ordered device view.
   */
  virtual bool producerReady() {
    synchronize();
    return true;
  }
  virtual bool supportsReplay() const { return false; }
  virtual void invalidateReplay() noexcept {}
  virtual void read(void *host, const void *native, size_t bytes) = 0;
  virtual void write(void *native, const void *host, size_t bytes) = 0;
};
} // namespace nntrainer::kv
#endif
