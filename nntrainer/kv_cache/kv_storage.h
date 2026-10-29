// SPDX-License-Identifier: Apache-2.0
/** @file kv_storage.h
 * @brief KV allocation ownership that survives every exported Tensor alias.
 * @author Codex (OpenAI)
 * @bug Only synchronous publication is supported.
 */
#ifndef NNTRAINER_KV_STORAGE_H
#define NNTRAINER_KV_STORAGE_H

#include <kv_cache/kv_storage_operations.h>
#include <mem_allocator.h>
#include <memory_data.h>

#include <cstdint>
#include <memory>

namespace nntrainer::kv {

/** One allocation owned through the existing allocator abstraction.
 * A device profile requires explicit operations and an actual allocation
 * result. data() is a native address, not permission to dereference it on the
 * host.
 */
class KvStorage : public std::enable_shared_from_this<KvStorage> {
public:
  static std::shared_ptr<KvStorage>
  create(size_t bytes, std::shared_ptr<MemAllocator> allocator,
         std::shared_ptr<KvStorageOperations> operations = nullptr);
  ~KvStorage();
  KvStorage(const KvStorage &) = delete;
  KvStorage &operator=(const KvStorage &) = delete;

  size_t bytes() const { return bytes_; }
  uint64_t identity() const { return identity_; }
  /** Stable allocation base, independent of exported wrapper metadata. */
  void *data() const { return allocation_; }
  const AllocationInfo &allocationInfo() const { return info_; }
  void read(void *host, size_t offset, size_t bytes) const;
  void write(const void *host, size_t offset, size_t bytes);
  /** Aliasing shared_ptr: the MemoryData itself owns no allocation.
   * Its address and metadata must not be changed by consumers. Rebind a Tensor
   * with Tensor::setData(), rather than mutating this shared wrapper.
   */
  std::shared_ptr<MemoryData> memory();

private:
  KvStorage(size_t bytes, std::shared_ptr<MemAllocator> allocator,
            std::shared_ptr<KvStorageOperations> operations);
  void checkRange(size_t offset, size_t bytes) const;
  std::shared_ptr<KvStorageOperations> operations_;
  AllocationInfo info_;
  std::shared_ptr<MemAllocator> allocator_;
  void *allocation_ = nullptr; /**< Immutable after successful allocation. */
  MemoryData memory_;
  size_t bytes_;
  uint64_t identity_;
};

} // namespace nntrainer::kv
#endif
