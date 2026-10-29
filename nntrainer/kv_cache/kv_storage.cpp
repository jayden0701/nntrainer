// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_storage.cpp
 * @brief Own aligned host or strictly described device allocations shared by
 * cache views and Tensor aliases.
 * @author Codex (OpenAI)
 * @bug Only synchronous publication is supported.
 */
#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_storage.h>

#include <atomic>
#include <cstring>
#include <utility>

namespace nntrainer::kv {

KvStorage::KvStorage(size_t bytes, std::shared_ptr<MemAllocator> allocator,
                     std::shared_ptr<KvStorageOperations> operations) :
  operations_(std::move(operations)),
  allocator_(std::move(allocator)),
  memory_(nullptr),
  bytes_(bytes) {
  if (!bytes || !allocator_)
    throw std::invalid_argument("KV storage requires an allocator and bytes");
  if (!operations_ &&
      (!allocator_->isHostAddressable() || allocator_->isDeviceVisible() ||
       allocator_->isSVM() || allocator_->needsRegister()))
    throw std::invalid_argument("KV storage requires a plain host allocator");
  checkedAdd(bytes, 63);
  void *ptr = nullptr;
  allocator_->alloc(&ptr, bytes, 64);
  if (!ptr)
    throw std::bad_alloc();
  try {
    info_ = allocator_->describe(ptr);
    const auto expected =
      operations_ ? operations_->kind() : AllocationKind::Host;
    if (info_.kind == AllocationKind::Unknown || info_.kind != expected ||
        info_.base != ptr || (info_.bytes && info_.bytes < bytes) ||
        (operations_ && (!info_.bytes || !operations_->accepts(info_))) ||
        reinterpret_cast<uintptr_t>(ptr) % 64)
      throw std::invalid_argument(
        "KV allocation differs from its strict profile");
    if (operations_)
      operations_->zero(ptr, bytes);
    else
      std::memset(ptr, 0, bytes);
  } catch (...) {
    allocator_->free(ptr);
    throw;
  }
  allocation_ = ptr;
  memory_.setAddr(ptr);
  memory_.setHostAddressable(info_.hostAddressable());
  memory_.setSVM(info_.openClSvm());
  static std::atomic<uint64_t> next{1};
  identity_ = next.fetch_add(1);
}

std::shared_ptr<KvStorage>
KvStorage::create(size_t bytes, std::shared_ptr<MemAllocator> allocator,
                  std::shared_ptr<KvStorageOperations> operations) {
  return std::shared_ptr<KvStorage>(
    new KvStorage(bytes, std::move(allocator), std::move(operations)));
}

KvStorage::~KvStorage() {
  // free() must release safely even after a backend execution error. CUDA free
  // handles pending use; normal successful execution was already synchronized.
  if (operations_) {
    try {
      operations_->synchronize();
    } catch (...) {
      // Destruction cannot publish data or throw during exception unwinding.
    }
  }
  allocator_->free(allocation_);
}

void KvStorage::checkRange(size_t offset, size_t bytes) const {
  if (offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("KV storage access exceeds its allocation");
}

void KvStorage::read(void *host, size_t offset, size_t bytes) const {
  checkRange(offset, bytes);
  if (!bytes)
    return;
  if (!host)
    throw std::invalid_argument("Null KV read destination");
  const auto *native = static_cast<const char *>(allocation_) + offset;
  if (operations_)
    operations_->read(host, native, bytes);
  else
    std::memcpy(host, native, bytes);
}

void KvStorage::write(const void *host, size_t offset, size_t bytes) {
  checkRange(offset, bytes);
  if (!bytes)
    return;
  if (!host)
    throw std::invalid_argument("Null KV write source");
  auto *native = static_cast<char *>(allocation_) + offset;
  if (operations_)
    operations_->write(native, host, bytes);
  else
    std::memcpy(native, host, bytes);
}

std::shared_ptr<MemoryData> KvStorage::memory() {
  return std::shared_ptr<MemoryData>(shared_from_this(), &memory_);
}

} // namespace nntrainer::kv
