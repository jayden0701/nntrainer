// SPDX-License-Identifier: Apache-2.0
/**
 * @file   android_memory_allocator.cpp
 * @brief  RPCMEM-backed shared memory allocator for QNN buffers on Android.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "android_memory_allocator.h"

#include <engine.h>
#include <mem_allocator.h>

#include <limits>
#include <memory>
#include <nntrainer_log.h>
#include <stdexcept>
#include <utility>

namespace {

// Preserve the headroom used by the legacy direct-rpcmem adapter.
constexpr std::size_t kQnnAllocationPadding = 140;

const std::shared_ptr<nntrainer::MemAllocator> &getQnnAllocator() {
  // Some CausalLM handles are function-static and may be destroyed after the
  // Engine singleton. Keep this small holder alive so deallocation never has
  // to re-enter a destroyed Engine. The QNN Context and plugin are already
  // process-lifetime resources in Engine.
  static const auto *qnn_allocator = []() {
    auto allocator = nntrainer::Engine::Global().getAllocator("qnn");
    if (allocator->getName() != "qnn") {
      throw std::runtime_error("Registered QNN allocator has the wrong type");
    }
    return new std::shared_ptr<nntrainer::MemAllocator>(std::move(allocator));
  }();
  return *qnn_allocator;
}

} // namespace

void *allocate(std::size_t file_size) {
  if (file_size >
      std::numeric_limits<std::size_t>::max() - kQnnAllocationPadding) {
    throw std::length_error("QNN RPC allocation size overflow");
  }

  void *buffer = nullptr;
  getQnnAllocator()->alloc(&buffer, file_size + kQnnAllocationPadding,
                           alignof(std::max_align_t));
  return buffer;
}

void deallocate(void *pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  try {
    getQnnAllocator()->free(pointer);
  } catch (const std::exception &e) {
    ml_loge("Failed to release a CausalLM QNN allocation: %s", e.what());
  } catch (...) {
    ml_loge("Failed to release a CausalLM QNN allocation");
  }
}
