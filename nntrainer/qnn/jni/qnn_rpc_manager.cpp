// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_rpc_manager.cpp
 * @date    06 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn rpc memory manager
 */
#include "qnn_rpc_manager.h"

#include "QnnTypes.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <inttypes.h>
#include <limits>
#include <new>
#include <nntrainer_log.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nntrainer {

QNNRpcManager::QNNRpcManager(const QNN_INTERFACE_VER_TYPE &qnn_interface) :
  qnnInterface_(qnn_interface) {
#ifdef ENABLE_QNN
  if (!RpcMem::global().valid()) {
    throw std::runtime_error(
      "QNN RPC memory functions are not available from libcdsprpc");
  }
  if (qnnInterface_.memRegister == nullptr ||
      qnnInterface_.memDeRegister == nullptr) {
    throw std::runtime_error("QNN memory function pointers are unavailable");
  }
#endif
}

QNNRpcManager::~QNNRpcManager() {
#ifdef ENABLE_QNN
  try {
    auto cleanup_guard = acquireRuntimeShutdownGuard();
    std::lock_guard<std::mutex> registration_guard(registration_mutex_);
    const auto report = deRegisterAllLocked();

    size_t retained = 0;
    for (auto allocation = allocations_.begin();
         allocation != allocations_.end();) {
      if (registrations_.find(allocation->first) != registrations_.end()) {
        ++retained;
        ++allocation;
        continue;
      }

      RpcMem::global().free(allocation->first);
      allocation = allocations_.erase(allocation);
    }

    if (!report.success() || retained > 0) {
      ml_loge("QNN RPC manager retained %zu backing allocations after "
              "deregistration failure",
              retained);
    }
  } catch (const std::exception &e) {
    // Destruction must never free backing after an uncertain deregistration.
    // Losing host bookkeeping is preferable to a driver-visible use-after-free
    // during process teardown.
    ml_loge("Exception during QNN RPC manager teardown: %s", e.what());
  } catch (...) {
    ml_loge("Unknown exception during QNN RPC manager teardown");
  }
#endif
}

QNNRpcManager::ExecutionGuard QNNRpcManager::acquireExecutionGuard() {
  ExecutionGuard execution_guard(lifecycle_mutex_);
  if (runtime_shutdown_) {
    throw std::runtime_error("QNN runtime execution is shut down");
  }
  return execution_guard;
}

QNNRpcManager::CleanupGuard QNNRpcManager::acquireCleanupGuard() {
  CleanupGuard cleanup_guard(lifecycle_mutex_);
  if (runtime_shutdown_) {
    throw std::runtime_error("QNN runtime cleanup is shut down");
  }
  return cleanup_guard;
}

QNNRpcManager::CleanupGuard QNNRpcManager::acquireRuntimeShutdownGuard() {
  CleanupGuard cleanup_guard(lifecycle_mutex_);
  runtime_shutdown_ = true;
  return cleanup_guard;
}

bool QNNRpcManager::ownsCleanupGuard(
  const CleanupGuard &cleanup_guard) noexcept {
  return cleanup_guard.owns_lock() &&
         cleanup_guard.mutex() == &lifecycle_mutex_;
}

void QNNRpcManager::alloc(void **ptr, size_t size, size_t alignment) {
  (void)alignment;
  if (ptr == nullptr) {
    throw std::invalid_argument("QNN RPC allocation output is null");
  }
  *ptr = nullptr;
  if (size == 0) {
    throw std::invalid_argument("QNN RPC allocation size is zero");
  }

#ifdef ENABLE_QNN
  auto execution_guard = acquireExecutionGuard();
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("QNN RPC allocation exceeds rpcmem size limit");
  }

  // Reserve the backing ledger node before calling into rpcmem. Holding the
  // registration lock across acquisition ensures no fallible lock or map-node
  // allocation remains between acquiring the RPC resource and publishing its
  // ownership record.
  std::lock_guard<std::mutex> registration_guard(registration_mutex_);
  char allocation_placeholder;
  auto placeholder = allocations_.emplace(&allocation_placeholder, size);
  if (!placeholder.second) {
    throw std::logic_error("Failed to reserve QNN RPC allocation ledger node");
  }
  auto allocation_node = allocations_.extract(placeholder.first);

  void *mem_pointer = RpcMem::global().alloc(
    kRpcMemHeapIdSystem, kRpcMemDefaultFlags, static_cast<int>(size));
  if (mem_pointer == nullptr) {
    throw std::bad_alloc();
  }

  allocation_node.key() = mem_pointer;
  auto inserted = allocations_.insert(std::move(allocation_node));
  if (!inserted.inserted) {
    // rpcmem must not return an address that is still live. Do not free the
    // ambiguous address because it may be the previously tracked allocation.
    ml_loge("QNN RPC allocator returned a duplicate live pointer: %p",
            mem_pointer);
    throw std::runtime_error("QNN RPC allocator returned a duplicate pointer");
  }
  *ptr = mem_pointer;
#else
  *ptr = std::calloc(1, size);
  if (*ptr == nullptr) {
    throw std::bad_alloc();
  }
#endif
}

void QNNRpcManager::registerQnnTensor(void *ptr, Qnn_Tensor_t &qnn_tensor,
                                      Qnn_ContextHandle_t context,
                                      const ExecutionGuard &execution_guard) {
#ifdef ENABLE_QNN
  if (!execution_guard.owns_lock() ||
      execution_guard.mutex() != &lifecycle_mutex_) {
    throw std::logic_error(
      "QNN tensor registration requires an execution guard");
  }
  if (ptr == nullptr || context == nullptr) {
    throw std::invalid_argument(
      "QNN tensor registration requires a pointer and context");
  }
  if (qnn_tensor.v1.rank > 0 && qnn_tensor.v1.dimensions == nullptr) {
    throw std::invalid_argument(
      "QNN tensor registration requires complete dimensions");
  }

  std::lock_guard<std::mutex> registration_guard(registration_mutex_);
  if (allocations_.find(ptr) == allocations_.end()) {
    throw std::invalid_argument(
      "QNN tensor pointer is not owned by the RPC allocator");
  }

  auto outer = registrations_.find(ptr);
  if (outer != registrations_.end()) {
    auto existing = outer->second.find(context);
    if (existing != outer->second.end()) {
      if (existing->second.state != RegistrationState::ACTIVE) {
        throw std::runtime_error(
          "QNN tensor registration is quarantined or incomplete");
      }
      const auto &registered_dimensions = existing->second.dimensions;
      const bool dimensions_match =
        registered_dimensions.size() == qnn_tensor.v1.rank &&
        (registered_dimensions.empty() ||
         std::equal(registered_dimensions.begin(), registered_dimensions.end(),
                    qnn_tensor.v1.dimensions));
      if (existing->second.data_type != qnn_tensor.v1.dataType ||
          !dimensions_match) {
        throw std::runtime_error(
          "QNN tensor pointer was reused with an incompatible descriptor");
      }
      qnn_tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
      qnn_tensor.v1.memHandle = existing->second.mem_handle;
      return;
    }
  }

  std::vector<uint32_t> dimensions;
  if (qnn_tensor.v1.rank > 0) {
    dimensions.assign(qnn_tensor.v1.dimensions,
                      qnn_tensor.v1.dimensions + qnn_tensor.v1.rank);
  }

  outer = registrations_.try_emplace(ptr).first;

  const int mem_fd = RpcMem::global().to_fd(ptr);
  if (mem_fd == -1) {
    if (outer->second.empty()) {
      registrations_.erase(outer);
    }
    throw std::runtime_error("rpcmem_to_fd failed for QNN tensor");
  }

  auto inserted = outer->second.try_emplace(context);
  auto &registration = inserted.first->second;
  registration.context = context;
  registration.fd = mem_fd;
  registration.data_type = qnn_tensor.v1.dataType;
  registration.dimensions = std::move(dimensions);

  Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
  descriptor.memShape = {qnn_tensor.v1.rank, qnn_tensor.v1.dimensions, nullptr};
  descriptor.dataType = qnn_tensor.v1.dataType;
  descriptor.memType = QNN_MEM_TYPE_ION;
  descriptor.ionInfo.fd = mem_fd;

  Qnn_MemHandle_t mem_handle{nullptr};
  const auto register_status =
    qnnInterface_.memRegister(context, &descriptor, 1u, &mem_handle);
  registration.last_error = register_status;
  registration.mem_handle = mem_handle;

  if (register_status != QNN_SUCCESS) {
    ml_loge("QNN memRegister failed: error=%" PRIu64
            ", public_error=%u, ptr=%p, fd=%d, context=%p, mem_handle=%p",
            static_cast<uint64_t>(register_status),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(register_status)), ptr,
            mem_fd, static_cast<void *>(context),
            static_cast<void *>(mem_handle));
    if (mem_handle == nullptr) {
      outer->second.erase(inserted.first);
      if (outer->second.empty()) {
        registrations_.erase(outer);
      }
    } else {
      registration.state = RegistrationState::QUARANTINED;
    }
    throw std::runtime_error("QNN tensor memory registration failed");
  }

  if (mem_handle == nullptr) {
    registration.state = RegistrationState::QUARANTINED;
    ml_loge("QNN memRegister succeeded without returning a handle: ptr=%p, "
            "fd=%d, context=%p",
            ptr, mem_fd, static_cast<void *>(context));
    throw std::runtime_error(
      "QNN tensor memory registration returned a null handle");
  }

  registration.state = RegistrationState::ACTIVE;
  qnn_tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  qnn_tensor.v1.memHandle = mem_handle;
#else
  (void)ptr;
  (void)qnn_tensor;
  (void)context;
  (void)execution_guard;
#endif
}

bool QNNRpcManager::deRegisterOneLocked(void *ptr,
                                        Registration &registration) noexcept {
#ifdef ENABLE_QNN
  if (registration.state != RegistrationState::ACTIVE) {
    registration.state = RegistrationState::QUARANTINED;
    ml_loge("Refusing to retry non-active QNN registration: ptr=%p, "
            "context=%p, mem_handle=%p, state=%u",
            ptr, static_cast<void *>(registration.context),
            static_cast<void *>(registration.mem_handle),
            static_cast<unsigned int>(registration.state));
    return false;
  }
  if (qnnInterface_.memDeRegister == nullptr ||
      registration.mem_handle == nullptr) {
    registration.state = RegistrationState::QUARANTINED;
    ml_loge("Cannot deregister QNN memory: ptr=%p, context=%p, "
            "mem_handle=%p",
            ptr, static_cast<void *>(registration.context),
            static_cast<void *>(registration.mem_handle));
    return false;
  }

  auto mem_handle = registration.mem_handle;
  const auto deregister_status = qnnInterface_.memDeRegister(&mem_handle, 1u);
  if (deregister_status != QNN_SUCCESS) {
    registration.state = RegistrationState::QUARANTINED;
    registration.last_error = deregister_status;
    ml_loge("QNN memDeRegister failed: error=%" PRIu64
            ", public_error=%u, ptr=%p, fd=%d, context=%p, mem_handle=%p",
            static_cast<uint64_t>(deregister_status),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(deregister_status)),
            ptr, registration.fd, static_cast<void *>(registration.context),
            static_cast<void *>(registration.mem_handle));
    return false;
  }
#else
  (void)ptr;
  (void)registration;
#endif
  return true;
}

QnnDeregistrationReport QNNRpcManager::deRegisterAllLocked() noexcept {
  QnnDeregistrationReport report{};
  for (auto outer = registrations_.begin(); outer != registrations_.end();) {
    for (auto inner = outer->second.begin(); inner != outer->second.end();) {
      if (inner->second.state == RegistrationState::QUARANTINED) {
        ++report.quarantined;
        ++inner;
        continue;
      }

      ++report.attempted;
      if (deRegisterOneLocked(outer->first, inner->second)) {
        ++report.succeeded;
        inner = outer->second.erase(inner);
      } else {
        ++report.failed;
        ++inner;
      }
    }

    if (outer->second.empty()) {
      outer = registrations_.erase(outer);
    } else {
      ++outer;
    }
  }
  return report;
}

size_t QNNRpcManager::registrationCount() const {
  std::lock_guard<std::mutex> registration_guard(registration_mutex_);
  size_t count = 0;
  for (const auto &outer : registrations_) {
    count += outer.second.size();
  }
  return count;
}

size_t QNNRpcManager::registrationCount(Qnn_ContextHandle_t context) const {
  std::lock_guard<std::mutex> registration_guard(registration_mutex_);
  size_t count = 0;
  for (const auto &outer : registrations_) {
    if (outer.second.find(context) != outer.second.end()) {
      ++count;
    }
  }
  return count;
}

size_t QNNRpcManager::quarantinedRegistrationCount() const {
  std::lock_guard<std::mutex> registration_guard(registration_mutex_);
  size_t count = 0;
  for (const auto &outer : registrations_) {
    for (const auto &inner : outer.second) {
      if (inner.second.state == RegistrationState::QUARANTINED) {
        ++count;
      }
    }
  }
  return count;
}

void QNNRpcManager::free(void *ptr) {
#ifdef ENABLE_QNN
  if (ptr == nullptr) {
    return;
  }

  CleanupGuard cleanup_guard{};
  try {
    cleanup_guard = acquireCleanupGuard();
  } catch (const std::exception &e) {
    ml_loge("Retaining QNN RPC backing because cleanup is closed: ptr=%p, "
            "reason=%s",
            ptr, e.what());
    throw;
  } catch (...) {
    ml_loge("Retaining QNN RPC backing because cleanup is closed: ptr=%p", ptr);
    throw;
  }
  bool release_backing = false;
  {
    std::lock_guard<std::mutex> registration_guard(registration_mutex_);
    auto allocation = allocations_.find(ptr);
    if (allocation == allocations_.end()) {
      ml_loge("Refusing to free unknown QNN RPC pointer: %p", ptr);
      throw std::invalid_argument(
        "QNN RPC pointer is not owned by this manager");
    }

    auto outer = registrations_.find(ptr);
    if (outer != registrations_.end()) {
      for (auto inner = outer->second.begin(); inner != outer->second.end();) {
        if (deRegisterOneLocked(ptr, inner->second)) {
          inner = outer->second.erase(inner);
        } else {
          ++inner;
        }
      }
      if (outer->second.empty()) {
        registrations_.erase(outer);
      } else {
        ml_loge("Retaining QNN RPC backing after deregistration failure: "
                "ptr=%p, size=%zu, registrations=%zu",
                ptr, allocation->second, outer->second.size());
        throw std::runtime_error(
          "QNN RPC backing retained after deregistration failure");
      }
    }

    allocations_.erase(allocation);
    release_backing = true;
  }

  if (release_backing) {
    RpcMem::global().free(ptr);
  }
#else
  std::free(ptr);
#endif
}

} // namespace nntrainer
