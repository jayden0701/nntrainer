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

#include "QnnMem.h"

#include <algorithm>
#include <cstdlib>
#include <inttypes.h>
#include <limits>
#include <new>
#include <nntrainer_log.h>
#include <stdexcept>
#include <utility>

namespace nntrainer {

QNNRuntimeLifecycle::ExecutionGuard
QNNRuntimeLifecycle::acquireExecutionGuard() {
  if (state_.load(std::memory_order_acquire) != QnnRuntimeState::RUNNING) {
    throw std::runtime_error("QNN runtime execution admission is closed");
  }

  ExecutionGuard guard(lifecycle_mutex_);
  if (state_.load(std::memory_order_acquire) != QnnRuntimeState::RUNNING) {
    guard.unlock();
    throw std::runtime_error("QNN runtime shutdown has started");
  }
  return guard;
}

QNNRuntimeLifecycle::CleanupGuard QNNRuntimeLifecycle::acquireCleanupGuard() {
  return CleanupGuard(lifecycle_mutex_);
}

QNNRuntimeLifecycle::CleanupGuard QNNRuntimeLifecycle::beginRuntimeShutdown() {
  auto expected = QnnRuntimeState::RUNNING;
  if (!state_.compare_exchange_strong(
        expected, QnnRuntimeState::SHUTDOWN_REQUESTED,
        std::memory_order_acq_rel, std::memory_order_acquire)) {
    throw std::runtime_error("QNN runtime shutdown has already started");
  }

  try {
    return CleanupGuard(lifecycle_mutex_);
  } catch (...) {
    // Admission is already closed, but active readers may not have drained.
    // Preserve the runtime and prohibit every later vendor cleanup attempt.
    state_.store(QnnRuntimeState::QUARANTINED, std::memory_order_release);
    throw;
  }
}

void QNNRuntimeLifecycle::finishRuntimeShutdown(
  const CleanupGuard &guard, QnnRuntimeState final_state) noexcept {
  if (!ownsCleanupGuard(guard)) {
    ml_loge("Cannot finish QNN runtime shutdown without its cleanup lease");
    return;
  }
  if (final_state != QnnRuntimeState::SHUT_DOWN &&
      final_state != QnnRuntimeState::QUARANTINED) {
    ml_loge("Invalid final QNN runtime state: %u",
            static_cast<unsigned int>(final_state));
    final_state = QnnRuntimeState::QUARANTINED;
  }
  state_.store(final_state, std::memory_order_release);
}

void QNNRuntimeLifecycle::quarantine(const CleanupGuard &guard) noexcept {
  if (!ownsCleanupGuard(guard)) {
    ml_loge("Cannot quarantine QNN runtime without its cleanup lease");
    return;
  }
  state_.store(QnnRuntimeState::QUARANTINED, std::memory_order_release);
}

bool QNNRuntimeLifecycle::ownsExecutionGuard(
  const ExecutionGuard &guard) const noexcept {
  return guard.owns_lock() && guard.mutex() == &lifecycle_mutex_;
}

bool QNNRuntimeLifecycle::ownsCleanupGuard(
  const CleanupGuard &guard) const noexcept {
  return guard.owns_lock() && guard.mutex() == &lifecycle_mutex_;
}

bool QNNRuntimeLifecycle::isRunning(const CleanupGuard &guard) const noexcept {
  return ownsCleanupGuard(guard) &&
         state_.load(std::memory_order_acquire) == QnnRuntimeState::RUNNING;
}

bool QNNRuntimeLifecycle::isShutdownRequested(
  const CleanupGuard &guard) const noexcept {
  return ownsCleanupGuard(guard) && state_.load(std::memory_order_acquire) ==
                                      QnnRuntimeState::SHUTDOWN_REQUESTED;
}

bool QNNRuntimeLifecycle::allowsVendorCleanup(
  const CleanupGuard &guard) const noexcept {
  if (!ownsCleanupGuard(guard)) {
    return false;
  }
  const auto current_state = state_.load(std::memory_order_acquire);
  return current_state == QnnRuntimeState::RUNNING ||
         current_state == QnnRuntimeState::SHUTDOWN_REQUESTED;
}

QNNRpcManager::QNNRpcManager(
  const QNN_INTERFACE_VER_TYPE &qnn_interface,
  std::shared_ptr<void> backend_library_lifetime,
  std::shared_ptr<QNNRuntimeLifecycle> runtime_lifecycle) :
  qnn_interface_(qnn_interface),
  backend_library_lifetime_(std::move(backend_library_lifetime)),
  runtime_lifecycle_(std::move(runtime_lifecycle)) {
  if (!runtime_lifecycle_) {
    throw std::invalid_argument("QNN runtime lifecycle is unavailable");
  }
#ifdef ENABLE_QNN
  if (!backend_library_lifetime_) {
    throw std::invalid_argument("QNN backend library lifetime is unavailable");
  }
  if (!RpcMem::global().valid()) {
    throw std::runtime_error(
      "QNN RPC memory functions are not available from libcdsprpc");
  }
  if (qnn_interface_.memRegister == nullptr ||
      qnn_interface_.memDeRegister == nullptr) {
    throw std::runtime_error("QNN memory function pointers are unavailable");
  }
#endif
}

QNNRpcManager::~QNNRpcManager() {
  try {
    auto cleanup_guard = runtime_lifecycle_->acquireCleanupGuard();
    (void)cleanup_guard;
#ifdef ENABLE_QNN
    size_t released_allocations = 0;
    {
      std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
      for (auto allocation = allocations_.begin();
           allocation != allocations_.end();) {
        if (allocation->second.state == AllocationState::ACTIVE &&
            allocation->second.registrations.empty()) {
          RpcMem::global().free(allocation->first);
          allocation = allocations_.erase(allocation);
          ++released_allocations;
        } else {
          ++allocation;
        }
      }
    }

    const auto report = resourceReport();
    if (released_allocations > 0) {
      ml_logw("QNN RPC manager released %zu unregistered allocations during "
              "final cleanup",
              released_allocations);
    }
    if (!report.clean()) {
      // Context/backend lifetime is not owned by this destructor. Calling QNN
      // here could target an already-freed context, so ambiguous resources are
      // deliberately left alive instead of risking a driver-visible UAF.
      ml_loge("QNN RPC manager retained %zu allocations and %zu registrations "
              "at teardown (quarantined allocations=%zu, registrations=%zu, "
              "duplicate acquisitions=%zu, allocation admission closed=%d)",
              report.allocations, report.registrations,
              report.quarantined_allocations, report.quarantined_registrations,
              report.duplicate_acquisitions,
              report.allocation_admission_closed ? 1 : 0);
    }
#endif
  } catch (const std::exception &e) {
    ml_loge("Exception while reporting retained QNN RPC resources: %s",
            e.what());
  } catch (...) {
    ml_loge("Unknown exception while reporting retained QNN RPC resources");
  }
}

bool QNNRpcManager::DescriptorSignatureLess::operator()(
  const DescriptorSignature &lhs,
  const DescriptorSignature &rhs) const noexcept {
  const auto lhs_type = static_cast<uint64_t>(lhs.data_type);
  const auto rhs_type = static_cast<uint64_t>(rhs.data_type);
  if (lhs_type != rhs_type) {
    return lhs_type < rhs_type;
  }
  if (lhs.dimensions != rhs.dimensions) {
    return std::lexicographical_compare(
      lhs.dimensions.begin(), lhs.dimensions.end(), rhs.dimensions.begin(),
      rhs.dimensions.end());
  }
  return lhs.required_bytes < rhs.required_bytes;
}

void QNNRpcManager::alloc(void **ptr, size_t size, size_t alignment) {
  if (ptr == nullptr) {
    throw std::invalid_argument("QNN RPC allocation output is null");
  }
  *ptr = nullptr;
  if (size == 0) {
    throw std::invalid_argument("QNN RPC allocation size is zero");
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument(
      "QNN RPC allocation alignment is not a power of two");
  }

  auto execution_guard = runtime_lifecycle_->acquireExecutionGuard();
  (void)execution_guard;

#ifdef ENABLE_QNN
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("QNN RPC allocation exceeds rpcmem size limit");
  }

  // Reserve a map node before acquiring RPC backing. No fallible host ledger
  // allocation remains between rpcmem_alloc and ownership publication.
  std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
  if (allocation_admission_closed_) {
    throw std::runtime_error(
      "QNN RPC allocation is disabled after an ambiguous acquisition");
  }
  auto placeholder = allocations_.emplace(
    nullptr, Allocation{size, -1, AllocationState::ACQUIRING, {}});
  if (!placeholder.second) {
    throw std::logic_error("QNN RPC allocation ledger reservation failed");
  }
  auto allocation_node = allocations_.extract(placeholder.first);

  void *mem_pointer = RpcMem::global().alloc(
    kRpcMemHeapIdSystem, kRpcMemDefaultFlags, static_cast<int>(size));
  if (mem_pointer == nullptr) {
    throw std::bad_alloc();
  }

  allocation_node.key() = mem_pointer;
  allocation_node.mapped().state = AllocationState::ACTIVE;
  auto inserted = allocations_.insert(std::move(allocation_node));
  if (!inserted.inserted) {
    // rpcmem returned an address that is already live. It is impossible to
    // know which acquisition a free would release, so preserve both and poison
    // the tracked allocation.
    inserted.position->second.state = AllocationState::QUARANTINED;
    ++duplicate_acquisitions_;
    allocation_admission_closed_ = true;
    ml_loge("QNN RPC allocator returned a duplicate live pointer: ptr=%p, "
            "existing_size=%zu, requested_size=%zu",
            mem_pointer, inserted.position->second.size, size);
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

void QNNRpcManager::registerQnnTensor(
  const QNNRuntimeLifecycle::ExecutionGuard &guard, void *ptr,
  Qnn_Tensor_t &qnn_tensor, Qnn_ContextHandle_t context,
  size_t required_bytes) {
  if (!runtime_lifecycle_->ownsExecutionGuard(guard)) {
    throw std::invalid_argument(
      "QNN tensor registration requires a runtime execution lease");
  }
#ifdef ENABLE_QNN
  if (ptr == nullptr || context == nullptr) {
    throw std::invalid_argument(
      "QNN tensor registration requires a pointer and context");
  }
  if (qnn_tensor.version != QNN_TENSOR_VERSION_1) {
    throw std::invalid_argument(
      "QNN tensor registration supports only V1 descriptors");
  }
  if (qnn_tensor.v1.rank > 0 && qnn_tensor.v1.dimensions == nullptr) {
    throw std::invalid_argument(
      "QNN tensor registration requires complete dimensions");
  }
  if (required_bytes == 0) {
    throw std::invalid_argument(
      "QNN tensor registration requires a nonzero byte count");
  }

  DescriptorSignature signature;
  signature.data_type = qnn_tensor.v1.dataType;
  signature.required_bytes = required_bytes;
  if (qnn_tensor.v1.rank > 0) {
    signature.dimensions.assign(qnn_tensor.v1.dimensions,
                                qnn_tensor.v1.dimensions + qnn_tensor.v1.rank);
  }

  std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
  auto allocation_it = allocations_.find(ptr);
  if (allocation_it == allocations_.end()) {
    throw std::invalid_argument(
      "QNN tensor pointer is not an owned RPC allocation base");
  }

  auto &allocation = allocation_it->second;
  if (allocation.state != AllocationState::ACTIVE) {
    throw std::runtime_error("QNN RPC allocation is quarantined");
  }
  if (required_bytes > allocation.size) {
    throw std::length_error("QNN tensor descriptor exceeds its RPC allocation");
  }

  auto context_it = allocation.registrations.find(context);
  if (context_it != allocation.registrations.end()) {
    auto existing = context_it->second.find(signature);
    if (existing != context_it->second.end()) {
      if (existing->second.state != RegistrationState::ACTIVE ||
          existing->second.mem_handle == nullptr) {
        throw std::runtime_error(
          "QNN tensor registration is incomplete or quarantined");
      }
      qnn_tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
      qnn_tensor.v1.memHandle = existing->second.mem_handle;
      return;
    }
  }

  const auto context_inserted = allocation.registrations.try_emplace(context);
  context_it = context_inserted.first;
  decltype(context_it->second.begin()) registration_it;
  try {
    auto registration_inserted =
      context_it->second.emplace(signature, Registration{});
    if (!registration_inserted.second) {
      throw std::logic_error("QNN registration ledger insertion failed");
    }
    registration_it = registration_inserted.first;
  } catch (...) {
    if (context_inserted.second && context_it->second.empty()) {
      allocation.registrations.erase(context_it);
    }
    throw;
  }

  auto erase_pending_registration = [&]() noexcept {
    context_it->second.erase(registration_it);
    if (context_it->second.empty()) {
      allocation.registrations.erase(context_it);
    }
  };

  if (allocation.fd == -1) {
    try {
      allocation.fd = RpcMem::global().to_fd(ptr);
    } catch (...) {
      erase_pending_registration();
      throw;
    }
    if (allocation.fd == -1) {
      erase_pending_registration();
      throw std::runtime_error("rpcmem_to_fd failed for QNN tensor");
    }
  }

  auto &registration = registration_it->second;
  Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
  descriptor.memShape = {
    static_cast<uint32_t>(signature.dimensions.size()),
    signature.dimensions.empty() ? nullptr : signature.dimensions.data(),
    nullptr};
  descriptor.dataType = signature.data_type;
  descriptor.memType = QNN_MEM_TYPE_ION;
  descriptor.ionInfo.fd = allocation.fd;

  Qnn_MemHandle_t mem_handle{nullptr};
  Qnn_ErrorHandle_t register_status = QNN_SUCCESS;
  try {
    register_status =
      qnn_interface_.memRegister(context, &descriptor, 1u, &mem_handle);
  } catch (...) {
    registration.mem_handle = mem_handle;
    registration.state = RegistrationState::QUARANTINED;
    registration.last_call_threw = true;
    allocation.state = AllocationState::QUARANTINED;
    ml_loge("QNN memRegister threw: ptr=%p, fd=%d, context=%p, "
            "mem_handle=%p",
            ptr, allocation.fd, static_cast<void *>(context),
            static_cast<void *>(mem_handle));
    throw;
  }

  registration.last_error = register_status;
  registration.mem_handle = mem_handle;
  if (QNN_GET_ERROR_CODE(register_status) != QNN_SUCCESS) {
    ml_loge("QNN memRegister failed: error=%" PRIu64
            ", public_error=%u, ptr=%p, fd=%d, context=%p, mem_handle=%p",
            static_cast<uint64_t>(register_status),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(register_status)), ptr,
            allocation.fd, static_cast<void *>(context),
            static_cast<void *>(mem_handle));
    if (mem_handle == nullptr) {
      erase_pending_registration();
    } else {
      registration.state = RegistrationState::QUARANTINED;
      allocation.state = AllocationState::QUARANTINED;
    }
    throw std::runtime_error("QNN tensor memory registration failed");
  }

  if (mem_handle == nullptr) {
    registration.state = RegistrationState::QUARANTINED;
    allocation.state = AllocationState::QUARANTINED;
    ml_loge("QNN memRegister succeeded without a handle: ptr=%p, fd=%d, "
            "context=%p",
            ptr, allocation.fd, static_cast<void *>(context));
    throw std::runtime_error(
      "QNN tensor memory registration returned a null handle");
  }

  registration.state = RegistrationState::ACTIVE;
  qnn_tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  qnn_tensor.v1.memHandle = mem_handle;
#else
  (void)guard;
  (void)ptr;
  (void)qnn_tensor;
  (void)context;
  (void)required_bytes;
#endif
}

bool QNNRpcManager::deRegisterOneLocked(void *ptr, Qnn_ContextHandle_t context,
                                        const DescriptorSignature &signature,
                                        Allocation &allocation,
                                        Registration &registration) noexcept {
#ifdef ENABLE_QNN
  if (registration.state != RegistrationState::ACTIVE ||
      registration.mem_handle == nullptr ||
      qnn_interface_.memDeRegister == nullptr) {
    const auto old_state = registration.state;
    registration.state = RegistrationState::QUARANTINED;
    allocation.state = AllocationState::QUARANTINED;
    ml_loge("Cannot deregister QNN memory: ptr=%p, context=%p, "
            "mem_handle=%p, state=%u",
            ptr, static_cast<void *>(context),
            static_cast<void *>(registration.mem_handle),
            static_cast<unsigned int>(old_state));
    return false;
  }

  registration.state = RegistrationState::DEREGISTERING;
  auto mem_handle = registration.mem_handle;
  Qnn_ErrorHandle_t deregister_status = QNN_SUCCESS;
  try {
    deregister_status = qnn_interface_.memDeRegister(&mem_handle, 1u);
  } catch (...) {
    registration.state = RegistrationState::QUARANTINED;
    registration.last_call_threw = true;
    allocation.state = AllocationState::QUARANTINED;
    ml_loge("QNN memDeRegister threw: ptr=%p, context=%p, mem_handle=%p", ptr,
            static_cast<void *>(context),
            static_cast<void *>(registration.mem_handle));
    return false;
  }

  registration.last_error = deregister_status;
  if (QNN_GET_ERROR_CODE(deregister_status) != QNN_SUCCESS) {
    registration.state = RegistrationState::QUARANTINED;
    allocation.state = AllocationState::QUARANTINED;
    ml_loge("QNN memDeRegister failed: error=%" PRIu64
            ", public_error=%u, ptr=%p, context=%p, mem_handle=%p, "
            "data_type=%" PRIu64 ", rank=%zu, bytes=%zu",
            static_cast<uint64_t>(deregister_status),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(deregister_status)),
            ptr, static_cast<void *>(context),
            static_cast<void *>(registration.mem_handle),
            static_cast<uint64_t>(signature.data_type),
            signature.dimensions.size(), signature.required_bytes);
    return false;
  }
#else
  (void)ptr;
  (void)context;
  (void)signature;
  (void)allocation;
  (void)registration;
#endif
  return true;
}

void QNNRpcManager::free(void *ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }

  try {
    auto cleanup_guard = runtime_lifecycle_->acquireCleanupGuard();
#ifdef ENABLE_QNN
    std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
    auto allocation_it = allocations_.find(ptr);
    if (allocation_it == allocations_.end()) {
      ml_loge("Refusing to free unknown QNN RPC pointer: %p", ptr);
      return;
    }

    auto &allocation = allocation_it->second;
    if (allocation.state != AllocationState::ACTIVE) {
      ml_loge("Retaining quarantined QNN RPC backing without further vendor "
              "calls: ptr=%p, size=%zu, state=%u",
              ptr, allocation.size,
              static_cast<unsigned int>(allocation.state));
      return;
    }

    if (!runtime_lifecycle_->allowsVendorCleanup(cleanup_guard) &&
        !allocation.registrations.empty()) {
      ml_loge("Retaining registered QNN RPC backing after runtime shutdown: "
              "ptr=%p, size=%zu, contexts=%zu",
              ptr, allocation.size, allocation.registrations.size());
      return;
    }

    bool cleanup_failed = false;
    for (auto context_it = allocation.registrations.begin();
         context_it != allocation.registrations.end();) {
      for (auto registration_it = context_it->second.begin();
           registration_it != context_it->second.end();) {
        if (deRegisterOneLocked(ptr, context_it->first, registration_it->first,
                                allocation, registration_it->second)) {
          registration_it = context_it->second.erase(registration_it);
        } else {
          cleanup_failed = true;
          break;
        }
      }

      if (cleanup_failed) {
        break;
      }
      if (context_it->second.empty()) {
        context_it = allocation.registrations.erase(context_it);
      } else {
        ++context_it;
      }
    }

    if (!allocation.registrations.empty() ||
        allocation.state != AllocationState::ACTIVE) {
      size_t registrations = 0;
      for (const auto &context : allocation.registrations) {
        registrations += context.second.size();
      }
      ml_loge("Retaining QNN RPC backing: ptr=%p, size=%zu, "
              "registrations=%zu, state=%u",
              ptr, allocation.size, registrations,
              static_cast<unsigned int>(allocation.state));
      return;
    }

    RpcMem::global().free(ptr);
    allocations_.erase(allocation_it);
#else
    (void)cleanup_guard;
    std::free(ptr);
#endif
  } catch (const std::exception &e) {
    ml_loge("Retaining QNN RPC backing after cleanup exception: ptr=%p, "
            "reason=%s",
            ptr, e.what());
  } catch (...) {
    ml_loge("Retaining QNN RPC backing after unknown cleanup exception: ptr=%p",
            ptr);
  }
}

QnnRpcResourceReport QNNRpcManager::resourceReport() const {
  std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
  QnnRpcResourceReport report;
  report.allocations = allocations_.size();
  report.duplicate_acquisitions = duplicate_acquisitions_;
  report.allocation_admission_closed = allocation_admission_closed_;
  for (const auto &allocation : allocations_) {
    if (allocation.second.state == AllocationState::QUARANTINED) {
      ++report.quarantined_allocations;
    }
    for (const auto &context : allocation.second.registrations) {
      report.registrations += context.second.size();
      for (const auto &registration : context.second) {
        if (registration.second.state == RegistrationState::QUARANTINED) {
          ++report.quarantined_registrations;
        }
      }
    }
  }
  return report;
}

size_t
QNNRpcManager::registrationCount(const QNNRuntimeLifecycle::CleanupGuard &guard,
                                 Qnn_ContextHandle_t context) const {
  if (!runtime_lifecycle_->ownsCleanupGuard(guard)) {
    throw std::invalid_argument(
      "QNN registration count requires a runtime cleanup lease");
  }
  std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);
  size_t count = 0;
  for (const auto &allocation : allocations_) {
    auto context_it = allocation.second.registrations.find(context);
    if (context_it != allocation.second.registrations.end()) {
      count += context_it->second.size();
    }
  }
  return count;
}

QnnRegistrationDrainReport QNNRpcManager::drainRegistrationsForShutdown(
  const QNNRuntimeLifecycle::CleanupGuard &guard) noexcept {
  QnnRegistrationDrainReport report;
  if (!runtime_lifecycle_ || !runtime_lifecycle_->ownsCleanupGuard(guard)) {
    report.failure = QnnRegistrationDrainFailure::INVALID_GUARD;
    return report;
  }
  if (!runtime_lifecycle_->isShutdownRequested(guard)) {
    report.failure = QnnRegistrationDrainFailure::INVALID_RUNTIME_STATE;
    return report;
  }

#ifdef ENABLE_QNN
  try {
    std::lock_guard<std::mutex> ledger_guard(ledger_mutex_);

    for (auto &allocation : allocations_) {
      for (auto context = allocation.second.registrations.begin();
           context != allocation.second.registrations.end();) {
        if (context->second.empty()) {
          context = allocation.second.registrations.erase(context);
          continue;
        }
        if (allocation.first == nullptr || allocation.second.fd == -1 ||
            allocation.second.state == AllocationState::ACQUIRING ||
            context->second.size() >
              std::numeric_limits<size_t>::max() - report.discovered) {
          report.failure = QnnRegistrationDrainFailure::INVALID_REGISTRATION;
          return report;
        }
        report.discovered += context->second.size();
        for (const auto &registration : context->second) {
          if (registration.second.state == RegistrationState::QUARANTINED) {
            ++report.quarantined;
          }
        }
        ++context;
      }
    }
    report.remaining = report.discovered;

    if (report.discovered == 0) {
      return report;
    }
    if (qnn_interface_.memDeRegister == nullptr) {
      report.failure = QnnRegistrationDrainFailure::INVALID_REGISTRATION;
      return report;
    }

    // Complete all fallible host-side preflight before the first vendor call.
    // The global uniqueness check prevents calling memDeRegister twice for an
    // aliased handle returned in violation of the registration contract.
    std::vector<Qnn_MemHandle_t> handles;
    handles.reserve(report.discovered);
    for (const auto &allocation : allocations_) {
      for (const auto &context : allocation.second.registrations) {
        if (context.first == nullptr) {
          report.failure = QnnRegistrationDrainFailure::INVALID_REGISTRATION;
          return report;
        }
        for (const auto &registration : context.second) {
          const auto &resource = registration.second;
          if (resource.state != RegistrationState::ACTIVE ||
              resource.mem_handle == nullptr) {
            report.failure = QnnRegistrationDrainFailure::INVALID_REGISTRATION;
            return report;
          }
          if (std::find(handles.begin(), handles.end(), resource.mem_handle) !=
              handles.end()) {
            ml_loge("QNN shutdown found an aliased memory handle: %p",
                    static_cast<void *>(resource.mem_handle));
            report.failure = QnnRegistrationDrainFailure::DUPLICATE_HANDLE;
            return report;
          }
          handles.push_back(resource.mem_handle);
        }
      }
    }

    for (auto &allocation_entry : allocations_) {
      auto &allocation = allocation_entry.second;
      for (auto context = allocation.registrations.begin();
           context != allocation.registrations.end();) {
        for (auto registration = context->second.begin();
             registration != context->second.end();) {
          ++report.attempted;
          if (!deRegisterOneLocked(allocation_entry.first, context->first,
                                   registration->first, allocation,
                                   registration->second)) {
            report.failure = QnnRegistrationDrainFailure::DEREGISTER_FAILED;
            if (registration->second.state == RegistrationState::QUARANTINED) {
              ++report.quarantined;
            }
            return report;
          }
          registration = context->second.erase(registration);
          ++report.drained;
          --report.remaining;
        }

        if (context->second.empty()) {
          context = allocation.registrations.erase(context);
        } else {
          ++context;
        }
      }
    }
  } catch (const std::exception &e) {
    report.failure = QnnRegistrationDrainFailure::HOST_EXCEPTION;
    ml_loge("Exception while draining QNN registrations: %s", e.what());
  } catch (...) {
    report.failure = QnnRegistrationDrainFailure::HOST_EXCEPTION;
    ml_loge("Unknown exception while draining QNN registrations");
  }
#else
  (void)guard;
#endif
  return report;
}

} // namespace nntrainer
