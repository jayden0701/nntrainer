// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_rpc_manager.h
 * @date    06 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn rpc memory manager
 */
#ifndef __QNN_RPC_MANAGER_H__
#define __QNN_RPC_MANAGER_H__

#include "QnnInterface.h"
#include "rpc_mem.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mem_allocator.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace nntrainer {

/** @brief Runtime state shared by QNN objects and their late-lived allocator.
 */
enum class QnnRuntimeState : uint8_t {
  RUNNING,
  SHUTDOWN_REQUESTED,
  SHUT_DOWN,
  QUARANTINED,
};

/**
 * @brief Coordinates QNN execution, resource cleanup, and runtime shutdown.
 *
 * QNNVar and QNNRpcManager share this object because MemoryPool can retain the
 * allocator after the context owner is gone. Shutdown admission is closed
 * before waiting for active readers so new executions cannot starve teardown.
 */
class QNNRuntimeLifecycle {
public:
  using ExecutionGuard = std::shared_lock<std::shared_mutex>;
  using CleanupGuard = std::unique_lock<std::shared_mutex>;

  /** @brief Acquire a shared lease for one runtime execution or allocation. */
  ExecutionGuard acquireExecutionGuard();

  /** @brief Acquire an exclusive lease for normal resource cleanup. */
  CleanupGuard acquireCleanupGuard();

  /**
   * @brief Permanently close execution admission and drain active readers.
   */
  CleanupGuard beginRuntimeShutdown();

  /** @brief Mark the result of teardown while holding its exclusive lease. */
  void finishRuntimeShutdown(const CleanupGuard &guard,
                             QnnRuntimeState final_state) noexcept;

  /**
   * @brief Close new admission after an ambiguous call by the current reader.
   *
   * Existing readers retain their leases and unwind normally. No reader tries
   * to upgrade to the exclusive gate. Quarantine is sticky and prohibits
   * runtime teardown even if shutdown was already waiting for this reader.
   */
  void quarantine(const ExecutionGuard &guard) noexcept;

  /** @brief Permanently close admission after an ambiguous cleanup call. */
  void quarantine(const CleanupGuard &guard) noexcept;

  /**
   * @brief Check whether a previously admitted reader may start another call.
   *
   * Readers admitted before shutdown may finish, but a runtime fault revokes
   * readers that were queued behind another RPC ledger operation.
   */
  bool allowsExecutionContinuation(const ExecutionGuard &guard) const noexcept;

  /** @brief Verify that a shared lease belongs to this lifecycle. */
  bool ownsExecutionGuard(const ExecutionGuard &guard) const noexcept;

  /** @brief Verify that an exclusive lease belongs to this lifecycle. */
  bool ownsCleanupGuard(const CleanupGuard &guard) const noexcept;

  /** @brief Return whether normal vendor calls remain admitted. */
  bool isRunning(const CleanupGuard &guard) const noexcept;

  /** @brief Return whether shutdown-time cleanup owns the runtime gate. */
  bool isShutdownRequested(const CleanupGuard &guard) const noexcept;

  /** @brief Return whether QNN memory deregistration remains admitted. */
  bool allowsVendorCleanup(const CleanupGuard &guard) const noexcept;

  /** @brief Return the current sticky runtime state. */
  QnnRuntimeState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }

private:
  void quarantineState() noexcept;

  mutable std::shared_mutex lifecycle_mutex_;
  std::atomic<QnnRuntimeState> state_{QnnRuntimeState::RUNNING};
};

/** @brief Snapshot of resources retained by the QNN RPC manager. */
struct QnnRpcResourceReport {
  size_t allocations{0};
  size_t registrations{0};
  size_t quarantined_allocations{0};
  size_t quarantined_registrations{0};
  size_t duplicate_acquisitions{0};
  bool allocation_admission_closed{false};

  bool clean() const noexcept { return allocations == 0 && registrations == 0; }
};

/** @brief Reason a shutdown-time memory registration drain stopped. */
enum class QnnRegistrationDrainFailure : uint8_t {
  NONE,
  INVALID_GUARD,
  INVALID_RUNTIME_STATE,
  INVALID_REGISTRATION,
  DUPLICATE_HANDLE,
  DEREGISTER_FAILED,
  HOST_EXCEPTION,
};

/** @brief Exact outcome of a shutdown-time memory registration drain. */
struct QnnRegistrationDrainReport {
  size_t discovered{0};
  size_t attempted{0};
  size_t drained{0};
  size_t remaining{0};
  size_t quarantined{0};
  QnnRegistrationDrainFailure failure{QnnRegistrationDrainFailure::NONE};

  bool success() const noexcept {
    return failure == QnnRegistrationDrainFailure::NONE && remaining == 0;
  }
};

/** @brief Manages QNN RPC shared memory allocation via libcdsprpc. */
class QNNRpcManager : public MemAllocator {
public:
  /**
   * @brief Construct from the interface selected by QNNContext.
   *
   * @param qnn_interface copied function table for the selected backend
   * @param backend_library_lifetime shared owner of the selected backend DSO
   * @param runtime_lifecycle gate shared with the QNN runtime owner
   */
  explicit QNNRpcManager(
    const QNN_INTERFACE_VER_TYPE &qnn_interface,
    std::shared_ptr<void> backend_library_lifetime,
    std::shared_ptr<QNNRuntimeLifecycle> runtime_lifecycle);
  ~QNNRpcManager() override;

  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) noexcept override;

  std::string getName() override { return "qnn"; }

  /**
   * @brief Register an exact RPC allocation for a QNN tensor descriptor.
   *
   * A planned allocation can be reused by non-overlapping tensors. Distinct
   * descriptor signatures therefore own distinct registrations instead of
   * reusing an incompatible handle.
   */
  void registerQnnTensor(const QNNRuntimeLifecycle::ExecutionGuard &guard,
                         void *ptr, Qnn_Tensor_t &qnn_tensor,
                         Qnn_ContextHandle_t context, size_t required_bytes);

  /** @brief Return a thread-safe resource ledger snapshot. */
  QnnRpcResourceReport resourceReport() const;

  /** @brief Count registrations owned by one QNN context. */
  size_t registrationCount(const QNNRuntimeLifecycle::CleanupGuard &guard,
                           Qnn_ContextHandle_t context) const;

  /**
   * @brief Deregister every known memory handle before context teardown.
   *
   * Successful registration nodes are erased one at a time. Allocation nodes
   * and RPC backing remain owned until their MemoryPool releases them.
   */
  QnnRegistrationDrainReport drainRegistrationsForShutdown(
    const QNNRuntimeLifecycle::CleanupGuard &guard) noexcept;

private:
  enum class AllocationState : uint8_t {
    ACQUIRING,
    ACTIVE,
    QUARANTINED,
  };

  enum class RegistrationState : uint8_t {
    REGISTERING,
    ACTIVE,
    DEREGISTERING,
    QUARANTINED,
  };

  struct DescriptorSignature {
    Qnn_DataType_t data_type{};
    std::vector<uint32_t> dimensions;
    size_t required_bytes{0};
  };

  struct DescriptorSignatureLess {
    bool operator()(const DescriptorSignature &lhs,
                    const DescriptorSignature &rhs) const noexcept;
  };

  struct Registration {
    Qnn_MemHandle_t mem_handle{nullptr};
    RegistrationState state{RegistrationState::REGISTERING};
    Qnn_ErrorHandle_t last_error{QNN_SUCCESS};
    bool last_call_threw{false};
  };

  using SignatureRegistrations =
    std::map<DescriptorSignature, Registration, DescriptorSignatureLess>;
  using ContextRegistrations =
    std::map<Qnn_ContextHandle_t, SignatureRegistrations,
             std::less<Qnn_ContextHandle_t>>;

  struct Allocation {
    size_t size{0};
    int fd{-1};
    AllocationState state{AllocationState::ACQUIRING};
    ContextRegistrations registrations;
  };

  bool deRegisterOneLocked(void *ptr, Qnn_ContextHandle_t context,
                           const DescriptorSignature &signature,
                           Allocation &allocation,
                           Registration &registration) noexcept;

  bool quarantineRegistrationAliasLocked(
    Allocation &allocation, Registration &candidate_registration) noexcept;

  bool quarantineAnyRegistrationAliasLocked() noexcept;

  void quarantineAllocationLocked(Allocation &allocation) noexcept;

  bool quarantineInvalidRegistrationsLocked(void *ptr,
                                            Allocation &allocation) noexcept;

  size_t countQuarantinedRegistrationsLocked() const noexcept;

  QNN_INTERFACE_VER_TYPE qnn_interface_{};
  std::shared_ptr<void> backend_library_lifetime_;
  std::shared_ptr<QNNRuntimeLifecycle> runtime_lifecycle_;

  mutable std::mutex ledger_mutex_;
  std::map<void *, Allocation, std::less<void *>> allocations_;
  size_t duplicate_acquisitions_{0};
  bool allocation_admission_closed_{false};
};

} // namespace nntrainer

#endif
