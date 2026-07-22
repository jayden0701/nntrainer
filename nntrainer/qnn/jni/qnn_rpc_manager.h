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
#include "QnnTypes.h"
#include "rpc_mem.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mem_allocator.h>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace nntrainer {

/** Summary returned by a best-effort registration cleanup pass. */
struct QnnDeregistrationReport {
  size_t attempted{0};
  size_t succeeded{0};
  size_t failed{0};
  size_t quarantined{0};

  bool success() const noexcept { return failed == 0 && quarantined == 0; }
};

/** @brief Manages QNN RPC shared memory allocation via libcdsprpc. */
class QNNRpcManager : public MemAllocator {
public:
  using ExecutionGuard = std::shared_lock<std::shared_mutex>;
  using CleanupGuard = std::unique_lock<std::shared_mutex>;

  explicit QNNRpcManager(const QNN_INTERFACE_VER_TYPE &qnn_interface);
  ~QNNRpcManager() override;

  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;

  std::string getName() override { return "qnn"; }

  /** Block cleanup while a graph is binding tensors or executing. */
  ExecutionGuard acquireExecutionGuard();

  /** Drain active executions and exclude new ones during lifecycle cleanup. */
  CleanupGuard acquireCleanupGuard();

  /** Permanently close execution admission and drain the runtime. */
  CleanupGuard acquireRuntimeShutdownGuard();

  /** Verify that a cleanup guard exclusively owns this manager's gate. */
  bool ownsCleanupGuard(const CleanupGuard &cleanup_guard) noexcept;

  void registerQnnTensor(void *ptr, Qnn_Tensor_t &qnnTensor,
                         Qnn_ContextHandle_t context,
                         const ExecutionGuard &execution_guard);

  size_t registrationCount() const;
  size_t registrationCount(Qnn_ContextHandle_t context) const;
  size_t quarantinedRegistrationCount() const;

private:
  enum class RegistrationState : uint8_t {
    REGISTERING,
    ACTIVE,
    QUARANTINED,
  };

  struct Registration {
    Qnn_ContextHandle_t context{nullptr};
    int fd{-1};
    Qnn_MemHandle_t mem_handle{nullptr};
    Qnn_DataType_t data_type{};
    std::vector<uint32_t> dimensions;
    RegistrationState state{RegistrationState::REGISTERING};
    Qnn_ErrorHandle_t last_error{QNN_SUCCESS};
  };

  QnnDeregistrationReport deRegisterAllLocked() noexcept;
  bool deRegisterOneLocked(void *ptr, Registration &registration) noexcept;

  QNN_INTERFACE_VER_TYPE qnnInterface_{};

  /**
   * Lock order is lifecycle_mutex_ followed by registration_mutex_. Graph
   * execution holds a shared lifecycle lock; free/context teardown holds the
   * exclusive lock.
   */
  std::shared_mutex lifecycle_mutex_;
  bool runtime_shutdown_{false};
  mutable std::mutex registration_mutex_;

  // RPC backing allocations remain here until they are actually freed.
  std::map<void *, size_t, std::less<void *>> allocations_;

  using ContextRegistrations =
    std::map<Qnn_ContextHandle_t, Registration, std::less<Qnn_ContextHandle_t>>;

  // The same RPC allocation may be registered independently in multiple QNN
  // contexts. Each context owns a distinct memHandle.
  std::map<void *, ContextRegistrations, std::less<void *>> registrations_;
};

} // namespace nntrainer
#endif
