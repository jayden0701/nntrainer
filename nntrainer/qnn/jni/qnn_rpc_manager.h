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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mem_allocator.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nntrainer {

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

/** @brief Manages QNN RPC shared memory allocation via libcdsprpc. */
class QNNRpcManager : public MemAllocator {
public:
  /**
   * @brief Construct from the interface selected by QNNContext.
   *
   * @param qnn_interface copied function table for the selected backend
   * @param backend_library_lifetime shared owner of the selected backend DSO
   */
  explicit QNNRpcManager(const QNN_INTERFACE_VER_TYPE &qnn_interface,
                         std::shared_ptr<void> backend_library_lifetime);
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
  void registerQnnTensor(void *ptr, Qnn_Tensor_t &qnn_tensor,
                         Qnn_ContextHandle_t context, size_t required_bytes);

  /** @brief Return a thread-safe resource ledger snapshot. */
  QnnRpcResourceReport resourceReport() const;

  /** @brief Count registrations owned by one QNN context. */
  size_t registrationCount(Qnn_ContextHandle_t context) const;

private:
  enum class AllocationState : uint8_t {
    ACQUIRING,
    ACTIVE,
    QUARANTINED,
  };

  enum class RegistrationState : uint8_t {
    REGISTERING,
    ACTIVE,
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

  QNN_INTERFACE_VER_TYPE qnn_interface_{};
  std::shared_ptr<void> backend_library_lifetime_;

  mutable std::mutex ledger_mutex_;
  std::map<void *, Allocation, std::less<void *>> allocations_;
  size_t duplicate_acquisitions_{0};
  bool allocation_admission_closed_{false};
};

} // namespace nntrainer

#endif
