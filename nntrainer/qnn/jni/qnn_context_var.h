// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_context_var.h
 * @date    08 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context data related functions and classes
 * that manages the global configuration of the current QNN environment.
 */

#ifndef __QNN_CONTEXT_VAR_H__
#define __QNN_CONTEXT_VAR_H__

#include "BackendExtensions.hpp"
#include "IOTensor.hpp"
#include "Log/Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QNN/HTP/QnnHtpContext.h"
#include "QNN/QnnTypes.h"
#include "iotensor_wrapper.hpp"
#include "qnn_rpc_manager.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <inttypes.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>

#include <nntrainer_log.h>

using namespace qnn;
using namespace qnn::tools;

namespace nntrainer {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

enum class QnnContextEntryState : uint8_t {
  CREATING,
  ACTIVE,
  QUARANTINED,
};

struct Qnn_Context_Graph_t {
  Qnn_ContextHandle_t m_context = nullptr;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo = nullptr;
  std::map<std::string, qnn_wrapper_api::GraphInfo_t *>
    graph_map; /** graph name in Context - graph map **/
  std::map<std::string, uint32_t>
    graph_idx; /** graph name in Context - graph map **/

  uint32_t m_graphsCount = 0;
  QnnContextEntryState m_state = QnnContextEntryState::CREATING;
  Qnn_ErrorHandle_t m_lastError = QNN_CONTEXT_NO_ERROR;
  std::shared_ptr<uint8_t> m_binaryBuffer;
  uint64_t m_binarySize = 0;

  static bool validateTensorMetadata(const Qnn_Tensor_t *tensors,
                                     uint32_t count, const char *kind,
                                     uint32_t graph_index) {
    if (count == 0) {
      return true;
    }
    if (tensors == nullptr) {
      ml_loge("QNN graph %u has no %s tensor metadata", graph_index, kind);
      return false;
    }

    for (uint32_t i = 0; i < count; ++i) {
      const auto &tensor = tensors[i];
      if (tensor.version != QNN_TENSOR_VERSION_1) {
        ml_loge("QNN graph %u has unsupported %s tensor version at index %u",
                graph_index, kind, i);
        return false;
      }

      const auto &quant_params = tensor.v1.quantizeParams;
      const bool missing_axis_quantization =
        quant_params.quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET &&
        quant_params.axisScaleOffsetEncoding.numScaleOffsets > 0 &&
        quant_params.axisScaleOffsetEncoding.scaleOffset == nullptr;
      if (tensor.v1.name == nullptr || tensor.v1.name[0] == '\0' ||
          (tensor.v1.rank > 0 && tensor.v1.dimensions == nullptr) ||
          missing_axis_quantization) {
        ml_loge("QNN graph %u has invalid %s tensor metadata at index %u",
                graph_index, kind, i);
        return false;
      }
    }
    return true;
  }

  bool setGraphInfoMap() {
    graph_map.clear();
    if (m_graphsInfo == nullptr || m_graphsCount == 0) {
      ml_loge("QNN context metadata has no graphs");
      return false;
    }

    for (uint32_t i = 0; i < m_graphsCount; ++i) {
      auto *graph = m_graphsInfo[i];
      if (graph == nullptr || graph->graphName == nullptr ||
          graph->graphName[0] == '\0' ||
          !validateTensorMetadata(graph->inputTensors, graph->numInputTensors,
                                  "input", i) ||
          !validateTensorMetadata(graph->outputTensors, graph->numOutputTensors,
                                  "output", i)) {
        ml_loge("QNN context metadata is incomplete at graph index %u", i);
        graph_map.clear();
        return false;
      }

      auto inserted = graph_map.emplace(graph->graphName, graph);
      if (!inserted.second) {
        ml_loge("QNN context contains duplicate graph name: %s",
                graph->graphName);
        graph_map.clear();
        return false;
      }
    }
    return true;
  }

  qnn_wrapper_api::GraphInfo_t *getGraphPtr(const std::string &graph_name) {
    auto mapIt = graph_map.find(graph_name);
    if (mapIt != graph_map.end()) {
      return mapIt->second;
    }

    /**
     * nntrainer's GraphCore::ensureName() lowercases every layer name, but QNN
     * graph names are case-sensitive (e.g. "gemma_4_E2B_..."). When the layer
     * name was lowercased the exact lookup above misses, so fall back to a
     * case-insensitive match against the binary's real graph names.
     */
    auto ci_equal = [](const std::string &a, const std::string &b) {
      return a.size() == b.size() &&
             std::equal(a.begin(), a.end(), b.begin(),
                        [](unsigned char x, unsigned char y) {
                          return std::tolower(x) == std::tolower(y);
                        });
    };
    for (auto &kv : graph_map) {
      if (ci_equal(kv.first, graph_name))
        return kv.second;
    }

    ml_loge("cannot find graph");
    return nullptr;
  }

  int getGraphIdx(std::string graph_name) {
    auto mapIt = graph_idx.find(graph_name);
    if (mapIt != graph_idx.end()) {
      return mapIt->second;
    } else {
      ml_loge("cannot find graph");
      return -1;
    }
  }
};

struct QNNVar {
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  BackendExtensions *m_backendExtensions = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle = nullptr;
  iotensor::OutputDataType m_outputDataType =
    iotensor::OutputDataType::FLOAT_AND_NATIVE;
  iotensor::InputDataType m_inputDataType = iotensor::InputDataType::NATIVE;
  sample_app::ProfilingLevel m_profilingLevel = sample_app::ProfilingLevel::OFF;
  bool m_isBackendInitialized = false;
  void *m_backendLibraryHandle = nullptr;
  std::shared_ptr<void> m_backendLibraryLifetime;
  Qnn_LogHandle_t m_logHandle = nullptr;
  Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
  sample_app::QnnFunctionPointers m_qnnFunctionPointers{};
  std::shared_ptr<QNNRpcManager> RpcMem;
  IOTensorWrapper m_ioTensor;
  std::string name = "qnn_backend_param";
  std::map<std::string, Qnn_Context_Graph_t>
    ct_map; /** bin file name - Context map **/
  /**
   * Serializes registry mutation, context creation, and graph-handle
   * publication. References returned by this registry remain valid under the
   * existing owner contract: QNNContext teardown starts only after all QNN
   * layers stop using the shared runtime. Concurrent teardown is unsupported.
   */
  std::mutex context_registry_mutex;
  bool m_hasSystemContextFreeFailure = false;
  bool m_hasContextQuarantine = false;
  /**
   * Installed only after an ambiguous vendor free. The allocation-free
   * self-reference preserves this runtime and every resource it owns for the
   * remainder of the process.
   */
  std::shared_ptr<QNNVar> quarantine_self_reference;

  struct QuarantinedSystemContext {
    QnnSystemContext_Handle_t handle = nullptr;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    std::shared_ptr<uint8_t> binary_buffer;
    uint64_t binary_size = 0;
  };
  std::optional<QuarantinedSystemContext> m_quarantinedSystemContext;

  struct MalformedMetadataQuarantine {
    qnn_wrapper_api::GraphInfo_t **graphs_info = nullptr;
    std::shared_ptr<uint8_t> binary_buffer;
    uint64_t binary_size = 0;
  };
  /**
   * The SDK helper supplies no safe traversal bound when it returns a pointer
   * with a zero count. Retain at most one such result and never dereference or
   * free it by guessing an allocation layout.
   */
  std::optional<MalformedMetadataQuarantine> m_malformedMetadataQuarantine;

  std::optional<std::reference_wrapper<Qnn_Context_Graph_t>>
  findContext(const std::string &bin_path) {
    const std::lock_guard<std::mutex> lock(context_registry_mutex);
    return findContextLocked(bin_path);
  }

  std::optional<std::reference_wrapper<Qnn_Context_Graph_t>>
  findContextLocked(const std::string &bin_path) {
    if (m_hasSystemContextFreeFailure || m_hasContextQuarantine) {
      return std::nullopt;
    }

    auto mapIt = ct_map.find(bin_path);
    if (mapIt != ct_map.end() &&
        mapIt->second.m_state == QnnContextEntryState::ACTIVE) {
      return mapIt->second;
    }
    return std::nullopt;
  }

  void quarantineSystemContext(QnnSystemContext_Handle_t handle,
                               Qnn_ErrorHandle_t error,
                               std::shared_ptr<uint8_t> binary_buffer,
                               uint64_t binary_size) noexcept {
    m_hasSystemContextFreeFailure = true;
    if (!m_quarantinedSystemContext) {
      m_quarantinedSystemContext.emplace();
      auto &quarantine = *m_quarantinedSystemContext;
      quarantine.handle = handle;
      quarantine.error = error;
      quarantine.binary_buffer = std::move(binary_buffer);
      quarantine.binary_size = binary_size;
    } else {
      // A failed SystemContext free poisons makeContext(), so reaching this
      // branch would violate the registry's single-attempt invariant.
      ml_loge("QNN SystemContext quarantine slot is already occupied");
    }
    ml_loge("QNN SystemContext free failed: error=%" PRIu64
            ", public_error=%u, handle=%p, binary_size=%" PRIu64,
            static_cast<uint64_t>(error),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(error)), handle,
            binary_size);
  }

  static void
  releaseContextHostResources(Qnn_Context_Graph_t &context) noexcept {
    context.graph_map.clear();
    context.graph_idx.clear();

    if (context.m_graphsInfo != nullptr) {
      for (uint32_t i = 0; i < context.m_graphsCount; ++i) {
        auto *graph = context.m_graphsInfo[i];
        if (graph == nullptr) {
          continue;
        }
        free(graph->graphName);
        graph->graphName = nullptr;
        if (graph->inputTensors != nullptr) {
          qnn_wrapper_api::freeQnnTensors(graph->inputTensors,
                                          graph->numInputTensors);
        }
        if (graph->outputTensors != nullptr) {
          qnn_wrapper_api::freeQnnTensors(graph->outputTensors,
                                          graph->numOutputTensors);
        }
      }
      if (context.m_graphsCount > 0) {
        free(*context.m_graphsInfo);
      } else {
        ml_logw("QNN graph metadata pointer has no associated graph count");
      }
      free(context.m_graphsInfo);
      context.m_graphsInfo = nullptr;
    }
    context.m_graphsCount = 0;
    context.m_binaryBuffer.reset();
    context.m_binarySize = 0;
  }

  StatusCode
  rollbackContextCreationLocked(const std::string &bin_path) noexcept {
    auto it = ct_map.find(bin_path);
    if (it == ct_map.end() ||
        it->second.m_state != QnnContextEntryState::CREATING) {
      return StatusCode::FAILURE;
    }

    auto &context = it->second;
    if (context.m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextFree == nullptr) {
      context.m_state = QnnContextEntryState::QUARANTINED;
      m_hasContextQuarantine = true;
      ml_loge("Cannot roll back QNN context creation for: %s",
              bin_path.c_str());
      return StatusCode::FAILURE;
    }

    const auto free_status = m_qnnFunctionPointers.qnnInterface.contextFree(
      context.m_context, nullptr);
    if (QNN_CONTEXT_NO_ERROR != free_status) {
      context.m_state = QnnContextEntryState::QUARANTINED;
      context.m_lastError = free_status;
      m_hasContextQuarantine = true;
      ml_loge("Failed to roll back QNN context: error=%" PRIu64
              ", public_error=%u, binary=%s, context=%p",
              static_cast<uint64_t>(free_status),
              static_cast<unsigned int>(QNN_GET_ERROR_CODE(free_status)),
              bin_path.c_str(), static_cast<void *>(context.m_context));
      return StatusCode::FAILURE;
    }

    context.m_context = nullptr;
    releaseContextHostResources(context);
    ct_map.erase(it);
    return StatusCode::SUCCESS;
  }

  StatusCode freeContextLocked(const std::string &bin_path) {
    if (m_hasSystemContextFreeFailure || m_hasContextQuarantine) {
      ml_loge("QNN runtime is quarantined; refusing further context teardown");
      return StatusCode::FAILURE;
    }

    auto it = ct_map.find(bin_path);
    if (it == ct_map.end()) {
      ml_logw("Context not found for: %s", bin_path.c_str());
      return StatusCode::FAILURE;
    }

    auto &context = it->second;
    if (context.m_state != QnnContextEntryState::ACTIVE) {
      if (context.m_state == QnnContextEntryState::QUARANTINED) {
        m_hasContextQuarantine = true;
      }
      ml_loge("Refusing normal free of non-active QNN context: state=%u, "
              "binary=%s",
              static_cast<unsigned int>(context.m_state), bin_path.c_str());
      return StatusCode::FAILURE;
    }
    if (context.m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextFree == nullptr) {
      context.m_state = QnnContextEntryState::QUARANTINED;
      m_hasContextQuarantine = true;
      ml_loge("Cannot free QNN context for: %s", bin_path.c_str());
      return StatusCode::FAILURE;
    }

    if (RpcMem) {
      const size_t registration_count =
        RpcMem->registrationCount(context.m_context);
      if (registration_count > 0) {
        ml_loge("Refusing to free QNN context with live RPC registrations: "
                "binary=%s, context=%p, registrations=%zu",
                bin_path.c_str(), static_cast<void *>(context.m_context),
                registration_count);
        return StatusCode::FAILURE;
      }
    }

    const auto free_status = m_qnnFunctionPointers.qnnInterface.contextFree(
      context.m_context, nullptr);
    if (QNN_CONTEXT_NO_ERROR != free_status) {
      context.m_state = QnnContextEntryState::QUARANTINED;
      context.m_lastError = free_status;
      m_hasContextQuarantine = true;
      ml_loge("Failed to free QNN context: error=%" PRIu64
              ", public_error=%u, binary=%s, context=%p",
              static_cast<uint64_t>(free_status),
              static_cast<unsigned int>(QNN_GET_ERROR_CODE(free_status)),
              bin_path.c_str(), static_cast<void *>(context.m_context));
      return StatusCode::FAILURE;
    }

    context.m_context = nullptr;
    releaseContextHostResources(context);
    ct_map.erase(it);

    ml_logi("Freed QNN context for: %s", bin_path.c_str());
    return StatusCode::SUCCESS;
  }

  StatusCode freeContext(const std::string &bin_path) {
    const std::lock_guard<std::mutex> lock(context_registry_mutex);
    return freeContextLocked(bin_path);
  }

  StatusCode freeAllContexts() {
    const std::lock_guard<std::mutex> lock(context_registry_mutex);
    if (m_hasSystemContextFreeFailure || m_hasContextQuarantine) {
      return StatusCode::FAILURE;
    }

    std::vector<std::string> keys;
    keys.reserve(ct_map.size());
    for (auto &[k, _] : ct_map) {
      keys.push_back(k);
    }
    for (auto &k : keys) {
      if (freeContextLocked(k) != StatusCode::SUCCESS) {
        // Once a vendor free fails, the remaining runtime state is ambiguous.
        // Do not issue more teardown calls against the same backend.
        return StatusCode::FAILURE;
      }
    }
    return StatusCode::SUCCESS;
  }

  StatusCode makeContext(props::FilePath bin) {
    const std::lock_guard<std::mutex> lock(context_registry_mutex);
    const std::string bin_path = bin.get();

    if (m_hasSystemContextFreeFailure || m_hasContextQuarantine) {
      ml_loge("QNN runtime has quarantined state; refusing a new context");
      return StatusCode::FAILURE;
    }

    auto existing = ct_map.find(bin_path);
    if (existing != ct_map.end()) {
      if (existing->second.m_state == QnnContextEntryState::ACTIVE) {
        ml_logw("QNN context already exists for: %s", bin_path.c_str());
        return StatusCode::SUCCESS;
      }
      ml_loge("QNN context is not reusable: state=%u, binary=%s",
              static_cast<unsigned int>(existing->second.m_state),
              bin_path.c_str());
      return StatusCode::FAILURE;
    }
    if (m_malformedMetadataQuarantine) {
      ml_loge("QNN graph metadata is quarantined; refusing another context");
      return StatusCode::FAILURE;
    }

    if (m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ==
          nullptr ||
        m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ==
          nullptr ||
        m_qnnFunctionPointers.qnnSystemInterface.systemContextFree == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextFree == nullptr) {
      ml_loge("Required QNN context function pointers are not populated.");
      return StatusCode::FAILURE;
    }

    Qnn_Context_Graph_t candidate;
    bool entry_published = false;
    try {
      void *mapped_buffer = nullptr;
      size_t buffer_size = 0;
      if (!mmapBinaryFile(bin_path, &mapped_buffer, &buffer_size)) {
        return StatusCode::FAILURE;
      }
      candidate.m_binaryBuffer = std::shared_ptr<uint8_t>(
        static_cast<uint8_t *>(mapped_buffer), [buffer_size](uint8_t *ptr) {
          if (munmap(ptr, buffer_size) != 0) {
            ml_loge("Failed to unmap QNN context binary");
          }
        });
      candidate.m_binarySize = buffer_size;

      struct SystemContextGuard {
        SystemContextGuard(QNNVar &owner_,
                           std::shared_ptr<uint8_t> binary_buffer_,
                           uint64_t binary_size_) :
          owner(owner_),
          binary_buffer(std::move(binary_buffer_)),
          binary_size(binary_size_) {}

        SystemContextGuard(const SystemContextGuard &) = delete;
        SystemContextGuard &operator=(const SystemContextGuard &) = delete;

        QNNVar &owner;
        QnnSystemContext_Handle_t handle = nullptr;
        std::shared_ptr<uint8_t> binary_buffer;
        uint64_t binary_size = 0;

        Qnn_ErrorHandle_t close() noexcept {
          if (handle == nullptr) {
            return QNN_SUCCESS;
          }
          auto current = handle;
          handle = nullptr;
          const auto status =
            owner.m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(
              current);
          if (status != QNN_SUCCESS) {
            owner.quarantineSystemContext(
              current, status, std::move(binary_buffer), binary_size);
          }
          return status;
        }

        ~SystemContextGuard() { close(); }
      } system_context{*this, candidate.m_binaryBuffer, candidate.m_binarySize};

      auto system_status =
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
          &system_context.handle);
      if (system_status != QNN_SUCCESS) {
        ml_loge("Could not create QNN SystemContext: error=%" PRIu64
                ", public_error=%u",
                static_cast<uint64_t>(system_status),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(system_status)));
        system_context.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
      Qnn_ContextBinarySize_t binary_info_size = 0;
      system_status =
        m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
          system_context.handle, candidate.m_binaryBuffer.get(),
          candidate.m_binarySize, &binary_info, &binary_info_size);
      if (system_status != QNN_SUCCESS || binary_info == nullptr) {
        ml_loge("Failed to read QNN context metadata: error=%" PRIu64
                ", public_error=%u",
                static_cast<uint64_t>(system_status),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(system_status)));
        system_context.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      qnn_wrapper_api::GraphInfo_t **copied_graphs_info = nullptr;
      uint32_t copied_graph_count = 0;
      const bool metadata_copied =
        qnn::tools::sample_app::copyMetadataToGraphsInfo(
          binary_info, copied_graphs_info, copied_graph_count);
      if (!metadata_copied || copied_graphs_info == nullptr ||
          copied_graph_count == 0) {
        ml_loge("Failed to copy complete QNN context metadata");
        system_context.close();

        if (copied_graphs_info != nullptr && copied_graph_count > 0) {
          candidate.m_graphsInfo = copied_graphs_info;
          candidate.m_graphsCount = copied_graph_count;
        } else if (copied_graphs_info != nullptr) {
          // The SDK did not provide a safe traversal bound. Guessing its
          // allocation layout is more dangerous than retaining one bounded
          // failure-path allocation and its source mapping.
          m_malformedMetadataQuarantine.emplace();
          auto &quarantine = *m_malformedMetadataQuarantine;
          quarantine.graphs_info = copied_graphs_info;
          quarantine.binary_buffer = candidate.m_binaryBuffer;
          quarantine.binary_size = candidate.m_binarySize;
          ml_loge("Retaining malformed QNN graph metadata without a count");
        }
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      candidate.m_graphsInfo = copied_graphs_info;
      candidate.m_graphsCount = copied_graph_count;

      if (system_context.close() != QNN_SUCCESS) {
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      if (!candidate.setGraphInfoMap()) {
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      QnnContext_Config_t **custom_configs = nullptr;
      uint32_t custom_config_count = 0;
      auto *extension = m_backendExtensions == nullptr
                          ? nullptr
                          : m_backendExtensions->interface();
      if (extension != nullptr && !extension->beforeCreateFromBinary(
                                    &custom_configs, &custom_config_count)) {
        QNN_ERROR("Extensions Failure in beforeCreateFromBinary()");
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      if (custom_config_count > 0 && custom_configs == nullptr) {
        ml_loge("QNN extension returned an invalid context config list");
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      const size_t context_config_count = custom_config_count;
      if (context_config_count > std::numeric_limits<size_t>::max() - 2) {
        ml_loge("QNN extension returned too many context configs");
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      QnnHtpContext_CustomConfig_t io_mem_estimation{};
      io_mem_estimation.option = QnnHtpContext_ConfigOption_t::
        QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
      io_mem_estimation.ioMemEstimation = true;

      QnnContext_Config_t io_context_config = QNN_CONTEXT_CONFIG_INIT;
      io_context_config.option =
        QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      io_context_config.customConfig =
        reinterpret_cast<QnnContext_CustomConfig_t>(&io_mem_estimation);

      std::vector<const QnnContext_Config_t *> context_configs;
      context_configs.reserve(context_config_count + 2);
      context_configs.push_back(&io_context_config);
      for (uint32_t i = 0; i < custom_config_count; ++i) {
        if (custom_configs[i] == nullptr) {
          ml_loge("QNN extension returned a null context config at index %u",
                  i);
          releaseContextHostResources(candidate);
          return StatusCode::FAILURE;
        }
        context_configs.push_back(custom_configs[i]);
      }
      context_configs.push_back(nullptr);

      auto inserted = ct_map.try_emplace(bin_path, std::move(candidate));
      if (!inserted.second) {
        ml_loge("Failed to reserve QNN context entry for: %s",
                bin_path.c_str());
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      entry_published = true;
      auto &entry = inserted.first->second;
      candidate.m_context = nullptr;
      candidate.m_graphsInfo = nullptr;
      candidate.m_graphsCount = 0;
      candidate.m_binarySize = 0;

      const auto create_status =
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle, m_deviceHandle, context_configs.data(),
          entry.m_binaryBuffer.get(), entry.m_binarySize, &entry.m_context,
          m_profileBackendHandle);
      if (create_status != QNN_CONTEXT_NO_ERROR) {
        entry.m_lastError = create_status;
        ml_loge("Could not create QNN context: error=%" PRIu64
                ", public_error=%u, binary=%s, context=%p",
                static_cast<uint64_t>(create_status),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(create_status)),
                bin_path.c_str(), static_cast<void *>(entry.m_context));
        if (entry.m_context != nullptr) {
          entry.m_state = QnnContextEntryState::QUARANTINED;
          m_hasContextQuarantine = true;
        } else {
          releaseContextHostResources(entry);
          ct_map.erase(inserted.first);
        }
        return StatusCode::FAILURE;
      }
      if (entry.m_context == nullptr) {
        ml_loge("QNN context creation succeeded without returning a handle");
        entry.m_state = QnnContextEntryState::QUARANTINED;
        m_hasContextQuarantine = true;
        return StatusCode::FAILURE;
      }

      if (extension != nullptr && !extension->afterCreateFromBinary()) {
        QNN_ERROR("Extensions Failure in afterCreateFromBinary()");
        rollbackContextCreationLocked(bin_path);
        return StatusCode::FAILURE;
      }

      if (sample_app::ProfilingLevel::OFF != m_profilingLevel &&
          extractBackendProfilingInfo() != StatusCode::SUCCESS) {
        ml_logw("QNN context is usable, but profiling data is unavailable");
      }
      entry.m_state = QnnContextEntryState::ACTIVE;
      return StatusCode::SUCCESS;
    } catch (const std::exception &e) {
      ml_loge("Exception while creating QNN context for %s: %s",
              bin_path.c_str(), e.what());
    } catch (...) {
      ml_loge("Unknown exception while creating QNN context for %s",
              bin_path.c_str());
    }

    if (entry_published) {
      auto entry = ct_map.find(bin_path);
      if (entry != ct_map.end() && entry->second.m_context != nullptr) {
        rollbackContextCreationLocked(bin_path);
      } else if (entry != ct_map.end()) {
        releaseContextHostResources(entry->second);
        ct_map.erase(entry);
      }
    } else {
      releaseContextHostResources(candidate);
    }
    return StatusCode::FAILURE;
  }

  qnn_wrapper_api::GraphInfo_t *graphRetrieve(std::string bin_path,
                                              std::string graphName) {
    const std::lock_guard<std::mutex> lock(context_registry_mutex);

    std::optional<std::reference_wrapper<Qnn_Context_Graph_t>> op =
      findContextLocked(bin_path);

    if (!op) {
      ml_loge("Cannot find context");
      return nullptr;
    }

    Qnn_Context_Graph_t &context_i = *op;

    qnn_wrapper_api::GraphInfo_t *graphInfo = context_i.getGraphPtr(graphName);

    if (nullptr == graphInfo) {
      ml_loge("cannot find graph for graph name : %s", graphName.c_str());
      return nullptr;
    }

    if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
      ml_loge("graphRetrieveFnHandle is nullptr.");
      return nullptr;
    }

    if (graphInfo->graph != nullptr) {
      return graphInfo;
    }

    /**
     * QNN's graphRetrieve is case-sensitive, so pass the binary's real graph
     * name (graphInfo->graphName) rather than the possibly-lowercased lookup
     * key in graphName.
     */
    decltype(graphInfo->graph) retrieved_graph = nullptr;
    const auto retrieve_status =
      m_qnnFunctionPointers.qnnInterface.graphRetrieve(
        context_i.m_context, graphInfo->graphName, &retrieved_graph);
    if (retrieve_status != QNN_SUCCESS || retrieved_graph == nullptr) {
      ml_loge("Unable to retrieve graph handle for graph name : %s",
              graphInfo->graphName);
      return nullptr;
    }

    graphInfo->graph = retrieved_graph;
    return graphInfo;
  }

  StatusCode extractBackendProfilingInfo() {
    Qnn_ProfileHandle_t profileHandle = m_profileBackendHandle;

    if (nullptr == m_profileBackendHandle) {
      ml_loge("Backend Profile handle is nullptr; may not be initialized.");
      return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEvents(
          profileHandle, &profileEvents, &numEvents)) {
      ml_loge("Failure in profile get events.");
      return StatusCode::FAILURE;
    }
    ml_loge("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
    for (size_t event = 0; event < numEvents; event++) {
      extractProfilingEvent(*(profileEvents + event));
      extractProfilingSubEvents(*(profileEvents + event));
    }
    return StatusCode::SUCCESS;
  }

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId) {
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
          profileEventId, &profileSubEvents, &numSubEvents)) {
      ml_loge("Failure in profile get sub events.");
      return StatusCode::FAILURE;
    }
    ml_logd("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents,
            numSubEvents);
    for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
      extractProfilingEvent(*(profileSubEvents + subEvent));
      extractProfilingSubEvents(*(profileSubEvents + subEvent));
    }
    return StatusCode::SUCCESS;
  }

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId) {
    QnnProfile_EventData_t eventData;
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId,
                                                               &eventData)) {
      ml_loge("Failure in profile get event type.");
      return StatusCode::FAILURE;
    }
    ml_logd("Printing Event Info - Event Type: [%d], Event Value: [%lu], Event "
            "Identifier: [%s], Event Unit: [%d]",
            eventData.type, eventData.value, eventData.identifier,
            eventData.unit);
    return StatusCode::SUCCESS;
  }

  bool mmapBinaryFile(const std::string &file_path, void **buffer,
                      size_t *buffer_size) {
    if (buffer == nullptr || buffer_size == nullptr) {
      ml_loge("Invalid mmap request for QNN context binary");
      return false;
    }
    *buffer = nullptr;
    *buffer_size = 0;

    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd < 0) {
      ml_loge("Failed to open QNN context binary %s: %s", file_path.c_str(),
              strerror(errno));
      return false;
    }

    struct stat file_info {};
    if (fstat(fd, &file_info) != 0) {
      const int stat_error = errno;
      if (close(fd) != 0) {
        ml_logw("Failed to close QNN context binary %s: %s", file_path.c_str(),
                strerror(errno));
      }
      ml_loge("Failed to stat QNN context binary %s: %s", file_path.c_str(),
              strerror(stat_error));
      return false;
    }
    if (file_info.st_size <= 0 ||
        static_cast<uint64_t>(file_info.st_size) >
          std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(file_info.st_size) >
          std::numeric_limits<Qnn_ContextBinarySize_t>::max()) {
      if (close(fd) != 0) {
        ml_logw("Failed to close QNN context binary %s: %s", file_path.c_str(),
                strerror(errno));
      }
      ml_loge("Invalid QNN context binary size for: %s", file_path.c_str());
      return false;
    }

    const size_t mapped_size = static_cast<size_t>(file_info.st_size);
    void *mapped = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mmap_error = errno;
    if (close(fd) != 0) {
      ml_logw("Failed to close QNN context binary %s: %s", file_path.c_str(),
              strerror(errno));
    }
    if (mapped == MAP_FAILED) {
      ml_loge("Failed to mmap QNN context binary %s: %s", file_path.c_str(),
              strerror(mmap_error));
      return false;
    }

    *buffer = mapped;
    *buffer_size = mapped_size;
    if (madvise(mapped, mapped_size, MADV_NOHUGEPAGE) != 0) {
      ml_loge("Failed to advise OS on memory usage err: %s", strerror(errno));
    }
    return true;
  }
};

class QNNBackendVar : public ContextData {
public:
  QNNBackendVar() : data(std::make_shared<QNNVar>()) {}
  std::shared_ptr<QNNVar> &getVar() { return data; }

private:
  std::shared_ptr<QNNVar> data;
};
} // namespace nntrainer
#endif /* __QNN_CONTEXT_VAR_H__ */
