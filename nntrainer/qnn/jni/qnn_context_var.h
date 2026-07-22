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
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <inttypes.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
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
  Qnn_ContextHandle_t m_context{nullptr};
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo{nullptr};
  std::map<std::string, qnn_wrapper_api::GraphInfo_t *>
    graph_map; /** graph name in Context - graph map **/
  std::map<std::string, uint32_t>
    graph_idx; /** graph name in Context - graph map **/

  uint32_t m_graphsCount{0};
  QnnContextEntryState m_state{QnnContextEntryState::CREATING};
  Qnn_ErrorHandle_t m_lastError{QNN_CONTEXT_NO_ERROR};
  std::shared_ptr<uint8_t> m_binaryBuffer{};
  uint64_t m_binarySize{0};

  bool setGraphInfoMap() {
    graph_map.clear();
    if (m_graphsInfo == nullptr || m_graphsCount == 0) {
      ml_loge("QNN context metadata has no graphs");
      return false;
    }

    for (uint32_t i = 0; i < m_graphsCount; ++i) {
      auto *graph = m_graphsInfo[i];
      if (graph == nullptr || graph->graphName == nullptr ||
          (graph->numInputTensors > 0 && graph->inputTensors == nullptr) ||
          (graph->numOutputTensors > 0 && graph->outputTensors == nullptr)) {
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
  iotensor::OutputDataType m_outputDataType;
  iotensor::InputDataType m_inputDataType;
  sample_app::ProfilingLevel m_profilingLevel;
  bool m_isBackendInitialized;
  void *m_backendLibraryHandle = nullptr;
  Qnn_LogHandle_t m_logHandle = nullptr;
  Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
  sample_app::QnnFunctionPointers m_qnnFunctionPointers;
  std::shared_ptr<QNNRpcManager> RpcMem;
  IOTensorWrapper m_ioTensor;
  std::string name = "qnn_backend_param";
  std::map<std::string, Qnn_Context_Graph_t>
    ct_map; /** bin file name - Context map **/
  bool m_hasSystemContextFreeFailure{false};
  bool m_hasContextQuarantine{false};
  std::vector<QnnSystemContext_Handle_t> m_quarantinedSystemContexts;

  std::optional<std::reference_wrapper<Qnn_Context_Graph_t>>
  findContext(std::string bin_path) {
    auto mapIt = ct_map.find(bin_path);
    if (mapIt != ct_map.end() &&
        mapIt->second.m_state == QnnContextEntryState::ACTIVE) {
      return mapIt->second;
    }
    return std::nullopt;
  }

  void quarantineSystemContext(QnnSystemContext_Handle_t handle,
                               Qnn_ErrorHandle_t error) noexcept {
    m_hasSystemContextFreeFailure = true;
    try {
      m_quarantinedSystemContexts.push_back(handle);
    } catch (...) {
      ml_loge("Failed to retain quarantined QNN SystemContext handle");
    }
    ml_loge("QNN SystemContext free failed: error=%" PRIu64
            ", public_error=%u, handle=%p",
            static_cast<uint64_t>(error),
            static_cast<unsigned int>(QNN_GET_ERROR_CODE(error)), handle);
  }

  static void releaseContextHostResources(Qnn_Context_Graph_t &ctx) noexcept {
    ctx.graph_map.clear();
    ctx.graph_idx.clear();

    if (ctx.m_graphsInfo != nullptr) {
      for (uint32_t i = 0; i < ctx.m_graphsCount; i++) {
        auto *graph = ctx.m_graphsInfo[i];
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
      if (ctx.m_graphsCount > 0) {
        free(*ctx.m_graphsInfo);
      } else {
        ml_logw("QNN graph metadata pointer has no associated graph count");
      }
      free(ctx.m_graphsInfo);
      ctx.m_graphsInfo = nullptr;
    }
    ctx.m_graphsCount = 0;
    ctx.m_binaryBuffer.reset();
    ctx.m_binarySize = 0;
  }

  static uint32_t getBinaryGraphCount(
    const QnnSystemContext_BinaryInfo_t *binaryInfo) noexcept {
    if (binaryInfo == nullptr) {
      return 0;
    }
    switch (binaryInfo->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      return binaryInfo->contextBinaryInfoV1.numGraphs;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      return binaryInfo->contextBinaryInfoV2.numGraphs;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      return binaryInfo->contextBinaryInfoV3.numGraphs;
    default:
      return 0;
    }
  }

  StatusCode rollbackContextCreation(const std::string &bin_path) noexcept {
    auto it = ct_map.find(bin_path);
    if (it == ct_map.end() ||
        it->second.m_state != QnnContextEntryState::CREATING) {
      return StatusCode::FAILURE;
    }

    auto &ctx = it->second;
    if (ctx.m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextFree == nullptr) {
      ctx.m_state = QnnContextEntryState::QUARANTINED;
      m_hasContextQuarantine = true;
      ml_loge("Cannot roll back QNN context creation for: %s",
              bin_path.c_str());
      return StatusCode::FAILURE;
    }

    const auto freeStatus =
      m_qnnFunctionPointers.qnnInterface.contextFree(ctx.m_context, nullptr);
    if (QNN_CONTEXT_NO_ERROR != freeStatus) {
      ctx.m_state = QnnContextEntryState::QUARANTINED;
      ctx.m_lastError = freeStatus;
      m_hasContextQuarantine = true;
      ml_loge("Failed to roll back QNN context: error=%" PRIu64
              ", public_error=%u, binary=%s, context=%p",
              static_cast<uint64_t>(freeStatus),
              static_cast<unsigned int>(QNN_GET_ERROR_CODE(freeStatus)),
              bin_path.c_str(), static_cast<void *>(ctx.m_context));
      return StatusCode::FAILURE;
    }

    ctx.m_context = nullptr;
    releaseContextHostResources(ctx);
    ct_map.erase(it);
    return StatusCode::SUCCESS;
  }

  StatusCode freeContextWithCleanupGuard(
    const std::string &bin_path,
    const QNNRpcManager::CleanupGuard &cleanup_guard) {
    if (RpcMem && !RpcMem->ownsCleanupGuard(cleanup_guard)) {
      ml_loge("QNN context cleanup requires an exclusive lifecycle guard");
      return StatusCode::FAILURE;
    }

    auto it = ct_map.find(bin_path);
    if (it == ct_map.end()) {
      ml_logw("Context not found for: %s", bin_path.c_str());
      return StatusCode::FAILURE;
    }

    auto &ctx = it->second;

    if (ctx.m_state != QnnContextEntryState::ACTIVE) {
      if (ctx.m_state == QnnContextEntryState::QUARANTINED) {
        m_hasContextQuarantine = true;
      }
      ml_loge("Refusing normal free of non-active QNN context: state=%u, "
              "binary=%s",
              static_cast<unsigned int>(ctx.m_state), bin_path.c_str());
      return StatusCode::FAILURE;
    }
    if (ctx.m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextFree == nullptr) {
      ctx.m_state = QnnContextEntryState::QUARANTINED;
      m_hasContextQuarantine = true;
      ml_loge("Cannot free QNN context for: %s", bin_path.c_str());
      return StatusCode::FAILURE;
    }

    if (RpcMem) {
      const size_t registration_count =
        RpcMem->registrationCount(ctx.m_context);
      if (registration_count > 0) {
        ml_loge("Cannot free QNN context with live registrations: binary=%s, "
                "context=%p, registrations=%zu",
                bin_path.c_str(), static_cast<void *>(ctx.m_context),
                registration_count);
        return StatusCode::FAILURE;
      }
    }

    const auto freeStatus =
      m_qnnFunctionPointers.qnnInterface.contextFree(ctx.m_context, nullptr);
    if (QNN_CONTEXT_NO_ERROR != freeStatus) {
      ctx.m_state = QnnContextEntryState::QUARANTINED;
      ctx.m_lastError = freeStatus;
      m_hasContextQuarantine = true;
      ml_loge("Failed to free QNN context: error=%" PRIu64
              ", public_error=%u, binary=%s, context=%p",
              static_cast<uint64_t>(freeStatus),
              static_cast<unsigned int>(QNN_GET_ERROR_CODE(freeStatus)),
              bin_path.c_str(), static_cast<void *>(ctx.m_context));
      return StatusCode::FAILURE;
    }

    ctx.m_context = nullptr;
    releaseContextHostResources(ctx);
    ct_map.erase(it);

    ml_logi("Freed QNN context for: %s", bin_path.c_str());
    return StatusCode::SUCCESS;
  }

  StatusCode freeContext(const std::string &bin_path) {
    QNNRpcManager::CleanupGuard cleanup_guard{};
    try {
      if (RpcMem) {
        cleanup_guard = RpcMem->acquireCleanupGuard();
      }
    } catch (const std::exception &e) {
      ml_loge("Cannot free QNN context after runtime shutdown: %s", e.what());
      return StatusCode::FAILURE;
    } catch (...) {
      ml_loge("Cannot free QNN context after runtime shutdown");
      return StatusCode::FAILURE;
    }
    return freeContextWithCleanupGuard(bin_path, cleanup_guard);
  }

  StatusCode freeAllContextsWithCleanupGuard(
    const QNNRpcManager::CleanupGuard &cleanup_guard) {
    if (RpcMem && !RpcMem->ownsCleanupGuard(cleanup_guard)) {
      ml_loge("QNN runtime cleanup requires an exclusive lifecycle guard");
      return StatusCode::FAILURE;
    }

    // Copy the keys because successful cleanup erases entries from ct_map.
    std::vector<std::string> keys;
    keys.reserve(ct_map.size());
    for (auto &[k, _] : ct_map) {
      keys.push_back(k);
    }
    auto returnStatus =
      (m_hasSystemContextFreeFailure || m_hasContextQuarantine)
        ? StatusCode::FAILURE
        : StatusCode::SUCCESS;
    for (auto &k : keys) {
      if (freeContextWithCleanupGuard(k, cleanup_guard) !=
          StatusCode::SUCCESS) {
        returnStatus = StatusCode::FAILURE;
      }
    }
    if (RpcMem && RpcMem->registrationCount() > 0) {
      ml_loge("Cannot finish QNN runtime teardown with live registrations");
      returnStatus = StatusCode::FAILURE;
    }
    return returnStatus;
  }

  StatusCode freeAllContexts() {
    QNNRpcManager::CleanupGuard cleanup_guard{};
    try {
      if (RpcMem) {
        cleanup_guard = RpcMem->acquireCleanupGuard();
      }
    } catch (const std::exception &e) {
      ml_loge("Cannot free QNN contexts after runtime shutdown: %s", e.what());
      return StatusCode::FAILURE;
    } catch (...) {
      ml_loge("Cannot free QNN contexts after runtime shutdown");
      return StatusCode::FAILURE;
    }
    return freeAllContextsWithCleanupGuard(cleanup_guard);
  }

  StatusCode makeContext(props::FilePath bin) {
    const std::string binPath = bin.get();
    auto existing = ct_map.find(binPath);
    if (existing != ct_map.end()) {
      if (existing->second.m_state == QnnContextEntryState::ACTIVE) {
        ml_logw("QNN context already exists for: %s", binPath.c_str());
        return StatusCode::SUCCESS;
      }
      ml_loge("QNN context is not reusable: state=%u, binary=%s",
              static_cast<unsigned int>(existing->second.m_state),
              binPath.c_str());
      return StatusCode::FAILURE;
    }
    if (m_hasSystemContextFreeFailure || m_hasContextQuarantine) {
      ml_loge("QNN runtime has quarantined state; refusing a new context");
      return StatusCode::FAILURE;
    }

    if (nullptr ==
          m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr ==
          m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree ||
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary ||
        nullptr == m_qnnFunctionPointers.qnnInterface.contextFree) {
      ml_loge("Required QNN context function pointers are not populated.");
      return StatusCode::FAILURE;
    }

    const std::streamoff binarySize = bin.file_size();
    if (binarySize <= 0 || static_cast<uint64_t>(binarySize) >
                             std::numeric_limits<size_t>::max()) {
      ml_loge("Invalid QNN context binary size for: %s", binPath.c_str());
      return StatusCode::FAILURE;
    }
    const size_t bufferSize = static_cast<size_t>(binarySize);

    void *mappedBuffer = nullptr;
    if (!mmapBinaryFile(binPath, &mappedBuffer, bufferSize)) {
      return StatusCode::FAILURE;
    }

    Qnn_Context_Graph_t candidate{};
    candidate.m_binaryBuffer = std::shared_ptr<uint8_t>(
      static_cast<uint8_t *>(mappedBuffer), [bufferSize](uint8_t *ptr) {
        if (munmap(ptr, bufferSize)) {
          ml_loge("Failed to unmap QNN context binary");
        }
      });
    candidate.m_binarySize = bufferSize;

    struct SystemContextGuard {
      QNNVar &owner;
      QnnSystemContext_Handle_t handle{nullptr};

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
          owner.quarantineSystemContext(current, status);
        }
        return status;
      }

      ~SystemContextGuard() { close(); }
    } systemContext{*this};

    bool entryPublished = false;
    try {
      auto systemStatus =
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
          &systemContext.handle);
      if (systemStatus != QNN_SUCCESS) {
        ml_loge("Could not create QNN SystemContext: error=%" PRIu64
                ", public_error=%u",
                static_cast<uint64_t>(systemStatus),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(systemStatus)));
        systemContext.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
      Qnn_ContextBinarySize_t binaryInfoSize{0};
      systemStatus =
        m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
          systemContext.handle, candidate.m_binaryBuffer.get(),
          candidate.m_binarySize, &binaryInfo, &binaryInfoSize);
      if (systemStatus != QNN_SUCCESS || binaryInfo == nullptr) {
        ml_loge("Failed to read QNN context metadata: error=%" PRIu64
                ", public_error=%u",
                static_cast<uint64_t>(systemStatus),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(systemStatus)));
        systemContext.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      const uint32_t expectedGraphCount = getBinaryGraphCount(binaryInfo);
      if (expectedGraphCount == 0) {
        ml_loge("QNN context metadata contains no supported graphs");
        systemContext.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      // The SDK sample helper allocates an expected-count, zero-initialized
      // pointer array. Publish that count before copying so an exception or a
      // partially populated output can still be unwound completely.
      candidate.m_graphsCount = expectedGraphCount;
      uint32_t copiedGraphCount{0};
      const bool metadataCopied =
        qnn::tools::sample_app::copyMetadataToGraphsInfo(
          binaryInfo, candidate.m_graphsInfo, copiedGraphCount);
      if (!metadataCopied || candidate.m_graphsInfo == nullptr ||
          copiedGraphCount != expectedGraphCount) {
        ml_loge("Failed to copy complete QNN context metadata");
        systemContext.close();
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      if (systemContext.close() != QNN_SUCCESS) {
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      if (!candidate.setGraphInfoMap()) {
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      QnnContext_Config_t **customConfigs{nullptr};
      uint32_t customConfigCount{0};
      auto *extension = m_backendExtensions == nullptr
                          ? nullptr
                          : m_backendExtensions->interface();
      if (extension != nullptr && !extension->beforeCreateFromBinary(
                                    &customConfigs, &customConfigCount)) {
        QNN_ERROR("Extensions Failure in beforeCreateFromBinary()");
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      if (customConfigCount > 0 && customConfigs == nullptr) {
        ml_loge("QNN extension returned an invalid context config list");
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }

      QnnHtpContext_CustomConfig_t ioMemEstimation{};
      ioMemEstimation.option = QnnHtpContext_ConfigOption_t::
        QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
      ioMemEstimation.ioMemEstimation = true;

      QnnContext_Config_t ioContextConfig = QNN_CONTEXT_CONFIG_INIT;
      ioContextConfig.option =
        QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      ioContextConfig.customConfig =
        reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);

      std::vector<const QnnContext_Config_t *> contextConfigs;
      contextConfigs.reserve(customConfigCount + 2);
      contextConfigs.push_back(&ioContextConfig);
      for (uint32_t i = 0; i < customConfigCount; ++i) {
        if (customConfigs[i] == nullptr) {
          ml_loge("QNN extension returned a null context config at index %u",
                  i);
          releaseContextHostResources(candidate);
          return StatusCode::FAILURE;
        }
        contextConfigs.push_back(customConfigs[i]);
      }
      contextConfigs.push_back(nullptr);

      auto inserted = ct_map.try_emplace(binPath, std::move(candidate));
      if (!inserted.second) {
        ml_loge("Failed to reserve QNN context entry for: %s", binPath.c_str());
        releaseContextHostResources(candidate);
        return StatusCode::FAILURE;
      }
      entryPublished = true;
      auto &entry = inserted.first->second;
      // The generated move operation transfers STL owners but copies raw
      // pointer/scalar members. Neutralize the source so ownership remains
      // unambiguous if this struct later gains automatic cleanup.
      candidate.m_context = nullptr;
      candidate.m_graphsInfo = nullptr;
      candidate.m_graphsCount = 0;
      candidate.m_binarySize = 0;

      const auto createStatus =
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle, m_deviceHandle, contextConfigs.data(),
          entry.m_binaryBuffer.get(), entry.m_binarySize, &entry.m_context,
          m_profileBackendHandle);
      if (createStatus != QNN_CONTEXT_NO_ERROR) {
        entry.m_lastError = createStatus;
        ml_loge("Could not create QNN context: error=%" PRIu64
                ", public_error=%u, binary=%s, context=%p",
                static_cast<uint64_t>(createStatus),
                static_cast<unsigned int>(QNN_GET_ERROR_CODE(createStatus)),
                binPath.c_str(), static_cast<void *>(entry.m_context));
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
        rollbackContextCreation(binPath);
        return StatusCode::FAILURE;
      }

      if (sample_app::ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo();
      }
      entry.m_state = QnnContextEntryState::ACTIVE;
      return StatusCode::SUCCESS;
    } catch (const std::exception &e) {
      ml_loge("Exception while creating QNN context for %s: %s",
              binPath.c_str(), e.what());
    } catch (...) {
      ml_loge("Unknown exception while creating QNN context for %s",
              binPath.c_str());
    }

    systemContext.close();
    if (entryPublished) {
      auto entry = ct_map.find(binPath);
      if (entry != ct_map.end() && entry->second.m_context != nullptr) {
        rollbackContextCreation(binPath);
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

    std::optional<std::reference_wrapper<Qnn_Context_Graph_t>> op =
      findContext(bin_path);

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

    /**
     * QNN's graphRetrieve is case-sensitive, so pass the binary's real graph
     * name (graphInfo->graphName) rather than the possibly-lowercased lookup
     * key in graphName.
     */
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.graphRetrieve(
          context_i.m_context, graphInfo->graphName, &(graphInfo->graph))) {
      ml_loge("Unable to retrieve graph handle for graph name : %s",
              graphInfo->graphName);
      return nullptr;
    }

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

  bool mmapBinaryFile(const std::string &filePath, void **buffer,
                      size_t bufferSize) {
    if (buffer == nullptr || bufferSize == 0) {
      ml_loge("Invalid mmap request for QNN context binary");
      return false;
    }
    *buffer = nullptr;

    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0) {
      ml_loge("Failed to open QNN context binary %s: %s", filePath.c_str(),
              strerror(errno));
      return false;
    }

    void *mapped = mmap(nullptr, bufferSize, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mmapError = errno;
    if (close(fd) != 0) {
      ml_logw("Failed to close QNN context binary %s: %s", filePath.c_str(),
              strerror(errno));
    }
    if (mapped == MAP_FAILED) {
      ml_loge("Failed to mmap QNN context binary %s: %s", filePath.c_str(),
              strerror(mmapError));
      return false;
    }

    *buffer = mapped;
    if (madvise(mapped, bufferSize, MADV_NOHUGEPAGE)) {
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
