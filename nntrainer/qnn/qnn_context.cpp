// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_context.h
 * @date    10 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn context related functions and classes that
 * manages the global configuration of the current QNN environment.
 */

#include "Log/Logger.hpp"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QnnTypes.h"
#include "Utils/BuildId.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "iotensor_wrapper.hpp"

#include "BackendExtensions.hpp"

#include "qnn_context.h"
#include <QNNGraph.h>
#include <cstdlib>
#include <inttypes.h>
#include <iostream>
#include <limits.h>
#include <limits>
#include <unistd.h>
#include <utility>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QNNContext"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

namespace nntrainer {

static std::string g_default_backend_ext_config_path;

static std::string trim_trailing_slashes(std::string path) {
  while (path.size() > 1 && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

static bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

static std::string resolve_quick_dot_ai_base_dir() {
  const char *override_base_dir = std::getenv("QUICK_DOT_AI_BASE_DIR");
  if (override_base_dir != nullptr && override_base_dir[0] != '\0') {
    std::string resolved = trim_trailing_slashes(override_base_dir);
    LOGD("resolve_quick_dot_ai_base_dir: using QUICK_DOT_AI_BASE_DIR=%s",
         resolved.c_str());
    return resolved;
  }

  char cwd[PATH_MAX] = {
    0,
  };
  if (getcwd(cwd, sizeof(cwd)) != nullptr && cwd[0] != '\0') {
    std::string resolved = trim_trailing_slashes(cwd);
    LOGD("resolve_quick_dot_ai_base_dir: using cwd=%s", resolved.c_str());
    return resolved;
  }

  std::string fallback = "/sdcard/Download/aistudio-mobile/";
  LOGD("resolve_quick_dot_ai_base_dir: using fallback=%s", fallback.c_str());
  return fallback;
}

static std::string
resolve_backend_extensions_config_value(const std::string &path) {
  if (path.empty() || is_absolute_path(path)) {
    return path;
  }
  return resolve_quick_dot_ai_base_dir() + "/" + path;
}

static std::string resolve_backend_extensions_config_path() {
  const char *override_config_path =
    std::getenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
  if (override_config_path != nullptr && override_config_path[0] != '\0') {
    std::string config_path =
      resolve_backend_extensions_config_value(override_config_path);
    LOGD("resolve_backend_extensions_config_path: using "
         "QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH=%s",
         config_path.c_str());
    return config_path;
  }

  std::string config_path =
    resolve_quick_dot_ai_base_dir() + "/htp_backend_ext_config.json";
  LOGD("resolve_backend_extensions_config_path: %s", config_path.c_str());
  return config_path;
}

void QNNContext::setDefaultBackendExtConfigPath(const std::string &path) {
  g_default_backend_ext_config_path = path;
  LOGD("setDefaultBackendExtConfigPath: %s", path.c_str());
}

std::mutex qnn_factory_mutex;

static uint32_t publicQnnError(Qnn_ErrorHandle_t status) noexcept {
  return static_cast<uint32_t>(QNN_GET_ERROR_CODE(status));
}

static bool qnnCallSucceeded(Qnn_ErrorHandle_t status) noexcept {
  return publicQnnError(status) == QNN_SUCCESS;
}

static bool resourceStateMatchesHandle(QnnResourceState state,
                                       const void *handle) noexcept {
  if (state == QnnResourceState::OWNED) {
    return handle != nullptr;
  }
  if (state == QnnResourceState::QUARANTINED) {
    return false;
  }
  return handle == nullptr;
}

static void
quarantineRuntime(const std::shared_ptr<QNNVar> &qnn_data,
                  const std::shared_ptr<QNNRuntimeLifecycle> &runtime_lifecycle,
                  const QNNRuntimeLifecycle::CleanupGuard *shutdown_guard,
                  const char *reason) noexcept {
  qnn_data->m_hasRuntimeResourceQuarantine = true;
  qnn_data->quarantine_self_reference = qnn_data;
  if (shutdown_guard != nullptr && shutdown_guard->owns_lock()) {
    runtime_lifecycle->finishRuntimeShutdown(*shutdown_guard,
                                             QnnRuntimeState::QUARANTINED);
  }
  ml_loge("QNN runtime teardown quarantined: %s", reason);
}

static bool
releaseProfileResource(const std::shared_ptr<QNNVar> &qnn_data) noexcept {
  if (qnn_data->m_profileState != QnnResourceState::OWNED) {
    return true;
  }
  if (qnn_data->m_profileBackendHandle == nullptr ||
      qnn_data->m_qnnFunctionPointers.qnnInterface.profileFree == nullptr) {
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    ml_loge("Cannot release owned QNN profile resource");
    return false;
  }

  Qnn_ErrorHandle_t status = QNN_PROFILE_NO_ERROR;
  try {
    status = qnn_data->m_qnnFunctionPointers.qnnInterface.profileFree(
      qnn_data->m_profileBackendHandle);
  } catch (const std::exception &e) {
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    ml_loge("QNN profileFree threw: %s", e.what());
    return false;
  } catch (...) {
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    ml_loge("QNN profileFree threw");
    return false;
  }
  if (!qnnCallSucceeded(status)) {
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    ml_loge("QNN profileFree failed: error=%" PRIu64 ", public_error=%u",
            static_cast<uint64_t>(status), publicQnnError(status));
    return false;
  }

  qnn_data->m_profileBackendHandle = nullptr;
  qnn_data->m_profileState = QnnResourceState::RELEASED;
  return true;
}

static bool
releaseDeviceResource(const std::shared_ptr<QNNVar> &qnn_data) noexcept {
  if (!qnn_data->m_deviceLifecycleInitialized) {
    return true;
  }
  if (qnn_data->m_deviceState != QnnResourceState::OWNED &&
      qnn_data->m_deviceState != QnnResourceState::UNSUPPORTED) {
    qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
    ml_loge("QNN device lifecycle has an invalid ownership state");
    return false;
  }

  auto *extension = qnn_data->m_backendExtensions == nullptr
                      ? nullptr
                      : qnn_data->m_backendExtensions->interface();
  try {
    if (extension != nullptr && !extension->beforeFreeDevice()) {
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      ml_loge("QNN extension rejected device teardown");
      return false;
    }
  } catch (const std::exception &e) {
    qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
    ml_loge("QNN extension threw before device teardown: %s", e.what());
    return false;
  } catch (...) {
    qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
    ml_loge("QNN extension threw before device teardown");
    return false;
  }

  if (qnn_data->m_deviceState == QnnResourceState::OWNED) {
    if (qnn_data->m_deviceHandle == nullptr ||
        qnn_data->m_qnnFunctionPointers.qnnInterface.deviceFree == nullptr) {
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      ml_loge("Cannot release owned QNN device resource");
      return false;
    }

    Qnn_ErrorHandle_t status = QNN_SUCCESS;
    try {
      status = qnn_data->m_qnnFunctionPointers.qnnInterface.deviceFree(
        qnn_data->m_deviceHandle);
    } catch (const std::exception &e) {
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      ml_loge("QNN deviceFree threw: %s", e.what());
      return false;
    } catch (...) {
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      ml_loge("QNN deviceFree threw");
      return false;
    }
    if (!qnnCallSucceeded(status)) {
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      ml_loge("QNN deviceFree failed: error=%" PRIu64 ", public_error=%u",
              static_cast<uint64_t>(status), publicQnnError(status));
      return false;
    }
    qnn_data->m_deviceHandle = nullptr;
  }

  qnn_data->m_deviceState = QnnResourceState::RELEASED;
  try {
    if (extension != nullptr && !extension->afterFreeDevice()) {
      ml_loge("QNN extension failed after device teardown");
      return false;
    }
  } catch (const std::exception &e) {
    ml_loge("QNN extension threw after device teardown: %s", e.what());
    return false;
  } catch (...) {
    ml_loge("QNN extension threw after device teardown");
    return false;
  }
  qnn_data->m_deviceLifecycleInitialized = false;
  return true;
}

static bool
releaseBackendResource(const std::shared_ptr<QNNVar> &qnn_data) noexcept {
  if (qnn_data->m_backendState != QnnResourceState::OWNED) {
    return true;
  }

  auto *extension = qnn_data->m_backendExtensions == nullptr
                      ? nullptr
                      : qnn_data->m_backendExtensions->interface();
  try {
    if (extension != nullptr && !extension->beforeBackendTerminate()) {
      qnn_data->m_backendState = QnnResourceState::QUARANTINED;
      ml_loge("QNN extension rejected backend teardown");
      return false;
    }
  } catch (const std::exception &e) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("QNN extension threw before backend teardown: %s", e.what());
    return false;
  } catch (...) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("QNN extension threw before backend teardown");
    return false;
  }

  if (qnn_data->m_backendHandle == nullptr ||
      qnn_data->m_qnnFunctionPointers.qnnInterface.backendFree == nullptr) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("Cannot release owned QNN backend resource");
    return false;
  }

  Qnn_ErrorHandle_t status = QNN_BACKEND_NO_ERROR;
  try {
    status = qnn_data->m_qnnFunctionPointers.qnnInterface.backendFree(
      qnn_data->m_backendHandle);
  } catch (const std::exception &e) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("QNN backendFree threw: %s", e.what());
    return false;
  } catch (...) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("QNN backendFree threw");
    return false;
  }
  if (!qnnCallSucceeded(status)) {
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    ml_loge("QNN backendFree failed: error=%" PRIu64 ", public_error=%u",
            static_cast<uint64_t>(status), publicQnnError(status));
    return false;
  }
  qnn_data->m_backendHandle = nullptr;
  qnn_data->m_backendState = QnnResourceState::RELEASED;

  try {
    if (extension != nullptr && !extension->afterBackendTerminate()) {
      ml_loge("QNN extension failed after backend teardown");
      return false;
    }
  } catch (const std::exception &e) {
    ml_loge("QNN extension threw after backend teardown: %s", e.what());
    return false;
  } catch (...) {
    ml_loge("QNN extension threw after backend teardown");
    return false;
  }
  return true;
}

static bool
releaseLogResource(const std::shared_ptr<QNNVar> &qnn_data) noexcept {
  if (qnn_data->m_logState != QnnResourceState::OWNED) {
    return true;
  }
  if (qnn_data->m_logHandle == nullptr ||
      qnn_data->m_qnnFunctionPointers.qnnInterface.logFree == nullptr) {
    qnn_data->m_logState = QnnResourceState::QUARANTINED;
    ml_loge("Cannot release owned QNN log resource");
    return false;
  }

  Qnn_ErrorHandle_t status = QNN_SUCCESS;
  try {
    status = qnn_data->m_qnnFunctionPointers.qnnInterface.logFree(
      qnn_data->m_logHandle);
  } catch (const std::exception &e) {
    qnn_data->m_logState = QnnResourceState::QUARANTINED;
    ml_loge("QNN logFree threw: %s", e.what());
    return false;
  } catch (...) {
    qnn_data->m_logState = QnnResourceState::QUARANTINED;
    ml_loge("QNN logFree threw");
    return false;
  }
  if (!qnnCallSucceeded(status)) {
    qnn_data->m_logState = QnnResourceState::QUARANTINED;
    ml_loge("QNN logFree failed: error=%" PRIu64 ", public_error=%u",
            static_cast<uint64_t>(status), publicQnnError(status));
    return false;
  }

  qnn_data->m_logHandle = nullptr;
  qnn_data->m_logState = QnnResourceState::RELEASED;
  return true;
}

QNNContext::~QNNContext() {
  auto qnn_data = getQnnData();
  if (!qnn_data) {
    return;
  }

  auto runtime_lifecycle = qnn_data->runtime_lifecycle;
  if (!runtime_lifecycle) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    qnn_data->quarantine_self_reference = qnn_data;
    ml_loge("QNN runtime teardown quarantined without a lifecycle gate");
    return;
  }

  QNNRuntimeLifecycle::CleanupGuard shutdown_guard;
  try {
    shutdown_guard = runtime_lifecycle->beginRuntimeShutdown();
  } catch (const std::exception &e) {
    quarantineRuntime(qnn_data, runtime_lifecycle, nullptr, e.what());
    return;
  } catch (...) {
    quarantineRuntime(qnn_data, runtime_lifecycle, nullptr,
                      "could not drain active executions");
    return;
  }

  const bool resource_state_consistent =
    resourceStateMatchesHandle(qnn_data->m_logState, qnn_data->m_logHandle) &&
    resourceStateMatchesHandle(qnn_data->m_backendState,
                               qnn_data->m_backendHandle) &&
    resourceStateMatchesHandle(qnn_data->m_deviceState,
                               qnn_data->m_deviceHandle) &&
    resourceStateMatchesHandle(qnn_data->m_profileState,
                               qnn_data->m_profileBackendHandle) &&
    ((qnn_data->m_backendLibraryHandle == nullptr) ==
     !qnn_data->m_backendLibraryLifetime) &&
    (qnn_data->m_backendExtensions == nullptr ||
     qnn_data->m_resourceManager != nullptr) &&
    ((qnn_data->m_deviceLifecycleInitialized &&
      (qnn_data->m_deviceState == QnnResourceState::OWNED ||
       qnn_data->m_deviceState == QnnResourceState::UNSUPPORTED)) ||
     (!qnn_data->m_deviceLifecycleInitialized &&
      qnn_data->m_deviceState != QnnResourceState::OWNED &&
      qnn_data->m_deviceState != QnnResourceState::UNSUPPORTED));
  if (!resource_state_consistent || qnn_data->m_hasRuntimeResourceQuarantine ||
      qnn_data->m_hasSystemContextFreeFailure ||
      qnn_data->m_hasContextQuarantine) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "pre-existing ambiguous resource state");
    return;
  }

  if (qnn_data->RpcMem) {
    const auto drain_report =
      qnn_data->RpcMem->drainRegistrationsForShutdown(shutdown_guard);
    if (!drain_report.success()) {
      ml_loge("QNN registration drain failed: discovered=%zu, attempted=%zu, "
              "drained=%zu, remaining=%zu, quarantined=%zu, reason=%u",
              drain_report.discovered, drain_report.attempted,
              drain_report.drained, drain_report.remaining,
              drain_report.quarantined,
              static_cast<unsigned int>(drain_report.failure));
      quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                        "memory registration drain failed");
      return;
    }
  }

  try {
    if (qnn_data->freeAllContextsWithGuard(shutdown_guard) !=
        StatusCode::SUCCESS) {
      quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                        "context teardown failed");
      return;
    }
  } catch (const std::exception &e) {
    ml_loge("Exception during QNN context teardown: %s", e.what());
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "context teardown threw");
    return;
  } catch (...) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "context teardown threw");
    return;
  }

  if (!releaseProfileResource(qnn_data)) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "profile teardown failed");
    return;
  }
  if (!releaseDeviceResource(qnn_data)) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "device teardown failed");
    return;
  }
  if (!releaseBackendResource(qnn_data)) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "backend teardown failed");
    return;
  }

  qnn_data->m_backendExtensions.reset();
  if (!releaseLogResource(qnn_data)) {
    quarantineRuntime(qnn_data, runtime_lifecycle, &shutdown_guard,
                      "backend logging teardown failed");
    return;
  }

  qnn_data->m_qnnFunctionPointers.qnnSystemInterface = {};
  qnn_data->m_resourceManager.reset();

  void *backend_library_handle = qnn_data->m_backendLibraryHandle;
  qnn_data->m_backendLibraryHandle = nullptr;
  qnn_data->m_qnnFunctionPointers = {};
  if (qnn_data->m_backendLibraryLifetime) {
    qnn_data->m_backendLibraryLifetime.reset();
  } else if (backend_library_handle != nullptr) {
    pal::dynamicloading::dlClose(backend_library_handle);
  }

  runtime_lifecycle->finishRuntimeShutdown(shutdown_guard,
                                           QnnRuntimeState::SHUT_DOWN);
}

void QNNContext::initialize() noexcept {
  LOGD("initialize: START");
  try {
    LOGD("initialize: calling initBackend()");
    if (initBackend() != 0) {
      // Backend initialization failed (e.g. no usable QNN/HTP backend on this
      // device). Do not register the QNN layer factory or rpcmem allocator:
      // the process must survive so CPU models can still run.
      LOGE("initialize: initBackend() failed; QNN backend unavailable on this "
           "device, skipping QNN registration (CPU path stays alive)");
      ml_logw("qnn init failed; continuing without QNN backend");
      LOGD("initialize: END (init failed)");
      return;
    }
    LOGD("initialize: initBackend() completed");
    ml_logi("qnn init done");
    LOGD("initialize: creating QNNRpcManager");
    auto qnn_data = getQnnData();
    NNTR_THROW_IF(qnn_data->m_backendLibraryHandle == nullptr ||
                    !qnn_data->m_backendLibraryLifetime,
                  std::runtime_error)
      << "QNN backend library lifetime is unavailable";

    auto rpc_mem = std::make_shared<QNNRpcManager>(
      qnn_data->m_qnnFunctionPointers.qnnInterface,
      qnn_data->m_backendLibraryLifetime, qnn_data->runtime_lifecycle);

    LOGD("initialize: registering QNN layers");
    registerFactory(nntrainer::createLayer<QNNGraph>, QNNGraph::type, -1);
    ml_logi("qnn registerFactory done");
    LOGD("initialize: registerFactory done");

    qnn_data->RpcMem = rpc_mem;
    setMemAllocator(std::move(rpc_mem));
    LOGD("initialize: QNNRpcManager set");
    qnn_initialized = true;
  } catch (std::exception &e) {
    LOGE("initialize: registering qnn layers failed!!, reason: %s", e.what());
    ml_loge("registering qnn layers failed!!, reason: %s", e.what());
  } catch (...) {
    LOGE("initialize: registering qnn layer failed due to unknown reason");
    ml_loge("registering qnn layer failed due to unknown reason");
  }
  LOGD("initialize: END");
}

int QNNContext::init() {
  try {
    return ensureInitialized() ? 0 : -1;
  } catch (const std::exception &e) {
    ml_loge("QNN backend initialization raised an exception: %s", e.what());
  } catch (...) {
    ml_loge("QNN backend initialization raised an unknown exception");
  }
  return -1;
}

int QNNContext::initBackend() {
  LOGD("init: START");
  std::cout << "qnncontext::init called" << std::endl;
  if (!log::initializeLogging()) {
    LOGE("init: Unable to initialize logging!");
    ml_loge("ERROR: Unable to initialize logging!");
    return -1;
  }
  LOGD("init: logging initialized");
  log::setLogLevel(QnnLog_Level_t::QNN_LOG_LEVEL_ERROR);

  std::string backEndPath = "libQnnHtp.so";
  LOGD("init: backEndPath=%s", backEndPath.c_str());

  std::string opPackagePaths = "";

  auto qnn_data = getQnnData();
  LOGD("init: qnn_data obtained");

  qnn_data->m_outputDataType = iotensor::OutputDataType::FLOAT_AND_NATIVE;
  qnn_data->m_inputDataType = iotensor::InputDataType::NATIVE;
  qnn_data->m_profilingLevel = ProfilingLevel::OFF;

  m_isContextCreated = false;

  qnn::tools::sample_app::split(m_opPackagePaths, opPackagePaths, ',');

  if (backEndPath.empty()) {
    LOGE("init: Cannot find backend Path : libQnnHtp.so");
    ml_loge("ERROR: Cannot fine backend Path : libQnnHtp.so");
    return -1;
  }
  LOGD("init: calling dynamicloadutil::getQnnFunctionPointers");

  auto statusCode = dynamicloadutil::getQnnFunctionPointers(
    backEndPath, "", &qnn_data->m_qnnFunctionPointers,
    &qnn_data->m_backendLibraryHandle, false, nullptr);
  LOGD("init: getQnnFunctionPointers returned status=%d", (int)statusCode);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
      LOGE("init: could not load backend");
      ml_loge(
        "Error: initializing QNN Function Pointers: could not load backend");
    } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
      LOGE("init: could not load model");
      ml_loge("Error initializing QNN Function Pointers: could not load");
    } else {
      LOGE("init: unknown error initializing QNN Function Pointers");
      ml_loge("Error initializing QNN Function Pointers");
    }
    if (qnn_data->m_backendLibraryHandle != nullptr) {
      pal::dynamicloading::dlClose(qnn_data->m_backendLibraryHandle);
      qnn_data->m_backendLibraryHandle = nullptr;
    }
    qnn_data->m_qnnFunctionPointers = {};
    return -1;
  }

  if (qnn_data->m_backendLibraryHandle == nullptr) {
    qnn_data->m_qnnFunctionPointers = {};
    ml_loge("QNN function-pointer loading succeeded without a backend handle");
    return -1;
  }

  // Establish shared ownership before any vendor resource is created. The RPC
  // allocator can outlive QNNContext through MemoryPool, so both retain this
  // one loader handle instead of independently reopening the backend.
  try {
    qnn_data->m_backendLibraryLifetime = std::shared_ptr<void>(
      qnn_data->m_backendLibraryHandle, [](void *handle) noexcept {
        if (handle != nullptr) {
          pal::dynamicloading::dlClose(handle);
        }
      });
  } catch (...) {
    // shared_ptr invokes the deleter when control-block allocation fails.
    // Invalidate the copied table so teardown cannot call into the closed DSO.
    qnn_data->m_backendLibraryHandle = nullptr;
    qnn_data->m_qnnFunctionPointers = {};
    throw;
  }

  const auto &qnn_interface = qnn_data->m_qnnFunctionPointers.qnnInterface;
  if (qnn_interface.backendCreate == nullptr ||
      qnn_interface.backendFree == nullptr ||
      (qnn_interface.logCreate == nullptr) !=
        (qnn_interface.logFree == nullptr) ||
      (qnn_interface.deviceCreate != nullptr &&
       qnn_interface.deviceFree == nullptr)) {
    ml_loge("QNN function table has incomplete resource lifecycle pairs");
    return -1;
  }

  try {
    qnn_data->m_resourceManager = std::make_shared<genie::ResourceManager>();
    qnn_data->m_qnnFunctionPointers.qnnSystemInterface =
      qnn_data->m_resourceManager->getQnnSystemInterface();
  } catch (const std::exception &e) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN ResourceManager construction failed: %s", e.what());
    return -1;
  } catch (...) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN ResourceManager construction failed");
    return -1;
  }

  const auto &system_interface =
    qnn_data->m_qnnFunctionPointers.qnnSystemInterface;
  if (system_interface.systemContextCreate == nullptr ||
      system_interface.systemContextGetBinaryInfo == nullptr ||
      system_interface.systemContextFree == nullptr) {
    ml_loge("QNN system interface is incomplete");
    return -1;
  }

  if (log::isLogInitialized() && qnn_interface.logCreate != nullptr) {
    auto logCallback = log::getLogCallback();
    auto logLevel = log::getLogLevel();

    Qnn_LogHandle_t log_handle = nullptr;
    Qnn_ErrorHandle_t log_status = QNN_SUCCESS;
    try {
      log_status = qnn_interface.logCreate(logCallback, logLevel, &log_handle);
    } catch (const std::exception &e) {
      qnn_data->m_logHandle = log_handle;
      qnn_data->m_logState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN logCreate threw: %s", e.what());
      return -1;
    } catch (...) {
      qnn_data->m_logHandle = log_handle;
      qnn_data->m_logState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN logCreate threw");
      return -1;
    }

    const auto log_error = publicQnnError(log_status);
    if (log_error == QNN_SUCCESS && log_handle != nullptr) {
      qnn_data->m_logHandle = log_handle;
      qnn_data->m_logState = QnnResourceState::OWNED;
      LOGD("init: Logging initialized in the backend");
    } else if (log_error == QNN_COMMON_ERROR_NOT_SUPPORTED &&
               log_handle == nullptr) {
      qnn_data->m_logState = QnnResourceState::UNSUPPORTED;
      ml_logw("QNN backend logging is not supported");
    } else if (log_handle != nullptr || log_error == QNN_SUCCESS) {
      qnn_data->m_logHandle = log_handle;
      qnn_data->m_logState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN logCreate returned an ambiguous result: error=%" PRIu64
              ", public_error=%u, handle=%p",
              static_cast<uint64_t>(log_status), log_error,
              static_cast<void *>(log_handle));
      return -1;
    } else {
      ml_loge("QNN logCreate failed: error=%" PRIu64 ", public_error=%u",
              static_cast<uint64_t>(log_status), log_error);
      return -1;
    }
  } else {
    qnn_data->m_logState = QnnResourceState::UNSUPPORTED;
  }

  LOGD("init: Creating backend extensions");
  BackendExtensionsConfigs backend_extensions_config;
  std::string config_path;
  if (!m_backendExtConfigPath.empty()) {
    config_path = m_backendExtConfigPath;
  } else if (!g_default_backend_ext_config_path.empty()) {
    config_path = g_default_backend_ext_config_path;
  } else {
    config_path = resolve_backend_extensions_config_path();
  }
  config_path = resolve_backend_extensions_config_value(config_path);
  LOGD("init: backend_extensions_config.configFilePath = %s",
       config_path.c_str());
  backend_extensions_config.configFilePath = config_path;
  backend_extensions_config.sharedLibraryPath = "libQnnHtpNetRunExtensions.so";

  try {
    qnn_data->m_backendExtensions = std::make_unique<BackendExtensions>(
      backend_extensions_config, qnn_data->m_backendLibraryHandle, false,
      nullptr, QNN_LOG_LEVEL_ERROR, qnn_data->m_resourceManager);
  } catch (const std::exception &e) {
    // The generated SDK wrapper does not own its extension DSO until
    // construction completes. A throw can therefore leave hidden extension
    // state that the caller cannot safely roll back.
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN BackendExtensions construction failed: %s", e.what());
    return -1;
  } catch (...) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN BackendExtensions construction failed");
    return -1;
  }
  auto *backend_extensions = qnn_data->m_backendExtensions.get();
  LOGD("init: Backend extensions created");

  QnnBackend_Config_t **customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (backend_extensions->interface()) {
    bool hook_succeeded = false;
    try {
      hook_succeeded = backend_extensions->interface()->beforeBackendInitialize(
        &customConfigs, &customConfigCount);
    } catch (const std::exception &e) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw before backend initialization: %s",
              e.what());
      return -1;
    } catch (...) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw before backend initialization");
      return -1;
    }
    if (!hook_succeeded) {
      LOGE("init: Extensions Failure in beforeBackendInitialize()");
      QNN_ERROR("Extensions Failure in beforeBackendInitialize()");
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      return -1;
    }
    LOGD("init: beforeBackendInitialize done, customConfigCount=%u",
         customConfigCount);
  }
  if (customConfigCount > 0 && customConfigs == nullptr) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN extension returned an invalid backend config list");
    return -1;
  }
  const size_t backend_config_count = customConfigCount;
  if (backend_config_count > std::numeric_limits<size_t>::max() - 1) {
    ml_loge("QNN extension returned too many backend configs");
    return -1;
  }

  std::vector<const QnnBackend_Config_t *> backend_config_pointers;
  if (backend_config_count > 0) {
    backend_config_pointers.reserve(backend_config_count + 1);
    for (size_t index = 0; index < backend_config_count; ++index) {
      if (customConfigs[index] == nullptr) {
        qnn_data->m_hasRuntimeResourceQuarantine = true;
        ml_loge("QNN extension returned a null backend config at index %zu",
                index);
        return -1;
      }
      backend_config_pointers.push_back(customConfigs[index]);
    }
    backend_config_pointers.push_back(nullptr);
  }

  LOGD("init: Calling backendCreate");
  Qnn_BackendHandle_t backend_handle = nullptr;
  Qnn_ErrorHandle_t qnnStatus = QNN_BACKEND_NO_ERROR;
  try {
    qnnStatus = qnn_interface.backendCreate(
      qnn_data->m_logHandle,
      backend_config_count == 0 ? nullptr : backend_config_pointers.data(),
      &backend_handle);
  } catch (const std::exception &e) {
    qnn_data->m_backendHandle = backend_handle;
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN backendCreate threw: %s", e.what());
    return -1;
  } catch (...) {
    qnn_data->m_backendHandle = backend_handle;
    qnn_data->m_backendState = QnnResourceState::QUARANTINED;
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN backendCreate threw");
    return -1;
  }
  const auto backend_error = publicQnnError(qnnStatus);
  LOGD("init: backendCreate returned public status=%u", backend_error);
  if (backend_error != QNN_BACKEND_NO_ERROR || backend_handle == nullptr) {
    qnn_data->m_backendHandle = backend_handle;
    if (backend_handle != nullptr || backend_error == QNN_BACKEND_NO_ERROR) {
      qnn_data->m_backendState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN backendCreate returned an ambiguous result: error=%" PRIu64
              ", public_error=%u, handle=%p",
              static_cast<uint64_t>(qnnStatus), backend_error,
              static_cast<void *>(backend_handle));
    } else {
      ml_loge("QNN backendCreate failed: error=%" PRIu64 ", public_error=%u",
              static_cast<uint64_t>(qnnStatus), backend_error);
    }
    return -1;
  }
  qnn_data->m_backendHandle = backend_handle;
  qnn_data->m_backendState = QnnResourceState::OWNED;
  ml_logi("Initialize Backend Returned Status = %u", backend_error);
  LOGD("init: Backend initialized successfully");

  if (backend_extensions->interface()) {
    bool hook_succeeded = false;
    try {
      hook_succeeded =
        backend_extensions->interface()->afterBackendInitialize();
    } catch (const std::exception &e) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw after backend initialization: %s", e.what());
      return -1;
    } catch (...) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw after backend initialization");
      return -1;
    }
    if (!hook_succeeded) {
      LOGE("init: Extensions Failure in afterBackendInitialize()");
      QNN_ERROR("Extensions Failure in afterBackendInitialize()");
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      return -1;
    }
    LOGD("init: afterBackendInitialize done");
  }

  LOGD("init: Creating device");
  auto devicePropertySupportStatus = this->isDevicePropertySupported();
  LOGD("init: isDevicePropertySupported returned %d",
       (int)devicePropertySupportStatus);
  if (StatusCode::FAILURE == devicePropertySupportStatus) {
    return -1;
  }
  auto createDeviceStatus = this->createDevice();
  LOGD("init: createDevice returned %d", (int)createDeviceStatus);
  if (StatusCode::SUCCESS != createDeviceStatus) {
    LOGE("init: Device Creation failure");
    ml_loge("Device Creation failure");
    return -1;
  }

  LOGD("init: Initializing profiling");
  if (StatusCode::SUCCESS != this->initializeProfiling()) {
    LOGE("init: Profiling Initialization failure");
    ml_loge("Profiling Initialization failure");
    return -1;
  }

  LOGD("init: Registering Op Packages");
  if (StatusCode::SUCCESS != this->registerOpPackages()) {
    LOGE("init: Register Op Packages failure");
    ml_loge("Register Op Packages failure");
    return -1;
  }

  LOGD("init: END (returning 0)");
  return 0;
}

template <typename T>
const int QNNContext::registerFactory(const FactoryType<T> factory,
                                      const std::string &key,
                                      const int int_key) {
  static_assert(isSupported<T>::value,
                "qnn_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(qnn_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    // std::stringstream ss;
    // ss << "qnn_context: cannot register factory with already taken key: "
    //    << key;
    // throw std::invalid_argument(ss.str().c_str());
    for (const auto &[ik, sk] : int_map) {
      if (sk == assigned_key)
        return ik;
    }

    return -1;
  }
  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    // std::stringstream ss;
    // ss << "qnn_context: cannot register factory with already taken int key: "
    //    << int_key;
    // throw std::invalid_argument(ss.str().c_str());
    return int_key;
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("qnn_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

StatusCode QNNContext::isDevicePropertySupported() {
  auto qnn_data = getQnnData();
  auto property_has_capability =
    qnn_data->m_qnnFunctionPointers.qnnInterface.propertyHasCapability;
  if (property_has_capability == nullptr) {
    return StatusCode::SUCCESS;
  }

  Qnn_ErrorHandle_t qnn_status = QNN_SUCCESS;
  try {
    qnn_status = property_has_capability(QNN_PROPERTY_GROUP_DEVICE);
  } catch (const std::exception &e) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN device property query threw: %s", e.what());
    return StatusCode::FAILURE;
  } catch (...) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN device property query threw");
    return StatusCode::FAILURE;
  }

  const auto qnn_error = publicQnnError(qnn_status);
  if (qnn_error == QNN_SUCCESS) {
    return StatusCode::SUCCESS;
  }
  if (qnn_error == QNN_PROPERTY_NOT_SUPPORTED) {
    ml_logw("Device property is not supported");
    return StatusCode::SUCCESS;
  }
  if (qnn_error == QNN_PROPERTY_ERROR_UNKNOWN_KEY) {
    ml_loge("Device property is not known to backend");
  } else {
    ml_loge("Device property query failed: error=%" PRIu64 ", public_error=%u",
            static_cast<uint64_t>(qnn_status), qnn_error);
  }
  return StatusCode::FAILURE;
}

StatusCode QNNContext::createDevice() {
  auto qnn_data = getQnnData();
  QnnDevice_Config_t **deviceConfigs{nullptr};
  uint32_t configCount{0};
  uint32_t socModel{0};
  auto *backend_extensions = qnn_data->m_backendExtensions.get();

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    bool hook_succeeded = false;
    try {
      hook_succeeded = backend_extensions->interface()->beforeCreateDevice(
        &deviceConfigs, &configCount, socModel);
    } catch (const std::exception &e) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw before device creation: %s", e.what());
      return StatusCode::FAILURE;
    } catch (...) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw before device creation");
      return StatusCode::FAILURE;
    }
    if (!hook_succeeded) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      QNN_ERROR("Extensions Failure in beforeCreateDevice()");
      return StatusCode::FAILURE;
    }
  }

  if (configCount > 0 && deviceConfigs == nullptr) {
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN extension returned an invalid device config list");
    return StatusCode::FAILURE;
  }
  const size_t device_config_count = configCount;
  if (device_config_count > std::numeric_limits<size_t>::max() - 1) {
    ml_loge("QNN extension returned too many device configs");
    return StatusCode::FAILURE;
  }

  std::vector<const QnnDevice_Config_t *> device_config_pointers;
  if (device_config_count > 0) {
    device_config_pointers.reserve(device_config_count + 1);
    for (size_t index = 0; index < device_config_count; ++index) {
      if (deviceConfigs[index] == nullptr) {
        qnn_data->m_hasRuntimeResourceQuarantine = true;
        ml_loge("QNN extension returned a null device config at index %zu",
                index);
        return StatusCode::FAILURE;
      }
      device_config_pointers.push_back(deviceConfigs[index]);
    }
    device_config_pointers.push_back(nullptr);
  }

  auto device_create =
    qnn_data->m_qnnFunctionPointers.qnnInterface.deviceCreate;
  if (device_create == nullptr) {
    qnn_data->m_deviceState = QnnResourceState::UNSUPPORTED;
  } else {
    Qnn_DeviceHandle_t device_handle = nullptr;
    Qnn_ErrorHandle_t qnn_status = QNN_SUCCESS;
    try {
      qnn_status = device_create(
        qnn_data->m_logHandle,
        device_config_count == 0 ? nullptr : device_config_pointers.data(),
        &device_handle);
    } catch (const std::exception &e) {
      qnn_data->m_deviceHandle = device_handle;
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN deviceCreate threw: %s", e.what());
      return StatusCode::FAILURE;
    } catch (...) {
      qnn_data->m_deviceHandle = device_handle;
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN deviceCreate threw");
      return StatusCode::FAILURE;
    }

    const auto qnn_error = publicQnnError(qnn_status);
    if (qnn_error == QNN_SUCCESS && device_handle != nullptr) {
      qnn_data->m_deviceHandle = device_handle;
      qnn_data->m_deviceState = QnnResourceState::OWNED;
    } else if (qnn_error == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE &&
               device_handle == nullptr) {
      qnn_data->m_deviceState = QnnResourceState::UNSUPPORTED;
    } else if (device_handle != nullptr || qnn_error == QNN_SUCCESS) {
      qnn_data->m_deviceHandle = device_handle;
      qnn_data->m_deviceState = QnnResourceState::QUARANTINED;
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN deviceCreate returned an ambiguous result: error=%" PRIu64
              ", public_error=%u, handle=%p",
              static_cast<uint64_t>(qnn_status), qnn_error,
              static_cast<void *>(device_handle));
      return StatusCode::FAILURE;
    } else {
      ml_loge("QNN deviceCreate failed: error=%" PRIu64 ", public_error=%u",
              static_cast<uint64_t>(qnn_status), qnn_error);
      return verifyFailReturnStatus(qnn_status);
    }
  }

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    bool hook_succeeded = false;
    try {
      hook_succeeded = backend_extensions->interface()->afterCreateDevice();
    } catch (const std::exception &e) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw after device creation: %s", e.what());
      return StatusCode::FAILURE;
    } catch (...) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      ml_loge("QNN extension threw after device creation");
      return StatusCode::FAILURE;
    }
    if (!hook_succeeded) {
      qnn_data->m_hasRuntimeResourceQuarantine = true;
      QNN_ERROR("Extensions Failure in afterCreateDevice()");
      return StatusCode::FAILURE;
    }
  }
  qnn_data->m_deviceLifecycleInitialized = true;
  return StatusCode::SUCCESS;
}

StatusCode QNNContext::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
  auto returnStatus = StatusCode::FAILURE;
  switch (publicQnnError(errCode)) {
  case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
    returnStatus = StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
    break;
  case QNN_COMMON_ERROR_SYSTEM:
    returnStatus = StatusCode::FAILURE_SYSTEM_ERROR;
    break;
  case QNN_COMMON_ERROR_NOT_SUPPORTED:
    returnStatus = StatusCode::QNN_FEATURE_UNSUPPORTED;
    break;
  default:
    break;
  }
  return returnStatus;
}

StatusCode QNNContext::initializeProfiling() {
  auto qnn_data = getQnnData();
  if (ProfilingLevel::OFF == qnn_data->m_profilingLevel) {
    return StatusCode::SUCCESS;
  }

  auto profile_create =
    qnn_data->m_qnnFunctionPointers.qnnInterface.profileCreate;
  auto profile_free = qnn_data->m_qnnFunctionPointers.qnnInterface.profileFree;
  if (profile_create == nullptr || profile_free == nullptr) {
    ml_loge("QNN profile lifecycle function table is incomplete");
    return StatusCode::FAILURE;
  }

  QnnProfile_Level_t profile_level;
  if (ProfilingLevel::BASIC == qnn_data->m_profilingLevel) {
    profile_level = QNN_PROFILE_LEVEL_BASIC;
  } else if (ProfilingLevel::DETAILED == qnn_data->m_profilingLevel) {
    profile_level = QNN_PROFILE_LEVEL_DETAILED;
  } else {
    ml_loge("Unsupported QNN profiling level");
    return StatusCode::FAILURE;
  }

  Qnn_ProfileHandle_t profile_handle = nullptr;
  Qnn_ErrorHandle_t qnn_status = QNN_PROFILE_NO_ERROR;
  try {
    qnn_status =
      profile_create(qnn_data->m_backendHandle, profile_level, &profile_handle);
  } catch (const std::exception &e) {
    qnn_data->m_profileBackendHandle = profile_handle;
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN profileCreate threw: %s", e.what());
    return StatusCode::FAILURE;
  } catch (...) {
    qnn_data->m_profileBackendHandle = profile_handle;
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN profileCreate threw");
    return StatusCode::FAILURE;
  }

  const auto qnn_error = publicQnnError(qnn_status);
  if (qnn_error == QNN_PROFILE_NO_ERROR && profile_handle != nullptr) {
    qnn_data->m_profileBackendHandle = profile_handle;
    qnn_data->m_profileState = QnnResourceState::OWNED;
    return StatusCode::SUCCESS;
  }
  if (profile_handle != nullptr || qnn_error == QNN_PROFILE_NO_ERROR) {
    qnn_data->m_profileBackendHandle = profile_handle;
    qnn_data->m_profileState = QnnResourceState::QUARANTINED;
    qnn_data->m_hasRuntimeResourceQuarantine = true;
    ml_loge("QNN profileCreate returned an ambiguous result: error=%" PRIu64
            ", public_error=%u, handle=%p",
            static_cast<uint64_t>(qnn_status), qnn_error,
            static_cast<void *>(profile_handle));
  } else {
    ml_loge("QNN profileCreate failed: error=%" PRIu64 ", public_error=%u",
            static_cast<uint64_t>(qnn_status), qnn_error);
  }
  return StatusCode::FAILURE;
}

StatusCode QNNContext::registerOpPackages() {
  const size_t pathIdx = 0;
  const size_t interfaceProviderIdx = 1;
  auto qnn_data = getQnnData();
  std::cout << qnn_data->name << std::endl;

  for (auto const &opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    qnn::tools::sample_app::split(opPackage, opPackagePath, ':');
    ml_logi("opPackagePath: %s", opPackagePath.c_str());
    const char *target = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      ml_loge("Malformed opPackageString provided: %s", opPackagePath.c_str());
      return StatusCode::FAILURE;
    }
    if (opPackage.size() == 3) {
      target = (char *)opPackage[targetIdx].c_str();
    }
    if (nullptr ==
        qnn_data->m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
      ml_loge("backendRegisterOpPackageFnHandle is nullptr.");
      return StatusCode::FAILURE;
    }
    const auto qnn_status =
      qnn_data->m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
        qnn_data->m_backendHandle, (char *)opPackage[pathIdx].c_str(),
        (char *)opPackage[interfaceProviderIdx].c_str(), target);
    if (!qnnCallSucceeded(qnn_status)) {
      ml_loge("Could not register Op Package: %s and interface provider: %s",
              opPackage[pathIdx].c_str(),
              opPackage[interfaceProviderIdx].c_str());
      return StatusCode::FAILURE;
    }
    ml_logi("Registered Op Package: %s and interface provider: %s",
            opPackage[pathIdx].c_str(),
            opPackage[interfaceProviderIdx].c_str());
  }
  return StatusCode::SUCCESS;
}

/**
 * @copydoc const int QNNContext::registerFactory
 */
template const int QNNContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

#ifdef PLUGGABLE
nntrainer::Context *create_qnn_context() {
  nntrainer::QNNContext *qnn_context = new nntrainer::QNNContext();
  if (!g_default_backend_ext_config_path.empty()) {
    qnn_context->setBackendExtConfigPath(g_default_backend_ext_config_path);
  }
  // Register a lightweight context. Vendor libraries, backend extensions, and
  // the RPC allocator are initialized only when a QNN layer or binary is used.
  return qnn_context;
}

void destory_qnn_context(nntrainer::Context *ct) { delete ct; }

extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{create_qnn_context,
                                                       destory_qnn_context};
}
#endif
} // namespace nntrainer
