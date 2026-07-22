// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   engine.cpp
 * @date   27 December 2024
 * @brief  This file contains engine context related functions and classes that
 * manages the engines (NPU, GPU, CPU) of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <app_context.h>
#include <base_properties.h>
#include <compute_ops.h>
#include <context.h>
#include <dynamic_library_loader.h>
#include <engine.h>

static std::string solib_suffix = ".so";
static std::string contextlib_suffix = "context.so";
static const std::string func_tag = "[Engine] ";

namespace nntrainer {

namespace {

struct ContextPluginRecord {
  std::once_flag initialization_once;
  std::string context_name;
  nntrainer::Context *context = nullptr;
  void *library_handle = nullptr;
  DestroyContextFunc destroy_func = nullptr;
};

struct ContextPluginRegistry {
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<ContextPluginRecord>> records;
};

struct ContextPluginIdentity {
  std::string key;
  std::string load_path;
};

ContextPluginRegistry &contextPluginRegistry() {
  // Engine contexts and their DSOs are process-lifetime resources today.
  // Keep the registry alive as well so static destruction cannot unload a
  // plugin while a late model destructor still references its Context.
  static auto *registry = new ContextPluginRegistry();
  return *registry;
}

ContextPluginIdentity resolvePluginIdentity(const std::string &path) {
  const std::filesystem::path requested(path);
  if (!requested.is_absolute() && requested.parent_path().empty()) {
    auto loader_name = requested.generic_string();
#if defined(_WIN32)
    std::transform(loader_name.begin(), loader_name.end(), loader_name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
#endif
    return {"loader-name:" + loader_name, path};
  }

  const auto load_path =
    std::filesystem::absolute(requested).lexically_normal();
  std::error_code error;
  auto key_path = std::filesystem::weakly_canonical(load_path, error);
  if (error)
    key_path = load_path;

  auto file_key = key_path.generic_string();
#if defined(_WIN32)
  std::transform(file_key.begin(), file_key.end(), file_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });
#endif
  return {"file-path:" + file_key, load_path.string()};
}

std::shared_ptr<ContextPluginRecord>
getContextPluginRecord(const std::string &path) {
  auto &registry = contextPluginRegistry();
  const std::lock_guard<std::mutex> lock(registry.mutex);

  auto found = registry.records.find(path);
  if (found != registry.records.end()) {
    return found->second;
  }

  auto record = std::make_shared<ContextPluginRecord>();
  registry.records.emplace(path, record);
  return record;
}

} // namespace

std::mutex engine_mutex;

std::once_flag global_engine_init_flag;

nntrainer::Context
  *Engine::nntrainerRegisteredContext[Engine::RegisterContextMax];

/** @brief Return the Engine singleton. */
template <> NNTRAINER_SINGLETON_API Engine &Singleton<Engine>::Global() {
  static Engine instance;
  instance.initializeOnce();
  return instance;
}

/** @brief Return the process-wide Engine singleton. */
NNTRAINER_SINGLETON_API Engine &Engine::Global() {
  return Singleton<Engine>::Global();
}

void Engine::add_default_object() {
  /// @note all layers should be added to the app_context to guarantee that
  /// createLayer/createOptimizer class is created

  auto &app_context = nntrainer::AppContext::Global();

  // Ensure CPU backend compute-ops table is bound. ensureComputeOps() is
  // std::call_once-guarded, so this call is safe even if AppContext or
  // another Context already initialized it.
  ensureComputeOps();
  registerContext("cpu", &app_context);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  auto &cl_context = nntrainer::ClContext::Global();

  registerContext("gpu", &cl_context);
#endif

#if defined(ENABLE_NPU) && ENABLE_NPU == 1
  // QNN context is loaded as a plugin .so for decoupling from QNN SDK.
  // libqnn_context.so exports ml_train_context_pluggable symbol.
  try {
    registerContext("libqnn_context.so", "");
  } catch (std::exception &e) {
    ml_logw("QNN context plugin not available: %s", e.what());
  }
#endif
}

void Engine::initialize() noexcept {
  try {
    add_default_object();
  } catch (std::exception &e) {
    ml_loge("registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering layer failed due to unknown reason");
  }
};

void Engine::release() {}

std::string
Engine::parseComputeEngine(const std::vector<std::string> &props) const {
  for (auto &prop : props) {
    std::string key, value;
    int status = nntrainer::getKeyValue(prop, key, value);
    if (nntrainer::istrequal(key, "engine")) {
      constexpr const auto data =
        std::data(props::ComputeEngineTypeInfo::EnumList);
      for (unsigned int i = 0;
           i < props::ComputeEngineTypeInfo::EnumList.size(); ++i) {
        if (nntrainer::istrequal(value.c_str(),
                                 props::ComputeEngineTypeInfo::EnumStr[i])) {
          return props::ComputeEngineTypeInfo::EnumStr[i];
        }
      }
    }
  }

  return "cpu";
}

/**
 * @brief Get the Full Path from given string
 * @details path is resolved in the following order
 * 1) if @a path is absolute, return path
 * ----------------------------------------
 * 2) if @a base == "" && @a path == "", return "."
 * 3) if @a base == "" && @a path != "", return @a path
 * 4) if @a base != "" && @a path == "", return @a base
 * 5) if @a base != "" && @a path != "", return @a base + "/" + path
 *
 * @param path path to calculate from base
 * @param base base path
 * @return const std::string
 */
const std::string getFullPath(const std::string &path,
                              const std::string &base) {
  /// if path is absolute, return path
  if (!path.empty() && std::filesystem::path(path).is_absolute()) {
    return path;
  }

  if (base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? base : base + "/" + path;
}

const std::string Engine::getWorkingPath(const std::string &path) const {
  return getFullPath(path, working_path_base);
}

void Engine::setWorkingDirectory(const std::string &base) {
  std::filesystem::path base_path(base);

  if (!std::filesystem::is_directory(base_path)) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }

  char *ret = getRealpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

int Engine::registerContext(const std::string &library_path,
                            const std::string &base_path) {
  NNTR_THROW_IF(library_path.empty(), std::invalid_argument)
    << func_tag << "context plugin path must not be empty";

  const std::string full_path = getFullPath(library_path, base_path);
  const auto plugin_identity = resolvePluginIdentity(full_path);
  auto plugin_record = getContextPluginRecord(plugin_identity.key);

  // Plugin factories must not recursively register the same identity:
  // std::call_once is intentionally non-reentrant for one record.
  std::call_once(
    plugin_record->initialization_once,
    [plugin_record, load_path = plugin_identity.load_path] {
      void *handle = DynamicLibraryLoader::loadLibrary(load_path.c_str(),
                                                       RTLD_LAZY | RTLD_LOCAL);
      std::unique_ptr<void, decltype(&DynamicLibraryLoader::freeLibrary)>
        library(handle, &DynamicLibraryLoader::freeLibrary);
      const auto load_error = handle == nullptr
                                ? DynamicLibraryLoader::getLastErrorString()
                                : std::string();

      NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
        << func_tag << "open context plugin failed, reason: " << load_error;

      auto *pluggable = reinterpret_cast<nntrainer::ContextPluggable *>(
        DynamicLibraryLoader::loadSymbol(handle, "ml_train_context_pluggable"));
      const auto symbol_error = pluggable == nullptr
                                  ? DynamicLibraryLoader::getLastErrorString()
                                  : std::string();

      NNTR_THROW_IF(pluggable == nullptr, std::invalid_argument)
        << func_tag
        << "loading context plugin symbol failed, reason: " << symbol_error;
      NNTR_THROW_IF(pluggable->createfunc == nullptr ||
                      pluggable->destroyfunc == nullptr,
                    std::invalid_argument)
        << func_tag << "context plugin factory is incomplete";

      auto *candidate = pluggable->createfunc();
      NNTR_THROW_IF(candidate == nullptr, std::invalid_argument)
        << func_tag << "created pluggable context is null";
      std::unique_ptr<nntrainer::Context, DestroyContextFunc> context(
        candidate, pluggable->destroyfunc);

      auto type = context->getName();
      NNTR_THROW_IF(type.empty(), std::invalid_argument)
        << func_tag << "custom context must specify a non-empty name";

      // Publish potentially throwing record state before releasing either
      // resource. A successful record owns one Context/DSO independent of the
      // Engine instance that performed the first load.
      plugin_record->context_name = type;
      plugin_record->context = context.get();
      plugin_record->library_handle = library.get();
      plugin_record->destroy_func = pluggable->destroyfunc;
      (void)context.release();
      (void)library.release();
    });

  NNTR_THROW_IF(plugin_record->context == nullptr ||
                  plugin_record->context_name.empty(),
                std::logic_error)
    << func_tag << "context plugin initialization completed without a Context";

  // A successful process-wide plugin initialization can still need attaching
  // to another Engine instance. Do not silently bind this path to an unrelated
  // Context that happens to have the same name in that Engine.
  auto *registered_context =
    registerContextAndGet(plugin_record->context_name, plugin_record->context);
  NNTR_THROW_IF(registered_context != plugin_record->context,
                std::invalid_argument)
    << func_tag << "context name collision for plugin " << full_path << ": "
    << plugin_record->context_name;

  return 0;
}

} // namespace nntrainer
