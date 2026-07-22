// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file unittest_engine_context_plugin.cpp
 * @brief Validate context plugin initialization and registration lifecycle
 */

#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "fake_context_plugin_api.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dynamic_library_loader.h>
#include <engine.h>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string pluginPath(int id) {
  const std::string variable = "NNTR_FAKE_CONTEXT_PLUGIN_" + std::to_string(id);
  const char *path = std::getenv(variable.c_str());
  if (path == nullptr || path[0] == '\0') {
    throw std::runtime_error("missing test plugin path: " + variable);
  }
  return path;
}

class PluginProbe {
public:
  explicit PluginProbe(const std::string &path) {
    handle_ = nntrainer::DynamicLibraryLoader::loadLibrary(
      path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      throw std::runtime_error(
        "failed to open fake context plugin: " +
        nntrainer::DynamicLibraryLoader::getLastErrorString());
    }

    try {
      reset_ = loadSymbol<decltype(reset_)>("nntr_fake_context_reset");
      create_count_ =
        loadSymbol<decltype(create_count_)>("nntr_fake_context_create_count");
      destroy_count_ =
        loadSymbol<decltype(destroy_count_)>("nntr_fake_context_destroy_count");
      name_ = loadSymbol<decltype(name_)>("nntr_fake_context_name");
    } catch (...) {
      close();
      throw;
    }
  }

  PluginProbe(const PluginProbe &) = delete;
  PluginProbe &operator=(const PluginProbe &) = delete;

  ~PluginProbe() { close(); }

  void reset(int failures_before_success) { reset_(failures_before_success); }

  int createCount() const { return create_count_(); }
  int destroyCount() const { return destroy_count_(); }
  std::string name() const { return name_(); }

  void close() noexcept {
    if (handle_ != nullptr) {
      (void)nntrainer::DynamicLibraryLoader::freeLibrary(handle_);
      handle_ = nullptr;
    }
  }

private:
  template <typename T> T loadSymbol(const char *name) {
    void *symbol = nntrainer::DynamicLibraryLoader::loadSymbol(handle_, name);
    if (symbol == nullptr) {
      throw std::runtime_error(
        std::string("missing fake context plugin symbol ") + name + ": " +
        nntrainer::DynamicLibraryLoader::getLastErrorString());
    }
    return reinterpret_cast<T>(symbol);
  }

  void *handle_ = nullptr;
  decltype(&nntr_fake_context_reset) reset_ = nullptr;
  decltype(&nntr_fake_context_create_count) create_count_ = nullptr;
  decltype(&nntr_fake_context_destroy_count) destroy_count_ = nullptr;
  decltype(&nntr_fake_context_name) name_ = nullptr;
};

void testSequentialRegistration(nntrainer::Engine &engine) {
  const auto path = pluginPath(1);
  PluginProbe probe(path);
  probe.reset(0);
  const auto context_name = probe.name();

  require(engine.registerContext(path) == 0,
          "first sequential registration failed");
  require(engine.registerContext(path) == 0,
          "second sequential registration failed");
  require(probe.createCount() == 1,
          "same path initialized more than once sequentially");
  require(probe.destroyCount() == 0,
          "successful process-lifetime Context was destroyed");

  nntrainer::Engine secondary_engine;
  require(secondary_engine.registerContext(path) == 0,
          "secondary Engine registration failed");
  require(secondary_engine.getRegisteredContext(context_name) ==
            engine.getRegisteredContext(context_name),
          "same plugin path resolved to different Context pointers");
  require(probe.createCount() == 1,
          "secondary Engine initialized the same plugin again");

  probe.close();
  require(engine.getRegisteredContext(context_name)->getName() == context_name,
          "Engine did not retain the successful plugin DSO");
}

void testConcurrentRegistration(nntrainer::Engine &engine) {
  constexpr int caller_count = 16;
  const auto path = pluginPath(2);
  PluginProbe probe(path);
  probe.reset(0);

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  callers.reserve(caller_count);

  try {
    for (int i = 0; i < caller_count; ++i) {
      callers.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        try {
          if (engine.registerContext(path) != 0) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
  } catch (...) {
    start.store(true, std::memory_order_release);
    for (auto &caller : callers) {
      caller.join();
    }
    throw;
  }

  while (ready.load(std::memory_order_acquire) != caller_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &caller : callers) {
    caller.join();
  }

  require(failures.load(std::memory_order_relaxed) == 0,
          "concurrent context registration failed");
  require(probe.createCount() == 1,
          "same path initialized more than once concurrently");
  require(probe.destroyCount() == 0, "concurrent winner Context was destroyed");
}

void testExplicitPathAlias(nntrainer::Engine &engine) {
  const std::filesystem::path path(pluginPath(3));
  PluginProbe probe(path.string());
  probe.reset(0);

  require(engine.registerContext(path.filename().string(),
                                 path.parent_path().string()) == 0,
          "base-path registration failed");
  const auto alias = path.parent_path() / "." / path.filename();
  require(engine.registerContext(alias.string()) == 0,
          "lexical alias registration failed");
  require(probe.createCount() == 1,
          "explicit aliases did not share one plugin record");
}

void testFailureRetry(nntrainer::Engine &engine) {
  const auto path = pluginPath(4);
  PluginProbe probe(path);
  probe.reset(1);

  bool first_failed = false;
  try {
    (void)engine.registerContext(path);
  } catch (const std::invalid_argument &) {
    first_failed = true;
  }
  require(first_failed, "initial plugin factory failure was not propagated");
  require(probe.createCount() == 1,
          "failed plugin factory call count is incorrect");

  require(engine.registerContext(path) == 0,
          "context registration retry failed");
  require(engine.registerContext(path) == 0,
          "post-retry duplicate registration failed");
  require(probe.createCount() == 2,
          "failed call_once did not retry exactly once");
  require(probe.destroyCount() == 0, "successful retry Context was destroyed");
}

void testNameCollision(nntrainer::Engine &engine) {
  const auto path = pluginPath(5);
  PluginProbe probe(path);
  probe.reset(0);

  for (int attempt = 0; attempt < 2; ++attempt) {
    bool collision_detected = false;
    try {
      (void)engine.registerContext(path);
    } catch (const std::invalid_argument &) {
      collision_detected = true;
    }
    require(collision_detected,
            "different Context pointer reused an existing name");
  }

  require(probe.createCount() == 1,
          "collision path initialized more than once");
  require(probe.destroyCount() == 0,
          "process-wide collision record did not retain its Context");
}

} // namespace

int main() {
  try {
    auto &engine = nntrainer::Engine::Global();
    testSequentialRegistration(engine);
    testConcurrentRegistration(engine);
    testExplicitPathAlias(engine);
    testFailureRetry(engine);
    testNameCollision(engine);
  } catch (const std::exception &exception) {
    std::cerr << "context plugin lifecycle test failed: " << exception.what()
              << '\n';
    return 1;
  }

  return 0;
}
