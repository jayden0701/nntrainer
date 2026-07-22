// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file fake_context_plugin.cpp
 * @brief Test-only dynamically loaded Context implementation
 */

#include "fake_context_plugin_api.h"

#include <atomic>
#include <memory>
#include <string>

#include <context_data.h>

#ifndef NNTR_FAKE_CONTEXT_ID
#error "NNTR_FAKE_CONTEXT_ID must be defined"
#endif

#ifndef NNTR_FAKE_CONTEXT_NAME_ID
#define NNTR_FAKE_CONTEXT_NAME_ID NNTR_FAKE_CONTEXT_ID
#endif

#define NNTR_STRINGIFY_IMPL(value) #value
#define NNTR_STRINGIFY(value) NNTR_STRINGIFY_IMPL(value)

namespace {

constexpr char context_name[] =
  "fake_context_" NNTR_STRINGIFY(NNTR_FAKE_CONTEXT_NAME_ID);

std::atomic<int> create_count{0};
std::atomic<int> destroy_count{0};
std::atomic<int> failures_remaining{0};

class FakeContext final : public nntrainer::Context {
public:
  FakeContext() : Context(std::make_shared<nntrainer::ContextData>()) {}

  std::string getName() override { return context_name; }
};

nntrainer::Context *createContext() {
  create_count.fetch_add(1, std::memory_order_relaxed);

  int remaining = failures_remaining.load(std::memory_order_relaxed);
  while (remaining > 0 &&
         !failures_remaining.compare_exchange_weak(remaining, remaining - 1,
                                                   std::memory_order_relaxed)) {
  }
  if (remaining > 0) {
    return nullptr;
  }

  return new FakeContext();
}

void destroyContext(nntrainer::Context *context) {
  destroy_count.fetch_add(1, std::memory_order_relaxed);
  delete context;
}

} // namespace

extern "C" {

NNTR_FAKE_CONTEXT_PLUGIN_API nntrainer::ContextPluggable
  ml_train_context_pluggable{createContext, destroyContext};

NNTR_FAKE_CONTEXT_PLUGIN_API void
nntr_fake_context_reset(int failures_before_success) noexcept {
  create_count.store(0, std::memory_order_relaxed);
  destroy_count.store(0, std::memory_order_relaxed);
  failures_remaining.store(failures_before_success, std::memory_order_relaxed);
}

NNTR_FAKE_CONTEXT_PLUGIN_API int nntr_fake_context_create_count() noexcept {
  return create_count.load(std::memory_order_relaxed);
}

NNTR_FAKE_CONTEXT_PLUGIN_API int nntr_fake_context_destroy_count() noexcept {
  return destroy_count.load(std::memory_order_relaxed);
}

NNTR_FAKE_CONTEXT_PLUGIN_API const char *nntr_fake_context_name() noexcept {
  return context_name;
}
}
