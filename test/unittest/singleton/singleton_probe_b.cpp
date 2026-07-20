// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file singleton_probe_b.cpp
 * @brief Second shared-library singleton regression probe
 */

#include "singleton_probe_api.h"

#include <profiler.h>
#include <thread_manager.h>

const void *nntr_singleton_probe_b_profiler() noexcept {
  return &nntrainer::profile::Profiler::Global();
}

const void *nntr_singleton_probe_b_thread_manager() noexcept {
  return &nntrainer::ThreadManager::Global();
}

const void *nntr_singleton_probe_b_thread_manager_base() noexcept {
  return &nntrainer::Singleton<nntrainer::ThreadManager>::Global();
}
