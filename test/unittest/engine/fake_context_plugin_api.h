// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file fake_context_plugin_api.h
 * @brief Test-only context plugin lifecycle probe API
 */

#ifndef __NNTRAINER_FAKE_CONTEXT_PLUGIN_API_H__
#define __NNTRAINER_FAKE_CONTEXT_PLUGIN_API_H__

#include <context.h>

#if defined(_WIN32)
#define NNTR_FAKE_CONTEXT_PLUGIN_API
#else
#define NNTR_FAKE_CONTEXT_PLUGIN_API __attribute__((visibility("default")))
#endif

extern "C" {

/** @brief Reset lifecycle counters and configure initial create failures. */
NNTR_FAKE_CONTEXT_PLUGIN_API void
nntr_fake_context_reset(int failures_before_success) noexcept;

/** @brief Return the number of factory calls. */
NNTR_FAKE_CONTEXT_PLUGIN_API int nntr_fake_context_create_count() noexcept;

/** @brief Return the number of Context destroy calls. */
NNTR_FAKE_CONTEXT_PLUGIN_API int nntr_fake_context_destroy_count() noexcept;

/** @brief Return this plugin's Context name. */
NNTR_FAKE_CONTEXT_PLUGIN_API const char *nntr_fake_context_name() noexcept;
}

#endif // __NNTRAINER_FAKE_CONTEXT_PLUGIN_API_H__
