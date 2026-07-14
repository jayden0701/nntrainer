// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api_internal.h
 * @brief   Internal adapter hooks; not part of the installed QuickDotAI API.
 */
#ifndef __QUICK_DOT_AI_API_INTERNAL_H__
#define __QUICK_DOT_AI_API_INTERNAL_H__

#include "quick_dot_ai_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Announce that an adapter is about to enter a blocking generation call.
 *
 * This closes the cancellation window while JNI marshals arguments. Every
 * successful arm must be paired with quickAiDisarmRunCancellation(), including
 * when marshalling fails before the public run API starts.
 */
WIN_EXPORT ErrorCode quickAiArmRunCancellation(CausalLmHandle handle);

/** Clear an adapter announcement that was not already consumed by a run. */
WIN_EXPORT void quickAiDisarmRunCancellation(CausalLmHandle handle);

#ifdef __cplusplus
}
#endif

#endif // __QUICK_DOT_AI_API_INTERNAL_H__
