// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api_internal.h
 * @brief   Internal adapter hooks; not part of the installed Quick.AI API.
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __QUICK_DOT_AI_API_INTERNAL_H__
#define __QUICK_DOT_AI_API_INTERNAL_H__

#include "quick_dot_ai_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Announce that an adapter will enter a blocking generation call.
 *
 * Every successful arm must be paired with quickAiDisarmRunCancellation(),
 * including when argument marshalling fails before generation starts.
 */
WIN_EXPORT ErrorCode quickAiArmRunCancellation(CausalLmHandle handle);

/**
 * @brief Clear an adapter announcement not consumed by a generation call.
 * @param handle Handle previously passed to quickAiArmRunCancellation()
 */
WIN_EXPORT void quickAiDisarmRunCancellation(CausalLmHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* __QUICK_DOT_AI_API_INTERNAL_H__ */
