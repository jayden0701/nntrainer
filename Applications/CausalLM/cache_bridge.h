// SPDX-License-Identifier: Apache-2.0
/** @file cache_bridge.h
 * @brief Synchronous CPU cross-model KV handoff CLI.
 */
#ifndef NNTRAINER_CAUSALLM_CACHE_BRIDGE_H
#define NNTRAINER_CAUSALLM_CACHE_BRIDGE_H

#include <json.hpp>

namespace causallm {
class CausalLM;

/** Run exact-token CacheBridge requests on two resident Qwen3 models. */
void runCacheBridge(CausalLM &source, CausalLM &target,
                    const nlohmann::json &config);
} // namespace causallm

#endif
