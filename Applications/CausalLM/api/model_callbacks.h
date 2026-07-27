// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_callbacks.h
 * @brief  Thread-safe callback registries owned by the Quick.AI API DSO
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef QUICK_DOT_AI_MODEL_CALLBACKS_H_
#define QUICK_DOT_AI_MODEL_CALLBACKS_H_

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "quick_dot_ai_extension_api.h"

/** Host-owned descriptor copied from QuickAiModelExtensionV1. */
struct RegisteredExtensionDescriptor {
  std::string id;
  std::string family;
  std::string display_name;
  uint32_t runtime = 0;
  uint32_t backend_mask = 0;
  uint32_t capabilities = 0;
  std::string config_name;
  std::string sd_variant_id;
};

/**
 * Host-owned snapshot of one process-lifetime extension registration.
 * Function pointers and user_data remain owned by the loaded plugin.
 */
struct RegisteredModelExtension {
  std::string architecture;
  uint32_t feature_mask = 0;
  RegisteredExtensionDescriptor descriptor;
  QuickAiExtensionRunOpenAIV1 run_openai = nullptr;
  QuickAiExtensionConfigureSpeculativeV1 configure_speculative = nullptr;
  void *user_data = nullptr;
};

enum class ModelExtensionRegistrationResult {
  SUCCESS,
  DUPLICATE_ARCHITECTURE,
  DUPLICATE_MODEL_ID,
};

/**
 * @brief Atomic descriptor-plus-callback extension registry.
 *
 * The registry owns one complete record per architecture. A single insertion
 * publishes the descriptor and callbacks together, preventing a catalog entry
 * without its implementation. All lookups return value snapshots and no
 * plugin callback is invoked while this registry's mutex is held.
 */
class ModelExtensionRegistry {
public:
  static ModelExtensionRegistry &instance();

  ModelExtensionRegistrationResult
  register_extension(RegisteredModelExtension extension);
  std::optional<RegisteredModelExtension>
  lookup(const std::string &architecture) const;
  std::optional<RegisteredModelExtension>
  find_by_model_id(const std::string &model_id) const;
  bool has_model_id(const std::string &model_id) const;
  std::vector<RegisteredModelExtension> snapshot() const;

private:
  ModelExtensionRegistry() = default;
  ModelExtensionRegistry(const ModelExtensionRegistry &) = delete;
  ModelExtensionRegistry &operator=(const ModelExtensionRegistry &) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, RegisteredModelExtension> by_architecture_;
};

#endif // QUICK_DOT_AI_MODEL_CALLBACKS_H_
