// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_callbacks.cpp
 * @brief  Thread-safe callback registries owned by the Quick.AI API DSO
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "model_callbacks.h"

#include <algorithm>
#include <utility>

ModelExtensionRegistry &ModelExtensionRegistry::instance() {
  static ModelExtensionRegistry registry;
  return registry;
}

ModelExtensionRegistrationResult
ModelExtensionRegistry::register_extension(RegisteredModelExtension extension) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (by_architecture_.find(extension.architecture) != by_architecture_.end()) {
    return ModelExtensionRegistrationResult::DUPLICATE_ARCHITECTURE;
  }
  for (const auto &entry : by_architecture_) {
    const auto &existing = entry.second.descriptor;
    const auto &candidate = extension.descriptor;
    const bool duplicate_id =
      existing.id == candidate.id ||
      (!candidate.sd_variant_id.empty() &&
       existing.id == candidate.sd_variant_id) ||
      (!existing.sd_variant_id.empty() &&
       existing.sd_variant_id == candidate.id) ||
      (!existing.sd_variant_id.empty() && !candidate.sd_variant_id.empty() &&
       existing.sd_variant_id == candidate.sd_variant_id);
    if (duplicate_id)
      return ModelExtensionRegistrationResult::DUPLICATE_MODEL_ID;
  }

  by_architecture_.emplace(extension.architecture, std::move(extension));
  return ModelExtensionRegistrationResult::SUCCESS;
}

std::optional<RegisteredModelExtension>
ModelExtensionRegistry::lookup(const std::string &architecture) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = by_architecture_.find(architecture);
  if (iterator == by_architecture_.end())
    return std::nullopt;
  return iterator->second;
}

std::optional<RegisteredModelExtension>
ModelExtensionRegistry::find_by_model_id(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &entry : by_architecture_) {
    if (entry.second.descriptor.id == model_id ||
        (!entry.second.descriptor.sd_variant_id.empty() &&
         entry.second.descriptor.sd_variant_id == model_id)) {
      return entry.second;
    }
  }
  return std::nullopt;
}

bool ModelExtensionRegistry::has_model_id(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &entry : by_architecture_) {
    if (entry.second.descriptor.id == model_id ||
        (!entry.second.descriptor.sd_variant_id.empty() &&
         entry.second.descriptor.sd_variant_id == model_id)) {
      return true;
    }
  }
  return false;
}

std::vector<RegisteredModelExtension> ModelExtensionRegistry::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<RegisteredModelExtension> records;
  records.reserve(by_architecture_.size());
  for (const auto &entry : by_architecture_)
    records.push_back(entry.second);
  std::sort(records.begin(), records.end(),
            [](const RegisteredModelExtension &left,
               const RegisteredModelExtension &right) {
              if (left.descriptor.id != right.descriptor.id)
                return left.descriptor.id < right.descriptor.id;
              return left.architecture < right.architecture;
            });
  return records;
}
