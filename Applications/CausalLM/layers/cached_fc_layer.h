// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   cached_fc_layer.h
 * @brief  Cached fully connected layer for CausalLM
 */

#ifndef __CACHED_FC_LAYER_H__
#define __CACHED_FC_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <array>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace causallm {

namespace props {

class CacheMode : public nntrainer::Property<std::string> {
public:
  CacheMode(std::string value = "incremental") { set(std::move(value)); }
  static constexpr const char *key = "cache_mode";
  using prop_tag = nntrainer::str_prop_tag;
};

class EnableCache : public nntrainer::Property<bool> {
public:
  EnableCache(bool value = true) { set(value); }
  static constexpr const char *key = "enable_cache";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props

class CachedFCLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT CachedFCLayer();
  WIN_EXPORT ~CachedFCLayer() = default;

  WIN_EXPORT CachedFCLayer(CachedFCLayer &&rhs) noexcept = default;
  WIN_EXPORT CachedFCLayer &operator=(CachedFCLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return CachedFCLayer::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "cached_fc_layer";

private:
  struct RuntimeState {
    bool initialized;
    unsigned int cached_length;
  };

  RuntimeState &getRuntimeState(nntrainer::RunLayerContext &context);
  void resetRuntimeState(nntrainer::RunLayerContext &context);

  void runProjection(nntrainer::Tensor &input, nntrainer::Tensor &output,
                     nntrainer::Tensor &weight,
                     nntrainer::Tensor *bias = nullptr) const;

  void copyCacheToOutput(nntrainer::RunLayerContext &context,
                         unsigned int cached_length);

  std::tuple<nntrainer::props::Unit, props::CacheMode, props::EnableCache>
    cached_fc_props;
  std::array<unsigned int, 2> weight_idx;
  std::array<unsigned int, 1> tensor_idx;

  mutable std::mutex runtime_state_mutex;
  std::unordered_map<const nntrainer::RunLayerContext *, RuntimeState>
    runtime_state;
};

} // namespace causallm

#endif
#endif
