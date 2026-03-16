// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   cached_fc_layer.h
 * @date   16 March 2026
 * @brief  Cached fully connected projection layer for CausalLM
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

namespace causallm {

class CachedFullyConnectedLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT CachedFullyConnectedLayer();
  WIN_EXPORT ~CachedFullyConnectedLayer() = default;

  WIN_EXPORT CachedFullyConnectedLayer(CachedFullyConnectedLayer &&rhs) noexcept =
    default;
  WIN_EXPORT CachedFullyConnectedLayer &
  operator=(CachedFullyConnectedLayer &&rhs) = default;

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
    return CachedFullyConnectedLayer::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  static constexpr const char *type = "cached_fc_layer";

private:
  float lora_scaling;
  std::tuple<nntrainer::props::Unit, nntrainer::props::LoraRank,
             nntrainer::props::LoraAlpha>
    fc_props;
  std::array<unsigned int, 2> weight_idx;
  std::array<unsigned int, 4> lora_idx;
  std::unique_ptr<nntrainer::Quantizer> quantizer;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CACHED_FC_LAYER_H__ */
