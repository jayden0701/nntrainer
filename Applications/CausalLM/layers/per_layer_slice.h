// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_slice.h
 * @date   07 Apr 2026
 * @brief  Selects per-layer input chunk from packed per-layer embedding tensor.
 */

#ifndef __PER_LAYER_SLICE_H__
#define __PER_LAYER_SLICE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <base_properties.h>
#include <causallm_common_properties.h>

namespace causallm {

namespace props {
class LayerIndex : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "layer_index";
  using prop_tag = nntrainer::uint_prop_tag;
  LayerIndex(unsigned int value = 0) { set(value); }
};
} // namespace props

class PerLayerSliceLayer final : public nntrainer::Layer {
public:
  PerLayerSliceLayer() :
    Layer(), slice_props(props::FeatureSize(), props::LayerIndex()) {}

  ~PerLayerSliceLayer() {}

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  bool supportBackwarding() const override { return false; }

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  const std::string getType() const override { return type; }

  void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = nntrainer::loadProperties(values, slice_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[per_layer_slice] Unknown Layer Properties count " +
           std::to_string(values.size());
  }

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "per_layer_slice";

private:
  std::tuple<props::FeatureSize, props::LayerIndex> slice_props;
};

} // namespace causallm

#endif
