// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   fused_fc_reshaped_rms_norm.h
 * @brief  Fused fully-connected + reshaped RMSNorm layer for CausalLM.
 */

#ifndef __FUSED_FC_RESHAPED_RMS_NORM_H__
#define __FUSED_FC_RESHAPED_RMS_NORM_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <causallm_common_properties.h>
#include <layer_impl.h>

#include <array>
#include <tuple>

namespace causallm {

class FusedFCReshapedRMSNormLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT FusedFCReshapedRMSNormLayer();
  WIN_EXPORT ~FusedFCReshapedRMSNormLayer() = default;

  WIN_EXPORT FusedFCReshapedRMSNormLayer(
    FusedFCReshapedRMSNormLayer &&rhs) noexcept = default;
  WIN_EXPORT FusedFCReshapedRMSNormLayer &
  operator=(FusedFCReshapedRMSNormLayer &&rhs) = default;

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
    return FusedFCReshapedRMSNormLayer::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "fused_fc_reshaped_rms_norm";

private:
  void runProjection(nntrainer::Tensor &input, nntrainer::Tensor &output,
                     nntrainer::Tensor &weight,
                     nntrainer::Tensor *bias = nullptr) const;

  void normalizeProjection(nntrainer::Tensor &output,
                           const nntrainer::Tensor &gamma) const;

  std::tuple<nntrainer::props::Unit, nntrainer::props::Epsilon,
             props::FeatureSize>
    fused_props;
  std::array<unsigned int, 3> weight_idx;
};

} // namespace causallm

#endif
#endif
