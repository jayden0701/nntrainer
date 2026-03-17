// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026
 *
 * @file   geglu.h
 * @date   18 March 2026
 * @brief  Implementation of fused GeGLU activation layer
 */

#ifndef __GEGLU_LAYER_H__
#define __GEGLU_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief GeGLU fused layer: gelu(gate) * up
 */
WIN_EXPORT class GeGLULayer final : public nntrainer::Layer {
public:
  WIN_EXPORT GeGLULayer() : Layer() {}
  WIN_EXPORT ~GeGLULayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return true; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {};

  WIN_EXPORT const std::string getType() const override {
    return GeGLULayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {};

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "geglu";
};

} // namespace causallm

#endif // __GEGLU_LAYER_H__
