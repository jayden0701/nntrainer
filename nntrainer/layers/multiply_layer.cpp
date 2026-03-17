// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is multiply layer class (operation layer)
 *
 */

#include <multiply_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void MultiplyLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void MultiplyLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
  input0.multiply(input1, hidden);
}

void MultiplyLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(1)));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(0)));
}

void MultiplyLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multiply_props);
  if (!remain_props.empty()) {
    std::string msg = "[MultiplyLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void MultiplyLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {

  ml::train::TensorDim input_dim1 = context.getInput(0).getDim();
  ml::train::TensorDim input_dim2 = context.getInput(1).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(0, input_dim1);
  context.updateInput(1, input_dim2);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}
} /* namespace nntrainer */
