// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>
 *
 * @file   geglu.cpp
 * @date   18 March 2026
 * @brief  Implementation of fused GeGLU activation layer
 * @see		 https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug		 No known bugs except for NYI items
 *
 */

#include <acti_func.h>
#include <cmath>

#include "geglu.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace {

template <typename T>
void applyTanhGeluAndMultiply(const nntrainer::Tensor &gate,
                              const nntrainer::Tensor &up,
                              nntrainer::Tensor &out, unsigned int from,
                              unsigned int to) {
  const auto row_dim = nntrainer::TensorDim({1, 1, 1, gate.width()},
                                            gate.getTensorType());
  nntrainer::Tensor activated_gate(row_dim);

  for (unsigned int b = 0; b < gate.batch(); ++b) {
    for (unsigned int c = 0; c < gate.channel(); ++c) {
      for (unsigned int h = from; h < to; ++h) {
        const auto gate_off = gate.getIndex(b, c, h, 0);
        const auto up_off = up.getIndex(b, c, h, 0);
        const auto out_off = out.getIndex(b, c, h, 0);

        const nntrainer::Tensor gate_row =
          gate.getSharedDataTensor(row_dim, gate_off, true);
        const nntrainer::Tensor up_row =
          up.getSharedDataTensor(row_dim, up_off, true);
        nntrainer::Tensor out_row =
          out.getSharedDataTensor(row_dim, out_off, true);

        nntrainer::ActiFunc::tanhGelu<T>(gate_row, activated_gate);
        activated_gate.multiply(up_row, out_row);
      }
    }
  }
}

} // namespace

void GeGLULayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void GeGLULayer::forwarding(nntrainer::RunLayerContext &context,
                            bool training) {}

void GeGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  nntrainer::Tensor &gate = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &up = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  if (gate.getDataType() == ml::train::TensorDim::DataType::FP32) {
    applyTanhGeluAndMultiply<float>(gate, up, out, from, to);
  } else if (gate.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    applyTanhGeluAndMultiply<_FP16>(gate, up, out, from, to);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void GeGLULayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim gate_dim = context.getInput(INPUT_IDX_1).getDim();
  ml::train::TensorDim up_dim = context.getInput(INPUT_IDX_2).getDim();
  ml::train::TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  gate_dim.height(input_dimensions[0].height());
  up_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, gate_dim);
  context.updateInput(INPUT_IDX_2, up_dim);
  context.updateOutput(OUT_IDX, output_dim);
}

void GeGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {
nntrainer::Layer *create_geglu_layer() { return new causallm::GeGLULayer(); }
void destroy_geglu_layer(nntrainer::Layer *layer) { delete layer; }
nntrainer::LayerPluggable ml_train_layer_pluggable{create_geglu_layer,
                                                   destroy_geglu_layer};
}
#endif
