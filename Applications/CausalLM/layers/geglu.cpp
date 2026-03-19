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
#include <cmath>

#include "geglu.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace {
inline float tanh_gelu(float x) {
  static constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  static constexpr float coeff = 0.044715f;
  return 0.5f * x *
         (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
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

  const unsigned int iter = to - from;

  if (gate.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < gate.batch(); ++b) {
      for (unsigned int c = 0; c < gate.channel(); ++c) {
        for (unsigned int h = 0; h < iter; ++h) {
          auto gate_off = gate.getIndex(b, c, h, 0);
          auto up_off = up.getIndex(b, c, h, 0);
          auto out_off = out.getIndex(b, c, h, 0);

          float *gate_ptr = gate.getData<float>() + gate_off;
          float *up_ptr = up.getData<float>() + up_off;
          float *out_ptr = out.getData<float>() + out_off;

          for (unsigned int w = 0; w < gate.width(); ++w) {
            out_ptr[w] = tanh_gelu(gate_ptr[w]) * up_ptr[w];
          }
        }
      }
    }
  } else if (gate.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < gate.batch(); ++b) {
      for (unsigned int c = 0; c < gate.channel(); ++c) {
        for (unsigned int h = 0; h < iter; ++h) {
          auto gate_off = gate.getIndex(b, c, h, 0);
          auto up_off = up.getIndex(b, c, h, 0);
          auto out_off = out.getIndex(b, c, h, 0);

          _FP16 *gate_ptr = gate.getData<_FP16>() + gate_off;
          _FP16 *up_ptr = up.getData<_FP16>() + up_off;
          _FP16 *out_ptr = out.getData<_FP16>() + out_off;

          for (unsigned int w = 0; w < gate.width(); ++w) {
            out_ptr[w] = static_cast<_FP16>(
              tanh_gelu(static_cast<float>(gate_ptr[w])) *
              static_cast<float>(up_ptr[w]));
          }
        }
      }
    }
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
