// SPDX-License-Identifier: Apache-2.0
/**
 * @file   geglu.cpp
 * @date   18 March 2026
 * @brief  Implementation of fused GeGLU activation layer
 */

#include <acti_func.h>
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
  auto step_dim = gate.getDim();
  step_dim.batch(1);
  step_dim.height(to - from);

  for (unsigned int b = 0; b < gate.batch(); ++b) {
    const auto step_offset = b * gate.getDim().getFeatureLen() +
                             from * gate.getDim().width();
    const nntrainer::Tensor gate_step =
      gate.getSharedDataTensor(step_dim, step_offset, false);
    const nntrainer::Tensor up_step =
      up.getSharedDataTensor(step_dim, step_offset, false);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(step_dim, step_offset, false);
    nntrainer::Tensor activated_gate(step_dim);

    nntrainer::ActiFunc::tanhGelu<T>(gate_step, activated_gate);
    activated_gate.multiply_strided(up_step, out_step);
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

  context.updateTensor(INPUT_IDX_1, gate_dim);
  context.updateTensor(INPUT_IDX_2, up_dim);
  context.updateTensor(OUT_IDX, output_dim);
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
