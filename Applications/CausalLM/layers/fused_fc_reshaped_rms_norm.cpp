// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   fused_fc_reshaped_rms_norm.cpp
 * @brief  Fused fully-connected + reshaped RMSNorm layer for CausalLM.
 */

#include <fused_fc_reshaped_rms_norm.h>

#include <cmath>
#include <limits>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FusedParams {
  WEIGHT = 0,
  BIAS = 1,
  GAMMA = 2,
};

FusedFCReshapedRMSNormLayer::FusedFCReshapedRMSNormLayer() :
  LayerImpl(),
  fused_props(nntrainer::props::Unit(), nntrainer::props::Epsilon(),
              props::FeatureSize()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FusedFCReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "fused_fc_reshaped_rms_norm takes exactly one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  const auto unit = std::get<nntrainer::props::Unit>(fused_props).get();
  const auto feature_size = std::get<props::FeatureSize>(fused_props).get();
  const bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  auto in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  auto out_dim = in_dim;
  if (is_nchw)
    out_dim.width(unit);
  else
    out_dim.channel(unit);

  NNTR_THROW_IF(out_dim.width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of fused output width";

  out_dim.setTensorType({context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({out_dim});

  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ml::train::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));

  weight_idx[FusedParams::WEIGHT] = context.requestWeight(
    weight_dim, nntrainer::Initializer::NONE, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FusedParams::BIAS] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  weight_idx[FusedParams::GAMMA] = context.requestWeight(
    gamma_dim, nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "gamma", false);
}

void FusedFCReshapedRMSNormLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fused_props);
  LayerImpl::setProperty(remain_props);
}

void FusedFCReshapedRMSNormLayer::runProjection(nntrainer::Tensor &input,
                                                nntrainer::Tensor &output,
                                                nntrainer::Tensor &weight,
                                                nntrainer::Tensor *bias) const {
  input.dot(weight, output, false, false);
  if (bias != nullptr) {
    output.add_i(*bias);
  }
}

void FusedFCReshapedRMSNormLayer::normalizeProjection(
  nntrainer::Tensor &output, const nntrainer::Tensor &gamma) const {
  const auto feature_size = std::get<props::FeatureSize>(fused_props).get();
  const auto epsilon = std::get<nntrainer::props::Epsilon>(fused_props).get();
  auto out_dim = output.getDim();

  NNTR_THROW_IF(output.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "fused_fc_reshaped_rms_norm currently supports FP32 activations only";

  float *out_data = output.getData<float>();
  const float *gamma_data = gamma.getData<float>();
  const unsigned int width = out_dim.width();
  const unsigned int height = out_dim.height();
  const unsigned int rows_per_height = width / feature_size;

  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int row = 0; row < rows_per_height; ++row) {
      float *segment = out_data + h * width + row * feature_size;
      float sum_sq = 0.0f;
      for (unsigned int i = 0; i < feature_size; ++i) {
        sum_sq += segment[i] * segment[i];
      }

      const float inv_rms =
        1.0f / std::sqrt(sum_sq / static_cast<float>(feature_size) + epsilon);

      for (unsigned int i = 0; i < feature_size; ++i) {
        segment[i] = segment[i] * inv_rms * gamma_data[i];
      }
    }
  }
}

void FusedFCReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                             bool training) {
  auto &input = context.getInput(SINGLE_INOUT_IDX);
  auto &output = context.getOutput(SINGLE_INOUT_IDX);
  auto &weight = context.getWeight(weight_idx[FusedParams::WEIGHT]);
  auto &gamma = context.getWeight(weight_idx[FusedParams::GAMMA]);

  nntrainer::Tensor *bias = nullptr;
  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    bias = &context.getWeight(weight_idx[FusedParams::BIAS]);
  }

  runProjection(input, output, weight, bias);
  normalizeProjection(output, gamma);
}

void FusedFCReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(to < from, std::invalid_argument)
    << "incremental_forwarding expects to >= from";

  auto &input = context.getInput(SINGLE_INOUT_IDX);
  auto &output = context.getOutput(SINGLE_INOUT_IDX);
  auto &weight = context.getWeight(weight_idx[FusedParams::WEIGHT]);
  auto &gamma = context.getWeight(weight_idx[FusedParams::GAMMA]);

  nntrainer::Tensor *bias = nullptr;
  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    bias = &context.getWeight(weight_idx[FusedParams::BIAS]);
  }

  auto input_dim = input.getDim();
  auto output_dim = output.getDim();
  const unsigned int step_height = to - from;

  if (step_height == 0) {
    return;
  }

  auto step_in_dim = input_dim;
  step_in_dim.batch(1);
  step_in_dim.height(step_height);

  auto step_out_dim = output_dim;
  step_out_dim.batch(1);
  step_out_dim.height(step_height);

  for (unsigned int b = 0; b < output_dim.batch(); ++b) {
    const unsigned int input_offset =
      b * input_dim.getFeatureLen() + from * input_dim.width();
    const unsigned int output_offset =
      b * output_dim.getFeatureLen() + from * output_dim.width();

    nntrainer::Tensor input_step =
      input.getSharedDataTensor(step_in_dim, input_offset, true);
    nntrainer::Tensor output_step =
      output.getSharedDataTensor(step_out_dim, output_offset, true);

    runProjection(input_step, output_step, weight, bias);
    normalizeProjection(output_step, gamma);
  }
}

void FusedFCReshapedRMSNormLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for FusedFCReshapedRMSNormLayer is not supported");
}

void FusedFCReshapedRMSNormLayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for FusedFCReshapedRMSNormLayer is not supported");
}

void FusedFCReshapedRMSNormLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fused_props, method, this);
}

void FusedFCReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  const auto height = input_dimensions[0].height();

  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);

  auto out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();
  out_dim.height(height);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_fused_fc_reshaped_rms_norm_layer() {
  return new FusedFCReshapedRMSNormLayer();
}

void destroy_fused_fc_reshaped_rms_norm_layer(nntrainer::Layer *layer) {
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_fused_fc_reshaped_rms_norm_layer,
  destroy_fused_fc_reshaped_rms_norm_layer};
}

#endif

} // namespace causallm
