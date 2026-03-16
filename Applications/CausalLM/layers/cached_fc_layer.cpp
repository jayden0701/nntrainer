// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   cached_fc_layer.cpp
 * @brief  Cached fully connected layer for CausalLM
 */

#include <cached_fc_layer.h>

#include <algorithm>
#include <limits>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum CachedFCParams {
  WEIGHT = 0,
  BIAS = 1,
};

enum CachedFCTensors {
  CACHE = 0,
};

CachedFCLayer::CachedFCLayer() :
  LayerImpl(),
  cached_fc_props(nntrainer::props::Unit(), props::CacheMode(),
                  props::EnableCache()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

void CachedFCLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "cached_fc_layer takes exactly one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  const auto unit = std::get<nntrainer::props::Unit>(cached_fc_props).get();
  const bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  auto in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  auto out_dim = in_dim;
  if (is_nchw)
    out_dim.width(unit);
  else
    out_dim.channel(unit);

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
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  weight_idx[CachedFCParams::WEIGHT] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[CachedFCParams::BIAS] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  tensor_idx[CachedFCTensors::CACHE] = context.requestTensor(
    out_dim, "projection_cache", nntrainer::Initializer::NONE, true,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void CachedFCLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, cached_fc_props);
  LayerImpl::setProperty(remain_props);

  auto &cache_mode = std::get<props::CacheMode>(cached_fc_props);
  if (!cache_mode.empty()) {
    const auto &mode = cache_mode.get();
    if (mode != "one_time" && mode != "incremental") {
      throw std::invalid_argument("unknown cache_mode: " + mode);
    }
  }
}

void CachedFCLayer::runProjection(nntrainer::Tensor &input,
                                  nntrainer::Tensor &output,
                                  nntrainer::Tensor &weight,
                                  nntrainer::Tensor *bias) const {
  input.dot(weight, output, false, false);
  if (bias != nullptr) {
    output.add_i(*bias);
  }
}

CachedFCLayer::RuntimeState &
CachedFCLayer::getRuntimeState(nntrainer::RunLayerContext &context) {
  std::lock_guard<std::mutex> guard(runtime_state_mutex);
  auto [it, inserted] =
    runtime_state.try_emplace(&context, RuntimeState{false, 0u});
  return it->second;
}

void CachedFCLayer::resetRuntimeState(nntrainer::RunLayerContext &context) {
  std::lock_guard<std::mutex> guard(runtime_state_mutex);
  runtime_state[&context] = RuntimeState{false, 0u};
}

void CachedFCLayer::copyCacheToOutput(nntrainer::RunLayerContext &context,
                                      unsigned int cached_length) {
  nntrainer::Tensor &cache =
    context.getTensor(tensor_idx[CachedFCTensors::CACHE]);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  auto cache_dim = cache.getDim();
  auto output_dim = output.getDim();

  cached_length =
    std::min(cached_length, static_cast<unsigned int>(std::min(
                              cache_dim.height(), output_dim.height())));

  auto copied_dim = output_dim;
  copied_dim.height(cached_length);

  for (unsigned int b = 0; b < output_dim.batch(); ++b) {
    nntrainer::Tensor cache_slice = cache.getSharedDataTensor(
      copied_dim, b * cache_dim.getFeatureLen(), true);
    nntrainer::Tensor output_slice = output.getSharedDataTensor(
      copied_dim, b * output_dim.getFeatureLen(), true);
    output_slice.copy(cache_slice);
  }
}

void CachedFCLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  auto &input = context.getInput(SINGLE_INOUT_IDX);
  auto &output = context.getOutput(SINGLE_INOUT_IDX);
  auto weight = context.getWeight(weight_idx[CachedFCParams::WEIGHT]);

  nntrainer::Tensor *bias = nullptr;
  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    bias = &context.getWeight(weight_idx[CachedFCParams::BIAS]);
  }

  auto &cache_mode = std::get<props::CacheMode>(cached_fc_props);
  auto &enable_cache = std::get<props::EnableCache>(cached_fc_props);
  auto &state = getRuntimeState(context);

  if (enable_cache.get() == false) {
    runProjection(input, output, weight, bias);
    state.initialized = false;
    state.cached_length = 0;
    return;
  }

  if (cache_mode.get() == "one_time") {
    if (!state.initialized) {
      runProjection(input, output, weight, bias);
      auto &cache = context.getTensor(tensor_idx[CachedFCTensors::CACHE]);
      cache.copy(output);
      state.initialized = true;
      state.cached_length = output.getDim().height();
    } else {
      copyCacheToOutput(context, state.cached_length);
    }
    return;
  }

  runProjection(input, output, weight, bias);
  auto &cache = context.getTensor(tensor_idx[CachedFCTensors::CACHE]);
  cache.copy(output);
  state.initialized = true;
  state.cached_length = output.getDim().height();
}

void CachedFCLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  auto &cache_mode = std::get<props::CacheMode>(cached_fc_props);
  auto &enable_cache = std::get<props::EnableCache>(cached_fc_props);

  if (enable_cache.get() == false) {
    forwarding(context, training);
    return;
  }

  auto &state = getRuntimeState(context);

  if (cache_mode.get() == "one_time") {
    if (!state.initialized) {
      forwarding(context, training);
    } else {
      copyCacheToOutput(context, state.cached_length);
    }
    return;
  }

  NNTR_THROW_IF(to < from, std::invalid_argument)
    << "incremental_forwarding expects to >= from";

  auto &input = context.getInput(SINGLE_INOUT_IDX);
  auto &output = context.getOutput(SINGLE_INOUT_IDX);
  auto &cache = context.getTensor(tensor_idx[CachedFCTensors::CACHE]);

  auto weight = context.getWeight(weight_idx[CachedFCParams::WEIGHT]);

  nntrainer::Tensor *bias = nullptr;
  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    bias = &context.getWeight(weight_idx[CachedFCParams::BIAS]);
  }

  auto input_dim = input.getDim();
  auto output_dim = output.getDim();
  auto cache_dim = cache.getDim();

  unsigned int new_steps = to - from;
  if (!state.initialized || from == 0) {
    state.cached_length = 0;
    state.initialized = true;
  }

  NNTR_THROW_IF(state.cached_length + new_steps > cache_dim.height(),
                std::invalid_argument)
    << "cache capacity exceeded";

  auto step_in_dim = input_dim;
  step_in_dim.height(new_steps);
  auto step_out_dim = output_dim;
  step_out_dim.height(new_steps);

  for (unsigned int b = 0; b < output_dim.batch(); ++b) {
    const unsigned int input_offset =
      b * input_dim.getFeatureLen() + from * input.width();
    const unsigned int cache_offset =
      b * cache_dim.getFeatureLen() + state.cached_length * cache.width();

    nntrainer::Tensor input_step =
      input.getSharedDataTensor(step_in_dim, input_offset, true);
    nntrainer::Tensor cache_step =
      cache.getSharedDataTensor(step_out_dim, cache_offset, true);

    runProjection(input_step, cache_step, weight, bias);
  }

  state.cached_length += new_steps;
  copyCacheToOutput(context, state.cached_length);
}

void CachedFCLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for CachedFCLayer is not supported");
}

void CachedFCLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for CachedFCLayer is not supported");
}

void CachedFCLayer::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(cached_fc_props, method, this);
}

void CachedFCLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  const auto height = input_dimensions[0].height();

  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);

  auto out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();
  out_dim.height(height);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
  context.updateTensor(tensor_idx[CachedFCTensors::CACHE], out_dim);

  resetRuntimeState(context);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_cached_fc_layer() { return new CachedFCLayer(); }

void destroy_cached_fc_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_cached_fc_layer,
                                                   destroy_cached_fc_layer};
}

#endif

} // namespace causallm
