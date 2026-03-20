// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   merged_attention_cache_layer.cpp
 * @brief  Cache layer for T5Gemma2 merged attention KV tensors
 */

#include <merged_attention_cache_layer.h>

#include <algorithm>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

MergedAttentionCacheLayer::MergedAttentionCacheLayer() :
  LayerImpl(), merged_attention_cache_props(props::MaxDecoderCacheLen()) {
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void MergedAttentionCacheLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 4, std::invalid_argument)
    << "merged_attention_cache takes exactly four inputs";

  const auto &encoder_key_dim = context.getInputDimensions()[ENCODER_KEY];
  const auto &encoder_value_dim = context.getInputDimensions()[ENCODER_VALUE];
  const auto &decoder_key_dim = context.getInputDimensions()[DECODER_KEY];
  const auto &decoder_value_dim = context.getInputDimensions()[DECODER_VALUE];

  NNTR_THROW_IF(encoder_key_dim.width() != decoder_key_dim.width(),
                std::invalid_argument)
    << "encoder/decoder key widths must match";
  NNTR_THROW_IF(encoder_value_dim.width() != decoder_value_dim.width(),
                std::invalid_argument)
    << "encoder/decoder value widths must match";

  const unsigned int max_decoder_cache_len =
    std::get<props::MaxDecoderCacheLen>(merged_attention_cache_props).get();

  auto merged_key_dim = encoder_key_dim;
  merged_key_dim.height(encoder_key_dim.height() + max_decoder_cache_len);
  merged_key_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  auto merged_value_dim = encoder_value_dim;
  merged_value_dim.height(encoder_value_dim.height() + max_decoder_cache_len);
  merged_value_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions({merged_key_dim, merged_value_dim});


  tensor_idx[ENCODER_KEY_CACHE] = context.requestTensor(
    encoder_key_dim, "encoder_key_cache", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  tensor_idx[ENCODER_VALUE_CACHE] = context.requestTensor(
    encoder_value_dim, "encoder_value_cache", nntrainer::Initializer::NONE,
    false, nntrainer::TensorLifespan::MAX_LIFESPAN);

  auto decoder_key_cache_dim = decoder_key_dim;
  decoder_key_cache_dim.height(max_decoder_cache_len);
  tensor_idx[DECODER_KEY_CACHE] = context.requestTensor(
    decoder_key_cache_dim, "decoder_key_cache", nntrainer::Initializer::NONE,
    false, nntrainer::TensorLifespan::MAX_LIFESPAN);

  auto decoder_value_cache_dim = decoder_value_dim;
  decoder_value_cache_dim.height(max_decoder_cache_len);
  tensor_idx[DECODER_VALUE_CACHE] =
    context.requestTensor(decoder_value_cache_dim, "decoder_value_cache",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void MergedAttentionCacheLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, merged_attention_cache_props);
  LayerImpl::setProperty(remain_props);
}

MergedAttentionCacheLayer::RuntimeState &
MergedAttentionCacheLayer::getRuntimeState(
  nntrainer::RunLayerContext &context) {
  std::lock_guard<std::mutex> guard(runtime_state_mutex);
  auto [it, inserted] =
    runtime_state.try_emplace(&context, RuntimeState{false, 0u, 0u});
  return it->second;
}

void MergedAttentionCacheLayer::resetRuntimeState(
  nntrainer::RunLayerContext &context) {
  std::lock_guard<std::mutex> guard(runtime_state_mutex);
  runtime_state[&context] = RuntimeState{false, 0u, 0u};
}

void MergedAttentionCacheLayer::copyTensorByHeight(
  nntrainer::Tensor &src, nntrainer::Tensor &dst,
  unsigned int src_offset_height, unsigned int dst_offset_height,
  unsigned int copy_height) const {
  if (copy_height == 0) {
    return;
  }

  auto src_dim = src.getDim();
  auto dst_dim = dst.getDim();
  auto copy_dim = src_dim;
  copy_dim.height(copy_height);

  for (unsigned int b = 0; b < src_dim.batch(); ++b) {
    const unsigned int src_offset =
      b * src_dim.getFeatureLen() + src_offset_height * src.width();
    const unsigned int dst_offset =
      b * dst_dim.getFeatureLen() + dst_offset_height * dst.width();

    nntrainer::Tensor src_slice =
      src.getSharedDataTensor(copy_dim, src_offset, true);
    nntrainer::Tensor dst_slice =
      dst.getSharedDataTensor(copy_dim, dst_offset, true);
    dst_slice.copy(src_slice);
  }
}

void MergedAttentionCacheLayer::cacheEncoderIfNeeded(
  nntrainer::RunLayerContext &context, RuntimeState &state, unsigned int from = 0,
  unsigned int to = 1) {
  if (state.encoder_cached) {
    return;
  }

  auto &encoder_key = context.getInput(ENCODER_KEY);
  auto &encoder_value = context.getInput(ENCODER_VALUE);
  auto &encoder_key_cache = context.getTensor(tensor_idx[ENCODER_KEY_CACHE]);
  auto &encoder_value_cache =
    context.getTensor(tensor_idx[ENCODER_VALUE_CACHE]);

  copyTensorByHeight(encoder_key, encoder_key_cache, 0, 0,
                     encoder_key.getDim().height());
  copyTensorByHeight(encoder_value, encoder_value_cache, 0, 0,
                     encoder_value.getDim().height());
  state.encoder_cached = true;

  // initial height is encoder_seq_len + (to - from)
  state.encoder_cached_length = encoder_key.getDim().height() - (to - from);
}

void MergedAttentionCacheLayer::appendDecoderChunk(
  nntrainer::RunLayerContext &context, RuntimeState &state, unsigned int from,
  unsigned int to) {
  NNTR_THROW_IF(to < from, std::invalid_argument)
    << "merged_attention_cache expects to >= from";

  const unsigned int requested_steps = to - from;
  if (requested_steps == 0) {
    return;
  }

  auto &decoder_key = context.getInput(DECODER_KEY);
  auto &decoder_value = context.getInput(DECODER_VALUE);
  auto &decoder_key_cache = context.getTensor(tensor_idx[DECODER_KEY_CACHE]);
  auto &decoder_value_cache =
    context.getTensor(tensor_idx[DECODER_VALUE_CACHE]);

  const unsigned int input_height = decoder_key.getDim().height();
  // decoder side key/value is always from 0
  // change this if from - to mechanism changes
  const unsigned int src_offset_height = 0;



  NNTR_THROW_IF(state.decoder_cached_length + requested_steps >
                  decoder_key_cache.getDim().height(),
                std::invalid_argument)
    << "decoder key cache capacity exceeded";
  NNTR_THROW_IF(state.decoder_cached_length + requested_steps >
                  decoder_value_cache.getDim().height(),
                std::invalid_argument)
    << "decoder value cache capacity exceeded";

  copyTensorByHeight(decoder_key, decoder_key_cache, src_offset_height,
                     state.decoder_cached_length, requested_steps);
  copyTensorByHeight(decoder_value, decoder_value_cache, src_offset_height,
                     state.decoder_cached_length, requested_steps);
  state.decoder_cached_length += requested_steps;
}

void MergedAttentionCacheLayer::writeMergedOutputs(
  nntrainer::RunLayerContext &context, const RuntimeState &state) {
  auto &encoder_key_cache = context.getTensor(tensor_idx[ENCODER_KEY_CACHE]);
  auto &encoder_value_cache =
    context.getTensor(tensor_idx[ENCODER_VALUE_CACHE]);
  auto &decoder_key_cache = context.getTensor(tensor_idx[DECODER_KEY_CACHE]);
  auto &decoder_value_cache =
    context.getTensor(tensor_idx[DECODER_VALUE_CACHE]);
  auto &merged_key = context.getOutput(MERGED_KEY);
  auto &merged_value = context.getOutput(MERGED_VALUE);

  const unsigned int encoder_height =
    state.encoder_cached ? state.encoder_cached_length : 0u;

  // concat {K/V, crossed_K/V}

  copyTensorByHeight(decoder_key_cache, merged_key, 0, 0,
                     state.decoder_cached_length);

  copyTensorByHeight(encoder_key_cache, merged_key, 0,
                     state.decoder_cached_length, encoder_height);

  copyTensorByHeight(decoder_value_cache, merged_value, 0, 0,
                     state.decoder_cached_length);
  copyTensorByHeight(encoder_value_cache, merged_value, 0,
                     state.decoder_cached_length, encoder_height);


  std::string d1 = "decoder_layer0";
  size_t found = context.getName().find(d1);
  if (found == std::string::npos) {
    return;
  }

  auto &encoder_key = context.getInput(ENCODER_KEY);
  auto &encoder_value = context.getInput(ENCODER_VALUE);

  auto &decoder_key = context.getInput(DECODER_KEY);
  auto &decoder_value = context.getInput(DECODER_VALUE);

  encoder_key.print(std::cout);
  encoder_value.print(std::cout);

  decoder_key.print(std::cout);
  decoder_value.print(std::cout);

  std::cout << "----------------------------from here cache-------------------" << std::endl;

  encoder_key_cache.print(std::cout);
  encoder_value_cache.print(std::cout);

  decoder_key_cache.print(std::cout);
  decoder_value_cache.print(std::cout);


  merged_key.print(std::cout);
  merged_value.print(std::cout);
}

void MergedAttentionCacheLayer::forwarding(nntrainer::RunLayerContext &context,
                                           bool training) {
  auto &state = getRuntimeState(context);
  state = RuntimeState{false, 0u, 0u};
  cacheEncoderIfNeeded(context, state);
  appendDecoderChunk(context, state, 0,
                     context.getInput(DECODER_KEY).getDim().height());
  writeMergedOutputs(context, state);
}

void MergedAttentionCacheLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &state = getRuntimeState(context);

  // First time accessing this layer
  if (from == 0) {
    state = RuntimeState{false, 0u, 0u};
  }

  std::string d1 = "decoder_layer0";
  size_t found = context.getName().find(d1);
  if (found != std::string::npos) {
    std::cout << "tmtm" << std::endl;
  }

  cacheEncoderIfNeeded(context, state, from, to);
  appendDecoderChunk(context, state, from, to);
  writeMergedOutputs(context, state);
}

void MergedAttentionCacheLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for MergedAttentionCacheLayer is not supported");
}

void MergedAttentionCacheLayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for MergedAttentionCacheLayer is not supported");
}

void MergedAttentionCacheLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(merged_attention_cache_props, method, this);
}

void MergedAttentionCacheLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto encoder_key_dim = context.getInput(ENCODER_KEY).getDim();
  auto encoder_value_dim = context.getInput(ENCODER_VALUE).getDim();
  auto decoder_key_dim = context.getInput(DECODER_KEY).getDim();
  auto decoder_value_dim = context.getInput(DECODER_VALUE).getDim();

  encoder_key_dim.height(input_dimensions[0].height());
  encoder_value_dim.height(input_dimensions[0].height());
  decoder_key_dim.height(input_dimensions[0].height());
  decoder_value_dim.height(input_dimensions[0].height());

  context.updateInput(ENCODER_KEY, encoder_key_dim);
  context.updateInput(ENCODER_VALUE, encoder_value_dim);
  context.updateInput(DECODER_KEY, decoder_key_dim);
  context.updateInput(DECODER_VALUE, decoder_value_dim);

  // TODO : I think cache(tensor) should not be updated

  // context.updateTensor(tensor_idx[ENCODER_KEY_CACHE], encoder_key_dim);
  // context.updateTensor(tensor_idx[ENCODER_VALUE_CACHE], encoder_value_dim);

  // auto decoder_key_cache_dim =
  // context.getTensor(tensor_idx[DECODER_KEY_CACHE]).getDim();
  // decoder_key_cache_dim.width(decoder_key_dim.width());
  // context.updateTensor(tensor_idx[DECODER_KEY_CACHE], decoder_key_cache_dim);

  // auto decoder_value_cache_dim =
  // context.getTensor(tensor_idx[DECODER_VALUE_CACHE]).getDim();
  // decoder_value_cache_dim.width(decoder_value_dim.width());
  // context.updateTensor(tensor_idx[DECODER_VALUE_CACHE],
  // decoder_value_cache_dim);

  auto merged_key_dim = context.getOutput(MERGED_KEY).getDim();
  // merged_key_dim.width(encoder_key_dim.width());
  merged_key_dim.height(input_dimensions[0].height());
  context.updateOutput(MERGED_KEY, merged_key_dim);

  auto merged_value_dim = context.getOutput(MERGED_VALUE).getDim();
  // merged_value_dim.width(encoder_value_dim.width());
  merged_value_dim.height(input_dimensions[0].height());
  context.updateOutput(MERGED_VALUE, merged_value_dim);

  // resetRuntimeState(context);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_merged_attention_cache_layer() {
  return new MergedAttentionCacheLayer();
}

void destroy_merged_attention_cache_layer(nntrainer::Layer *layer) {
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_merged_attention_cache_layer, destroy_merged_attention_cache_layer};
}

#endif

} // namespace causallm
