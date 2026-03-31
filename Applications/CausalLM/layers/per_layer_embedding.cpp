// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.cpp
 * @date   31 March 2026
 * @brief  Per Layer Embedding layer for CausalLM.
 */

#include <per_layer_embedding.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

#include <algorithm>
#include <iostream>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

PerLayerEmbedding::PerLayerEmbedding() :
  LayerImpl(),
  ple_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
            props::LayerId(1), props::NumLayers(1), props::FlashMode(false),
            props::CacheSize(0), props::FlashWeightPath(),
            nntrainer::props::Scale()),
  weight_idx(std::numeric_limits<unsigned>::max()),
  flash_row_bytes(0) {}

void PerLayerEmbedding::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "per_layer_embedding takes exactly one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "per_layer_embedding expects channel size to be 1";

  NNTR_THROW_IF(input_dim.getDataType() != nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "per_layer_embedding only supports FP32 token index input";

  size_t vocab_size =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(ple_props));
  size_t ple_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(ple_props));
  size_t layer_id = static_cast<size_t>(std::get<props::LayerId>(ple_props));
  size_t num_layers =
    static_cast<size_t>(std::get<props::NumLayers>(ple_props));

  NNTR_THROW_IF(vocab_size == 0 || ple_dim == 0, std::invalid_argument)
    << "in_dim and out_dim must be positive";
  NNTR_THROW_IF(layer_id >= num_layers, std::invalid_argument)
    << "layer_id must be smaller than num_layers";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  nntrainer::TensorDim output_dim = input_dim;
  output_dim.height(input_dim.width());
  output_dim.width(ple_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  nntrainer::TensorDim weight_dim = output_dim;
  weight_dim.setTensorType({context.getFormat(), context.getWeightDataType()});
  weight_dim.height(vocab_size * num_layers);
  weight_dim.width(ple_dim);
  weight_dim.batch(1);

  weight_idx = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "PerLayerEmbedding", true);

  const auto &flash_path = std::get<props::FlashWeightPath>(ple_props).get();
  bool flash_mode = std::get<props::FlashMode>(ple_props);
  if (flash_mode && !flash_path.empty()) {
    flash_row_bytes = ple_dim * sizeof(float);
    flash_reader.open(flash_path, std::ios::binary);
    NNTR_THROW_IF(!flash_reader.is_open(), std::invalid_argument)
      << "flash_mode is enabled but flash_weight_path cannot be opened: "
      << flash_path;
  }
}

void PerLayerEmbedding::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, ple_props);
  LayerImpl::setProperty(remain_props);
}

void PerLayerEmbedding::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {}

bool PerLayerEmbedding::getCachedRow(size_t table_row_index, float *destination,
                                     size_t out_dim) {
  (void)out_dim;
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto cache_it = row_cache.find(table_row_index);
  if (cache_it == row_cache.end()) {
    return false;
  }

  std::copy(cache_it->second.begin(), cache_it->second.end(), destination);

  auto lru_it = lru_index.find(table_row_index);
  if (lru_it != lru_index.end()) {
    lru_keys.erase(lru_it->second);
    lru_keys.push_front(table_row_index);
    lru_it->second = lru_keys.begin();
  }

  return true;
}

void PerLayerEmbedding::cacheRow(size_t table_row_index, const float *row_data,
                                 size_t out_dim) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  unsigned int cache_size = std::get<props::CacheSize>(ple_props);
  if (cache_size == 0) {
    return;
  }

  auto cache_it = row_cache.find(table_row_index);
  if (cache_it == row_cache.end()) {
    if (row_cache.size() >= cache_size) {
      size_t remove_key = lru_keys.back();
      lru_keys.pop_back();
      lru_index.erase(remove_key);
      row_cache.erase(remove_key);
    }

    row_cache.emplace(table_row_index,
                      std::vector<float>(row_data, row_data + out_dim));
    lru_keys.push_front(table_row_index);
    lru_index[table_row_index] = lru_keys.begin();
    return;
  }

  cache_it->second.assign(row_data, row_data + out_dim);
  auto lru_it = lru_index.find(table_row_index);
  if (lru_it != lru_index.end()) {
    lru_keys.erase(lru_it->second);
  }
  lru_keys.push_front(table_row_index);
  lru_index[table_row_index] = lru_keys.begin();
}

bool PerLayerEmbedding::fetchEmbeddingRow(nntrainer::RunLayerContext &context,
                                          size_t table_row_index,
                                          float *destination) {
  size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(ple_props));

  if (getCachedRow(table_row_index, destination, out_dim)) {
    return true;
  }

  bool flash_mode = std::get<props::FlashMode>(ple_props);
  if (flash_mode && flash_reader.is_open()) {
    const std::streamoff row_offset =
      static_cast<std::streamoff>(table_row_index * flash_row_bytes);
    flash_reader.seekg(row_offset, std::ios::beg);
    if (!flash_reader.good()) {
      return false;
    }

    flash_reader.read(reinterpret_cast<char *>(destination), flash_row_bytes);
    if (static_cast<size_t>(flash_reader.gcount()) != flash_row_bytes) {
      return false;
    }

    cacheRow(table_row_index, destination, out_dim);
    return true;
  }

  nntrainer::Tensor &weight_tensor = context.getWeight(weight_idx);
  nntrainer::TensorDim row_dim({1, 1, 1, out_dim},
                               weight_tensor.getTensorType());
  nntrainer::Tensor row_view =
    weight_tensor.getSharedDataTensor(row_dim, out_dim * table_row_index);

  std::copy(row_view.getData<float>(), row_view.getData<float>() + out_dim,
            destination);
  cacheRow(table_row_index, destination, out_dim);
  return true;
}

void PerLayerEmbedding::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const size_t vocab_size =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(ple_props));
  const size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(ple_props));
  const size_t layer_id =
    static_cast<size_t>(std::get<props::LayerId>(ple_props));
  const float scale = std::get<nntrainer::props::Scale>(ple_props).empty()
                        ? 1.0f
                        : std::get<nntrainer::props::Scale>(ple_props).get();

  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim out_tensor_dim({1, 1, 1, out_dim},
                                            output.getTensorType());

  unsigned int b_size = input.batch();
  unsigned int token_count = to - from;

  for (unsigned int batch = 0; batch < b_size; ++batch) {
    float *in_data =
      input.getAddress<float>(batch * input.getDim().getFeatureLen());
    nntrainer::Tensor batch_output = output.getBatchSlice(batch, 1);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(token_count); ++i) {
      size_t token_id = static_cast<size_t>(in_data[i]);
      if (token_id >= vocab_size) {
        throw std::invalid_argument("token index exceeds vocab size");
      }

      size_t table_row = layer_id * vocab_size + token_id;

      nntrainer::Tensor out_token =
        batch_output.getSharedDataTensor(out_tensor_dim, out_dim * i);
      float *out_data = out_token.getData<float>();

      if (!fetchEmbeddingRow(context, table_row, out_data)) {
        throw std::runtime_error(
          "failed to fetch per-layer embedding row from flash or memory");
      }

      if (scale != 1.0f) {
        out_token.multiply_i(scale);
      }
    }
  }
}

void PerLayerEmbedding::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for per_layer_embedding is not supported");
}

void PerLayerEmbedding::calcGradient(nntrainer::RunLayerContext &context) {}

void PerLayerEmbedding::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(ple_props, method, this);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_per_layer_embedding() {
  auto layer = new PerLayerEmbedding();
  std::cout << "per_layer_embedding layer created\n";
  return layer;
}

void destroy_per_layer_embedding(nntrainer::Layer *layer) {
  std::cout << "per_layer_embedding layer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_per_layer_embedding,
                                                   destroy_per_layer_embedding};
}

#endif

} // namespace causallm
