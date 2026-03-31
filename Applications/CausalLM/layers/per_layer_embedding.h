// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_embedding.h
 * @date   31 March 2026
 * @brief  Per Layer Embedding layer for CausalLM.
 */

#ifndef __PER_LAYER_EMBEDDING_H__
#define __PER_LAYER_EMBEDDING_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <cstddef>
#include <fstream>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace causallm {

namespace props {

class LayerId : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "layer_id";
  using prop_tag = nntrainer::uint_prop_tag;
};

class NumLayers : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "num_layers";
  using prop_tag = nntrainer::uint_prop_tag;
};

class FlashMode : public nntrainer::Property<bool> {
public:
  FlashMode(bool val = false) : nntrainer::Property<bool>(val) {}
  static constexpr const char *key = "flash_mode";
  using prop_tag = nntrainer::bool_prop_tag;
};

class CacheSize : public nntrainer::Property<unsigned int> {
public:
  CacheSize(unsigned int val = 0) : nntrainer::Property<unsigned int>(val) {}
  static constexpr const char *key = "cache_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

class FlashWeightPath : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "flash_weight_path";
  using prop_tag = nntrainer::str_prop_tag;
};

} // namespace props

WIN_EXPORT class PerLayerEmbedding : public nntrainer::LayerImpl {
public:
  WIN_EXPORT PerLayerEmbedding();
  WIN_EXPORT ~PerLayerEmbedding() = default;

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
    return PerLayerEmbedding::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "per_layer_embedding";

private:
  bool fetchEmbeddingRow(nntrainer::RunLayerContext &context,
                         size_t table_row_index, float *destination);

  void cacheRow(size_t table_row_index, const float *row_data, size_t out_dim);

  bool getCachedRow(size_t table_row_index, float *destination, size_t out_dim);

  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             props::LayerId, props::NumLayers, props::FlashMode,
             props::CacheSize, props::FlashWeightPath, nntrainer::props::Scale>
    ple_props;

  unsigned int weight_idx;

  std::unordered_map<size_t, std::vector<float>> row_cache;
  std::list<size_t> lru_keys;
  std::unordered_map<size_t, std::list<size_t>::iterator> lru_index;
  std::mutex cache_mutex;

  std::ifstream flash_reader;
  size_t flash_row_bytes;
};

} // namespace causallm

#endif // __PER_LAYER_EMBEDDING_H__
