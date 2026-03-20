// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   merged_attention_cache_layer.h
 * @brief  Cache layer for T5Gemma2 merged attention KV tensors
 */

#ifndef __MERGED_ATTENTION_CACHE_LAYER_H__
#define __MERGED_ATTENTION_CACHE_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

#include <array>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace causallm {

namespace props {

class MaxDecoderCacheLen : public nntrainer::PositiveIntegerProperty {
public:
  MaxDecoderCacheLen(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "max_decoder_cache_len";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

class MergedAttentionCacheLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT MergedAttentionCacheLayer();
  WIN_EXPORT ~MergedAttentionCacheLayer() = default;

  WIN_EXPORT MergedAttentionCacheLayer(MergedAttentionCacheLayer &&rhs) noexcept = default;
  WIN_EXPORT MergedAttentionCacheLayer &operator=(MergedAttentionCacheLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override {
    return MergedAttentionCacheLayer::type;
  }
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "merged_attention_cache";

private:
  enum InputIndex : unsigned int {
    ENCODER_KEY = 0,
    ENCODER_VALUE = 1,
    DECODER_KEY = 2,
    DECODER_VALUE = 3,
  };

  enum OutputIndex : unsigned int {
    MERGED_KEY = 0,
    MERGED_VALUE = 1,
  };

  enum TensorIndex : unsigned int {
    ENCODER_KEY_CACHE = 0,
    ENCODER_VALUE_CACHE = 1,
    DECODER_KEY_CACHE = 2,
    DECODER_VALUE_CACHE = 3,
  };

  struct RuntimeState {
    bool encoder_cached;
    unsigned int encoder_cached_length;
    unsigned int decoder_cached_length;
  };

  RuntimeState &getRuntimeState(nntrainer::RunLayerContext &context);
  void resetRuntimeState(nntrainer::RunLayerContext &context);
  void cacheEncoderIfNeeded(nntrainer::RunLayerContext &context, RuntimeState &state, unsigned int from,
  unsigned int to);
  void appendDecoderChunk(nntrainer::RunLayerContext &context, RuntimeState &state,
                          unsigned int from, unsigned int to);
  void writeMergedOutputs(nntrainer::RunLayerContext &context, const RuntimeState &state);
  void copyTensorByHeight(nntrainer::Tensor &src, nntrainer::Tensor &dst,
                          unsigned int src_offset_height,
                          unsigned int dst_offset_height,
                          unsigned int copy_height) const;

  std::tuple<props::MaxDecoderCacheLen> merged_attention_cache_props;
  std::array<unsigned int, 4> tensor_idx;

  mutable std::mutex runtime_state_mutex;
  std::unordered_map<const nntrainer::RunLayerContext *, RuntimeState> runtime_state;
};

} // namespace causallm

#endif
#endif
