// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   snapkv_policy.h
 * @date   03 August 2026
 * @brief  CPU helpers for SnapKV prompt-cache eviction
 * @see    https://arxiv.org/abs/2404.14469
 */

#ifndef CAUSALLM_SNAPKV_POLICY_H_
#define CAUSALLM_SNAPKV_POLICY_H_

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Supported one-dimensional pooling operations for SnapKV scores.
 */
enum class SnapKVPooling { MAX, AVERAGE };

/**
 * @brief Logical and physical cursor state for a compacted KV cache.
 */
struct SnapKVCachePositionState {
  unsigned int logical_position = 0;
  unsigned int physical_position = 0;
  unsigned int logical_to_physical_offset = 0;
  bool has_compacted = false;
};

/**
 * @brief Parse a user-facing SnapKV pooling name.
 * @param[in] value `max`, `avg`, or `average` (case-insensitive)
 * @return parsed pooling operation
 * @throw std::invalid_argument for an unsupported value
 */
SnapKVPooling parseSnapKVPooling(const std::string &value);

/**
 * @brief Pure CPU implementation of SnapKV selection and cache compaction.
 *
 * Attention tensors use nntrainer's causal triangular layout:
 *
 *   attention[(q * (q + 1) / 2 + key) * query_heads + query_head]
 *
 * where `0 <= key <= q < prompt_length`. Cache tensors use token-major
 * `[token][kv_head][head_dim]` layout.
 */
class SnapKVPolicy {
public:
  /**
   * @brief Validate the configured cache geometry.
   */
  static void validateConfig(unsigned int cache_capacity,
                             unsigned int observation_window,
                             unsigned int pooling_kernel);

  /**
   * @brief Map an absolute logical position to the next physical cache row.
   *        Setting position zero starts a fresh, uncompacted sequence.
   */
  static SnapKVCachePositionState
  mapLogicalPosition(const SnapKVCachePositionState &state,
                     unsigned int logical_position);

  /**
   * @brief Advance logical/physical cursors after one successful MHA step.
   */
  static SnapKVCachePositionState
  advanceCachePosition(const SnapKVCachePositionState &state,
                       unsigned int step_size, bool compact_current_step,
                       unsigned int cache_capacity);

  /**
   * @brief Sum observation-window attention for each query head.
   *
   * The returned layout is `[query_head][prefix_position]`. GQA reduction is
   * intentionally deferred until after pooling because max pooling and group
   * reduction do not commute.
   */
  static std::vector<float> observationScores(const float *attention,
                                              size_t attention_elements,
                                              unsigned int prompt_length,
                                              unsigned int observation_window,
                                              unsigned int query_heads);

  /**
   * @brief Typed implementation shared by FP32 and FP16 MHA paths.
   */
  template <typename AttentionType>
  static std::vector<float>
  observationScoresTyped(const AttentionType *attention,
                         size_t attention_elements, unsigned int prompt_length,
                         unsigned int observation_window,
                         unsigned int query_heads) {
    if (attention == nullptr) {
      throw std::invalid_argument("SnapKV: attention must not be null");
    }
    if (prompt_length <= observation_window || observation_window == 0) {
      throw std::invalid_argument(
        "SnapKV: prompt must be longer than the observation window");
    }
    if (query_heads == 0) {
      throw std::invalid_argument("SnapKV: query_heads must be greater than 0");
    }

    auto checked_multiply = [](size_t lhs, size_t rhs,
                               const char *description) {
      if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(std::string("SnapKV: overflow computing ") +
                                  description);
      }
      return lhs * rhs;
    };
    auto triangular_elements = [&checked_multiply](unsigned int length) {
      const size_t converted = length;
      if (converted == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("SnapKV: prompt length overflow");
      }
      return checked_multiply(converted, converted + 1,
                              "triangular attention size") /
             2;
    };

    const size_t expected_elements = checked_multiply(
      triangular_elements(prompt_length), query_heads, "attention elements");
    if (attention_elements != expected_elements) {
      throw std::invalid_argument(
        "SnapKV: attention element count does not match triangular layout");
    }

    const unsigned int prefix_length = prompt_length - observation_window;
    std::vector<float> scores(
      checked_multiply(query_heads, prefix_length, "observation scores"), 0.0f);

    for (unsigned int query = prefix_length; query < prompt_length; ++query) {
      const size_t query_offset = triangular_elements(query);
      for (unsigned int prefix = 0; prefix < prefix_length; ++prefix) {
        const size_t attention_row = query_offset + prefix;
        for (unsigned int query_head = 0; query_head < query_heads;
             ++query_head) {
          const size_t attention_index =
            attention_row * query_heads + query_head;
          scores[static_cast<size_t>(query_head) * prefix_length + prefix] +=
            static_cast<float>(attention[attention_index]);
        }
      }
    }

    return scores;
  }

  /**
   * @brief Apply stride-one, same-length pooling to per-head prefix scores.
   *
   * Max pooling pads with negative infinity. Average pooling pads with zero
   * and includes the padding in the divisor, matching PyTorch's default
   * `avg_pool1d` behavior.
   */
  static std::vector<float> poolScores(const std::vector<float> &scores,
                                       unsigned int heads,
                                       unsigned int prefix_length,
                                       unsigned int pooling_kernel,
                                       SnapKVPooling pooling);

  /**
   * @brief Sum pooled query-head scores for each shared GQA/MQA KV head.
   *
   * Group sum and mean produce the same per-KV-head ranking because every
   * group has the same size. The returned layout is
   * `[kv_head][prefix_position]`.
   */
  static std::vector<float>
  aggregateGQAScores(const std::vector<float> &pooled_query_scores,
                     unsigned int query_heads, unsigned int kv_heads,
                     unsigned int prefix_length);

  /**
   * @brief Select top prefix positions independently for each KV head.
   *
   * Results use `[kv_head][rank]` layout and descending score order. Equal
   * scores prefer the smaller original position for deterministic CPU output.
   */
  static std::vector<unsigned int>
  selectTopK(const std::vector<float> &pooled_scores, unsigned int kv_heads,
             unsigned int prefix_length, unsigned int retained_prefix);

  /**
   * @brief Compact one batch's K and V cache through temporary byte buffers.
   *
   * `selected_indices` must use `[kv_head][rank]` layout. Selected prefix
   * vectors are written first, followed by the complete chronological
   * observation window. Data beyond `cache_capacity` is left unchanged.
   */
  static void compactCache(void *key_cache, void *value_cache,
                           size_t element_size, unsigned int max_seq_len,
                           unsigned int kv_heads, unsigned int head_dim,
                           unsigned int prompt_length,
                           unsigned int observation_window,
                           unsigned int cache_capacity,
                           const std::vector<unsigned int> &selected_indices);
};

} // namespace causallm

#endif // CAUSALLM_SNAPKV_POLICY_H_
