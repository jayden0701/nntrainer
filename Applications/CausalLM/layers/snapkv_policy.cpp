// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   snapkv_policy.cpp
 * @date   03 August 2026
 * @brief  CPU helpers for SnapKV prompt-cache eviction
 * @see    https://arxiv.org/abs/2404.14469
 */

#include "snapkv_policy.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace causallm {

namespace {

size_t checkedMultiply(size_t lhs, size_t rhs, const char *description) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(std::string("SnapKV: overflow computing ") +
                              description);
  }
  return lhs * rhs;
}

size_t checkedAdd(size_t lhs, size_t rhs, const char *description) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(std::string("SnapKV: overflow computing ") +
                              description);
  }
  return lhs + rhs;
}

bool isNaNBits(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t),
                "SnapKV requires a 32-bit IEEE-754 float");
  static_assert(std::numeric_limits<float>::is_iec559,
                "SnapKV requires IEEE-754 floating-point semantics");
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t exponent_mask = 0x7f800000U;
  constexpr uint32_t mantissa_mask = 0x007fffffU;
  return (bits & exponent_mask) == exponent_mask && (bits & mantissa_mask) != 0;
}

float rankableScore(float value) {
  return isNaNBits(value) ? -std::numeric_limits<float>::infinity() : value;
}

void validatePositionState(const SnapKVCachePositionState &state) {
  if (state.logical_position < state.physical_position ||
      state.logical_position - state.physical_position !=
        state.logical_to_physical_offset ||
      state.has_compacted != (state.logical_to_physical_offset != 0)) {
    throw std::logic_error("SnapKV: inconsistent cache position state");
  }
}

} // namespace

SnapKVPooling parseSnapKVPooling(const std::string &value) {
  std::string normalized(value);
  std::transform(
    normalized.begin(), normalized.end(), normalized.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (normalized == "max")
    return SnapKVPooling::MAX;
  if (normalized == "avg" || normalized == "average")
    return SnapKVPooling::AVERAGE;

  throw std::invalid_argument("SnapKV: pooling must be 'max' or 'avg'");
}

void SnapKVPolicy::validateConfig(unsigned int cache_capacity,
                                  unsigned int observation_window,
                                  unsigned int pooling_kernel) {
  if (cache_capacity == 0) {
    throw std::invalid_argument(
      "SnapKV: cache_capacity must be greater than 0");
  }
  if (observation_window == 0) {
    throw std::invalid_argument(
      "SnapKV: observation_window must be greater than 0");
  }
  if (cache_capacity <= observation_window) {
    throw std::invalid_argument(
      "SnapKV: cache_capacity must exceed observation_window");
  }
  if (pooling_kernel == 0 || pooling_kernel % 2 == 0) {
    throw std::invalid_argument(
      "SnapKV: pooling_kernel must be a positive odd number");
  }
}

SnapKVCachePositionState
SnapKVPolicy::mapLogicalPosition(const SnapKVCachePositionState &state,
                                 unsigned int logical_position) {
  if (logical_position == 0)
    return {};
  validatePositionState(state);
  if (state.has_compacted && logical_position != state.logical_position) {
    throw std::out_of_range(
      "SnapKV: repositioning a compacted cache is unsupported");
  }
  if (logical_position < state.logical_to_physical_offset) {
    throw std::out_of_range(
      "SnapKV: logical position precedes the cache offset");
  }

  auto mapped = state;
  mapped.logical_position = logical_position;
  mapped.physical_position =
    logical_position - state.logical_to_physical_offset;
  return mapped;
}

SnapKVCachePositionState SnapKVPolicy::advanceCachePosition(
  const SnapKVCachePositionState &state, unsigned int step_size,
  bool compact_current_step, unsigned int cache_capacity) {
  if (step_size == 0)
    throw std::invalid_argument("SnapKV: cache step size must be positive");
  validatePositionState(state);
  if (state.logical_position >
      std::numeric_limits<unsigned int>::max() - step_size) {
    throw std::overflow_error("SnapKV: logical cache position overflow");
  }

  auto advanced = state;
  advanced.logical_position += step_size;
  if (compact_current_step) {
    if (state.has_compacted || cache_capacity == 0 ||
        advanced.logical_position <= cache_capacity) {
      throw std::invalid_argument(
        "SnapKV: invalid cache compaction position transition");
    }
    advanced.physical_position = cache_capacity;
    advanced.logical_to_physical_offset =
      advanced.logical_position - cache_capacity;
    advanced.has_compacted = true;
  } else {
    if (state.physical_position >
        std::numeric_limits<unsigned int>::max() - step_size) {
      throw std::overflow_error("SnapKV: physical cache position overflow");
    }
    advanced.physical_position += step_size;
  }
  return advanced;
}

std::vector<float> SnapKVPolicy::observationScores(
  const float *attention, size_t attention_elements, unsigned int prompt_length,
  unsigned int observation_window, unsigned int query_heads) {
  return observationScoresTyped(attention, attention_elements, prompt_length,
                                observation_window, query_heads);
}

std::vector<float> SnapKVPolicy::poolScores(const std::vector<float> &scores,
                                            unsigned int heads,
                                            unsigned int prefix_length,
                                            unsigned int pooling_kernel,
                                            SnapKVPooling pooling) {
  if (heads == 0 || prefix_length == 0) {
    throw std::invalid_argument(
      "SnapKV: heads and prefix_length must be greater than 0");
  }
  if (pooling_kernel == 0 || pooling_kernel % 2 == 0) {
    throw std::invalid_argument(
      "SnapKV: pooling_kernel must be a positive odd number");
  }
  if (pooling != SnapKVPooling::MAX && pooling != SnapKVPooling::AVERAGE) {
    throw std::invalid_argument("SnapKV: unsupported pooling operation");
  }

  const size_t expected =
    checkedMultiply(heads, prefix_length, "score elements");
  if (scores.size() != expected) {
    throw std::invalid_argument(
      "SnapKV: score count does not match head geometry");
  }

  std::vector<float> pooled(expected, 0.0f);
  const size_t radius = static_cast<size_t>(pooling_kernel / 2);
  for (unsigned int head = 0; head < heads; ++head) {
    const size_t head_offset = static_cast<size_t>(head) * prefix_length;
    for (unsigned int position = 0; position < prefix_length; ++position) {
      const size_t position_index = position;
      const size_t last_position = static_cast<size_t>(prefix_length) - 1;
      const size_t first_source =
        position_index > radius ? position_index - radius : 0;
      const size_t last_source = radius > last_position - position_index
                                   ? last_position
                                   : position_index + radius;
      if (pooling == SnapKVPooling::MAX) {
        float value = -std::numeric_limits<float>::infinity();
        for (size_t source = first_source;; ++source) {
          value = std::max(value, rankableScore(scores[head_offset + source]));
          if (source == last_source)
            break;
        }
        pooled[head_offset + position] = value;
      } else {
        float sum = 0.0f;
        bool saw_nan = false;
        for (size_t source = first_source;; ++source) {
          const float score = scores[head_offset + source];
          if (isNaNBits(score)) {
            saw_nan = true;
          } else {
            sum += score;
          }
          if (source == last_source)
            break;
        }
        pooled[head_offset + position] =
          saw_nan ? -std::numeric_limits<float>::infinity()
                  : sum / static_cast<float>(pooling_kernel);
      }
    }
  }
  return pooled;
}

std::vector<float> SnapKVPolicy::aggregateGQAScores(
  const std::vector<float> &pooled_query_scores, unsigned int query_heads,
  unsigned int kv_heads, unsigned int prefix_length) {
  if (query_heads == 0 || kv_heads == 0 || prefix_length == 0 ||
      query_heads % kv_heads != 0) {
    throw std::invalid_argument(
      "SnapKV: query_heads must be divisible by kv_heads and geometry must "
      "be nonzero");
  }

  const size_t expected =
    checkedMultiply(query_heads, prefix_length, "pooled query score elements");
  if (pooled_query_scores.size() != expected) {
    throw std::invalid_argument(
      "SnapKV: pooled score count does not match query-head geometry");
  }

  std::vector<float> aggregated(
    checkedMultiply(kv_heads, prefix_length, "aggregated GQA scores"), 0.0f);
  const unsigned int group_size = query_heads / kv_heads;
  for (unsigned int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    for (unsigned int member = 0; member < group_size; ++member) {
      const unsigned int query_head = kv_head * group_size + member;
      const size_t query_offset =
        static_cast<size_t>(query_head) * prefix_length;
      const size_t kv_offset = static_cast<size_t>(kv_head) * prefix_length;
      for (unsigned int position = 0; position < prefix_length; ++position) {
        aggregated[kv_offset + position] +=
          pooled_query_scores[query_offset + position];
      }
    }
  }
  return aggregated;
}

std::vector<unsigned int>
SnapKVPolicy::selectTopK(const std::vector<float> &pooled_scores,
                         unsigned int kv_heads, unsigned int prefix_length,
                         unsigned int retained_prefix) {
  if (kv_heads == 0 || prefix_length == 0 || retained_prefix == 0 ||
      retained_prefix > prefix_length) {
    throw std::invalid_argument("SnapKV: invalid top-k geometry");
  }

  const size_t expected =
    checkedMultiply(kv_heads, prefix_length, "pooled score elements");
  if (pooled_scores.size() != expected) {
    throw std::invalid_argument(
      "SnapKV: pooled score count does not match KV-head geometry");
  }

  std::vector<unsigned int> selected(
    checkedMultiply(kv_heads, retained_prefix, "selected indices"));
  std::vector<unsigned int> order(prefix_length);
  for (unsigned int head = 0; head < kv_heads; ++head) {
    std::iota(order.begin(), order.end(), 0U);
    const size_t score_offset = static_cast<size_t>(head) * prefix_length;
    auto more_important = [&](unsigned int lhs, unsigned int rhs) {
      const float lhs_score = rankableScore(pooled_scores[score_offset + lhs]);
      const float rhs_score = rankableScore(pooled_scores[score_offset + rhs]);
      if (lhs_score == rhs_score)
        return lhs < rhs;
      return lhs_score > rhs_score;
    };
    std::partial_sort(order.data(), order.data() + retained_prefix,
                      order.data() + order.size(), more_important);
    std::copy_n(order.data(), retained_prefix,
                selected.data() + static_cast<size_t>(head) * retained_prefix);
  }
  return selected;
}

void SnapKVPolicy::compactCache(
  void *key_cache, void *value_cache, size_t element_size,
  unsigned int max_seq_len, unsigned int kv_heads, unsigned int head_dim,
  unsigned int prompt_length, unsigned int observation_window,
  unsigned int cache_capacity,
  const std::vector<unsigned int> &selected_indices) {
  if (key_cache == nullptr || value_cache == nullptr || element_size == 0) {
    throw std::invalid_argument(
      "SnapKV: cache pointers and element_size must be valid");
  }
  validateConfig(cache_capacity, observation_window, 1);
  if (prompt_length > max_seq_len || prompt_length <= cache_capacity) {
    throw std::invalid_argument(
      "SnapKV: prompt length must exceed capacity and fit the cache");
  }
  if (kv_heads == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "SnapKV: kv_heads and head_dim must be greater than 0");
  }

  const unsigned int prefix_length = prompt_length - observation_window;
  const unsigned int retained_prefix = cache_capacity - observation_window;
  const size_t expected_indices =
    checkedMultiply(kv_heads, retained_prefix, "selected index elements");
  if (selected_indices.size() != expected_indices) {
    throw std::invalid_argument(
      "SnapKV: selected index count does not match cache geometry");
  }

  const size_t head_elements = head_dim;
  const size_t token_elements =
    checkedMultiply(kv_heads, head_elements, "cache token width");
  const size_t head_bytes =
    checkedMultiply(head_elements, element_size, "cache head bytes");
  const size_t token_bytes =
    checkedMultiply(token_elements, element_size, "cache token bytes");
  const size_t cache_bytes =
    checkedMultiply(max_seq_len, token_bytes, "cache allocation bytes");
  const size_t prompt_bytes =
    checkedMultiply(prompt_length, token_bytes, "prompt cache bytes");
  const size_t compacted_bytes =
    checkedMultiply(cache_capacity, token_bytes, "compacted cache bytes");
  if (prompt_bytes > cache_bytes) {
    throw std::invalid_argument(
      "SnapKV: prompt byte range exceeds the cache allocation");
  }

  std::vector<unsigned char> compacted_key(compacted_bytes);
  std::vector<unsigned char> compacted_value(compacted_bytes);
  const auto *source_key = static_cast<const unsigned char *>(key_cache);
  const auto *source_value = static_cast<const unsigned char *>(value_cache);

  for (unsigned int destination = 0; destination < retained_prefix;
       ++destination) {
    for (unsigned int head = 0; head < kv_heads; ++head) {
      const unsigned int source =
        selected_indices[static_cast<size_t>(head) * retained_prefix +
                         destination];
      if (source >= prefix_length) {
        throw std::out_of_range(
          "SnapKV: selected prefix index is outside the prefix");
      }
      const size_t source_offset = checkedAdd(
        checkedMultiply(source, token_bytes, "source token byte offset"),
        checkedMultiply(head, head_bytes, "source head byte offset"),
        "source vector byte offset");
      const size_t destination_offset = checkedAdd(
        checkedMultiply(destination, token_bytes,
                        "destination token byte offset"),
        checkedMultiply(head, head_bytes, "destination head byte offset"),
        "destination vector byte offset");
      if (checkedAdd(source_offset, head_bytes, "source vector byte end") >
            prompt_bytes ||
          checkedAdd(destination_offset, head_bytes,
                     "destination vector byte end") > compacted_bytes) {
        throw std::out_of_range("SnapKV: cache vector byte range is invalid");
      }
      std::memcpy(compacted_key.data() + destination_offset,
                  source_key + source_offset, head_bytes);
      std::memcpy(compacted_value.data() + destination_offset,
                  source_value + source_offset, head_bytes);
    }
  }

  const size_t observation_source = checkedMultiply(
    prefix_length, token_bytes, "observation source byte offset");
  const size_t observation_destination = checkedMultiply(
    retained_prefix, token_bytes, "observation destination byte offset");
  const size_t observation_bytes =
    checkedMultiply(observation_window, token_bytes, "observation bytes");
  if (checkedAdd(observation_source, observation_bytes,
                 "observation source byte end") > prompt_bytes ||
      checkedAdd(observation_destination, observation_bytes,
                 "observation destination byte end") > compacted_bytes) {
    throw std::out_of_range("SnapKV: observation byte range is invalid");
  }
  std::memcpy(compacted_key.data() + observation_destination,
              source_key + observation_source, observation_bytes);
  std::memcpy(compacted_value.data() + observation_destination,
              source_value + observation_source, observation_bytes);

  std::memcpy(key_cache, compacted_key.data(), compacted_bytes);
  std::memcpy(value_cache, compacted_value.data(), compacted_bytes);
}

} // namespace causallm
