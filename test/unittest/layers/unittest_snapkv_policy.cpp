// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_snapkv_policy.cpp
 * @date   03 August 2026
 * @brief  Unit tests for the CPU SnapKV policy helpers
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <snapkv_policy.h>

namespace {

size_t triangularOffset(unsigned int query) {
  return static_cast<size_t>(query) * (query + 1) / 2;
}

uint32_t floatBits(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t),
                "SnapKV tests require a 32-bit IEEE-754 float");
  static_assert(std::numeric_limits<float>::is_iec559,
                "SnapKV tests require IEEE-754 floating-point semantics");
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool isNaNBits(float value) {
  const uint32_t bits = floatBits(value);
  return (bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0;
}

bool isNegativeInfinityBits(float value) {
  return floatBits(value) == 0xff800000U;
}

std::vector<float>
referenceObservationScores(const std::vector<float> &attention,
                           unsigned int prompt_length, unsigned int window,
                           unsigned int query_heads) {
  const unsigned int prefix = prompt_length - window;
  std::vector<float> result(static_cast<size_t>(query_heads) * prefix, 0.0f);
  for (unsigned int query_head = 0; query_head < query_heads; ++query_head) {
    const size_t head_offset = static_cast<size_t>(query_head) * prefix;
    for (unsigned int key = 0; key < prefix; ++key) {
      for (unsigned int query = prefix; query < prompt_length; ++query) {
        result[head_offset + key] +=
          attention[(triangularOffset(query) + key) * query_heads + query_head];
      }
    }
  }
  return result;
}

std::vector<float> referenceAggregateGQA(const std::vector<float> &query_scores,
                                         unsigned int query_heads,
                                         unsigned int kv_heads,
                                         unsigned int prefix) {
  const unsigned int group = query_heads / kv_heads;
  std::vector<float> result(static_cast<size_t>(kv_heads) * prefix, 0.0f);
  for (unsigned int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const size_t kv_offset = static_cast<size_t>(kv_head) * prefix;
    for (unsigned int member = 0; member < group; ++member) {
      const unsigned int query_head = kv_head * group + member;
      const size_t query_offset = static_cast<size_t>(query_head) * prefix;
      for (unsigned int key = 0; key < prefix; ++key)
        result[kv_offset + key] += query_scores[query_offset + key];
    }
  }
  return result;
}

std::vector<float> referencePool(const std::vector<float> &scores,
                                 unsigned int heads, unsigned int length,
                                 unsigned int kernel,
                                 causallm::SnapKVPooling pooling) {
  const int radius = static_cast<int>(kernel / 2);
  std::vector<float> result(scores.size(), 0.0f);
  for (unsigned int head = 0; head < heads; ++head) {
    const size_t head_offset = static_cast<size_t>(head) * length;
    for (unsigned int position = 0; position < length; ++position) {
      float value = pooling == causallm::SnapKVPooling::MAX
                      ? -std::numeric_limits<float>::infinity()
                      : 0.0f;
      for (int delta = -radius; delta <= radius; ++delta) {
        const int source = static_cast<int>(position) + delta;
        if (source < 0 || source >= static_cast<int>(length))
          continue;
        const size_t source_index = static_cast<size_t>(source);
        if (pooling == causallm::SnapKVPooling::MAX) {
          value = std::max(value, scores[head_offset + source_index]);
        } else {
          value += scores[head_offset + source_index];
        }
      }
      if (pooling == causallm::SnapKVPooling::AVERAGE)
        value /= static_cast<float>(kernel);
      result[head_offset + position] = value;
    }
  }
  return result;
}

std::vector<unsigned int> referenceTopK(const std::vector<float> &scores,
                                        unsigned int heads, unsigned int length,
                                        unsigned int retained) {
  std::vector<unsigned int> result(static_cast<size_t>(heads) * retained);
  for (unsigned int head = 0; head < heads; ++head) {
    const size_t head_offset = static_cast<size_t>(head) * length;
    const size_t result_offset = static_cast<size_t>(head) * retained;
    std::vector<unsigned int> indices(length);
    for (unsigned int position = 0; position < length; ++position)
      indices[position] = position;
    std::sort(
      indices.begin(), indices.end(), [&](unsigned int lhs, unsigned int rhs) {
        const float lhs_score = scores[head_offset + lhs];
        const float rhs_score = scores[head_offset + rhs];
        return lhs_score == rhs_score ? lhs < rhs : lhs_score > rhs_score;
      });
    for (unsigned int rank = 0; rank < retained; ++rank)
      result[result_offset + rank] = indices[rank];
  }
  return result;
}

TEST(SnapKVPolicyTest, validates_configuration) {
  EXPECT_NO_THROW(causallm::SnapKVPolicy::validateConfig(8, 2, 3));
  EXPECT_THROW(causallm::SnapKVPolicy::validateConfig(0, 2, 3),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::validateConfig(8, 0, 3),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::validateConfig(2, 2, 3),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::validateConfig(8, 2, 0),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::validateConfig(8, 2, 4),
               std::invalid_argument);
}

TEST(SnapKVPolicyTest, parses_pooling_names_case_insensitively) {
  EXPECT_EQ(causallm::parseSnapKVPooling("MAX"), causallm::SnapKVPooling::MAX);
  EXPECT_EQ(causallm::parseSnapKVPooling("avg"),
            causallm::SnapKVPooling::AVERAGE);
  EXPECT_EQ(causallm::parseSnapKVPooling("Average"),
            causallm::SnapKVPooling::AVERAGE);
  EXPECT_THROW(causallm::parseSnapKVPooling("median"), std::invalid_argument);
}

TEST(SnapKVPolicyTest, tracks_logical_and_physical_positions_after_eviction) {
  causallm::SnapKVCachePositionState state;
  state = causallm::SnapKVPolicy::advanceCachePosition(state, 6, true, 4);
  EXPECT_EQ(state.logical_position, 6u);
  EXPECT_EQ(state.physical_position, 4u);
  EXPECT_EQ(state.logical_to_physical_offset, 2u);
  EXPECT_TRUE(state.has_compacted);

  state = causallm::SnapKVPolicy::mapLogicalPosition(state, 6);
  EXPECT_EQ(state.physical_position, 4u);
  state = causallm::SnapKVPolicy::advanceCachePosition(state, 1, false, 4);
  EXPECT_EQ(state.logical_position, 7u);
  EXPECT_EQ(state.physical_position, 5u);
  EXPECT_EQ(state.logical_to_physical_offset, 2u);

  state = causallm::SnapKVPolicy::mapLogicalPosition(state, 0);
  EXPECT_EQ(state.logical_position, 0u);
  EXPECT_EQ(state.physical_position, 0u);
  EXPECT_EQ(state.logical_to_physical_offset, 0u);
  EXPECT_FALSE(state.has_compacted);
}

TEST(SnapKVPolicyTest, rejects_invalid_cache_position_transitions) {
  causallm::SnapKVCachePositionState inconsistent{5, 4, 2, true};
  EXPECT_THROW(causallm::SnapKVPolicy::mapLogicalPosition(inconsistent, 5),
               std::logic_error);
  causallm::SnapKVCachePositionState forged_uncompacted{5, 3, 2, false};
  EXPECT_THROW(
    causallm::SnapKVPolicy::mapLogicalPosition(forged_uncompacted, 5),
    std::logic_error);
  causallm::SnapKVCachePositionState forged_compacted{5, 5, 0, true};
  EXPECT_THROW(
    causallm::SnapKVPolicy::advanceCachePosition(forged_compacted, 1, false, 4),
    std::logic_error);
  EXPECT_THROW(causallm::SnapKVPolicy::advanceCachePosition({}, 0, false, 4),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::advanceCachePosition({}, 4, true, 4),
               std::invalid_argument);

  causallm::SnapKVCachePositionState compacted{6, 4, 2, true};
  EXPECT_THROW(
    causallm::SnapKVPolicy::advanceCachePosition(compacted, 1, true, 4),
    std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::mapLogicalPosition(compacted, 1),
               std::out_of_range);
  EXPECT_THROW(causallm::SnapKVPolicy::mapLogicalPosition(compacted, 7),
               std::out_of_range);
}

TEST(SnapKVPolicyTest, computes_query_scores_then_aggregates_gqa_groups) {
  constexpr unsigned int prompt_length = 5;
  constexpr unsigned int window = 2;
  constexpr unsigned int query_heads = 4;
  constexpr unsigned int kv_heads = 2;
  constexpr unsigned int prefix = prompt_length - window;
  const size_t rows = triangularOffset(prompt_length);
  std::vector<float> attention(rows * query_heads, 0.0f);

  auto set_score = [&](unsigned int query, unsigned int key, unsigned int head,
                       float value) {
    attention[(triangularOffset(query) + key) * query_heads + head] = value;
  };

  // KV head 0 owns query heads 0 and 1.
  set_score(3, 0, 0, 0.10f);
  set_score(3, 0, 1, 0.20f);
  set_score(4, 0, 0, 0.30f);
  set_score(4, 0, 1, 0.40f);
  set_score(3, 1, 0, 0.05f);
  set_score(4, 1, 1, 0.15f);
  set_score(3, 2, 0, 0.25f);

  // KV head 1 owns query heads 2 and 3.
  set_score(3, 0, 2, 0.07f);
  set_score(4, 1, 2, 0.11f);
  set_score(4, 1, 3, 0.13f);
  set_score(3, 2, 3, 0.17f);
  set_score(4, 2, 2, 0.19f);

  const auto query_scores = causallm::SnapKVPolicy::observationScores(
    attention.data(), attention.size(), prompt_length, window, query_heads);

  ASSERT_EQ(query_scores.size(), static_cast<size_t>(query_heads * prefix));
  EXPECT_FLOAT_EQ(query_scores[0 * prefix + 0], 0.40f);
  EXPECT_FLOAT_EQ(query_scores[0 * prefix + 1], 0.05f);
  EXPECT_FLOAT_EQ(query_scores[0 * prefix + 2], 0.25f);
  EXPECT_FLOAT_EQ(query_scores[1 * prefix + 0], 0.60f);
  EXPECT_FLOAT_EQ(query_scores[1 * prefix + 1], 0.15f);
  EXPECT_FLOAT_EQ(query_scores[1 * prefix + 2], 0.00f);

  const auto scores = causallm::SnapKVPolicy::aggregateGQAScores(
    query_scores, query_heads, kv_heads, prefix);

  ASSERT_EQ(scores.size(), static_cast<size_t>(kv_heads * prefix));
  EXPECT_FLOAT_EQ(scores[0 * prefix + 0], 1.00f);
  EXPECT_FLOAT_EQ(scores[0 * prefix + 1], 0.20f);
  EXPECT_FLOAT_EQ(scores[0 * prefix + 2], 0.25f);
  EXPECT_FLOAT_EQ(scores[1 * prefix + 0], 0.07f);
  EXPECT_FLOAT_EQ(scores[1 * prefix + 1], 0.24f);
  EXPECT_FLOAT_EQ(scores[1 * prefix + 2], 0.36f);
}

TEST(SnapKVPolicyTest, matches_independent_scalar_oracle) {
  constexpr unsigned int prompt_length = 7;
  constexpr unsigned int window = 3;
  constexpr unsigned int query_heads = 6;
  constexpr unsigned int kv_heads = 3;
  constexpr unsigned int prefix = prompt_length - window;
  constexpr unsigned int retained = 2;
  std::vector<float> attention(triangularOffset(prompt_length) * query_heads,
                               0.0f);

  // Deterministic, normalized causal rows with no dependency on policy code.
  for (unsigned int query = 0; query < prompt_length; ++query) {
    for (unsigned int head = 0; head < query_heads; ++head) {
      float denominator = 0.0f;
      for (unsigned int key = 0; key <= query; ++key)
        denominator +=
          1.0f + static_cast<float>((query * 11 + head * 7 + key * 5) % 17);
      for (unsigned int key = 0; key <= query; ++key) {
        const float numerator =
          1.0f + static_cast<float>((query * 11 + head * 7 + key * 5) % 17);
        attention[(triangularOffset(query) + key) * query_heads + head] =
          numerator / denominator;
      }
    }
  }

  const auto expected_query_scores =
    referenceObservationScores(attention, prompt_length, window, query_heads);
  const auto actual_query_scores = causallm::SnapKVPolicy::observationScores(
    attention.data(), attention.size(), prompt_length, window, query_heads);
  ASSERT_EQ(actual_query_scores.size(), expected_query_scores.size());
  for (size_t index = 0; index < actual_query_scores.size(); ++index)
    EXPECT_NEAR(actual_query_scores[index], expected_query_scores[index],
                1.0e-6f);

  for (const auto pooling :
       {causallm::SnapKVPooling::MAX, causallm::SnapKVPooling::AVERAGE}) {
    const auto expected_query_pooled =
      referencePool(expected_query_scores, query_heads, prefix, 3, pooling);
    const auto expected_pooled = referenceAggregateGQA(
      expected_query_pooled, query_heads, kv_heads, prefix);
    const auto actual_query_pooled = causallm::SnapKVPolicy::poolScores(
      actual_query_scores, query_heads, prefix, 3, pooling);
    const auto actual_pooled = causallm::SnapKVPolicy::aggregateGQAScores(
      actual_query_pooled, query_heads, kv_heads, prefix);
    ASSERT_EQ(actual_pooled.size(), expected_pooled.size());
    for (size_t index = 0; index < actual_pooled.size(); ++index)
      EXPECT_NEAR(actual_pooled[index], expected_pooled[index], 1.0e-6f);
    EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(actual_pooled, kv_heads,
                                                 prefix, retained),
              referenceTopK(expected_pooled, kv_heads, prefix, retained));
  }
}

TEST(SnapKVPolicyTest, matches_scalar_oracle_across_gqa_geometries) {
  for (unsigned int trial = 0; trial < 128; ++trial) {
    const unsigned int prompt_length = 4 + trial % 7;
    const unsigned int window = 1 + trial % (prompt_length - 2);
    const unsigned int prefix = prompt_length - window;
    const unsigned int kv_heads = 1 + trial % 3;
    const unsigned int group_size = 1 + (trial / 3) % 4;
    const unsigned int query_heads = kv_heads * group_size;
    const unsigned int kernel = 1 + 2 * (trial % 4);
    const unsigned int retained = 1 + trial % prefix;
    std::vector<float> attention(triangularOffset(prompt_length) * query_heads,
                                 0.0f);

    for (unsigned int query = 0; query < prompt_length; ++query) {
      for (unsigned int head = 0; head < query_heads; ++head) {
        float denominator = 0.0f;
        for (unsigned int key = 0; key <= query; ++key) {
          denominator +=
            1.0f + static_cast<float>(
                     (trial * 13 + query * 11 + head * 7 + key * 5) % 29);
        }
        for (unsigned int key = 0; key <= query; ++key) {
          const float numerator =
            1.0f + static_cast<float>(
                     (trial * 13 + query * 11 + head * 7 + key * 5) % 29);
          attention[(triangularOffset(query) + key) * query_heads + head] =
            numerator / denominator;
        }
      }
    }

    const auto expected_query =
      referenceObservationScores(attention, prompt_length, window, query_heads);
    const auto actual_query = causallm::SnapKVPolicy::observationScores(
      attention.data(), attention.size(), prompt_length, window, query_heads);
    ASSERT_EQ(actual_query.size(), expected_query.size());
    for (size_t index = 0; index < actual_query.size(); ++index)
      EXPECT_NEAR(actual_query[index], expected_query[index], 1.0e-6f);

    for (const auto pooling :
         {causallm::SnapKVPooling::MAX, causallm::SnapKVPooling::AVERAGE}) {
      const auto expected_query_pooled =
        referencePool(expected_query, query_heads, prefix, kernel, pooling);
      const auto expected_kv = referenceAggregateGQA(
        expected_query_pooled, query_heads, kv_heads, prefix);
      const auto actual_query_pooled = causallm::SnapKVPolicy::poolScores(
        actual_query, query_heads, prefix, kernel, pooling);
      const auto actual_kv = causallm::SnapKVPolicy::aggregateGQAScores(
        actual_query_pooled, query_heads, kv_heads, prefix);

      ASSERT_EQ(actual_kv.size(), expected_kv.size());
      for (size_t index = 0; index < actual_kv.size(); ++index)
        EXPECT_NEAR(actual_kv[index], expected_kv[index], 1.0e-6f);
      EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(actual_kv, kv_heads, prefix,
                                                   retained),
                referenceTopK(expected_kv, kv_heads, prefix, retained));
    }
  }
}

TEST(SnapKVPolicyTest, rejects_malformed_triangular_attention) {
  std::vector<float> attention(3, 0.0f);
  EXPECT_THROW(causallm::SnapKVPolicy::observationScores(
                 attention.data(), attention.size(), 5, 2, 4),
               std::invalid_argument);
  EXPECT_THROW(causallm::SnapKVPolicy::observationScores(
                 attention.data(), attention.size(), 5, 2, 0),
               std::invalid_argument);
}

TEST(SnapKVPolicyTest, pools_each_query_head_before_gqa_reduction) {
  const std::vector<float> query_scores{10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f};
  const auto pooled_query_scores = causallm::SnapKVPolicy::poolScores(
    query_scores, 2, 3, 3, causallm::SnapKVPooling::MAX);
  const auto aggregated =
    causallm::SnapKVPolicy::aggregateGQAScores(pooled_query_scores, 2, 1, 3);

  EXPECT_EQ(aggregated, (std::vector<float>{10.0f, 20.0f, 10.0f}));
  EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(aggregated, 1, 3, 1),
            (std::vector<unsigned int>{1}));

  // Reducing before max-pooling would incorrectly produce a three-way tie.
  const auto reduce_first =
    causallm::SnapKVPolicy::aggregateGQAScores(query_scores, 2, 1, 3);
  const auto wrong_order = causallm::SnapKVPolicy::poolScores(
    reduce_first, 1, 3, 3, causallm::SnapKVPooling::MAX);
  EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(wrong_order, 1, 3, 1),
            (std::vector<unsigned int>{0}));
}

TEST(SnapKVPolicyTest, rejects_invalid_gqa_score_geometry) {
  EXPECT_THROW(causallm::SnapKVPolicy::aggregateGQAScores({}, 2, 1, 3),
               std::invalid_argument);
  EXPECT_THROW(
    causallm::SnapKVPolicy::aggregateGQAScores(std::vector<float>(9), 3, 2, 3),
    std::invalid_argument);
  EXPECT_THROW(
    causallm::SnapKVPolicy::aggregateGQAScores(std::vector<float>(5), 2, 1, 3),
    std::invalid_argument);
}

TEST(SnapKVPolicyTest, max_pool_uses_negative_infinity_padding) {
  const std::vector<float> scores{-5.0f, -4.0f, -10.0f, -6.0f};
  const auto pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 4, 3, causallm::SnapKVPooling::MAX);
  const std::vector<float> expected{-4.0f, -4.0f, -4.0f, -6.0f};
  EXPECT_EQ(pooled, expected);
}

TEST(SnapKVPolicyTest, average_pool_includes_zero_padding_in_divisor) {
  const std::vector<float> scores{1.0f, 2.0f, 10.0f, 4.0f};
  const auto pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 4, 3, causallm::SnapKVPooling::AVERAGE);
  ASSERT_EQ(pooled.size(), 4u);
  EXPECT_FLOAT_EQ(pooled[0], 1.0f);
  EXPECT_FLOAT_EQ(pooled[1], 13.0f / 3.0f);
  EXPECT_FLOAT_EQ(pooled[2], 16.0f / 3.0f);
  EXPECT_FLOAT_EQ(pooled[3], 14.0f / 3.0f);
}

TEST(SnapKVPolicyTest, handles_nan_pooling_deterministically) {
  const std::vector<float> scores{std::numeric_limits<float>::quiet_NaN(), 2.0f,
                                  3.0f};
  const auto max_pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 3, 3, causallm::SnapKVPooling::MAX);
  EXPECT_EQ(max_pooled, (std::vector<float>{2.0f, 3.0f, 3.0f}));

  const auto average_pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 3, 3, causallm::SnapKVPooling::AVERAGE);
  EXPECT_TRUE(isNegativeInfinityBits(average_pooled[0]));
  EXPECT_TRUE(isNegativeInfinityBits(average_pooled[1]));
  EXPECT_FLOAT_EQ(average_pooled[2], 5.0f / 3.0f);

  const std::vector<float> all_nan{std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::quiet_NaN()};
  const auto all_nan_max = causallm::SnapKVPolicy::poolScores(
    all_nan, 1, 2, 3, causallm::SnapKVPooling::MAX);
  ASSERT_EQ(all_nan_max.size(), 2u);
  for (float value : all_nan_max) {
    EXPECT_TRUE(isNegativeInfinityBits(value));
  }
}

TEST(SnapKVPolicyTest, rejects_unknown_pooling_enum) {
  const std::vector<float> scores{1.0f};
  EXPECT_THROW(causallm::SnapKVPolicy::poolScores(
                 scores, 1, 1, 1, static_cast<causallm::SnapKVPooling>(99)),
               std::invalid_argument);
}

TEST(SnapKVPolicyTest, kernel_one_is_identity) {
  const std::vector<float> scores{1.0f, -2.0f, 3.0f, 4.0f};
  EXPECT_EQ(causallm::SnapKVPolicy::poolScores(scores, 2, 2, 1,
                                               causallm::SnapKVPooling::MAX),
            scores);
  EXPECT_EQ(causallm::SnapKVPolicy::poolScores(
              scores, 2, 2, 1, causallm::SnapKVPooling::AVERAGE),
            scores);
}

TEST(SnapKVPolicyTest, very_large_kernel_avoids_signed_index_overflow) {
  const std::vector<float> scores{1.0f, 2.0f, 3.0f};
  const unsigned int kernel = std::numeric_limits<unsigned int>::max();
  const auto max_pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 3, kernel, causallm::SnapKVPooling::MAX);
  EXPECT_EQ(max_pooled, (std::vector<float>{3.0f, 3.0f, 3.0f}));

  const auto average_pooled = causallm::SnapKVPolicy::poolScores(
    scores, 1, 3, kernel, causallm::SnapKVPooling::AVERAGE);
  const float expected = 6.0f / static_cast<float>(kernel);
  for (float value : average_pooled)
    EXPECT_FLOAT_EQ(value, expected);
}

TEST(SnapKVPolicyTest, topk_is_per_head_and_deterministic_on_ties) {
  const std::vector<float> pooled{
    5.0f, 5.0f, 1.0f, 4.0f, // head 0
    0.0f, 3.0f, 2.0f, 3.0f  // head 1
  };
  const auto selected = causallm::SnapKVPolicy::selectTopK(pooled, 2, 4, 2);
  const std::vector<unsigned int> expected{0, 1, 1, 3};
  EXPECT_EQ(selected, expected);
}

TEST(SnapKVPolicyTest, nan_score_ranks_below_finite_score) {
  const std::vector<float> pooled{std::numeric_limits<float>::quiet_NaN(),
                                  -100.0f};
  const auto selected = causallm::SnapKVPolicy::selectTopK(pooled, 1, 2, 1);
  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0], 1u);

  const std::vector<float> tied_non_finite{
    std::numeric_limits<float>::quiet_NaN(),
    -std::numeric_limits<float>::infinity()};
  EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(tied_non_finite, 1, 2, 1),
            (std::vector<unsigned int>{0}));

  const std::vector<float> mixed_infinity{
    std::numeric_limits<float>::infinity(), 0.0f,
    -std::numeric_limits<float>::infinity(), 1.0f};
  const auto mixed_aggregated =
    causallm::SnapKVPolicy::aggregateGQAScores(mixed_infinity, 2, 1, 2);
  ASSERT_EQ(mixed_aggregated.size(), 2u);
  EXPECT_TRUE(isNaNBits(mixed_aggregated[0]));
  EXPECT_FLOAT_EQ(mixed_aggregated[1], 1.0f);
  EXPECT_EQ(causallm::SnapKVPolicy::selectTopK(mixed_aggregated, 1, 2, 1),
            (std::vector<unsigned int>{1}));
}

template <typename T> void expectCompactionPreservesPairs() {
  constexpr unsigned int max_seq_len = 8;
  constexpr unsigned int prompt_length = 6;
  constexpr unsigned int window = 2;
  constexpr unsigned int capacity = 4;
  constexpr unsigned int kv_heads = 2;
  constexpr unsigned int head_dim = 2;
  constexpr unsigned int width = kv_heads * head_dim;

  std::vector<T> key(max_seq_len * width);
  std::vector<T> value(max_seq_len * width);
  for (unsigned int token = 0; token < max_seq_len; ++token) {
    for (unsigned int head = 0; head < kv_heads; ++head) {
      for (unsigned int dim = 0; dim < head_dim; ++dim) {
        const size_t index = static_cast<size_t>(token) * width +
                             static_cast<size_t>(head) * head_dim + dim;
        key[index] = static_cast<T>(token * 100 + head * 10 + dim);
        value[index] = static_cast<T>(token * 100 + head * 10 + dim + 1000);
      }
    }
  }
  const auto original_key = key;
  const auto original_value = value;

  // Retain two prefix positions. Each KV head deliberately uses a different
  // source order, including sources that an unsafe in-place gather can clobber.
  const std::vector<unsigned int> selected{2, 0, 1, 3};
  causallm::SnapKVPolicy::compactCache(
    key.data(), value.data(), sizeof(T), max_seq_len, kv_heads, head_dim,
    prompt_length, window, capacity, selected);

  auto expect_head = [&](unsigned int destination, unsigned int head,
                         unsigned int source) {
    for (unsigned int dim = 0; dim < head_dim; ++dim) {
      const size_t destination_index =
        static_cast<size_t>(destination) * width +
        static_cast<size_t>(head) * head_dim + dim;
      const size_t source_index = static_cast<size_t>(source) * width +
                                  static_cast<size_t>(head) * head_dim + dim;
      EXPECT_EQ(key[destination_index], original_key[source_index]);
      EXPECT_EQ(value[destination_index], original_value[source_index]);
    }
  };

  expect_head(0, 0, 2);
  expect_head(0, 1, 1);
  expect_head(1, 0, 0);
  expect_head(1, 1, 3);
  for (unsigned int head = 0; head < kv_heads; ++head) {
    expect_head(2, head, 4);
    expect_head(3, head, 5);
  }
  for (unsigned int token = capacity; token < max_seq_len; ++token) {
    for (unsigned int element = 0; element < width; ++element) {
      const size_t index = static_cast<size_t>(token) * width + element;
      EXPECT_EQ(key[index], original_key[index]);
      EXPECT_EQ(value[index], original_value[index]);
    }
  }
}

TEST(SnapKVPolicyTest, compacts_fp32_cache_without_overlap_corruption) {
  expectCompactionPreservesPairs<float>();
}

TEST(SnapKVPolicyTest, compacts_two_byte_cache_byte_exactly) {
  expectCompactionPreservesPairs<uint16_t>();
}

TEST(SnapKVPolicyTest, compacts_batches_independently) {
  constexpr unsigned int batches = 2;
  constexpr unsigned int max_seq_len = 6;
  constexpr unsigned int prompt_length = 5;
  constexpr unsigned int window = 2;
  constexpr unsigned int capacity = 4;
  constexpr size_t batch_stride = max_seq_len;
  std::vector<uint16_t> key(batches * batch_stride);
  std::vector<uint16_t> value(batches * batch_stride);
  for (unsigned int batch = 0; batch < batches; ++batch) {
    for (unsigned int token = 0; token < max_seq_len; ++token) {
      const size_t index = batch * batch_stride + token;
      key[index] = static_cast<uint16_t>(batch * 100 + token);
      value[index] = static_cast<uint16_t>(batch * 100 + token + 1000);
    }
  }
  const auto original_key = key;
  const auto original_value = value;
  const std::vector<std::vector<unsigned int>> selected{{0, 2}, {2, 0}};

  for (unsigned int batch = 0; batch < batches; ++batch) {
    causallm::SnapKVPolicy::compactCache(
      key.data() + batch * batch_stride, value.data() + batch * batch_stride,
      sizeof(uint16_t), max_seq_len, 1, 1, prompt_length, window, capacity,
      selected[batch]);
  }

  for (unsigned int batch = 0; batch < batches; ++batch) {
    const std::vector<unsigned int> expected_sources{selected[batch][0],
                                                     selected[batch][1], 3, 4};
    for (unsigned int destination = 0; destination < capacity; ++destination) {
      const size_t destination_index = batch * batch_stride + destination;
      const size_t source_index =
        batch * batch_stride + expected_sources[destination];
      EXPECT_EQ(key[destination_index], original_key[source_index]);
      EXPECT_EQ(value[destination_index], original_value[source_index]);
    }
    for (unsigned int token = capacity; token < max_seq_len; ++token) {
      const size_t index = batch * batch_stride + token;
      EXPECT_EQ(key[index], original_key[index]);
      EXPECT_EQ(value[index], original_value[index]);
    }
  }
}

TEST(SnapKVPolicyTest, rejects_selected_index_outside_prefix) {
  std::vector<float> key(8, 0.0f);
  std::vector<float> value(8, 0.0f);
  const auto original_key = key;
  const auto original_value = value;
  const std::vector<unsigned int> selected{3};
  EXPECT_THROW(causallm::SnapKVPolicy::compactCache(key.data(), value.data(),
                                                    sizeof(float), 4, 1, 2, 3,
                                                    1, 2, selected),
               std::out_of_range);
  EXPECT_EQ(key, original_key);
  EXPECT_EQ(value, original_value);
}

TEST(SnapKVPolicyTest, rejects_cache_byte_geometry_overflow) {
  unsigned char key = 0;
  unsigned char value = 0;
  constexpr unsigned int kv_heads = 65536;
  constexpr unsigned int head_dim = 131072;
  const std::vector<unsigned int> selected(kv_heads, 0U);

  EXPECT_THROW(causallm::SnapKVPolicy::compactCache(
                 &key, &value, 1, std::numeric_limits<unsigned int>::max(),
                 kv_heads, head_dim, 3, 1, 2, selected),
               std::overflow_error);
}

} // namespace
