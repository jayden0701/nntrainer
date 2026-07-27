// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_generate_qnn_sample.cpp
 * @brief  Unit tests for QNN token sampling
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 */

#include "generate_qnn_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr float kUnitScale = 1.0f;
constexpr float kNoRepetitionPenalty = 1.0f;
constexpr float kGreedyTemperature = 0.0f;
constexpr float kFullTopP = 1.0f;

int sampleGreedy(std::vector<uint16_t> &logits, int top_k,
                 const int32_t *allowed_token_bitmask = nullptr,
                 size_t allowed_token_bitmask_size = 0) {
  std::mt19937 random_engine(0);
  return sample(logits.data(), static_cast<int>(logits.size()), nullptr, 0,
                kUnitScale, 0, kNoRepetitionPenalty, kGreedyTemperature,
                kFullTopP, top_k, random_engine, 0.0f, allowed_token_bitmask,
                allowed_token_bitmask_size);
}

TEST(GenerateQnnSample, GreedySelectsHighestLogit) {
  std::vector<uint16_t> logits{10, 40, 30};

  EXPECT_EQ(sampleGreedy(logits, static_cast<int>(logits.size())), 1);
}

TEST(GenerateQnnSample, GreedyTieSelectsLowestTokenId) {
  std::vector<uint16_t> logits{40, 40, 30};

  EXPECT_EQ(sampleGreedy(logits, 1), 0);
}

TEST(GenerateQnnSample, AppliesRepetitionPenaltyBeforeTopK) {
  std::vector<uint16_t> logits{100, 90, 10};
  std::array<int, 1> token_history{0};
  std::mt19937 random_engine(0);

  EXPECT_EQ(sample(logits.data(), static_cast<int>(logits.size()),
                   token_history.data(), static_cast<int>(token_history.size()),
                   kUnitScale, 0, 2.0f, kGreedyTemperature, kFullTopP, 1,
                   random_engine),
            1);
}

TEST(GenerateQnnSample, GrammarMaskExcludesDisallowedTokens) {
  std::vector<uint16_t> logits{10, 100, 50};
  std::array<int32_t, 1> allowed_token_bitmask{
    static_cast<int32_t>((1U << 0U) | (1U << 2U))};

  EXPECT_EQ(sampleGreedy(logits, static_cast<int>(logits.size()),
                         allowed_token_bitmask.data(),
                         allowed_token_bitmask.size()),
            2);
}

TEST(GenerateQnnSample, GrammarMaskRejectsAllMaskedVocabulary) {
  std::vector<uint16_t> logits{10, 100, 50};
  std::array<int32_t, 1> allowed_token_bitmask{0};

  EXPECT_THROW(sampleGreedy(logits, static_cast<int>(logits.size()),
                            allowed_token_bitmask.data(),
                            allowed_token_bitmask.size()),
               std::runtime_error);
}

TEST(GenerateQnnSample, RejectsInvalidInputs) {
  std::vector<uint16_t> logits{10, 20, 30};
  std::array<int, 1> token_history{0};
  std::array<int32_t, 1> allowed_token_bitmask{1};
  std::mt19937 random_engine(0);

  EXPECT_THROW(sample(nullptr, static_cast<int>(logits.size()), nullptr, 0,
                      kUnitScale, 0, kNoRepetitionPenalty, kGreedyTemperature,
                      kFullTopP, static_cast<int>(logits.size()),
                      random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), 0, nullptr, 0, kUnitScale, 0,
                      kNoRepetitionPenalty, kGreedyTemperature, kFullTopP, 1,
                      random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      static_cast<int>(token_history.size()), kUnitScale, 0,
                      kNoRepetitionPenalty, kGreedyTemperature, kFullTopP, 1,
                      random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      -1, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, kFullTopP, 1, random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, kFullTopP, 0, random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, 0.0f, 0, kNoRepetitionPenalty, kGreedyTemperature,
                      kFullTopP, 1, random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, 0.0f, kGreedyTemperature, kFullTopP, 1,
                      random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty, -0.1f, kFullTopP,
                      1, random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, 1.1f, 1, random_engine),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, kFullTopP, 1, random_engine, -1.0f),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, kFullTopP, 1, random_engine, 0.0f,
                      nullptr, allowed_token_bitmask.size()),
               std::invalid_argument);
  EXPECT_THROW(sample(logits.data(), static_cast<int>(logits.size()), nullptr,
                      0, kUnitScale, 0, kNoRepetitionPenalty,
                      kGreedyTemperature, kFullTopP, 1, random_engine, 0.0f,
                      allowed_token_bitmask.data(), 0),
               std::invalid_argument);

  std::vector<uint16_t> large_vocabulary(33, 0);
  EXPECT_THROW(
    sample(large_vocabulary.data(), static_cast<int>(large_vocabulary.size()),
           nullptr, 0, kUnitScale, 0, kNoRepetitionPenalty, kGreedyTemperature,
           kFullTopP, 1, random_engine, 0.0f, allowed_token_bitmask.data(),
           allowed_token_bitmask.size()),
    std::invalid_argument);
}

TEST(GenerateQnnSample, SeededSamplingIsDeterministic) {
  std::vector<uint16_t> logits{10, 11, 12, 13};
  std::mt19937 first_engine(1729);
  std::mt19937 second_engine(1729);

  for (int draw = 0; draw < 16; ++draw) {
    const int first =
      sample(logits.data(), static_cast<int>(logits.size()), nullptr, 0,
             kUnitScale, 0, kNoRepetitionPenalty, 1.0f, kFullTopP,
             static_cast<int>(logits.size()), first_engine);
    const int second =
      sample(logits.data(), static_cast<int>(logits.size()), nullptr, 0,
             kUnitScale, 0, kNoRepetitionPenalty, 1.0f, kFullTopP,
             static_cast<int>(logits.size()), second_engine);
    EXPECT_EQ(first, second);
  }
}

} // namespace
