// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gemma4.cpp
 * @date   15 June 2026
 * @brief  Tiny Gemma4 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma4_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr int tiny_gemma4_num_layers = 2;
constexpr unsigned int tiny_gemma4_per_layer_input_width = 32;

std::vector<char> readBinaryFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open test weight file: " +
                             path.string());

  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

/**
 * @brief Tiny Gemma4 CausalLM adapter for common model tests
 *
 * Thin subclass of the shared CausalLMTestAdapter: only the constructor
 * differs because Gemma4 must sanitize its configs (flattening text_config)
 * before initializing the (virtual) Transformer base.
 */
class TinyGemma4CausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM> {
public:
  /**
   * @brief Construct a tiny Gemma4 CausalLM test adapter
   */
  TinyGemma4CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate deterministic tiny Gemma4 weights for golden token tests
 */
void setupGemma4DeterministicWeights(TinyGemma4CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        } else if (layer.getName().find("_layer_scalar") != std::string::npos) {
          // layer_scalar scales decoder_output (including residual) before the
          // next layer receives it.  A value of 0 zeros out the entire hidden
          // state; 1 preserves it so the residual path is exercised.
          weight.setValue(1.0f);
        }
      }
    });
}

/**
 * @brief Populate non-zero PLE weights for the FSU regression.
 */
void setupGemma4PleSensitiveWeights(TinyGemma4CausalLM &model) {
  setupGemma4DeterministicWeights(model);

  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "embedding0") {
        auto &weight = context.getWeight(0);
        ASSERT_GT(weight.getDim().width(), 1u);
        weight.setValue(0, 0, 1, 1, 0.25f);
        weight.setValue(0, 0, 2, 0, 1.5f);
        weight.setValue(0, 0, 3, 0, 0.5f);
      } else if (layer.getName().rfind("layer", 0) == 0 &&
                 layer.getName().find("_per_layer_input_embedding") !=
                   std::string::npos) {
        auto &weight = context.getWeight(0);
        ASSERT_GT(weight.getDim().width(), 1u);
        for (unsigned int token_id : {1u, 2u, 3u, 4u})
          weight.setValue(0, 0, token_id, 1, static_cast<float>(token_id));
      } else if (layer.getName() == "per_layer_input_embedding") {
        auto &weight = context.getWeight(0);
        ASSERT_GT(weight.getDim().width(), tiny_gemma4_per_layer_input_width);
        // The packed table concatenates each layer's PLE columns. Its
        // layer_id * P + 1 column matches split table feature 1.
        for (unsigned int layer_id = 0;
             layer_id < static_cast<unsigned int>(tiny_gemma4_num_layers);
             ++layer_id) {
          for (unsigned int token_id : {1u, 2u, 3u, 4u}) {
            weight.setValue(0, 0, token_id,
                            layer_id * tiny_gemma4_per_layer_input_width + 1,
                            static_cast<float>(token_id));
          }
        }
      } else if (layer.getName().find("_per_layer_input_gate") !=
                 std::string::npos) {
        // FC weights are [1, 1, input, output] in the NCHW test graph. Route
        // decoder feature 0 to PLE feature 1, not a pure scale RMSNorm erases.
        auto &weight = context.getWeight(0);
        ASSERT_GT(weight.getDim().height(), 1u);
        ASSERT_GT(weight.getDim().width(), 1u);
        weight.setValue(0, 0, 0, 1, 1.0f);
      } else if (layer.getName().find("_per_layer_input_proj") !=
                 std::string::npos) {
        // Route PLE feature 1 to hidden feature 1 with the same layout.
        auto &weight = context.getWeight(0);
        ASSERT_GT(weight.getDim().height(), 1u);
        ASSERT_GT(weight.getDim().width(), 1u);
        weight.setValue(0, 0, 1, 1, 1.0f);
      }
    });
}

/**
 * @brief Remove only split PLE table values while keeping its
 * gate/projection.
 */
void clearGemma4SplitPleEmbeddingWeights(TinyGemma4CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName().find("_per_layer_input_embedding") !=
          std::string::npos) {
        context.getWeight(0).setValue(0.0f);
      }
    });
}

/**
 * @brief Read whether every per-decoder PLE embedding has the FSU state
 */
struct PleWeightFsuState {
  unsigned int count = 0;
  bool all_virtual = true;
  bool all_allocated = true;
  bool any_allocated = false;
};

PleWeightFsuState getPleWeightFsuState(TinyGemma4CausalLM &model) {
  PleWeightFsuState state;

  model.forEachLayer([&state](ml::train::Layer &layer,
                              nntrainer::RunLayerContext &context, void *) {
    if (layer.getName().find("_per_layer_input_embedding") == std::string::npos)
      return;

    ASSERT_EQ(context.getNumWeights(), 1u);
    ++state.count;
    state.all_virtual &= context.getWeight(0).isVirtual();
    state.all_allocated &= context.getWeight(0).isAllocated();
    state.any_allocated |= context.getWeight(0).isAllocated();
  });

  return state;
}

/**
 * @brief Make the tiny Gemma4 model config
 *
 * Fields are wrapped in text_config as the real HF config would be.
 * sanitizeConfig() in TinyGemma4CausalLM flattens them before construction.
 */
causallm::json makeTinyGemma4Config() {
  return {
    {"architectures", {"Gemma4ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"text_config",
     {
       {"head_dim", 8},
       {"hidden_size", 64},
       {"hidden_size_per_layer_input", 32},
       {"intermediate_size", 64},
       {"layer_types", {"sliding_attention", "full_attention"}},
       {"max_position_embeddings", 8},
       {"num_attention_heads", 8},
       {"num_hidden_layers", tiny_gemma4_num_layers},
       {"num_key_value_heads", 4},
       {"rms_norm_eps", 1e-6},
       {"rope_theta", 1000000},
       {"sliding_window", 4},
       {"tie_word_embeddings", true},
       {"vocab_size", 32},
       {"vocab_size_per_layer_input", 32},
     }},
  };
}

/**
 * @brief Make the tiny Gemma4 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeGemma4LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32") {
    const auto emb_dtype =
      causallm_test::toTensorDataType(data_type.embedding_dtype);
    dtype_map["embedding0"] = emb_dtype;
    // per_layer_input_embedding: [vocab_per_layer, num_layers*hidden_per_layer]
    // with hidden_size_per_layer_input=32: width=64, divisible by 32
    dtype_map["per_layer_input_embedding"] = emb_dtype;
  }

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_gemma4_num_layers; ++i) {
      const std::string prefix = "layer" + std::to_string(i);
      dtype_map[prefix + "_wq"] = dtype;
      dtype_map[prefix + "_wk"] = dtype;
      dtype_map[prefix + "_wv"] = dtype;
      dtype_map[prefix + "_attention_out"] = dtype;
      dtype_map[prefix + "_ffn_gate"] = dtype;
      dtype_map[prefix + "_ffn_up"] = dtype;
      dtype_map[prefix + "_ffn_down"] = dtype;
      // Gemma4-specific per-layer FC weights and split PLE embedding.
      // hidden_size_per_layer_input=32 ensures width is divisible by 32
      dtype_map[prefix + "_per_layer_input_embedding"] = dtype;
      dtype_map[prefix + "_per_layer_input_gate"] = dtype;
      dtype_map[prefix + "_per_layer_input_proj"] = dtype;
    }
    dtype_map["per_layer_input_projection"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the expected tiny Gemma4 prefill logits
 *
 * With deterministic weights (embedding[1,0]=1, embedding[4,0]=2, all FC=0,
 * all rms_norm=1, all scalar_multiply=0), the hidden state passes unchanged
 * through zero-output decoder layers.  The final rms_norm normalises the
 * embedding vector, and the tied word-embedding lm_head projects it back:
 *   logit[j] = hidden_norm[0] * embedding[j,0]
 * giving logit[1]=8, logit[4]=16, all others=0.
 */
std::vector<float> makeExpectedGemma4Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 8.0f;
  logits[4] = 16.0f;
  return logits;
}

/**
 * @brief Make a Gemma4 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeGemma4Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Gemma4_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGemma4Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-2f
                                    : 1e-3f},
    makeTinyGemma4Config,
    makeGemma4LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGemma4CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGemma4DeterministicWeights(
        static_cast<TinyGemma4CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Gemma4 model cases
 */
class Gemma4TinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Gemma4TinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

/**
 * @brief Test that greedy generation chooses the argmax logit
 */
TEST_P(Gemma4TinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

/**
 * @brief Test that a save/load round-trip preserves logits
 */
TEST_P(Gemma4TinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(Gemma4TinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

/**
 * @brief Split PLE remains resident, including Windows FSU fallback.
 */
TEST(Gemma4PleFsuTest, SplitPleWeightsRemainResidentAndWindowsFsuFallback) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "SplitPleWeightsRemainResidentAndWindowsFsuFallback",
    "FP32");
  const auto data_type = causallm_test::makeTinyFp32DataType();
  auto model_cfg = makeTinyGemma4Config();
  auto generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  nntr_cfg["ple_split"] = true;
#if defined(_WIN32)
  // Windows must construct split PLE weights as resident even when FSU is on.
  nntr_cfg["fsu"] = true;
  nntr_cfg["fsu_lookahead"] = 1;
#endif

  TinyGemma4CausalLM model(model_cfg, generation_cfg, nntr_cfg);
  model.initializeModel();

  const auto state = getPleWeightFsuState(model);
  EXPECT_EQ(state.count, tiny_gemma4_num_layers);
  EXPECT_FALSE(state.all_virtual);
  EXPECT_TRUE(state.all_allocated);
}

/**
 * @brief Split Tensor PLE cannot be combined with a sidecar LUT
 */
TEST(Gemma4PleFsuTest, RejectsSidecarPleWithSplitLayout) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "RejectsSidecarPleWithSplitLayout", "FP32");
  const auto data_type = causallm_test::makeTinyFp32DataType();
  auto model_cfg = makeTinyGemma4Config();
  auto generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  nntr_cfg["ple_split"] = true;
  nntr_cfg["ple_file_name"] = (files.dir / "ple_sidecar.bin").string();

  TinyGemma4CausalLM model(model_cfg, generation_cfg, nntr_cfg);
  EXPECT_THROW(model.initializeModel(), std::invalid_argument);
}

/**
 * @brief Packed and resident split PLE graphs have equivalent algebra.
 */
TEST(Gemma4PleFsuTest, LegacyPackedPleMatchesResidentSplitGraph) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "LegacyPackedPleMatchesResidentSplitGraph", "FP32");
  const auto data_type = causallm_test::makeTinyFp32DataType();
  const std::vector<unsigned int> ids = {1, 4, 2, 3};

  auto legacy_model_cfg = makeTinyGemma4Config();
  auto legacy_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto legacy_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  TinyGemma4CausalLM legacy(legacy_model_cfg, legacy_generation_cfg,
                            legacy_nntr_cfg);
  legacy.initializeModel();
  setupGemma4PleSensitiveWeights(legacy);

  auto split_model_cfg = makeTinyGemma4Config();
  auto split_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto split_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  split_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM split(split_model_cfg, split_generation_cfg,
                           split_nntr_cfg);
  split.initializeModel();
  setupGemma4PleSensitiveWeights(split);

  const auto legacy_prefill = legacy.prefillLogitsFromIds(ids);
  const auto split_prefill = split.prefillLogitsFromIds(ids);
  ASSERT_EQ(legacy_prefill.size(), split_prefill.size());
  for (size_t i = 0; i < legacy_prefill.size(); ++i)
    EXPECT_NEAR(legacy_prefill[i], split_prefill[i], 1e-4f) << "logit " << i;
  EXPECT_EQ(legacy.greedyGenerateFromIds(ids, 2),
            split.greedyGenerateFromIds(ids, 2));
}

/**
 * @brief Virtual PLE preserves resident prefill and decode results.
 */
TEST(Gemma4PleFsuTest, LazyPleWeightMatchesResidentPrefillAndDecode) {
#if defined(_WIN32)
  GTEST_SKIP() << "virtual tensor mmap is not supported on Windows";
#else
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "LazyPleWeightMatchesResidentPrefillAndDecode", "FP32");
  const auto data_type = causallm_test::makeTinyFp32DataType();
  const std::vector<unsigned int> ids = {1, 4, 2, 3};

  auto source_model_cfg = makeTinyGemma4Config();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  source_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  setupGemma4PleSensitiveWeights(source);
  source.saveWeight(files.weight_path.string());

  const auto resident_prefill = source.prefillLogitsFromIds(ids);
  const auto resident_tokens = source.greedyGenerateFromIds(ids, 2);

  // Use a separate model so its KV cache cannot affect the source result.
  // Keep the PLE gate/projection identical and clear only the PLE table. The
  // non-zero path is feature 0 -> feature 1 -> feature 1, which changes the
  // normalized hidden-state direction rather than only its magnitude.
  auto zero_ple_model_cfg = makeTinyGemma4Config();
  auto zero_ple_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto zero_ple_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  zero_ple_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM zero_ple_model(zero_ple_model_cfg, zero_ple_generation_cfg,
                                    zero_ple_nntr_cfg);
  zero_ple_model.initializeModel();
  setupGemma4PleSensitiveWeights(zero_ple_model);
  clearGemma4SplitPleEmbeddingWeights(zero_ple_model);
  const auto zero_ple_prefill = zero_ple_model.prefillLogitsFromIds(ids);
  ASSERT_EQ(resident_prefill.size(), zero_ple_prefill.size());
  bool has_ple_contribution = false;
  for (size_t i = 0; i < resident_prefill.size(); ++i) {
    if (std::fabs(resident_prefill[i] - zero_ple_prefill[i]) > 1e-5f) {
      has_ple_contribution = true;
      break;
    }
  }
  EXPECT_TRUE(has_ple_contribution)
    << "non-zero split PLE must change at least one prefill logit";

  auto fsu_model_cfg = makeTinyGemma4Config();
  auto fsu_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto fsu_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  fsu_nntr_cfg["fsu"] = true;
  fsu_nntr_cfg["fsu_lookahead"] = 1;
  fsu_nntr_cfg["ple_split"] = true;

  TinyGemma4CausalLM fsu_model(fsu_model_cfg, fsu_generation_cfg, fsu_nntr_cfg);
  fsu_model.initializeModel();
  fsu_model.loadWeight(files.weight_path.string());

  const auto backing_bytes = readBinaryFile(files.weight_path);
  const auto backing_size = std::filesystem::file_size(files.weight_path);
  ASSERT_FALSE(backing_bytes.empty());
  const auto rejected_path = files.dir / "fsu_save_rejected.bin";
  EXPECT_FALSE(std::filesystem::exists(rejected_path));

  // Transformer wraps NeuralNetwork's pre-open invalid_argument as
  // runtime_error. The backing file must remain usable after either rejection.
  EXPECT_THROW(fsu_model.saveWeight(files.weight_path.string()),
               std::runtime_error);
  EXPECT_EQ(std::filesystem::file_size(files.weight_path), backing_size);
  EXPECT_EQ(readBinaryFile(files.weight_path), backing_bytes);
  EXPECT_THROW(fsu_model.saveWeight(rejected_path.string()),
               std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(rejected_path));

  const auto state_before = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_before.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_before.all_virtual);
  EXPECT_FALSE(state_before.any_allocated);

  const auto fsu_prefill = fsu_model.prefillLogitsFromIds(ids);
  ASSERT_EQ(fsu_prefill.size(), resident_prefill.size());
  for (size_t i = 0; i < resident_prefill.size(); ++i)
    EXPECT_NEAR(fsu_prefill[i], resident_prefill[i], 1e-4f) << "logit " << i;
  EXPECT_EQ(std::filesystem::file_size(files.weight_path), backing_size);
  EXPECT_EQ(readBinaryFile(files.weight_path), backing_bytes);

  const auto state_after_prefill = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_after_prefill.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_after_prefill.all_virtual);
  EXPECT_FALSE(state_after_prefill.any_allocated);

  const auto fsu_tokens = fsu_model.greedyGenerateFromIds(ids, 2);
  EXPECT_EQ(fsu_tokens, resident_tokens);

  const auto state_after_decode = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_after_decode.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_after_decode.all_virtual);
  EXPECT_FALSE(state_after_decode.any_allocated);
#endif
}

/**
 * @brief Q4_0 split PLE keeps resident and virtual inference identical.
 */
TEST(Gemma4PleFsuTest, Q40LazyPleWeightMatchesResidentSplitLayout) {
#if defined(_WIN32)
  GTEST_SKIP() << "virtual tensor mmap is not supported on Windows";
#else
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "Q40LazyPleWeightMatchesResidentSplitLayout",
    "Q40_FP32");
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  const std::vector<unsigned int> ids = {1, 4, 2, 3};

  auto source_model_cfg = makeTinyGemma4Config();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, fp32_data_type);
  source_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  setupGemma4PleSensitiveWeights(source);
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeGemma4LayerDtypeMap(q40_data_type));

  auto resident_model_cfg = makeTinyGemma4Config();
  auto resident_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto resident_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, q40_data_type);
  resident_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM resident(resident_model_cfg, resident_generation_cfg,
                              resident_nntr_cfg);
  resident.initializeModel();
  resident.loadWeight(files.weight_path.string());
  const auto resident_prefill = resident.prefillLogitsFromIds(ids);
  const auto resident_tokens = resident.greedyGenerateFromIds(ids, 2);

  auto fsu_model_cfg = makeTinyGemma4Config();
  auto fsu_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto fsu_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, q40_data_type);
  fsu_nntr_cfg["fsu"] = true;
  fsu_nntr_cfg["fsu_lookahead"] = 1;
  fsu_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM fsu_model(fsu_model_cfg, fsu_generation_cfg, fsu_nntr_cfg);
  fsu_model.initializeModel();
  fsu_model.loadWeight(files.weight_path.string());

  const auto state_before = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_before.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_before.all_virtual);
  EXPECT_FALSE(state_before.any_allocated);

  const auto fsu_prefill = fsu_model.prefillLogitsFromIds(ids);
  ASSERT_EQ(fsu_prefill.size(), resident_prefill.size());
  for (size_t i = 0; i < resident_prefill.size(); ++i)
    EXPECT_NEAR(fsu_prefill[i], resident_prefill[i], 1e-4f) << "logit " << i;

  EXPECT_EQ(fsu_model.greedyGenerateFromIds(ids, 2), resident_tokens);
  const auto state_after_decode = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_after_decode.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_after_decode.all_virtual);
  EXPECT_FALSE(state_after_decode.any_allocated);
#endif
}

/**
 * @brief Virtual PLE is unmapped when its embedding lookup throws.
 */
TEST(Gemma4PleFsuTest, VirtualPleUnmapsAfterInvalidPleToken) {
#if defined(_WIN32)
  GTEST_SKIP() << "virtual tensor mmap is not supported on Windows";
#else
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4PleFsuTest", "VirtualPleUnmapsAfterInvalidPleToken", "FP32");
  const auto data_type = causallm_test::makeTinyFp32DataType();

  auto source_model_cfg = makeTinyGemma4Config();
  source_model_cfg["text_config"]["vocab_size_per_layer_input"] = 4;
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  source_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  setupGemma4DeterministicWeights(source);
  source.saveWeight(files.weight_path.string());

  auto fsu_model_cfg = makeTinyGemma4Config();
  fsu_model_cfg["text_config"]["vocab_size_per_layer_input"] = 4;
  auto fsu_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto fsu_nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  fsu_nntr_cfg["fsu"] = true;
  fsu_nntr_cfg["fsu_lookahead"] = 1;
  fsu_nntr_cfg["ple_split"] = true;
  TinyGemma4CausalLM fsu_model(fsu_model_cfg, fsu_generation_cfg, fsu_nntr_cfg);
  fsu_model.initializeModel();
  fsu_model.loadWeight(files.weight_path.string());

  // Token 4 is valid for the primary vocabulary (size 32), but invalid for
  // the intentionally smaller PLE vocabulary (size 4), so the exception is
  // thrown after the virtual PLE weight is activated.
  EXPECT_THROW(fsu_model.prefillLogitsFromIds({4}), std::invalid_argument);
  const auto state_after_error = getPleWeightFsuState(fsu_model);
  EXPECT_EQ(state_after_error.count, tiny_gemma4_num_layers);
  EXPECT_TRUE(state_after_error.all_virtual);
  EXPECT_FALSE(state_after_error.any_allocated);
#endif
}

INSTANTIATE_TEST_SUITE_P(
  Gemma4, Gemma4TinyModelTest,
  ::testing::Values(makeGemma4Case(causallm_test::makeTinyFp32DataType()),
                    makeGemma4Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Gemma4Fp16, Gemma4TinyModelTest,
  ::testing::Values(makeGemma4Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

} // namespace
