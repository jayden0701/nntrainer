// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen2.cpp
 * @date   18 May 2026
 * @brief  Tiny Qwen2 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen2_causallm.h>
#include <qwen2_embedding.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

/**
 * @brief Anonymous namespace for tiny Qwen2 test helpers
 */
namespace {

/**
 * @brief Tiny Qwen2 CausalLM adapter for common model tests
 */
class TinyQwen2CausalLM final : public causallm::Qwen2CausalLM,
                                public causallm_test::TinyCausalLMRunner {
public:
  /**
   * @brief Construct a tiny Qwen2 CausalLM test adapter
   */
  TinyQwen2CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                    causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm::Qwen2CausalLM(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen2 model
   */
  void initializeModel() override { initialize(); }

  /**
   * @brief Save tiny Qwen2 model weights
   */
  void saveWeight(const std::string &path) override { save_weight(path); }

  /**
   * @brief Save tiny Qwen2 model weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) override {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen2 model weights
   */
  void loadWeight(const std::string &path) override { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen2 weights for golden token tests
   */
  void setDeterministicWeights() override {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
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
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  /**
   * @brief Run one prompt through the tiny Qwen2 model
   */
  void runPrompt(const std::string &prompt) override {
    run(prompt, false, "", "", false);
  }

  /**
   * @brief Run Qwen2 prefill and return logits before sampling
   */
  std::vector<float> prefillLogits(const std::string &prompt) override {
    allocateAndBindKVCache();

    auto encoded = tokenizer->Encode(prompt);
    if (encoded.empty())
      throw std::invalid_argument("tiny Qwen2 prompt encoded to no tokens");

    const unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
    const unsigned int init_len = static_cast<unsigned int>(
      std::min<size_t>(encoded.size(), num_allow_str));
    std::vector<float> input_sample(
      static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN, 0.0f);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      for (unsigned int i = 0; i < init_len; ++i) {
        const auto token_id = static_cast<unsigned int>(encoded[i]);
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
          static_cast<float>(token_id);
        ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = token_id;
      }
    }

    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }

    std::sort(
      cache_inputs.begin(), cache_inputs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float *> input;
    input.reserve(1 + cache_inputs.size());
    input.push_back(input_sample.data());
    for (const auto &cache_input : cache_inputs)
      input.push_back(cache_input.second);

    std::vector<float *> label;
    setKVCachePosition(0);
    auto output = model->incremental_inference(BATCH_SIZE, input, label,
                                               init_len, 0, init_len, false);
    std::vector<float> logits(output[0], output[0] + NUM_VOCAB);
    for (auto &out : output)
      delete[] out;

    return logits;
  }

  /**
   * @brief Get generated output text
   */
  std::string getOutputText(int batch_idx = 0) const override {
    return getOutput(batch_idx);
  }

  /**
   * @brief Get whether the tiny Qwen2 model has completed run()
   */
  bool hasRun() const override { return causallm::CausalLM::hasRun(); }

  /**
   * @brief Read one token from the Qwen2 input/output history
   */
  unsigned int tokenAt(size_t idx) const override { return ids_history[idx]; }

  /**
   * @brief Generate ids from logits through Qwen2 decoding logic
   */
  std::vector<unsigned int>
  generateFromLogits(float *logits, bool do_sample, float repetition_penalty,
                     unsigned int *input_ids,
                     unsigned int num_input_ids) override {
    return generate(logits, do_sample, repetition_penalty, input_ids,
                    num_input_ids);
  }
};

/**
 * @brief Files generated for one tiny Qwen2.5 embedding test invocation
 */
struct TinyQwen25EmbeddingFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny Qwen2.5 Embedding adapter for model-level tests
 */
class TinyQwen25Embedding final : public causallm::Qwen2Embedding {
public:
  /**
   * @brief Construct a tiny Qwen2.5 Embedding test adapter
   */
  TinyQwen25Embedding(causallm::json &cfg, causallm::json &generation_cfg,
                      causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm::Qwen2Embedding(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen2.5 embedding model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny Qwen2.5 embedding weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen2.5 embedding model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen2.5 embedding weights
   */
  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
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
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  /**
   * @brief Encode one prompt and copy the embedding output
   */
  std::vector<float> encodePrompt(const std::string &prompt) {
    auto output = encode(prompt);
    std::vector<float> embedding(output[0], output[0] + BATCH_SIZE * DIM);
    for (auto &out : output)
      delete[] out;
    return embedding;
  }

  /**
   * @brief Return whether the tiny embedding model uses causal attention
   */
  bool isCausalForTest() const { return IS_CAUSAL; }
};

/**
 * @brief Write a string into a file
 */
void writeTextFile(const std::filesystem::path &path,
                   const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());

  file << content;
  if (!file.good())
    throw std::runtime_error("failed to write " + path.string());
}

/**
 * @brief Make tiny Qwen2.5 SentenceTransformer module config files
 */
std::filesystem::path
writeTinyQwen25EmbeddingModules(const std::filesystem::path &dir) {
  auto modules_path = dir / "modules.json";
  auto pooling_dir = dir / "1_Pooling";
  std::filesystem::create_directories(pooling_dir);

  writeTextFile(modules_path, R"([
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "",
    "type": "sentence_transformers.models.Normalize"
  }
])");

  writeTextFile(pooling_dir / "config.json", R"({
  "word_embedding_dimension": 64,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": false,
  "include_prompt": false
})");

  return modules_path;
}

/**
 * @brief Make the tiny Qwen2 model config
 */
causallm::json makeTinyQwen2Config() {
  return {
    {"architectures", {"Qwen2ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny Qwen2.5 embedding model config
 */
causallm::json makeTinyQwen25EmbeddingConfig() {
  auto cfg = makeTinyQwen2Config();
  cfg["architectures"] = {"Qwen2Model"};
  cfg.erase("is_causal");
  return cfg;
}

/**
 * @brief Make tiny Qwen2.5 embedding test files
 */
TinyQwen25EmbeddingFiles makeQwen25EmbeddingFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "Qwen25EmbeddingTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "Qwen25Embedding_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyQwen25EmbeddingModules(files.dir),
    files.dir / "qwen25_embedding_tiny.bin",
  };
}

/**
 * @brief Make the tiny Qwen2.5 embedding nntrainer config
 */
causallm::json makeTinyQwen25EmbeddingNntrainerConfig(
  const TinyQwen25EmbeddingFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Qwen2 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen2LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the tiny Qwen2.5 embedding Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen25EmbeddingQ40LayerDtypeMap() {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["embedding0"] = dtype;
  dtype_map["layer0_wq"] = dtype;
  dtype_map["layer0_wk"] = dtype;
  dtype_map["layer0_wv"] = dtype;
  dtype_map["layer0_attention_out"] = dtype;
  dtype_map["layer0_ffn_gate"] = dtype;
  dtype_map["layer0_ffn_up"] = dtype;
  dtype_map["layer0_ffn_down"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny Qwen2.5 embedding model
 */
std::unique_ptr<TinyQwen25Embedding>
makeLoadedQwen25Embedding(const TinyQwen25EmbeddingFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyQwen25EmbeddingConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyQwen25EmbeddingNntrainerConfig(files, fp32_data_type);

  TinyQwen25Embedding source(source_model_cfg, source_generation_cfg,
                             source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeQwen25EmbeddingQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyQwen25EmbeddingConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyQwen25EmbeddingNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyQwen25Embedding>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Qwen2 prefill logits
 */
std::vector<float> makeExpectedQwen2Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen2 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen2Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen2_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen2Logits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyQwen2Config,
    makeQwen2LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen2CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Qwen2 CausalLM model cases
 */
class Qwen2CausalLMTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Qwen2CausalLMTinyModelTest";
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
TEST_P(Qwen2CausalLMTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
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
TEST_P(Qwen2CausalLMTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(Qwen2CausalLMTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

/**
 * @brief Test that embedding_file_name reaches the embedding layer sidecar path
 */
TEST_P(Qwen2CausalLMTinyModelTest,
       EmbeddingFileNameIsPassedToEmbeddingLayerSidecarPath) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  config.model["tie_word_embeddings"] = false;
  config.nntrainer["embedding_file_name"] =
    (files.dir / "missing_sidecar_lut.bin").string();
  auto model = std::make_unique<TinyQwen2CausalLM>(
    config.model, config.generation, config.nntrainer);

  EXPECT_THROW(model->initializeModel(), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
  Qwen2, Qwen2CausalLMTinyModelTest,
  ::testing::Values(makeQwen2Case(causallm_test::makeTinyFp32DataType()),
                    makeQwen2Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

/**
 * @brief Test that a tiny Qwen2.5 embedding model can save/load and encode
 */
TEST(Qwen25EmbeddingTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40BidirectionalModel) {
  const auto files = makeQwen25EmbeddingFiles();
  auto model = makeLoadedQwen25Embedding(files);

  EXPECT_FALSE(model->isCausalForTest());

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero = false;
  for (float value : embedding) {
    ASSERT_TRUE(std::isfinite(value));
    has_non_zero = has_non_zero || value != 0.0f;
  }
  EXPECT_TRUE(has_non_zero);
}

} // namespace
