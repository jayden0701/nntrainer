// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_kv_cache_cuda_graph.cpp
 * @brief FP16 Qwen-shaped actual wrapper/graph tests for owned CUDA KV.
 * @author Codex (OpenAI)
 * @bug Synthetic weights; public model parity and capture/replay are separate.
 */
#include <causallm_test_utils.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <gtest/gtest.h>
#include <layer.h>
#include <layer_context.h>
#include <neuralnet.h>
#include <qwen3_causallm.h>
#include <vector>

namespace {
class Model
  : public causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM> {
public:
  Model(causallm::json &cfg, causallm::json &generation,
        causallm::json &config) :
    causallm::Transformer(cfg, generation, config,
                          causallm::ModelType::CAUSALLM),
    CausalLMTestAdapter(cfg, generation, config) {}

  void replayFeeds(bool enabled) {
    auto *nn = dynamic_cast<nntrainer::NeuralNetwork *>(this->model.get());
    nn->setGraphReplayFeedNodes(enabled ? std::vector<std::string>{"embedding0"}
                                        : std::vector<std::string>{});
  }
};
class CudaKvGraph : public ::testing::Test {
protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
      GTEST_SKIP() << "No CUDA device";
  }
  std::unique_ptr<Model> makeModel(const std::string &name,
                                   unsigned int capacity = 8) {
    auto files = causallm_test::makeTinyCausalLMFiles("CudaKv", name, "FP16");
    causallm::json cfg = {{"architectures", {"Qwen3ForCausalLM"}},
                          {"bos_token_id", 0},
                          {"eos_token_id", {31}},
                          {"head_dim", 16},
                          {"hidden_size", 64},
                          {"intermediate_size", 64},
                          {"is_causal", true},
                          {"max_position_embeddings", capacity},
                          {"num_attention_heads", 4},
                          {"num_hidden_layers", 1},
                          {"num_key_value_heads", 2},
                          {"rms_norm_eps", 1e-5},
                          {"rope_theta", 10000},
                          {"tie_word_embeddings", true},
                          {"vocab_size", 32}};
    auto generation = causallm_test::makeTinyGenerationConfig();
    causallm_test::TinyCausalLMDataType dtype{"FP16", "FP32", "FP16", "FP32",
                                              "FP16-FP16"};
    auto config =
      causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, dtype);
    config["max_seq_len"] = capacity;
    auto result = std::make_unique<Model>(cfg, generation, config);
    result->initializeModel();
    result->forEachLayer([](ml::train::Layer &layer,
                            nntrainer::RunLayerContext &context, void *) {
      for (unsigned int w = 0; w < context.getNumWeights(); ++w) {
        auto &weight = context.getWeight(w);
        if (!weight.getMemoryData()->isHostAddressable())
          throw std::runtime_error("Synthetic weights require host access");
        const bool norm = causallm_test::isRmsNormLayerType(layer.getType());
        for (size_t i = 0; i < weight.size(); ++i) {
          const float value =
            norm ? 1.0f
                 : (static_cast<int>((i + layer.getName().size()) % 17) - 8) *
                     0.025f;
          if (weight.getDataType() == ml::train::TensorDim::DataType::FP16)
            weight.getData<_FP16>()[i] = static_cast<_FP16>(value);
          else if (weight.getDataType() == ml::train::TensorDim::DataType::FP32)
            weight.getData<float>()[i] = value;
          else
            throw std::runtime_error("Unexpected synthetic weight dtype");
        }
      }
    });
    return result;
  }
  void near(const std::vector<float> &a, const std::vector<float> &b) {
    ASSERT_EQ(a.size(), b.size());
    float largest = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      ASSERT_TRUE(std::isfinite(a[i]));
      ASSERT_TRUE(std::isfinite(b[i]));
      EXPECT_NEAR(a[i], b[i], 0.01f) << i;
      largest = std::max(largest, std::fabs(a[i]));
    }
    EXPECT_GT(largest, 0.001f);
  }
};
} // namespace

TEST_F(CudaKvGraph, realWrapperMatchesChunkedPrefillAndDecode) {
  auto full = makeModel("full");
  const auto expected = full->stepLogitsFromIds({1, 4, 2}, 0);
  const auto expected_decode = full->stepLogitsFromIds({3}, 3);
  auto chunked = makeModel("chunked");
  chunked->stepLogitsFromIds({1}, 0);
  near(expected, chunked->stepLogitsFromIds({4, 2}, 1));
  near(expected_decode, chunked->stepLogitsFromIds({3}, 3));
  auto &cache = chunked->cacheManager();
  EXPECT_TRUE(cache.usesCoreStorage());
  EXPECT_FALSE(cache.usesCpuStorage());
  EXPECT_FALSE(cache.getKeyCache(0).getMemoryData()->isHostAddressable());
  EXPECT_EQ(cache.getPosition(), 4u);
  EXPECT_EQ(cache.coreStats().device_attention_calls, 3u);
  if (cache.directValueWrites()) {
    EXPECT_EQ(cache.coreStats().append_copy_bytes, 0u);
    EXPECT_EQ(cache.coreStats().direct_value_writes, 3u);
  } else {
    EXPECT_GT(cache.coreStats().append_copy_bytes, 0u);
  }
}

TEST_F(CudaKvGraph, capacityIntervalsAndRepeatedResetUseRealGraph) {
  auto model = makeModel("capacity");
  model->prepareKVCache();
  EXPECT_THROW(model->stepLogitsFromIds({1}, 1), std::logic_error);
  for (int repeat = 0; repeat < 3; ++repeat) {
    model->cacheManager().reset();
    for (unsigned int position = 0; position < 8; position += 2)
      model->stepLogitsFromIds({1, 4}, position);
    EXPECT_EQ(model->cacheManager().getPosition(), 8u);
    EXPECT_THROW(model->stepLogitsFromIds({1}, 8), std::out_of_range);
  }
  EXPECT_EQ(model->cacheManager().coreStats().device_attention_calls, 12u);
}

TEST_F(CudaKvGraph, snapshotRestoresSameDeviceContinuation) {
  auto model = makeModel("snapshot");
  model->stepLogitsFromIds({1, 4, 2}, 0);
  auto files =
    causallm_test::makeTinyCausalLMFiles("CudaKv", "snapshot_io", "FP16");
  auto path = (files.dir / "kv.bin").string();
  model->cacheManager().save(path, 3);
  auto expected = model->stepLogitsFromIds({3}, 3);
  EXPECT_EQ(model->cacheManager().load(path, 3), 3u);
  near(expected, model->stepLogitsFromIds({3}, 3));
}

TEST_F(CudaKvGraph, missingProducerCannotPublishAfterActualMha) {
  auto model = makeModel("missing");
  model->addUnproducedCacheOwner();
  EXPECT_THROW(model->stepLogitsFromIds({1, 4}, 0), std::logic_error);
  EXPECT_EQ(model->cacheManager().getPosition(), 0u);
  EXPECT_EQ(model->cacheManager().coreStats().device_attention_calls,
            model->cacheManager().directValueWrites() ? 0u : 1u);
}

TEST_F(CudaKvGraph, foreignCacheIsRejectedBeforeProducerWrites) {
  auto model = makeModel("foreign");
  model->prepareKVCache();
  nntrainer::Tensor foreign(model->cacheManager().getKeyCache(0).getDim(),
                            true);
  auto &plane = model->cacheManager().getKeyCache(0);
  auto original = plane.getMemoryData();
  auto offset = plane.getOffset();
  plane.setData(foreign.getMemoryData(), 0, false);
  EXPECT_THROW(model->stepLogitsFromIds({1, 4}, 0), std::logic_error);
  EXPECT_EQ(model->cacheManager().getPosition(), 0u);
  plane.setData(original, offset, false);
  EXPECT_NO_THROW(model->stepLogitsFromIds({1, 4}, 0));
  EXPECT_EQ(model->cacheManager().getPosition(), 2u);
}

TEST_F(CudaKvGraph, directProducerMatchesCopyPathAndRestoresBindings) {
  std::vector<float> expected_prefill, expected_decode;
  std::vector<unsigned short> expected_key, expected_value;
  {
    causallm_test::ScopedEnv mode("NNTR_CUDA_KV_DIRECT", "0");
    auto compatibility = makeModel("copy_reference");
    expected_prefill = compatibility->stepLogitsFromIds({1, 4, 2}, 0);
    expected_decode = compatibility->stepLogitsFromIds({3}, 3);
    auto &key = compatibility->cacheManager().getKeyCache(0);
    auto &value = compatibility->cacheManager().getValueCache(0);
    expected_key.resize(key.size());
    expected_value.resize(value.size());
    ASSERT_EQ(cudaMemcpy(expected_key.data(), key.getData<char>(), key.bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(expected_value.data(), value.getData<char>(),
                         value.bytes(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(compatibility->cacheManager().coreStats().append_copy_bytes,
              256u);
  }
  causallm_test::ScopedEnv mode("NNTR_CUDA_KV_DIRECT", "1");
  auto direct = makeModel("direct");
  near(expected_prefill, direct->stepLogitsFromIds({1, 4, 2}, 0));
  near(expected_decode, direct->stepLogitsFromIds({3}, 3));
  auto &cache = direct->cacheManager();
  EXPECT_EQ(cache.coreStats().append_copy_bytes, 0u);
  EXPECT_EQ(cache.coreStats().direct_value_writes, 2u);
  std::vector<unsigned short> actual(expected_key.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), cache.getKeyCache(0).getData<char>(),
                       cache.getKeyCache(0).bytes(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, expected_key);
  ASSERT_EQ(cudaMemcpy(actual.data(), cache.getValueCache(0).getData<char>(),
                       cache.getValueCache(0).bytes(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, expected_value);
  const void *producer = nullptr, *consumer = nullptr;
  direct->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "layer0_wv")
        producer = context.getOutput(0).getData<char>();
      if (layer.getName() == "layer0_attention")
        consumer = context.getInput(2).getData<char>();
    });
  EXPECT_NE(producer, nullptr);
  EXPECT_EQ(producer, consumer);
  EXPECT_NE(producer, cache.getValueCache(0).getData<char>());
}

TEST_F(CudaKvGraph, unsupportedProfilesCannotAllocateStrictBacking) {
  // Register the CUDA context before testing live backend resolution.
  auto model = makeModel("admission");
  using Type = ml::train::TensorDim::DataType;
  causallm::KVCacheManager cache;
  EXPECT_THROW(cache.allocate(1, 2, 8, 2, 16), std::invalid_argument);
  EXPECT_THROW(cache.allocate(1, 1, 8, 2, 16, Type::FP32),
               std::invalid_argument);
  EXPECT_THROW(cache.allocate(1, 1, 8, std::vector<unsigned int>{32}),
               std::invalid_argument);
  for (const auto *name :
       {"NNTR_CUDA_GRAPH", "NNTR_CUDA_M2B", "NNTR_CUDA_PREFILL_GRAPH",
        "NNTR_KV_INT8", "NNTR_KV_OHWI"}) {
    causallm_test::ScopedEnv unsupported(name, "1");
    EXPECT_THROW(cache.allocate(1, 1, 8, 2, 16), std::invalid_argument) << name;
  }
  for (const auto *name : {"NNTR_CUDA_ATTN", "NNTR_CUDA_ROPE",
                           "NNTR_CUDA_QKNORM", "NNTR_CUDA_ELTWISE"}) {
    causallm_test::ScopedEnv missing(name, "0");
    EXPECT_THROW(cache.allocate(1, 1, 8, 2, 16), std::invalid_argument) << name;
  }
  {
    causallm_test::ScopedEnv cpu("NNTR_ENGINE", "cpu");
    EXPECT_THROW(cache.allocate(1, 1, 8, 2, 16), std::invalid_argument);
  }
  cache.setLayerKVSources({-1, 0});
  EXPECT_THROW(cache.allocate(2, 1, 8, 2, 16), std::invalid_argument);
  cache.setLayerKVSources({});
  cache.setLayerProducers({false});
  EXPECT_THROW(cache.allocate(1, 1, 8, 2, 16), std::invalid_argument);
  EXPECT_FALSE(cache.isAllocated());
}

TEST_F(CudaKvGraph, replayMatchesEagerEveryLogitAndKvByte) {
  std::vector<std::vector<float>> golden;
  std::vector<unsigned short> keys, values;
  {
    auto eager = makeModel("replay_reference", 96);
    eager->replayFeeds(false);
    golden.push_back(eager->stepLogitsFromIds({1, 4}, 0));
    for (unsigned int pos = 2; pos < 72; ++pos)
      golden.push_back(eager->stepLogitsFromIds({pos % 20 + 1}, pos));
    auto &cache = eager->cacheManager();
    keys.resize(cache.getKeyCache(0).size());
    values.resize(keys.size());
    ASSERT_EQ(cudaMemcpy(keys.data(), cache.getKeyCache(0).getData<char>(),
                         keys.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(values.data(), cache.getValueCache(0).getData<char>(),
                         values.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cache.coreStats().graph_replays, 0u);
  }
  auto replay = makeModel("replay_actual", 96);
  EXPECT_EQ(golden[0], replay->stepLogitsFromIds({1, 4}, 0));
  for (unsigned int pos = 2; pos < 72; ++pos)
    EXPECT_EQ(golden[pos - 1], replay->stepLogitsFromIds({pos % 20 + 1}, pos))
      << pos;
  auto &cache = replay->cacheManager();
  std::vector<unsigned short> actual(keys.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), cache.getKeyCache(0).getData<char>(),
                       actual.size() * 2, cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, keys);
  ASSERT_EQ(cudaMemcpy(actual.data(), cache.getValueCache(0).getData<char>(),
                       actual.size() * 2, cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual, values);
  EXPECT_GT(cache.coreStats().graph_captures, 0u);
  EXPECT_GT(cache.coreStats().graph_replays, 1u);
  EXPECT_EQ(cache.coreStats().append_copy_bytes, 0u);
  EXPECT_EQ(cache.getPosition(), 72u);
  EXPECT_THROW(replay->stepLogitsFromIds(std::vector<unsigned int>(25, 1), 72),
               std::out_of_range);
}

TEST_F(CudaKvGraph, replayInvalidatesOnRestoreResetAndOwnerSwitch) {
  auto a = makeModel("replay_lifetime_a", 16);
  a->stepLogitsFromIds({1, 4}, 0);
  for (unsigned int pos = 2; pos < 5; ++pos)
    a->stepLogitsFromIds({3}, pos);
  ASSERT_GT(a->cacheManager().coreStats().graph_replays, 0u);
  auto files =
    causallm_test::makeTinyCausalLMFiles("CudaKv", "replay_file", "FP16");
  auto path = (files.dir / "kv.bin").string();
  a->cacheManager().save(path, 5);
  auto expected = a->stepLogitsFromIds({2}, 5);
  auto captures = a->cacheManager().coreStats().graph_captures;
  ASSERT_EQ(a->cacheManager().load(path, 5), 5u);
  EXPECT_EQ(expected, a->stepLogitsFromIds({2}, 5));
  EXPECT_GT(a->cacheManager().coreStats().graph_captures, captures);
  {
    auto b = makeModel("replay_lifetime_b", 16);
    b->stepLogitsFromIds({1, 4}, 0);
    for (unsigned int pos = 2; pos < 5; ++pos)
      b->stepLogitsFromIds({3}, pos);
    EXPECT_EQ(expected, b->stepLogitsFromIds({2}, 5));
    EXPECT_EQ(a->stepLogitsFromIds({4}, 6), b->stepLogitsFromIds({4}, 6));
  }
  EXPECT_NO_THROW(a->stepLogitsFromIds({1}, 7));
  a->cacheManager().reset();
  EXPECT_NO_THROW(a->stepLogitsFromIds({1, 4}, 0));
  a->cacheManager().allocate(1, 1, 16, 2, 16);
  EXPECT_NO_THROW(a->stepLogitsFromIds({1, 4}, 0));
  EXPECT_EQ(a->cacheManager().getPosition(), 2u);
}

TEST_F(CudaKvGraph, replayCaptureExceptionDrainsAndRetries) {
  auto model = makeModel("replay_failure", 16);
  model->stepLogitsFromIds({1, 4}, 0);
  model->stepLogitsFromIds({3}, 2);
  auto &cache = model->cacheManager();
  auto &plane = cache.getKeyCache(0);
  auto saved = plane.getMemoryData();
  auto offset = plane.getOffset();
  nntrainer::Tensor foreign(plane.getDim(), true);
  plane.setData(foreign.getMemoryData(), 0, false);
  EXPECT_THROW(model->stepLogitsFromIds({2}, 3), std::logic_error);
  EXPECT_EQ(cache.getPosition(), 3u);
  EXPECT_FALSE(nntrainer::cuda::StreamManager::Global().isCapturing());
  plane.setData(saved, offset, false);
  EXPECT_NO_THROW(model->stepLogitsFromIds({2}, 3));
  EXPECT_NO_THROW(model->stepLogitsFromIds({4}, 4));
  EXPECT_EQ(cache.getPosition(), 5u);
  EXPECT_GT(cache.coreStats().graph_replays, 0u);
}

TEST_F(CudaKvGraph, replayAdmissionRequiresCoreAndCoherentSwitches) {
  auto model = makeModel("replay_admission", 16);
  causallm::KVCacheManager cache;
  {
    causallm_test::ScopedEnv missing_core("NNTR_CUDA_KV_CORE", "0");
    EXPECT_THROW(cache.allocate(1, 1, 16, 2, 16), std::invalid_argument);
  }
  {
    causallm_test::ScopedEnv mismatch("NNTR_CUDA_M2B", "0");
    EXPECT_THROW(cache.allocate(1, 1, 16, 2, 16), std::invalid_argument);
  }
  {
    causallm_test::ScopedEnv copy("NNTR_CUDA_KV_DIRECT", "0");
    EXPECT_THROW(cache.allocate(1, 1, 16, 2, 16), std::invalid_argument);
  }
  {
    causallm_test::ScopedEnv device_activations("NNTR_CUDA_DEV_ACT", "1");
    EXPECT_THROW(cache.allocate(1, 1, 16, 2, 16), std::invalid_argument);
  }
  {
    causallm_test::ScopedEnv prefill("NNTR_CUDA_PREFILL_GRAPH", "1");
    EXPECT_THROW(cache.allocate(1, 1, 16, 2, 16), std::invalid_argument);
  }
  EXPECT_FALSE(cache.isAllocated());
}
