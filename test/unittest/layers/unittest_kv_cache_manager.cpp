// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  Unit tests for KVCacheManager
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <kv_cache_manager.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

std::string makeTempPath(const std::string &filename) {
  return (std::filesystem::temp_directory_path() / filename).string();
}

} // namespace

/**
 * @class   KVCacheManagerTest
 * @brief   gtest fixture for the standalone host-side KVCacheManager
 *          (allocate / read+write views / position bookkeeping /
 *          save+load / multi-session / multi-turn / branching). Sized
 *          for a 4-layer, batch-2, seq-128 toy config so every test
 *          runs in well under a millisecond on host.
 */
class KVCacheManagerTest : public ::testing::Test {
protected:
  static constexpr unsigned int NUM_LAYERS = 4;
  static constexpr unsigned int BATCH_SIZE = 2;
  static constexpr unsigned int MAX_SEQ_LEN = 128;
  static constexpr unsigned int NUM_HEADS_KV = 4;
  static constexpr unsigned int HEAD_DIM = 8;
  static constexpr unsigned int KV_WIDTH = NUM_HEADS_KV * HEAD_DIM;

  void SetUp() override {
    manager.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV,
                     HEAD_DIM, ml::train::TensorDim::DataType::FP32);
  }

  causallm::KVCacheManager manager;
};

TEST_F(KVCacheManagerTest, allocate_basic) {
  EXPECT_TRUE(manager.isAllocated());
  EXPECT_EQ(manager.getNumLayers(), NUM_LAYERS);
  EXPECT_EQ(manager.getMaxSeqLen(), MAX_SEQ_LEN);
  EXPECT_EQ(manager.getBatchSize(), BATCH_SIZE);
  EXPECT_EQ(manager.getKVWidth(), KV_WIDTH);
  EXPECT_EQ(manager.getPosition(), 0u);
}

TEST_F(KVCacheManagerTest, allocate_invalid_params) {
  causallm::KVCacheManager m;
  EXPECT_THROW(m.allocate(0, 1, 128, 4, 8), std::invalid_argument);
  EXPECT_THROW(m.allocate(4, 0, 128, 4, 8), std::invalid_argument);
  EXPECT_THROW(m.allocate(4, 1, 0, 4, 8), std::invalid_argument);
}

TEST_F(KVCacheManagerTest, cache_tensor_dimensions) {
  auto &k = manager.getKeyCache(0);
  auto &v = manager.getValueCache(0);

  EXPECT_EQ(k.batch(), BATCH_SIZE);
  EXPECT_EQ(k.channel(), 1u);
  EXPECT_EQ(k.height(), MAX_SEQ_LEN);
  EXPECT_EQ(k.width(), KV_WIDTH);

  EXPECT_EQ(v.batch(), BATCH_SIZE);
  EXPECT_EQ(v.channel(), 1u);
  EXPECT_EQ(v.height(), MAX_SEQ_LEN);
  EXPECT_EQ(v.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, allocate_per_layer_capacities) {
  causallm::KVCacheManager compact;
  const std::vector<unsigned int> capacities = {4, MAX_SEQ_LEN, 8, MAX_SEQ_LEN};

  compact.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM,
                   capacities, ml::train::TensorDim::DataType::FP32);

  EXPECT_EQ(compact.getMaxSeqLen(), MAX_SEQ_LEN);
  for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
    EXPECT_EQ(compact.getCacheCapacity(i), capacities[i]);
    EXPECT_EQ(compact.getKeyCache(i).height(), capacities[i]);
    EXPECT_EQ(compact.getValueCache(i).height(), capacities[i]);
  }
}

TEST_F(KVCacheManagerTest, logical_position_independent_of_physical_capacity) {
  causallm::KVCacheManager compact;
  compact.allocate(2, 1, 8, 1, 2, {4, 8}, ml::train::TensorDim::DataType::FP32);

  EXPECT_NO_THROW(compact.setPosition(5));
  EXPECT_EQ(compact.getPosition(), 5u);
  EXPECT_NO_THROW(compact.advance(3));
  EXPECT_EQ(compact.getPosition(), 8u);
}

TEST_F(KVCacheManagerTest, allocate_invalid_cache_capacity_count) {
  causallm::KVCacheManager compact;
  EXPECT_THROW(compact.allocate(2, 1, 8, 1, 2, {4}), std::invalid_argument);
}

TEST_F(KVCacheManagerTest, allocate_invalid_cache_capacity_values) {
  causallm::KVCacheManager compact;
  EXPECT_THROW(compact.allocate(2, 1, 8, 1, 2, {0, 8}), std::invalid_argument);
  EXPECT_THROW(compact.allocate(2, 1, 8, 1, 2, {4, 9}), std::invalid_argument);
}

TEST_F(KVCacheManagerTest, position_management) {
  EXPECT_EQ(manager.getPosition(), 0u);

  manager.advance(10);
  EXPECT_EQ(manager.getPosition(), 10u);

  manager.advance(5);
  EXPECT_EQ(manager.getPosition(), 15u);

  manager.setPosition(50);
  EXPECT_EQ(manager.getPosition(), 50u);

  manager.reset();
  EXPECT_EQ(manager.getPosition(), 0u);
}

TEST_F(KVCacheManagerTest, position_bounds_check) {
  EXPECT_THROW(manager.setPosition(MAX_SEQ_LEN + 1), std::out_of_range);
  manager.setPosition(MAX_SEQ_LEN); // exactly at limit is ok

  manager.reset();
  manager.advance(MAX_SEQ_LEN);
  EXPECT_THROW(manager.advance(1), std::out_of_range);
}

TEST_F(KVCacheManagerTest, invalid_layer_idx) {
  EXPECT_THROW(manager.getCacheCapacity(NUM_LAYERS), std::out_of_range);
  EXPECT_THROW(manager.getKeyCache(NUM_LAYERS), std::out_of_range);
  EXPECT_THROW(manager.getValueCache(NUM_LAYERS), std::out_of_range);
  EXPECT_THROW(manager.getKeyCacheWriteView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheWriteView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getKeyCacheReadView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheReadView(NUM_LAYERS, 0, 1),
               std::out_of_range);
}

TEST_F(KVCacheManagerTest, invalid_batch_idx) {
  EXPECT_THROW(manager.getKeyCacheWriteView(0, BATCH_SIZE, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheWriteView(0, BATCH_SIZE, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getKeyCacheReadView(0, BATCH_SIZE, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheReadView(0, BATCH_SIZE, 1),
               std::out_of_range);
}

TEST_F(KVCacheManagerTest, write_view_dimensions) {
  unsigned int step_size = 3;
  auto view = manager.getKeyCacheWriteView(0, 0, step_size);

  EXPECT_EQ(view.batch(), 1u);
  EXPECT_EQ(view.channel(), 1u);
  EXPECT_EQ(view.height(), step_size);
  EXPECT_EQ(view.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, read_view_dimensions) {
  unsigned int read_len = 10;
  auto view = manager.getKeyCacheReadView(0, 0, read_len);

  EXPECT_EQ(view.batch(), 1u);
  EXPECT_EQ(view.channel(), 1u);
  EXPECT_EQ(view.height(), read_len);
  EXPECT_EQ(view.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, write_view_points_to_correct_location) {
  // Write at position 0
  auto write_view = manager.getKeyCacheWriteView(0, 0, 1);
  float *write_ptr = write_view.getData<float>();

  // Read from position 0
  auto read_view = manager.getKeyCacheReadView(0, 0, 1);
  float *read_ptr = read_view.getData<float>();

  // Should point to same memory
  EXPECT_EQ(write_ptr, read_ptr);
}

TEST_F(KVCacheManagerTest, write_and_read_data_consistency) {
  // Write some data at position 0
  auto write_view = manager.getKeyCacheWriteView(0, 0, 1);
  float *data = write_view.getData<float>();
  for (unsigned int i = 0; i < KV_WIDTH; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  // Read it back
  auto read_view = manager.getKeyCacheReadView(0, 0, 1);
  float *read_data = read_view.getData<float>();
  for (unsigned int i = 0; i < KV_WIDTH; ++i) {
    EXPECT_FLOAT_EQ(read_data[i], static_cast<float>(i + 1));
  }
}

TEST_F(KVCacheManagerTest, sequential_write_positions) {
  // Simulate prefill: write 5 tokens
  auto &k_cache = manager.getKeyCache(0);
  float *cache_base = k_cache.getData<float>();

  auto view0 = manager.getKeyCacheWriteView(0, 0, 5);
  float *ptr0 = view0.getData<float>();
  EXPECT_EQ(ptr0, cache_base); // starts at beginning

  // Advance position
  manager.advance(5);

  // Write 1 more token
  auto view1 = manager.getKeyCacheWriteView(0, 0, 1);
  float *ptr1 = view1.getData<float>();
  EXPECT_EQ(ptr1, cache_base + 5 * KV_WIDTH); // offset by 5 tokens
}

TEST_F(KVCacheManagerTest, batch_offset_correct) {
  auto &k_cache = manager.getKeyCache(0);
  float *cache_base = k_cache.getData<float>();
  size_t feature_len = k_cache.getDim().getFeatureLen();

  // Batch 0
  auto view_b0 = manager.getKeyCacheWriteView(0, 0, 1);
  float *ptr_b0 = view_b0.getData<float>();
  EXPECT_EQ(ptr_b0, cache_base);

  // Batch 1
  auto view_b1 = manager.getKeyCacheWriteView(0, 1, 1);
  float *ptr_b1 = view_b1.getData<float>();
  EXPECT_EQ(ptr_b1, cache_base + feature_len);
}

TEST_F(KVCacheManagerTest, multi_layer_independence) {
  // Write different data to layer 0 and layer 1
  auto view_l0 = manager.getKeyCacheWriteView(0, 0, 1);
  auto view_l1 = manager.getKeyCacheWriteView(1, 0, 1);

  view_l0.getData<float>()[0] = 42.0f;
  view_l1.getData<float>()[0] = 99.0f;

  auto read_l0 = manager.getKeyCacheReadView(0, 0, 1);
  auto read_l1 = manager.getKeyCacheReadView(1, 0, 1);

  EXPECT_FLOAT_EQ(read_l0.getData<float>()[0], 42.0f);
  EXPECT_FLOAT_EQ(read_l1.getData<float>()[0], 99.0f);
}

TEST_F(KVCacheManagerTest, save_and_load) {
  // Write data to all layers
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto k_view = manager.getKeyCacheWriteView(l, 0, 3);
    auto v_view = manager.getValueCacheWriteView(l, 0, 3);
    float *kd = k_view.getData<float>();
    float *vd = v_view.getData<float>();
    for (unsigned int i = 0; i < 3 * KV_WIDTH; ++i) {
      kd[i] = static_cast<float>(l * 1000 + i);
      vd[i] = static_cast<float>(l * 1000 + i + 500);
    }
  }
  manager.advance(3);

  // Save
  const std::string path = makeTempPath("test_kv_cache.bin");
  manager.save(path);

  // Create a new manager and load
  causallm::KVCacheManager loaded;
  loaded.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM,
                  ml::train::TensorDim::DataType::FP32);

  loaded.load(path, 3);
  EXPECT_EQ(loaded.getPosition(), 3u);

  // Verify data
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto k_read = loaded.getKeyCacheReadView(l, 0, 3);
    auto v_read = loaded.getValueCacheReadView(l, 0, 3);
    float *kd = k_read.getData<float>();
    float *vd = v_read.getData<float>();
    for (unsigned int i = 0; i < 3 * KV_WIDTH; ++i) {
      EXPECT_FLOAT_EQ(kd[i], static_cast<float>(l * 1000 + i))
        << "Key mismatch at layer=" << l << " i=" << i;
      EXPECT_FLOAT_EQ(vd[i], static_cast<float>(l * 1000 + i + 500))
        << "Value mismatch at layer=" << l << " i=" << i;
    }
  }

  // Cleanup
  std::remove(path.c_str());
}

TEST_F(KVCacheManagerTest, save_preserves_legacy_multibatch_layout) {
  constexpr unsigned int logical_len = 3;
  auto &key_cache = manager.getKeyCache(0);
  float *key_data = key_cache.getData<float>();
  const size_t saved_elements =
    static_cast<size_t>(BATCH_SIZE) * logical_len * KV_WIDTH;
  for (size_t i = 0; i < saved_elements; ++i) {
    key_data[i] = static_cast<float>(i + 1);
  }
  manager.setPosition(logical_len);

  const std::string path = makeTempPath("test_legacy_multibatch_kv_cache.bin");
  manager.save(path);

  std::ifstream saved(path, std::ios::binary);
  ASSERT_TRUE(saved.is_open());
  std::vector<float> stored(saved_elements);
  saved.read(reinterpret_cast<char *>(stored.data()),
             static_cast<std::streamsize>(stored.size() * sizeof(float)));
  ASSERT_EQ(saved.gcount(),
            static_cast<std::streamsize>(stored.size() * sizeof(float)));
  for (size_t i = 0; i < stored.size(); ++i) {
    EXPECT_FLOAT_EQ(stored[i], key_data[i]);
  }

  saved.close();
  std::remove(path.c_str());
}

TEST_F(KVCacheManagerTest, uint16_save_preserves_legacy_tensor_framing) {
  constexpr unsigned int max_seq_len = 8;
  constexpr unsigned int logical_len = 3;
  constexpr unsigned int width = 2;

  causallm::KVCacheManager uint16_cache;
  uint16_cache.allocate(1, 1, max_seq_len, 1, width,
                        ml::train::TensorDim::DataType::UINT16);

  auto fill_cache = [](nntrainer::Tensor &cache, std::uint16_t base) {
    auto *data = cache.getData<std::uint16_t>();
    for (size_t i = 0; i < cache.size(); ++i) {
      data[i] = static_cast<std::uint16_t>(base + i);
    }
  };
  fill_cache(uint16_cache.getKeyCache(0), 1);
  fill_cache(uint16_cache.getValueCache(0), 101);
  uint16_cache.setPosition(logical_len);

  std::vector<unsigned char> expected;
  auto append_legacy_tensor = [&](nntrainer::Tensor &cache) {
    const auto qscheme = static_cast<std::uint16_t>(cache.q_scheme());
    const auto *qscheme_bytes =
      reinterpret_cast<const unsigned char *>(&qscheme);
    expected.insert(expected.end(), qscheme_bytes,
                    qscheme_bytes + sizeof(qscheme));

    const size_t data_bytes =
      static_cast<size_t>(logical_len) * width * sizeof(std::uint16_t);
    const size_t trailing_bytes = cache.getMemoryBytes() - cache.bytes();
    const auto *data = reinterpret_cast<const unsigned char *>(cache.getData());
    expected.insert(expected.end(), data, data + data_bytes + trailing_bytes);
  };
  append_legacy_tensor(uint16_cache.getKeyCache(0));
  append_legacy_tensor(uint16_cache.getValueCache(0));

  const std::string path = makeTempPath("test_uint16_legacy_kv_cache.bin");
  uint16_cache.save(path);
  EXPECT_EQ(std::filesystem::file_size(path), expected.size());

  std::ifstream saved(path, std::ios::binary);
  ASSERT_TRUE(saved.is_open());
  std::vector<unsigned char> stored(expected.size());
  saved.read(reinterpret_cast<char *>(stored.data()),
             static_cast<std::streamsize>(stored.size()));
  ASSERT_EQ(saved.gcount(), static_cast<std::streamsize>(stored.size()));
  EXPECT_EQ(stored, expected);

  saved.close();
  std::remove(path.c_str());
}

TEST_F(KVCacheManagerTest, uint16_compact_load_save_preserves_framing) {
  constexpr unsigned int max_seq_len = 8;
  constexpr unsigned int logical_len = 6;
  constexpr unsigned int capacity = 4;

  causallm::KVCacheManager legacy;
  legacy.allocate(1, 1, max_seq_len, 1, 1,
                  ml::train::TensorDim::DataType::UINT16);
  auto *legacy_key = legacy.getKeyCache(0).getData<std::uint16_t>();
  auto *legacy_value = legacy.getValueCache(0).getData<std::uint16_t>();
  for (unsigned int i = 0; i < max_seq_len; ++i) {
    legacy_key[i] = static_cast<std::uint16_t>(i);
    legacy_value[i] = static_cast<std::uint16_t>(100 + i);
  }
  legacy.setPosition(logical_len);

  const std::string legacy_path =
    makeTempPath("test_uint16_legacy_compact_kv_cache.bin");
  const std::string compact_path =
    makeTempPath("test_uint16_compact_kv_cache.bin");
  legacy.save(legacy_path);

  causallm::KVCacheManager compact;
  compact.allocate(1, 1, max_seq_len, 1, 1, {capacity},
                   ml::train::TensorDim::DataType::UINT16);
  compact.load(legacy_path, logical_len);

  const auto *compact_key = compact.getKeyCache(0).getData<std::uint16_t>();
  const auto *compact_value = compact.getValueCache(0).getData<std::uint16_t>();
  for (unsigned int i = 0; i < capacity; ++i) {
    EXPECT_EQ(compact_key[i], static_cast<std::uint16_t>(i + 2));
    EXPECT_EQ(compact_value[i], static_cast<std::uint16_t>(i + 102));
  }

  compact.save(compact_path);
  EXPECT_EQ(std::filesystem::file_size(compact_path),
            std::filesystem::file_size(legacy_path));

  std::ifstream saved(compact_path, std::ios::binary);
  ASSERT_TRUE(saved.is_open());
  std::uint16_t qscheme = 0;
  saved.read(reinterpret_cast<char *>(&qscheme), sizeof(qscheme));
  EXPECT_EQ(qscheme,
            static_cast<std::uint16_t>(compact.getKeyCache(0).q_scheme()));
  std::vector<std::uint16_t> serialized_key(logical_len);
  saved.read(reinterpret_cast<char *>(serialized_key.data()),
             static_cast<std::streamsize>(serialized_key.size() *
                                          sizeof(std::uint16_t)));
  const std::vector<std::uint16_t> expected_key = {0, 0, 2, 3, 4, 5};
  EXPECT_EQ(serialized_key, expected_key);
  saved.close();

  causallm::KVCacheManager reloaded;
  reloaded.allocate(1, 1, max_seq_len, 1, 1, {capacity},
                    ml::train::TensorDim::DataType::UINT16);
  reloaded.load(compact_path, logical_len);
  const auto *reloaded_key = reloaded.getKeyCache(0).getData<std::uint16_t>();
  const auto *reloaded_value =
    reloaded.getValueCache(0).getData<std::uint16_t>();
  for (unsigned int i = 0; i < capacity; ++i) {
    EXPECT_EQ(reloaded_key[i], compact_key[i]);
    EXPECT_EQ(reloaded_value[i], compact_value[i]);
  }

  std::remove(legacy_path.c_str());
  std::remove(compact_path.c_str());
}

TEST_F(KVCacheManagerTest, save_load_mixed_capacities_after_wrap) {
  constexpr unsigned int logical_len = 8;
  constexpr unsigned int width = 2;
  const std::vector<unsigned int> capacities = {4, logical_len};

  causallm::KVCacheManager legacy;
  legacy.allocate(2, 1, logical_len, 1, width,
                  ml::train::TensorDim::DataType::FP32);

  auto fill_cache = [](nntrainer::Tensor &cache, float base) {
    float *data = cache.getData<float>();
    for (unsigned int row = 0; row < cache.height(); ++row) {
      for (unsigned int col = 0; col < cache.width(); ++col) {
        data[row * cache.width() + col] =
          base + static_cast<float>(row * 10 + col);
      }
    }
  };

  fill_cache(legacy.getKeyCache(0), 0.0f);
  fill_cache(legacy.getValueCache(0), 100.0f);
  fill_cache(legacy.getKeyCache(1), 0.0f);
  fill_cache(legacy.getValueCache(1), 100.0f);
  legacy.setPosition(logical_len);

  const std::string legacy_path = makeTempPath("test_legacy_kv_cache.bin");
  const std::string compact_path = makeTempPath("test_compact_kv_cache.bin");
  legacy.save(legacy_path);

  causallm::KVCacheManager compact;
  compact.allocate(2, 1, logical_len, 1, width, capacities,
                   ml::train::TensorDim::DataType::FP32);
  compact.load(legacy_path, logical_len);

  EXPECT_EQ(compact.getPosition(), logical_len);
  EXPECT_FLOAT_EQ(compact.getKeyCache(0).getData<float>()[0], 40.0f);
  EXPECT_FLOAT_EQ(compact.getKeyCache(0).getData<float>()[7], 71.0f);
  EXPECT_FLOAT_EQ(compact.getValueCache(0).getData<float>()[0], 140.0f);
  EXPECT_FLOAT_EQ(compact.getKeyCache(1).getData<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(compact.getKeyCache(1).getData<float>()[15], 71.0f);

  compact.save(compact_path);
  std::ifstream saved(compact_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(saved.is_open());
  const auto expected_size =
    static_cast<std::streamoff>(2 * 2 * logical_len * width * sizeof(float));
  EXPECT_EQ(saved.tellg(), expected_size);
  saved.close();

  causallm::KVCacheManager loaded;
  loaded.allocate(2, 1, logical_len, 1, width, capacities,
                  ml::train::TensorDim::DataType::FP32);
  loaded.load(compact_path, logical_len);

  EXPECT_EQ(loaded.getPosition(), logical_len);
  EXPECT_FLOAT_EQ(loaded.getKeyCache(0).getData<float>()[0], 40.0f);
  EXPECT_FLOAT_EQ(loaded.getKeyCache(0).getData<float>()[7], 71.0f);
  EXPECT_FLOAT_EQ(loaded.getValueCache(0).getData<float>()[0], 140.0f);
  EXPECT_FLOAT_EQ(loaded.getKeyCache(1).getData<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(loaded.getKeyCache(1).getData<float>()[15], 71.0f);

  std::remove(legacy_path.c_str());
  std::remove(compact_path.c_str());
}

TEST_F(KVCacheManagerTest, compact_multibatch_uses_legacy_layout_intersection) {
  constexpr unsigned int max_seq_len = 8;
  constexpr unsigned int logical_len = 5;
  constexpr unsigned int capacity = 4;
  constexpr unsigned int batch_size = 2;

  causallm::KVCacheManager legacy;
  legacy.allocate(1, batch_size, max_seq_len, 1, 1,
                  ml::train::TensorDim::DataType::FP32);
  float *legacy_key = legacy.getKeyCache(0).getData<float>();
  float *legacy_value = legacy.getValueCache(0).getData<float>();
  for (unsigned int i = 0; i < batch_size * max_seq_len; ++i) {
    legacy_key[i] = static_cast<float>(i);
    legacy_value[i] = static_cast<float>(100 + i);
  }
  legacy.setPosition(logical_len);

  const std::string legacy_path =
    makeTempPath("test_legacy_multibatch_compact_kv_cache.bin");
  const std::string compact_path =
    makeTempPath("test_compact_multibatch_kv_cache.bin");
  legacy.save(legacy_path);

  causallm::KVCacheManager compact;
  compact.allocate(1, batch_size, max_seq_len, 1, 1, {capacity},
                   ml::train::TensorDim::DataType::FP32);
  compact.getKeyCache(0).setValue(-1.0f);
  compact.getValueCache(0).setValue(-1.0f);
  compact.load(legacy_path, logical_len);

  const float *compact_key = compact.getKeyCache(0).getData<float>();
  const float *compact_value = compact.getValueCache(0).getData<float>();
  for (unsigned int i = 0; i < capacity; ++i) {
    EXPECT_FLOAT_EQ(compact_key[i], static_cast<float>(i + 1));
    EXPECT_FLOAT_EQ(compact_value[i], static_cast<float>(101 + i));
  }
  EXPECT_FLOAT_EQ(compact_key[capacity], 9.0f);
  EXPECT_FLOAT_EQ(compact_value[capacity], 109.0f);
  for (unsigned int i = capacity + 1; i < batch_size * capacity; ++i) {
    EXPECT_FLOAT_EQ(compact_key[i], 0.0f);
    EXPECT_FLOAT_EQ(compact_value[i], 0.0f);
  }

  compact.save(compact_path);
  std::ifstream saved(compact_path, std::ios::binary);
  ASSERT_TRUE(saved.is_open());
  std::vector<float> serialized(batch_size * logical_len);
  saved.read(reinterpret_cast<char *>(serialized.data()),
             static_cast<std::streamsize>(serialized.size() * sizeof(float)));
  ASSERT_EQ(saved.gcount(),
            static_cast<std::streamsize>(serialized.size() * sizeof(float)));
  const std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 9.0f};
  EXPECT_EQ(serialized, expected);
  saved.close();

  causallm::KVCacheManager reloaded;
  reloaded.allocate(1, batch_size, max_seq_len, 1, 1, {capacity},
                    ml::train::TensorDim::DataType::FP32);
  reloaded.getKeyCache(0).setValue(-1.0f);
  reloaded.getValueCache(0).setValue(-1.0f);
  reloaded.load(compact_path, logical_len);

  const float *reloaded_key = reloaded.getKeyCache(0).getData<float>();
  const float *reloaded_value = reloaded.getValueCache(0).getData<float>();
  for (unsigned int i = 0; i < batch_size * capacity; ++i) {
    EXPECT_FLOAT_EQ(reloaded_key[i], compact_key[i]);
    EXPECT_FLOAT_EQ(reloaded_value[i], compact_value[i]);
  }

  std::remove(legacy_path.c_str());
  std::remove(compact_path.c_str());
}

TEST_F(KVCacheManagerTest, compact_save_rejects_unavailable_position) {
  causallm::KVCacheManager compact;
  compact.allocate(1, 1, 8, 1, 1, {4}, ml::train::TensorDim::DataType::FP32);
  const std::string path = makeTempPath("test_compact_save_position.bin");

  compact.setPosition(3);
  EXPECT_NO_THROW(compact.save(path, 2));
  EXPECT_THROW(compact.save(path, 4), std::invalid_argument);

  compact.setPosition(6);
  EXPECT_THROW(compact.save(path, 5), std::invalid_argument);
  EXPECT_NO_THROW(compact.save(path, 6));

  std::remove(path.c_str());
}

TEST_F(KVCacheManagerTest, save_load_not_allocated) {
  causallm::KVCacheManager empty;
  const std::string path = makeTempPath("test_unallocated_kv_cache.bin");
  EXPECT_THROW(empty.save(path), std::runtime_error);
  EXPECT_THROW(empty.load(path, 1), std::runtime_error);
}

TEST_F(KVCacheManagerTest, write_view_overflow) {
  manager.setPosition(MAX_SEQ_LEN - 1);
  // Writing 1 should be ok
  EXPECT_NO_THROW(manager.getKeyCacheWriteView(0, 0, 1));
  // Writing 2 should overflow
  EXPECT_THROW(manager.getKeyCacheWriteView(0, 0, 2), std::out_of_range);
}

TEST_F(KVCacheManagerTest, compact_cache_view_rejects_logical_position) {
  causallm::KVCacheManager compact;
  compact.allocate(2, 1, 8, 1, 2, {4, 8}, ml::train::TensorDim::DataType::FP32);
  compact.setPosition(5);

  EXPECT_THROW(compact.getKeyCacheWriteView(0, 0, 1), std::out_of_range);
  EXPECT_THROW(compact.getValueCacheWriteView(0, 0, 1), std::out_of_range);
  EXPECT_THROW(compact.getKeyCacheReadView(0, 0, 5), std::out_of_range);
  EXPECT_THROW(compact.getValueCacheReadView(0, 0, 5), std::out_of_range);
}

TEST_F(KVCacheManagerTest, typical_inference_flow) {
  // Simulate: prefill 10 tokens, then generate 5 tokens one by one

  // Prefill: write 10 tokens
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      auto k_write = manager.getKeyCacheWriteView(l, b, 10);
      auto v_write = manager.getValueCacheWriteView(l, b, 10);
      // Fill with identifiable data
      float *kd = k_write.getData<float>();
      for (unsigned int i = 0; i < 10 * KV_WIDTH; ++i) {
        kd[i] = static_cast<float>(l * 10000 + b * 1000 + i);
      }
    }
  }
  manager.advance(10);
  EXPECT_EQ(manager.getPosition(), 10u);

  // Generate: 5 tokens one by one
  for (unsigned int step = 0; step < 5; ++step) {
    unsigned int current_pos = manager.getPosition();
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        // Write new K/V
        auto k_write = manager.getKeyCacheWriteView(l, b, 1);
        float *kd = k_write.getData<float>();
        for (unsigned int i = 0; i < KV_WIDTH; ++i) {
          kd[i] = static_cast<float>(current_pos * 100 + l * 10 + i);
        }

        // Read all cached K for attention
        auto k_read = manager.getKeyCacheReadView(l, b, current_pos + 1);
        EXPECT_EQ(k_read.height(), current_pos + 1);
      }
    }
    manager.advance(1);
  }

  EXPECT_EQ(manager.getPosition(), 15u);

  // Verify first token of prefill is still intact (layer 0, batch 0)
  auto k_full = manager.getKeyCacheReadView(0, 0, 15);
  float *kd = k_full.getData<float>();
  EXPECT_FLOAT_EQ(kd[0], 0.0f); // l=0, b=0, i=0
  EXPECT_FLOAT_EQ(kd[1], 1.0f); // l=0, b=0, i=1
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
