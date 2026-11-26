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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#ifdef __linux__
#include <unistd.h>
#endif

#include <kv_cache_manager.h>
#include <kv_ring.h>
#include <mem_allocator.h>
#include <tensor.h>
#include <tensor_dim.h>

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
  m.setActiveLayers({true});
  EXPECT_THROW(m.allocate(4, 1, 128, 4, 8), std::invalid_argument);
}

TEST(KVCacheManagerHybrid, inactive_layers_allocate_no_storage) {
  unsigned allocations = 0, releases = 0;
  {
    causallm::KVCacheManager m;
    m.setActiveLayers({false, true, false, true});
    m.setSharedAllocator(
      [&](size_t bytes) {
        ++allocations;
        return std::malloc(bytes);
      },
      [&](void *p) {
        ++releases;
        std::free(p);
      });
    m.allocate(4, 1, 64, 2, 8, ml::train::TensorDim::DataType::FP32);
    ASSERT_EQ(allocations, 4u);
    EXPECT_EQ(m.getKeyCache(0).height(), 64u);
    EXPECT_FALSE(m.getKeyCache(0).isAllocated());
    EXPECT_FALSE(m.getValueCache(2).isAllocated());
    EXPECT_TRUE(m.getKeyCache(1).isAllocated());
    EXPECT_TRUE(m.getValueCache(3).isAllocated());
    EXPECT_THROW(m.getKeyCacheWriteView(0, 0, 1), std::logic_error);
    EXPECT_THROW(m.getValueCacheReadView(2, 0, 1), std::logic_error);
    EXPECT_THROW(m.setActiveLayers({}), std::logic_error);

    causallm::KVCacheManager moved;
    moved = std::move(m);
    EXPECT_FALSE(moved.isLayerActive(0));
    EXPECT_TRUE(moved.isLayerActive(3));
  }
  EXPECT_EQ(releases, allocations);
}

TEST(KVCacheManagerHybrid, all_inactive_layers_keep_only_metadata) {
  unsigned allocations = 0, releases = 0;
  {
    causallm::KVCacheManager m;
    m.setActiveLayers({true, false});
    m.setActiveLayers({false, false});
    m.setSharedAllocator(
      [&](size_t bytes) {
        ++allocations;
        return std::malloc(bytes);
      },
      [&](void *p) {
        ++releases;
        std::free(p);
      });
    for (unsigned rows : {16u, 32u}) {
      ASSERT_NO_THROW(
        m.allocate(2, 1, rows, 2, 8, ml::train::TensorDim::DataType::FP32));
      EXPECT_TRUE(m.isAllocated());
      EXPECT_EQ(m.getNumLayers(), 2u);
      for (unsigned i = 0; i < 2; ++i) {
        EXPECT_FALSE(m.isLayerActive(i));
        EXPECT_EQ(m.getKeyCache(i).height(), rows);
        EXPECT_EQ(m.getValueCache(i).width(), 16u);
        EXPECT_FALSE(m.getKeyCache(i).isAllocated());
        EXPECT_FALSE(m.getValueCache(i).isAllocated());
      }
      m.advance(3);
      m.reset();
      EXPECT_EQ(m.getPosition(), 0u);
      EXPECT_EQ(allocations, 0u);
    }
    // A new prompt can replace the manager and select a different layout.
    m = causallm::KVCacheManager{};
    m.setActiveLayers({});
    m.allocate(2, 1, 16, 2, 8, ml::train::TensorDim::DataType::FP32);
    EXPECT_TRUE(m.isLayerActive(0));
    EXPECT_TRUE(m.getKeyCache(0).isAllocated());
  }
  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(releases, 0u);
}

TEST(KVCacheManagerHybrid, compact_skips_inactive_windows) {
  causallm::KVCacheManager m;
  m.setActiveLayers({false, true});
#ifdef ENABLE_FP16
  const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  m.allocate(2, 1, 8, std::vector<unsigned int>{2, 2}, dtype,
             ml::train::TensorDim::Format::NCHW, {4, 4}, {2, 2});
  for (auto *cache : {&m.getKeyCache(1), &m.getValueCache(1)}) {
    auto *bits = cache->getData<uint16_t>();
    for (unsigned i = 0; i < 8; ++i)
      bits[i] = static_cast<uint16_t>(100 + i);
  }
  EXPECT_TRUE(m.compact(4));
  EXPECT_EQ(m.getBase(0), 0u);
  EXPECT_FALSE(m.getKeyCache(0).isAllocated());
  EXPECT_FALSE(m.getValueCache(0).isAllocated());
  EXPECT_EQ(m.getBase(1), 2u);
  for (auto *cache : {&m.getKeyCache(1), &m.getValueCache(1)})
    for (unsigned i = 0; i < 4; ++i)
      EXPECT_EQ(cache->getData<uint16_t>()[i], 104 + i);
  EXPECT_FALSE(m.compact(4));
}

TEST(KVCacheManagerHybrid, compact_folds_the_dropped_rows_amax) {
  causallm::KVCacheManager m;
#ifdef ENABLE_FP16
  const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  // one windowed layer: 4 rows of width 2, a window of 2
  m.allocate(1, 1, 8, std::vector<unsigned int>{2}, dtype,
             ml::train::TensorDim::Format::NCHW, {4}, {2});
  // fp16 bits: 1.0 = 0x3C00, 2.0 = 0x4000, -3.0 = 0xC200, 0.5 = 0x3800,
  // -2.0 = 0xC000; rows 0..3 as (col0, col1)
  const uint16_t kb[8] = {0x3C00, 0xC200, 0x4000, 0x3800,
                          0x3800, 0x3800, 0x3800, 0x3800};
  const uint16_t vb[8] = {0x3800, 0x3C00, 0xC000, 0x3C00,
                          0x3800, 0x3800, 0x3800, 0x3800};
  std::copy(kb, kb + 8, m.getKeyCache(0).getData<uint16_t>());
  std::copy(vb, vb + 8, m.getValueCache(0).getData<uint16_t>());
  EXPECT_TRUE(m.droppedAmaxK(0).empty());
  EXPECT_TRUE(m.droppedAmaxV(0).empty());
  EXPECT_TRUE(m.droppedAmaxK(7).empty()); // no such layer: empty, no throw
  // rows 0 and 1 go: |max| per column of what went
  EXPECT_TRUE(m.compact(4));
  ASSERT_EQ(m.droppedAmaxK(0).size(), 2u);
  EXPECT_FLOAT_EQ(m.droppedAmaxK(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxK(0)[1], 3.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxV(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxV(0)[1], 1.0f);
  // the next compaction drops smaller rows (old 2 and 3): the fold only grows
  EXPECT_TRUE(m.compact(6));
  EXPECT_EQ(m.getBase(0), 4u);
  EXPECT_FLOAT_EQ(m.droppedAmaxK(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxK(0)[1], 3.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxV(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(m.droppedAmaxV(0)[1], 1.0f);
  // a new sequence starts from nothing
  m.reset();
  EXPECT_TRUE(m.droppedAmaxK(0).empty());
  EXPECT_TRUE(m.droppedAmaxV(0).empty());
}

TEST(KVCacheManagerHybrid, invalid_later_window_leaves_every_layer_unchanged) {
  causallm::KVCacheManager m;
#ifdef ENABLE_FP16
  const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  m.allocate(2, 1, 8, std::vector<unsigned int>{2, 2}, dtype,
             ml::train::TensorDim::Format::NCHW, {4, 2}, {1, 1});
  for (unsigned l = 0; l < 2; ++l)
    for (auto *t : {&m.getKeyCache(l), &m.getValueCache(l)})
      std::fill_n(t->getData<uint16_t>(), m.getRows(l) * 2, 0x3c00u);
  // Layer 0 can compact at 3, but layer 1 cannot hold that position.
  for (unsigned pos : {3u, 9u}) {
    EXPECT_THROW(m.compact(pos), std::out_of_range);
    for (unsigned l = 0; l < 2; ++l) {
      EXPECT_EQ(m.getBase(l), 0u);
      EXPECT_TRUE(m.droppedAmaxK(l).empty());
      EXPECT_TRUE(m.droppedAmaxV(l).empty());
      for (auto *t : {&m.getKeyCache(l), &m.getValueCache(l)})
        for (unsigned i = 0; i < m.getRows(l) * 2; ++i)
          EXPECT_EQ(t->getData<uint16_t>()[i], 0x3c00u);
    }
  }
}

TEST(KVCacheManagerHybrid, rejected_snapshot_preserves_existing_file) {
  causallm::KVCacheManager m;
#ifdef ENABLE_FP16
  const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  m.allocate(2, 1, 8, std::vector<unsigned int>{2, 2}, dtype,
             ml::train::TensorDim::Format::NCHW, {4, 2}, {1, 1});
  const auto path =
    (std::filesystem::temp_directory_path() / "nntr_kv_rejected_snapshot.bin")
      .string();
  const std::string original = "keep the previous snapshot";
  const auto check_refused = [&](unsigned pos) {
    {
      std::ofstream file(path, std::ios::binary);
      file << original;
    }
    EXPECT_THROW(m.save(path, pos), std::out_of_range);
    std::ifstream file(path, std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_EQ(content, original);
  };
  check_refused(3);
  check_refused(9);
  ASSERT_TRUE(m.compact(2));
  check_refused(0); // The earlier rows have already been evicted.
  EXPECT_EQ(m.getBase(0), 1u);
  EXPECT_EQ(m.getBase(1), 1u);
  std::filesystem::remove(path);
}

TEST(KVCacheManagerHybrid, valid_window_snapshot_round_trip) {
  causallm::KVCacheManager m, restored;
#ifdef ENABLE_FP16
  const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  for (auto *cache : {&m, &restored})
    cache->allocate(2, 1, 8, std::vector<unsigned int>{2, 2}, dtype,
                    ml::train::TensorDim::Format::NCHW, {4, 2}, {1, 1});
  for (unsigned l = 0; l < 2; ++l)
    for (auto *t : {&m.getKeyCache(l), &m.getValueCache(l)})
      for (unsigned i = 0; i < m.getRows(l) * 2; ++i)
        t->getData<uint16_t>()[i] = 0x3c00u + l * 16u + i;
  const auto path =
    (std::filesystem::temp_directory_path() / "nntr_kv_window_round_trip.bin")
      .string();
  ASSERT_NO_THROW(m.validateSaveRange(2));
  ASSERT_TRUE(m.compact(2));
  ASSERT_NO_THROW(m.save(path, 2));
  ASSERT_EQ(restored.load(path, 2), 2u);
  for (unsigned l = 0; l < 2; ++l) {
    EXPECT_EQ(restored.getBase(l), 1u);
    for (unsigned i = 0; i < 2; ++i) {
      EXPECT_EQ(restored.getKeyCache(l).getData<uint16_t>()[i],
                m.getKeyCache(l).getData<uint16_t>()[i]);
      EXPECT_EQ(restored.getValueCache(l).getData<uint16_t>()[i],
                m.getValueCache(l).getData<uint16_t>()[i]);
    }
  }
  std::filesystem::remove(path);
}

TEST(KVCacheManagerHybrid, dense_files_remain_compatible_in_both_directions) {
  const std::string path = "/tmp/test_kv_cache_hybrid.bin";
  using Dtype = ml::train::TensorDim::DataType;
  std::vector<Dtype> dtypes{Dtype::FP32};
#ifdef ENABLE_FP16
  dtypes.push_back(Dtype::FP16);
#endif
  for (auto dtype : dtypes) {
    causallm::KVCacheManager dense, sparse, restored;
    sparse.setActiveLayers({false, true, false, true});
    for (auto *m : {&dense, &sparse, &restored})
      m->allocate(4, 1, 16, 2, 8, dtype);
    for (unsigned i = 0; i < 4; ++i) {
      dense.getKeyCache(i).setValue(float(i + 1));
      dense.getValueCache(i).setValue(float(i + 11));
    }
    const size_t bytes =
      5u * 16u * dense.getKeyCache(0).getDim().getDataTypeSize();
    dense.save(path, 5);
    EXPECT_EQ(sparse.load(path, 5), 5u);
    EXPECT_FALSE(sparse.getKeyCache(0).isAllocated());
    for (unsigned i : {1u, 3u}) {
      EXPECT_EQ(std::memcmp(sparse.getKeyCache(i).getData<char>(),
                            dense.getKeyCache(i).getData<char>(), bytes),
                0);
      EXPECT_EQ(std::memcmp(sparse.getValueCache(i).getData<char>(),
                            dense.getValueCache(i).getData<char>(), bytes),
                0);
    }
    sparse.save(path, 5);
    EXPECT_EQ(restored.load(path, 5), 5u);
    const std::vector<char> zeros(bytes, 0);
    for (unsigned i = 0; i < 4; ++i) {
      EXPECT_EQ(
        std::memcmp(restored.getKeyCache(i).getData<char>(),
                    i % 2 ? dense.getKeyCache(i).getData<char>() : zeros.data(),
                    bytes),
        0);
      EXPECT_EQ(std::memcmp(restored.getValueCache(i).getData<char>(),
                            i % 2 ? dense.getValueCache(i).getData<char>()
                                  : zeros.data(),
                            bytes),
                0);
    }
  }
  std::remove(path.c_str());
}

TEST(KVCacheManagerHybrid, aliases_cannot_reference_inactive_layers) {
  causallm::KVCacheManager m;
  m.setActiveLayers({false, true});
  m.setLayerKVSources({-1, 0});
  EXPECT_THROW(m.allocate(2, 1, 16, 2, 8, ml::train::TensorDim::DataType::FP32),
               std::invalid_argument);
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
  std::string path = "/tmp/test_kv_cache.bin";
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

TEST_F(KVCacheManagerTest, save_load_not_allocated) {
  causallm::KVCacheManager empty;
  EXPECT_THROW(empty.save("/tmp/test.bin"), std::runtime_error);
  EXPECT_THROW(empty.load("/tmp/test.bin", 1), std::runtime_error);
}

TEST_F(KVCacheManagerTest, write_view_overflow) {
  manager.setPosition(MAX_SEQ_LEN - 1);
  // Writing 1 should be ok
  EXPECT_NO_THROW(manager.getKeyCacheWriteView(0, 0, 1));
  // Writing 2 should overflow
  EXPECT_THROW(manager.getKeyCacheWriteView(0, 0, 2), std::out_of_range);
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

/**
 * @brief [kv-share] A layer declared as sharing an earlier layer's KV must end
 *        up ALIASING that layer's storage -- same address, full dimensions,
 *        never empty. Getting this wrong the other way (an empty or null slab)
 *        does not crash: getKeyCache(i).getData() returns nullptr, the graph
 *        binds it, and attention silently reads zeros. So the post-conditions
 *        pinned here are "non-null" and "identical to the source", not merely
 *        "allocate() returned".
 */
TEST(KVCacheManagerShare, shared_layers_alias_the_source_plane) {
  constexpr unsigned int LAYERS = 6;
  constexpr unsigned int BATCH = 2;
  constexpr unsigned int SEQ = 32;
  constexpr unsigned int HEADS_KV = 2;
  constexpr unsigned int HEAD_D = 4;

  causallm::KVCacheManager m;
  // layers 0..2 own their cache; 3->1, 4->0, 5->1 (sources strictly earlier,
  // and one source feeding two sharers).
  m.setLayerKVSources({-1, -1, -1, 1, 0, 1});
  m.allocate(LAYERS, BATCH, SEQ, HEADS_KV, HEAD_D,
             ml::train::TensorDim::DataType::FP32);

  const std::vector<int> expect = {-1, -1, -1, 1, 0, 1};
  for (unsigned int l = 0; l < LAYERS; ++l) {
    EXPECT_EQ(m.getLayerKVSource(l), expect[l]) << "layer=" << l;

    // No layer -- shared or not -- may present an empty/null cache.
    ASSERT_NE(m.getKeyCache(l).getData<float>(), nullptr) << "layer=" << l;
    ASSERT_NE(m.getValueCache(l).getData<float>(), nullptr) << "layer=" << l;
    EXPECT_EQ(m.getKeyCache(l).getDim(), m.getKeyCache(0).getDim());
    EXPECT_EQ(m.getValueCache(l).getDim(), m.getValueCache(0).getDim());
    EXPECT_EQ(m.getKeyCache(l).bytes(), static_cast<size_t>(BATCH) * SEQ *
                                          HEADS_KV * HEAD_D * sizeof(float));
  }

  for (unsigned int l = 0; l < LAYERS; ++l) {
    const int src = m.getLayerKVSource(l);
    if (src < 0)
      continue;
    EXPECT_EQ(m.getKeyCache(l).getData<float>(),
              m.getKeyCache(static_cast<unsigned int>(src)).getData<float>())
      << "layer=" << l;
    EXPECT_EQ(m.getValueCache(l).getData<float>(),
              m.getValueCache(static_cast<unsigned int>(src)).getData<float>())
      << "layer=" << l;
  }

  // Distinct owners stay distinct -- aliasing must not collapse everything.
  EXPECT_NE(m.getKeyCache(0).getData<float>(),
            m.getKeyCache(1).getData<float>());
  EXPECT_NE(m.getKeyCache(0).getData<float>(),
            m.getValueCache(0).getData<float>());

  // The alias is the same memory in both directions: this is what makes the
  // shared layer's (redundant, byte-identical) write harmless.
  m.getKeyCache(1).getData<float>()[7] = 42.0f;
  EXPECT_FLOAT_EQ(m.getKeyCache(3).getData<float>()[7], 42.0f);
  m.getKeyCache(5).getData<float>()[7] = -1.5f;
  EXPECT_FLOAT_EQ(m.getKeyCache(1).getData<float>()[7], -1.5f);
  EXPECT_FLOAT_EQ(m.getKeyCache(3).getData<float>()[7], -1.5f);
}

/**
 * @brief [kv-share] An unusable source map must be rejected at allocate(),
 *        before any memory is requested -- a forward/self reference would
 *        alias a not-yet-allocated (empty) plane, and a geometry mismatch
 *        would reinterpret the source's rows. Both fail silently at runtime.
 */
TEST(KVCacheManagerShare, rejects_unusable_source_maps) {
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, 2, -1, -1}); // forward reference
    EXPECT_THROW(m.allocate(4, 1, 16, 2, 4), std::invalid_argument);
  }
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, 1, -1, -1}); // self reference
    EXPECT_THROW(m.allocate(4, 1, 16, 2, 4), std::invalid_argument);
  }
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, -1}); // size != num_layers
    EXPECT_THROW(m.allocate(4, 1, 16, 2, 4), std::invalid_argument);
  }
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, 0}); // width(1) != width(0)
    EXPECT_THROW(m.allocate(2, 1, 16, std::vector<unsigned int>{8u, 16u},
                            ml::train::TensorDim::DataType::FP32),
                 std::invalid_argument);
  }
}

/**
 * @brief [kv-share] The default (no setLayerKVSources call) must stay exactly
 *        what it always was: every layer owns a private plane.
 */
TEST(KVCacheManagerShare, no_declaration_means_no_aliasing) {
  causallm::KVCacheManager m;
  m.allocate(3, 1, 16, 2, 4, ml::train::TensorDim::DataType::FP32);
  for (unsigned int l = 0; l < 3; ++l) {
    EXPECT_EQ(m.getLayerKVSource(l), -1);
    EXPECT_FALSE(m.isLayerKVAliased(l));
    ASSERT_NE(m.getKeyCache(l).getData<float>(), nullptr);
  }
  EXPECT_NE(m.getKeyCache(0).getData<float>(),
            m.getKeyCache(1).getData<float>());
  EXPECT_NE(m.getKeyCache(1).getData<float>(),
            m.getKeyCache(2).getData<float>());
}

/**
 * Shared-memory allocator (the accelerator-visible KV cache): every slab is
 * one block from the allocator, the cache tensors and their views resolve
 * into those blocks with no copy, and each block goes back exactly once
 * when the manager is destroyed.
 */
TEST(KVCacheManagerSharedMemory, slabs_come_from_allocator_and_return) {
  std::vector<void *> blocks;
  std::vector<size_t> sizes;
  std::vector<void *> released;
  {
    causallm::KVCacheManager m;
    m.setSharedAllocator(
      [&](size_t bytes) {
        void *p = std::malloc(bytes);
        blocks.push_back(p);
        sizes.push_back(bytes);
        return p;
      },
      [&](void *p) {
        released.push_back(p);
        std::free(p);
      });
    m.allocate(3, 1, 64, 2, 8, ml::train::TensorDim::DataType::FP32);

    EXPECT_TRUE(m.usesSharedMemory());
    ASSERT_EQ(blocks.size(), 6u); // key and value per layer
    for (size_t b : sizes) {
      EXPECT_EQ(b, 64u * 16u * sizeof(float));
    }
    for (unsigned int l = 0; l < 3; ++l) {
      EXPECT_EQ(m.getKeyCache(l).getData<float>(), blocks[2 * l]);
      EXPECT_EQ(m.getValueCache(l).getData<float>(), blocks[2 * l + 1]);
    }

    // A write view at position 5 lands at row 5 of the value block.
    m.setPosition(5);
    auto view = m.getValueCacheWriteView(1, 0, 1);
    float *vd = view.getData<float>();
    EXPECT_EQ(vd, static_cast<float *>(blocks[3]) + 5 * 16);
    vd[3] = 42.0f;
    EXPECT_FLOAT_EQ(static_cast<float *>(blocks[3])[5 * 16 + 3], 42.0f);
    EXPECT_TRUE(released.empty());
  }
  ASSERT_EQ(released.size(), 6u);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_NE(std::find(blocks.begin(), blocks.end(), released[i]),
              blocks.end());
  }
}

TEST(KVCacheManagerSharedMemory, nullptr_from_allocator_means_heap) {
  int releases = 0;
  {
    causallm::KVCacheManager m;
    m.setSharedAllocator([](size_t) -> void * { return nullptr; },
                         [&](void *) { ++releases; });
    m.allocate(2, 1, 16, 2, 8, ml::train::TensorDim::DataType::FP32);
    EXPECT_TRUE(m.isAllocated());
    EXPECT_FALSE(m.usesSharedMemory());
    EXPECT_NE(m.getKeyCache(0).getData<float>(), nullptr);
  }
  EXPECT_EQ(releases, 0);
}

TEST(KVCacheManagerSharedMemory, allocator_must_be_set_before_allocate) {
  causallm::KVCacheManager m;
  m.allocate(1, 1, 16, 2, 8, ml::train::TensorDim::DataType::FP32);
  EXPECT_THROW(m.setSharedAllocator([](size_t) -> void * { return nullptr; },
                                    [](void *) {}),
               std::logic_error);
  causallm::KVCacheManager n;
  EXPECT_THROW(
    n.setSharedAllocator([](size_t) -> void * { return nullptr; }, nullptr),
    std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(KVCacheManagerShared, MovePreservesGeometryAndReleasesEachAllocationOnce) {
  unsigned int allocated = 0, released = 0;
  auto alloc = [&](size_t bytes) -> void * {
    ++allocated;
    return std::malloc(bytes);
  };
  auto free = [&](void *p) {
    ++released;
    std::free(p);
  };
  {
    causallm::KVCacheManager source, destination;
    source.setSharedAllocator(alloc, free);
    source.setLayerCaps({4, 4});
    source.setLayerKVSources({-1, 0});
    source.allocate(2, 1, 8, std::vector<unsigned int>{16, 16},
                    ml::train::TensorDim::DataType::FP32);
    source.setPosition(3);
    auto *key = source.getKeyCache(0).getData<float>();
    key[0] = 7.0f;
    destination.setSharedAllocator(alloc, free);
    destination.allocate(1, 1, 8, 1, 16, ml::train::TensorDim::DataType::FP32);
    EXPECT_EQ(allocated, 4u);
    destination = std::move(source);
    EXPECT_EQ(released, 2u);
    EXPECT_EQ(destination.getPosition(), 3u);
    EXPECT_EQ(destination.getLayerCap(1), 4u);
    EXPECT_EQ(destination.getLayerKVSource(1), 0);
    EXPECT_EQ(destination.getKeyCache(1).getData<float>(), key);
    EXPECT_FLOAT_EQ(destination.getKeyCache(0).getData<float>()[0], 7.0f);
  }
  EXPECT_EQ(released, allocated);
}

TEST(KVCacheManagerCoreShared, aliases_pin_slabs_and_move_keeps_ownership) {
  auto allocator = std::make_shared<nntrainer::MemAllocator>();
  size_t allocated = 0, freed = 0;
  std::shared_ptr<nntrainer::MemoryData> held;
  {
    causallm::KVCacheManager source;
    source.setCoreSharedStorage(true);
    source.setSharedAllocator(
      [&](size_t bytes) {
        void *p = nullptr;
        allocator->alloc(&p, bytes, 64);
        ++allocated;
        return p;
      },
      [&](void *p) {
        ++freed;
        allocator->free(p);
      });
    source.setLayerKVSources({-1, 0});
    source.allocate(2, 1, 8, std::vector<unsigned int>{4, 4},
                    ml::train::TensorDim::DataType::FP32);
    ASSERT_TRUE(source.usesCoreSharedStorage());
    ASSERT_TRUE(source.usesSharedMemory());
    ASSERT_EQ(allocated, 2u);
    EXPECT_EQ(source.getKeyCache(0).getData<float>(),
              source.getKeyCache(1).getData<float>());
    source.getKeyCache(0).getData<float>()[0] = 23.0f;
    held = source.getKeyCache(1).getMemoryData();
    source.setPosition(3);
    causallm::KVCacheManager destination;
    destination = std::move(source);
    EXPECT_TRUE(destination.usesCoreSharedStorage());
    EXPECT_EQ(destination.getPosition(), 3u);
    EXPECT_FLOAT_EQ(destination.getKeyCache(0).getData<float>()[0], 23.0f);
    destination.advance(2);
    EXPECT_EQ(destination.getPosition(), 5u);
    destination.reset();
    EXPECT_EQ(destination.getPosition(), 0u);
    EXPECT_EQ(freed, 0u);
  }
  EXPECT_EQ(freed, 1u);
  EXPECT_FLOAT_EQ(held->getAddr<float>()[0], 23.0f);
  held.reset();
  EXPECT_EQ(freed, 2u);
}

TEST(KVCacheManagerCoreShared, null_shared_allocation_is_not_a_heap_fallback) {
  causallm::KVCacheManager manager;
  manager.setCoreSharedStorage(true);
  manager.setSharedAllocator([](size_t) -> void * { return nullptr; },
                             [](void *) {});
  EXPECT_THROW(
    manager.allocate(1, 1, 8, 1, 4, ml::train::TensorDim::DataType::FP32),
    std::bad_alloc);
}

TEST(KVCacheManagerCoreShared, preserves_window_compaction_and_snapshot_bytes) {
  auto allocator = std::make_shared<nntrainer::MemAllocator>();
  causallm::KVCacheManager manager;
  manager.setCoreSharedStorage(true);
  manager.setSharedAllocator(
    [allocator](size_t bytes) {
      void *p = nullptr;
      allocator->alloc(&p, bytes, 64);
      return p;
    },
    [allocator](void *p) { allocator->free(p); });
  manager.allocate(1, 1, 8, std::vector<unsigned int>{4},
                   ml::train::TensorDim::DataType::UINT16,
                   ml::train::TensorDim::Format::NCHW, {4}, {2});
  auto *key = manager.getKeyCache(0).getData<uint16_t>();
  auto *value = manager.getValueCache(0).getData<uint16_t>();
  for (size_t i = 0; i < 16; ++i) {
    key[i] = static_cast<uint16_t>(0x3c00u + i);
    value[i] = static_cast<uint16_t>(0x4000u + i);
  }
  manager.setPosition(4);
  ASSERT_TRUE(manager.compact(4));
  EXPECT_EQ(manager.getBase(0), 2u);
  EXPECT_EQ(key, manager.getKeyCache(0).getData<uint16_t>());
  EXPECT_EQ(key[0], 0x3c08u);
  EXPECT_EQ(value[0], 0x4008u);
  EXPECT_EQ(manager.rowsToPersist(0, 4), 2u);
  EXPECT_FALSE(manager.droppedAmaxK(0).empty());
  EXPECT_THROW(manager.setCoreSharedStorage(false), std::logic_error);
}

#if defined(__linux__) && defined(ENABLE_FP16)
TEST(KVCacheManagerFlash, persists_ring_alias_and_existing_snapshot_format) {
  using DT = ml::train::TensorDim::DataType;
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  for (bool alias : {false, true}) {
    for (unsigned lookahead : {0u, 1u, 3u}) {
      SCOPED_TRACE(alias);
      SCOPED_TRACE(lookahead);
      causallm::KVCacheManager dram, flash;
      flash.setFlashStorage(directory.path, lookahead);
      constexpr unsigned width = 1024, ring_rows = 65, context = 128;
      for (auto *cache : {&dram, &flash}) {
        cache->setActiveLayers({true, false, true, true});
        cache->setLayerCaps({ring_rows, 0, ring_rows, 0});
        cache->setLayerKVSources({-1, -1, alias ? 0 : -1, -1});
        cache->allocate(4, 1, context, 1, width, DT::FP16);
      }

      // The graph caches raw addresses for its persistent KV bindings.
      std::vector<const uint16_t *> key_addresses(4), value_addresses(4);
      for (unsigned layer : {0u, 2u, 3u}) {
        key_addresses[layer] = flash.getKeyCache(layer).getData<uint16_t>();
        value_addresses[layer] = flash.getValueCache(layer).getData<uint16_t>();
      }
      const auto compareLayer = [&](unsigned layer) {
        const auto &k = dram.getKeyCache(layer);
        const auto &v = dram.getValueCache(layer);
        EXPECT_EQ(0, std::memcmp(k.getData<uint16_t>(),
                                 flash.getKeyCache(layer).getData<uint16_t>(),
                                 k.bytes()));
        EXPECT_EQ(0, std::memcmp(v.getData<uint16_t>(),
                                 flash.getValueCache(layer).getData<uint16_t>(),
                                 v.bytes()));
      };

      unsigned from = 0;
      for (unsigned to : {64u, 70u, 71u}) {
        flash.beginFlashStep(from);
        for (unsigned layer : {0u, 2u, 3u}) {
          flash.beginFlashLayer(layer);
          compareLayer(layer);
          if (layer != 2u || !alias) {
            for (auto *cache : {&dram, &flash}) {
              for (unsigned row = from; row < to; ++row) {
                const auto offset = row % cache->getLayerCap(layer) * width;
                for (unsigned col = 0; col < width; ++col) {
                  auto value = static_cast<uint16_t>(layer * 500 + row + col);
                  cache->getKeyCache(layer).getData<uint16_t>()[offset + col] =
                    value;
                  cache->getValueCache(layer)
                    .getData<uint16_t>()[offset + col] =
                    0xff00u + (value & 0xffu);
                }
              }
            }
          }
          flash.finishFlashLayer(layer, from, to);
          EXPECT_EQ(key_addresses[layer],
                    flash.getKeyCache(layer).getData<uint16_t>());
          EXPECT_EQ(value_addresses[layer],
                    flash.getValueCache(layer).getData<uint16_t>());
          EXPECT_EQ(0u, flash.getKeyCache(layer).getData<uint16_t>()[0]);
        }
        flash.finishFlashStep();
        dram.setPosition(to);
        flash.setPosition(to);
        from = to;
      }

      const auto expected = directory.path + "/dram.bin";
      const auto actual = directory.path + "/flash.bin";
      const auto bytes = [](const std::string &file) {
        std::ifstream in(file, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
      };
      dram.save(expected);
      const auto baseline = bytes(expected);
      ASSERT_FALSE(baseline.empty());
      const auto compareSnapshot = [&] {
        const auto result = bytes(actual);
        ASSERT_EQ(baseline.size(), result.size());
        const auto mismatch =
          std::mismatch(baseline.begin(), baseline.end(), result.begin());
        EXPECT_EQ(baseline.size(),
                  static_cast<size_t>(mismatch.first - baseline.begin()));
      };
      flash.save(actual);
      compareSnapshot();

      flash.reset();
      EXPECT_EQ(from, flash.load(expected, 0));
      flash.save(actual);
      compareSnapshot();

      flash.beginFlashStep(0);
      flash.beginFlashLayer(0);
      EXPECT_EQ(0u, flash.getValueCache(0).getData<uint16_t>()[0]);
      flash.finishFlashStep();
    }
  }
}
namespace {
/**
 * @brief Sets an environment variable for one test and restores the previous
 *        state on scope exit, so a failed assertion cannot leak it into the
 *        tests that run next in the same process.
 */
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    const char *previous = std::getenv(name);
    had_previous_ = previous != nullptr;
    if (had_previous_)
      previous_ = previous;
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_previous_)
      setenv(name_, previous_.c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  bool had_previous_ = false;
  std::string previous_;
};

/** @brief Whole raw bytes of a file, for snapshot byte comparisons. */
std::string readFileBytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/** @brief Bit-compare one layer's whole K/V planes. */
void expectSameLayerBytes(causallm::KVCacheManager &dram,
                          causallm::KVCacheManager &flash, unsigned layer) {
  const auto &k = dram.getKeyCache(layer);
  const auto &v = dram.getValueCache(layer);
  EXPECT_EQ(0, std::memcmp(k.getData<uint16_t>(),
                           flash.getKeyCache(layer).getData<uint16_t>(),
                           k.bytes()));
  EXPECT_EQ(0, std::memcmp(v.getData<uint16_t>(),
                           flash.getValueCache(layer).getData<uint16_t>(),
                           v.bytes()));
}

/** @brief Bit-compare two snapshot files, with the first differing offset. */
void expectSameFileBytes(const std::string &baseline,
                         const std::string &actual) {
  ASSERT_EQ(baseline.size(), actual.size());
  const auto mismatch =
    std::mismatch(baseline.begin(), baseline.end(), actual.begin());
  EXPECT_EQ(baseline.size(),
            static_cast<size_t>(mismatch.first - baseline.begin()))
    << "snapshot bytes diverge at offset "
    << (mismatch.first - baseline.begin());
}

/**
 * @brief [kv-fsu mixed] The mixed fixture: a DRAM reference and a flash
 *        manager with the same geometry -- layers 0/2 windowed (rows=8,
 *        keep=4, host DRAM tensors), layer 1 full (streams through the FSU),
 *        layer 3 an alias of layer 1.
 */
void allocateMixedPair(causallm::KVCacheManager &dram,
                       causallm::KVCacheManager &flash,
                       const std::string &flash_dir) {
  flash.setFlashStorage(flash_dir, 1);
  constexpr unsigned width = 256, max_seq = 20;
  const std::vector<unsigned int> widths(4, width);
  const std::vector<unsigned int> rows{8u, 0u, 8u, 0u};
  const std::vector<unsigned int> keep{4u, 0u, 4u, 0u};
  for (auto *cache : {&dram, &flash}) {
    cache->setLayerKVSources({-1, -1, -1, 1});
    cache->allocate(4, 1, max_seq, widths, ml::train::TensorDim::DataType::FP16,
                    ml::train::TensorDim::Format::NCHW, rows, keep);
  }
  ASSERT_FALSE(flash.isFlashOwned(0));
  ASSERT_TRUE(flash.isFlashOwned(1));
  ASSERT_FALSE(flash.isFlashOwned(2));
  ASSERT_TRUE(flash.isFlashOwned(3));
}

/**
 * @brief One forward step [from, to) over the mixed fixture, then the window
 *        compaction both managers run at the new position. Every layer enters
 *        the flash bracket; the manager skips the windowed ones. A flash
 *        layer's live rows are only resident inside its bracket, so each
 *        layer is bit-compared against the DRAM reference there.
 */
void mixedStep(causallm::KVCacheManager &dram, causallm::KVCacheManager &flash,
               unsigned from, unsigned to) {
  constexpr unsigned width = 256;
  static constexpr bool windowed[4] = {true, false, true, false};
  flash.beginFlashStep(from);
  for (unsigned layer : {0u, 1u, 2u, 3u}) {
    SCOPED_TRACE(layer);
    flash.beginFlashLayer(layer);
    if (layer != 3u) { // the alias has no plane of its own
      for (auto *cache : {&dram, &flash}) {
        for (unsigned row = from; row < to; ++row) {
          const size_t offset =
            (windowed[layer] ? row - cache->getBase(layer)
                             : row % cache->getLayerCap(layer)) *
            width;
          for (unsigned col = 0; col < width; ++col) {
            const auto value = static_cast<uint16_t>(layer * 500 + row + col);
            cache->getKeyCache(layer).getData<uint16_t>()[offset + col] = value;
            cache->getValueCache(layer).getData<uint16_t>()[offset + col] =
              static_cast<uint16_t>(0xab00u + (value & 0xffu));
          }
        }
      }
    }
    expectSameLayerBytes(dram, flash, layer);
    flash.finishFlashLayer(layer, from, to);
  }
  flash.finishFlashStep();
  dram.setPosition(to);
  flash.setPosition(to);
  EXPECT_TRUE(dram.compact(to));
  EXPECT_TRUE(flash.compact(to));
  EXPECT_EQ(dram.getBase(0), flash.getBase(0));
  EXPECT_EQ(dram.getBase(2), flash.getBase(2));
}
} // namespace

TEST(KVCacheManagerFlash, lookahead_counts_distinct_upcoming_file_reads) {
  const auto page = static_cast<unsigned>(sysconf(_SC_PAGESIZE));
  const auto path = std::filesystem::temp_directory_path().string();
  using Dim = ml::train::TensorDim;
  for (bool mixed : {false, true}) {
    for (bool keep_middle : {false, true}) {
      const auto budget = std::to_string(keep_middle ? 2 * page : 0);
      ScopedEnv retention("NNTR_KV_RESIDENT_BYTES", budget.c_str());
      for (unsigned depth : {0u, 1u, 2u, 8u}) {
        SCOPED_TRACE(::testing::Message()
                     << mixed << "/" << budget << "/" << depth);
        causallm::KVCacheManager cache;
        cache.setFlashStorage(path, depth);
        const unsigned mid = mixed ? 3 : 1, last = mixed ? 6 : 2;
        std::vector<unsigned> widths(last + 1, 1), caps(last + 1),
          keep(last + 1);
        widths[0] = page;
        widths[last] = 2 * page;
        if (mixed) {
          // Window, inactive layer, flash owner, its alias, another window.
          caps[1] = caps[5] = 3;
          keep[1] = keep[5] = 1;
          cache.setActiveLayers({true, true, false, true, true, true, true});
          cache.setLayerKVSources({-1, -1, -1, -1, 3, -1, -1});
        }
        cache.allocate(last + 1, 1, 8, widths, Dim::DataType::FP16,
                       Dim::Format::NCHW, caps, keep);
        const auto read_bytes = [&cache]() -> size_t {
          ::testing::internal::CaptureStderr();
          cache.reportFlashStorage();
          const auto report = ::testing::internal::GetCapturedStderr();
          const std::string field = " read_bytes=";
          return std::stoull(report.substr(report.find(field) + field.size()));
        };
        cache.beginFlashStep(0);
        for (unsigned owner : {0u, mid, last}) {
          cache.beginFlashLayer(owner);
          cache.getKeyCache(owner).getData<uint16_t>()[0] = owner + 1;
          cache.finishFlashLayer(owner, 0, 1);
        }
        cache.finishFlashStep();
        const auto initial = read_bytes();
        cache.beginFlashStep(1);
        cache.beginFlashLayer(0);
        EXPECT_EQ(cache.getKeyCache(0).getData<uint16_t>()[0], 1u);
        cache.finishFlashLayer(0, 1, 2);
        // Drain before entering another attention. Each owner's byte count
        // is distinct, so this checks exactly which owners were read ahead.
        cache.finishFlashStep();
        size_t expected = 4 * page;
        if (!keep_middle && depth >= 1)
          expected += 2 * page;
        if (depth >= (keep_middle ? 1u : 2u))
          expected += 8 * page;
        EXPECT_EQ(read_bytes() - initial, expected);
      }
    }
  }
}

TEST(KVCacheManagerFlash, capped_rolling_preserves_dirty_owners_across_growth) {
  using Dim = ml::train::TensorDim;
  const unsigned page = getpagesize(), width = page / 4;
  // Four stream owners, one DRAM window (3 FP16 rows x K,V) and an alias.
  const auto cap = 4 * page + 12;
  const auto limit = std::to_string(cap);
  ScopedEnv total("NNTR_KV_TOTAL_BYTES", limit.c_str());
  ScopedEnv retention("NNTR_KV_RESIDENT_BYTES", limit.c_str());
  for (const char *mode : {"0", "1"}) {
    ScopedEnv policy("NNTR_KV_ROLLING", mode);
    causallm::KVCacheManager cache;
    cache.setFlashStorage(std::filesystem::temp_directory_path().string(), 3);
    cache.setLayerKVSources({-1, -1, -1, -1, 2, -1});
    cache.allocate(6, 1, 8, {width, 1, width, width, width, width},
                   Dim::DataType::FP16, Dim::Format::NCHW, {0, 3, 0, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0});
    for (unsigned row = 0; row < 4; ++row) {
      cache.beginFlashStep(row, row + 1);
      for (unsigned layer : {0u, 2u, 3u, 4u, 5u}) {
        const unsigned owner = layer == 4 ? 2 : layer;
        cache.beginFlashLayer(layer);
        auto *k = cache.getKeyCache(layer).getData<uint16_t>();
        auto *v = cache.getValueCache(layer).getData<uint16_t>();
        for (unsigned r = 0; r < row + (layer != owner); ++r) {
          EXPECT_EQ(k[r * width], 10 * owner + r + 1);
          EXPECT_EQ(v[r * width], 100 * owner + r + 1);
        }
        if (layer == owner) {
          k[row * width] = 10 * owner + row + 1;
          v[row * width] = 100 * owner + row + 1;
        }
        cache.finishFlashLayer(layer, row, row + 1);
      }
      cache.finishFlashStep();
    }
    ::testing::internal::CaptureStderr();
    cache.reportFlashStorage();
    const auto report = ::testing::internal::GetCapturedStderr();
    const auto field = [&report](const std::string &name) {
      const auto pos = report.find(" " + name + "=");
      if (pos == std::string::npos)
        throw std::runtime_error("missing FSU field: " + name);
      return std::stoull(report.substr(pos + name.size() + 2));
    };
    EXPECT_EQ(field("total_budget_bytes"), cap);
    EXPECT_EQ(field("kv_payload_peak_bytes"), cap);
    EXPECT_EQ(field("borrowed_owners") > 0, *mode == '1');
    cache.beginFlashStep(4, 5);
    EXPECT_THROW(cache.beginFlashLayer(0), std::runtime_error);
    cache.abortFlashStep();
    cache.beginFlashStep(0, 1);
    cache.beginFlashLayer(0);
    EXPECT_EQ(cache.getKeyCache(0).getData<uint16_t>()[0], 0u);
    cache.finishFlashLayer(0, 0, 1);
    cache.finishFlashStep();
  }
}

TEST(KVCacheManagerFlash, rejects_total_budget_smaller_than_sliding_storage) {
  ScopedEnv total("NNTR_KV_TOTAL_BYTES", "11");
  using Dim = ml::train::TensorDim;
  causallm::KVCacheManager cache;
  cache.setFlashStorage(std::filesystem::temp_directory_path().string(), 1);
  EXPECT_THROW(cache.allocate(1, 1, 8, std::vector<unsigned>{1},
                              Dim::DataType::FP16, Dim::Format::NCHW, {3}, {1}),
               std::invalid_argument);
}

TEST(KVCacheManagerFlash, capped_demand_reclaims_async_write_staging) {
  using Dim = ml::train::TensorDim;
  constexpr unsigned chunk = 65536;
  ScopedEnv total("NNTR_KV_TOTAL_BYTES", "262144");
  ScopedEnv retention("NNTR_KV_RESIDENT_BYTES", "0");
  ScopedEnv async_write("NNTR_KV_ASYNC_WRITE", "1");
  causallm::KVCacheManager cache;
  cache.setFlashStorage(std::filesystem::temp_directory_path().string(), 1);
  cache.allocate(1, 1, 8, 1, chunk / 8, Dim::DataType::FP16);
  for (unsigned from : {0u, 4u}) {
    cache.beginFlashStep(from, from + 4);
    cache.beginFlashLayer(0);
    auto *k = cache.getKeyCache(0).getData<uint16_t>();
    if (from)
      EXPECT_EQ(k[0], 12u);
    k[from * chunk / 8] = 12 + from;
    cache.finishFlashLayer(0, from, from + 4);
    cache.finishFlashStep();
    ::testing::internal::CaptureStderr();
    cache.reportFlashStorage();
    const auto report = ::testing::internal::GetCapturedStderr();
    if (report.find("async_writer=1") != std::string::npos) {
      EXPECT_NE(
        report.find(from ? "staging_bytes=0 " : "staging_bytes=131072 "),
        std::string::npos);
      EXPECT_NE(report.find("async_writes=1 "), std::string::npos);
    }
    if (from)
      EXPECT_NE(report.find("kv_payload_peak_bytes=262144 "),
                std::string::npos);
  }
}

TEST(KVCacheManagerFlash, mixed_storage_keeps_windowed_layers_off_flash) {
  using DT = ml::train::TensorDim::DataType;
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

#ifdef ENABLE_FP16
  const auto dtype = DT::FP16;
#else
  const auto dtype = DT::UINT16;
#endif
  // [kv-fsu mixed, design §2.5 D] rows/keep with flash storage used to be
  // refused outright; now only a windowed FLASH OWNER is impossible: a
  // windowed layer keeps a host DRAM tensor (its compaction memmoves rows,
  // which a flash owner's pool buffer cannot honor) while the remaining
  // layers stream through the FSU. Pin the split from both sides.
  causallm::KVCacheManager mixed;
  mixed.setFlashStorage(directory.path, 1);
  mixed.setActiveLayers({true, false, true, true});
  mixed.allocate(4, 1, 128, std::vector<unsigned int>(4, 1024), dtype,
                 ml::train::TensorDim::Format::NCHW, {8u, 0u, 8u, 0u},
                 {4u, 0u, 4u, 0u});
  EXPECT_FALSE(mixed.isFlashOwned(0)); // windowed: host DRAM tensor
  EXPECT_FALSE(mixed.isFlashOwned(1)); // inactive: metadata only
  EXPECT_FALSE(mixed.isFlashOwned(2)); // windowed
  EXPECT_TRUE(mixed.isFlashOwned(3));  // full: streams through the FSU
  EXPECT_TRUE(mixed.getKeyCache(0).isAllocated());
  EXPECT_TRUE(mixed.getValueCache(0).isAllocated());
  EXPECT_FALSE(mixed.getKeyCache(1).isAllocated());
  EXPECT_TRUE(mixed.getKeyCache(3).isAllocated());

  // The windowed layers compact on their host tensors while flash storage
  // is on -- the old all-or-nothing guard threw here.
  for (unsigned row = 0; row < 8; ++row)
    for (unsigned col = 0; col < 1024; ++col) {
      mixed.getKeyCache(0).getData<uint16_t>()[row * 1024 + col] =
        static_cast<uint16_t>(0x3c00u + row);
      mixed.getValueCache(0).getData<uint16_t>()[row * 1024 + col] =
        static_cast<uint16_t>(0x4000u + row);
    }
  mixed.setPosition(8);
  EXPECT_TRUE(mixed.compact(8));
  EXPECT_EQ(mixed.getBase(0), 4u);
  EXPECT_EQ(mixed.getKeyCache(0).getData<uint16_t>()[0], 0x3c04u);
  EXPECT_EQ(mixed.getValueCache(0).getData<uint16_t>()[0], 0x4004u);
  EXPECT_EQ(mixed.getKeyCache(0).getData<uint16_t>()[4095], 0x3c07u);
  EXPECT_FALSE(mixed.droppedAmaxK(0).empty());
  EXPECT_FALSE(mixed.droppedAmaxV(0).empty());
  // A full (flash-owned) layer never compacts and never moves.
  EXPECT_EQ(mixed.getBase(3), 0u);
  EXPECT_TRUE(mixed.droppedAmaxK(3).empty());

  // Without rows/keep nothing changes: every layer is flash-owned and a
  // compaction is a no-op.
  causallm::KVCacheManager flash;
  flash.setFlashStorage(directory.path, 1);
  flash.setActiveLayers({true, false, true, true});
  flash.allocate(4, 1, 128, 1, 1024, dtype);
  EXPECT_TRUE(flash.isFlashOwned(0));
  EXPECT_FALSE(flash.isFlashOwned(1));
  EXPECT_FALSE(flash.compact(64));
  EXPECT_EQ(flash.getBase(0), 0u);
  EXPECT_TRUE(flash.droppedAmaxK(0).empty());
}

TEST(KVCacheManagerFlash,
     mixed_windowed_and_flash_layers_match_dram_reference) {
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  causallm::KVCacheManager dram, flash;
  allocateMixedPair(dram, flash, directory.path);

  // The windowed layers cross a compaction boundary at every step (bases
  // 0 -> 1 -> 5 -> 8 -> 12) while the flash layer streams the same
  // positions through the FSU file; the alias reads layer 1's plane.
  unsigned from = 0;
  for (unsigned to : {5u, 9u, 12u, 16u}) {
    SCOPED_TRACE(to);
    mixedStep(dram, flash, from, to);
    from = to;
  }
  EXPECT_EQ(flash.getBase(0), 12u);
  EXPECT_EQ(flash.getBase(2), 12u);

  // The post-compaction window of the host layers, and a full read-back of
  // the flash layer's file after the last boundary. Layer 3's plane is
  // layer 1's MemoryData, so one bracket covers both.
  expectSameLayerBytes(dram, flash, 0);
  expectSameLayerBytes(dram, flash, 2);
  flash.beginFlashLayer(1);
  expectSameLayerBytes(dram, flash, 1);
  expectSameLayerBytes(dram, flash, 3);
  flash.finishFlashLayer(1, 16, 16);
}

TEST(KVCacheManagerFlash, mixed_snapshot_round_trip_matches_dram_bytes) {
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  causallm::KVCacheManager dram, flash;
  allocateMixedPair(dram, flash, directory.path);
  unsigned from = 0;
  for (unsigned to : {5u, 9u, 12u, 16u}) {
    mixedStep(dram, flash, from, to);
    from = to;
  }

  // The file layout stays one K plane + one V plane per layer in layer
  // order, regardless of which storage owns the plane: a mixed save and a
  // DRAM save of the same positions must be byte-identical.
  const auto dram_path = directory.path + "/dram.bin";
  const auto flash_path = directory.path + "/flash.bin";
  const auto roundtrip_path = directory.path + "/roundtrip.bin";
  dram.save(dram_path);
  const auto baseline = readFileBytes(dram_path);
  ASSERT_FALSE(baseline.empty());
  flash.save(flash_path);
  expectSameFileBytes(baseline, readFileBytes(flash_path));

  // Loading the DRAM-written file into the mixed manager: the windowed
  // layers land in their host tensors at the live window, the flash layers
  // go through the FSU, and re-saving reproduces the same bytes.
  flash.reset();
  EXPECT_EQ(flash.load(dram_path, 0), 16u);
  EXPECT_EQ(flash.getBase(0), 12u);
  EXPECT_EQ(flash.getBase(2), 12u);
  flash.save(roundtrip_path);
  expectSameFileBytes(baseline, readFileBytes(roundtrip_path));

  // Continue writing after the load: the next step must match the DRAM
  // reference that resumed from the same file, compaction included.
  dram.reset();
  EXPECT_EQ(dram.load(dram_path, 0), 16u);
  mixedStep(dram, flash, 16, 18);
  expectSameLayerBytes(dram, flash, 0);
  expectSameLayerBytes(dram, flash, 2);
  flash.beginFlashLayer(1);
  expectSameLayerBytes(dram, flash, 1);
  expectSameLayerBytes(dram, flash, 3);
  flash.finishFlashLayer(1, 18, 18);
}

TEST(KVCacheManagerFlash, legacy_dense_window_reads_tail_and_skips_alias_slot) {
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-legacy-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  const std::filesystem::path dir(path);
  const auto file = dir / "dense.bin";
  constexpr unsigned rows = 11, cap = 4, keep = 2;
  using Dim = ml::train::TensorDim;
  causallm::KVCacheManager cache;
  cache.setFlashStorage(path, 1);
  cache.setLayerKVSources({-1, 0, -1});
  cache.allocate(3, 1, 16, std::vector<unsigned>{1, 1, 1}, Dim::DataType::FP16,
                 Dim::Format::NCHW, {cap, cap, 0}, {keep, keep, 0});
  {
    std::ofstream out(file, std::ios::binary);
    for (unsigned plane = 0; plane < 6; ++plane)
      for (unsigned row = 0; row < rows; ++row) {
        const uint16_t value = 0x3c00 + 16 * plane + row;
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
  }
  ASSERT_EQ(cache.load(file.string(), rows), rows);
  for (unsigned layer : {0u, 1u}) {
    EXPECT_EQ(cache.getBase(layer), rows - keep);
    EXPECT_EQ(cache.getKeyCache(layer).getData<uint16_t>()[0], 0x3c09);
    EXPECT_EQ(cache.getValueCache(layer).getData<uint16_t>()[1], 0x3c1a);
  }
  ASSERT_EQ(cache.droppedAmaxK(0).size(), 1u);
  EXPECT_FLOAT_EQ(cache.droppedAmaxK(0)[0], 1.0f + 8.0f / 1024);
  cache.beginFlashLayer(2);
  EXPECT_EQ(cache.getKeyCache(2).getData<uint16_t>()[10], 0x3c4a);
  cache.finishFlashLayer(2, rows, rows);
  std::filesystem::resize_file(file, 12);
  EXPECT_THROW(cache.load(file.string(), rows), std::runtime_error);
  EXPECT_EQ(cache.getBase(0), rows - keep);
  std::filesystem::remove_all(dir);
}

TEST(KVCacheManagerFlash, rejects_invalid_byte_budgets) {
  for (const char *value : {"-1", "", "24MiB", "18446744073709551616"}) {
    ScopedEnv env("NNTR_KV_RESIDENT_BYTES", value);
    causallm::KVCacheManager cache;
    cache.setFlashStorage(std::filesystem::temp_directory_path().string(), 1);
    EXPECT_THROW(
      cache.allocate(1, 1, 16, 1, 16, ml::train::TensorDim::DataType::FP16),
      std::invalid_argument)
      << value;
  }
}

TEST_F(KVCacheManagerTest, legacy_window_preserves_batches_and_plane_encoding) {
  using Dim = ml::train::TensorDim;
  const unsigned cap = 4, keep = 2;
  const auto file =
    std::filesystem::temp_directory_path() / "nntr-kv-legacy-batch-window.bin";
  for (auto dtype : {Dim::DataType::FP16, Dim::DataType::UINT16}) {
    for (unsigned rows : {2u, 7u}) {
      causallm::KVCacheManager cache;
      cache.allocate(1, 2, 8, std::vector<unsigned>{1}, dtype,
                     Dim::Format::NCHW, {cap}, {keep});
      {
        std::ofstream out(file, std::ios::binary);
        nntrainer::Tensor source(
          Dim({2, 1, rows, 1}, {Dim::Format::NCHW, dtype}), true);
        for (unsigned plane : {0u, 1u}) {
          for (unsigned n = 0; n < 2 * rows; ++n)
            source.getData<uint16_t>()[n] = 0x3c00 + 32 * plane + n;
          source.save(out);
        }
      }
      ASSERT_EQ(cache.load(file.string(), rows), rows);
      const unsigned dropped = rows - keep;
      EXPECT_EQ(cache.getBase(0), dropped);
      for (unsigned batch : {0u, 1u}) {
        EXPECT_EQ(cache.getKeyCache(0).getData<uint16_t>()[batch * cap],
                  0x3c00u + batch * rows + dropped);
        EXPECT_EQ(cache.getValueCache(0).getData<uint16_t>()[batch * cap + 1],
                  0x3c20u + batch * rows + dropped + 1);
      }
    }
  }
  std::filesystem::remove(file);
}

TEST(KVCacheManagerFlash, resident_budget_keeps_owners_across_steps) {
  using DT = ml::train::TensorDim::DataType;
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  // 16 MiB keeps every owner of this fixture in the pool across steps.
  ScopedEnv resident_budget("NNTR_KV_RESIDENT_BYTES", "16777216");
  causallm::KVCacheManager dram, flash;
  flash.setFlashStorage(directory.path, 1);
  constexpr unsigned width = 1024, ring_rows = 65, context = 128;
  for (auto *cache : {&dram, &flash}) {
    cache->setActiveLayers({true, false, true, true});
    cache->setLayerCaps({ring_rows, 0, ring_rows, 0});
    cache->setLayerKVSources({-1, -1, 0, -1});
    cache->allocate(4, 1, context, 1, width, DT::FP16);
  }

  const auto compareLayer = [&](unsigned layer) {
    const auto &k = dram.getKeyCache(layer);
    const auto &v = dram.getValueCache(layer);
    EXPECT_EQ(0, std::memcmp(k.getData<uint16_t>(),
                             flash.getKeyCache(layer).getData<uint16_t>(),
                             k.bytes()));
    EXPECT_EQ(0, std::memcmp(v.getData<uint16_t>(),
                             flash.getValueCache(layer).getData<uint16_t>(),
                             v.bytes()));
  };

  unsigned from = 0;
  for (unsigned to : {64u, 70u, 71u}) {
    SCOPED_TRACE(to);
    flash.beginFlashStep(from);
    for (unsigned layer : {0u, 2u, 3u}) {
      SCOPED_TRACE(layer);
      flash.beginFlashLayer(layer);
      compareLayer(layer);
      if (layer != 2u) {
        for (auto *cache : {&dram, &flash}) {
          for (unsigned row = from; row < to; ++row) {
            const auto offset = row % cache->getLayerCap(layer) * width;
            for (unsigned col = 0; col < width; ++col) {
              auto value = static_cast<uint16_t>(layer * 500 + row + col);
              cache->getKeyCache(layer).getData<uint16_t>()[offset + col] =
                value;
              cache->getValueCache(layer).getData<uint16_t>()[offset + col] =
                0xab00u + (value & 0xffu);
            }
          }
        }
      }
      flash.finishFlashLayer(layer, from, to);
    }
    flash.finishFlashStep();
    dram.setPosition(to);
    flash.setPosition(to);
    from = to;
    // Every owner fits the budget, so all three planes stay in the pool and
    // the next step must not read them back from the file.
    EXPECT_GT(flash.flashRetainedBytes(), 0u);
  }
  // Nearly the full ring of every owner (layers 0 and 3; layer 2 aliases
  // layer 0) is retained. The accounting must track the grown rows, not
  // stay at the first step's value. Layer 0 wraps its 65-row ring; layer 3
  // runs linearly to row 71, so their page-aligned plane sizes differ.
  const auto aligned_rows = [](unsigned rows, unsigned cols) {
    return (rows * cols * 2u + 4095u) / 4096u * 4096u;
  };
  EXPECT_EQ(flash.flashRetainedBytes(),
            2u *
              (aligned_rows(std::min(ring_rows, flash.getLayerCap(0u)), width) +
               aligned_rows(std::min(71u, flash.getLayerCap(3u)), width)));

  // A reset drops the retained owners and the pool, and a new prompt still
  // produces the same bytes as the DRAM reference.
  flash.reset();
  EXPECT_EQ(flash.flashRetainedBytes(), 0u);
  from = 0;
  flash.beginFlashStep(from);
  for (unsigned layer : {0u, 2u, 3u}) {
    flash.beginFlashLayer(layer);
    for (unsigned row = from; row < 64; ++row) {
      const auto offset = row % flash.getLayerCap(layer) * width;
      for (unsigned col = 0; col < width; ++col) {
        flash.getKeyCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(row + col);
        flash.getValueCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(0x1000u + row + col);
      }
    }
    flash.finishFlashLayer(layer, from, 64);
  }
  flash.finishFlashStep();
  dram.reset();
  dram.setPosition(0);
  for (unsigned layer : {0u, 2u, 3u}) {
    for (unsigned row = 0; row < 64; ++row) {
      const auto offset = row % dram.getLayerCap(layer) * width;
      for (unsigned col = 0; col < width; ++col) {
        dram.getKeyCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(0x2000u + row + col);
        dram.getValueCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(0x3000u + row + col);
      }
    }
  }
  for (unsigned layer : {0u, 2u, 3u}) {
    flash.beginFlashLayer(layer);
    for (unsigned row = 0; row < 64; ++row) {
      const auto offset = row % flash.getLayerCap(layer) * width;
      for (unsigned col = 0; col < width; ++col) {
        flash.getKeyCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(0x2000u + row + col);
        flash.getValueCache(layer).getData<uint16_t>()[offset + col] =
          static_cast<uint16_t>(0x3000u + row + col);
      }
    }
    flash.finishFlashLayer(layer, 0, 64);
  }
  // Reading back must go through the graph's begin/finish access pattern:
  // finishFlashLayer leaves the binding on the pristine mapping when the
  // owner is retained. Only the rewritten rows are valid on both sides, so
  // the stale ring rows beyond them are excluded from the comparison.
  const auto compareRows = [&](unsigned layer, unsigned rows) {
    const auto &k = dram.getKeyCache(layer);
    const auto &v = dram.getValueCache(layer);
    EXPECT_EQ(0, std::memcmp(k.getData<uint16_t>(),
                             flash.getKeyCache(layer).getData<uint16_t>(),
                             rows * width * 2));
    EXPECT_EQ(0, std::memcmp(v.getData<uint16_t>(),
                             flash.getValueCache(layer).getData<uint16_t>(),
                             rows * width * 2));
  };
  for (unsigned layer : {0u, 2u, 3u}) {
    SCOPED_TRACE("post-reset");
    SCOPED_TRACE(layer);
    flash.beginFlashLayer(layer);
    compareRows(layer, 64);
    flash.finishFlashLayer(layer, 64, 64);
  }
}
TEST(KVCacheManagerFlash, resident_budget_flushes_dirty_on_eviction) {
  using DT = ml::train::TensorDim::DataType;
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  // The budget holds exactly one owner, so every step retains layer 0 and
  // evicts layer 3; its deferred writes must reach the file before release.
  ScopedEnv resident_budget("NNTR_KV_RESIDENT_BYTES", "300000");
  causallm::KVCacheManager dram, flash;
  flash.setFlashStorage(directory.path, 1);
  constexpr unsigned width = 1024, ring_rows = 65, context = 128;
  for (auto *cache : {&dram, &flash}) {
    cache->setActiveLayers({true, false, true, true});
    cache->setLayerCaps({ring_rows, 0, ring_rows, 0});
    cache->setLayerKVSources({-1, -1, 0, -1});
    cache->allocate(4, 1, context, 1, width, DT::FP16);
  }

  const auto compareLayer = [&](unsigned layer) {
    const auto &k = dram.getKeyCache(layer);
    const auto &v = dram.getValueCache(layer);
    EXPECT_EQ(0, std::memcmp(k.getData<uint16_t>(),
                             flash.getKeyCache(layer).getData<uint16_t>(),
                             k.bytes()));
    EXPECT_EQ(0, std::memcmp(v.getData<uint16_t>(),
                             flash.getValueCache(layer).getData<uint16_t>(),
                             v.bytes()));
  };

  unsigned from = 0;
  for (unsigned to : {64u, 70u, 71u}) {
    SCOPED_TRACE(to);
    flash.beginFlashStep(from);
    for (unsigned layer : {0u, 2u, 3u}) {
      SCOPED_TRACE(layer);
      flash.beginFlashLayer(layer);
      compareLayer(layer);
      if (layer != 2u) {
        for (auto *cache : {&dram, &flash}) {
          for (unsigned row = from; row < to; ++row) {
            const auto offset = row % cache->getLayerCap(layer) * width;
            for (unsigned col = 0; col < width; ++col) {
              auto value = static_cast<uint16_t>(layer * 500 + row + col);
              cache->getKeyCache(layer).getData<uint16_t>()[offset + col] =
                value;
              cache->getValueCache(layer).getData<uint16_t>()[offset + col] =
                0xcd00u + (value & 0xffu);
            }
          }
        }
      }
      flash.finishFlashLayer(layer, from, to);
    }
    flash.finishFlashStep();
    dram.setPosition(to);
    flash.setPosition(to);
    from = to;
  }
  // Only layer 0 stays in the pool; its accounting follows the grown rows.
  EXPECT_EQ(flash.flashRetainedBytes(),
            2u * ((65u * 1024u * 2u + 4095) / 4096 * 4096));
}

/**
 * @brief [kv-fsu windowed] reset() after compactions: a fresh request must
 *        find the windowed state at its ORIGIN.
 *
 * The regression: the first request's compactions left bases_ > 0, and the
 * next request (setSessionTokenPosition(0) + compact(0)) died with "layer 0
 * does not hold rows [0, 0)" because setPosition(0) rewinds only the write
 * cursor, not the windowed metadata. reset() restores bases_/decode
 * offsets/dropped maxima (and releases the flash owners); compact(0) is then
 * a clean no-op and a fresh cycle of writes + compactions works from origin.
 */
TEST(KVCacheManagerWindowed, reset_returns_windowed_state_to_origin) {
  using DT = ml::train::TensorDim::DataType;
#ifdef ENABLE_FP16
  const auto dtype = DT::FP16;
#else
  const auto dtype = DT::UINT16;
#endif
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  constexpr unsigned width = 64;
  const auto fill = [](causallm::KVCacheManager &m, unsigned layer,
                       unsigned from, unsigned to) {
    for (unsigned row = from; row < to; ++row)
      for (unsigned col = 0; col < width; ++col) {
        m.getKeyCache(layer)
          .getData<uint16_t>()[(row - m.getBase(layer)) * width + col] =
          static_cast<uint16_t>(0x3c00u + row);
        m.getValueCache(layer)
          .getData<uint16_t>()[(row - m.getBase(layer)) * width + col] =
          static_cast<uint16_t>(0x4000u + row);
      }
  };

  // Host-DRAM windowed layers: two requests of writes/compactions around a
  // reset, with the exact FATAL the review reproduced as the regression.
  causallm::KVCacheManager m;
  m.allocate(2, 1, 32, std::vector<unsigned int>(2, width), dtype,
             ml::train::TensorDim::Format::NCHW, {8u, 0u}, {4u, 0u});
  // request 1: prefill [0, 8) and the compactions it triggers
  fill(m, 0, 0, 8);
  m.setPosition(8);
  ASSERT_TRUE(m.compact(8));
  ASSERT_EQ(m.getBase(0), 4u);
  ASSERT_FALSE(m.droppedAmaxK(0).empty());
  // request 2 (reset:true): the cache must go back to origin, not just rewind
  // the cursor -- compact(0) used to throw "does not hold rows [0, 0)" here.
  m.reset();
  EXPECT_EQ(m.getBase(0), 0u);
  EXPECT_EQ(m.getPosition(), 0u);
  EXPECT_TRUE(m.droppedAmaxK(0).empty());
  EXPECT_TRUE(m.droppedAmaxV(0).empty());
  EXPECT_FALSE(m.compact(0)); // nothing to move at origin, and no throw
  // the fresh request re-writes from absolute 0 and compacts normally again
  fill(m, 0, 0, 8);
  m.setPosition(8);
  EXPECT_TRUE(m.compact(8));
  EXPECT_EQ(m.getBase(0), 4u);
  EXPECT_EQ(m.getKeyCache(0).getData<uint16_t>()[0], 0x3c04u);

  // The FSU variant: the reset also releases the previous request's flash
  // owners, so the next request streams from an empty storage.
  causallm::KVCacheManager flash;
  flash.setFlashStorage(directory.path, 1);
  flash.allocate(2, 1, 32, std::vector<unsigned int>(2, width), dtype,
                 ml::train::TensorDim::Format::NCHW, {8u, 0u}, {4u, 0u});
  ASSERT_FALSE(flash.isFlashOwned(0)); // windowed: host DRAM
  ASSERT_TRUE(flash.isFlashOwned(1));  // full: streams through the FSU
  flash.beginFlashStep(0);
  {
    flash.beginFlashLayer(1);
    for (unsigned row = 0; row < 8; ++row)
      for (unsigned col = 0; col < width; ++col) {
        flash.getKeyCache(1).getData<uint16_t>()[row * width + col] =
          static_cast<uint16_t>(0x3c00u + row);
        flash.getValueCache(1).getData<uint16_t>()[row * width + col] =
          static_cast<uint16_t>(0x4000u + row);
      }
    flash.finishFlashLayer(1, 0, 8);
  }
  flash.finishFlashStep();
  fill(flash, 0, 0, 8);
  flash.setPosition(8);
  ASSERT_TRUE(flash.compact(8));
  ASSERT_EQ(flash.getBase(0), 4u);
  flash.reset();
  EXPECT_EQ(flash.getBase(0), 0u);
  EXPECT_EQ(flash.getPosition(), 0u);
  EXPECT_EQ(flash.flashRetainedBytes(), 0u);
  EXPECT_FALSE(flash.compact(0));
}

/**
 * @brief [kv-fsu mixed] The windowed rows formula (windowedKVRows): the
 *        plane holds the window plus one prefill chunk or the generated
 *        tail; an UNBOUNDED num_to_generate (<= 0, the setupParameters
 *        sentinel) spans the rest of the context and degenerates to full
 *        height. The regression: NUM_TO_GENERATE 0 left only INIT_SEQ_LEN
 *        of headroom and an unlimited decode ran off the end of the plane
 *        mid-generation ("step write [140, 141) does not fit a 64-row
 *        windowed plane based at 76") where the full-plane control ran to
 *        cached_after 511.
 */
TEST(KVCacheManagerWindowed, rows_formula_bounds_the_decode_tail) {
  // bounded: window 4 + max(init 60, ntg 1) -> roundUp64(64) = 64 rows (the
  // gemma4_tiny_win sliding layer at max_seq 128).
  EXPECT_EQ(causallm::windowedKVRows(4, 60, 1, 128), 64u);
  // bounded: the decode tail binds -- window 4 + ntg 100 -> roundUp64(104).
  EXPECT_EQ(causallm::windowedKVRows(4, 4, 100, 512), 128u);
  // bounded but capped at the context height.
  EXPECT_EQ(causallm::windowedKVRows(4, 4, 1000, 128), 128u);
  // unbounded (num_to_generate <= 0): the decode loop stops only at
  // max_seq, so the headroom is max_seq - window and the layer stays a
  // FULL-HEIGHT plane (planKVCache's keep is then 0: not windowed at all).
  EXPECT_EQ(causallm::windowedKVRows(4, 4, 0, 512), 512u);
  EXPECT_EQ(causallm::windowedKVRows(4, 60, 0, 128), 128u);
  EXPECT_EQ(causallm::windowedKVRows(4, 4, -5, 512), 512u);
}

/**
 * @brief [kv-fsu mixed] A WINDOWED alias tracks its owner's compaction base.
 *
 * gemma4's num_kv_shared_layers resolves a trailing shared layer to the last
 * SAME-TYPE layer of the prefix -- when that type is sliding_attention, a
 * windowed layer aliases a windowed owner. compact() moves only owners, so
 * the sharer's base lives ONLY through getBase()'s resolution; a stale 0
 * there made the sharer's attention read plane row 0 as absolute 0 (silent
 * wrong answers) and offset its snapshot slice past the live window.
 */
TEST(KVCacheManagerFlash, windowed_alias_tracks_the_owner_base) {
  using DT = ml::train::TensorDim::DataType;
  auto path =
    (std::filesystem::temp_directory_path() / "nntr-kv-fsu-test-XXXXXX")
      .string();
  ASSERT_NE(mkdtemp(path.data()), nullptr);
  struct Directory {
    std::string path;
    ~Directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } directory{path};

  constexpr unsigned width = 64, max_seq = 20;
  // layer 0: windowed owner; layer 1: full (flash); layer 2: WINDOWED alias
  // of 0; layer 3: alias of 1.
  const std::vector<unsigned int> rows{8u, 0u, 8u, 0u};
  const std::vector<unsigned int> keep{4u, 0u, 4u, 0u};
  causallm::KVCacheManager dram, flash;
  flash.setFlashStorage(directory.path, 1);
  for (auto *cache : {&dram, &flash}) {
    cache->setLayerKVSources({-1, -1, 0, 1});
    cache->allocate(4, 1, max_seq, std::vector<unsigned int>(4, width),
                    DT::FP16, ml::train::TensorDim::Format::NCHW, rows, keep);
  }
  ASSERT_FALSE(flash.isFlashOwned(0));
  ASSERT_TRUE(flash.isFlashOwned(1));
  ASSERT_FALSE(flash.isFlashOwned(2)); // windowed: resolves through owner 0
  ASSERT_TRUE(flash.isFlashOwned(3));

  // The written VALUES, per absolute row: what the owners (and, in the real
  // model, the sharer's byte-identical redundant write) leave in the plane.
  const auto value = [](unsigned layer, unsigned row, unsigned col) {
    return static_cast<uint16_t>(layer * 500u + row + col);
  };
  auto writeOwners = [&](causallm::KVCacheManager &m, unsigned from,
                         unsigned to) {
    for (unsigned layer : {0u, 1u})
      for (unsigned row = from; row < to; ++row)
        for (unsigned col = 0; col < width; ++col) {
          const size_t plane_row =
            layer == 0u ? row - m.getBase(layer) : row % m.getLayerCap(layer);
          m.getKeyCache(layer).getData<uint16_t>()[plane_row * width + col] =
            value(layer, row, col);
          m.getValueCache(layer).getData<uint16_t>()[plane_row * width + col] =
            static_cast<uint16_t>(0xab00u + (value(layer, row, col) & 0xffu));
        }
  };

  unsigned from = 0;
  for (unsigned to : {5u, 8u, 9u}) {
    SCOPED_TRACE(to);
    flash.beginFlashStep(from);
    flash.beginFlashLayer(1);
    writeOwners(dram, from, to);
    writeOwners(flash, from, to);
    flash.finishFlashLayer(1, from, to);
    flash.finishFlashStep();
    dram.setPosition(to);
    flash.setPosition(to);
    ASSERT_TRUE(dram.compact(to));
    ASSERT_TRUE(flash.compact(to));
    // THE regression assertion: the sharer's base is the OWNER's base. It
    // stayed 0 before the fix, and every cache_base push / read span / save
    // offset derived from it was wrong from the owner's first compaction.
    EXPECT_EQ(dram.getBase(2), dram.getBase(0));
    EXPECT_EQ(flash.getBase(2), flash.getBase(0));
    EXPECT_EQ(flash.getBase(3), flash.getBase(1));
    from = to;
  }
  EXPECT_EQ(dram.getBase(0), 5u); // liveFrom(9, keep 4)

  // And the snapshot must stay format-transparent: an aliased layer's
  // region is byte-identical to a DRAM build where the sharer owns a real
  // plane filled with the same byte-identical values (what its redundant
  // write produces). A save offset derived from the stale base 0 wrote the
  // wrong rows here.
  causallm::KVCacheManager dram_ref;
  dram_ref.allocate(4, 1, max_seq, std::vector<unsigned int>(4, width),
                    DT::FP16, ml::train::TensorDim::Format::NCHW, rows, keep);
  from = 0;
  for (unsigned to : {5u, 8u, 9u}) {
    writeOwners(dram_ref, from, to); // layer 2/3 own planes here
    // the sharer's redundant write: byte-identical values, same rows
    for (unsigned layer : {2u, 3u})
      for (unsigned row = from; row < to; ++row)
        for (unsigned col = 0; col < width; ++col) {
          const size_t plane_row = layer == 2u
                                     ? row - dram_ref.getBase(layer)
                                     : row % dram_ref.getLayerCap(layer);
          const unsigned source = layer == 2u ? 0u : 1u;
          dram_ref.getKeyCache(layer)
            .getData<uint16_t>()[plane_row * width + col] =
            value(source, row, col);
          dram_ref.getValueCache(layer)
            .getData<uint16_t>()[plane_row * width + col] =
            static_cast<uint16_t>(0xab00u + (value(source, row, col) & 0xffu));
        }
    dram_ref.setPosition(to);
    ASSERT_TRUE(dram_ref.compact(to));
    from = to;
  }
  const auto aliased_path = directory.path + "/aliased.bin";
  const auto owned_path = directory.path + "/owned.bin";
  dram.save(aliased_path);
  dram_ref.save(owned_path);
  const auto aliased = readFileBytes(aliased_path);
  const auto owned = readFileBytes(owned_path);
  ASSERT_EQ(aliased.size(), owned.size());
  // The 128-byte header differs in one field by design: the geom digest
  // encodes the alias map itself. The PAYLOAD is what the transparency
  // claim is about -- one K plane + one V plane per layer, byte-identical
  // whether the sharer owns its plane or aliases the owner's.
  constexpr size_t kKvHeaderBytes = 128;
  EXPECT_EQ(0, std::memcmp(aliased.data() + kKvHeaderBytes,
                           owned.data() + kKvHeaderBytes,
                           aliased.size() - kKvHeaderBytes));
}

/**
 * @brief [kv-fsu mixed] An alias family slides by ONE window: a sharer whose
 *        keep differs from its owner's would track the owner's base while its
 *        own liveFrom() derives a different window start -- silently shifted
 *        rows. Rejected at allocate(), before any memory is requested.
 */
TEST(KVCacheManagerShare, rejects_windowed_alias_with_mismatched_window) {
  const std::vector<unsigned int> rows{8u, 0u, 8u, 0u};
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, -1, 0, -1});
    // sharer keep 2 vs owner keep 4
    EXPECT_THROW(m.allocate(4, 1, 32, std::vector<unsigned int>(4, 64),
                            ml::train::TensorDim::DataType::FP16,
                            ml::train::TensorDim::Format::NCHW, rows,
                            {4u, 0u, 2u, 0u}),
                 std::invalid_argument);
  }
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, -1, 0, -1});
    // a windowed sharer over a FULL owner is the same mismatch in the other
    // direction (and would also fail the plane-dim check later)
    EXPECT_THROW(m.allocate(4, 1, 32, std::vector<unsigned int>(4, 64),
                            ml::train::TensorDim::DataType::FP16,
                            ml::train::TensorDim::Format::NCHW,
                            {0u, 0u, 8u, 0u}, {0u, 0u, 4u, 0u}),
                 std::invalid_argument);
  }
  {
    causallm::KVCacheManager m;
    m.setLayerKVSources({-1, -1, 0, -1});
    // same window: accepted, and the sharer reads the owner's plane
    m.allocate(4, 1, 32, std::vector<unsigned int>(4, 64),
               ml::train::TensorDim::DataType::FP16,
               ml::train::TensorDim::Format::NCHW, rows, {4u, 0u, 4u, 0u});
    EXPECT_EQ(m.getKeyCache(2).getData<uint16_t>(),
              m.getKeyCache(0).getData<uint16_t>());
  }
}
#endif
