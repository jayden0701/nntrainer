// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  KV Cache Manager implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_manager.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace causallm {

namespace {

constexpr size_t STREAM_CHUNK_SIZE = 4096;

void checkedWrite(std::ostream &stream, const char *data, size_t bytes) {
  if (bytes >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(
      "KVCacheManager::save: tensor data is too large to write");
  }

  stream.write(data, static_cast<std::streamsize>(bytes));
  if (!stream.good()) {
    throw std::runtime_error("KVCacheManager::save: write failed");
  }
}

void writeZeros(std::ostream &stream, size_t bytes) {
  static const std::array<char, STREAM_CHUNK_SIZE> zeros{};
  while (bytes > 0) {
    const size_t chunk = std::min(bytes, zeros.size());
    checkedWrite(stream, zeros.data(), chunk);
    bytes -= chunk;
  }
}

void checkedRead(std::ifstream &stream, char *data, size_t bytes) {
  if (bytes >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(
      "KVCacheManager::load: tensor data is too large to read");
  }

  stream.read(data, static_cast<std::streamsize>(bytes));
  if (!stream.good()) {
    throw std::runtime_error("KVCacheManager::load: read failed");
  }
}

void discardBytes(std::ifstream &stream, size_t bytes) {
  std::array<char, STREAM_CHUNK_SIZE> buffer;
  while (bytes > 0) {
    const size_t chunk = std::min(bytes, buffer.size());
    checkedRead(stream, buffer.data(), chunk);
    bytes -= chunk;
  }
}

bool usesUIntTensorSerialization(const nntrainer::Tensor &cache) {
  switch (cache.getDataType()) {
  case ml::train::TensorDim::DataType::UINT8:
  case ml::train::TensorDim::DataType::UINT16:
  case ml::train::TensorDim::DataType::UINT32:
    return true;
  default:
    return false;
  }
}

void validateCompactSerialization(const nntrainer::Tensor &cache) {
  if (cache.getMemoryBytes() != cache.bytes() &&
      !usesUIntTensorSerialization(cache)) {
    throw std::invalid_argument(
      "KVCacheManager: compact persistence does not support this cache dtype");
  }
}

void saveSerializationHeader(std::ostream &stream,
                             const nntrainer::Tensor &cache) {
  if (!usesUIntTensorSerialization(cache)) {
    return;
  }

  const auto qscheme = static_cast<std::uint16_t>(cache.q_scheme());
  checkedWrite(stream, reinterpret_cast<const char *>(&qscheme),
               sizeof(qscheme));
}

void loadSerializationHeader(std::ifstream &stream,
                             const nntrainer::Tensor &cache) {
  if (!usesUIntTensorSerialization(cache)) {
    return;
  }

  std::uint16_t qscheme = 0;
  checkedRead(stream, reinterpret_cast<char *>(&qscheme), sizeof(qscheme));
  if (qscheme != static_cast<std::uint16_t>(cache.q_scheme())) {
    throw std::runtime_error(
      "KVCacheManager::load: incompatible cache quantization scheme");
  }
}

void saveCacheTensor(std::ostream &stream, const nntrainer::Tensor &cache,
                     unsigned int seq_len, unsigned int max_seq_len) {
  const auto &dim = cache.getDim();

  /**
   * The legacy implementation saved a B x seq_len shared tensor starting at
   * offset zero. For B > 1 and seq_len < max_seq_len, this is the first
   * B * seq_len physical rows of the full cache, rather than seq_len rows from
   * each batch. Preserve both that payload layout and Tensor's dtype-specific
   * serialization framing for existing cache files.
   */
  if (dim.height() == max_seq_len) {
    ml::train::TensorDim save_dim = dim;
    save_dim.height(seq_len);
    nntrainer::Tensor slice =
      const_cast<nntrainer::Tensor &>(cache).getSharedDataTensor(save_dim, 0,
                                                                 true);
    slice.save(stream);
    return;
  }

  validateCompactSerialization(cache);
  saveSerializationHeader(stream, cache);

  const size_t row_bytes =
    static_cast<size_t>(dim.width()) * dim.getDataTypeSize();
  const size_t total_rows =
    static_cast<size_t>(dim.batch()) * static_cast<size_t>(seq_len);
  const auto *data = reinterpret_cast<const char *>(cache.getData());
  const unsigned int retained_len =
    std::min(seq_len, static_cast<unsigned int>(dim.height()));
  const unsigned int retained_from = seq_len - retained_len;
  const size_t batch_stride =
    static_cast<size_t>(dim.getFeatureLen()) * dim.getDataTypeSize();
  size_t written_rows = 0;

  for (unsigned int batch = 0; batch < dim.batch(); ++batch) {
    const size_t retained_begin =
      static_cast<size_t>(batch) * max_seq_len + retained_from;
    if (retained_begin >= total_rows) {
      break;
    }

    writeZeros(stream, (retained_begin - written_rows) * row_bytes);
    const size_t retained_end =
      std::min(static_cast<size_t>(batch) * max_seq_len + seq_len, total_rows);
    const size_t rows_to_write = retained_end - retained_begin;
    checkedWrite(stream, data + batch * batch_stride,
                 rows_to_write * row_bytes);
    written_rows = retained_end;
  }

  writeZeros(stream, (total_rows - written_rows) * row_bytes);

  const size_t trailing_bytes = cache.getMemoryBytes() - cache.bytes();
  if (trailing_bytes > 0) {
    checkedWrite(stream, data + cache.bytes(), trailing_bytes);
  }
  cache.putData();
}

void loadCacheTensor(std::ifstream &stream, nntrainer::Tensor &cache,
                     unsigned int seq_len, unsigned int max_seq_len) {
  const auto &dim = cache.getDim();

  if (dim.height() == max_seq_len) {
    ml::train::TensorDim load_dim = dim;
    load_dim.height(seq_len);
    nntrainer::Tensor slice = cache.getSharedDataTensor(load_dim, 0, true);
    slice.read(stream);
    return;
  }

  validateCompactSerialization(cache);
  loadSerializationHeader(stream, cache);

  cache.setZero();
  const size_t row_bytes =
    static_cast<size_t>(dim.width()) * dim.getDataTypeSize();
  const size_t total_rows =
    static_cast<size_t>(dim.batch()) * static_cast<size_t>(seq_len);
  auto *data = reinterpret_cast<char *>(cache.getData());
  const unsigned int retained_len =
    std::min(seq_len, static_cast<unsigned int>(dim.height()));
  const unsigned int retained_from = seq_len - retained_len;
  const size_t batch_stride =
    static_cast<size_t>(dim.getFeatureLen()) * dim.getDataTypeSize();
  size_t read_rows = 0;

  for (unsigned int batch = 0; batch < dim.batch(); ++batch) {
    const size_t retained_begin =
      static_cast<size_t>(batch) * max_seq_len + retained_from;
    if (retained_begin >= total_rows) {
      break;
    }

    discardBytes(stream, (retained_begin - read_rows) * row_bytes);
    const size_t retained_end =
      std::min(static_cast<size_t>(batch) * max_seq_len + seq_len, total_rows);
    const size_t rows_to_read = retained_end - retained_begin;
    checkedRead(stream, data + batch * batch_stride, rows_to_read * row_bytes);
    read_rows = retained_end;
  }

  discardBytes(stream, (total_rows - read_rows) * row_bytes);

  const size_t trailing_bytes = cache.getMemoryBytes() - cache.bytes();
  discardBytes(stream, trailing_bytes);
  cache.putData();
}

} // namespace

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  allocate(num_layers, batch_size, max_seq_len, num_heads_kv, head_dim,
           std::vector<unsigned int>(num_layers, max_seq_len), dtype, format);
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              const std::vector<unsigned int> &cache_capacities,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  allocate(num_layers, batch_size, max_seq_len,
           std::vector<unsigned int>(num_layers, num_heads_kv * head_dim),
           cache_capacities, dtype, format);

  num_heads_kv_ = num_heads_kv;
  head_dim_ = head_dim;
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              const std::vector<unsigned int> &kv_widths,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  allocate(num_layers, batch_size, max_seq_len, kv_widths,
           std::vector<unsigned int>(num_layers, max_seq_len), dtype, format);
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              const std::vector<unsigned int> &kv_widths,
                              const std::vector<unsigned int> &cache_capacities,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_layers == 0 || batch_size == 0 || max_seq_len == 0 ||
      kv_widths.size() != num_layers || cache_capacities.size() != num_layers) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: invalid layer, batch, KV width, or cache "
      "capacity count");
  }

  for (unsigned int i = 0; i < num_layers; ++i) {
    if (kv_widths[i] == 0) {
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV widths must be > 0");
    }
    if (cache_capacities[i] == 0 || cache_capacities[i] > max_seq_len) {
      throw std::invalid_argument(
        "KVCacheManager::allocate: cache capacities must be in "
        "[1, max_seq_len]");
    }
  }

  batch_size_ = batch_size;
  max_seq_len_ = max_seq_len;
  num_heads_kv_ = 0;
  head_dim_ = 0;
  kv_width_ = kv_widths[0];
  kv_widths_ = kv_widths;
  cache_capacities_ = cache_capacities;
  dtype_ = dtype;
  format_ = format;
  cache_pos_ = 0;

  layer_caches_.resize(num_layers);
  for (unsigned int i = 0; i < num_layers; ++i) {
    ml::train::TensorDim cache_dim(
      {batch_size, 1, cache_capacities_[i], kv_widths_[i]}, {format, dtype});
    layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, true);
    layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, true);
  }
}

void KVCacheManager::setPosition(unsigned int pos) {
  if (pos > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::setPosition: pos exceeds max_seq_len");
  }
  cache_pos_ = pos;
}

void KVCacheManager::advance(unsigned int step_size) {
  if (cache_pos_ > max_seq_len_ || step_size > max_seq_len_ - cache_pos_) {
    throw std::out_of_range(
      "KVCacheManager::advance: position would exceed max_seq_len");
  }
  cache_pos_ += step_size;
}

void KVCacheManager::reset() { cache_pos_ = 0; }

nntrainer::Tensor &KVCacheManager::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].key_cache;
}

nntrainer::Tensor &KVCacheManager::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].value_cache;
}

unsigned int KVCacheManager::getCacheCapacity(unsigned int layer_idx) const {
  if (layer_idx >= cache_capacities_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getCacheCapacity: invalid layer_idx");
  }
  return cache_capacities_[layer_idx];
}

nntrainer::Tensor KVCacheManager::getKeyCacheWriteView(unsigned int layer_idx,
                                                       unsigned int batch,
                                                       unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: invalid layer_idx");
  }
  if (batch >= batch_size_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: invalid batch");
  }
  const unsigned int capacity = cache_capacities_[layer_idx];
  if (cache_pos_ > capacity || step_size > capacity - cache_pos_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: would exceed layer capacity");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheWriteView(
  unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: invalid layer_idx");
  }
  if (batch >= batch_size_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: invalid batch");
  }
  const unsigned int capacity = cache_capacities_[layer_idx];
  if (cache_pos_ > capacity || step_size > capacity - cache_pos_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: would exceed layer capacity");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getKeyCacheReadView(unsigned int layer_idx,
                                                      unsigned int batch,
                                                      unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: invalid layer_idx");
  }
  if (batch >= batch_size_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: invalid batch");
  }
  if (read_len > cache_capacities_[layer_idx]) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: read_len exceeds layer capacity");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheReadView(unsigned int layer_idx,
                                                        unsigned int batch,
                                                        unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: invalid layer_idx");
  }
  if (batch >= batch_size_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: invalid batch");
  }
  if (read_len > cache_capacities_[layer_idx]) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: read_len exceeds layer capacity");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

void KVCacheManager::save(const std::string &path) const {
  save(path, cache_pos_);
}

void KVCacheManager::save(const std::string &path, unsigned int seq_len) const {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::save: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::save: seq_len exceeds max_seq_len");
  }
  for (unsigned int capacity : cache_capacities_) {
    if (capacity == max_seq_len_) {
      continue;
    }
    const bool has_evicted_rows = cache_pos_ > capacity;
    if (seq_len > cache_pos_ || (has_evicted_rows && seq_len != cache_pos_)) {
      throw std::invalid_argument(
        "KVCacheManager::save: requested logical position is unavailable in "
        "a compact cache");
    }
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::save: cannot open file: " + path);
  }

  for (const auto &lc : layer_caches_) {
    saveCacheTensor(f, lc.key_cache, seq_len, max_seq_len_);
    saveCacheTensor(f, lc.value_cache, seq_len, max_seq_len_);
  }
}

void KVCacheManager::load(const std::string &path, unsigned int seq_len) {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::load: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::load: seq_len exceeds max_seq_len");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::load: cannot open file: " + path);
  }

  for (auto &lc : layer_caches_) {
    loadCacheTensor(f, lc.key_cache, seq_len, max_seq_len_);
    loadCacheTensor(f, lc.value_cache, seq_len, max_seq_len_);
  }

  cache_pos_ = seq_len;
}

} // namespace causallm
