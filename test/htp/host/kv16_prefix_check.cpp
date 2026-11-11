// SPDX-License-Identifier: Apache-2.0
/** @file kv16_prefix_check.cpp
 * @brief Opt-in host FP16 checkpoint storage and transactional restoration.
 * No model, SDK or DSP required. Link the production KV manager and codec.
 */
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <kv_cache_manager.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Manager = causallm::KVCacheManager;
using DataType = ml::train::TensorDim::DataType;
using Format = ml::train::TensorDim::Format;

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

template <typename Function> void rejects(Function operation) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("Expected invalid prefix rejection");
}

causallm::kv16::Geometry geometry() {
  causallm::kv16::Geometry g;
  g.tokens = 0;
  g.heads = 2;
  g.dimensions = 32;
  g.block_rows = 16;
  g.gamma_bits.assign(32, 0x3f800000u);
  g.rotary_dims = {0, 1, 16, 17};
  return g;
}

void fill(Manager &cache) {
  for (unsigned layer = 0; layer < cache.getNumLayers(); ++layer) {
    auto &key = cache.getKeyCache(layer);
    auto &value = cache.getValueCache(layer);
    for (size_t i = 0; i < size_t(key.height()) * key.width(); ++i) {
      // Includes signed zero, subnormal and NaN patterns; no float casts.
      value.getData<uint16_t>()[i] = uint16_t(i * 29u + layer);
      key.getData<uint16_t>()[i] = value.getData<uint16_t>()[i];
      if (i % 32u == 0 || i % 32u == 16)
        key.getData<uint16_t>()[i] ^= 0x8011u;
    }
  }
}

std::vector<uint16_t> plane(Manager &cache, unsigned layer, bool key) {
  auto &tensor = key ? cache.getKeyCache(layer) : cache.getValueCache(layer);
  const auto *data = tensor.getData<uint16_t>();
  return {data, data + size_t(tensor.height()) * tensor.width()};
}

void roundTripAndFailureIsolation() {
  Manager cache;
  cache.allocate(2, 1, 256, std::vector<unsigned>{64, 64}, DataType::UINT16,
                 Format::NCHW);
  fill(cache);
  cache.setPosition(129);
  const auto original_k = plane(cache, 0, true);
  const auto original_v = plane(cache, 0, false);
  const auto disabled = cache.capturePrefix(128);
  require(!disabled->layers[0].pages[0]->compressed(), "Disabled path changed");

  cache.configurePrefixCodec(0, geometry(), true);
  const auto compressed = cache.capturePrefix(129, disabled.get());
  const auto &pages = compressed->layers[0].pages;
  require(pages.size() == 3 && pages[0]->compressed() && pages[1]->compressed(),
          "Full pages were not compressed");
  require(!pages[2]->compressed(), "One-token metadata overhead was retained");
  require(!compressed->layers[1].pages[0]->compressed(),
          "Unconfigured layer changed");
  require(pages[0]->storedBytes() < pages[0]->keySize() + pages[0]->valueSize(),
          "Stored byte accounting is not compressed");
  require(pages[0]->key.empty() && pages[0]->value.empty() &&
            !pages[0]->key_view && !pages[0]->key_owner,
          "Compressed page retains original backing");
  try {
    (void)pages[0]->keyData();
    throw std::runtime_error("Compressed page exposed a raw pointer");
  } catch (const std::logic_error &) {
  }
  const auto materialized = cache.materializePrefix(*compressed);
  require(!materialized->layers[0].pages[0]->compressed() &&
            pages[0]->compressed(),
          "Materialization modified immutable backing");
  const auto longer = cache.capturePrefix(129, compressed.get());
  require(longer->layers[0].pages[0] == pages[0],
          "Full page sharing regressed");
  require(longer->layers[0].pages.back() != pages.back(),
          "Partial page was incorrectly shared");

  // A later corrupt page must fail before copying even the earlier valid one.
  auto broken = *compressed;
  auto corrupt = std::make_shared<Manager::PrefixSnapshot::Page>(*pages[1]);
  corrupt->encoded.back() ^= 1;
  broken.layers[0].pages[1] = corrupt;
  for (unsigned layer = 0; layer < 2; ++layer) {
    auto &k = cache.getKeyCache(layer);
    auto &v = cache.getValueCache(layer);
    std::fill_n(k.getData<uint16_t>(), size_t(k.height()) * k.width(), 0xa55au);
    std::fill_n(v.getData<uint16_t>(), size_t(v.height()) * v.width(), 0x55aau);
  }
  cache.setPosition(43);
  const auto before_k = plane(cache, 0, true),
             before_v = plane(cache, 0, false);
  const auto other_k = plane(cache, 1, true), other_v = plane(cache, 1, false);
  rejects([&] { cache.restorePrefix(broken); });
  require(
    cache.getPosition() == 43 && cache.getBase(0) == 0 &&
      plane(cache, 0, true) == before_k && plane(cache, 0, false) == before_v &&
      plane(cache, 1, true) == other_k && plane(cache, 1, false) == other_v,
    "Corrupt restore changed live bytes or state");

  auto changed = geometry();
  changed.gamma_bits[5] = 0xbf800000u;
  cache.configurePrefixCodec(0, changed);
  rejects([&] { cache.restorePrefix(*compressed); });
  require(plane(cache, 0, true) == before_k && cache.getPosition() == 43,
          "Gamma mismatch changed live cache");
  cache.configurePrefixCodec(0, geometry());
  cache.restorePrefix(*compressed);
  const size_t restored_words = 129u * 64u;
  const auto restored_k = plane(cache, 0, true),
             restored_v = plane(cache, 0, false);
  require(cache.getPosition() == 129 &&
            std::equal(original_k.begin(), original_k.begin() + restored_words,
                       restored_k.begin()) &&
            std::equal(original_v.begin(), original_v.begin() + restored_words,
                       restored_v.begin()),
          "Restored FP16 patterns differ");
  cache.clearPrefixCodecs();
  rejects([&] { cache.restorePrefix(*compressed); });
  const auto raw_again = cache.capturePrefix(129, compressed.get());
  require(!raw_again->layers[0].pages[0]->compressed(),
          "Disabled codec reused compressed prior backing");
}

void ownerReleaseAndEligibility() {
  std::map<void *, std::shared_ptr<void>> slabs;
  std::vector<std::weak_ptr<void>> weak;
  std::shared_ptr<const Manager::PrefixSnapshot> snapshot;
  {
    Manager cache;
    cache.setSharedAllocator(
      [&](size_t bytes) -> void * {
        auto *p = std::malloc(bytes);
        if (p) {
          auto owner = std::shared_ptr<void>(p, std::free);
          weak.push_back(owner);
          slabs.emplace(p, std::move(owner));
        }
        return p;
      },
      [&](void *p) { slabs.erase(p); });
    cache.setSharedRetain([&](void *p) { return slabs.at(p); });
    cache.allocate(1, 1, 256, std::vector<unsigned>{64}, DataType::UINT16,
                   Format::NCHW);
    fill(cache);
    cache.setPosition(128);
    cache.configurePrefixCodec(0, geometry());
    snapshot = cache.capturePrefix(128);
    require(snapshot->layers[0].pages[0]->compressed(),
            "Borrowed source was not compressed");
  }
  require(slabs.empty() && !weak.empty(),
          "Allocator test did not release slabs");
  for (const auto &owner : weak)
    require(owner.expired(), "Compressed snapshot pins an RPC slab");
  require(snapshot->layers[0].pages[0]->decodeCopy()->key.size() ==
            64u * 64u * 2u,
          "Compressed snapshot outlived source incorrectly");

  Manager unsupported;
  rejects([&] { unsupported.configurePrefixCodec(0, geometry()); });
  unsupported.allocate(1, 1, 256, std::vector<unsigned>{64}, DataType::FP32,
                       Format::NCHW);
  rejects([&] { unsupported.configurePrefixCodec(0, geometry()); });
  unsupported.allocate(1, 1, 256, std::vector<unsigned>{64}, DataType::UINT16,
                       Format::NCHW, {192}, {64});
  rejects([&] { unsupported.configurePrefixCodec(0, geometry()); });
  Manager aliased;
  aliased.setLayerKVSources({-1, 0});
  aliased.allocate(2, 1, 256, std::vector<unsigned>{64, 64}, DataType::UINT16,
                   Format::NCHW);
  rejects([&] { aliased.configurePrefixCodec(1, geometry()); });
}

void rawFallbackSharing() {
  Manager cache;
  cache.allocate(1, 1, 256, std::vector<unsigned>{64}, DataType::UINT16,
                 Format::NCHW);
  uint32_t state = 4481;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return uint16_t(state >> 16u);
  };
  auto *k = cache.getKeyCache(0).getData<uint16_t>();
  auto *v = cache.getValueCache(0).getData<uint16_t>();
  for (unsigned i = 0; i < 256u * 64u; ++i) {
    k[i] = next();
    v[i] = next();
  }
  cache.setPosition(128);
  cache.configurePrefixCodec(0, geometry());
  const auto first = cache.capturePrefix(128);
  require(!first->layers[0].pages[0]->compressed(),
          "Incompressible page retained codec overhead");
  const auto next_snapshot = cache.capturePrefix(128, first.get());
  require(first->layers[0].pages[0] == next_snapshot->layers[0].pages[0],
          "Unprofitable full page was unnecessarily re-encoded");
}
} // namespace

int main() {
  roundTripAndFailureIsolation();
  ownerReleaseAndEligibility();
  rawFallbackSharing();
  std::cout << "KV16 prefix mixed storage, identity, owner release, failure "
               "isolation: PASS\n";
}
