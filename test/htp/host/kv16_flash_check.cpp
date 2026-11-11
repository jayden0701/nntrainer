// SPDX-License-Identifier: Apache-2.0
/** @file kv16_flash_check.cpp
 * @brief Mixed raw/KV16 Flash persistence, framing and budget isolation.
 * No model inference, Qualcomm SDK, Tensor runtime or DSP required.
 */
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <prefix_flash_store.h>
#include <stdexcept>
#include <vector>

namespace {
using Store = causallm::PrefixFlashStore;
using Snapshot = Store::Snapshot;
using Page = Snapshot::Page;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void rejects(Function operation) {
  try {
    operation();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("Corrupt Flash payload was accepted");
}

void word(std::vector<unsigned char> &plane, size_t index, uint16_t value) {
  plane[index * 2] = static_cast<unsigned char>(value);
  plane[index * 2 + 1] = static_cast<unsigned char>(value >> 8);
}

std::shared_ptr<Snapshot> sample(bool compressed) {
  auto snapshot = std::make_shared<Snapshot>();
  snapshot->position = 159;
  snapshot->element_bytes = 2;
  snapshot->layers.resize(2);
  for (unsigned layer = 0; layer < snapshot->layers.size(); ++layer) {
    auto &current = snapshot->layers[layer];
    current.base = layer ? 64 : 0;
    current.width = layer ? 8 : 64;
    current.dropped_k = {0.5f, 1.5f};
    current.dropped_v = {2.5f, 3.5f};
    for (unsigned row = current.base; row < snapshot->position; row += 64) {
      const auto rows = std::min(64u, snapshot->position - row);
      auto page = std::make_shared<Page>();
      page->key.resize(size_t(rows) * current.width * 2);
      page->value.resize(page->key.size());
      for (size_t i = 0; i < page->key.size() / 2; ++i) {
        // Preserve arbitrary FP16 patterns without any floating-point cast.
        const uint16_t bits = uint16_t(i * 71u + row * 13u + layer);
        word(page->value, i, bits);
        word(page->key, i, uint16_t(bits ^ ((i % 32u == 0) ? 0x8011u : 0u)));
      }
      if (compressed && layer == 0 && row != 64) {
        causallm::kv16::Geometry geometry;
        geometry.tokens = rows;
        geometry.heads = 2;
        geometry.dimensions = 32;
        geometry.gamma_bits.assign(32, 0x3f800000u);
        geometry.rotary_dims = {0, 1, 16, 17};
        auto encoded = causallm::kv16::encode(
          page->key.data(), page->value.data(), page->key.size(), geometry);
        require(encoded.size() < page->key.size() * 2,
                "Fixture did not compress");
        page->logical_plane_bytes = page->key.size();
        page->encoded = std::move(encoded);
        std::vector<unsigned char>().swap(page->key);
        std::vector<unsigned char>().swap(page->value);
      }
      current.pages.push_back(std::move(page));
    }
  }
  return snapshot;
}

Snapshot layoutOf(const Snapshot &snapshot) {
  auto layout = snapshot;
  for (auto &layer : layout.layers)
    layer.pages.clear();
  return layout;
}

void same(const Snapshot &expected, const Snapshot &actual,
          bool check_backing = true) {
  require(expected.position == actual.position &&
            expected.element_bytes == actual.element_bytes &&
            expected.layers.size() == actual.layers.size(),
          "Snapshot geometry changed");
  for (size_t l = 0; l < expected.layers.size(); ++l) {
    const auto &a = expected.layers[l], &b = actual.layers[l];
    require(a.base == b.base && a.width == b.width &&
              a.dropped_k == b.dropped_k && a.dropped_v == b.dropped_v &&
              a.pages.size() == b.pages.size(),
            "Layer state changed");
    for (size_t p = 0; p < a.pages.size(); ++p) {
      const auto raw_a = a.pages[p]->decodeCopy();
      const auto raw_b = b.pages[p]->decodeCopy();
      require(raw_a->key == raw_b->key && raw_a->value == raw_b->value,
              "Original K/V bytes changed");
      if (check_backing)
        require(a.pages[p]->compressed() == b.pages[p]->compressed() &&
                  a.pages[p]->encoded == b.pages[p]->encoded,
                "Compressed/raw backing changed");
    }
  }
}

std::vector<uint8_t> read(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  require(bool(file), "Cannot read fixture");
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

void write(const std::filesystem::path &path,
           const std::vector<uint8_t> &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  require(bool(file), "Cannot write fixture");
}

uint64_t nativeWord(const std::vector<uint8_t> &bytes, size_t offset) {
  uint64_t value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

void nativeWord(std::vector<uint8_t> &bytes, size_t offset, uint64_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void rehash(std::vector<uint8_t> &bytes) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 40; i < bytes.size() - 8; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  nativeWord(bytes, bytes.size() - 8, hash);
}

void bakedChecks(const std::filesystem::path &directory,
                 const std::shared_ptr<Snapshot> &snapshot) {
  const auto layout = layoutOf(*snapshot);
  const auto path = directory / "mixed.kv";
  Store::saveSnapshot(path.string(), *snapshot);
  same(*snapshot, *Store::loadSnapshot(path.string(), layout));
  const auto original = read(path);
  require(nativeWord(original, 0) == 0x32464b5652544e4eULL,
          "Mixed snapshot did not use v2 framing");

  // The raw-only v1 format remains readable and contains identical logical KV.
  const auto raw = sample(false);
  const auto raw_path = directory / "raw.kv";
  Store::saveSnapshot(raw_path.string(), *raw);
  require(nativeWord(read(raw_path), 0) == 0x31464b5652544e4eULL,
          "Raw snapshot no longer uses v1 framing");
  same(*raw, *Store::loadSnapshot(raw_path.string(), layoutOf(*raw)));
  same(*raw, *snapshot, false);

  auto bad = original;
  bad.back() ^= 1;
  write(path, bad);
  rejects([&] { Store::loadSnapshot(path.string(), layout); });

  bad = original;
  bad[64 + 48] ^= 1; // Inner KV16 CRC; outer hash recomputed deliberately.
  rehash(bad);
  write(path, bad);
  rejects([&] { Store::loadSnapshot(path.string(), layout); });

  for (const auto &change : std::vector<std::pair<size_t, uint64_t>>{
         {40, 2}, {48, UINT64_MAX}, {56, 1}, {32, UINT64_MAX}}) {
    bad = original;
    nativeWord(bad, change.first, change.second);
    rehash(bad);
    write(path, bad);
    rejects([&] { Store::loadSnapshot(path.string(), layout); });
  }
  for (size_t extent : {size_t(39), original.size() - 1}) {
    bad.assign(original.begin(), original.begin() + extent);
    write(path, bad);
    rejects([&] { Store::loadSnapshot(path.string(), layout); });
  }
  bad = original;
  bad.push_back(0);
  write(path, bad);
  rejects([&] { Store::loadSnapshot(path.string(), layout); });
  write(path, original);
  same(*snapshot, *Store::loadSnapshot(path.string(), layout));

  auto corrupt_snapshot = std::make_shared<Snapshot>(*snapshot);
  auto corrupt_page = std::make_shared<Page>(*snapshot->layers[0].pages[0]);
  corrupt_page->encoded.back() ^= 1;
  corrupt_snapshot->layers[0].pages[0] = corrupt_page;
  const auto invalid_path = directory / "invalid.kv";
  rejects(
    [&] { Store::saveSnapshot(invalid_path.string(), *corrupt_snapshot); });
  require(!std::filesystem::exists(invalid_path) &&
            !std::filesystem::exists(invalid_path.string() + ".tmp"),
          "Invalid compressed payload published a file");
}

void asynchronousChecks(const std::filesystem::path &directory,
                        const std::shared_ptr<Snapshot> &snapshot) {
  size_t payload = 0, logical_raw = 0;
  for (const auto &layer : snapshot->layers)
    for (const auto &page : layer.pages) {
      payload += page->storedBytes() + 24;
      logical_raw += page->keySize() * 2;
    }
  const size_t file_bytes = payload + 48;
  require(file_bytes < logical_raw, "Fixture must demonstrate stored budget");
  const auto store_path = directory / "async";
  Store store(store_path.string(), file_bytes, file_bytes);
  require(store.canStage(payload) && !store.canStage(logical_raw),
          "Store did not use encoded payload budget");
  auto entry = store.submit(snapshot);
  require(bool(entry), "Exact encoded-byte budget rejected snapshot");
  same(*snapshot, *entry->load());
  store.drain();
  same(*snapshot, *entry->load());
  require(store.stats().flash_bytes == file_bytes &&
            store.stats().peak_pending_bytes == file_bytes &&
            store.stats().pending_bytes == 0 && store.stats().written == 1,
          "Encoded page/framing budget accounting incorrect");
  for (const auto &subdirectory :
       std::filesystem::directory_iterator(store_path))
    for (const auto &file : std::filesystem::directory_iterator(subdirectory)) {
      auto bad = read(file.path());
      bad.back() ^= 1;
      write(file.path(), bad);
    }
  require(!entry->load() && !entry->available() && store.stats().failed == 1 &&
            store.stats().flash_bytes == 0,
          "Corruption did not become isolated cache miss");
  // A failed worker must release both the encoded-byte budget and its entry.
  auto corrupted = std::make_shared<Snapshot>(*snapshot);
  auto page = std::make_shared<Page>(*snapshot->layers[0].pages[0]);
  page->encoded.back() ^= 1;
  corrupted->layers[0].pages[0] = page;
  auto invalid = store.submit(corrupted);
  require(bool(invalid), "Corrupt fixture did not enter worker validation");
  store.drain();
  require(!invalid->load() && !invalid->available() &&
            store.stats().failed == 2 && store.stats().pending_bytes == 0 &&
            store.stats().flash_bytes == 0,
          "Failed worker leaked budget or published payload");
}
} // namespace

int main() {
  const auto directory =
    std::filesystem::temp_directory_path() /
    ("nntr-kv16-flash-" +
     std::to_string(
       std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(directory);
  try {
    const auto snapshot = sample(true);
    bakedChecks(directory, snapshot);
    asynchronousChecks(directory, snapshot);
    std::filesystem::remove_all(directory);
    std::cout << "KV16 mixed/raw Flash: baked, async, bytes, corruption and "
                 "budgets PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::filesystem::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
