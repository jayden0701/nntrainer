// SPDX-License-Identifier: Apache-2.0
/** @file baked_prefix_check.cpp
 * @brief Persistent asset roundtrip, bit-exact calibration and rejection paths.
 */
#include <baked_prefix.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <htp_backend/htp_attention_state.h>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <kv16_lossless.h>
#include <sstream>
#include <stdexcept>
using Bake = causallm::BakedPrefix;
using json = nlohmann::json;
static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Baked prefix check failed");
}
template <class F> static void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::exception &) {
    bad = true;
  }
  require(bad);
}
static std::string digest(const std::string &s) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << h;
  return out.str();
}

static void compressedBakedChecks(
  const std::filesystem::path &root, const json &identity,
  const std::shared_ptr<nntrainer::HtpPrefixCalibration> &calibration) {
  using Store = causallm::PrefixFlashStore;
  Store::Snapshot snapshot;
  snapshot.position = 128;
  snapshot.element_bytes = 2;
  snapshot.layers.resize(2);
  size_t raw_file_bytes = 48;
  for (unsigned layer = 0; layer < snapshot.layers.size(); ++layer) {
    auto &current = snapshot.layers[layer];
    current.base = layer ? 64 : 0;
    current.width = layer ? 8 : 64;
    current.dropped_k = {0.0f, 1.25f};
    current.dropped_v = {0.5f, 2.0f};
    for (unsigned row = current.base; row < snapshot.position; row += 64) {
      auto page = std::make_shared<Store::Snapshot::Page>();
      page->key.resize(64 * current.width * 2);
      page->value.resize(page->key.size());
      for (size_t i = 0; i < page->key.size() / 2; ++i) {
        const uint16_t special[] = {0, 0x8000, 0x7d11, 0x7fff, 0x7c00};
        const uint16_t value =
          i < 5 ? special[i] : uint16_t(i * 71u + row * 13u + layer);
        // These are FP16 bit patterns, including NaN payloads and signed zero.
        page->key[i * 2] = page->value[i * 2] = uint8_t(value);
        page->key[i * 2 + 1] = page->value[i * 2 + 1] = uint8_t(value >> 8);
      }
      raw_file_bytes += page->key.size() * 2;
      if (!layer && !row) {
        causallm::kv16::Geometry geometry;
        geometry.tokens = 64;
        geometry.heads = 2;
        geometry.dimensions = 32;
        geometry.gamma_bits.assign(32, 0x3f800000u);
        geometry.rotary_dims = {0, 1, 16, 17};
        page->encoded = causallm::kv16::encode(
          page->key.data(), page->value.data(), page->key.size(), geometry);
        require(page->encoded.size() < page->key.size() * 2);
        page->logical_plane_bytes = page->key.size();
        page->key.clear();
        page->value.clear();
      }
      current.pages.push_back(std::move(page));
    }
  }
  const auto directory = root / "compressed-asset";
  {
    Bake::Writer writer(directory.string(), identity,
                        std::vector<unsigned>(128, 9), 140);
    writer.add(0, 128, 128, snapshot, {calibration, calibration});
    const auto info = writer.finish();
    require(info["baked_tokens"] == 128 &&
            info["remaining_system_tokens"] == 12);
  }
  auto asset = Bake::load(directory.string(), identity, 256, 2);
  const auto frame = asset.frames.at(128);
  // This regression must exercise a genuinely smaller v2 payload: the old
  // raw-file-size equality in BakedPrefix::load rejected this valid asset.
  require(std::filesystem::file_size(frame->payload) < raw_file_bytes);
  auto restored = Store::loadSnapshot(frame->payload, *frame->layout);
  require(restored->layers[0].pages[0]->compressed() &&
          !restored->layers[0].pages[1]->compressed() &&
          !restored->layers[1].pages[0]->compressed());
  require(restored->layers[0].pages[0]->encoded ==
          snapshot.layers[0].pages[0]->encoded);
  for (size_t layer = 0; layer < snapshot.layers.size(); ++layer) {
    const auto &expected = snapshot.layers[layer];
    const auto &actual = restored->layers[layer];
    require(expected.base == actual.base && expected.width == actual.width &&
            expected.dropped_k == actual.dropped_k &&
            expected.dropped_v == actual.dropped_v &&
            expected.pages.size() == actual.pages.size());
    for (size_t page = 0; page < expected.pages.size(); ++page) {
      const auto a = expected.pages[page]->decodeCopy();
      const auto b = actual.pages[page]->decodeCopy();
      require(a->key == b->key && a->value == b->value);
    }
    const auto &saved = *frame->calibration[layer];
    require(saved.q_enc.size() == calibration->q_enc.size() &&
            std::memcmp(saved.q_enc.data(), calibration->q_enc.data(),
                        saved.q_enc.size() * sizeof(float)) == 0);
  }
  rejects([&] {
    Bake::load(directory.string(), json{{"runtime", "changed"}}, 256, 2);
  });
  std::ifstream source(frame->payload, std::ios::binary);
  const std::vector<unsigned char> original{
    std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
  source.close();
  auto write = [&](const std::vector<unsigned char> &bytes) {
    std::ofstream file(frame->payload, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    require(bool(file));
  };
  auto bad = original;
  const uint64_t huge_extent = UINT64_MAX;
  std::memcpy(bad.data() + 32, &huge_extent, sizeof(huge_extent));
  write(bad);
  rejects([&] { Bake::load(directory.string(), identity, 256, 2); });
  write(original);
  std::filesystem::resize_file(frame->payload, original.size() - 1);
  rejects([&] { Bake::load(directory.string(), identity, 256, 2); });
  bad = original;
  bad.push_back(0);
  write(bad);
  rejects([&] { Bake::load(directory.string(), identity, 256, 2); });
  // Same-size corruption remains a lazy payload-integrity failure rather
  // than loading damaged K/V. Metadata load should still be bounded and safe.
  bad = original;
  bad.back() ^= 1;
  write(bad);
  auto corrupt = Bake::load(directory.string(), identity, 256, 2);
  rejects([&] {
    Store::loadSnapshot(corrupt.frames.at(128)->payload,
                        *corrupt.frames.at(128)->layout);
  });
  write(original);
  asset = Bake::load(directory.string(), identity, 256, 2);
  require(Store::loadSnapshot(asset.frames.at(128)->payload,
                              *asset.frames.at(128)->layout)
            ->layers[0]
            .pages[0]
            ->encoded == snapshot.layers[0].pages[0]->encoded);
}

int main() {
  const auto root =
    std::filesystem::temp_directory_path() /
    ("nntr-baked-check-" +
     std::to_string(
       std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  try {
    const auto weights = root / "weights.bin";
    {
      std::ofstream f(weights);
      f << "0123456789abcdef";
    }
    const auto stamp = Bake::fingerprintFile(weights.string());
    const json identity = {{"weights", stamp}, {"runtime", "original"}};
    auto page = std::make_shared<causallm::PrefixFlashStore::Snapshot::Page>();
    page->key.resize(64 * 2 * 2);
    page->value.resize(64 * 2 * 2);
    for (size_t i = 0; i < page->key.size(); ++i) {
      page->key[i] = i % 256;
      page->value[i] = (i * 3) % 256;
    }
    causallm::PrefixFlashStore::Snapshot snapshot;
    snapshot.position = 64;
    snapshot.element_bytes = 2;
    snapshot.layers.resize(1);
    snapshot.layers[0].width = 2;
    snapshot.layers[0].pages = {page};
    auto cal = std::make_shared<nntrainer::HtpPrefixCalibration>();
    cal->calibrated = true;
    cal->scale_k = {1.25f};
    cal->scale_v = {0.25f, 0.75f};
    cal->q_enc = {1e-12f, -123.0f};
    cal->out_enc = {0.5f, 32768.0f};
    cal->kmax = {3.25f};
    cal->vmax = {9.5f, 7.0f};
    const auto asset_dir = root / "asset";
    {
      Bake::Writer writer(asset_dir.string(), identity,
                          std::vector<unsigned>(64, 7), 70);
      writer.add(0, 64, 64, snapshot, {cal});
      const auto info = writer.finish();
      require(info["baked_tokens"] == 64 &&
              info["remaining_system_tokens"] == 6);
    }
    auto asset = Bake::load(asset_dir.string(), identity, 128, 1);
    require(asset.tokens.size() == 64 && asset.frames.size() == 1);
    auto restored = causallm::PrefixFlashStore::loadSnapshot(
      asset.frames.at(64)->payload, *asset.frames.at(64)->layout);
    require(restored->layers[0].pages[0]->key == page->key &&
            restored->layers[0].pages[0]->value == page->value);
    const auto &saved = *asset.frames.at(64)->calibration[0];
    require(saved.scale_k == cal->scale_k &&
            saved.q_enc.size() == cal->q_enc.size() &&
            std::memcmp(saved.q_enc.data(), cal->q_enc.data(),
                        cal->q_enc.size() * sizeof(float)) == 0);
    compressedBakedChecks(root, identity, cal);
    rejects([&] {
      Bake::Writer writer(asset_dir.string(), identity,
                          std::vector<unsigned>(64, 7), 70);
    });
    rejects([&] {
      Bake::load(asset_dir.string(), json{{"weights", "different"}}, 128, 1);
    });
    const auto mtime = std::filesystem::last_write_time(weights);
    {
      std::fstream f(weights, std::ios::in | std::ios::out);
      f.seekp(8);
      f.put('X');
    }
    std::filesystem::last_write_time(weights, mtime);
    require(Bake::fingerprintFile(weights.string()) != stamp);
    // Metadata damage is rejected before its scale values can reach inference.
    const auto manifest = asset_dir / "manifest.json";
    json doc;
    {
      std::ifstream f(manifest);
      f >> doc;
    }
    const auto original = doc;
    auto write = [&] {
      std::ofstream f(manifest);
      f << doc.dump();
    };
    doc["body"]["system_tokens"] = 75;
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    doc["body"]["frames"][0]["calibration"][0]["scale_k"][0] =
      uint32_t(0x7fc00000);
    doc["checksum"] = digest(doc["body"].dump());
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    doc["body"]["frames"][0]["payload"] = "../weights.bin";
    doc["checksum"] = digest(doc["body"].dump());
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    write();
    // Metadata can load lazily, but corrupted same-size payload must not
    // restore.
    {
      std::fstream f(asset.frames.at(64)->payload,
                     std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(48);
      f.put('\x5a');
    }
    auto corrupt = Bake::load(asset_dir.string(), identity, 128, 1);
    rejects([&] {
      causallm::PrefixFlashStore::loadSnapshot(corrupt.frames.at(64)->payload,
                                               *corrupt.frames.at(64)->layout);
    });
    std::filesystem::resize_file(asset.frames.at(64)->payload, 32);
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    const auto aborted = root / "aborted";
    {
      Bake::Writer writer(aborted.string(), identity,
                          std::vector<unsigned>(64, 7), 64);
    }
    require(!std::filesystem::exists(aborted));
    require(
      std::filesystem::exists(manifest)); // asset ownership survives readers.
    std::filesystem::remove_all(root);
    std::cout << "Persistent prefix: raw/v2 mixed codec, exact calibration, "
                 "content identity, extents, lazy corruption, rollback PASS\n";
  } catch (...) {
    std::filesystem::remove_all(root);
    throw;
  }
}
