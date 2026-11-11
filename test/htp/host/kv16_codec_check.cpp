// SPDX-License-Identifier: Apache-2.0
/** @file kv16_codec_check.cpp
 * @brief Standalone codec oracle bridge; no DSP or model inference required.
 */
#include <kv16_lossless.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<uint8_t> read(const char *path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot open codec fixture");
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}
void write(const char *path, const std::vector<uint8_t> &data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(data.data()), data.size());
  if (!stream)
    throw std::runtime_error("Cannot write codec output");
}
uint32_t word(const uint8_t *data, unsigned count) {
  uint32_t result = 0;
  for (unsigned i = 0; i < count; ++i)
    result |= uint32_t(data[i]) << (8 * i);
  return result;
}
void require(bool condition) {
  if (!condition)
    throw std::runtime_error("Codec self-check failed");
}
void selfcheck() {
  using namespace causallm::kv16;
  Geometry geometry;
  geometry.heads = 2;
  geometry.dimensions = 4;
  geometry.gamma_bits.assign(4, 0x3f800000);
  geometry.rotary_dims = {0, 2};
  const auto empty = encode(nullptr, nullptr, 0, geometry);
  const auto decoded = decode(empty.data(), empty.size());
  require(decoded.key.empty() && decoded.value.empty());
  require(inspect(empty.data(), empty.size()).heads == 2);
  require(predict(0x8000, 0xbf800000) == 0);
  require(predict(0x8001, 0x3f000000) == 0x8000);
  require(predict(0x7c00, 0) == 0x7e00);
  require(predict(0x7bff, 0x40000000) == 0x7c00);
  geometry.tokens = UINT64_MAX;
  bool rejected = false;
  try {
    (void)encode(nullptr, nullptr, 0, geometry);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
  std::cout
    << "KV16 empty geometry, resource bounds and special patterns PASS\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) {
      selfcheck();
    } else if (argc == 4 && std::string(argv[1]) == "predict") {
      const auto input = read(argv[2]);
      if (input.size() % 6)
        throw std::runtime_error(
          "Predictor fixture must contain u16/u32 records");
      std::vector<uint8_t> output;
      output.reserve(input.size() / 3);
      for (size_t i = 0; i < input.size(); i += 6) {
        const auto result = causallm::kv16::predict(
          uint16_t(word(input.data() + i, 2)), word(input.data() + i + 2, 4));
        output.push_back(uint8_t(result));
        output.push_back(uint8_t(result >> 8));
      }
      write(argv[3], output);
    } else if (argc == 7 && std::string(argv[1]) == "roundtrip") {
      const auto input = read(argv[2]);
      const auto inspected =
        causallm::kv16::inspect(input.data(), input.size());
      const auto decoded = causallm::kv16::decode(input.data(), input.size());
      const auto encoded =
        causallm::kv16::encode(decoded.key.data(), decoded.value.data(),
                               decoded.key.size(), inspected);
      write(argv[3], encoded);
      write(argv[4], decoded.key);
      write(argv[5], decoded.value);
      std::ofstream status(argv[6]);
      status << "PASS\n";
      if (!status)
        throw std::runtime_error("Cannot write status");
    } else if (argc == 3 && std::string(argv[1]) == "inspect") {
      const auto input = read(argv[2]);
      (void)causallm::kv16::inspect(input.data(), input.size());
    } else {
      std::cerr << "Usage: codec_check [predict in out | roundtrip in out k v "
                   "status | inspect in]\n";
      return 1;
    }
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << error.what() << '\n';
    return 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
