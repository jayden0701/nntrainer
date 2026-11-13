// SPDX-License-Identifier: Apache-2.0
/** @file kv16_lossless.cpp
 * @brief Integer predictor and portable lossless FP16 cache representation.
 */
#include <kv16_lossless.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace causallm {
namespace kv16 {
namespace {
constexpr size_t header_size = 52, index_size = 16;
enum Mode : uint8_t { Raw = 0, Fixed = 1, Sparse = 2 };

[[noreturn]] void invalid(const char *message) {
  throw std::invalid_argument(message);
}

unsigned bits(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  // Count leading zeros is a single instruction on all supported targets;
  // this helper sits on the per-word encode path (twice per predict).
  return value ? unsigned(64 - __builtin_clzll(value)) : 0;
#else
  unsigned n = 0;
  for (; value; value >>= 1)
    ++n;
  return n;
#endif
}

uint64_t roundShift(uint64_t value, int shift) {
  if (shift <= 0)
    return value << unsigned(-shift);
  if (shift > 64)
    return 0;
  if (shift == 64)
    return value > (uint64_t{1} << 63);
  const uint64_t quotient = value >> unsigned(shift);
  const uint64_t remainder = value & ((uint64_t{1} << shift) - 1);
  const uint64_t halfway = uint64_t{1} << (shift - 1);
  return quotient +
         (remainder > halfway || (remainder == halfway && (quotient & 1)));
}

uint32_t roundBinary(unsigned sign, uint64_t significand, int exponent,
                     unsigned exponent_bits, unsigned fraction_bits) {
  const uint32_t sign_word = sign << (exponent_bits + fraction_bits);
  if (!significand)
    return sign_word;
  const int bias = (1 << (exponent_bits - 1)) - 1;
  const int minimum = 1 - bias;
  int actual = int(bits(significand)) - 1 + exponent;
  if (actual < minimum)
    return sign_word | uint32_t(roundShift(
                         significand, minimum - int(fraction_bits) - exponent));
  uint64_t rounded =
    roundShift(significand, actual - int(fraction_bits) - exponent);
  if (rounded == (uint64_t{1} << (fraction_bits + 1))) {
    rounded >>= 1;
    ++actual;
  }
  if (actual > bias)
    return sign_word | (((uint32_t{1} << exponent_bits) - 1) << fraction_bits);
  return sign_word | (uint32_t(actual + bias) << fraction_bits) |
         uint32_t(rounded - (uint64_t{1} << fraction_bits));
}

struct Parts {
  unsigned sign;
  uint64_t mantissa;
  int exponent;
  bool nan, inf;
};
Parts parts(uint32_t word, unsigned exponent_bits, unsigned fraction_bits) {
  const uint32_t exponent_max = (uint32_t{1} << exponent_bits) - 1;
  const uint32_t exponent = (word >> fraction_bits) & exponent_max;
  const uint32_t fraction = word & ((uint32_t{1} << fraction_bits) - 1);
  const int bias = (1 << (exponent_bits - 1)) - 1;
  return {word >> (exponent_bits + fraction_bits),
          exponent ? (uint64_t{1} << fraction_bits) | fraction : fraction,
          exponent ? int(exponent) - bias - int(fraction_bits)
                   : 1 - bias - int(fraction_bits),
          exponent == exponent_max && fraction != 0,
          exponent == exponent_max && fraction == 0};
}

uint16_t get16(const uint8_t *p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
uint32_t get32(const uint8_t *p) {
  return uint32_t(get16(p)) | (uint32_t(get16(p + 2)) << 16);
}
uint64_t get64(const uint8_t *p) {
  return uint64_t(get32(p)) | (uint64_t(get32(p + 4)) << 32);
}
void put(std::vector<uint8_t> &out, uint64_t value, unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    out.push_back(uint8_t(value >> (8 * i)));
}
void store16(uint8_t *p, uint16_t word) {
  p[0] = uint8_t(word);
  p[1] = uint8_t(word >> 8);
}
void append(std::vector<uint8_t> &out, const uint8_t *data, size_t size) {
  if (size)
    out.insert(out.end(), data, data + size);
}

uint32_t checksum(const uint8_t *data, size_t size) {
  // Slice-by-8 CRC32 (reflected, polynomial 0xedb88320) -- same value as the
  // byte-at-a-time table, ~4x faster on multi-megabyte containers.
  static const auto table = [] {
    std::array<std::array<uint32_t, 256>, 8> result{};
    for (unsigned i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (unsigned bit = 0; bit < 8; ++bit)
        c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0u);
      result[0][i] = c;
    }
    for (unsigned i = 0; i < 256; ++i)
      for (unsigned j = 1; j < 8; ++j)
        result[j][i] =
          (result[j - 1][i] >> 8) ^ result[0][result[j - 1][i] & 255];
    return result;
  }();
  uint32_t c = 0xffffffffu;
  const auto range = [&](const uint8_t *p, size_t n) {
    for (; n >= 8; n -= 8, p += 8) {
      const uint32_t lo = c ^ (uint32_t(p[0]) | uint32_t(p[1]) << 8 |
                               uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24);
      c = table[7][lo & 255] ^ table[6][(lo >> 8) & 255] ^
          table[5][(lo >> 16) & 255] ^ table[4][(lo >> 24) & 255] ^
          table[3][p[4]] ^ table[2][p[5]] ^ table[1][p[6]] ^ table[0][p[7]];
    }
    for (; n; --n, ++p)
      c = table[0][(c ^ *p) & 255] ^ (c >> 8);
  };
  range(data, size < 48 ? size : 48); // the CRC field itself is skipped
  if (size > header_size)
    range(data + header_size, size - header_size);
  return c ^ 0xffffffffu;
}

size_t planeSize(const Geometry &g) {
  if (!g.heads || !g.dimensions || !g.block_rows)
    invalid("KV16 requires positive heads, dimensions and block rows");
  const uint64_t width = uint64_t(g.heads) * g.dimensions;
  if (g.tokens > max_decoded_bytes / 4 / width)
    invalid("KV16 decoded size exceeds limit");
  if ((g.gamma_bits.size() != g.dimensions && g.gamma_bits.size() != width) ||
      g.gamma_bits.size() > std::numeric_limits<uint32_t>::max())
    invalid("KV16 gamma count does not match geometry");
  if (g.rotary_dims.size() > g.dimensions)
    invalid("KV16 rotary count exceeds dimensions");
  for (size_t i = 0; i < g.rotary_dims.size(); ++i)
    if (g.rotary_dims[i] >= g.dimensions ||
        (i && g.rotary_dims[i] <= g.rotary_dims[i - 1]))
      invalid("KV16 rotary indices must be sorted and unique");
  return size_t(g.tokens * width * 2);
}

struct Entry {
  size_t offset, length, rows;
  uint8_t mode, width;
};
struct Parsed {
  Geometry geometry;
  std::vector<Entry> entries;
};

unsigned popcount(uint8_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return unsigned(__builtin_popcount(value));
#else
  unsigned count = 0;
  for (; value; value &= uint8_t(value - 1))
    ++count;
  return count;
#endif
}

Parsed parse(const uint8_t *data, size_t size) {
  if (!data || size < header_size)
    invalid("Truncated KV16 header");
  if (std::memcmp(data, "KV16", 4) || get16(data + 4) != 1 ||
      get16(data + 6) != 1)
    invalid("Unsupported KV16 magic/version/predictor");
  if (get64(data + 40) != size)
    invalid("KV16 file size mismatch");
  Parsed p;
  auto &g = p.geometry;
  g.tokens = get64(data + 8);
  g.heads = get32(data + 16);
  g.dimensions = get32(data + 20);
  g.block_rows = get32(data + 24);
  const uint64_t gamma_count = get32(data + 28);
  const uint64_t rotary_count = get32(data + 32);
  const uint64_t block_count = get32(data + 36);
  if (!g.heads || !g.dimensions || !g.block_rows)
    invalid("Invalid KV16 geometry");
  const uint64_t width = uint64_t(g.heads) * g.dimensions;
  if (g.tokens > max_decoded_bytes / 4 / width ||
      (gamma_count != g.dimensions && gamma_count != width) ||
      rotary_count > g.dimensions)
    invalid("Invalid KV16 geometry");
  const uint64_t rows = g.tokens * g.heads;
  if (block_count != rows / g.block_rows + (rows % g.block_rows != 0))
    invalid("Invalid KV16 block count");
  const uint64_t gamma_end = header_size + gamma_count * 4;
  const uint64_t rotary_end = gamma_end + rotary_count * 4;
  const uint64_t payload_start = rotary_end + block_count * index_size;
  if (payload_start > size)
    invalid("Truncated KV16 metadata/index");
  if (checksum(data, size) != get32(data + 48))
    invalid("KV16 checksum mismatch");
  g.gamma_bits.reserve(size_t(gamma_count));
  for (size_t i = 0; i < gamma_count; ++i)
    g.gamma_bits.push_back(get32(data + header_size + i * 4));
  g.rotary_dims.reserve(size_t(rotary_count));
  for (size_t i = 0; i < rotary_count; ++i)
    g.rotary_dims.push_back(get32(data + gamma_end + i * 4));
  (void)planeSize(g);
  p.entries.reserve(size_t(block_count));
  size_t expected_offset = size_t(payload_start);
  for (size_t block = 0; block < block_count; ++block) {
    const auto *idx = data + size_t(rotary_end) + block * index_size;
    const uint64_t offset = get64(idx);
    const size_t length = get32(idx + 8);
    const uint8_t mode = idx[12], residual_width = idx[13];
    if (offset != expected_offset || get16(idx + 14) || residual_width > 16 ||
        offset > size || length > size - size_t(offset))
      invalid("Invalid KV16 index");
    const size_t count_rows =
      size_t(std::min<uint64_t>(g.block_rows, rows - block * g.block_rows));
    const size_t raw_length = count_rows * size_t(g.dimensions) * 4;
    size_t expected_length = raw_length;
    if (mode == Raw) {
      if (residual_width)
        invalid("RAW16 residual width must be zero");
    } else if (mode == Fixed || mode == Sparse) {
      const size_t count = count_rows * (g.dimensions - size_t(rotary_count));
      const size_t base =
        count_rows * (g.dimensions + size_t(rotary_count)) * 2;
      if (mode == Fixed) {
        const size_t meaningful_bits = count * residual_width;
        expected_length = base + (meaningful_bits + 7) / 8;
        if (length != expected_length)
          invalid("Invalid KV16 fixed payload length");
        if (meaningful_bits % 8 &&
            (data[size_t(offset) + length - 1] >> (meaningful_bits % 8)))
          invalid("Nonzero KV16 residual padding");
      } else {
        if (residual_width)
          invalid("Sparse residual width must be zero");
        const size_t bitmap_size = (count + 7) / 8;
        if (length < base + bitmap_size)
          invalid("Truncated KV16 sparse bitmap");
        const auto *bitmap = data + size_t(offset) + base;
        if (count % 8 && (bitmap[bitmap_size - 1] >> (count % 8)))
          invalid("Nonzero KV16 bitmap padding");
        size_t nonzero = 0;
        for (size_t i = 0; i < bitmap_size; ++i)
          nonzero += popcount(bitmap[i]);
        expected_length = base + bitmap_size + nonzero * 2;
        if (length != expected_length)
          invalid("KV16 sparse popcount/length mismatch");
        for (size_t i = 0; i < nonzero; ++i)
          if (!get16(bitmap + bitmap_size + i * 2))
            invalid("Noncanonical zero KV16 sparse residual");
      }
      if (expected_length >= raw_length)
        invalid("KV16 encoded payload is not smaller than raw");
    } else {
      invalid("Unknown KV16 mode");
    }
    if (length != expected_length)
      invalid("Invalid KV16 payload length");
    p.entries.push_back(
      {size_t(offset), length, count_rows, mode, residual_width});
    expected_offset += length;
  }
  if (expected_offset != size)
    invalid("Trailing KV16 payload");
  return p;
}

std::vector<uint32_t> nonrotary(const Geometry &g) {
  std::vector<uint32_t> result;
  result.reserve(g.dimensions - g.rotary_dims.size());
  size_t rotary = 0;
  for (uint32_t d = 0; d < g.dimensions; ++d)
    if (rotary < g.rotary_dims.size() && g.rotary_dims[rotary] == d)
      ++rotary;
    else
      result.push_back(d);
  return result;
}
/** Same as the public predict, with the gamma word decomposition hoisted out
 *  of per-word loops: parts() of the gamma word is loop-invariant across rows
 *  and blocks for a fixed (head, dimension). */
/** Double rounding for the common normal-normal case: both inputs are
 *  finite normals and the result stays a normal FP16, so the two rounding
 *  shifts are known-width (the product mantissa is 34~35 bits, the FP32
 *  stage rounds it to 24, the FP16 stage to 11) and the whole emulation
 *  is a handful of integer ops. Anything else falls back to the generic
 *  path; results are bit-identical either way. */
uint16_t predict_with_gamma(const Parts &v, const Parts &g) {
  const unsigned sign = v.sign ^ g.sign;
  if (!v.nan && !g.nan && !v.inf && !g.inf && v.mantissa >= (1u << 10) &&
      g.mantissa >= (1u << 23)) {
    const uint64_t product = v.mantissa * g.mantissa; // 34~35 bits
    // FP32 RNE round of the product to 24 significant bits.
    const unsigned shift1 = unsigned(bits(product)) - 24;
    const uint64_t rem1 = product & ((1ull << shift1) - 1);
    uint64_t r1 = product >> shift1;
    r1 += rem1 > (1ull << (shift1 - 1)) ||
          (rem1 == (1ull << (shift1 - 1)) && (r1 & 1));
    // FP32 actual exponent of the rounded significand.
    int exponent = int(bits(product)) - 1 + v.exponent + g.exponent;
    if (r1 == (1ull << 24)) {
      r1 = 1ull << 23;
      ++exponent;
    }
    if (exponent > 127) // FP32 overflow: generic yields inf
      return uint16_t(sign << 15 | 0x7c00);
    if (exponent < -14) // FP16-subnormal (or FP32-subnormal) result: generic
      goto generic;
    // FP16 RNE round to 11 significant bits: r1 is in [2^23, 2^24).
    const uint64_t rem2 = r1 & ((1ull << 13) - 1);
    uint32_t r2 = uint32_t(r1 >> 13);
    r2 += rem2 > (1ull << 12) || (rem2 == (1ull << 12) && (r2 & 1));
    exponent += r2 == (1u << 11);
    if (r2 == (1u << 11))
      r2 = 1u << 10;
    if (exponent > 15) // FP16 overflow: generic yields inf
      return uint16_t(sign << 15 | 0x7c00);
    return uint16_t(sign << 15 | uint32_t(exponent + 15) << 10 |
                    (r2 - (1u << 10)));
  }
generic:
  const unsigned sign2 = v.sign ^ g.sign;
  if (v.nan || g.nan)
    return 0x7e00;
  if (v.inf || g.inf) {
    if ((!v.inf && !v.mantissa) || (!g.inf && !g.mantissa))
      return 0x7e00;
    return uint16_t(sign2 << 15 | 0x7c00);
  }
  const uint32_t product =
    roundBinary(sign2, v.mantissa * g.mantissa, v.exponent + g.exponent, 8, 23);
  const auto p = parts(product, 8, 23);
  if (p.inf)
    return uint16_t((p.sign << 15) | 0x7c00);
  return uint16_t(roundBinary(p.sign, p.mantissa, p.exponent, 5, 10));
}

} // namespace

uint16_t predict(uint16_t value_bits, uint32_t gamma_bits) {
  return predict_with_gamma(parts(value_bits, 5, 10), parts(gamma_bits, 8, 23));
}

std::vector<uint8_t> encode(const uint8_t *key, const uint8_t *value,
                            size_t plane_bytes, const Geometry &g) {
  if (planeSize(g) != plane_bytes || (plane_bytes && (!key || !value)))
    invalid("KV16 input length/pointer mismatch");
  const size_t rows = size_t(g.tokens * g.heads);
  const auto plain = nonrotary(g);
  // Gamma word decompositions, hoisted out of the per-word loop: the gamma
  // for a (head, dimension) pair is constant across rows and blocks.
  std::vector<Parts> gammas;
  if (rows) {
    const size_t gamma_count = g.gamma_bits.size() == g.dimensions
                                 ? g.dimensions
                                 : size_t(g.heads) * g.dimensions;
    gammas.reserve(gamma_count);
    for (size_t i = 0; i < gamma_count; ++i)
      gammas.push_back(parts(g.gamma_bits[i], 8, 23));
  }
  struct Block {
    uint8_t mode, width;
    std::vector<uint8_t> data;
  };
  std::vector<Block> blocks;
  for (size_t start = 0; start < rows;) {
    const size_t count = std::min<size_t>(g.block_rows, rows - start);
    const size_t word_start = start * g.dimensions;
    const size_t word_count = count * g.dimensions;
    // Pass 1: residuals, max width and nonzero count -- the only per-block
    // state the three block sizes depend on. No candidate is built yet.
    std::vector<uint16_t> residuals(count * plain.size());
    auto *residual_out = residuals.data();
    unsigned width = 0;
    size_t nonzero = 0;
    for (size_t row = start; row < start + count; ++row) {
      const size_t gamma_offset = g.gamma_bits.size() == g.dimensions
                                    ? 0
                                    : (row % g.heads) * g.dimensions;
      for (auto d : plain) {
        const size_t offset = (row * g.dimensions + d) * 2;
        const auto residual =
          uint16_t(get16(key + offset) ^
                   predict_with_gamma(parts(get16(value + offset), 5, 10),
                                      gammas[gamma_offset + d]));
        *residual_out++ = residual;
        width = std::max(width, bits(residual));
        nonzero += residual != 0;
      }
    }
    // Pass 2: closed-form sizes. base (the value plane plus the rotary dims
    // copied out of the key) is shared by fixed and sparse and unused by raw,
    // so the winner can be materialized directly without building the two
    // losers. Sizes match the built form byte for byte:
    //   raw   = key plane + value plane
    //   fixed = base + ceil(n * width / 8) packed bits
    //   sparse= base + bitmap(n) + 2 bytes per nonzero
    const size_t base_size = word_count * 2 + count * g.rotary_dims.size() * 2;
    const size_t raw_size = word_count * 4;
    const size_t fixed_size = base_size + (width * residuals.size() + 7u) / 8u;
    const size_t sparse_size =
      base_size + (residuals.size() + 7u) / 8u + nonzero * 2u;
    // Identical tie-breaking to comparing fully built blocks: fixed beats
    // raw first, then sparse must beat the current winner.
    enum Pick { PickRaw, PickFixed, PickSparse };
    Pick pick = PickRaw;
    size_t best = raw_size;
    if (fixed_size < best) {
      pick = PickFixed;
      best = fixed_size;
    }
    if (sparse_size < best) {
      pick = PickSparse;
      best = sparse_size;
    }
    Block chosen{Raw, 0, {}};
    chosen.data.reserve(best);
    switch (pick) {
    case PickRaw:
      append(chosen.data, key + word_start * 2, word_count * 2);
      append(chosen.data, value + word_start * 2, word_count * 2);
      break;
    default: {
      auto &data = chosen.data;
      chosen.mode = uint8_t(pick); // Fixed=1, Sparse=2 match the wire values
      append(data, value + word_start * 2, word_count * 2);
      {
        // Extend first (value-init zero-fills), then overwrite through a
        // pointer: per-byte push_back was dominating the packing cost.
        data.resize(data.size() + count * g.rotary_dims.size() * 2);
        auto *cursor =
          data.data() + data.size() - count * g.rotary_dims.size() * 2;
        for (size_t row = start; row < start + count; ++row)
          for (auto d : g.rotary_dims) {
            store16(cursor, get16(key + (row * g.dimensions + d) * 2));
            cursor += 2;
          }
      }
      if (pick == PickFixed) {
        chosen.width = uint8_t(width);
        data.resize(data.size() + (width * residuals.size() + 7u) / 8u);
        auto *cursor =
          data.data() + data.size() - (width * residuals.size() + 7u) / 8u;
        if (width == 16) {
          std::memcpy(cursor, residuals.data(), residuals.size() * 2);
        } else if (width) {
          uint32_t accumulator = 0;
          unsigned available = 0;
          for (auto residual : residuals) {
            accumulator |= uint32_t(residual) << available;
            available += width;
            while (available >= 8) {
              *cursor++ = uint8_t(accumulator);
              accumulator >>= 8;
              available -= 8;
            }
          }
          if (available)
            *cursor++ = uint8_t(accumulator);
        }
      } else {
        const size_t bitmap_start = data.size();
        data.resize(bitmap_start + (residuals.size() + 7u) / 8u, 0);
        for (size_t i = 0; i < residuals.size(); ++i)
          if (residuals[i]) {
            data[bitmap_start + i / 8] |= uint8_t(1u << (i % 8));
            put(data, residuals[i], 2);
          }
      }
      break;
    }
    }
    blocks.push_back(std::move(chosen));
    start += count;
  }

  size_t offset = header_size +
                  (g.gamma_bits.size() + g.rotary_dims.size()) * 4 +
                  blocks.size() * index_size;
  for (const auto &block : blocks)
    offset += block.data.size();
  std::vector<uint8_t> out{'K', 'V', '1', '6'};
  // Avoid geometric capacity growth retaining as much memory as the raw page.
  // All block choices are known, so reserve the complete container once.
  out.reserve(offset);
  put(out, 1, 2);
  put(out, 1, 2);
  put(out, g.tokens, 8);
  for (uint32_t word :
       {g.heads, g.dimensions, g.block_rows, uint32_t(g.gamma_bits.size()),
        uint32_t(g.rotary_dims.size()), uint32_t(blocks.size())})
    put(out, word, 4);
  put(out, offset, 8);
  put(out, 0, 4);
  for (auto word : g.gamma_bits)
    put(out, word, 4);
  for (auto word : g.rotary_dims)
    put(out, word, 4);
  offset = header_size + (g.gamma_bits.size() + g.rotary_dims.size()) * 4 +
           blocks.size() * index_size;
  for (const auto &block : blocks) {
    put(out, offset, 8);
    put(out, block.data.size(), 4);
    put(out, block.mode, 1);
    put(out, block.width, 1);
    put(out, 0, 2);
    offset += block.data.size();
  }
  for (const auto &block : blocks)
    append(out, block.data.data(), block.data.size());
  const uint32_t crc = checksum(out.data(), out.size());
  for (unsigned i = 0; i < 4; ++i)
    out[48 + i] = uint8_t(crc >> (8 * i));
  return out;
}

Geometry inspect(const uint8_t *data, size_t size) {
  return parse(data, size).geometry;
}

Decoded decode(const uint8_t *data, size_t size) {
  const auto parsed = parse(data, size);
  const auto &g = parsed.geometry;
  const auto plain = nonrotary(g);
  Decoded result;
  result.geometry = g;
  result.key.resize(planeSize(g));
  result.value.resize(result.key.size());
  for (size_t block = 0; block < parsed.entries.size(); ++block) {
    const auto &entry = parsed.entries[block];
    const size_t words = entry.rows * g.dimensions;
    const size_t output_start = block * size_t(g.block_rows) * g.dimensions * 2;
    const auto *payload = data + entry.offset;
    if (entry.mode == Raw) {
      std::memcpy(result.key.data() + output_start, payload, words * 2);
      std::memcpy(result.value.data() + output_start, payload + words * 2,
                  words * 2);
      continue;
    }
    std::memcpy(result.value.data() + output_start, payload, words * 2);
    const auto *rotated = payload + words * 2;
    const auto *residuals = rotated + entry.rows * g.rotary_dims.size() * 2;
    const size_t residual_count = entry.rows * plain.size();
    size_t residual_index = 0, input_offset = 0;
    size_t sparse_offset = (residual_count + 7) / 8;
    uint32_t accumulator = 0;
    unsigned available = 0;
    for (size_t local_row = 0; local_row < entry.rows; ++local_row) {
      const size_t row = block * size_t(g.block_rows) + local_row;
      const size_t row_start = output_start + local_row * g.dimensions * 2;
      for (auto d : g.rotary_dims) {
        store16(result.key.data() + row_start + d * 2, get16(rotated));
        rotated += 2;
      }
      const size_t gamma_offset = g.gamma_bits.size() == g.dimensions
                                    ? 0
                                    : (row % g.heads) * g.dimensions;
      for (auto d : plain) {
        uint16_t residual = 0;
        if (entry.mode == Fixed) {
          while (available < entry.width) {
            accumulator |= uint32_t(residuals[input_offset++]) << available;
            available += 8;
          }
          residual = uint16_t(accumulator & ((uint32_t{1} << entry.width) - 1));
          accumulator >>= entry.width;
          available -= entry.width;
        } else if (residuals[residual_index / 8] &
                   (1u << (residual_index % 8))) {
          residual = get16(residuals + sparse_offset);
          sparse_offset += 2;
        }
        ++residual_index;
        const size_t output = row_start + d * 2;
        store16(result.key.data() + output,
                predict(get16(result.value.data() + output),
                        g.gamma_bits[gamma_offset + d]) ^
                  residual);
      }
    }
  }
  return result;
}

} // namespace kv16
} // namespace causallm
