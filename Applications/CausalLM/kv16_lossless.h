// SPDX-License-Identifier: Apache-2.0
/** @file kv16_lossless.h
 * @brief Portable, bit-exact FP16 K/V storage codec (experimental format v1).
 */
#ifndef NNTRAINER_KV16_LOSSLESS_H
#define NNTRAINER_KV16_LOSSLESS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace causallm {
namespace kv16 {

/** Logical contiguous little-endian [token, head, dimension] geometry. */
struct Geometry {
  uint64_t tokens = 0;
  uint32_t heads = 0, dimensions = 0, block_rows = 16;
  std::vector<uint32_t> gamma_bits, rotary_dims;
};

/** Newly owned output; no partial cache publication on decode failure. */
struct Decoded {
  Geometry geometry;
  std::vector<uint8_t> key, value;
};

/** Maximum decoded K+V bytes, not a bound on total temporary allocations. */
constexpr uint64_t max_decoded_bytes = uint64_t{1} << 30;

/** Integer IEEE FP32 multiply RNE followed by FP16 RNE; canonical NaN 0x7e00.
 */
uint16_t predict(uint16_t value_bits, uint32_t gamma_bits);

/** Encode original FP16 bytes. The whole container can exceed raw K+V bytes.
 *  Throws std::invalid_argument on unsupported geometry or input length. */
std::vector<uint8_t> encode(const uint8_t *key, const uint8_t *value,
                            size_t plane_bytes, const Geometry &geometry);

/** Fully validate version, CRC, geometry, offsets and all block
 * representations. Does not materialize K/V. Throws std::invalid_argument on
 * invalid input. */
Geometry inspect(const uint8_t *data, size_t size);

/** Validate before allocating output; input is never mutated. */
Decoded decode(const uint8_t *data, size_t size);

} // namespace kv16
} // namespace causallm
#endif
