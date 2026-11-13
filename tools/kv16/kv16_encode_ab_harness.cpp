// SPDX-License-Identifier: Apache-2.0
/** @file kv16_encode_ab_harness.cpp
 * @brief Review harness for the KV16 encode optimization stack: byte-exact
 *        A/B against the pre-optimization implementation, an exhaustive
 *        predict() comparison and an encode throughput benchmark.
 *
 * The optimized implementation comes from this tree (kv16_lossless.cpp). The
 * baseline comes from a git revision supplied on the command line, extracted,
 * renamed to namespace causalleg and compiled as a second translation unit:
 *
 *   OLD_REV=<sha>  # e.g. the commit before the optimization stack
 *   git show $OLD_REV:Applications/CausalLM/kv16_lossless.cpp |
 *     sed -e 's/^namespace causallm {/namespace causalleg {/' \
 *         -e 's/^} \/\/ namespace causallm$/} \/\/ namespace causalleg/' \
 *         -e '13a using causallm::kv16::Geometry;\nusing
 * causallm::kv16::Decoded;\nusing causallm::kv16::max_decoded_bytes;' \ >
 * /tmp/kv16_old_renamed.cpp
 *   g++ -O2 -std=c++17 -I. -Iapi/ccapi/include -IApplications/CausalLM \
 *       -IApplications/CausalLM/app_config \
 *       tools/kv16/kv16_encode_ab_harness.cpp /tmp/kv16_old_renamed.cpp \
 *       -o /tmp/kv16_ab
 *   /tmp/kv16_ab            # byte-exact encode A/B + decode round-trips
 *   /tmp/kv16_ab predict    # exhaustive predict(): 65536 values x 4096 gammas
 *   /tmp/kv16_ab bench      # encode throughput, old vs new, mix pattern
 *
 * Exit code 0 means byte-exact. The A/B covers 2,048 randomized
 * geometry/data-pattern cases exercising raw, fixed and sparse blocks; any
 * byte mismatch, decode round-trip failure or thrown exception fails the run.
 */
#include "Applications/CausalLM/kv16_lossless.cpp"

#include <chrono>
#include <cstdio>
#include <map>
#include <string>

namespace causalleg {
namespace kv16 {
std::vector<uint8_t> encode(const uint8_t *key, const uint8_t *value,
                            size_t plane_bytes,
                            const causallm::kv16::Geometry &geometry);
uint16_t predict(uint16_t value_bits, uint32_t gamma_bits);
} // namespace kv16
} // namespace causalleg

using causallm::kv16::Geometry;

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

// Per-block modes parsed from the container index: after header (52) +
// (gamma + rotary) words, each entry is offset(8) length(4) rows(2) mode(1)
// width(1) padding(2).
static std::map<uint8_t, size_t> mode_histogram(const std::vector<uint8_t> &e) {
  auto rd = [&](size_t off, unsigned n) {
    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i)
      v |= uint64_t(e[off + i]) << (8 * i);
    return v;
  };
  const size_t idx = 52 + (rd(28, 4) + rd(32, 4)) * 4;
  std::map<uint8_t, size_t> hist;
  for (size_t b = 0; b < rd(36, 4); ++b)
    ++hist[uint8_t(e[idx + b * 16 + 12])];
  return hist;
}

static void run_ab_case(const Geometry &g, unsigned pattern, size_t &cases,
                        size_t &failures, std::map<uint8_t, size_t> &modes) {
  const size_t rows = size_t(g.tokens) * g.heads;
  std::vector<uint16_t> key(rows * g.dimensions), value(rows * g.dimensions);
  for (size_t row = 0; row < rows; ++row) {
    // Residual width target per row: 0 -> zero residuals (sparse wins),
    // small -> narrow (fixed wins), 16 -> noise (raw wins), mix otherwise.
    unsigned target_bits;
    if (pattern == 0)
      target_bits = 0;
    else if (pattern == 1)
      target_bits = 1 + unsigned(rnd() % 8);
    else if (pattern == 2)
      target_bits = 16;
    else
      target_bits = row % 3 == 0 ? 0 : row % 3 == 1 ? 4 : 16;
    const size_t gamma_offset =
      g.gamma_bits.size() == g.dimensions ? 0 : (row % g.heads) * g.dimensions;
    for (unsigned d = 0; d < g.dimensions; ++d) {
      const uint16_t v = uint16_t(rnd());
      const uint16_t p =
        causallm::kv16::predict(v, g.gamma_bits[gamma_offset + d]);
      uint16_t residual = 0;
      if (target_bits)
        residual = uint16_t(rnd() & ((1u << target_bits) - 1u));
      value[row * g.dimensions + d] = v;
      key[row * g.dimensions + d] = uint16_t(p ^ residual);
    }
  }
  const size_t plane_bytes = rows * g.dimensions * 2;
  const auto *kp = reinterpret_cast<const uint8_t *>(key.data());
  const auto *vp = reinterpret_cast<const uint8_t *>(value.data());
  try {
    const auto e_new = causallm::kv16::encode(kp, vp, plane_bytes, g);
    const auto e_old = causalleg::kv16::encode(kp, vp, plane_bytes, g);
    ++cases;
    if (e_new != e_old) {
      ++failures;
      std::printf("MISMATCH: tokens=%llu heads=%u dims=%u block=%u pat=%u "
                  "new=%zu old=%zu\n",
                  (unsigned long long)g.tokens, g.heads, g.dimensions,
                  g.block_rows, pattern, e_new.size(), e_old.size());
      return;
    }
    const auto dec = causallm::kv16::decode(e_new.data(), e_new.size());
    bool ok = dec.key.size() == plane_bytes && dec.value.size() == plane_bytes;
    for (size_t i = 0; ok && i < plane_bytes / 2; ++i)
      ok = dec.key[i * 2] == uint8_t(key[i]) &&
           dec.key[i * 2 + 1] == uint8_t(key[i] >> 8) &&
           dec.value[i * 2] == uint8_t(value[i]) &&
           dec.value[i * 2 + 1] == uint8_t(value[i] >> 8);
    if (!ok) {
      ++failures;
      std::printf("ROUNDTRIP FAIL: tokens=%llu heads=%u dims=%u block=%u "
                  "pat=%u\n",
                  (unsigned long long)g.tokens, g.heads, g.dimensions,
                  g.block_rows, pattern);
    }
    for (const auto &kv : mode_histogram(e_new))
      modes[kv.first] += kv.second;
  } catch (const std::exception &ex) {
    ++failures;
    std::printf("EXCEPTION: %s (tokens=%llu heads=%u dims=%u block=%u)\n",
                ex.what(), (unsigned long long)g.tokens, g.heads, g.dimensions,
                g.block_rows);
  }
}

static int ab_test() {
  size_t cases = 0, failures = 0;
  std::map<uint8_t, size_t> modes;
  const unsigned tokens_list[] = {1, 3, 15, 16, 17, 32, 64, 129};
  const unsigned heads_list[] = {1, 2, 4, 8};
  const unsigned dims_list[] = {8, 64, 128, 256};
  const unsigned block_list[] = {1, 4, 16, 32};
  for (unsigned tokens : tokens_list)
    for (unsigned heads : heads_list)
      for (unsigned dims : dims_list)
        for (unsigned block : block_list)
          for (unsigned pattern = 0; pattern < 4; ++pattern) {
            Geometry g;
            g.tokens = tokens;
            g.heads = heads;
            g.dimensions = dims;
            g.block_rows = block;
            const bool per_head = rnd() & 1;
            for (unsigned i = 0; i < dims * (per_head ? heads : 1u); ++i)
              g.gamma_bits.push_back(uint32_t(rnd()));
            for (unsigned d = 0; d < dims; d += 2)
              if (rnd() & 1)
                g.rotary_dims.push_back(d);
            run_ab_case(g, pattern, cases, failures, modes);
          }
  std::printf("cases=%zu failures=%zu modes:", cases, failures);
  for (const auto &kv : modes)
    std::printf(" m%u=%zu", kv.first, kv.second);
  std::printf("\n%s\n", failures ? "A/B TEST FAILED" : "A/B BYTE-EXACT PASS");
  return failures ? 1 : 0;
}

// Every FP16 value x 4096 gammas: half fully random (exercise the
// inf/nan/subnormal fallbacks), half near 1.0 (realistic prefill range).
static int predict_test() {
  uint64_t s = 0x12345678abcdefull;
  auto next = [&s] {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  };
  long checked = 0, failures = 0;
  for (int gi = 0; gi < 4096; ++gi) {
    const uint32_t g = gi < 2048
                         ? uint32_t(next())
                         : 0x3f800000u + uint32_t(next() % 0x1000) - 0x800;
    for (unsigned v = 0; v < 65536; ++v) {
      ++checked;
      if (causallm::kv16::predict(uint16_t(v), g) !=
          causalleg::kv16::predict(uint16_t(v), g)) {
        if (++failures < 10)
          std::printf("PREDICT MISMATCH v=%04x g=%08x\n", v, g);
      }
    }
  }
  std::printf("checked=%ld failures=%ld -> %s\n", checked, failures,
              failures ? "FAILED" : "PASS");
  return failures ? 1 : 0;
}

static Geometry bench_geometry() {
  uint64_t s = 7;
  auto next = [&s] {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  };
  Geometry g;
  g.tokens = 2154;
  g.heads = 8;
  g.dimensions = 128;
  g.block_rows = 16;
  for (unsigned i = 0; i < 128; ++i)
    g.gamma_bits.push_back(0x3f000000u + uint32_t(next()));
  for (unsigned d = 0; d < 128; d += 2)
    g.rotary_dims.push_back(d);
  return g;
}

static int bench_test() {
  uint64_t s = 7;
  auto next = [&s] {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  };
  const Geometry g = bench_geometry();
  const size_t rows = size_t(g.tokens) * g.heads;
  std::vector<uint16_t> key(rows * 128), value(rows * 128);
  for (size_t row = 0; row < rows; ++row)
    for (unsigned d = 0; d < 128; ++d) {
      const uint16_t v = uint16_t(next());
      value[row * 128 + d] = v;
      const uint16_t p = causallm::kv16::predict(v, g.gamma_bits[d]);
      const unsigned bits = row % 4 == 0 ? 0 : row % 4 == 1 ? 5 : 16;
      key[row * 128 + d] =
        uint16_t(p ^ (bits ? unsigned(next()) & ((1u << bits) - 1u) : 0));
    }
  const auto *kp = reinterpret_cast<const uint8_t *>(key.data());
  const auto *vp = reinterpret_cast<const uint8_t *>(value.data());
  const size_t plane_bytes = rows * 256;
  volatile size_t sink = 0;
  for (const char *who : {"old", "new"}) {
    double ms = 0;
    for (unsigned rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto e = who[0] == 'o'
                       ? causalleg::kv16::encode(kp, vp, plane_bytes, g)
                       : causallm::kv16::encode(kp, vp, plane_bytes, g);
      const auto t1 = std::chrono::steady_clock::now();
      ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      sink += e.size();
    }
    std::printf("%s pat=mix: %.2f ms/encode, %.4f ms/token, sink=%zu\n", who,
                ms / 5, ms / 5 / g.tokens, size_t(sink));
  }
  (void)sink;
  return 0;
}

int main(int argc, char **argv) {
  const std::string what = argc > 1 ? argv[1] : "";
  if (what == "predict")
    return predict_test();
  if (what == "bench")
    return bench_test();
  return ab_test();
}
