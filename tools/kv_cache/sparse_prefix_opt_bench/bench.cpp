// SPDX-License-Identifier: Apache-2.0
/** Per-optimization gather benchmark.
 *
 * One templated gather enables the reviewed sparse-prefix optimizations
 * cumulatively. Every variant is compared bitwise against the unmodified
 * production function (extracted verbatim into golden.cpp by run.py) and
 * against the baseline variant, then measured on the large 24K -> 20K
 * fixture: allocation/copy counters while counting, wall-clock time while
 * not counting. Host-only: it does not exercise HTP hardware.
 */
#include "bench_support.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fp16.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <vector>

bool counting = false;
Counts counts;
void *operator new(std::size_t n) {
  if (void *p = std::malloc(n ? n : 1)) {
    if (counting) {
      ++counts.allocations;
      counts.requested_bytes += n;
    }
    return p;
  }
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }

struct Fixture {
  History history;
  Runs runs;
  std::vector<unsigned> dims;
  Rope rope;
};

static Fixture fixture(bool fp32, bool large) {
  const unsigned rows = large ? 24576 : 256, window = large ? 1024 : 64;
  const unsigned layers = large ? 30 : 2, globals = large ? 5 : 1;
  auto snapshot = std::make_shared<PrefixSnapshot>();
  snapshot->position = rows;
  snapshot->element_bytes = fp32 ? 4 : 2;
  snapshot->layers.resize(layers);
  Fixture f;
  f.history.push_back(snapshot);
  f.dims.assign(layers, 64);
  f.rope.resize(layers);
  for (unsigned l = 0; l < layers; ++l) {
    auto &dst = snapshot->layers[l];
    dst.base = l < globals ? 0 : rows - window;
    dst.width = 64;
    auto &rope = f.rope[l];
    rope.frequencies.resize(32, 0);
    for (unsigned d = 0; d < (l < globals ? 8u : 32u); ++d)
      rope.frequencies[d] = 1.0f / std::pow(10000.0f, float(2 * d) / 64);
    for (unsigned start = dst.base; start < rows; start += 64) {
      auto page = std::make_shared<PrefixSnapshot::Page>();
      page->key.resize(64 * 64 * snapshot->element_bytes);
      page->value.resize(page->key.size());
      for (unsigned t = 0; t < 64; ++t)
        for (unsigned d = 0; d < 64; ++d) {
          const float k =
            d == 0
              ? -0.0f
              : float(int(((start + t) * 17 + d * 13 + l * 3) % 2047) - 1023) *
                  0.0005f;
          const float v =
            float(int(((start + t) * 7 + d + l) % 997) - 498) * 0.001f;
          const size_t off = (t * 64 + d) * snapshot->element_bytes;
          if (fp32) {
            std::memcpy(page->key.data() + off, &k, 4);
            std::memcpy(page->value.data() + off, &v, 4);
          } else {
            const auto hk = nntrainer::compute_fp32_to_fp16(k),
                       hv = nntrainer::compute_fp32_to_fp16(v);
            std::memcpy(page->key.data() + off, &hk, 2);
            std::memcpy(page->value.data() + off, &hv, 2);
          }
        }
      dst.pages.push_back(page);
    }
  }
  f.runs = large ? Runs{{0, 0, 4096}, {8192, 4096, 16384}}
                 : Runs{{0, 0, 64}, {128, 64, 128}};
  return f;
}

static uint64_t checksum(const PrefixSnapshot &s) {
  uint64_t h = 14695981039346656037ull;
  for (const auto &l : s.layers)
    for (const auto &p : l.pages)
      for (bool key : {true, false}) {
        const auto *b = key ? p->keyData() : p->valueData();
        const size_t n = key ? p->keySize() : p->valueSize();
        for (size_t i = 0; i < n; ++i)
          h = (h ^ b[i]) * 1099511628211ull;
      }
  return h;
}

static bool equal(const PrefixSnapshot &a, const PrefixSnapshot &b) {
  if (a.position != b.position || a.element_bytes != b.element_bytes ||
      a.layers.size() != b.layers.size())
    return false;
  for (size_t l = 0; l < a.layers.size(); ++l) {
    const auto &la = a.layers[l], &lb = b.layers[l];
    if (la.base != lb.base || la.width != lb.width ||
        la.pages.size() != lb.pages.size())
      return false;
    for (size_t p = 0; p < la.pages.size(); ++p)
      if (la.pages[p]->keySize() != lb.pages[p]->keySize() ||
          la.pages[p]->valueSize() != lb.pages[p]->valueSize() ||
          std::memcmp(la.pages[p]->keyData(), lb.pages[p]->keyData(),
                      la.pages[p]->keySize()) != 0 ||
          std::memcmp(la.pages[p]->valueData(), lb.pages[p]->valueData(),
                      la.pages[p]->valueSize()) != 0)
        return false;
  }
  return true;
}

/** Cumulative optimization flags. The all-false instantiation is the
 * unmodified baseline body; the all-true instantiation must stay byte-identical
 * to the production gather extracted into golden.cpp by run.py. */
template <bool Reserve, bool SkipRows, bool NoPrecopy, bool SharePages,
          bool RotaryTail>
static std::shared_ptr<const PrefixSnapshot>
gatherVariant(const History &history, const Runs &runs,
              const std::vector<unsigned> &head_dims, const Rope &rope) {
  using namespace nntrainer::kv;
  const size_t length = sparsePrefixLength(runs);
  if (history.empty() || !history.front() || !length ||
      length > std::numeric_limits<unsigned>::max())
    throw std::invalid_argument("Empty or oversized sparse prefix history");
  size_t expected = 0, source_end = 0;
  for (const auto &run : runs) {
    if (!run.count || run.destination != expected || run.source < source_end)
      throw std::invalid_argument("Unordered sparse prefix plan");
    expected = checkedAdd(expected, run.count);
    source_end = checkedAdd(run.source, run.count);
  }
  const auto &first = *history.front();
  const size_t layers = first.layers.size();
  if (head_dims.size() != layers || rope.size() != layers ||
      (first.element_bytes != 2 && first.element_bytes != 4))
    throw std::invalid_argument("Invalid sparse prefix geometry");
  auto result = std::make_shared<PrefixSnapshot>();
  result->position = static_cast<unsigned>(length);
  result->element_bytes = first.element_bytes;
  result->layers.resize(layers);
  for (size_t l = 0; l < layers; ++l) {
    auto &dst = result->layers[l];
    dst.width = first.layers[l].width;
    const size_t dim = head_dims[l];
    if (!dim || !dst.width || dst.width % dim)
      throw std::invalid_argument("Sparse prefix head geometry mismatch");
    rope[l].validate(dim);
    size_t retained = 0;
    for (const auto &snapshot : history) {
      if (!snapshot || snapshot->layers.size() != layers ||
          snapshot->element_bytes != result->element_bytes ||
          snapshot->layers[l].width != dst.width ||
          snapshot->layers[l].base > snapshot->position)
        throw std::invalid_argument("Incompatible sparse source history");
      retained = std::max<size_t>(retained, snapshot->position -
                                              snapshot->layers[l].base);
    }
    dst.base = length > retained ? static_cast<unsigned>(length - retained) : 0;
    const size_t row_bytes = checkedMultiply(dst.width, result->element_bytes);
    if constexpr (Reserve)
      dst.pages.reserve(checkedAdd(length - dst.base, 63) / 64);
    std::vector<float> key(dim);
    std::shared_ptr<PrefixSnapshot::Page> page;
    for (const auto &run : runs) {
      if (run.destination + run.count <= dst.base)
        continue;
      const auto rotation =
        run.source != run.destination && !rope[l].frequencies.empty()
          ? rope[l].rotation(run.source, run.destination)
          : std::vector<float>{};
      const size_t discarded =
        SkipRows && dst.base > run.destination ? dst.base - run.destination : 0;
      const size_t rotary_dims = 2 * rope[l].frequencies.size();
      for (size_t t = discarded; t < run.count; ++t) {
        const size_t src = run.source + t, pos = run.destination + t;
        if constexpr (!SkipRows) {
          if (pos < dst.base) {
            if (counting)
              ++counts.skipped_rows;
            continue;
          }
        }
        if (counting)
          ++counts.visited_rows;
        const PrefixSnapshot::Layer *found = nullptr;
        size_t found_end = 0;
        for (const auto &snapshot : history)
          if (snapshot->layers[l].base <= src && src < snapshot->position) {
            found = &snapshot->layers[l];
            found_end = snapshot->position;
            break;
          }
        if (!found)
          throw std::invalid_argument("Missing sparse sliding history");
        const size_t row = src - found->base;
        const auto &input = found->pages.at(row / 64);
        const size_t offset = checkedMultiply(row % 64, row_bytes);
        if (!input || input->keySize() < checkedAdd(offset, row_bytes) ||
            input->valueSize() < checkedAdd(offset, row_bytes))
          throw std::invalid_argument("Incomplete sparse source page");
        const size_t output_row = pos - dst.base;
        if (output_row % 64 == 0) {
          if constexpr (SharePages) {
            const size_t full = checkedMultiply(64, row_bytes);
            if (run.count - t >= 64 &&
                (run.source == run.destination ||
                 rope[l].frequencies.empty()) &&
                row % 64 == 0 && src + 64 <= found_end && !input->key_view &&
                input->keySize() >= full && input->valueSize() >= full) {
              dst.pages.push_back(input);
              if (counting) {
                ++counts.shared_pages;
                counts.shared_rows += 64;
              }
              page.reset();
              t += 63;
              continue;
            }
          }
          page = std::make_shared<PrefixSnapshot::Page>();
          const size_t bytes =
            checkedMultiply(std::min<size_t>(64, length - pos), row_bytes);
          auditResize(page->key, bytes);
          auditResize(page->value, bytes);
          dst.pages.push_back(page);
        }
        auto *out = page->key.data() + output_row % 64 * row_bytes;
        auditCopy(false, page->value.data() + output_row % 64 * row_bytes,
                  input->valueData() + offset, row_bytes);
        if (src == pos || rope[l].frequencies.empty()) {
          auditCopy(true, out, input->keyData() + offset, row_bytes);
          continue;
        }
        if constexpr (!NoPrecopy)
          auditCopy(true, out, input->keyData() + offset, row_bytes);
        const auto *src_key = NoPrecopy ? input->keyData() + offset : out;
        const size_t element_bytes = result->element_bytes;
        for (size_t h = 0; h < dst.width / dim; ++h) {
          const auto *head = src_key + h * dim * element_bytes;
          auto *out_head = out + h * dim * element_bytes;
          const size_t rotated = RotaryTail ? rotary_dims : dim;
          for (size_t d = 0; d < rotated; ++d) {
            if (element_bytes == 4)
              std::memcpy(&key[d], head + d * 4, 4);
            else {
              uint16_t bits;
              std::memcpy(&bits, head + d * 2, 2);
              key[d] = nntrainer::compute_fp16_to_fp32(bits);
            }
          }
          rope[l].reposition(key.data(), rotation);
          for (size_t d = 0; d < rotated; ++d) {
            if (!sparseFinite(key[d]))
              throw std::invalid_argument("Nonfinite sparse prefix key");
            if (element_bytes == 4)
              std::memcpy(out_head + d * 4, &key[d], 4);
            else {
              const uint16_t bits = nntrainer::compute_fp32_to_fp16(key[d]);
              if (!sparseFinite(nntrainer::compute_fp16_to_fp32(bits)))
                throw std::invalid_argument("Sparse prefix FP16 overflow");
              std::memcpy(out_head + d * 2, &bits, 2);
            }
          }
          if constexpr (RotaryTail) {
            for (size_t d = rotary_dims; d < dim; ++d) {
              float value;
              if (element_bytes == 4)
                std::memcpy(&value, head + d * 4, 4);
              else {
                uint16_t bits;
                std::memcpy(&bits, head + d * 2, 2);
                value = nntrainer::compute_fp16_to_fp32(bits);
              }
              if (!sparseFinite(value))
                throw std::invalid_argument("Nonfinite sparse prefix key");
              std::memcpy(out_head + d * element_bytes,
                          head + d * element_bytes, element_bytes);
            }
          }
        }
      }
    }
  }
  return result;
}

struct Variant {
  const char *name;
  std::shared_ptr<const PrefixSnapshot> (*gather)(const History &, const Runs &,
                                                  const std::vector<unsigned> &,
                                                  const Rope &);
};

template <bool R, bool S, bool N, bool P, bool T>
constexpr Variant make(const char *name) {
  return {name, gatherVariant<R, S, N, P, T>};
}

static const std::vector<Variant> &variants() {
  static const std::vector<Variant> list{
    make<false, false, false, false, false>("baseline"),
    make<true, false, false, false, false>("1_pages_reserve"),
    make<true, true, false, false, false>("2_skip_discarded_rows"),
    make<true, true, true, false, false>("3_no_key_precopy"),
    make<true, true, true, true, false>("4_share_exact_pages"),
    make<true, true, true, true, true>("5_rotary_tail_bitcopy")};
  return list;
}

static std::string countsJson(const Counts &c) {
  std::ostringstream o;
  o << "{\"allocations\":" << c.allocations
    << ",\"requested_bytes\":" << c.requested_bytes
    << ",\"payload_allocations\":" << c.payload_allocations
    << ",\"payload_zero_bytes\":" << c.payload_zero_bytes
    << ",\"key_initial_copy_bytes\":" << c.key_initial_copy_bytes
    << ",\"value_copy_bytes\":" << c.value_copy_bytes
    << ",\"visited_rows\":" << c.visited_rows
    << ",\"skipped_rows\":" << c.skipped_rows
    << ",\"shared_pages\":" << c.shared_pages
    << ",\"shared_rows\":" << c.shared_rows << "}";
  return o.str();
}

int main(int argc, char **argv) {
  try {
    const size_t reps = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 12;
    const size_t warmups = 2;
    std::ostringstream report;
    report << "{\"counting_meaning\":\"source-level requested "
              "allocations/initialization/copies and loop iterations, not "
              "measured DRAM traffic or RSS\","
           << "\"hardware_exercised\":false,\"faults_injected\":false,"
           << "\"fixture_large\":\"24K source -> 20K retained, 30 layers, 5 "
              "global, sliding window 1024, D64/one KV head\","
           << "\"fixture_small\":\"256 rows, 2 layers, six run plans, "
              "fp16/fp32 x split/interleaved/no-RoPE\",";
    static const std::array<Runs, 6> plans{{{{{0, 0, 64}, {128, 64, 128}}},
                                            {{{0, 0, 128}, {192, 128, 64}}},
                                            {{{64, 0, 192}}},
                                            {{{0, 0, 65}, {129, 65, 127}}},
                                            {{{0, 0, 63}, {128, 63, 128}}},
                                            {{{0, 0, 64}, {129, 64, 127}}}}};
    size_t controls = 0;
    for (bool fp32 : {false, true})
      for (unsigned mode = 0; mode < 3; ++mode) {
        auto f = fixture(fp32, false);
        const auto before = checksum(*f.history.front());
        for (auto &r : f.rope) {
          if (mode == 1)
            r.interleaved = true;
          if (mode == 2)
            r.frequencies.clear();
        }
        for (const auto &plan : plans) {
          f.runs = plan;
          auto golden =
            Manager::gatherSparsePrefix(f.history, f.runs, f.dims, f.rope);
          for (const auto &v : variants()) {
            auto actual = v.gather(f.history, f.runs, f.dims, f.rope);
            if (!equal(*golden, *actual) ||
                checksum(*f.history.front()) != before)
              throw std::runtime_error(std::string("control mismatch: ") +
                                       v.name);
            ++controls;
          }
        }
      }
    report << "\"small_fp16_fp32_bitwise_controls\":true,"
           << "\"control_comparisons\":" << controls << ",";
    auto f = fixture(false, true);
    const auto before = checksum(*f.history.front());
    auto golden =
      Manager::gatherSparsePrefix(f.history, f.runs, f.dims, f.rope);
    for (const auto &v : variants()) {
      auto actual = v.gather(f.history, f.runs, f.dims, f.rope);
      if (!equal(*golden, *actual) || checksum(*f.history.front()) != before)
        throw std::runtime_error(std::string("large fixture mismatch: ") +
                                 v.name);
    }
    report << "\"large_bitwise_matches_production\":true,";
    report << "\"variants\":[";
    for (size_t vi = 0; vi < variants().size(); ++vi) {
      const auto &v = variants()[vi];
      counts = {};
      counting = true;
      std::shared_ptr<const PrefixSnapshot> hold;
      try {
        hold = v.gather(f.history, f.runs, f.dims, f.rope);
        counting = false;
      } catch (...) {
        counting = false;
        throw;
      }
      hold.reset();
      std::vector<double> ms;
      for (size_t r = 0; r < warmups + reps; ++r) {
        const auto start = std::chrono::steady_clock::now();
        auto out = v.gather(f.history, f.runs, f.dims, f.rope);
        const double elapsed = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        if (r >= warmups)
          ms.push_back(elapsed);
      }
      std::sort(ms.begin(), ms.end());
      const double total = std::accumulate(ms.begin(), ms.end(), 0.0);
      if (vi)
        report << ",";
      report << "{\"name\":\"" << v.name
             << "\",\"counters\":" << countsJson(counts)
             << ",\"gather_ms_min\":" << ms.front()
             << ",\"gather_ms_median\":" << ms[ms.size() / 2]
             << ",\"gather_ms_mean\":" << total / ms.size() << "}";
    }
    report << "]}";
    const std::string text = report.str();
    if (argc > 2) {
      std::ofstream out(argv[2]);
      out << text << '\n';
      if (!out)
        return 2;
    }
    std::cout << text << '\n';
    return 0;
  } catch (const std::exception &e) {
    counting = false;
    std::cerr << e.what() << '\n';
    return 2;
  }
}
