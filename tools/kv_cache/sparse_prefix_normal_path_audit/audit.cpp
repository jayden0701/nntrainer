// SPDX-License-Identifier: Apache-2.0
#include "audit_support.h"
#include <cmath>
#include <cstdlib>
#include <fp16.h>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <new>

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
using json = nlohmann::json;
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
    const auto &x = a.layers[l], &y = b.layers[l];
    if (x.base != y.base || x.width != y.width ||
        x.pages.size() != y.pages.size())
      return false;
    for (size_t p = 0; p < x.pages.size(); ++p) {
      if (x.pages[p]->keySize() != y.pages[p]->keySize() ||
          x.pages[p]->valueSize() != y.pages[p]->valueSize())
        return false;
      if (std::memcmp(x.pages[p]->keyData(), y.pages[p]->keyData(),
                      x.pages[p]->keySize()) ||
          std::memcmp(x.pages[p]->valueData(), y.pages[p]->valueData(),
                      x.pages[p]->valueSize()))
        return false;
    }
  }
  return true;
}
static json asJson(const Counts &s) {
  return {{"allocations", s.allocations},
          {"requested_bytes", s.requested_bytes},
          {"payload_allocations", s.payload_allocations},
          {"payload_zero_bytes", s.payload_zero_bytes},
          {"key_initial_copy_bytes", s.key_initial_copy_bytes},
          {"value_copy_bytes", s.value_copy_bytes},
          {"visited_rows", s.visited_rows},
          {"skipped_rows", s.skipped_rows},
          {"shared_rows", s.shared_rows}};
}
int main(int argc, char **argv) {
  try {
    const std::array<std::pair<const char *, Gather>, 5> variants{
      {{"instrumented_baseline", auditBaseline},
       {"no_key_precopy", auditNoKeyPrecopy},
       {"plus_reserve_pages", auditReserve},
       {"plus_skip_discarded_rows", auditSkipRows},
       {"plus_share_exact_pages", auditShareExact}}};
    json report{
      {"hardware_exercised", false},
      {"faults_injected", false},
      {"counter_meaning",
       "source-level requested allocations/initialization/copies, not measured "
       "DRAM traffic or RSS"},
      {"fixture", "24K source ->20K retained, 30 layers, 5 global, sliding "
                  "window1024; D64/one KV head to scale host payload"}};
    const std::array<Runs, 6> plans{{{{0, 0, 64}, {128, 64, 128}},
                                     {{0, 0, 128}, {192, 128, 64}},
                                     {{64, 0, 192}},
                                     {{0, 0, 65}, {129, 65, 127}},
                                     {{0, 0, 63}, {128, 63, 128}},
                                     {{0, 0, 64}, {129, 64, 127}}}};
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
          for (const auto &v : variants) {
            auto actual = v.second(f.history, f.runs, f.dims, f.rope);
            if (!equal(*golden, *actual) ||
                checksum(*f.history.front()) != before)
              throw std::runtime_error(std::string("control mismatch: ") +
                                       v.first);
            ++controls;
          }
        }
      }
    report["small_fp16_fp32_bitwise_controls"] = true;
    report["control_comparisons"] = controls;
    auto f = fixture(false, true);
    const auto before = checksum(*f.history.front());
    auto golden =
      Manager::gatherSparsePrefix(f.history, f.runs, f.dims, f.rope);
    report["variants"] = json::array();
    for (const auto &v : variants) {
      counts = {};
      counting = true;
      std::shared_ptr<const PrefixSnapshot> actual;
      try {
        actual = v.second(f.history, f.runs, f.dims, f.rope);
        counting = false;
      } catch (...) {
        counting = false;
        throw;
      }
      const auto measured = counts;
      auto row = asJson(measured);
      row["name"] = v.first;
      row["bitwise_equal_to_production"] = equal(*golden, *actual);
      row["source_immutable"] = checksum(*f.history.front()) == before;
      if (row["bitwise_equal_to_production"] != true ||
          row["source_immutable"] != true)
        throw std::runtime_error(std::string("large fixture mismatch: ") +
                                 v.first);
      report["variants"].push_back(row);
    }
    report["passed"] = true;
    if (argc > 1) {
      std::ofstream out(argv[1]);
      out << report.dump(2) << '\n';
      if (!out)
        return 2;
    }
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    counting = false;
    std::cerr << e.what() << '\n';
    return 2;
  }
}
