// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstring>
#include <kv_cache_manager.h>
#include <vector>
using Manager = causallm::KVCacheManager;
using PrefixSnapshot = Manager::PrefixSnapshot;
using Runs = std::vector<nntrainer::kv::SparsePrefixRun>;
using Rope = std::vector<nntrainer::kv::SparseRope>;
using History = std::vector<std::shared_ptr<const PrefixSnapshot>>;
struct Counts {
  size_t allocations = 0, requested_bytes = 0;
  size_t payload_allocations = 0, payload_zero_bytes = 0;
  size_t key_initial_copy_bytes = 0, value_copy_bytes = 0;
  size_t visited_rows = 0, skipped_rows = 0, shared_rows = 0;
};
// The fixture invokes gather on one thread; no TLS wrapper is needed.
extern bool counting;
extern Counts counts;
inline void auditResize(std::vector<unsigned char> &v, size_t n) {
  if (counting) {
    counts.payload_zero_bytes += n > v.size() ? n - v.size() : 0;
    counts.payload_allocations += n > v.capacity();
  }
  v.resize(n);
}
inline void auditCopy(bool key, void *dst, const void *src, size_t n) {
  if (counting) {
    if (key)
      counts.key_initial_copy_bytes += n;
    else
      counts.value_copy_bytes += n;
  }
  std::memcpy(dst, src, n);
}
using Gather = std::shared_ptr<const PrefixSnapshot> (*)(
  const History &, const Runs &, const std::vector<unsigned> &, const Rope &);
std::shared_ptr<const PrefixSnapshot>
auditBaseline(const History &, const Runs &, const std::vector<unsigned> &,
              const Rope &);
std::shared_ptr<const PrefixSnapshot>
auditNoKeyPrecopy(const History &, const Runs &, const std::vector<unsigned> &,
                  const Rope &);
std::shared_ptr<const PrefixSnapshot>
auditReserve(const History &, const Runs &, const std::vector<unsigned> &,
             const Rope &);
std::shared_ptr<const PrefixSnapshot>
auditSkipRows(const History &, const Runs &, const std::vector<unsigned> &,
              const Rope &);
std::shared_ptr<const PrefixSnapshot>
auditShareExact(const History &, const Runs &, const std::vector<unsigned> &,
                const Rope &);
