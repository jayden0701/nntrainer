// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  KV Cache Manager implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_manager.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include <fp16.h>
#include <memory_data.h>

#include <basic_planner.h>
#include <engine.h>
#include <env_compat.h>
#include <mem_allocator.h>
#include <nntrainer_log.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_mem_allocator.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace causallm {

namespace {
/** Shared slabs whose RPC registration and completion stay with the backend. */
class SharedSlabAllocator final : public nntrainer::MemAllocator {
public:
  SharedSlabAllocator(KVCacheManager::SharedAlloc allocate,
                      KVCacheManager::SharedFree release) :
    allocate_(std::move(allocate)), release_(std::move(release)) {}
  void alloc(void **ptr, size_t bytes, size_t alignment) override {
    *ptr = allocate_(bytes);
    if (!*ptr)
      throw std::bad_alloc();
    if (reinterpret_cast<uintptr_t>(*ptr) % alignment) {
      release_(*ptr);
      *ptr = nullptr;
      throw std::invalid_argument("Shared KV allocation is not aligned");
    }
  }
  void free(void *ptr) override { release_(ptr); }
  bool isDeviceVisible() const override { return true; }
  bool isSVM() const override { return false; }
  std::string getName() override { return "htp-shared"; }

private:
  KVCacheManager::SharedAlloc allocate_;
  KVCacheManager::SharedFree release_;
};
} // namespace

#if defined(__linux__)
/**
 * CPU inference leases page-aligned KV buffers from a bounded pool. Each
 * tensor keeps its own MemoryData so graph views and aliases follow the
 * active buffer during attention. Outside the lease the tensor points back
 * to its original mapping. The worker reads buffers without rebinding them.
 */
class KVFlashStorage {
public:
  KVFlashStorage(const std::string &directory, unsigned layers,
                 unsigned lookahead) :
    lookahead_(lookahead),
    layers_(layers),
    buffers_(std::min<size_t>(lookahead, layers - 1) + 1) {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
      throw std::runtime_error("KV FSU: cannot determine page size");
    page_ = static_cast<size_t>(page);
    // Optional stage timing for the device experiments. A disabled run keeps
    // the original wait, copy and sync paths untouched.
    timing_ = std::getenv("NNTR_KV_TIMING") != nullptr;
    if (const char *budget = std::getenv("NNTR_KV_RESIDENT_BYTES")) {
      // Owner retention budget in DRAM bytes. Owners kept in the pool skip
      // the full-history read on the next step; the others still stream.
      resident_budget_ = std::strtoull(budget, nullptr, 10);
      // Pool buffers are the working copy now, so a flush can be given the
      // durability a write-through store pays per write in buffered mode.
      flush_sync_ = std::getenv("NNTR_KV_FLUSH_SYNC") != nullptr;
      // layers_.size() - 1 would underflow for a manager with no layers.
      const size_t slack =
        layers_.empty() ? 0 : std::min<size_t>(lookahead_, layers_.size() - 1);
      buffers_.resize(layers_.size() + slack + 1);
    }

    std::string path = directory + "/nntr-kv-XXXXXX";
    fd_ = mkostemp(path.data(), O_CLOEXEC);
    if (fd_ < 0)
      fail("create backing file");
    try {
      if (unlink(path.c_str()))
        fail("unlink backing file");
      if (lookahead_ != 0)
        prefetch_worker_ = std::make_unique<IOWorker>();
    } catch (...) {
      close(fd_);
      throw;
    }
  }

  ~KVFlashStorage() {
    try {
      waitWrite();
    } catch (...) {
      // Destructors cannot report a pending I/O error.
    }
    write_worker_.reset();
    prefetch_worker_.reset();
    value_worker_.reset();
    close(fd_);
  }

  void addLayer(unsigned layer, nntrainer::Tensor &k, nntrainer::Tensor &v) {
    const size_t bytes = alignUp(k.bytes());
    if (bytes > (static_cast<size_t>(std::numeric_limits<off_t>::max()) -
                 next_offset_) /
                  2)
      throw std::overflow_error("KV FSU: backing file offset overflow");

    auto &entry = layers_[layer];
    entry.owner = layer;
    entry.offset = next_offset_;
    entry.capacity = bytes;
    for (auto *tensor : {&k, &v}) {
      auto data = std::make_shared<Mapping>(bytes);
      const auto plane = tensor == &v;
      entry.bindings[plane] = data.get();
      tensor->setData(data, 0, false);
    }
    next_offset_ += 2 * bytes;

    // Try direct I/O before the first inference. Some filesystems accept the
    // flag but refuse direct writes to encrypted or compressed files.
    if (entry.offset == 0) {
      // NNTR_KV_BUFFERED skips the probe so an experiment can compare I/O
      // policies on the buffered path, where every write goes through the
      // page cache and a sync is what makes them durable.
      if (!std::getenv("NNTR_KV_BUFFERED")) {
        const int flags = fcntl(fd_, F_GETFL);
        if (flags >= 0 && fcntl(fd_, F_SETFL, flags | O_DIRECT) == 0) {
          ssize_t n;
          do {
            n = pwrite(fd_, entry.bindings[0]->address(), page_, 0);
          } while (n < 0 && errno == EINTR);
          direct_ = n == static_cast<ssize_t>(page_);
          if (!direct_ && fcntl(fd_, F_SETFL, flags) != 0)
            fail("disable unsupported direct I/O");
          if (madvise(entry.bindings[0]->address(), page_, MADV_DONTNEED))
            fail("release direct I/O probe");
        }
      }
      // Retained owners make the pool buffer the working copy, so small
      // writes can mark dirty ranges and flush when the owner leaves the
      // pool. NNTR_KV_WRITE_THROUGH restores the write-at-every-store path.
      write_back_ =
        resident_budget_ != 0 && !std::getenv("NNTR_KV_WRITE_THROUGH");
      if (direct_ && lookahead_ != 0) {
        try {
          if (std::getenv("NNTR_KV_ASYNC_WRITE"))
            write_worker_ = std::make_unique<IOWorker>();
          value_worker_ = std::make_unique<IOWorker>();
        } catch (const std::system_error &) {
          // A process with no spare threads can still use serial reads.
        }
      }
    }
  }

  void alias(unsigned layer, unsigned source) {
    layers_[layer].owner = layers_[source].owner;
    has_aliases_ = true;
  }

  void begin(unsigned layer) {
    wait();
    load(layer);
    const unsigned end =
      layer + 1 + std::min<unsigned>(lookahead_, layers_.size() - layer - 1);
    if (layer + 1 == end)
      return;
    prefetch(layer + 1, end, layers_[layer].owner);
  }

  void load(unsigned layer, bool bind = true) {
    if (bind && write_owner_ == layers_[layer].owner)
      waitWrite();
    auto &entry = layers_[layers_[layer].owner];
    if (!entry.buffer) {
      entry.buffer = &acquire(entry.capacity, entry.saved);
      try {
        readPlanes(entry);
      } catch (...) {
        releaseBuffer(entry);
        throw;
      }
      read_bytes_ += 2 * entry.saved;
    }
    if (bind) {
      for (unsigned plane = 0; plane < 2; ++plane)
        entry.bindings[plane]->setAddr(entry.buffer->planes[plane]->address());
    }
  }

  void store(unsigned layer, size_t start, size_t end, bool replace = false) {
    if (layers_[layer].owner != layer)
      return;
    auto &entry = layers_[layer];
    const size_t first = start / page_ * page_;
    const size_t last = alignUp(end);
    if (last > entry.capacity || first > last)
      throw std::out_of_range("KV FSU: write outside layer capacity");
    account(*entry.buffer, std::max(entry.buffer->resident, last));
    const size_t bytes = last - first;
    if (write_back_ && !replace) {
      // The retained buffer holds the live rows; persist when the owner
      // leaves the pool. Snapshot restores still write through.
      entry.dirty.push_back({first, last});
      ++deferred_writes_;
      entry.saved = std::max(entry.saved, last);
      write_bytes_ += 2 * (last - first);
      return;
    }
    // Serialize dirty ranges even when a later write uses the sync path.
    waitWrite();
    // Small decode writes can contend with prefetch on mobile storage.
    // Pipeline larger prefill chunks; leave decode writes synchronous.
    // Snapshot restore must report write errors before returning.
    if (write_worker_ && !has_aliases_ && !replace && bytes >= 64 * 1024) {
      if (write_capacity_ < bytes) {
        write_planes_[0] = std::make_unique<Mapping>(bytes);
        write_planes_[1] = std::make_unique<Mapping>(bytes);
        write_capacity_ = bytes;
      }
      uint64_t t0 = 0;
      if (timing_)
        t0 = now_ns();
      for (unsigned p = 0; p < 2; ++p)
        std::memcpy(write_planes_[p]->address(),
                    entry.buffer->planes[p]->address() + first, bytes);
      if (timing_)
        copy_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
      const auto offset = entry.offset + first;
      const auto capacity = entry.capacity;
      write_owner_ = layer;
      write_pending_ = write_worker_->submit(
        std::packaged_task<void()>([this, offset, capacity, bytes] {
          for (unsigned p = 0; p < 2; ++p)
            transferRaw(write_planes_[p]->address(), offset + p * capacity,
                        bytes, true);
        }));
      ++async_writes_;
    } else {
      for (unsigned p = 0; p < 2; ++p)
        transfer(entry, p, first, bytes, true);
    }
    if (!direct_ && last != first)
      syncBackingFile();
    entry.saved = replace ? last : std::max(entry.saved, last);
    write_bytes_ += 2 * (last - first);
  }

  void evict(unsigned layer) {
    auto &entry = layers_[layers_[layer].owner];
    if (!entry.buffer)
      return;
    // The tensor binding always returns to the original mapping address, so
    // the graph keeps seeing the pristine plane it was built with.
    const auto restore = [&entry] {
      for (unsigned plane = 0; plane < 2; ++plane)
        entry.bindings[plane]->setAddr(entry.bindings[plane]->address());
    };
    if (entry.retained) {
      // The owner is already in the pool for the next step; this step wrote
      // new rows into the same buffer, so only the accounting changes.
      // store() already grew buffer.resident, so subtract the value this
      // owner contributed when it was retained, not the current one.
      retained_bytes_ -= 2 * entry.retained_resident;
      entry.retained = false;
    }
    if (resident_budget_ != 0 &&
        retained_bytes_ + 2 * entry.buffer->resident <= resident_budget_) {
      // Keep the owner in the pool so the next step skips its full-history
      // read.
      entry.retained = true;
      entry.retained_resident = entry.buffer->resident;
      retained_bytes_ += 2 * entry.buffer->resident;
      restore();
      return;
    }
    flushDirty(entry);
    for (unsigned plane = 0; plane < 2; ++plane) {
      restore();
      if (!direct_ && entry.saved != 0) {
        const auto off =
          static_cast<off_t>(entry.offset + plane * entry.capacity);
        posix_fadvise(fd_, off, static_cast<off_t>(entry.saved),
                      POSIX_FADV_DONTNEED);
      }
    }
    releaseBuffer(entry);
  }

  void finishStep() {
    const uint64_t t0 = timing_ ? now_ns() : 0;
    wait();
    waitWrite();
    if (timing_)
      drain_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    for (unsigned i = 0; i < layers_.size(); ++i)
      if (layers_[i].owner == i)
        evict(i);
  }

  void startStep(unsigned from) {
    // A snapshot load can leave a write pending before the next step.
    waitWrite();
    if (from == 0)
      reset();
    else if (failed_)
      throw std::runtime_error("KV FSU: start a new prompt after an I/O error");
    if (lookahead_ == 0)
      return;
    const auto first =
      std::find_if(layers_.begin(), layers_.end(),
                   [](const Layer &layer) { return layer.owner != inactive; });
    if (first != layers_.end() && layers_[first->owner].saved != 0) {
      const auto index = static_cast<unsigned>(first - layers_.begin());
      prefetch(index, index + 1, inactive);
    }
  }

  void abortStep() noexcept {
    failed_ = true;
    try {
      finishStep();
    } catch (...) {
      // Preserve the inference or I/O exception that triggered the abort.
    }
  }

  size_t retainedBytes() const { return retained_bytes_; }

  void reset() {
    // A new prompt must not retain the previous request's owners. The
    // bindings must return to their own mappings before the pool pages go.
    wait();
    waitWrite();
    for (auto &layer : layers_) {
      layer.dirty.clear();
      if (!layer.buffer)
        continue;
      for (unsigned plane = 0; plane < 2; ++plane)
        layer.bindings[plane]->setAddr(layer.bindings[plane]->address());
      layer.retained = false;
      releaseBuffer(layer);
    }
    for (auto &buffer : buffers_)
      discardTail(buffer, 0);
    failed_ = false;
    retained_bytes_ = 0;
    for (auto &layer : layers_) {
      layer.retained = false;
      layer.retained_resident = 0;
      layer.saved = 0;
    }
  }

  void report() const {
    std::fprintf(stderr,
                 "[kv-fsu] lookahead=%u direct_io=%d peak_kv_bytes=%zu "
                 "read_bytes=%zu write_bytes=%zu reused_buffers=%zu "
                 "parallel_reads=%zu async_writes=%zu staging_bytes=%zu "
                 "retained_bytes=%zu deferred_writes=%zu\n",
                 lookahead_, direct_, peak_bytes_.load(), read_bytes_.load(),
                 write_bytes_, reuse_count_, parallel_reads_, async_writes_,
                 2 * write_capacity_, retained_bytes_, deferred_writes_);
    if (!timing_)
      return;
    // Per-report deltas for one request, next to the totals since start.
    const Timing now{ready_wait_ns_.load(), write_wait_ns_.load(),
                     fdatasync_ns_.load(),  copy_ns_.load(),
                     io_read_ns_.load(),    io_write_ns_.load(),
                     drain_ns_.load()};
    std::fprintf(stderr,
                 "[kv-fsu-timing] step ready_wait_us=%.1f write_wait_us=%.1f "
                 "fdatasync_us=%.1f copy_us=%.1f io_read_us=%.1f "
                 "io_write_us=%.1f drain_us=%.1f | total "
                 "ready_wait_us=%.1f write_wait_us=%.1f fdatasync_us=%.1f "
                 "copy_us=%.1f io_read_us=%.1f io_write_us=%.1f "
                 "drain_us=%.1f\n",
                 us(now.ready - last_report_.ready),
                 us(now.write_wait - last_report_.write_wait),
                 us(now.fdatasync - last_report_.fdatasync),
                 us(now.copy - last_report_.copy),
                 us(now.io_read - last_report_.io_read),
                 us(now.io_write - last_report_.io_write),
                 us(now.drain - last_report_.drain), us(now.ready),
                 us(now.write_wait), us(now.fdatasync), us(now.copy),
                 us(now.io_read), us(now.io_write), us(now.drain));
    last_report_ = now;
  }

private:
  class IOWorker final {
  public:
    IOWorker() :
      thread_([this] {
        for (;;) {
          std::unique_lock<std::mutex> lock(mutex_);
          work_.wait(lock, [this] { return job_.valid() || stopping_; });
          if (!job_.valid())
            return;
          auto job = std::move(job_);
          lock.unlock();
          job();
        }
      }) {}

    ~IOWorker() {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
      }
      work_.notify_one();
      thread_.join();
    }

    std::future<void> submit(std::packaged_task<void()> task) {
      auto future = task.get_future();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = std::move(task);
      }
      work_.notify_one();
      return future;
    }

  private:
    std::mutex mutex_;
    std::condition_variable work_;
    std::packaged_task<void()> job_;
    bool stopping_ = false;
    std::thread thread_;
  };

  class Mapping final : public nntrainer::MemoryData {
  public:
    explicit Mapping(size_t size) :
      MemoryData(mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)),
      original_(getAddr<char>()),
      size_(size) {
      if (original_ == MAP_FAILED)
        fail("reserve KV address space");
    }
    ~Mapping() override { munmap(original_, size_); }
    char *address() const { return original_; }

  private:
    char *const original_;
    size_t size_;
  };

  struct Buffer {
    std::unique_ptr<Mapping> planes[2];
    size_t capacity = 0;
    size_t resident = 0;
    bool used = false;
  };

  static constexpr unsigned inactive = std::numeric_limits<unsigned>::max();
  struct Layer {
    unsigned owner = inactive;
    Mapping *bindings[2]{};
    size_t offset = 0;
    size_t capacity = 0;
    size_t saved = 0;
    Buffer *buffer = nullptr;
    bool retained = false;
    size_t retained_resident = 0;
    std::vector<std::pair<size_t, size_t>> dirty;
  };

  void flushDirty(Layer &entry) {
    if (entry.dirty.empty())
      return;
    std::sort(entry.dirty.begin(), entry.dirty.end());
    std::vector<std::pair<size_t, size_t>> merged;
    for (const auto &range : entry.dirty) {
      if (!merged.empty() && range.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, range.second);
      else
        merged.push_back(range);
    }
    // Clear only after every range reached the file: if a write throws, the
    // ledger still says which rows exist only in DRAM.
    for (const auto &range : merged)
      for (unsigned plane = 0; plane < 2; ++plane)
        transfer(entry, plane, range.first, range.second - range.first, true);
    entry.dirty.clear();
    // A write-through store syncs after each write when buffered; syncing
    // here is what makes batching and durability comparable in an A/B.
    if (flush_sync_ && !direct_)
      syncBackingFile();
  }

  void syncBackingFile() {
    const uint64_t t0 = timing_ ? now_ns() : 0;
    int rc;
    do {
      rc = fdatasync(fd_);
    } while (rc && errno == EINTR);
    if (timing_)
      fdatasync_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    if (rc)
      fail("sync backing file");
  }

  [[noreturn]] static void fail(const char *operation) {
    throw std::system_error(errno, std::generic_category(),
                            std::string("KV FSU: ") + operation);
  }

  size_t alignUp(size_t bytes) const {
    return (bytes + page_ - 1) / page_ * page_;
  }

  void prefetch(unsigned first, unsigned end, unsigned exclude) {
    pending_ = prefetch_worker_->submit(
      std::packaged_task<void()>([this, first, end, exclude] {
        for (unsigned next = first; next < end; ++next) {
          const auto owner = layers_[next].owner;
          if (owner != inactive && owner != exclude &&
              layers_[owner].saved != 0)
            load(next, false);
        }
      }));
  }

  void readPlanes(const Layer &entry) {
    // Independent positioned reads keep more than one large request in
    // flight, using the same two aligned planes already leased by this layer.
    constexpr size_t min_parallel_read = 1024 * 1024;
    if (!value_worker_ || entry.saved < min_parallel_read) {
      transfer(entry, 0, 0, entry.saved, false);
      transfer(entry, 1, 0, entry.saved, false);
      return;
    }
    auto value = value_worker_->submit(std::packaged_task<void()>(
      [this, &entry] { transfer(entry, 1, 0, entry.saved, false); }));
    try {
      transfer(entry, 0, 0, entry.saved, false);
    } catch (...) {
      // Do not release the value plane while its I/O is still running.
      value.wait();
      throw;
    }
    value.get();
    ++parallel_reads_;
  }

  void wait() {
    if (pending_.valid()) {
      const uint64_t t0 = timing_ ? now_ns() : 0;
      pending_.get();
      if (timing_)
        ready_wait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
  }

  void waitWrite() {
    if (write_pending_.valid()) {
      write_owner_ = inactive;
      const uint64_t t0 = timing_ ? now_ns() : 0;
      write_pending_.get();
      if (timing_)
        write_wait_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
  }

  Buffer &acquire(size_t capacity, size_t saved) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto available = std::find_if(
      buffers_.begin(), buffers_.end(), [capacity](const Buffer &buffer) {
        return !buffer.used && buffer.capacity == capacity;
      });
    if (available == buffers_.end()) {
      available = std::find_if(buffers_.begin(), buffers_.end(),
                               [](const Buffer &buf) { return !buf.used; });
      if (available == buffers_.end())
        throw std::logic_error("KV FSU: no available layer buffer");
      auto key = std::make_unique<Mapping>(capacity);
      auto value = std::make_unique<Mapping>(capacity);
      resident_bytes_ -= 2 * available->resident;
      available->planes[0] = std::move(key);
      available->planes[1] = std::move(value);
      available->capacity = capacity;
      available->resident = 0;
    } else if (available->resident != 0 && saved != 0) {
      ++reuse_count_;
    }

    // Recycled pages within saved are overwritten by pread. The uncached
    // tail must still enter the layer zero-filled (including after reset).
    if (saved < available->resident)
      discardTail(*available, saved);
    account(*available, saved);
    available->used = true;
    return *available;
  }

  void releaseBuffer(Layer &entry) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    entry.buffer->used = false;
    entry.buffer = nullptr;
  }

  void discardTail(Buffer &buffer, size_t bytes) {
    if (buffer.resident <= bytes)
      return;
    const size_t discarded = buffer.resident - bytes;
    for (const auto &plane : buffer.planes) {
      if (madvise(plane->address() + bytes, discarded, MADV_DONTNEED))
        fail("release unused KV buffer pages");
    }
    resident_bytes_ -= 2 * discarded;
    buffer.resident = bytes;
  }

  void account(Buffer &buffer, size_t bytes) {
    const size_t added = 2 * (bytes - buffer.resident);
    buffer.resident = bytes;
    const size_t now = resident_bytes_.fetch_add(added) + added;
    size_t peak = peak_bytes_.load();
    while (now > peak && !peak_bytes_.compare_exchange_weak(peak, now)) {
    }
  }

  void transfer(const Layer &entry, unsigned plane, size_t first, size_t bytes,
                bool write) {
    char *address = entry.buffer->planes[plane]->address() + first;
    transferRaw(address, entry.offset + plane * entry.capacity + first, bytes,
                write);
  }

  void transferRaw(char *address, size_t file_offset, size_t bytes,
                   bool write) {
    auto offset = static_cast<off_t>(file_offset);
    const uint64_t t0 = timing_ ? now_ns() : 0;
    while (bytes) {
      const ssize_t n = write ? pwrite(fd_, address, bytes, offset)
                              : pread(fd_, address, bytes, offset);
      if (n < 0 && errno == EINTR)
        continue;
      if (n < 0)
        fail(write ? "write KV pages" : "read KV pages");
      if (n == 0)
        throw std::runtime_error("KV FSU: incomplete backing file transfer");
      bytes -= static_cast<size_t>(n);
      address += n;
      offset += n;
    }
    if (timing_) {
      auto &counter = write ? io_write_ns_ : io_read_ns_;
      counter.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
  }

  const unsigned lookahead_;
  std::vector<Layer> layers_;
  std::vector<Buffer> buffers_;
  std::mutex buffer_mutex_;
  size_t reuse_count_ = 0;
  size_t page_ = 0;
  size_t next_offset_ = 0;
  int fd_ = -1;
  bool direct_ = false;
  bool failed_ = false;
  std::atomic<size_t> resident_bytes_{0}, peak_bytes_{0}, read_bytes_{0};
  size_t write_bytes_ = 0;
  size_t parallel_reads_ = 0;
  size_t async_writes_ = 0, write_capacity_ = 0;
  bool has_aliases_ = false;
  unsigned write_owner_ = inactive;
  std::unique_ptr<Mapping> write_planes_[2];
  std::unique_ptr<IOWorker> write_worker_;
  std::future<void> write_pending_;
  std::unique_ptr<IOWorker> prefetch_worker_;
  std::unique_ptr<IOWorker> value_worker_;
  std::future<void> pending_;

  // Stage timing state, only accumulated when the NNTR_KV_TIMING environment
  // variable is present. The write and value workers add from their threads,
  // so the counters are atomic. report() prints deltas since the last report
  // next to the totals.
  static double us(uint64_t ns) { return ns / 1000.0; }
  static uint64_t now_ns() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count());
  }
  struct Timing {
    uint64_t ready, write_wait, fdatasync, copy, io_read, io_write, drain;
  };
  bool timing_ = false;
  std::atomic<uint64_t> ready_wait_ns_{0}, write_wait_ns_{0}, fdatasync_ns_{0},
    copy_ns_{0}, io_read_ns_{0}, io_write_ns_{0}, drain_ns_{0};
  mutable Timing last_report_{};
  size_t resident_budget_ = 0;
  size_t retained_bytes_ = 0;
  size_t deferred_writes_ = 0;
  bool write_back_ = false;
  bool flush_sync_ = false;
};
#else
class KVFlashStorage {
public:
  KVFlashStorage(const std::string &, unsigned, unsigned) {
    throw std::invalid_argument("KV FSU requires Android or Linux");
  }
  void addLayer(unsigned, nntrainer::Tensor &, nntrainer::Tensor &) {}
  void alias(unsigned, unsigned) {}
  void begin(unsigned) {}
  void load(unsigned) {}
  void store(unsigned, size_t, size_t, bool = false) {}
  void evict(unsigned) {}
  void finishStep() {}
  void startStep(unsigned) {}
  void abortStep() noexcept {}
  void reset() {}
  void report() const {}
};
#endif

namespace {
class FlashSnapshotLease {
public:
  FlashSnapshotLease(KVFlashStorage *storage, unsigned layer) :
    storage_(storage), layer_(layer) {
    if (storage_)
      storage_->load(layer_);
  }

  ~FlashSnapshotLease() {
    if (storage_)
      storage_->evict(layer_);
  }

  FlashSnapshotLease(const FlashSnapshotLease &) = delete;
  FlashSnapshotLease &operator=(const FlashSnapshotLease &) = delete;

private:
  KVFlashStorage *storage_;
  unsigned layer_;
};
} // namespace

KVCacheManager::KVCacheManager() = default;
KVCacheManager::KVCacheManager(KVCacheManager &&) = default;

void KVCacheManager::setFlashStorage(std::string directory,
                                     unsigned lookahead) {
  if (isAllocated() || directory.empty())
    throw std::invalid_argument(
      "KV FSU: specify a backing directory before allocating KV");
  flash_directory_ = std::move(directory);
  flash_lookahead_ = lookahead;
}

void KVCacheManager::beginFlashLayer(unsigned layer) { flash_->begin(layer); }

void KVCacheManager::finishFlashLayer(unsigned layer, unsigned from,
                                      unsigned to) {
  const unsigned cap = getLayerCap(layer);
  const size_t row_bytes = layer_caches_[layer].key_cache.bytes() / cap;
  const unsigned count = std::min(to - from, cap);
  const unsigned start = count == cap ? 0 : from % cap;
  const unsigned first_rows = std::min(count, cap - start);
  flash_->store(layer, start * row_bytes, (start + first_rows) * row_bytes);
  if (first_rows < count)
    flash_->store(layer, 0, (count - first_rows) * row_bytes);
  flash_->evict(layer);
}

void KVCacheManager::beginFlashStep(unsigned from) {
  if (flash_)
    flash_->startStep(from);
}

void KVCacheManager::finishFlashStep() {
  if (flash_)
    flash_->finishStep();
}

void KVCacheManager::abortFlashStep() noexcept {
  if (flash_)
    flash_->abortStep();
}

size_t KVCacheManager::flashRetainedBytes() const {
  return flash_ ? flash_->retainedBytes() : 0;
}

void KVCacheManager::reportFlashStorage() const {
  if (flash_)
    flash_->report();
}

KVCacheManager::~KVCacheManager() { releaseShared(); }

KVCacheManager &KVCacheManager::operator=(KVCacheManager &&other) noexcept {
  if (this != &other) {
    // The tensors go first so nothing still points into the blocks the
    // allocator gets back.
    flash_.reset();
    layer_caches_.clear();
    flash_ = std::move(other.flash_);
    flash_directory_ = std::move(other.flash_directory_);
    flash_lookahead_ = other.flash_lookahead_;
    core_shared_ = std::move(other.core_shared_);
    core_shared_requested_ = other.core_shared_requested_;
    releaseShared();
    layer_caches_ = std::move(other.layer_caches_);
    active_layers_ = std::move(other.active_layers_);
    svm_pool_ = std::move(other.svm_pool_);
    cache_pos_ = other.cache_pos_;
    batch_size_ = other.batch_size_;
    max_seq_len_ = other.max_seq_len_;
    num_heads_kv_ = other.num_heads_kv_;
    head_dim_ = other.head_dim_;
    kv_width_ = other.kv_width_;
    kv_widths_ = std::move(other.kv_widths_);
    layer_kv_sources_ = std::move(other.layer_kv_sources_);
    layer_caps_ = std::move(other.layer_caps_);
    keep_ = std::move(other.keep_);
    bases_ = std::move(other.bases_);
    dropped_amax_k_ = std::move(other.dropped_amax_k_);
    dropped_amax_v_ = std::move(other.dropped_amax_v_);
    dtype_ = other.dtype_;
    format_ = other.format_;
    shared_retain_ = std::move(other.shared_retain_);
    prefix_views_ = std::move(other.prefix_views_);
    decode_offsets_ = std::move(other.decode_offsets_);
    shared_alloc_ = std::move(other.shared_alloc_);
    shared_free_ = std::move(other.shared_free_);
    shared_blocks_ = std::move(other.shared_blocks_);
    other.shared_blocks_.clear();
  }
  return *this;
}

void KVCacheManager::setCoreSharedStorage(bool enabled) {
  if (isAllocated())
    throw std::logic_error("Select core shared storage before allocate()");
  core_shared_requested_ = enabled;
}

void KVCacheManager::setSharedAllocator(SharedAlloc alloc, SharedFree release) {
  if (isAllocated()) {
    throw std::logic_error(
      "KVCacheManager::setSharedAllocator: call before allocate()");
  }
  if (static_cast<bool>(alloc) != static_cast<bool>(release)) {
    throw std::invalid_argument(
      "KVCacheManager::setSharedAllocator: need both alloc and release");
  }
  shared_alloc_ = std::move(alloc);
  shared_free_ = std::move(release);
}

nntrainer::Tensor
KVCacheManager::makeCacheTensor(const ml::train::TensorDim &dim) {
  if (shared_alloc_) {
    const size_t bytes = dim.getDataLen() * dim.getDataTypeSize();
    void *block = shared_alloc_(bytes);
    if (block) {
      shared_blocks_.push_back(block);
      // An unallocated tensor of this shape, then bound to the block: the
      // same setData() path the memory pool uses, so every view built from
      // it (getSharedDataTensor) resolves into the block.
      nntrainer::Tensor t(dim, false);
      t.setData(std::make_shared<nntrainer::MemoryData>(block), 0, false);
      return t;
    }
  }
  return nntrainer::Tensor(dim, true);
}

void KVCacheManager::releaseShared() {
  if (shared_free_) {
    for (void *block : shared_blocks_) {
      shared_free_(block);
    }
  }
  shared_blocks_.clear();
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  allocate(num_layers, batch_size, max_seq_len,
           std::vector<unsigned int>(num_layers, num_heads_kv * head_dim),
           dtype, format);

  num_heads_kv_ = num_heads_kv;
  head_dim_ = head_dim;
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              const std::vector<unsigned int> &kv_widths,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format,
                              const std::vector<unsigned int> &rows,
                              const std::vector<unsigned int> &keep) {
  if (num_layers == 0 || batch_size == 0 || max_seq_len == 0 ||
      kv_widths.size() != num_layers ||
      (!active_layers_.empty() && active_layers_.size() != num_layers) ||
      (!rows.empty() &&
       (rows.size() != num_layers || keep.size() != num_layers))) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: invalid layer, batch, or KV width count");
  }

  // [HTP window] rows / keep: a layer with fewer rows than max_seq_len keeps
  // its last keep rows contiguous at the front (compact()); the physical
  // capacity goes through the same per-layer cap the ring uses.
  keep_.assign(num_layers, 0u);
  bases_.assign(num_layers, 0u);
  decode_offsets_.assign(num_layers, 0u);
  dropped_amax_k_.assign(num_layers, {});
  dropped_amax_v_.assign(num_layers, {});
  if (!rows.empty()) {
    layer_caps_.assign(num_layers, 0u);
    for (unsigned int i = 0; i < num_layers; ++i) {
      if (rows[i] == 0 || rows[i] >= max_seq_len)
        continue;
      // a step must fit behind the window it attends to
      if (keep[i] == 0 || keep[i] >= rows[i])
        throw std::invalid_argument(
          "KVCacheManager::allocate: a windowed layer needs 0 < keep < rows");
      layer_caps_[i] = rows[i];
      keep_[i] = keep[i];
    }
  }

  for (auto kv_width : kv_widths) {
    if (kv_width == 0) {
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV widths must be > 0");
    }
  }

  batch_size_ = batch_size;
  max_seq_len_ = max_seq_len;
  num_heads_kv_ = 0;
  head_dim_ = 0;
  kv_width_ = kv_widths[0];
  kv_widths_ = kv_widths;
  dtype_ = dtype;
  format_ = format;
  cache_pos_ = 0;

  // Re-allocation: drop the old tensors before their blocks go back.
  flash_.reset();
  layer_caches_.clear();
  core_shared_.reset();
  releaseShared();
  layer_caches_.resize(num_layers);

  // [kv-share] Reject an unusable alias map BEFORE a single byte is requested:
  // a source that is not strictly earlier, or whose geometry differs, would
  // produce a cache that reads the wrong plane without ever crashing.
  validateKVSources(num_layers);
  if (!flash_directory_.empty()) {
    using DataType = ml::train::TensorDim::DataType;
    if (batch_size != 1 || shared_alloc_ || core_shared_requested_ ||
        !rows.empty() || dtype != DataType::FP16)
      throw std::invalid_argument("KV FSU requires batch-one CPU attention "
                                  "with FP16 externally bound KV");
    flash_ = std::make_unique<KVFlashStorage>(flash_directory_, num_layers,
                                              flash_lookahead_);
    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim dim({batch_size, 1, getLayerCap(i), kv_widths_[i]},
                               {format, dtype});
      const int source = getLayerKVSource(i);
      if (source >= 0) {
        aliasLayerCache(i, static_cast<unsigned>(source), dim);
        flash_->alias(i, static_cast<unsigned>(source));
      } else {
        auto &entry = layer_caches_[i];
        entry.key_cache = nntrainer::Tensor(dim, false);
        entry.value_cache = nntrainer::Tensor(dim, false);
        if (isLayerActive(i))
          flash_->addLayer(i, entry.key_cache, entry.value_cache);
      }
    }
    reportKVShare(num_layers, batch_size, dtype, "flash");
    return;
  }
  if (core_shared_requested_) {
    using namespace nntrainer::kv;
    using DataType = ml::train::TensorDim::DataType;
    if (!shared_alloc_ || !shared_free_ ||
        (dtype != DataType::FP32 && dtype != DataType::FP16 &&
         dtype != DataType::UINT16) ||
        (!active_layers_.empty() &&
         std::find(active_layers_.begin(), active_layers_.end(), false) !=
           active_layers_.end()))
      throw std::invalid_argument(
        "Core shared KV requires a shared allocator and active float planes");
    CacheSpec spec;
    for (unsigned int i = 0; i < num_layers; ++i) {
      LayerSpec layer;
      layer.batch = batch_size;
      layer.head_dim = kv_widths[i];
      layer.max_tokens = max_seq_len;
      layer.capacity = getLayerCap(i);
      layer.source = getLayerKVSource(i);
      layer.encoding =
        dtype == DataType::FP32 ? Encoding::Float32 : Encoding::Float16;
      layer.storage_type = dtype == DataType::FP32   ? StorageType::Float32
                           : dtype == DataType::FP16 ? StorageType::Float16
                                                     : StorageType::UInt16;
      spec.layers.push_back(layer);
    }
    auto core = std::make_unique<nntrainer::kv::KvCacheManager>();
    core->initialize(
      spec, std::make_shared<SharedSlabAllocator>(shared_alloc_, shared_free_));
    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim dim({batch_size, 1, getLayerCap(i), kv_widths[i]},
                               {format, dtype});
      layer_caches_[i].key_cache = nntrainer::Tensor(dim, false);
      layer_caches_[i].value_cache = nntrainer::Tensor(dim, false);
      layer_caches_[i].key_cache.setData(core->keyMemory(i), 0, false);
      layer_caches_[i].value_cache.setData(core->valueMemory(i), 0, false);
    }
    const auto stats = core->stats();
    core_shared_ = std::move(core);
    std::fprintf(stderr,
                 "[kv-htp-core] owners=%zu bytes=%zu shared=1 paged=0\n",
                 stats.owner_layers, stats.backing_bytes);
    reportKVShare(num_layers, batch_size, dtype, "core-htp-shared");
    return;
  }
  // An all-convolution model needs only the tensor metadata. In particular,
  // do not ask a backend MemoryPool to plan or allocate an empty set of slots.
  if (!active_layers_.empty() &&
      std::none_of(active_layers_.begin(), active_layers_.end(),
                   [](bool active) { return active; })) {
    svm_pool_.reset();
    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim cache_dim(
        {batch_size, 1, getLayerCap(i), kv_widths_[i]}, {format, dtype});
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
    }
    return;
  }
  {
    // [kv-window-ring] one-time diagnostic, printed only when the ring is
    // actually engaged: KV rows allocated vs the full-max_seq baseline.
    bool ring_on = false;
    for (auto c : layer_caps_)
      if (c > 0) {
        ring_on = true;
        break;
      }
    if (ring_on) {
      const size_t es =
        (dtype == ml::train::TensorDim::DataType::FP16) ? 2u : 4u;
      size_t rows = 0, bytes = 0;
      for (unsigned int i = 0; i < num_layers; ++i) {
        rows += getLayerCap(i);
        bytes += (size_t)batch_size * getLayerCap(i) * kv_widths_[i] * es * 2u;
      }
      ml_logi("[kvcache] window-ring: %u layers, KV rows %zu (vs %zu full), "
              "%.1f MB",
              num_layers, rows, (size_t)num_layers * max_seq_len,
              bytes / (1024.0 * 1024.0));
    }
  }

  // GPU-resident KV cache: on the OpenCL gpu engine, unless
  // NNTR_GPU_SVM_POOL=0, and when the gpu-svm allocator is available, allocate
  // the per-layer K/V from an SVM MemoryPool so their MemoryData reports
  // isSVM()=true. That is the precondition for mha_core's GPU flash attention
  // path (svm_ok); without it attention falls back to the host (CPU) GEMM and
  // is ~60x slower.
  // [engine=gpu fold] the SVM-resident pool is now the default — but ONLY for
  // the OpenCL gpu engine. On a CUDA build the gpu-svm (OpenCL) allocator is
  // also registered, so without the engine guard a cuda run would wrongly bind
  // the KV cache to it; the #if ENABLE_CUDA block below owns the cuda-uvm KV
  // cache. The env proxy (NNTR_GPU_SVM_POOL set ⇒ OpenCL) used to carry that
  // distinction; now it is explicit. NNTR_GPU_SVM_POOL=0 reverts to a host KV
  // cache.
  std::shared_ptr<nntrainer::MemAllocator> svm_alloc;
  const char *_svm_pool_env = std::getenv("NNTR_GPU_SVM_POOL");
  const std::string _eng = nntr_engine_env();
  const bool svm_pool_on = !_svm_pool_env || std::atoi(_svm_pool_env) != 0;
  const bool gpu_engine = _eng != "cpu" && _eng != "cuda";
  if (svm_pool_on && gpu_engine) {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    auto it = allocs.find("gpu");
    if (it != allocs.end() && it->second &&
        it->second->getName() == "gpu-svm") {
      svm_alloc = it->second;
    }
  }
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // CUDA UVM-resident KV cache: route the per-layer K/V through the cuda-uvm
  // (cudaMallocManaged) allocator so the cache is device-accessible. That lets
  // GPU attention read it without the per-call host->device mirror and lets GPU
  // RoPE write K straight into the device cache -- the precondition for a fully
  // on-GPU decode chain. Same pooled path as the OpenCL SVM cache above.
  // Opt-in and VALUE-checked: any value other than one starting with '0'
  // enables it; unset or =0 keeps a plain-host KV cache (needed on WDDM,
  // where the UVM setZero host-faults / attention can't reach a managed
  // cache).
  // Device-resident KV (NNTR_CUDA_KV_DEV=1, opt-in): the cache lives in
  // cudaMalloc DEVICE memory instead of UVM/pinned. WDDM (cMA==0) campaign
  // tier: managed KV hangs (remigration storm) and pinned KV pays PCIe on
  // every attention read; device-resident reads at VRAM speed. Steady state
  // has zero host KV-byte touches on the SAFE profile (GPU rope writes K,
  // GPU scalar-mul writes V -- the V-copy gate auto-routes for a device-only
  // destination -- and split-KV flash reads); setZero/save/load stage through
  // cuda::device_memset0/copy_any. Falls back to the UVM path if the cuda
  // engine is not registered. Takes precedence over NNTR_CUDA_KV_UVM.
  const char *kv_dev = std::getenv("NNTR_CUDA_KV_DEV");
  if (!svm_alloc && kv_dev != nullptr && kv_dev[0] == '1') {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    if (allocs.find("cuda") != allocs.end()) {
      svm_alloc =
        std::make_shared<nntrainer::CudaMemAllocator>(/*device_only=*/true);
    }
  }

  const char *kv_uvm = std::getenv("NNTR_CUDA_KV_UVM");
  if (!svm_alloc && kv_uvm != nullptr && kv_uvm[0] != '0') {
    auto allocs = nntrainer::Engine::Global().getAllocators();
    auto it = allocs.find("cuda");
    if (it != allocs.end() && it->second &&
        it->second->getName() == "cuda-uvm") {
      svm_alloc = it->second;
    }
  }
#endif

  const size_t elem_size =
    (dtype == ml::train::TensorDim::DataType::FP16) ? 2u : 4u;

  if (svm_alloc) {
    svm_pool_ = std::make_shared<nntrainer::MemoryPool>(svm_alloc);
    // [kv-share] A layer that aliases an earlier layer's K/V requests NO
    // token -- it gets no bytes of its own, which is the whole point. Index
    // the tokens per layer (0 = "none"; MemoryPool tokens start at 1) instead
    // of the old 2*i / 2*i+1 arithmetic, which cannot survive the gaps.
    std::vector<unsigned int> ktok(num_layers, 0u), vtok(num_layers, 0u);
    // All caches are live for the whole run; BasicPlanner gives each its own
    // (non-overlapping) region so the total pool is the sum. Size each region
    // per-layer so models with non-uniform KV widths stay correct.
    for (unsigned int i = 0; i < num_layers; ++i) {
      if (!isLayerActive(i) || isLayerKVAliased(i))
        continue;
      // [kv-window-ring] a ring layer only needs its Wcap rows.
      const size_t bytes =
        (size_t)batch_size * getLayerCap(i) * kv_widths_[i] * elem_size;
      ktok[i] = svm_pool_->requestMemory(bytes, 1, 2); // key
      vtok[i] = svm_pool_->requestMemory(bytes, 1, 2); // value
    }
    svm_pool_->planLayout(nntrainer::BasicPlanner());
    svm_pool_->allocate();

    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim cache_dim(
        {batch_size, 1, getLayerCap(i), kv_widths_[i]}, {format, dtype});
      if (!isLayerActive(i)) {
        layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
        layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
        continue;
      }
      // [kv-share] The source is strictly earlier, so its storage is already
      // in place -- one pass suffices, no second fixup loop. The source's
      // plane was zeroed when the source itself was set up, so the sharer must
      // NOT be zeroed again: it is the same memory, and a second pass over it
      // is a wasted device kernel (and, on the managed path, a wasted
      // migration).
      const int src = getLayerKVSource(i);
      if (src >= 0) {
        aliasLayerCache(i, static_cast<unsigned int>(src), cache_dim);
        continue;
      }
      layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].key_cache.setData(svm_pool_->getMemory(ktok[i]), 0,
                                         false);
      layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
      layer_caches_[i].value_cache.setData(svm_pool_->getMemory(vtok[i]), 0,
                                           false);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Zero a CUDA-resident KV cache ON THE DEVICE, never with a host memset.
      //
      // Device-only KV (NNTR_CUDA_KV_DEV) has no host mapping at all, so a host
      // setZero would dereference a cudaMalloc pointer. But the managed (UVM)
      // cache needs the same treatment for a performance reason: it IS host
      // addressable, and a host memset faults every page of the cache into HOST
      // residency. The first GPU writer -- the RoPE that lands K in the cache
      // and the V copy -- then has to migrate the whole cache back across PCIe
      // page by page, which showed up as 15 ms of a 1K gemma4 prefill (K-write
      // 10.4 ms and V-write 4.8 ms per prefill, versus 0.5 ms each once the
      // pages start out on the device). Zeroing through the stream leaves the
      // pages where every consumer of this cache wants them.
      //
      // dev_accessible() is false unless the CUDA engine is actually selected,
      // so the OpenCL SVM pool keeps the host memset.
      auto zero_kv = [](nntrainer::Tensor &t) {
        void *ptr = (void *)t.getData<char>();
        const auto md = t.getMemoryData();
        const bool host_only = md && !md->isHostAddressable();
        if (host_only || nntrainer::cuda::dev_accessible(ptr)) {
          if (!nntrainer::cuda::device_memset0(ptr, t.bytes()))
            throw std::runtime_error(
              "KVCacheManager: device memset of the KV cache failed");
        } else {
          t.setZero();
        }
      };
      zero_kv(layer_caches_[i].key_cache);
      zero_kv(layer_caches_[i].value_cache);
#else
      layer_caches_[i].key_cache.setZero();
      layer_caches_[i].value_cache.setZero();
#endif
    }
  } else {
    for (unsigned int i = 0; i < num_layers; ++i) {
      ml::train::TensorDim cache_dim(
        {batch_size, 1, getLayerCap(i), kv_widths_[i]}, {format, dtype});
      if (!isLayerActive(i)) {
        layer_caches_[i].key_cache = nntrainer::Tensor(cache_dim, false);
        layer_caches_[i].value_cache = nntrainer::Tensor(cache_dim, false);
        continue;
      }
      // [kv-share] Same rule as the pool branch above: an aliased layer gets a
      // view, never an allocation. Tensor(dim, /*alloc=*/false) + setData()
      // shares the source's MemoryData shared_ptr, so the host buffer outlives
      // every view of it regardless of destruction order.
      const int src = getLayerKVSource(i);
      if (src >= 0) {
        aliasLayerCache(i, static_cast<unsigned int>(src), cache_dim);
        continue;
      }
      layer_caches_[i].key_cache = makeCacheTensor(cache_dim);
      layer_caches_[i].value_cache = makeCacheTensor(cache_dim);
      layer_caches_[i].key_cache.setZero();
      layer_caches_[i].value_cache.setZero();
    }
  }

  reportKVShare(num_layers, batch_size, dtype, svm_alloc ? "svm-pool" : "host");
}

void KVCacheManager::validateKVSources(unsigned int num_layers) const {
  if (layer_kv_sources_.empty())
    return; // no sharing declared -- every layer owns its cache (the default)

  if (layer_kv_sources_.size() != num_layers)
    throw std::invalid_argument(
      "KVCacheManager::allocate: setLayerKVSources() size (" +
      std::to_string(layer_kv_sources_.size()) + ") != num_layers (" +
      std::to_string(num_layers) + ")");

  for (unsigned int i = 0; i < num_layers; ++i) {
    const int src = layer_kv_sources_[i];
    if (src < 0)
      continue;

    // Strictly earlier: the allocation pass walks layers in order and aliases
    // onto storage that already exists. A forward or self reference would
    // alias an empty Tensor -- and an empty KV cache does not crash here, it
    // silently feeds zeros into attention.
    if (static_cast<unsigned int>(src) >= i)
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV source layer " + std::to_string(src) +
        " for layer " + std::to_string(i) + " must be strictly earlier");
    if (!isLayerActive(i) || !isLayerActive(static_cast<unsigned int>(src)))
      throw std::invalid_argument(
        "KVCacheManager::allocate: inactive layers cannot share KV storage");

    // Same physical plane means same physical geometry. KV width is
    // per-layer, so a source/sharer drift (e.g. a narrow layer pointed at a
    // wide one) is exactly the mistake this check exists to catch, and it is
    // cheap. Only width is checked here; a row-count (per-layer cap)
    // mismatch is caught by the full-dim check in aliasLayerCache().
    if (kv_widths_[i] != kv_widths_[static_cast<unsigned int>(src)])
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV alias geometry mismatch, layer " +
        std::to_string(i) + " (width " + std::to_string(kv_widths_[i]) +
        ") -> layer " + std::to_string(src) + " (width " +
        std::to_string(kv_widths_[static_cast<unsigned int>(src)]) + ")");
  }
}

void KVCacheManager::aliasLayerCache(unsigned int dst, unsigned int src,
                                     const ml::train::TensorDim &cache_dim) {
  auto &s = layer_caches_[src];

  // The source must already hold storage; validateKVSources() guarantees the
  // ordering, this catches the case where the source's own allocation failed
  // to attach memory. Never fall through to "leave the sharer empty".
  if (s.key_cache.getMemoryData() == nullptr ||
      s.value_cache.getMemoryData() == nullptr)
    throw std::runtime_error(
      "KVCacheManager::allocate: KV alias source layer " + std::to_string(src) +
      " has no storage (layer " + std::to_string(dst) + " would read nulls)");

  if (s.key_cache.getDim() != cache_dim || s.value_cache.getDim() != cache_dim)
    throw std::runtime_error(
      "KVCacheManager::allocate: KV alias dim mismatch, layer " +
      std::to_string(dst) + " -> layer " + std::to_string(src));

  // Share the source's MemoryData shared_ptr (NOT a fresh MemoryPool::
  // getMemory(), which mints a new MemoryData object every call): ownership is
  // refcounted, so the sharer keeps the plane alive and teardown order is
  // irrelevant on both the pool and the host branch. This is byte-for-byte the
  // same setData() the model's allocateAndBindKVCache() performs when it hands
  // a cache to a graph placeholder.
  layer_caches_[dst].key_cache = nntrainer::Tensor(cache_dim, false);
  layer_caches_[dst].key_cache.setData(s.key_cache.getMemoryData(),
                                       s.key_cache.getOffset(), false);
  layer_caches_[dst].value_cache = nntrainer::Tensor(cache_dim, false);
  layer_caches_[dst].value_cache.setData(s.value_cache.getMemoryData(),
                                         s.value_cache.getOffset(), false);

  // Post-condition, asserted rather than assumed: the sharer resolves to a
  // non-null address and it is the source's address.
  const void *dk = layer_caches_[dst].key_cache.getData<char>();
  const void *dv = layer_caches_[dst].value_cache.getData<char>();
  if (dk == nullptr || dv == nullptr || dk != s.key_cache.getData<char>() ||
      dv != s.value_cache.getData<char>())
    throw std::runtime_error(
      "KVCacheManager::allocate: KV alias did not land, layer " +
      std::to_string(dst) + " -> layer " + std::to_string(src));
}

void KVCacheManager::reportKVShare(unsigned int num_layers,
                                   unsigned int batch_size,
                                   ml::train::TensorDim::DataType dtype,
                                   const char *where) const {
  // [kv-share] Independent witness of the RESOLVED layer -> source map.
  // Emitted whenever sharing is declared (i.e. not gated on the window ring,
  // which can collapse to off), because "which layer reads whose K/V" is the
  // one fact an aliased cache makes impossible to infer after the fact -- and
  // a silent source/sharer disagreement produces fluent, wrong output with no
  // crash. It also states the bytes NOT allocated, so a memory measurement
  // never has to assume the aliasing engaged.
  //
  // STDERR, deliberately, and not ml_logi like the window-ring line above.
  // ml_logi is not a witness in a shipping build: under -D__LOGGING__ (which
  // is how this app is built) it goes to a logs/*.out file (NNTR_LOG_DIR, else
  // the user cache dir; CWD-relative on Android) that no run harness reads,
  // and in the non-__LOGGING__ DEBUG variant INFO goes to STDOUT -- which is
  // the generated-text stream, i.e. inside the slice a golden md5 hashes.
  // stderr is the channel the app already uses for run diagnostics and the only
  // one that is both visible and safe here.
  if (layer_kv_sources_.empty())
    return;

  const size_t es = (dtype == ml::train::TensorDim::DataType::FP16) ? 2u : 4u;
  size_t saved = 0;
  unsigned int n_ok = 0;
  std::vector<std::string> entries;
  for (unsigned int i = 0; i < num_layers; ++i) {
    const int src = getLayerKVSource(i);
    if (src < 0)
      continue;
    saved += (size_t)batch_size * max_seq_len_ * kv_widths_[i] * es * 2u;

    // Prove the alias landed instead of asserting it in prose: both planes
    // must resolve to a non-null address, and to the SAME address as the
    // source. getData() only computes MemoryData::addr + offset, so this is
    // safe for a device-only plane too -- nothing is dereferenced.
    const auto &d = layer_caches_[i];
    const auto &s = layer_caches_[static_cast<unsigned int>(src)];
    const void *dk = d.key_cache.getData<char>();
    const void *dv = d.value_cache.getData<char>();
    const bool ok = dk != nullptr && dv != nullptr &&
                    dk == s.key_cache.getData<char>() &&
                    dv == s.value_cache.getData<char>();
    if (ok)
      ++n_ok;

    entries.push_back(std::to_string(i) + "->" + std::to_string(src) +
                      (ok ? "" : "(BROKEN)"));
  }

  std::fprintf(stderr,
               "[kvcache] kv-share (%s): %u of %u layers alias an earlier "
               "layer's K/V, %.1f MB not allocated, "
               "alias pointer check %zu/%zu ok\n",
               where, static_cast<unsigned int>(entries.size()), num_layers,
               saved / (1024.0 * 1024.0), (size_t)n_ok, entries.size());
  for (size_t b = 0; b < entries.size(); b += 10) {
    std::string line;
    for (size_t j = b; j < entries.size() && j < b + 10; ++j)
      line += " " + entries[j];
    std::fprintf(stderr, "[kvcache] kv-share map:%s\n", line.c_str());
  }
  std::fflush(stderr);
}

void KVCacheManager::setPosition(unsigned int pos) {
  if (pos > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::setPosition: pos exceeds max_seq_len");
  }
  if (core_shared_)
    core_shared_->setPosition(pos);
  cache_pos_ = pos;
}

void KVCacheManager::advance(unsigned int step_size) {
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::advance: position would exceed max_seq_len");
  }
  setPosition(cache_pos_ + step_size);
}

void KVCacheManager::reset() {
  if (flash_)
    flash_->reset();
  for (unsigned l = 0; l < prefix_views_.size(); ++l)
    freezePrefixViews(l);
  if (core_shared_)
    core_shared_->reset();
  cache_pos_ = 0;
  bases_.assign(bases_.size(), 0u);
  decode_offsets_.assign(bases_.size(), 0u);
  for (auto *v : {&dropped_amax_k_, &dropped_amax_v_})
    for (auto &a : *v)
      a.clear();
}

void KVCacheManager::foldDroppedRows(unsigned int l, unsigned int n_rows) {
  // fp16 magnitudes order as their bit patterns, so the max is taken on
  // the bits (one compare a value, vectorizable) and widened once a column
  const size_t w = kv_widths_[l];
  const size_t rows = getLayerCap(l);
  std::vector<uint16_t> mk(w, 0u), mv(w, 0u);
  for (auto [t, m] : {std::pair{&layer_caches_[l].key_cache, &mk},
                      std::pair{&layer_caches_[l].value_cache, &mv}}) {
    for (unsigned int b = 0; b < batch_size_; ++b) {
      const uint16_t *d = t->getData<uint16_t>() + b * rows * w;
      uint16_t *mm = m->data();
      for (unsigned int r = 0; r < n_rows; ++r, d += w)
        for (size_t c = 0; c < w; ++c)
          mm[c] = std::max<uint16_t>(mm[c], d[c] & 0x7FFFu);
    }
  }
  for (auto [m, a] : {std::pair{&mk, &dropped_amax_k_[l]},
                      std::pair{&mv, &dropped_amax_v_[l]}}) {
    if (a->empty())
      a->assign(w, 0.0f);
    for (size_t c = 0; c < w; ++c)
      (*a)[c] = std::max((*a)[c], nntrainer::compute_fp16_to_fp32((*m)[c]));
  }
}

void KVCacheManager::validateRange(unsigned int pos, bool compacting) const {
  if (pos > max_seq_len_)
    throw std::out_of_range("KVCacheManager: position exceeds max_seq_len");

  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    if (!isLayerActive(l) || !windowed(l) ||
        (compacting && isLayerKVAliased(l)))
      continue;
    const unsigned int from = liveFrom(l, pos);
    if (from < bases_[l] || pos - bases_[l] > getLayerCap(l))
      throw std::out_of_range("KVCacheManager: layer " + std::to_string(l) +
                              " does not hold rows [" + std::to_string(from) +
                              ", " + std::to_string(pos) + ")");
    if (compacting && from > bases_[l]) {
      for (const auto *t :
           {&layer_caches_[l].key_cache, &layer_caches_[l].value_cache}) {
        const auto md = t->getMemoryData();
        if (md && !md->isHostAddressable())
          throw std::runtime_error(
            "KVCacheManager::compact: a windowed layer's "
            "cache must be host addressable");
      }
    }
  }
}

void KVCacheManager::validateSaveRange(unsigned int seq_len) const {
  if (!isAllocated())
    throw std::runtime_error("KVCacheManager::save: not allocated");
  validateRange(seq_len, false);
}

std::shared_ptr<const KVCacheManager::PrefixSnapshot>
KVCacheManager::capturePrefix(unsigned position, const PrefixSnapshot *previous,
                              bool owned) const {
  if (flash_)
    throw std::logic_error("KV FSU supports file snapshots only");
  if (!isAllocated() || batch_size_ != 1 || position == 0 ||
      position > max_seq_len_ || position > cache_pos_)
    throw std::invalid_argument(
      "Prefill prefix requires an available original-call endpoint");
  auto state = std::make_shared<PrefixSnapshot>();
  state->position = position;
  state->element_bytes =
    layer_caches_.front().key_cache.getDim().getDataTypeSize();
  state->layers.resize(layer_caches_.size());
  prefix_views_.resize(layer_caches_.size());
  for (unsigned l = 0; l < state->layers.size(); ++l) {
    auto &dst = state->layers[l];
    dst.base = getBase(l);
    dst.width = kv_widths_[l];
    if (position < dst.base ||
        position - dst.base + getDecodeOffset(l) > getLayerCap(l) ||
        !isLayerActive(l))
      throw std::invalid_argument("Prefill prefix has an unavailable layer");
    dst.dropped_k = droppedAmaxK(l);
    dst.dropped_v = droppedAmaxV(l);
    const auto &src = layer_caches_[l];
    auto *k = src.key_cache.getData<unsigned char>();
    auto *v = src.value_cache.getData<unsigned char>();
    const auto key_owner =
      !owned && shared_retain_ ? shared_retain_(k) : nullptr;
    const auto value_owner =
      !owned && shared_retain_ ? shared_retain_(v) : nullptr;
    for (unsigned p = dst.base; p < position; p += 64u) {
      if (previous && l < previous->layers.size() &&
          previous->element_bytes == state->element_bytes &&
          previous->layers[l].width == dst.width &&
          p >= previous->layers[l].base &&
          (p - previous->layers[l].base) % 64u == 0 &&
          p + 64u <= previous->position &&
          (!owned || !previous->layers[l]
                        .pages.at((p - previous->layers[l].base) / 64u)
                        ->key_view)) {
        dst.pages.push_back(
          previous->layers[l].pages.at((p - previous->layers[l].base) / 64u));
      } else {
        auto page = std::make_shared<PrefixSnapshot::Page>();
        const size_t page_bytes =
          std::min(64u, position - p) * dst.width * state->element_bytes;
        const size_t off = size_t(p - dst.base + getDecodeOffset(l)) *
                           dst.width * state->element_bytes;
        if (key_owner && value_owner) {
          page->key_view = k + off;
          page->value_view = v + off;
          page->view_bytes = page_bytes;
          page->slab_bytes =
            size_t(getLayerCap(l)) * dst.width * state->element_bytes;
          page->key_owner = key_owner;
          page->value_owner = value_owner;
          prefix_views_[l].push_back(page);
        } else {
          page->key.assign(k + off, k + off + page_bytes);
          page->value.assign(v + off, v + off + page_bytes);
        }
        dst.pages.push_back(std::move(page));
      }
    }
  }
  return state;
}

size_t KVCacheManager::prefixPayloadBytes(unsigned position) const {
  if (!isAllocated() || position > cache_pos_)
    throw std::invalid_argument("Unavailable prefix payload");
  size_t bytes = 0;
  for (unsigned l = 0; l < layer_caches_.size(); ++l) {
    if (position < getBase(l))
      throw std::invalid_argument("Unavailable prefix window");
    bytes += size_t(position - getBase(l)) * kv_widths_[l] *
             layer_caches_[l].key_cache.getDim().getDataTypeSize() * 2u;
  }
  return bytes;
}

void KVCacheManager::restorePrefix(const PrefixSnapshot &state) {
  if (flash_)
    throw std::logic_error("KV FSU supports file snapshots only");
  if (!isAllocated() || batch_size_ != 1 || !state.position ||
      state.position > max_seq_len_ ||
      state.layers.size() != layer_caches_.size() ||
      state.element_bytes !=
        layer_caches_.front().key_cache.getDim().getDataTypeSize())
    throw std::invalid_argument("Incompatible prefill prefix geometry");
  // Validate and allocate metadata before changing any live payload.
  auto bases = bases_;
  auto dropped_k = dropped_amax_k_, dropped_v = dropped_amax_v_;
  for (unsigned l = 0; l < state.layers.size(); ++l) {
    const auto &src = state.layers[l];
    if (src.base > state.position ||
        state.position - src.base > getLayerCap(l) ||
        src.width != kv_widths_[l] ||
        src.pages.size() != (state.position - src.base + 63u) / 64u)
      throw std::invalid_argument("Incompatible prefill prefix layer");
    unsigned row = src.base;
    for (const auto &page : src.pages) {
      const size_t bytes =
        std::min(64u, state.position - row) * src.width * state.element_bytes;
      if (!page || page->keySize() != bytes || page->valueSize() != bytes)
        throw std::invalid_argument("Incomplete prefill prefix page");
      row += std::min(64u, state.position - row);
    }
    bases[l] = src.base;
    dropped_k[l] = src.dropped_k;
    dropped_v[l] = src.dropped_v;
  }
  for (unsigned l = 0; l < state.layers.size(); ++l)
    freezePrefixViews(l);
  for (unsigned l = 0; l < state.layers.size(); ++l) {
    auto *k = layer_caches_[l].key_cache.getData<unsigned char>();
    auto *v = layer_caches_[l].value_cache.getData<unsigned char>();
    for (const auto &page : state.layers[l].pages) {
      std::memcpy(k, page->keyData(), page->keySize());
      std::memcpy(v, page->valueData(), page->valueSize());
      k += page->keySize();
      v += page->valueSize();
    }
  }
  bases_.swap(bases);
  dropped_amax_k_.swap(dropped_k);
  dropped_amax_v_.swap(dropped_v);
  cache_pos_ = state.position;
  decode_offsets_.assign(bases_.size(), 0u);
}

void KVCacheManager::freezePrefixViews(unsigned layer) {
  if (layer >= prefix_views_.size())
    return;
  for (auto &weak : prefix_views_[layer])
    if (auto page = weak.lock())
      page->freeze();
  prefix_views_[layer].clear();
}

void KVCacheManager::prepareDecodeView(unsigned int pos) {
  if (!isAllocated() || batch_size_ != 1)
    throw std::logic_error("Decode views require allocated batch-one KV");
  validateRange(pos, true);
  for (unsigned l = 0; l < layer_caches_.size(); ++l)
    if (getDecodeOffset(l))
      throw std::logic_error("Host KV is already a read-only decode view");
  decode_offsets_.assign(layer_caches_.size(), 0u);
  for (unsigned l = 0; l < layer_caches_.size(); ++l) {
    if (!isLayerActive(l) || !windowed(l) || isLayerKVAliased(l))
      continue;
    const unsigned from = liveFrom(l, pos);
    if (from <= bases_[l])
      continue;
    const unsigned skip = from - bases_[l];
    // Identical dropped-row maxima and logical window as compact(), while
    // preserving the shared prefill bytes for radix snapshots.
    foldDroppedRows(l, skip);
    decode_offsets_[l] = skip;
    bases_[l] = from;
  }
}

bool KVCacheManager::compact(unsigned int pos) {
  if (!isAllocated())
    return false;
  for (unsigned l = 0; l < layer_caches_.size(); ++l)
    if (getDecodeOffset(l))
      throw std::logic_error(
        "Restore or reset the read-only decode view before prefill");
  // Validate every layer before folding dropped rows or moving any buffer.
  validateRange(pos, true);
  bool changed = false;
  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    if (!isLayerActive(l) || !windowed(l) || isLayerKVAliased(l))
      continue;
    const unsigned int from = liveFrom(l, pos);
    if (from <= bases_[l])
      continue;
    // compact() moves rows of a windowed layer, and flash storage keeps the
    // live rows of an evicted owner in a pool buffer whose layout
    // finishFlashLayer() derived from the unshifted base: the memmove would
    // hit the pristine original mapping and leave the real data behind.
    // allocate() already refuses rows/keep together with flash storage, so
    // this is unreachable today; it stays as the belt until a ring shrink or
    // the HTP port reconciles the two layouts.
    if (usesFlashStorage())
      throw std::logic_error(
        "KVCacheManager::compact: flash streaming cannot move rows of a "
        "windowed layer; disable the sliding ring shrink or flash storage");
    // A later sliding compaction may overwrite an immutable borrowed prefix.
    // Materialize only that layer's live pinned pages before the original move.
    freezePrefixViews(l);
    const size_t w = kv_widths_[l];
    const unsigned int rows = getLayerCap(l);
    const size_t skip = (from - bases_[l]) * w, live = (pos - from) * w;
    // [#4428 window] the rows about to go: their |max| for the decode seed
    foldDroppedRows(l, from - bases_[l]);
    for (auto *t :
         {&layer_caches_[l].key_cache, &layer_caches_[l].value_cache}) {
      for (unsigned int b = 0; b < batch_size_; ++b) {
        // fp16 bits on every build (FP16 or UINT16 storage)
        uint16_t *d =
          t->getData<uint16_t>() + b * static_cast<size_t>(rows) * w;
        std::memmove(d, d + skip, live * sizeof(uint16_t));
      }
    }
    bases_[l] = from;
    changed = true;
  }
  return changed;
}

nntrainer::Tensor &KVCacheManager::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].key_cache;
}

nntrainer::Tensor &KVCacheManager::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].value_cache;
}

nntrainer::Tensor KVCacheManager::getKeyCacheWriteView(unsigned int layer_idx,
                                                       unsigned int batch,
                                                       unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: would exceed max_seq_len");
  }
  if (!isLayerActive(layer_idx))
    throw std::logic_error("KVCacheManager: inactive layer has no cache view");

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheWriteView(
  unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: would exceed max_seq_len");
  }
  if (!isLayerActive(layer_idx))
    throw std::logic_error("KVCacheManager: inactive layer has no cache view");

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getKeyCacheReadView(unsigned int layer_idx,
                                                      unsigned int batch,
                                                      unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: read_len exceeds max_seq_len");
  }
  if (!isLayerActive(layer_idx))
    throw std::logic_error("KVCacheManager: inactive layer has no cache view");

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheReadView(unsigned int layer_idx,
                                                        unsigned int batch,
                                                        unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: read_len exceeds max_seq_len");
  }
  if (!isLayerActive(layer_idx))
    throw std::logic_error("KVCacheManager: inactive layer has no cache view");

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

/**
 * @brief On-disk header of a saved KV cache.
 *
 * A KV cache file used to be a bare concatenation of planes: no magic, no
 * geometry, no identity. Handed the wrong file -- another model, another
 * max_seq_len, another dtype -- load() would reinterpret whatever bytes were
 * there and the model would answer fluently from someone else's attention
 * state, because the only check was "did the read succeed". This header is what
 * turns those into named refusals, and it also records the ONE number the
 * planes cannot carry: the absolute token position they belong to (a ring
 * layer's rows mean nothing without it).
 *
 * Layout is fixed-size and little-endian-as-written (both targets are LE; a
 * future big-endian port has the version field to branch on). Everything after
 * `header_bytes` is the plane payload, in the same layer order the headerless
 * format used -- so the payload of a v1 file and of a legacy file are
 * byte-identical for a non-ring model, and a legacy file still loads (see
 * load()).
 */
namespace {

constexpr char kKvMagic[8] = {'N', 'N', 'T', 'R', 'K', 'V', 'C', '\0'};
constexpr unsigned int kKvVersion = 1u;
constexpr unsigned int kKvTagMax = 64u;

struct KvCacheFileHeader {
  char magic[8];
  unsigned int version;
  unsigned int header_bytes;
  unsigned int num_layers;
  unsigned int batch_size;
  unsigned int max_seq_len;  /**< cache capacity at save time */
  unsigned int seq_len;      /**< absolute token position saved */
  unsigned int dtype;        /**< ml::train::TensorDim::DataType */
  unsigned int format;       /**< ml::train::TensorDim::Format */
  unsigned int geom_digest;  /**< per-layer widths / caps / alias map */
  unsigned int payload_lo;   /**< plane bytes, low 32 */
  unsigned int payload_hi;   /**< plane bytes, high 32 */
  char model_tag[kKvTagMax]; /**< NUL-padded; empty = unnamed */
  unsigned int reserved[3];
};

/** 128 bytes exactly: 8 magic + 11 words + a 64-byte tag + 3 reserved words. A
 *  round number is not cosmetic here -- it is the one thing a person reading
 *  a hex dump of the file has to know, and `header_bytes` in the header is
 *  checked against it on load, so a build with a different layout refuses
 *  rather than reinterprets. */
static_assert(sizeof(KvCacheFileHeader) == 128,
              "KV cache header must stay packed at 128 bytes as written");

std::string tag_of(const KvCacheFileHeader &h) {
  size_t n = 0;
  while (n < kKvTagMax && h.model_tag[n] != '\0')
    ++n;
  return std::string(h.model_tag, n);
}

/** Read the header if the file carries one. `false` = legacy headerless (or
 *  unreadable, which the caller reports on its own). */
bool read_header(std::ifstream &f, KvCacheFileHeader &h) {
  f.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!f || std::memcmp(h.magic, kKvMagic, sizeof(kKvMagic)) != 0) {
    f.clear();
    f.seekg(0, std::ios::beg);
    return false;
  }
  return true;
}

} // namespace

unsigned int KVCacheManager::geometryDigest() const {
  unsigned int hash = 2166136261u;
  auto mix = [&hash](unsigned int v) {
    for (int b = 0; b < 4; ++b) {
      hash ^= (v >> (8 * b)) & 0xffu;
      hash *= 16777619u;
    }
  };
  const unsigned int n = static_cast<unsigned int>(layer_caches_.size());
  mix(n);
  for (unsigned int i = 0; i < n; ++i) {
    mix(i < kv_widths_.size() ? kv_widths_[i] : 0u);
    mix(getLayerCap(i));
    mix(windowed(i) ? keep_[i] : 0u);
    mix(static_cast<unsigned int>(getLayerKVSource(i) + 1));
  }
  return hash;
}

size_t KVCacheManager::fileBytesFor(unsigned int seq_len) const {
  size_t bytes = sizeof(KvCacheFileHeader);
  for (unsigned int i = 0; i < layer_caches_.size(); ++i) {
    ml::train::TensorDim d = layer_caches_[i].key_cache.getDim();
    d.height(rowsToPersist(i, seq_len));
    bytes += 2u * d.getDataLen() * d.getDataTypeSize();
  }
  return bytes;
}

std::string KVCacheManager::describeFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    return "cannot open '" + path + "'";
  KvCacheFileHeader h{};
  if (!read_header(f, h))
    return "'" + path +
           "' has no KV-cache header (legacy headerless file: no model, "
           "geometry or token length recorded)";
  std::string tag = tag_of(h);
  return "'" + path + "' v" + std::to_string(h.version) + " model='" +
         (tag.empty() ? std::string("(unnamed)") : tag) +
         "' tokens=" + std::to_string(h.seq_len) +
         " layers=" + std::to_string(h.num_layers) +
         " batch=" + std::to_string(h.batch_size) +
         " capacity=" + std::to_string(h.max_seq_len) +
         " dtype=" + std::to_string(h.dtype) +
         " geometry=" + std::to_string(h.geom_digest);
}

void KVCacheManager::save(const std::string &path) const {
  save(path, cache_pos_);
}

void KVCacheManager::save(const std::string &path, unsigned int seq_len,
                          const std::string &model_tag) const {
  validateSaveRange(seq_len);

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::save: cannot open file: " + path);
  }

  {
    KvCacheFileHeader h{};
    std::memcpy(h.magic, kKvMagic, sizeof(kKvMagic));
    h.version = kKvVersion;
    h.header_bytes = static_cast<unsigned int>(sizeof(h));
    h.num_layers = static_cast<unsigned int>(layer_caches_.size());
    h.batch_size = batch_size_;
    h.max_seq_len = max_seq_len_;
    h.seq_len = seq_len;
    h.dtype = static_cast<unsigned int>(dtype_);
    h.format = static_cast<unsigned int>(format_);
    h.geom_digest = geometryDigest();
    const size_t payload = fileBytesFor(seq_len) - sizeof(h);
    h.payload_lo = static_cast<unsigned int>(payload & 0xffffffffull);
    h.payload_hi = static_cast<unsigned int>(payload >> 32);
    if (model_tag.size() >= kKvTagMax)
      throw std::invalid_argument("KVCacheManager::save: model tag '" +
                                  model_tag + "' exceeds " +
                                  std::to_string(kKvTagMax - 1) + " bytes");
    std::memcpy(h.model_tag, model_tag.data(), model_tag.size());
    f.write(reinterpret_cast<const char *>(&h), sizeof(h));
    if (!f)
      throw std::runtime_error(
        "KVCacheManager::save: cannot write the header of " + path);
  }

  // [kv-share] The file format stays one K plane + one V plane per layer, in
  // layer order, exactly as before -- so a file written by an aliasing build
  // and one written by a non-aliasing build of the same model are byte
  // identical, and either loads into either. An aliased layer simply emits its
  // source's plane a second time, which is not an approximation: the values
  // written into a sharer's private slab always WERE its source's values (same
  // wk/wv, same k_norm/v_norm, same RoPE, same absolute position -- see
  // Gemma4Transformer::createSharedAttention). load() below skips the
  // duplicates rather than replaying them.
  for (unsigned int li = 0; li < layer_caches_.size(); ++li) {
    const auto &lc = layer_caches_[li];
    ml::train::TensorDim save_dim = lc.key_cache.getDim();
    // [kv-window-ring] A ring layer's plane is Wcap rows and the live rows
    // wrap, so seq_len rows would reach past the plane (getSharedDataTensor
    // throws) -- write the whole ring and let the header's absolute position
    // re-derive which row is which. A full layer is unchanged: rowsToPersist ==
    // seq_len.
    save_dim.height(rowsToPersist(li, seq_len));
    if (!isLayerActive(li)) {
      // Preserve the dense file layout, including dtype-specific metadata.
      // This temporary plane exists only during an explicit cache save.
      nntrainer::Tensor padding(save_dim, true);
      padding.setZero();
      padding.save(f);
      padding.save(f);
      continue;
    }
    const FlashSnapshotLease lease(flash_.get(), li);
    // [HTP window] a windowed layer's live rows start at liveFrom - base
    size_t save_off = 0;
    if (windowed(li)) {
      const unsigned int from = liveFrom(li, seq_len);
      save_off = static_cast<size_t>(from - bases_[li]) * kv_widths_[li];
    }

    nntrainer::Tensor k_slice =
      const_cast<nntrainer::Tensor &>(lc.key_cache)
        .getSharedDataTensor(save_dim, save_off, true);
    nntrainer::Tensor v_slice =
      const_cast<nntrainer::Tensor &>(lc.value_cache)
        .getSharedDataTensor(save_dim, save_off, true);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Device-only KV: Tensor::save reads on the host -- stage D2H first.
    auto save_slice = [&f](nntrainer::Tensor &slice) {
      void *ptr = (void *)slice.getData<char>();
      const auto md = slice.getMemoryData();
      if (md && !md->isHostAddressable()) {
        nntrainer::Tensor host_t(slice.getDim(), true);
        if (!nntrainer::cuda::copy_any((void *)host_t.getData<char>(), ptr,
                                       host_t.bytes()))
          throw std::runtime_error(
            "KVCacheManager::save: D2H staging of the device KV failed");
        host_t.save(f);
      } else {
        slice.save(f);
      }
    };
    save_slice(k_slice);
    save_slice(v_slice);
#else
    k_slice.save(f);
    v_slice.save(f);
#endif
  }
}

unsigned int KVCacheManager::load(const std::string &path, unsigned int seq_len,
                                  const std::string &model_tag) {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::load: not allocated");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::load: cannot open file: " + path);
  }

  // [kv-share] File extent, taken once: seeking PAST the end of an ifstream is
  // not itself an error (the failure only surfaces on the next read), and the
  // aliased planes below are skipped by seeking. Without this, a file truncated
  // inside the trailing aliased region -- every layer after the last owner, so
  // most of the file for a KV-sharing model -- would be skipped over and
  // "loaded" silently.
  f.seekg(0, std::ios::end);
  const std::streamoff file_end = f.tellg();
  f.seekg(0, std::ios::beg);

  // The header, when the file has one. A named refusal here is the whole point:
  // without it, a file from another model / capacity / dtype is simply a byte
  // count, and a byte count that happens to match loads someone else's
  // attention state into this model (two models in this tree have the SAME
  // per-layer KV width and layer count, so a length check alone passes).
  KvCacheFileHeader h{};
  const bool versioned = read_header(f, h);
  auto refuse = [&path](const std::string &what) {
    throw std::runtime_error("KVCacheManager::load: " + path +
                             " does not belong to this model: " + what);
  };
  if (versioned) {
    if (h.version != kKvVersion)
      refuse("file format version " + std::to_string(h.version) +
             ", this build reads version " + std::to_string(kKvVersion));
    if (h.header_bytes != sizeof(KvCacheFileHeader))
      refuse("header is " + std::to_string(h.header_bytes) +
             " bytes, this build's is " +
             std::to_string(sizeof(KvCacheFileHeader)));
    const std::string file_tag = tag_of(h);
    if (!model_tag.empty() && !file_tag.empty() && file_tag != model_tag)
      refuse("written for model '" + file_tag + "', this model is '" +
             model_tag + "'");
    if (h.num_layers != layer_caches_.size())
      refuse("written with " + std::to_string(h.num_layers) +
             " attention layers, this model has " +
             std::to_string(layer_caches_.size()));
    if (h.batch_size != batch_size_)
      refuse("written at batch " + std::to_string(h.batch_size) +
             ", this model runs at batch " + std::to_string(batch_size_));
    if (h.dtype != static_cast<unsigned int>(dtype_))
      refuse("written with cache dtype " + std::to_string(h.dtype) +
             ", this model's is " +
             std::to_string(static_cast<unsigned int>(dtype_)));
    if (h.format != static_cast<unsigned int>(format_))
      refuse("written with tensor format " + std::to_string(h.format) +
             ", this model's is " +
             std::to_string(static_cast<unsigned int>(format_)));
    if (h.geom_digest != geometryDigest())
      refuse("per-layer geometry digest " + std::to_string(h.geom_digest) +
             " != this model's " + std::to_string(geometryDigest()) +
             " (KV widths, sliding-window ring capacities or the KV-sharing "
             "alias map differ)");
    // The capacity may legitimately differ (a bigger window still holds a
    // shorter cache), as long as this model can hold what the file carries.
    if (h.seq_len > max_seq_len_)
      refuse("holds " + std::to_string(h.seq_len) +
             " tokens, this model's cache capacity is " +
             std::to_string(max_seq_len_));
    if (seq_len != 0 && seq_len != h.seq_len)
      refuse("holds " + std::to_string(h.seq_len) +
             " tokens, the caller asked to resume at " +
             std::to_string(seq_len));
    seq_len = h.seq_len;
    const size_t payload =
      (static_cast<size_t>(h.payload_hi) << 32) | h.payload_lo;
    const size_t expect = fileBytesFor(seq_len) - sizeof(KvCacheFileHeader);
    if (payload != expect)
      refuse("declares " + std::to_string(payload) +
             " plane bytes, this geometry needs " + std::to_string(expect));
    if (static_cast<size_t>(file_end) < sizeof(KvCacheFileHeader) + payload)
      refuse("is truncated: " + std::to_string(file_end) + " bytes on disk, " +
             std::to_string(sizeof(KvCacheFileHeader) + payload) + " expected");
  } else {
    // A file written before the header existed: no identity, no length. Honour
    // the caller's seq_len (the only source there is) and say so once, because
    // every check above is unavailable for it.
    if (seq_len == 0)
      throw std::runtime_error(
        "KVCacheManager::load: " + path +
        " carries no KV-cache header, so the token length cannot be read from "
        "it; pass the position the cache was saved at");
    ml_logw(
      "[kv-cache] %s has no header (legacy headerless format): loading %u "
      "tokens on the caller's word, with no model or geometry check",
      path.c_str(), seq_len);
  }

  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::load: seq_len exceeds max_seq_len");
  }

  if (flash_)
    flash_->reset();
  for (unsigned int i = 0; i < layer_caches_.size(); ++i) {
    auto &lc = layer_caches_[i];
    ml::train::TensorDim load_dim = lc.key_cache.getDim();
    load_dim.height(versioned ? rowsToPersist(i, seq_len) : seq_len);
    if (!isLayerActive(i)) {
      // Consume the old dense slot without retaining unused convolution KV.
      nntrainer::Tensor padding(load_dim, true);
      padding.read(f);
      padding.read(f);
      continue;
    }
    // [HTP window] the loaded rows land at row 0; the base says which
    // absolute position that is
    if (windowed(i))
      bases_[i] = liveFrom(i, seq_len);

    const FlashSnapshotLease lease(isLayerKVAliased(i) ? nullptr : flash_.get(),
                                   i);
    nntrainer::Tensor k_slice =
      lc.key_cache.getSharedDataTensor(load_dim, 0, true);
    nntrainer::Tensor v_slice =
      lc.value_cache.getSharedDataTensor(load_dim, 0, true);

    // [kv-share] An aliased layer's plane IS its source's plane. Reading it
    // again would write the same bytes over memory that was already filled by
    // the source's own read -- harmless when the file came from save() above,
    // but it makes the final contents depend on which sharer happened to be
    // read last instead of on the source. Skip the duplicate bytes (the file
    // still carries them, so the format stays interchangeable in both
    // directions) and let the source's read be the single writer.
    if (isLayerKVAliased(i)) {
      f.seekg(static_cast<std::streamoff>(k_slice.bytes() + v_slice.bytes()),
              std::ios::cur);
      if (!f || f.tellg() > file_end)
        throw std::runtime_error(
          "KVCacheManager::load: truncated file while skipping the aliased "
          "plane of layer " +
          std::to_string(i));
      continue;
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Device-only KV: Tensor::read writes on the host -- read into a host
    // temp, then push H2D.
    auto load_slice = [&f](nntrainer::Tensor &slice) {
      void *ptr = (void *)slice.getData<char>();
      const auto md = slice.getMemoryData();
      if (md && !md->isHostAddressable()) {
        nntrainer::Tensor host_t(slice.getDim(), true);
        host_t.read(f);
        if (!nntrainer::cuda::copy_any(
              ptr, (const void *)host_t.getData<char>(), host_t.bytes()))
          throw std::runtime_error(
            "KVCacheManager::load: H2D staging of the device KV failed");
      } else {
        slice.read(f);
      }
    };
    load_slice(k_slice);
    load_slice(v_slice);
#else
    k_slice.read(f);
    v_slice.read(f);
#endif
    if (flash_)
      flash_->store(i, 0, k_slice.bytes(), true);
  }

  cache_pos_ = seq_len;
  return seq_len;
}

} // namespace causallm
