// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_cache_manager.cpp
 * @brief Synchronous host/device KV ownership, views and transactional
 * publication.
 * @author Codex (OpenAI)
 * @bug Sessions are thread confined; asynchronous publication is unsupported.
 */
#include <kv_cache/kv_cache_manager.h>

#include <algorithm>
#include <utility>

namespace nntrainer::kv {

struct ViewEpoch {
  uint64_t generation = 1;
  uint64_t active_step = 0;
};

enum class ProducerState { Missing, Ordered, Complete };

struct SessionState {
  CacheSpec spec;
  std::shared_ptr<KvStorageOperations> operations;
  std::vector<std::shared_ptr<KvStorage>> keys, values;
  std::vector<ProducerState> ready;
  uint64_t replay_generation = 1;
  std::shared_ptr<ViewEpoch> epoch = std::make_shared<ViewEpoch>();
  size_t position = 0;
  uint64_t next_step = 0;
  bool valid = true;
  bool overwrites = false;
  CacheStats stats;
};

void KvView::validate(bool write) const {
  if (!storage_ || epoch_->generation != generation_ ||
      (write && (!step_ || epoch_->active_step != step_)))
    throw std::logic_error("Stale or non-writable KV view");
}

const AllocationInfo &KvView::allocationInfo() const {
  validate(false);
  return storage_->allocationInfo();
}
const void *KvView::address() const {
  validate(false);
  return static_cast<const char *>(storage_->data()) + offset_;
}
void *KvView::writableAddress() const {
  validate(true);
  return static_cast<char *>(storage_->data()) + offset_;
}
const void *KvView::data() const {
  if (!allocationInfo().hostAddressable())
    throw std::logic_error("Device KV view cannot be dereferenced on the host");
  return address();
}
void *KvView::writableData() const {
  if (!allocationInfo().hostAddressable())
    throw std::logic_error("Device KV view cannot be dereferenced on the host");
  return writableAddress();
}

StepLease::StepLease(std::shared_ptr<SessionState> state, uint64_t id,
                     size_t from, size_t to) :
  state_(std::move(state)), id_(id), from_(from), to_(to) {}

StepLease::StepLease(StepLease &&other) noexcept :
  state_(std::move(other.state_)),
  id_(other.id_),
  from_(other.from_),
  to_(other.to_) {}

StepLease &StepLease::operator=(StepLease &&other) noexcept {
  if (this != &other) {
    abort();
    state_ = std::move(other.state_);
    id_ = other.id_;
    from_ = other.from_;
    to_ = other.to_;
  }
  return *this;
}

void StepLease::abort() noexcept {
  if (state_ && state_->epoch->active_step == id_) {
    if (state_->operations) {
      try {
        state_->operations->synchronize();
      } catch (...) {
        state_->valid = false;
      }
    }
    if (state_->overwrites)
      state_->valid = false;
    state_->epoch->active_step = 0;
    ++state_->epoch->generation;
    ++state_->replay_generation;
    if (state_->operations)
      state_->operations->invalidateReplay();
  }
  state_.reset();
}

StepLease::~StepLease() { abort(); }

KvCacheManager::KvCacheManager() = default;
KvCacheManager::~KvCacheManager() {
  if (state_ && state_->operations)
    state_->operations->invalidateReplay();
}

void KvCacheManager::initialize(
  const CacheSpec &spec, std::shared_ptr<MemAllocator> allocator,
  std::shared_ptr<KvStorageOperations> operations) {
  const auto bytes = spec.validate();
  if (state_ && state_->epoch->active_step)
    throw std::logic_error("KV step is in flight");
  auto next = std::make_shared<SessionState>();
  next->spec = spec;
  next->operations = operations;
  next->stats.backing_bytes = operations ? 0 : bytes;
  for (const auto &s : spec.layers) {
    if (s.source >= 0) {
      next->keys.push_back(next->keys[s.source]);
      next->values.push_back(next->values[s.source]);
    } else {
      next->keys.push_back(
        KvStorage::create(s.planeBytes(), allocator, operations));
      next->values.push_back(
        KvStorage::create(s.planeBytes(), allocator, operations));
      if (operations)
        next->stats.backing_bytes =
          checkedAdd(next->stats.backing_bytes,
                     checkedAdd(next->keys.back()->allocationInfo().bytes,
                                next->values.back()->allocationInfo().bytes));
      ++next->stats.owner_layers;
    }
  }
  next->ready.resize(spec.layers.size());
  if (state_) {
    ++state_->epoch->generation;
    if (state_->operations)
      state_->operations->invalidateReplay();
  }
  state_ = std::move(next);
}

StepLease KvCacheManager::beginStep(size_t count) {
  if (!state_ || !state_->valid || state_->epoch->active_step)
    throw std::logic_error("KV session unavailable or step already active");
  if (!count ||
      count > state_->spec.layers.front().max_tokens - state_->position)
    throw std::out_of_range("KV step exceeds logical capacity");
  for (const auto &s : state_->spec.layers)
    if (count > s.capacity)
      throw std::out_of_range("KV step exceeds physical capacity; split it");
  const size_t to = state_->position + count;
  state_->overwrites = false;
  for (const auto &s : state_->spec.layers)
    state_->overwrites |= to > s.capacity;
  std::fill(state_->ready.begin(), state_->ready.end(), ProducerState::Missing);
  // Old read ranges must expire before a new step can overwrite ring rows.
  ++state_->epoch->generation;
  state_->epoch->active_step = ++state_->next_step;
  return StepLease(state_, state_->epoch->active_step, state_->position, to);
}

void KvCacheManager::checkStep(const StepLease &step) const {
  if (!state_ || step.state_ != state_ || !step.id_ ||
      state_->epoch->active_step != step.id_ || !state_->valid)
    throw std::logic_error("Invalid KV step lease");
}

std::vector<KvView> KvCacheManager::views(const StepLease &step, size_t layer,
                                          size_t batch, bool key, bool write) {
  checkStep(step);
  const auto &s = state_->spec.layers.at(layer);
  if (!s.has_producer || batch >= s.batch || (write && s.source >= 0))
    throw std::out_of_range("Invalid KV batch or alias write");
  if (!write) {
    size_t owner = layer;
    while (state_->spec.layers[owner].source >= 0)
      owner = state_->spec.layers[owner].source;
    if (state_->ready[owner] == ProducerState::Missing)
      throw std::logic_error("KV producer is not ready");
  }
  const size_t from =
    write ? step.from_ : (step.to_ > s.capacity ? step.to_ - s.capacity : 0);
  const size_t count = step.to_ - from;
  const size_t row = from % s.capacity;
  const size_t first = std::min(count, s.capacity - row);
  std::vector<KvView> result;
  auto append = [&](size_t at, size_t rows) {
    KvView v;
    v.storage_ = key ? state_->keys[layer] : state_->values[layer];
    v.epoch_ = state_->epoch;
    v.generation_ = state_->epoch->generation;
    v.step_ = write ? step.id_ : 0;
    v.offset_ = (batch * s.capacity + at) * s.rowBytes();
    v.bytes_ = rows * s.rowBytes();
    result.push_back(std::move(v));
  };
  append(row, first);
  if (first < count)
    append(0, count - first);
  return result;
}

std::vector<KvView> KvCacheManager::writeKey(const StepLease &step, size_t l,
                                             size_t b) {
  return views(step, l, b, true, true);
}
std::vector<KvView> KvCacheManager::writeValue(const StepLease &step, size_t l,
                                               size_t b) {
  return views(step, l, b, false, true);
}
std::vector<KvView> KvCacheManager::readKey(const StepLease &step, size_t l,
                                            size_t b) {
  return views(step, l, b, true, false);
}
std::vector<KvView> KvCacheManager::readValue(const StepLease &step, size_t l,
                                              size_t b) {
  return views(step, l, b, false, false);
}

void KvCacheManager::markReady(const StepLease &step, size_t layer) {
  checkStep(step);
  if (state_->spec.layers.at(layer).source >= 0 ||
      !state_->spec.layers.at(layer).has_producer)
    throw std::invalid_argument("KV alias has no producer");
  bool completed = true;
  if (state_->operations) {
    try {
      completed = state_->operations->producerReady();
    } catch (...) {
      state_->valid = false;
      throw;
    }
  }
  state_->ready[layer] =
    completed ? ProducerState::Complete : ProducerState::Ordered;
}

void KvCacheManager::commit(StepLease &&step) {
  checkStep(step);
  for (size_t i = 0; i < state_->ready.size(); ++i)
    if (state_->spec.layers[i].source < 0 &&
        state_->spec.layers[i].has_producer &&
        state_->ready[i] == ProducerState::Missing)
      throw std::logic_error("KV commit before all producers completed");
  if (state_->operations) {
    try {
      state_->operations->synchronize();
    } catch (...) {
      state_->valid = false;
      throw;
    }
  }
  state_->position = step.to_;
  state_->epoch->active_step = 0;
  step.state_.reset();
}

void KvCacheManager::abort(StepLease &&step) {
  // A failed device completion invalidates the session, but its live lease must
  // still be drainable. Publication checks require valid(); abort does not.
  if (!state_ || step.state_ != state_ || !step.id_ ||
      state_->epoch->active_step != step.id_)
    throw std::logic_error("Invalid KV step lease");
  step.abort();
}

void KvCacheManager::reset() {
  if (!state_ || state_->epoch->active_step)
    throw std::logic_error("Cannot reset an uninitialized or active KV cache");
  if (state_->operations)
    state_->operations->synchronize();
  state_->valid = true;
  state_->position = 0;
  ++state_->epoch->generation;
  ++state_->replay_generation;
  if (state_->operations)
    state_->operations->invalidateReplay();
}

void KvCacheManager::setPosition(size_t position) {
  if (!state_ || !state_->valid || state_->epoch->active_step)
    throw std::logic_error("Cannot seek an unavailable KV session");
  if (position > state_->spec.layers.front().max_tokens)
    throw std::out_of_range("KV position exceeds capacity");
  state_->position = position;
  ++state_->epoch->generation;
  ++state_->replay_generation;
  if (state_->operations)
    state_->operations->invalidateReplay();
}

size_t KvCacheManager::committedLength() const {
  return state_ ? state_->position : 0;
}
bool KvCacheManager::valid() const { return state_ && state_->valid; }
bool KvCacheManager::hasActiveStep() const {
  return state_ && state_->epoch->active_step != 0;
}
CacheStats KvCacheManager::stats() const {
  return state_ ? state_->stats : CacheStats{};
}
const LayerSpec &KvCacheManager::layerSpec(size_t layer) const {
  if (!state_)
    throw std::logic_error("KV cache is uninitialized");
  return state_->spec.layers.at(layer);
}
std::shared_ptr<MemoryData> KvCacheManager::keyMemory(size_t layer) {
  layerSpec(layer);
  return state_->keys[layer]->memory();
}
std::shared_ptr<MemoryData> KvCacheManager::valueMemory(size_t layer) {
  layerSpec(layer);
  return state_->values[layer]->memory();
}

thread_local KvStepScope *KvStepScope::current_ = nullptr;

std::unique_ptr<KvStepScope> KvCacheManager::bindStep(const StepLease &step) {
  checkStep(step);
  return std::unique_ptr<KvStepScope>(new KvStepScope(*this, step));
}

KvStepScope::KvStepScope(KvCacheManager &manager, const StepLease &step) :
  manager_(manager), step_(step) {
  if (current_)
    throw std::logic_error("Nested CPU KV graph scopes are unsupported");
  for (const auto &layer : manager_.state_->spec.layers)
    if (layer.batch != 1 || layer.capacity != layer.max_tokens)
      throw std::invalid_argument(
        "CPU graph KV views require batch-1 linear storage");
  value_producers_.resize(manager_.state_->spec.layers.size());
  current_ = this;
}

KvStepScope::~KvStepScope() { current_ = nullptr; }

void KvStepScope::validate() const {
  if (current_ != this)
    throw std::logic_error("CPU KV access escaped its graph scope");
  manager_.checkStep(step_);
}

void KvStepScope::registerValueProducer(
  size_t layer, const void *producer,
  std::function<void(const KvView &)> bind) {
  validate();
  if (!producer || !bind || layer >= value_producers_.size() ||
      value_producers_[layer].token)
    throw std::invalid_argument("Invalid or duplicate KV value producer");
  for (const auto &registered : value_producers_)
    if (registered.token == producer)
      throw std::invalid_argument("A KV producer cannot bind multiple owners");
  manager_.writeValue(step_, layer); // Validate owner and write permission now.
  value_producers_[layer] = {producer, std::move(bind), false, false};
}

void KvStepScope::prepareValueProducer(const void *producer) {
  if (!current_)
    return;
  auto &scope = *current_;
  scope.validate();
  for (size_t i = 0; i < scope.value_producers_.size(); ++i) {
    auto &registered = scope.value_producers_[i];
    if (registered.token != producer)
      continue;
    if (registered.bound)
      throw std::logic_error("KV value producer executed twice in one step");
    auto views = scope.manager_.writeValue(scope.step_, i);
    if (views.size() != 1)
      throw std::logic_error("Direct V producer requires linear KV storage");
    registered.bind(views[0]);
    registered.bound = true;
    return;
  }
}

void KvStepScope::valueProduced(const void *producer, const void *address,
                                size_t bytes) {
  if (!current_)
    return;
  auto &scope = *current_;
  scope.validate();
  for (size_t i = 0; i < scope.value_producers_.size(); ++i) {
    auto &registered = scope.value_producers_[i];
    if (registered.token != producer)
      continue;
    auto views = scope.manager_.writeValue(scope.step_, i);
    if (!registered.bound || registered.submitted || views.size() != 1 ||
        views[0].writableAddress() != address || views[0].sizeBytes() != bytes)
      throw std::logic_error(
        "Final V producer wrote outside its checked slice");
    // This is submission, not GPU completion. markReady synchronizes the
    // stream.
    registered.submitted = true;
    ++scope.manager_.state_->stats.direct_value_writes;
    return;
  }
}

KvReplayBinding KvStepScope::replayBinding() {
  if (!current_)
    return {};
  auto &scope = *current_;
  scope.validate();
  auto &state = *scope.manager_.state_;
  if (!state.operations || !state.operations->supportsReplay())
    return {};
  KvReplayBinding result{scope.manager_.state_, state.replay_generation, {}};
  for (const auto &producer : scope.value_producers_) {
    if (!producer.token)
      throw std::logic_error(
        "Replay requires every final V producer declaration");
    result.producers.push_back(producer.token);
  }
  return result;
}

void *KvStepScope::slotValueBase(const void *producer) {
  if (!current_)
    return nullptr;
  auto &scope = *current_;
  scope.validate();
  auto &state = *scope.manager_.state_;
  if (!state.operations || !state.operations->supportsReplay() ||
      scope.step_.to() - scope.step_.from() != 1 || scope.step_.from() == 0)
    return nullptr;
  for (size_t i = 0; i < scope.value_producers_.size(); ++i)
    if (scope.value_producers_[i].token == producer)
      return state.values[i]->data();
  return nullptr;
}

void KvStepScope::capturedSubmission() {
  if (!current_ || !replayBinding().owner)
    return;
  auto &scope = *current_;
  for (size_t i = 0; i < scope.value_producers_.size(); ++i)
    if (!scope.value_producers_[i].submitted ||
        scope.manager_.state_->ready[i] == ProducerState::Missing)
      throw std::logic_error("Captured KV graph has an unsubmitted producer");
  ++scope.manager_.state_->stats.graph_captures;
}

void KvStepScope::replaySubmission(const KvReplayBinding &binding) {
  if (!current_ || !binding.owner || binding != replayBinding())
    throw std::logic_error("Stale KV replay binding");
  auto &scope = *current_;
  for (size_t i = 0; i < scope.value_producers_.size(); ++i) {
    scope.manager_.markReady(scope.step_, i);
    ++scope.manager_.state_->stats.direct_value_writes;
    ++scope.manager_.state_->stats.device_attention_calls;
  }
  ++scope.manager_.state_->stats.graph_replays;
}

bool KvStepScope::deviceStep() {
  if (!current_)
    return false;
  current_->validate();
  return current_->manager_.state_->operations != nullptr;
}

void KvStepScope::noteAppendCopy(size_t bytes) {
  if (!current_)
    return;
  current_->validate();
  auto &stats = current_->manager_.state_->stats;
  stats.append_copy_bytes = checkedAdd(stats.append_copy_bytes, bytes);
}

void KvStepScope::noteDeviceRead(const void *key, const void *value,
                                 size_t bytes) {
  if (!current_)
    return;
  auto &scope = *current_;
  scope.validate();
  const auto &state = *scope.manager_.state_;
  for (size_t i = 0; i < state.spec.layers.size(); ++i) {
    if (state.keys[i]->data() != key || state.values[i]->data() != value)
      continue;
    const auto views = scope.manager_.readKey(scope.step_, i);
    if (views.size() != 1 || views[0].address() != key ||
        views[0].sizeBytes() != bytes)
      throw std::logic_error(
        "Device attention read differs from checked KV view");
    ++scope.manager_.state_->stats.device_attention_calls;
    return;
  }
  throw std::logic_error("Device attention does not read its core KV backing");
}

std::unique_ptr<KvLayerAccess> KvStepScope::access(const void *key,
                                                   const void *value,
                                                   size_t batch, size_t from,
                                                   size_t to) {
  if (!current_)
    return nullptr; // Existing unmanaged Tensor paths remain compatible.
  auto &scope = *current_;
  scope.validate();
  if (batch || from != scope.step_.from() || to != scope.step_.to())
    throw std::logic_error("CPU MHA interval differs from its KV lease");
  const auto &state = *scope.manager_.state_;
  for (size_t i = 0; i < state.spec.layers.size(); ++i)
    if (state.spec.layers[i].source < 0 && state.keys[i]->data() == key &&
        state.values[i]->data() == value)
      return std::unique_ptr<KvLayerAccess>(new KvLayerAccess(scope, i));
  throw std::logic_error("CPU MHA cache does not belong to the active KV step");
}

KvLayerAccess::KvLayerAccess(KvStepScope &scope, size_t layer) :
  scope_(&scope),
  scope_lifetime_(scope.lifetime_),
  layer_(layer),
  key_write_(scope.manager_.writeKey(scope.step_, layer)),
  value_write_(scope.manager_.writeValue(scope.step_, layer)) {}

KvStepScope &KvLayerAccess::checkedScope() const {
  if (scope_lifetime_.expired())
    throw std::logic_error("CPU KV access outlived its graph scope");
  scope_->validate();
  return *scope_;
}

void KvLayerAccess::validateWrite(const void *key, const void *value,
                                  size_t bytes) const {
  checkedScope();
  if (key_write_.size() != 1 || value_write_.size() != 1 ||
      key_write_[0].writableAddress() != key ||
      value_write_[0].writableAddress() != value ||
      key_write_[0].sizeBytes() != bytes ||
      value_write_[0].sizeBytes() != bytes)
    throw std::logic_error(
      "CPU producer Tensor is outside its checked KV write view");
}

bool KvLayerAccess::directValue(const void *value, size_t bytes) const {
  auto &scope = checkedScope();
  const auto &producer = scope.value_producers_[layer_];
  if (!producer.token)
    return false;
  if (!producer.submitted || value_write_.size() != 1 ||
      value_write_[0].writableAddress() != value ||
      value_write_[0].sizeBytes() != bytes)
    throw std::logic_error("MHA did not receive its final V producer output");
  return true;
}

void KvLayerAccess::producerReady() {
  auto &scope = checkedScope();
  const auto &producer = scope.value_producers_[layer_];
  if (producer.token && !producer.submitted)
    throw std::logic_error("KV V producer did not execute");
  scope.manager_.markReady(scope.step_, layer_);
  key_read_ = scope.manager_.readKey(scope.step_, layer_);
  value_read_ = scope.manager_.readValue(scope.step_, layer_);
}

void KvLayerAccess::validateRead(const void *key, const void *value,
                                 size_t bytes) const {
  checkedScope();
  if (key_read_.size() != 1 || value_read_.size() != 1 ||
      key_read_[0].address() != key || value_read_[0].address() != value ||
      key_read_[0].sizeBytes() != bytes || value_read_[0].sizeBytes() != bytes)
    throw std::logic_error(
      "CPU consumer Tensor is outside its checked KV read view");
}

} // namespace nntrainer::kv
