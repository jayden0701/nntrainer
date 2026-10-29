// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_kv_storage.cpp
 * @brief Checked CUDA completion, initialization and explicit snapshot I/O.
 * @author Codex (OpenAI)
 * @bug Only one backend stream and explicit dense FP16 replay are supported.
 */
#include <cstring>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_kv_storage.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <stdexcept>

namespace nntrainer::cuda {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
class Operations final : public kv::KvStorageOperations {
public:
  explicit Operations(AllocationKind kind, bool ordered) :
    kind_(kind), ordered_(ordered) {
    if (kind != AllocationKind::CudaDevice &&
        kind != AllocationKind::CudaHostMapped)
      throw std::invalid_argument("Unsupported CUDA KV allocation kind");
    if (!ContextManager::Global().isAvailable())
      throw std::runtime_error("CUDA KV context unavailable");
    if (ordered_) {
      if (kind != AllocationKind::CudaDevice)
        throw std::invalid_argument("Replay KV requires device-only storage");
      check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
  }
  ~Operations() override {
    if (event_)
      cudaEventDestroy(event_);
  }
  bool supportsReplay() const override { return ordered_; }
  void invalidateReplay() noexcept override {
    if (ordered_)
      cuda_reset_decode_graph_cache();
  }
  bool producerReady() override {
    if (!ordered_) {
      synchronize();
      return true;
    }
    auto &stream = StreamManager::Global();
    check(cudaEventRecord(event_, stream.GetStream()));
    check(cudaStreamWaitEvent(stream.GetStream(), event_, 0));
    return false;
  }
  AllocationKind kind() const override { return kind_; }
  bool accepts(const AllocationInfo &info) const override {
    auto &context = ContextManager::Global();
    return info.kind == kind_ && info.device == context.GetDeviceOrdinal() &&
           info.context == reinterpret_cast<uintptr_t>(context.GetContext());
  }
  void synchronize() override {
    ContextManager::Global().EnsureCurrent();
    auto &manager = StreamManager::Global();
    if (manager.isCapturing())
      throw std::logic_error(
        "Synchronous core KV cannot execute during capture");
    if (!manager.GetStream())
      throw std::runtime_error("CUDA KV backend stream unavailable");
    if (ordered_) {
      check(cudaEventRecord(event_, manager.GetStream()));
      check(cudaEventSynchronize(event_));
    } else {
      check(cudaStreamSynchronize(manager.GetStream()));
    }
  }
  void zero(void *native, size_t bytes) override {
    synchronize();
    if (kind_ == AllocationKind::CudaHostMapped)
      std::memset(native, 0, bytes);
    else {
      check(
        cudaMemsetAsync(native, 0, bytes, StreamManager::Global().GetStream()));
      synchronize();
    }
  }
  void read(void *host, const void *native, size_t bytes) override {
    synchronize();
    if (kind_ == AllocationKind::CudaHostMapped)
      std::memcpy(host, native, bytes);
    else {
      check(cudaMemcpyAsync(host, native, bytes, cudaMemcpyDeviceToHost,
                            StreamManager::Global().GetStream()));
      synchronize();
    }
  }
  void write(void *native, const void *host, size_t bytes) override {
    synchronize();
    if (kind_ == AllocationKind::CudaHostMapped)
      std::memcpy(native, host, bytes);
    else {
      check(cudaMemcpyAsync(native, host, bytes, cudaMemcpyHostToDevice,
                            StreamManager::Global().GetStream()));
      synchronize();
    }
  }

private:
  AllocationKind kind_;
  bool ordered_;
  cudaEvent_t event_ = nullptr;
};
} // namespace
std::shared_ptr<kv::KvStorageOperations>
makeKvStorageOperations(AllocationKind kind, bool graph_ordered) {
  return std::make_shared<Operations>(kind, graph_ordered);
}

bool kvValueSlot(const unsigned short *input, const unsigned short *weight,
                 unsigned short *base, int outputs, int inputs) {
  if (outputs <= 0 || inputs <= 0 || !dev_accessible(input) ||
      !dev_accessible(weight) || !dev_accessible(base))
    return false;
  static const char *source = R"CU(
extern "C" __global__ void kv_value_slot(const unsigned short *x,
 const unsigned short *w, unsigned short *base, const int *pos, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a, b;
    unsigned short ah = x[k], bh = w[(long long)k * N + n];
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(ah));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(bh));
    sum = __fmaf_rn(a, b, sum);
  }
  unsigned short result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(sum));
  base[(long long)pos[0] * N + n] = result;
}
)CU";
  auto kernel =
    CudaContext::Global().registerCudaKernel(source, "kv_value_slot");
  if (!kernel)
    return false;
  auto *pos = cuda_pos_buffer();
  kernel->SetKernelArguments(0, &input, sizeof(input));
  kernel->SetKernelArguments(1, &weight, sizeof(weight));
  kernel->SetKernelArguments(2, &base, sizeof(base));
  kernel->SetKernelArguments(3, &pos, sizeof(pos));
  kernel->SetKernelArguments(4, &outputs, sizeof(outputs));
  kernel->SetKernelArguments(5, &inputs, sizeof(inputs));
  const int grid[3] = {(outputs + 127) / 128, 1, 1};
  const int block[3] = {128, 1, 1};
  return StreamManager::Global().DispatchCommand(*kernel, grid, block);
}

} // namespace nntrainer::cuda
