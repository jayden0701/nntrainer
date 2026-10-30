// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_kv_value_benchmark.cpp
 * @brief CUDA-event comparison of captured final V projections.
 * @author Codex (OpenAI)
 * @bug Measures repeated fixed operands, not end-to-end model latency.
 */
#include <algorithm>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_fp16.h>
#include <cuda_kernel.h>
#include <cuda_kv_storage.h>
#include <cuda_mem_allocator.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <json.hpp>
#include <kv_cache/kv_storage.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nntrainer;
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct TimedGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  ~TimedGraph() {
    if (begin)
      cudaEventDestroy(begin);
    if (end)
      cudaEventDestroy(end);
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
  }
  void capture(const std::function<bool()> &launch) {
    auto &sm = cuda::StreamManager::Global();
    if (!sm.beginCapture())
      throw std::runtime_error("begin capture failed");
    bool submitted = true;
    for (int i = 0; i < 100; ++i)
      submitted = launch() && submitted;
    const bool ended = sm.endCapture(&graph);
    if (!submitted || !ended)
      throw std::runtime_error("V projection capture failed");
    check(cudaGraphInstantiate(&exec, graph, 0));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int i = 0; i < 3; ++i)
      check(cudaGraphLaunch(exec, sm.GetStream()));
    check(cudaStreamSynchronize(sm.GetStream()));
  }
  double measure() {
    auto stream = cuda::StreamManager::Global().GetStream();
    check(cudaEventRecord(begin, stream));
    check(cudaGraphLaunch(exec, stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin, end));
    return ms * 1000.0 / 100; // average us/kernel in this 100-node batch
  }
};
const char *serial_source = R"CU(
extern "C" __global__ void kv_value_slot_serial_baseline(const unsigned short *x,
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
} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3 ||
      (argc == 3 && std::string(argv[2]) != "--mapped")) {
    std::cerr << "Usage: cuda_kv_value_benchmark NEW_OUTPUT.json [--mapped]\n";
    return 2;
  }
  try {
    if (std::ifstream(argv[1]).good())
      throw std::invalid_argument("output already exists");
    const bool mapped = argc == 3;
    cuda::ContextManager::Global().EnsureCurrent();
    auto &sm = cuda::StreamManager::Global();
    auto allocator = std::make_shared<CudaMemAllocator>(true);
    auto operations = cuda::makeKvStorageOperations();
    auto weight_allocator = std::make_shared<CudaMemAllocator>(!mapped);
    auto weight_operations = cuda::makeKvStorageOperations(
      mapped ? AllocationKind::CudaHostMapped : AllocationKind::CudaDevice);
    nlohmann::json result = {
      {"weight_kind", mapped ? "CudaHostMapped" : "CudaDevice"},
      {"device", cuda::ContextManager::Global().GetDeviceName()},
      {"timing_scope", "CUDA events around 100-node graph; us per node"},
      {"method", "ABBA, 3 batches/arm, four arms; 6 samples/arm"},
      {"warmup_graph_replays", 3},
      {"rows", nlohmann::json::array()}};
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime));
    check(cudaDriverGetVersion(&driver));
    result["cuda_runtime"] = runtime;
    result["cuda_driver"] = driver;
    for (const auto &shape : std::vector<std::pair<int, int>>{{31, 63},
                                                              {129, 65},
                                                              {128, 255},
                                                              {128, 256},
                                                              {128, 257},
                                                              {128, 511},
                                                              {128, 1024},
                                                              {1024, 1024},
                                                              {2048, 4096}}) {
      int N = shape.first, K = shape.second;
      std::vector<__half> x(K), w(size_t(N) * K);
      for (int k = 0; k < K; ++k)
        x[k] = __float2half_rn(float(k % 17 - 8) / 13.0f);
      for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2half_rn(float(int(i % 37) - 18) / 73.0f);
      auto input = kv::KvStorage::create(x.size() * 2, allocator, operations);
      auto weight = kv::KvStorage::create(w.size() * 2, weight_allocator,
                                          weight_operations);
      auto output =
        kv::KvStorage::create(size_t(N) * 3 * 2, allocator, operations);
      input->write(x.data(), 0, x.size() * 2);
      weight->write(w.data(), 0, w.size() * 2);
      auto *xp = static_cast<const unsigned short *>(input->data());
      auto *wp = static_cast<const unsigned short *>(weight->data());
      auto *yp = static_cast<unsigned short *>(output->data());
      cuda::cuda_set_pos(1, 2);
      auto *pos = cuda::cuda_pos_buffer();
      auto kernel = CudaContext::Global().registerCudaKernel(
        serial_source, "kv_value_slot_serial_baseline");
      if (!kernel)
        throw std::runtime_error("serial kernel unavailable");
      kernel->SetKernelArguments(0, &xp, sizeof(xp));
      kernel->SetKernelArguments(1, &wp, sizeof(wp));
      kernel->SetKernelArguments(2, &yp, sizeof(yp));
      kernel->SetKernelArguments(3, &pos, sizeof(pos));
      kernel->SetKernelArguments(4, &N, sizeof(N));
      kernel->SetKernelArguments(5, &K, sizeof(K));
      const int grid[3] = {(N + 127) / 128, 1, 1}, block[3] = {128, 1, 1};
      auto serial = [&]() { return sm.DispatchCommand(*kernel, grid, block); };
      auto candidate = [&]() { return cuda::kvValueSlot(xp, wp, yp, N, K); };
      std::vector<unsigned short> reference(size_t(N) * 3),
        actual(size_t(N) * 3);
      if (!serial())
        throw std::runtime_error("serial projection failed");
      output->read(reference.data(), 0, reference.size() * 2);
      // Compile candidate outside capture, then compare all rows/guard bytes.
      if (!candidate())
        throw std::runtime_error("candidate projection failed");
      output->read(actual.data(), 0, actual.size() * 2);
      if (reference != actual)
        throw std::runtime_error("projection bytes differ");
      TimedGraph baseline, production;
      baseline.capture(serial);
      production.capture(candidate);
      std::vector<double> a, b;
      for (int arm : {0, 1, 1, 0}) {
        auto &graph = arm ? production : baseline;
        for (int i = 0; i < 3; ++i)
          (arm ? b : a).push_back(graph.measure());
      }
      auto metrics = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return nlohmann::json{{"median_us", (values[2] + values[3]) / 2},
                              {"p95_us", values.back()},
                              {"batch_average_us", values}};
      };
      result["rows"].push_back({{"outputs", N},
                                {"inputs", K},
                                {"bitwise_equal", true},
                                {"serial", metrics(a)},
                                {"production", metrics(b)}});
    }
    std::ofstream out(argv[1]);
    out << result.dump(2) << "\n";
    if (!out)
      throw std::runtime_error("cannot write output");
    std::cout << result.dump(2) << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
