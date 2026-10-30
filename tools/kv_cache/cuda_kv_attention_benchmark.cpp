// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_kv_attention_benchmark.cpp
 * @brief CUDA-event profiling of actual split-KV decode graph kernels.
 * @author Codex (OpenAI)
 * @bug Measures repeated fixed operands, not end-to-end model latency.
 */
#include <algorithm>
#include <cstdlib>
#include <cuda_attention.h>
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
void driverCheck(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
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
      throw std::runtime_error("attention capture failed");
    check(cudaGraphInstantiate(&exec, graph, 0));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int i = 0; i < 3; ++i)
      check(cudaGraphLaunch(exec, sm.GetStream()));
    check(cudaStreamSynchronize(sm.GetStream()));
  }
  void cloneNode(const CUDA_KERNEL_NODE_PARAMS &params) {
    check(cudaGraphCreate(&graph, 0));
    CUgraphNode previous = nullptr;
    for (int i = 0; i < 100; ++i) {
      CUgraphNode node = nullptr;
      driverCheck(cuGraphAddKernelNode(&node, reinterpret_cast<CUgraph>(graph),
                                       previous ? &previous : nullptr,
                                       previous ? 1 : 0, &params));
      previous = node;
    }
    check(cudaGraphInstantiate(&exec, graph, 0));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    auto stream = cuda::StreamManager::Global().GetStream();
    for (int i = 0; i < 3; ++i)
      check(cudaGraphLaunch(exec, stream));
    check(cudaStreamSynchronize(stream));
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

} // namespace
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: cuda_kv_attention_benchmark NEW.json\n";
    return 2;
  }
  try {
    if (std::ifstream(argv[1]).good())
      throw std::invalid_argument("output exists");
    cuda::ContextManager::Global().EnsureCurrent();
    auto &sm = cuda::StreamManager::Global();
    auto allocator = std::make_shared<CudaMemAllocator>(true);
    auto ops = cuda::makeKvStorageOperations();
    nlohmann::json report = {
      {"device", cuda::ContextManager::Global().GetDeviceName()},
      {"scope",
       "CUDA events; each sample averages 100 calls; fixed device operands"},
      {"rows", nlohmann::json::array()}};
    for (const char *key : {"NNTR_CUDA_FLASH_DECODE", "NNTR_CUDA_ATTN_FUSE",
                            "NNTR_CUDA_DECODE_GRAPH", "NNTR_CUDA_GRAPH",
                            "NNTR_CUDA_M2B", "NNTR_CUDA_ASYNC"}) {
      const char *value = std::getenv(key);
      report["environment"][key] = value ? value : "(unset)";
    }
    for (auto shape :
         std::vector<std::pair<int, int>>{{16, 128}, {8, 256}, {8, 512}}) {
      const int HQ = shape.first, HKV = HQ == 16 ? 8 : 1, d = shape.second,
                capacity = 8192;
      if (!cuda::cuda_attention_splitkv_prewarm(capacity, HQ, d))
        throw std::runtime_error("prewarm");
      std::vector<__half> q(size_t(HQ) * d), k(size_t(capacity) * HKV * d),
        v(k.size());
      for (size_t i = 0; i < q.size(); ++i)
        q[i] = __float2half_rn(float(int((i * 13) % 29) - 14) / 17);
      for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __float2half_rn(float(int((i * 19) % 37) - 18) / 71);
        v[i] = __float2half_rn(float(int((i * 7) % 43) - 21) / 53);
      }
      auto qi = kv::KvStorage::create(q.size() * 2, allocator, ops),
           ki = kv::KvStorage::create(k.size() * 2, allocator, ops),
           vi = kv::KvStorage::create(v.size() * 2, allocator, ops),
           oi = kv::KvStorage::create(q.size() * 2, allocator, ops);
      qi->write(q.data(), 0, q.size() * 2);
      ki->write(k.data(), 0, k.size() * 2);
      vi->write(v.data(), 0, v.size() * 2);
      for (int length : {65, 129, 256, 257, 512, 1024, 4096, 4097, 8192}) {
        cuda::cuda_set_pos(length - 1, length);
        auto run = [&]() {
          return cuda::cuda_attention_interleaved_fp16(
            (unsigned short *)qi->data(), (unsigned short *)ki->data(),
            (unsigned short *)vi->data(), (unsigned short *)oi->data(), HQ, HKV,
            1, length, length - 1, d, capacity, 0);
        };
        if (!run())
          throw std::runtime_error("attention");
        ops->synchronize();
        TimedGraph whole;
        whole.capture(run);
        size_t count = 0;
        check(cudaGraphGetNodes(whole.graph, nullptr, &count));
        std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(whole.graph, nodes.data(), &count));
        CUDA_KERNEL_NODE_PARAMS pp{}, rp{};
        std::string partial_name, reduce_name;
        for (auto node : nodes) {
          cudaGraphNodeType type;
          check(cudaGraphNodeGetType(node, &type));
          if (type != cudaGraphNodeTypeKernel)
            continue;
          CUDA_KERNEL_NODE_PARAMS params{};
          driverCheck(cuGraphKernelNodeGetParams(
            reinterpret_cast<CUgraphNode>(node), &params));
          const char *name = nullptr;
          if (cuFuncGetName(&name, params.func) != CUDA_SUCCESS)
            throw std::runtime_error("function name");
          std::string n = name;
          if (n.find("attn_partial") == 0) {
            pp = params;
            partial_name = n;
          }
          if (n.find("attn_reduce") == 0) {
            rp = params;
            reduce_name = n;
          }
        }
        if (!pp.func || !rp.func)
          throw std::runtime_error("expected split-KV kernels");
        int registers = 0, local_bytes = 0;
        driverCheck(
          cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, pp.func));
        driverCheck(cuFuncGetAttribute(
          &local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, pp.func));
        TimedGraph partial, reduce;
        partial.cloneNode(pp);
        reduce.cloneNode(rp);
        auto metrics = [](TimedGraph &g) {
          std::vector<double> a;
          for (int i = 0; i < 9; ++i)
            a.push_back(g.measure());
          std::sort(a.begin(), a.end());
          return nlohmann::json{
            {"median_us", a[4]}, {"p95_us", a.back()}, {"samples", a}};
        };
        std::vector<unsigned short> bits(q.size());
        oi->read(bits.data(), 0, bits.size() * 2);
        nlohmann::json row = {
          {"hq", HQ},
          {"hkv", HKV},
          {"d", d},
          {"length", length},
          {"partial_name", partial_name},
          {"partial_registers", registers},
          {"partial_local_bytes", local_bytes},
          {"reduce_name", reduce_name},
          {"partial_grid", {pp.gridDimX, pp.gridDimY, pp.gridDimZ}},
          {"reduce_block", rp.blockDimX},
          {"whole", metrics(whole)},
          {"partial", metrics(partial)},
          {"reduce", metrics(reduce)},
          {"output_bits", bits}};
        report["rows"].push_back(row);
        std::cerr << HQ << "x" << d << " T=" << length
                  << " whole=" << row["whole"]["median_us"]
                  << " partial=" << row["partial"]["median_us"]
                  << " reduce=" << row["reduce"]["median_us"] << "\n";
      }
    }
    std::ofstream out(argv[1]);
    out << report.dump(2) << "\n";
    if (!out)
      throw std::runtime_error("write output");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
