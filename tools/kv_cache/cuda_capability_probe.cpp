// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_capability_probe.cpp
 * @brief Model-free CUDA/NVRTC and sequential shared-backing handoff probe.
 * @author Codex (OpenAI)
 * @bug Does not measure physical migration or concurrent host/device access.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void runtimeCheck(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void driverCheck(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}
void compilerCheck(nvrtcResult status) {
  if (status != NVRTC_SUCCESS)
    throw std::runtime_error(nvrtcGetErrorString(status));
}
} // namespace

int main() {
  try {
    int count = 0;
    runtimeCheck(cudaGetDeviceCount(&count));
    if (count == 0)
      throw std::runtime_error("No CUDA device");
    runtimeCheck(cudaSetDevice(0));
    cudaDeviceProp properties{};
    runtimeCheck(cudaGetDeviceProperties(&properties, 0));
    int runtime = 0, driver = 0, nvrtc_major = 0, nvrtc_minor = 0;
    runtimeCheck(cudaRuntimeGetVersion(&runtime));
    runtimeCheck(cudaDriverGetVersion(&driver));
    compilerCheck(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
    int arch_count = 0;
    compilerCheck(nvrtcGetNumSupportedArchs(&arch_count));
    std::vector<int> architectures(arch_count);
    compilerCheck(nvrtcGetSupportedArchs(architectures.data()));
    const int architecture = properties.major * 10 + properties.minor;
    if (std::find(architectures.begin(), architectures.end(), architecture) ==
        architectures.end())
      throw std::runtime_error(
        "NVRTC does not support the device architecture");

    const char *source =
      "extern \"C\" __global__ void adjust(int *p, int count) {"
      "int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<count) p[i]+=7;}";
    nvrtcProgram program = nullptr;
    compilerCheck(
      nvrtcCreateProgram(&program, source, "handoff.cu", 0, nullptr, nullptr));
    const auto option = "--gpu-architecture=sm_" + std::to_string(architecture);
    const char *options[] = {option.c_str()};
    const auto compile = nvrtcCompileProgram(program, 1, options);
    if (compile != NVRTC_SUCCESS) {
      size_t bytes = 0;
      nvrtcGetProgramLogSize(program, &bytes);
      std::string log(bytes, '\0');
      nvrtcGetProgramLog(program, log.data());
      nvrtcDestroyProgram(&program);
      throw std::runtime_error(log);
    }
    size_t bytes = 0;
    compilerCheck(nvrtcGetCUBINSize(program, &bytes));
    std::vector<char> cubin(bytes);
    compilerCheck(nvrtcGetCUBIN(program, cubin.data()));
    compilerCheck(nvrtcDestroyProgram(&program));
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    driverCheck(cuModuleLoadData(&module, cubin.data()));
    driverCheck(cuModuleGetFunction(&function, module, "adjust"));

    constexpr int elements = 72, begin = 4, active = 64;
    std::vector<int> pattern(elements);
    for (int i = 0; i < elements; ++i)
      pattern[i] = 1000 + i;
    int *device = nullptr;
    runtimeCheck(cudaMalloc(&device, elements * sizeof(int)));
    runtimeCheck(cudaMemcpy(device, pattern.data(), elements * sizeof(int),
                            cudaMemcpyHostToDevice));
    auto launch = [&](int *base) {
      int *slice = base + begin;
      int size = active;
      void *arguments[] = {&slice, &size};
      driverCheck(cuLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                 arguments, nullptr));
      runtimeCheck(cudaDeviceSynchronize());
    };
    launch(device);
    std::vector<int> result(elements);
    runtimeCheck(cudaMemcpy(result.data(), device, elements * sizeof(int),
                            cudaMemcpyDeviceToHost));
    runtimeCheck(cudaFree(device));
    for (int i = 0; i < elements; ++i)
      if (result[i] != pattern[i] + (i >= begin && i < begin + active ? 7 : 0))
        throw std::runtime_error("Device pattern/guard mismatch");

    int *host = nullptr, *mapped = nullptr;
    runtimeCheck(
      cudaHostAlloc(&host, elements * sizeof(int), cudaHostAllocMapped));
    runtimeCheck(cudaHostGetDevicePointer(&mapped, host, 0));
    const bool same_address = mapped == host;
    for (int round = 0; round < 8; ++round) {
      for (int i = 0; i < elements; ++i)
        host[i] = pattern[i] + round;
      launch(mapped);
      for (int i = 0; i < elements; ++i)
        if (host[i] !=
            pattern[i] + round + (i >= begin && i < begin + active ? 7 : 0))
          throw std::runtime_error("Mapped handoff/guard mismatch");
    }
    runtimeCheck(cudaFreeHost(host));
    driverCheck(cuModuleUnload(module));

    std::cout << "{\"device\":" << std::quoted(properties.name)
              << ",\"compute_capability\":" << architecture
              << ",\"total_bytes\":" << properties.totalGlobalMem
              << ",\"runtime_version\":" << runtime
              << ",\"driver_api_version\":" << driver << ",\"nvrtc_version\":"
              << std::quoted(std::to_string(nvrtc_major) + "." +
                             std::to_string(nvrtc_minor))
              << ",\"managed_memory\":" << properties.managedMemory
              << ",\"concurrent_managed_access\":"
              << properties.concurrentManagedAccess
              << ",\"can_map_host_memory\":" << properties.canMapHostMemory
              << ",\"unified_addressing\":" << properties.unifiedAddressing
              << ",\"host_native_atomics\":"
              << properties.hostNativeAtomicSupported
              << ",\"host_mapped_same_address\":"
              << (same_address ? "true" : "false")
              << ",\"device_pattern\":\"pass\",\"mapped_handoff_rounds\":8"
              << ",\"physical_migration\":\"unmeasured\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
