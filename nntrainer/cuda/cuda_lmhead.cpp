// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_lmhead.cpp
 * @brief Resident FP32 weights times FP16 hidden, FP32 accumulation on CUDA.
 * @author Codex (OpenAI)
 * @bug Stream capture is deliberately unsupported.
 */
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_lmhead.h>
#include <cuda_stream_manager.h>
#include <limits>
#include <utility>

namespace nntrainer::cuda {
static const char *HEAD_SRC = R"CU(
extern "C" __global__ void fp32_tied_head(const float *w,
    const unsigned short *x, void *y, unsigned int n, unsigned int k, int half_out) {
  unsigned int lane = threadIdx.x & 31, row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= n) return;
  float sum = 0;
  for (unsigned int j = lane; j < k; j += 32) {
    float v;
    unsigned short h = x[j];
    asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h));
    sum = fmaf(w[(unsigned long long)row * k + j], v, sum);
  }
  for (int offset = 16; offset; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane == 0) {
    if (half_out) {
      unsigned short h;
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(sum));
      ((unsigned short *)y)[row] = h;
    } else ((float *)y)[row] = sum;
  }
}
)CU";

bool Fp32LmHead::run(const float *weight, const unsigned short *input,
                     void *output, unsigned int vocab, unsigned int hidden,
                     bool output_fp16) {
  if (!engine_selected() || !weight || !input || !output || !vocab || !hidden ||
      vocab > static_cast<unsigned int>(std::numeric_limits<int>::max() - 7) ||
      hidden > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
      static_cast<size_t>(vocab) >
        std::numeric_limits<size_t>::max() / sizeof(float) / hidden)
    return false;
  auto &sm = StreamManager::Global();
  if (sm.isCapturing())
    return false;
  auto kernel =
    CudaContext::Global().registerCudaKernel(HEAD_SRC, "fp32_tied_head");
  if (!kernel)
    return false;
  const size_t output_bytes =
    static_cast<size_t>(vocab) * (output_fp16 ? 2 : 4);
  if (key_ != weight || vocab_ != vocab || hidden_ != hidden) {
    // Synchronize producers before the initial upload, including async mode.
    sm.finish();
    Buffer w(static_cast<size_t>(vocab) * hidden * sizeof(float));
    Buffer x(static_cast<size_t>(hidden) * sizeof(unsigned short));
    Buffer y(static_cast<size_t>(vocab) * sizeof(float));
    if (!w.GetBuffer() || !x.GetBuffer() || !y.GetBuffer() ||
        cudaMemcpy(w.GetBuffer(), weight, w.size(), cudaMemcpyDefault) !=
          cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    weight_ = std::move(w);
    input_ = std::move(x);
    output_ = std::move(y);
    key_ = weight;
    vocab_ = vocab;
    hidden_ = hidden;
  }
  auto stream = sm.GetStream();
  auto *w = static_cast<const float *>(weight_.GetBuffer());
  auto *x = static_cast<const unsigned short *>(input_.GetBuffer());
  void *y = output_.GetBuffer();
  int half_out = output_fp16 ? 1 : 0;
  auto failed = [&sm, stream]() {
    cudaStreamSynchronize(stream);
    cudaGetLastError();
    return false;
  };
  if (cudaMemcpyAsync(input_.GetBuffer(), input, input_.size(),
                      cudaMemcpyDefault, stream) != cudaSuccess)
    return failed();
  kernel->SetKernelArguments(0, &w, sizeof(w));
  kernel->SetKernelArguments(1, &x, sizeof(x));
  kernel->SetKernelArguments(2, &y, sizeof(y));
  kernel->SetKernelArguments(3, &vocab, sizeof(vocab));
  kernel->SetKernelArguments(4, &hidden, sizeof(hidden));
  kernel->SetKernelArguments(5, &half_out, sizeof(half_out));
  const int grid[3] = {static_cast<int>((vocab + 7) / 8), 1, 1};
  const int block[3] = {256, 1, 1};
  if (!sm.DispatchCommand(*kernel, grid, block))
    return failed();
  if (cudaMemcpyAsync(output, y, output_bytes, cudaMemcpyDefault, stream) !=
      cudaSuccess)
    return failed();
  // The following logits/bias consumers may run on the host in async mode.
  return cudaStreamSynchronize(stream) == cudaSuccess;
}
} // namespace nntrainer::cuda
