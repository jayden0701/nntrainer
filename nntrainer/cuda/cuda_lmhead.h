// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_lmhead.h
 * @brief Layer-owned resident FP32 tied-head projection.
 * @author Codex (OpenAI)
 * @bug Stream capture is deliberately unsupported.
 */
#ifndef __CUDA_LMHEAD_H__
#define __CUDA_LMHEAD_H__
#include <cuda_buffer.h>

namespace nntrainer::cuda {
/** @brief Own the immutable head weight and reusable device staging buffers. */
class Fp32LmHead {
public:
  /** @brief Project one FP16 row; false requests the host fallback.
   * @note Weight contents must stay immutable until this owner is destroyed.
   *       Layer weight reload destroys the owner before reading new weights.
   */
  bool run(const float *weight, const unsigned short *input, void *output,
           unsigned int vocab, unsigned int hidden, bool output_fp16);
  /** @brief Bytes retained in device memory by this layer. */
  size_t residentBytes() const {
    return weight_.size() + input_.size() + output_.size();
  }

private:
  Buffer weight_, input_, output_;
  const float *key_ = nullptr;
  unsigned int vocab_ = 0, hidden_ = 0;
};
} // namespace nntrainer::cuda
#endif
