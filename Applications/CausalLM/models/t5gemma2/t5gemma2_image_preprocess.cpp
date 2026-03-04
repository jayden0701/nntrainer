// SPDX-License-Identifier: Apache-2.0
/**
 * @file   t5gemma2_image_preprocess.cpp
 * @brief  Image preprocessing for T5Gemma2 model
 * @date   2025-02-24
 * @author Cline SR
 */

#include "t5gemma2_image_preprocess.h"
#include <llm_util.hpp>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iomanip>

namespace nntrainer {

/**
 * @brief Preprocess image for T5Gemma2 model with custom normalization parameters
 *
 * @param filepath Path to the input image file
 * @param target_width Target image width
 * @param target_height Target image height
 * @param mean Mean values for normalization (one per channel or single value)
 * @param std Standard deviation values for normalization (one per channel or single
 * value)
 * @return std::vector<float> Preprocessed image data in CHW format
 */
std::vector<float> preprocessT5Gemma2ImageCustom(
  const std::string &filepath, int target_width, int target_height,
  const std::vector<float> &mean, const std::vector<float> &std) {

  if (mean.empty() || mean.size() > 3) {
    throw std::runtime_error("mean must have 1-3 elements");
  }
  if (std.empty() || std.size() > 3) {
    throw std::runtime_error("std must have 1-3 elements");
  }

  const int channels = 3;

  // Step 1: Load and resize image without normalization
  std::vector<float> image_data =
    loadAndPreprocessImage(filepath, target_width, target_height, false);

  if (image_data.size() != channels * target_width * target_height) {
    throw std::runtime_error(
      "Invalid image data size after loading. Expected " +
      std::to_string(channels * target_width * target_height) + ", got " +
      std::to_string(image_data.size()));
  }

  // Step 2: Apply normalization with custom parameters
  const float rescale_factor = 1.0f / 255.0f;

  for (int c = 0; c < channels; ++c) {
    // Use channel-specific mean/std if provided, otherwise use the first value
    float channel_mean = (mean.size() > 1) ? mean[c] : mean[0];
    float channel_std = (std.size() > 1) ? std[c] : std[0];

    for (int h = 0; h < target_height; ++h) {
      for (int w = 0; w < target_width; ++w) {
        size_t idx = c * target_height * target_width + h * target_width + w;
        // Rescale from [0, 255] to [0, 1]
        float rescaled = image_data[idx] * rescale_factor;
        // Normalize: (rescaled - mean) / std
        image_data[idx] = (rescaled - channel_mean) / channel_std;
      }
    }
  }

  return image_data;
}


std::vector<float> preprocessT5Gemma2ImagesCustom(
  const std::vector<std::string> &filepaths, int target_width, int target_height,
  const std::vector<float> &mean, const std::vector<float> &std) {

  if (mean.empty() || mean.size() > 3) {
    throw std::runtime_error("mean must have 1-3 elements");
  }
  if (std.empty() || std.size() > 3) {
    throw std::runtime_error("std must have 1-3 elements");
  }

  const int channels = 3;
  const int batch_size = filepaths.size();

  if (batch_size == 0) {
    return std::vector<float>();
  }

  std::cout << "[T5Gemma2] Preprocessing " << batch_size << " images (custom params)" << std::endl;

  // Calculate total size: batch_size * channels * height * width
  const size_t single_image_size = channels * target_height * target_width;
  const size_t total_size = batch_size * single_image_size;

  std::vector<float> batch_data(total_size);

  // Process each image
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    std::cout << "[T5Gemma2] Processing image " << (batch_idx + 1) << "/" << batch_size 
              << ": " << filepaths[batch_idx] << std::endl;

    // Process single image
    std::vector<float> image_data = preprocessT5Gemma2ImageCustom(
      filepaths[batch_idx], target_width, target_height, mean, std);

    // Copy to batch output
    size_t batch_offset = batch_idx * single_image_size;
    std::copy(image_data.begin(), image_data.end(), batch_data.begin() + batch_offset);
  }

  std::cout << "[T5Gemma2] Preprocessing complete. Output shape: [" << batch_size 
            << ", " << channels << ", " << target_height << ", " << target_width << "]" << std::endl;

  return batch_data;
}

} // namespace nntrainer
