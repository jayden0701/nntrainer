// SPDX-License-Identifier: Apache-2.0
/**
 * @file   t5gemma2_image_preprocess.h
 * @brief  Image preprocessing for T5Gemma2 model
 * @date   2025-02-24
 * @author Cline SR
 */

#ifndef __T5GEMMA2_IMAGE_PREPROCESS_H__
#define __T5GEMMA2_IMAGE_PREPROCESS_H__

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @brief Preprocess image for T5Gemma2 model
 *
 * This function:
 * 1. Loads and resizes the image to 896x896 using loadAndPreprocessImage()
 * 2. Applies T5Gemma2-specific normalization:
 *    - Rescale: pixel_value / 255.0
 *    - Normalize: (pixel_value - mean) / std
 *    - Where mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
 *
 * @param filepath Path to the input image file
 * @return std::vector<float> Preprocessed image data in CHW format (3, 896, 896)
 */
std::vector<float> preprocessT5Gemma2Image(const std::string &filepath);

/**
 * @brief Preprocess multiple images for T5Gemma2 model
 *
 * This function processes multiple images and returns them as a batch.
 * Each image is preprocessed independently and then stacked into a batch tensor.
 *
 * @param filepaths List of paths to input image files
 * @return std::vector<float> Preprocessed image data in NCHW format (batch, 3, 896, 896)
 */
std::vector<float> preprocessT5Gemma2Images(const std::vector<std::string> &filepaths);

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
  const std::vector<float> &mean, const std::vector<float> &std);

/**
 * @brief Preprocess multiple images for T5Gemma2 model with custom parameters
 *
 * @param filepaths List of paths to input image files
 * @param target_width Target image width
 * @param target_height Target image height
 * @param mean Mean values for normalization (one per channel or single value)
 * @param std Standard deviation values for normalization (one per channel or single
 * value)
 * @return std::vector<float> Preprocessed image data in NCHW format (batch, 3, height, width)
 */
std::vector<float> preprocessT5Gemma2ImagesCustom(
  const std::vector<std::string> &filepaths, int target_width, int target_height,
  const std::vector<float> &mean, const std::vector<float> &std);

} // namespace nntrainer

#endif // __T5GEMMA2_IMAGE_PREPROCESS_H__
