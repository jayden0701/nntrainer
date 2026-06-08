// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_connector.h
 * @date   04 June 2026
 * @brief  Pixel-unshuffle + MLP connector for LFM2-VL.
 *
 *         Architecture (LiquidAI LFM2-VL-450M config.json):
 *           pixel_unshuffle x2 (downsample_factor=2)
 *           LayerNorm(3072) -> linear 3072->2560 -> gelu -> linear 2560->1024
 *         (projector_hidden_size=2560, text hidden_size=1024).
 *
 *         This is a NEW connector separate from the V-JEPA VoRA merger.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LFM2_VL_CONNECTOR_H__
#define __LFM2_VL_CONNECTOR_H__

#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Apply pixel-unshuffle (space-to-depth) on a flat patch sequence.
 *
 * Input shape:  [N, C]  where N = H*W patches, C = embed_dim.
 * Output shape: [N/(r*r), C*(r*r)]  where r = downsample_factor.
 *
 * @param features   Input feature vector (N * C floats, row-major).
 * @param n_patches  Number of input patches (N = H * W).
 * @param embed_dim  Embedding dimension per patch (C).
 * @param patch_h    Spatial grid height (sqrt(N) for square grids).
 * @param patch_w    Spatial grid width.
 * @param factor     Downsampling factor r (must divide patch_h and patch_w).
 * @return Pixel-unshuffled feature vector [(N/(r*r)) * (C*r*r) floats].
 */
std::vector<float> pixelUnshuffle(const std::vector<float> &features,
                                  unsigned int n_patches, unsigned int embed_dim,
                                  unsigned int patch_h, unsigned int patch_w,
                                  unsigned int factor);

/**
 * @brief Lfm2VlConnector: MLP projection after pixel-unshuffle.
 *
 * Weights are stored as plain FP32 row-major arrays loaded from a binary
 * checkpoint. All three linear layers use bias.
 *
 *   in_features  = embed_dim * factor * factor  (default 768*4=3072)
 *   hidden_size  = projector_hidden_size         (default 2560)
 *   out_features = text hidden_size              (default 1024)
 */
class Lfm2VlConnector {
public:
  /**
   * @brief Construct with explicit dimensions.
   * @param in_features  Input dimension after pixel-unshuffle (3072 default).
   * @param hidden_size  Projector hidden dimension (2560 default).
   * @param out_features Output dimension = LLM hidden size (1024 default).
   */
  Lfm2VlConnector(unsigned int in_features = 3072,
                  unsigned int hidden_size = 2560,
                  unsigned int out_features = 1024);

  /**
   * @brief Load weights from a flat binary file.
   *
   * Expected layout (all FP32, row-major):
   *   ln_weight  [in_features]
   *   ln_bias    [in_features]
   *   fc1_weight [hidden_size, in_features]
   *   fc1_bias   [hidden_size]
   *   fc2_weight [out_features, hidden_size]
   *   fc2_bias   [out_features]
   *
   * @param weight_path Absolute path to the connector weight binary.
   */
  void loadWeights(const std::string &weight_path);

  /**
   * @brief Run the MLP on already pixel-unshuffled features.
   *
   * @param x      Input: [n_out_patches * in_features] floats.
   * @param n_patches Number of output patches after pixel-unshuffle.
   * @return Output: [n_patches * out_features] floats.
   */
  std::vector<float> forward(const std::vector<float> &x,
                             unsigned int n_patches) const;

  /**
   * @brief Pixel-unshuffle raw vision features and run the MLP projection.
   *
   * @param features  Input: [n_patches * embed_dim] raw vision features.
   * @param n_patches Number of input vision patches.
   * @param embed_dim Embedding dimension per input patch.
   * @param patch_h   Input patch grid height.
   * @param patch_w   Input patch grid width.
   * @return Output: [outTokens(n_patches) * out_features] floats.
   */
  std::vector<float> project(const std::vector<float> &features,
                             unsigned int n_patches,
                             unsigned int embed_dim,
                             unsigned int patch_h,
                             unsigned int patch_w) const;

  /** @brief Number of output tokens (patches after unshuffle). */
  unsigned int outTokens(unsigned int n_input_patches) const {
    unsigned int r = factor_;
    return n_input_patches / (r * r);
  }

  unsigned int inFeatures() const { return in_features_; }
  unsigned int outFeatures() const { return out_features_; }
  unsigned int hiddenSize() const { return hidden_size_; }
  unsigned int downsampleFactor() const { return factor_; }

private:
  unsigned int in_features_;
  unsigned int hidden_size_;
  unsigned int out_features_;
  unsigned int factor_{2}; /**< pixel-unshuffle downsample factor */

  std::vector<float> fc1_weight_; /**< [hidden_size, in_features] */
  std::vector<float> fc1_bias_;   /**< [hidden_size] */
  std::vector<float> fc2_weight_; /**< [out_features, hidden_size] */
  std::vector<float> fc2_bias_;   /**< [out_features] */

  bool weights_loaded_{false};
  std::vector<float> ln_weight_; /**< LayerNorm weight [in_features] */
  std::vector<float> ln_bias_;   /**< LayerNorm bias [in_features] */

  /** @brief Element-wise GELU (erf-based, exact). */
  static float gelu(float x);

  /** @brief Matrix-vector multiply: y = W * x + b. */
  static std::vector<float> linearForward(const std::vector<float> &W,
                                          const std::vector<float> &b,
                                          const std::vector<float> &x,
                                          unsigned int rows,
                                          unsigned int cols);

  /** @brief Layer normalization over a single feature vector. */
  static std::vector<float> layerNorm(const std::vector<float> &x,
                                      const std::vector<float> &w,
                                      const std::vector<float> &b,
                                      float eps = 1e-5f);
};

} // namespace causallm

#endif // __LFM2_VL_CONNECTOR_H__
