// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.h
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief   This timm_vit_transformer.h constructs a class for timm ViT model
 * compatible with the PyTorch timm library.
 */

#ifndef __T5GEMMA2_H__
#define __T5GEMMA2_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief T5Gemma2Transformer class
 */
class T5Gemma2Transformer : virtual public Transformer {

public:

  // TODO : divide architecture to T5Gemma2 / T5Gemma2ForConditionalGeneration
  static constexpr const char *architectures = "T5Gemma2ForConditionalGeneration";

  T5Gemma2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);

    }

  virtual ~T5Gemma2Transformer() = default;

public:
  std::vector<LayerHandle> createPatchEmbed();
  std::vector<LayerHandle> createAttention(const int layer_id,
                                           const std::string &input_name);
  std::vector<LayerHandle> createMlp(const int layer_id,
                                     const std::string &input_name);

protected:
  void constructModel() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  void registerCustomLayers() override;

  /**
   * @brief Run the model (override for ViT specific behavior)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "") override;

  /**
   * @brief Initialize (override to skip compile/initialize for T5Gemma2)
   */
  void initialize() override;

  /**
   * @brief Check if the model has image input by looking for BOI_TOKEN
   * @param input_text the input for model (for checking if it contains image)
   * @return true if the model has image input (BOI_TOKEN found), false otherwise
   */
  bool checkImageInput(const std::string &input_text) override;

private:

  // TODO : get from config

  unsigned int IMG_SIZE = 224;    /**< Image height/width */
  unsigned int PATCH_SIZE = 16;   /**< Patch height/width */
  unsigned int NUM_PATCHES = 196; /**< Number of patches */
  unsigned int IMG_CHANNELS = 3;  /**< Image channels (RGB) */

  std::string BOI_TOKEN = "<start_of_image>";

};

} // namespace causallm

#endif /* __TIMM_VIT_TRANSFORMER_H__ */
