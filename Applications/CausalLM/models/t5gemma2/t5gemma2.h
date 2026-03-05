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
#include <memory>
#include "t5gemma2_processor.h"

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
    
    // Initialize processor (TODO: get parameters from config)
    processor = std::make_unique<nntrainer::T5Gemma2Processor>(256, 256000);
    }

  virtual ~T5Gemma2Transformer() = default;

public:
  std::vector<LayerHandle> createPatchEmbed();
  std::vector<LayerHandle> createEncoderAttention(const int layer_id,
                                           const std::string &input_name);
  std::vector<LayerHandle> createMlp(std::string prefix,
                                                      int dim, int hidden_dim,
                                                      std::string input_name);

protected:
  void constructModel() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create encoder layers for T5Gemma2 text encoder
   * @param input_name Name of the input layer
   * @return encoder layers
   */
  std::vector<LayerHandle> createEncoder(const std::string &input_name);



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

  // TODO : change these to ENC_ / DEC_ / VISION_ 

  int ENC_MLP_HIDDEN_SIZE;

  unsigned int IMG_SIZE = 224;    /**< Image height/width */
  unsigned int PATCH_SIZE = 16;   /**< Patch height/width */
  unsigned int NUM_PATCHES = 196; /**< Number of patches */
  unsigned int IMG_CHANNELS = 3;  /**< Image channels (RGB) */

  std::string BOI_TOKEN = "<start_of_image>";
  
  /** T5Gemma2 processor for multimodal input processing */
  std::unique_ptr<nntrainer::T5Gemma2Processor> processor;

};

} // namespace causallm

#endif /* __TIMM_VIT_TRANSFORMER_H__ */
