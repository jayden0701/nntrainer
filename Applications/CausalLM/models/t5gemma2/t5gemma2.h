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

#include <encoder_decoder.h>
#include "t5gemma2_processor.h"
#include <memory>

namespace causallm {

/**
 * @brief T5Gemma2Transformer class
 */
class T5Gemma2Transformer : virtual public EncoderDecoder {

public:
  // TODO : divide architecture to T5Gemma2 / T5Gemma2ForConditionalGeneration
  static constexpr const char *architectures =
    "T5Gemma2ForConditionalGeneration";

  T5Gemma2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    EncoderDecoder(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);

    // Initialize processor (TODO: get parameters from config)
    processor = std::make_unique<nntrainer::T5Gemma2Processor>(256, 256000);
  }

  virtual ~T5Gemma2Transformer() = default;

public:
  std::vector<LayerHandle> createPatchEmbed();
  
  std::vector<LayerHandle> createMergedAttention(
  std::string prefix, const int layer_id, int seq_len, int n_heads,
  int head_dim, int gqa_size, std::string query_name, std::string key_name,
  std::string value_name, std::string cross_key_name, std::string cross_value_name);

  std::vector<LayerHandle>
  createSelfAttention(std::string prefix, const int layer_id, int seq_len,
                      int n_heads, int head_dim, int gqa_size,
                      std::string query_name, std::string key_name,
                      std::string value_name);
  std::vector<LayerHandle> createMlp(std::string prefix, int dim,
                                     int hidden_dim, std::string input_name);

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
   * @return true if the model has image input (BOI_TOKEN found), false
   * otherwise
   */
  bool checkImageInput(const std::string &input_text) override;

  /**
   * @brief Load model weights from file (override parent)
   */
  void load_weight(const std::string &weight_path) override;

private:
  /**
   * @brief Generate next token from logits
   * @param logits Logits from model output
   * @param do_sample Whether to use sampling (true) or argmax (false)
   * @return Generated token ID
   */
  std::vector<unsigned int> generate(float *logits, bool do_sample);

  /**
   * @brief Run encoder inference with lazy initialization
   * @param input_data Input data pointer
   * @param input_len Input sequence length
   * @return Encoder output as vector
   */
  std::vector<float> runEncoder(float *input_data, unsigned int input_len);

  /**
   * @brief Run decoder inference with lazy initialization
   * @param encoder_output Encoder output for cross-attention
   * @return Generated text
   */
  std::string runDecoder(const std::vector<float> &encoder_output);

  /**
   * @brief Create encoder model (compile only)
   */
  void createEncoderModel();

  /**
   * @brief Create decoder model (compile only)
   */
  void createDecoderModel();

  /**
   * @brief Load encoder weights from file
   */
  void loadEncoderWeights(const std::string &weight_path);

  /**
   * @brief Load decoder weights from file
   */
  void loadDecoderWeights(const std::string &weight_path);

  // Separate models for encoder and decoder (lazy initialization)
  std::unique_ptr<ml::train::Model> encoder_model;
  std::unique_ptr<ml::train::Model> decoder_model;
  
  // Compile state flags (not initialized with memory)
  bool encoder_compiled = false;
  bool decoder_compiled = false;
  bool encoder_initialized = false;
  bool decoder_initialized = false;
  bool encoder_weights_loaded = false;
  bool decoder_weights_loaded = false;
  
  // Memory tracking
  size_t encoder_memory_size = 0;
  size_t decoder_memory_size = 0;
  
  // Weight file paths (for lazy loading)
  std::string encoder_weight_path;
  std::string decoder_weight_path;
  // For text generation
  std::vector<int> pending_ids_; /**< Pending token IDs for decoding */
  std::vector<std::string> output_list; /**< Generated output text */

  //
  int ACTUAL_SEQ_LEN;


  // shared configuration
  int TOKEN_INDEX_EOI;
  int TOKEN_INDEX_IMAGE;

  int EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;

  // Encoder configuration (text encoder for vision model)
  int ENC_NUM_LAYERS;
  int ENC_NUM_HEADS;
  int ENC_NUM_KEY_VALUE_HEADS;
  int ENC_HEAD_DIM;
  int ENC_HIDDEN_SIZE;
  int ENC_GQA_SIZE;
  int ENC_INTERMEDIATE_SIZE;
  int ENC_MAX_POSITION_EMBEDDINGS;
  float ENC_NORM_EPS;
  unsigned int ENC_SLIDING_WINDOW;
  float ENC_ROPE_THETA;
  float ENC_ROPE_THETA_SLIDING;

  bool ENC_USE_CROSS_ATTENTION;
  bool ENC_IS_BIDIRECTIONAL;
  int ENC_MLP_HIDDEN_SIZE;
  int ENC_MM_TOKENS_PER_IMAGE;
  int ENC_SLIDING_WINDOW_PATTERN;

  // Decoder configuration (text generation model)
  int DEC_NUM_LAYERS;
  int DEC_NUM_HEADS;
  int DEC_NUM_KEY_VALUE_HEADS;
  int DEC_HEAD_DIM;
  int DEC_HIDDEN_SIZE;
  int DEC_INTERMEDIATE_SIZE;
  int DEC_MAX_POSITION_EMBEDDINGS;
  float DEC_NORM_EPS;
  unsigned int DEC_SLIDING_WINDOW;
  float DEC_ROPE_THETA;
  float DEC_ROPE_THETA_SLIDING;   // RoPE theta for sliding attention layers
  float DEC_ROPE_THETA_FULL;      // RoPE theta for full attention layers
  int DEC_SLIDING_WINDOW_PATTERN; // Pattern for alternating attention types
  bool DEC_IS_CAUSAL;
  bool DEC_IS_BIDIRECTIONAL;
  int DEC_QUERY_PRE_ATTN_SCALAR;
  float DEC_ATTN_LOGIT_SOFTCAPPING;
  float DEC_FINAL_LOGIT_SOFTCAPPING;

  // Vision encoder configuration (SigLIP)
  int VISION_NUM_LAYERS;
  int VISION_NUM_CHANNELS;
  int VISION_HIDDEN_SIZE;
  int VISION_INTERMEDIATE_SIZE;
  int VISION_IMAGE_SIZE;
  int VISION_PATCH_SIZE;
  int VISION_NUM_PATCHES;
  int VISION_NUM_HEADS;
  int VISION_HEAD_DIM;
  float VISION_NORM_EPS;

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
