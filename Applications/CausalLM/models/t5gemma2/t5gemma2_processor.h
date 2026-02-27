// SPDX-License-Identifier: Apache-2.0
/**
 * @file   t5gemma2_processor.h
 * @brief  T5Gemma2 processor for handling text and image inputs
 * @date   2025-02-24
 * @author Cline SR
 */

#ifndef __T5GEMMA2_PROCESSOR_H__
#define __T5GEMMA2_PROCESSOR_H__

#include <string>
#include <vector>
#include <map>

namespace nntrainer {

/**
 * @brief Structure to hold processor outputs
 */
struct T5Gemma2ProcessorOutput {
  std::vector<float> pixel_values;  // Image tensor (batch, channels, height, width)
  std::vector<int> input_ids;        // Tokenized text input
  std::vector<int> attention_mask;   // Attention mask
  std::vector<int> token_type_ids;   // Token type IDs (0 for text, 1 for image)
  
  T5Gemma2ProcessorOutput() = default;
};

/**
 * @brief T5Gemma2 processor class
 * 
 * This processor handles both text and image inputs for the T5Gemma2 model.
 * It follows the pattern of the PyTorch Gemma3Processor.
 */
class T5Gemma2Processor {
public:
  /**
   * @brief Configuration for image processing
   */
  struct ImageProcessingConfig {
    int image_size = 896;
    int image_seq_length = 256;
    std::vector<float> image_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> image_std = {0.5f, 0.5f, 0.5f};
    bool do_resize = true;
    bool do_rescale = true;
    bool do_normalize = true;
    bool do_convert_rgb = true;
    
    ImageProcessingConfig() = default;
  };

  /**
   * @brief Configuration for text processing
   */
  struct TextProcessingConfig {
    bool add_bos_token = true;
    bool add_eos_token = false;
    bool padding = false;
    bool return_mm_token_type_ids = true;
    
    TextProcessingConfig() = default;
  };

  /**
   * @brief Constructor
   * 
   * @param image_seq_length Length of image token sequence
   * @param image_token_id Token ID for image tokens
   */
  T5Gemma2Processor(int image_seq_length = 256, int image_token_id = 256000);

  /**
   * @brief Process inputs (text and/or images)
   * 
   * @param text Input text (can be empty)
   * @param images List of image file paths (can be empty)
   * @return T5Gemma2ProcessorOutput Processed inputs
   */
  T5Gemma2ProcessorOutput process(
    const std::string &text,
    const std::vector<std::string> &images);

  /**
   * @brief Process inputs (images only)
   * 
   * @param images List of image file paths
   * @return T5Gemma2ProcessorOutput Processed inputs
   */
  T5Gemma2ProcessorOutput process(const std::vector<std::string> &images);

  /**
   * @brief Process inputs (text only)
   * 
   * @param text Input text
   * @return T5Gemma2ProcessorOutput Processed inputs
   */
  T5Gemma2ProcessorOutput process(const std::string &text);

  /**
   * @brief Set image processing configuration
   * 
   * @param config Image processing configuration
   */
  void setImageConfig(const ImageProcessingConfig &config);

  /**
   * @brief Set text processing configuration
   * 
   * @param config Text processing configuration
   */
  void setTextConfig(const TextProcessingConfig &config);

  /**
   * @brief Get image processing configuration
   * 
   * @return Current image processing configuration
   */
  const ImageProcessingConfig& getImageConfig() const { return image_config_; }

  /**
   * @brief Get text processing configuration
   * 
   * @return Current text processing configuration
   */
  const TextProcessingConfig& getTextConfig() const { return text_config_; }

  /**
   * @brief Get special tokens
   * 
   * @return Map of special token names to IDs
   */
  const std::map<std::string, int>& getSpecialTokens() const { return special_tokens_; }

  /**
   * @brief Enable or disable debug output
   * 
   * @param enable Whether to enable debug output
   */
  void setDebugOutput(bool enable) { debug_output_ = enable; }

private:
  // Configuration
  ImageProcessingConfig image_config_;
  TextProcessingConfig text_config_;
  
  // Special tokens
  std::map<std::string, int> special_tokens_;
  
  // Image sequence configuration
  int image_seq_length_;
  int image_token_id_;
  
  // Debug output flag
  bool debug_output_;

  // Special token names
  static const char* BOI_TOKEN;   // Beginning of image
  static const char* EOI_TOKEN;   // End of image
  static const char* IMAGE_TOKEN; // Image soft token

  /**
   * @brief Initialize special tokens
   */
  void initializeSpecialTokens();

  /**
   * @brief Preprocess images
   * 
   * @param images List of image file paths
   * @return std::vector<float> Preprocessed image tensor (batch, channels, height, width)
   */
  std::vector<float> preprocessImages(const std::vector<std::string> &images);

  /**
   * @brief Tokenize text
   * 
   * @param text Input text
   * @param image_placeholder_count Number of image tokens to insert
   * @return std::vector<int> Tokenized text
   */
  std::vector<int> tokenize(const std::string &text, int image_placeholder_count);

  /**
   * @brief Create attention mask
   * 
   * @param input_ids Token IDs
   * @return std::vector<int> Attention mask
   */
  std::vector<int> createAttentionMask(const std::vector<int> &input_ids);

  /**
   * @brief Create token type IDs
   * 
   * @param input_ids Token IDs
   * @param image_token_id Token ID for image tokens
   * @return std::vector<int> Token type IDs (0 for text, 1 for image)
   */
  std::vector<int> createTokenTypeIds(const std::vector<int> &input_ids, int image_token_id);

  /**
   * @brief Expand image placeholders in text
   * 
   * @param text Input text with image placeholders
   * @return std::string Text with expanded image tokens
   */
  std::string expandImagePlaceholders(const std::string &text);

  /**
   * @brief Get full image token sequence
   * 
   * @return std::string Full image token sequence (e.g., "<start_of_image><image_soft_token>...<end_of_image>")
   */
  std::string getFullImageSequence();
};

} // namespace nntrainer

#endif // __T5GEMMA2_PROCESSOR_H__
