// SPDX-License-Identifier: Apache-2.0
/**
 * @file   t5gemma2_processor.cpp
 * @brief  T5Gemma2 processor implementation
 * @date   2025-02-24
 * @author Cline SR
 */

#include "t5gemma2_processor.h"
#include "t5gemma2_image_preprocess.h"
#include <llm_util.hpp>
#include <algorithm>
#include <sstream>
#include <regex>

namespace nntrainer {

// Static member definitions
const char* T5Gemma2Processor::BOI_TOKEN = "<start_of_image>";
const char* T5Gemma2Processor::EOI_TOKEN = "<end_of_image>";
const char* T5Gemma2Processor::IMAGE_TOKEN = "<image_soft_token>";

T5Gemma2Processor::T5Gemma2Processor(int image_seq_length, int image_token_id)
    : image_seq_length_(image_seq_length),
      image_token_id_(image_token_id),
      debug_output_(true) {
  
  initializeSpecialTokens();
  
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Initialized" << std::endl;
    std::cout << "  image_seq_length: " << image_seq_length_ << std::endl;
    std::cout << "  image_token_id: " << image_token_id_ << std::endl;
    std::cout << "  boi_token: " << BOI_TOKEN << std::endl;
    std::cout << "  eoi_token: " << EOI_TOKEN << std::endl;
    std::cout << "  image_token: " << IMAGE_TOKEN << std::endl;
    std::cout << "  full_image_sequence: " << getFullImageSequence().substr(0, 50) << "..." << std::endl;
  }
}

void T5Gemma2Processor::initializeSpecialTokens() {
  // Initialize special tokens with placeholder IDs
  // These should be loaded from the tokenizer configuration in the future
  special_tokens_[BOI_TOKEN] = 255999;
  special_tokens_[EOI_TOKEN] = 256001;
  special_tokens_[IMAGE_TOKEN] = image_token_id_;
  special_tokens_["<bos>"] = 1;
  special_tokens_["<eos>"] = 2;
  special_tokens_["<pad>"] = 0;
  special_tokens_["<unk>"] = 3;
}

T5Gemma2ProcessorOutput T5Gemma2Processor::process(
    const std::string &text,
    const std::vector<std::string> &images) {
  
  if (debug_output_) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[T5Gemma2Processor::process]" << std::endl;
    std::cout << "  Input text: " << (text.empty() ? "(empty)" : text) << std::endl;
    std::cout << "  Input images: " << images.size() << " image(s)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
  }
  
  T5Gemma2ProcessorOutput output;
  
  // Validate inputs
  if (text.empty() && images.empty()) {
    throw std::runtime_error("Provide at least one of text or images.");
  }
  
  // Preprocess images if provided
  if (!images.empty()) {
    output.pixel_values = preprocessImages(images);
  }
  
  // Process text if provided or create placeholder for images
  std::string processed_text = text;
  
  if (images.empty() && !text.empty()) {
    // case 1) Text only processing
    if (debug_output_) {
      std::cout << "[T5Gemma2Processor] Processing text only" << std::endl;
    }
    output.input_ids = tokenize(processed_text, 0);
  } else if (!images.empty()) {
    // case 2) Images only or mixed processing

    int image_placeholder_count = images.size();
    
    if (text.empty()) {
      // create BOI_TOKEN for image holding
      processed_text = "";
      for (size_t i = 0; i < images.size(); ++i) {
        processed_text += BOI_TOKEN;
        processed_text += " ";
      }
      if (debug_output_) {
        std::cout << "[T5Gemma2Processor] Created placeholder text for images only" << std::endl;
      }
    }
    
    // Expand image placeholders in text
    processed_text = expandImagePlaceholders(processed_text);
    
    if (debug_output_) {
      std::cout << "[T5Gemma2Processor] Text after expansion: " 
                << processed_text << "..." << std::endl;
    }
    
    output.input_ids = tokenize(processed_text, image_placeholder_count);
  }
  
  // Create attention mask
  output.attention_mask = createAttentionMask(output.input_ids);
  
  // Create token type IDs if requested
  if (text_config_.return_mm_token_type_ids) {
    output.token_type_ids = createTokenTypeIds(output.input_ids, image_token_id_);
  }
  
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Processing complete" << std::endl;
    std::cout << "  input_ids length: " << output.input_ids.size() << std::endl;
    std::cout << "  pixel_values shape: [" 
              << (output.pixel_values.empty() ? 0 : output.pixel_values.size() / (3 * image_config_.image_size * image_config_.image_size))
              << ", 3, " << image_config_.image_size << ", " << image_config_.image_size << "]" << std::endl;
    std::cout << "  token_type_ids length: " << output.token_type_ids.size() << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
  }
  
  return output;
}

T5Gemma2ProcessorOutput T5Gemma2Processor::process(const std::vector<std::string> &images) {
  return process("", images);
}

T5Gemma2ProcessorOutput T5Gemma2Processor::process(const std::string &text) {
  return process(text, std::vector<std::string>());
}

void T5Gemma2Processor::setImageConfig(const ImageProcessingConfig &config) {
  image_config_ = config;
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Updated image config" << std::endl;
  }
}

void T5Gemma2Processor::setTextConfig(const TextProcessingConfig &config) {
  text_config_ = config;
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Updated text config" << std::endl;
  }
}

std::vector<float> T5Gemma2Processor::preprocessImages(const std::vector<std::string> &images) {
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Preprocessing " << images.size() << " image(s)" << std::endl;
  }
  
  // Use the multi-image preprocessing function
  return preprocessT5Gemma2ImagesCustom(
    images, 
    image_config_.image_size, 
    image_config_.image_size,
    image_config_.image_mean,
    image_config_.image_std
  );
}


// TODO change to NNTrainer Tokenizer
std::vector<int> T5Gemma2Processor::tokenize(const std::string &text, int image_placeholder_count) {
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Tokenizing text with " 
              << image_placeholder_count << " image placeholder(s)" << std::endl;
  }
  
  // TODO: Integrate with actual tokenizer from huggingface_tokenizer
  // For now, this is a placeholder implementation that:
  // 1. Splits text into words
  // 2. Maps words to token IDs (using placeholder IDs)
  // 3. Handles special tokens
  
  std::vector<int> input_ids;
  
  // Add BOS token if configured
  if (text_config_.add_bos_token) {
    input_ids.push_back(special_tokens_["<bos>"]);
  }
  
  // Placeholder tokenization: split text and map to simple IDs
  // In production, this would call the actual tokenizer
  std::istringstream iss(text);
  std::string word;
  int word_id = 100; // Starting word ID (placeholder)
  
  while (iss >> word) {
    // Check if word is a special token
    auto it = special_tokens_.find(word);
    if (it != special_tokens_.end()) {
      input_ids.push_back(it->second);
    } else {
      // Regular word - use placeholder ID
      input_ids.push_back(word_id++);
    }
  }
  
  // Add EOS token if configured
  if (text_config_.add_eos_token) {
    input_ids.push_back(special_tokens_["<eos>"]);
  }
  
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Tokenized to " << input_ids.size() << " tokens" << std::endl;
    std::cout << "[T5Gemma2Processor] First 10 input_ids: [";
    for (size_t i = 0; i < 10 && i < input_ids.size(); ++i) {
      std::cout << input_ids[i];
      if (i < 9 && i < input_ids.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  
  return input_ids;
}

std::vector<int> T5Gemma2Processor::createAttentionMask(const std::vector<int> &input_ids) {
  // Attention mask is 1 for all valid tokens
  // For now, all tokens are considered valid (no padding)
  std::vector<int> attention_mask(input_ids.size(), 1);
  
  // TODO: If padding is enabled, we should change attention mask algorithm to mark
  // padding tokens as 0. The attention mask should differentiate between actual
  // tokens (1) and padding tokens (0).
  
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Created attention mask: " 
              << attention_mask.size() << " tokens" << std::endl;
  }
  
  return attention_mask;
}

std::vector<int> T5Gemma2Processor::createTokenTypeIds(const std::vector<int> &input_ids, int image_token_id) {
  // Token type IDs: 0 for text, 1 for image tokens
  // For each image: BOI_TOKEN + 256 IMAGE_TOKENs + EOI_TOKEN = 258 tokens marked as 1
  // \n\n tokens are marked as 0 (text)
  std::vector<int> token_type_ids(input_ids.size(), 0);
  
  int boi_token_id = special_tokens_[BOI_TOKEN];
  int eoi_token_id = special_tokens_[EOI_TOKEN];
  
  int image_section_count = 0;
  bool in_image_section = false;
  
  for (size_t i = 0; i < input_ids.size(); ++i) {
    int token_id = input_ids[i];
    
    if (token_id == boi_token_id) {
      // Start of image section: mark BOI as 1
      token_type_ids[i] = 1;
      in_image_section = true;
      image_section_count++;
    } else if (token_id == eoi_token_id) {
      // End of image section: mark EOI as 1
      token_type_ids[i] = 1;
      in_image_section = false;
    } else if (token_id == image_token_id && in_image_section) {
      // Image soft token: mark as 1
      token_type_ids[i] = 1;
    }
    // All other tokens remain 0 (including \n\n tokens)
  }
  
  if (debug_output_) {
    // Count tokens marked as 1
    int marked_count = std::count(token_type_ids.begin(), token_type_ids.end(), 1);
    std::cout << "[T5Gemma2Processor] Created token_type_ids: " 
              << marked_count << " image tokens marked (BOI + 256*IMAGE + EOI per image)" << std::endl;
  }
  
  return token_type_ids;
}

std::string T5Gemma2Processor::expandImagePlaceholders(const std::string &text) {
  // Replace each BOI_TOKEN with the full image token sequence
  std::string expanded = text;
  
  // Find all occurrences of BOI_TOKEN and replace with full image sequence
  size_t pos = 0;
  std::string full_sequence = getFullImageSequence();
  
  while ((pos = expanded.find(BOI_TOKEN, pos)) != std::string::npos) {
    expanded.replace(pos, std::string(BOI_TOKEN).length(), full_sequence);
    pos += full_sequence.length();
  }
  
  return expanded;
}

std::string T5Gemma2Processor::getFullImageSequence() {
  // Create the full image token sequence:
  // \n\n<start_of_image><image_soft_token>*256<end_of_image>\n\n
  // Token order: \n\n token (108) + BOI token + 256 IMAGE tokens + EOI token + \n\n token (108)
  std::string image_tokens = "";
  for (int i = 0; i < image_seq_length_; ++i) {
    image_tokens += IMAGE_TOKEN;
  }
  
  std::string full_sequence = "\n\n" + std::string(BOI_TOKEN) + image_tokens + 
                             std::string(EOI_TOKEN) + "\n\n";
  
  return full_sequence;
}

} // namespace nntrainer
