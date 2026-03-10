// SPDX-License-Identifier: Apache-2.0
/**
 * @file   t5gemma2_processor.cpp
 * @brief  T5Gemma2 processor implementation
 * @date   2025-02-24
 * @author Cline SR
 */

#include "t5gemma2_processor.h"
#include "t5gemma2_image_preprocess.h"
#include <algorithm>
#include <llm_util.hpp>
#include <regex>
#include <sstream>

namespace nntrainer {

// Static member definitions
const char *T5Gemma2Processor::BOI_TOKEN = "<start_of_image>";
const char *T5Gemma2Processor::EOI_TOKEN = "<end_of_image>";
const char *T5Gemma2Processor::IMAGE_TOKEN = "<image_soft_token>";

T5Gemma2Processor::T5Gemma2Processor(int image_seq_length, int image_token_id) :
  image_seq_length_(image_seq_length),
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
    std::cout << "  full_image_sequence: "
              << getFullImageSequence().substr(0, 50) << "..." << std::endl;
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

T5Gemma2ProcessorOutput
T5Gemma2Processor::process(const std::string &text,
                           const std::vector<std::string> &images) {

  if (debug_output_) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[T5Gemma2Processor::process]" << std::endl;
    std::cout << "  Input text: " << (text.empty() ? "(empty)" : text)
              << std::endl;
    std::cout << "  Input images: " << images.size() << " image(s)"
              << std::endl;
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
    // case 1) Text only processing - return text as-is
    if (debug_output_) {
      std::cout << "[T5Gemma2Processor] Processing text only" << std::endl;
    }
    output.processed_text = processed_text;
  } else if (!images.empty()) {
    // case 2) Images only or mixed processing

    if (text.empty()) {
      // create BOI_TOKEN for image holding
      processed_text = "";
      for (size_t i = 0; i < images.size(); ++i) {
        processed_text += BOI_TOKEN;
        processed_text += " ";
      }
      if (debug_output_) {
        std::cout
          << "[T5Gemma2Processor] Created placeholder text for images only"
          << std::endl;
      }
    }

    // Expand image placeholders in text
    processed_text = expandImagePlaceholders(processed_text);

    if (debug_output_) {
      std::cout << "[T5Gemma2Processor] Text after expansion: "
                << processed_text << "..." << std::endl;
    }

    output.processed_text = processed_text;
  }

  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Processing complete" << std::endl;
    std::cout << "  processed_text length: " << output.processed_text.size() << std::endl;
    std::cout << "  pixel_values shape: ["
              << (output.pixel_values.empty()
                    ? 0
                    : output.pixel_values.size() /
                        (3 * image_config_.image_size *
                         image_config_.image_size))
              << ", 3, " << image_config_.image_size << ", "
              << image_config_.image_size << "]" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
  }

  return output;
}

T5Gemma2ProcessorOutput
T5Gemma2Processor::process(const std::string &input_prompt) {
  std::string processed_text = input_prompt;
  std::vector<std::string> images;

  // Check if BOI_TOKEN is present in the input
  if (input_prompt.find(BOI_TOKEN) != std::string::npos) {
    // Parse image paths from input text
    // Pattern: <BOI_TOKEN> followed by image path (with extension like .jpg,
    // .png, etc.)
    std::regex image_pattern(
      BOI_TOKEN +
      std::string(R"(\s+([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)))"));
    std::sregex_iterator it(input_prompt.begin(), input_prompt.end(),
                            image_pattern);
    std::sregex_iterator end;

    // Extract all image paths
    while (it != end) {
      std::smatch match = *it;
      if (match.size() > 1) {
        std::string image_path = match[1].str();
        images.push_back(image_path);

        if (debug_output_) {
          std::cout << "[T5Gemma2Processor] Found image path: " << image_path
                    << std::endl;
        }
      }
      ++it;
    }

    // Remove image paths from the text (keep BOI_TOKEN)
    processed_text = std::regex_replace(input_prompt, image_pattern, BOI_TOKEN);

    if (debug_output_) {
      std::cout << "[T5Gemma2Processor] Extracted " << images.size()
                << " image(s)" << std::endl;
      std::cout << "[T5Gemma2Processor] Processed text: " << processed_text
                << std::endl;
    }
  } else {
    // No BOI_TOKEN found, treat entire input as text
    if (debug_output_) {
      std::cout
        << "[T5Gemma2Processor] No BOI_TOKEN found, treating as text-only input"
        << std::endl;
    }
  }

  // Call the main process function with extracted text and images
  return process(processed_text, images);
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

std::vector<float>
T5Gemma2Processor::preprocessImages(const std::vector<std::string> &images) {
  if (debug_output_) {
    std::cout << "[T5Gemma2Processor] Preprocessing " << images.size()
              << " image(s)" << std::endl;
  }

  // Use the multi-image preprocessing function
  return preprocessT5Gemma2ImagesCustom(
    images, image_config_.image_size, image_config_.image_size,
    image_config_.image_mean, image_config_.image_std);
}

std::string
T5Gemma2Processor::expandImagePlaceholders(const std::string &text) {
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
  // Token order: \n\n token (108) + BOI token + 256 IMAGE tokens + EOI token +
  // \n\n token (108)
  std::string image_tokens = "";
  for (int i = 0; i < image_seq_length_; ++i) {
    image_tokens += IMAGE_TOKEN;
  }

  std::string full_sequence = "\n\n" + std::string(BOI_TOKEN) + image_tokens +
                              std::string(EOI_TOKEN) + "\n\n";

  return full_sequence;
}

} // namespace nntrainer
