// SPDX-License-Identifier: Apache-2.0
/**
 * @file   test_processor.cpp
 * @brief  Test program for T5Gemma2 processor
 * @date   2025-02-24
 * @author Cline SR
 */

#include "t5gemma2_processor.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace nntrainer;

void saveToFile(const std::vector<int> &data, const std::string &filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < data.size(); ++i) {
    outFile << data[i];
    if (i < data.size() - 1) {
      outFile << "\n";
    }
  }
  outFile.close();
  std::cout << "Saved " << data.size() << " values to " << filename << std::endl;
}

void saveToFile(const std::vector<float> &data, const std::string &filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  outFile << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < data.size(); ++i) {
    outFile << data[i];
    if (i < data.size() - 1) {
      outFile << "\n";
    }
  }
  outFile.close();
  std::cout << "Saved " << data.size() << " values to " << filename << std::endl;
}

void printProcessorOutput(const T5Gemma2ProcessorOutput &output, const std::string &name) {
  std::cout << "\n" << name << ":" << std::endl;
  std::cout << "  pixel_values size: " << output.pixel_values.size() << std::endl;
  std::cout << "  input_ids size: " << output.input_ids.size() << std::endl;
  std::cout << "  attention_mask size: " << output.attention_mask.size() << std::endl;
  std::cout << "  token_type_ids size: " << output.token_type_ids.size() << std::endl;
  
  if (!output.input_ids.empty()) {
    std::cout << "  First 20 input_ids: [";
    for (size_t i = 0; i < 20 && i < output.input_ids.size(); ++i) {
      std::cout << output.input_ids[i];
      if (i < 19 && i < output.input_ids.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  
  if (!output.token_type_ids.empty()) {
    std::cout << "  First 20 token_type_ids: [";
    for (size_t i = 0; i < 20 && i < output.token_type_ids.size(); ++i) {
      std::cout << output.token_type_ids[i];
      if (i < 19 && i < output.token_type_ids.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::cout << "\n=== T5Gemma2 Processor Test ===\n" << std::endl;

  try {
    // Test 1: Process images only
    std::cout << "\nTest 1: Process images only" << std::endl;
    std::cout << "========================================" << std::endl;
    
    T5Gemma2Processor processor1;
    std::vector<std::string> image_paths;
    
    // Use default test image
    image_paths.push_back("./Cat.jpg");

    
    
    T5Gemma2ProcessorOutput output1 = processor1.process(image_paths);
    printProcessorOutput(output1, "Output");
    
    // Save outputs to files
    saveToFile(output1.pixel_values, "processor_pixel_values.txt");
    saveToFile(output1.input_ids, "processor_input_ids.txt");
    saveToFile(output1.attention_mask, "processor_attention_mask.txt");
    saveToFile(output1.token_type_ids, "processor_token_type_ids.txt");
    
    // Test 2: Process text and images
    std::cout << "\n\nTest 2: Process text and images" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    T5Gemma2Processor processor2;
    std::string text = "Describe this image";
    
    T5Gemma2ProcessorOutput output2 = processor2.process(text, image_paths);
    printProcessorOutput(output2, "Output");
    
    // Test 3: Process multiple images
    std::cout << "\n\nTest 3: Process multiple images" << std::endl;
    std::cout << "======================================" << std::endl;
    
    T5Gemma2Processor processor3;
    std::vector<std::string> multiple_images;
    multiple_images.push_back("./Cat.jpg");
    multiple_images.push_back("./Dog.jpg");
    
    T5Gemma2ProcessorOutput output3 = processor3.process(multiple_images);
    printProcessorOutput(output3, "Output");
    
    // Test 4: Process text only
    std::cout << "\n\nTest 4: Process text only" << std::endl;
    std::cout << "=============================" << std::endl;
    
    T5Gemma2Processor processor4;
    T5Gemma2ProcessorOutput output4 = processor4.process(text);
    printProcessorOutput(output4, "Output");
    
    std::cout << "\n\n=== All tests completed successfully ===" << std::endl;
    std::cout << "\nOutput files saved:" << std::endl;
    std::cout << "  - processor_pixel_values.txt (flattened image tensor)" << std::endl;
    std::cout << "  - processor_input_ids.txt (tokenized text)" << std::endl;
    std::cout << "  - processor_attention_mask.txt" << std::endl;
    std::cout << "  - processor_token_type_ids.txt" << std::endl;
    
    return 0;
    
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
