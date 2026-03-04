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

    std::ofstream out("t5_log.log");
    
    // 1. 기존 cout의 버퍼를 저장해둡니다 (나중에 복구하기 위함)
    std::streambuf* coutbuf = std::cout.rdbuf();

    // 2. cout의 버퍼를 파일 스트림의 버퍼로 교체합니다.
    std::cout.rdbuf(out.rdbuf());

  std::cout << "\n=== T5Gemma2 Processor Test ===\n" << std::endl;

  try {
    
    
    std::string text = "I love korea because";
    // Test 1: Process text only
    std::cout << "\n\nTest 1: Process text only" << std::endl;
    std::cout << "=============================" << std::endl;
    
    T5Gemma2Processor processor1;
    T5Gemma2ProcessorOutput output1 = processor1.process(text);
    printProcessorOutput(output1, "Output");
    

       
    // Test 2: Process text and image
    std::cout << "\n\nTest 2: Process text and images" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    T5Gemma2Processor processor2;
    text = "<start_of_image> ./Cat.jpg Describe this image";
    
    T5Gemma2ProcessorOutput output2 = processor2.process(text);
    printProcessorOutput(output2, "Output");
    
        
    //TODO : Test 3 : process text + multi image
    
    return 0;
    
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
