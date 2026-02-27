// SPDX-License-Identifier: Apache-2.0
/**
 * @file   test_image_preprocess.cpp
 * @brief  Test program for T5Gemma2 image preprocessing
 * @date   2025-02-24
 * @author Cline SR
 */

#include "t5gemma2_image_preprocess.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace nntrainer;

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

void printStatistics(const std::vector<float> &data, const std::string &name) {
  if (data.empty()) {
    std::cout << name << ": Empty data" << std::endl;
    return;
  }

  float min_val = data[0];
  float max_val = data[0];
  float sum = 0.0f;

  for (const auto &val : data) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum += val;
  }

  float mean = sum / data.size();

  std::cout << name << ":" << std::endl;
  std::cout << "  Size: " << data.size() << std::endl;
  std::cout << "  Min: " << min_val << std::endl;
  std::cout << "  Max: " << max_val << std::endl;
  std::cout << "  Mean: " << mean << std::endl;
  std::cout << "  First 5 values: [";
  for (int i = 0; i < 5 && i < (int)data.size(); ++i) {
    std::cout << data[i];
    if (i < 4 && i < (int)data.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char *argv[]) {
  std::string image_path;

  std::ofstream out("output.txt");
    
    // 1. 기존 cout의 버퍼를 저장해둡니다 (나중에 복구하기 위함)
    std::streambuf* coutbuf = std::cout.rdbuf();

    // 2. cout의 버퍼를 파일 스트림의 버퍼로 교체합니다.
    std::cout.rdbuf(out.rdbuf());

  // Check if image path is provided as argument
  if (argc > 1) {
    image_path = argv[1];
  } else {
    // Use the default cat.png from the res directory
    image_path = "res/t5gemma2/cat.png";
    std::cout << "No image path provided, using default: " << image_path
              << std::endl;
  }

  std::cout << "\n=== T5Gemma2 Image Preprocessing Test ===" << std::endl;
  std::cout << "Processing image: " << image_path << "\n" << std::endl;

  try {
    // Test 1: Standard T5Gemma2 preprocessing (896x896)
    std::cout << "Test 1: Standard T5Gemma2 preprocessing (896x896)" << std::endl;
    std::cout << "Parameters: resize to 896x896, rescale by 1/255, "
                 "normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]"
              << std::endl;

    std::vector<float> preprocessed = preprocessT5Gemma2Image(image_path);
    printStatistics(preprocessed, "Preprocessed image");

    // Save to file for comparison with PyTorch
    std::string output_file = "t5gemma2_preprocessed_output.txt";
    saveToFile(preprocessed, output_file);

    // Test 2: Custom normalization parameters
    std::cout << "\nTest 2: Custom normalization parameters" << std::endl;
    std::vector<float> custom_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> custom_std = {0.5f, 0.5f, 0.5f};

    std::vector<float> preprocessed_custom = preprocessT5Gemma2ImageCustom(
      image_path, 896, 896, custom_mean, custom_std);
    printStatistics(preprocessed_custom, "Preprocessed image (custom params)");

    std::cout << "\n=== Comparison with Test 1 ===" << std::endl;
    if (preprocessed.size() == preprocessed_custom.size()) {
      bool match = true;
      for (size_t i = 0; i < preprocessed.size(); ++i) {
        if (std::abs(preprocessed[i] - preprocessed_custom[i]) > 1e-6f) {
          match = false;
          std::cout << "Mismatch at index " << i << ": " << preprocessed[i]
                    << " vs " << preprocessed_custom[i] << std::endl;
          break;
        }
      }
      if (match) {
        std::cout << "✓ Test 1 and Test 2 results match perfectly!" << std::endl;
      }
    } else {
      std::cout << "✗ Size mismatch between Test 1 and Test 2" << std::endl;
    }

    std::cout << "\n=== Test completed successfully ===" << std::endl;
    std::cout << "\nTo compare with PyTorch output:" << std::endl;
    std::cout << "1. Run the PyTorch preprocessing script on the same image"
              << std::endl;
    std::cout << "2. Compare the output with " << output_file << std::endl;
    std::cout << "3. The values should be identical (within floating point precision)"
              << std::endl;

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
