# T5Gemma2 Image Preprocessing Implementation

This directory contains the C++ implementation of image preprocessing for the T5Gemma2 model, which matches the PyTorch reference implementation.

## Overview

The T5Gemma2 model uses Gemma3's image processor with the following preprocessing steps:

1. **Resize**: Resize image to 896x896 pixels
2. **Rescale**: Scale pixel values from [0, 255] to [0, 1] by dividing by 255.0
3. **Normalize**: Apply normalization with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]

The final formula is: `(pixel_value / 255.0 - 0.5) / 0.5 = pixel_value / 127.5 - 1.0`

## Files

- `t5gemma2_image_preprocess.h` - Header file with function declarations
- `t5gemma2_image_preprocess.cpp` - Implementation of image preprocessing functions
- `test_image_preprocess.cpp` - Test program to verify the preprocessing
- `test_pytorch_comparison.py` - Python script to compare C++ output with PyTorch reference

## API Reference

### `preprocessT5Gemma2Image(const std::string &filepath)`

Main preprocessing function for T5Gemma2 model.

**Parameters:**
- `filepath`: Path to the input image file

**Returns:**
- `std::vector<float>`: Preprocessed image data in CHW format (3 channels, 896x896)

**Example:**
```cpp
#include "t5gemma2_image_preprocess.h"

using namespace nntrainer;

std::vector<float> pixel_values = preprocessT5Gemma2Image("path/to/image.png");
// pixel_values has size: 3 * 896 * 896 = 2,411,328
```

### `preprocessT5Gemma2ImageCustom(...)`

Preprocessing function with custom normalization parameters.

**Parameters:**
- `filepath`: Path to the input image file
- `target_width`: Target image width
- `target_height`: Target image height
- `mean`: Mean values for normalization (1-3 values)
- `std`: Standard deviation values for normalization (1-3 values)

**Returns:**
- `std::vector<float>`: Preprocessed image data in CHW format

**Example:**
```cpp
std::vector<float> mean = {0.5f, 0.5f, 0.5f};
std::vector<float> std = {0.5f, 0.5f, 0.5f};

std::vector<float> pixel_values = preprocessT5Gemma2ImageCustom(
    "path/to/image.png", 896, 896, mean, std);
```

## Building

The implementation is integrated into the nntrainer meson build system and will be built automatically when you build from the root.

### Building with Meson

The implementation requires:
- C++17 or later
- STB image library (already included via `llm_util.cpp`)

**Build the entire project:**
```bash
cd /path/to/nntrainer
meson setup build
ninja -C build
```

This will:
- Compile `t5gemma2_image_preprocess.cpp` and link it into the `libcausallm` library
- Build the test executable `t5gemma2_image_preprocess_test` (tests are enabled by default)
- The functions will be available for use in other parts of the codebase

**Note:** Tests are enabled by default, so the test executable is built automatically. No additional flags needed!

## Testing

### 1. Run the C++ test program

The test executable is built automatically at:
```
build/Applications/CausalLM/t5gemma2_image_preprocess_test
```

Run it with:
```bash
cd build/Applications/CausalLM
./t5gemma2_image_preprocess_test [image_path]
```

If no image path is provided, it will use `Applications/CausalLM/models/t5gemma2/res/t5gemma2/cat.png` by default.

**Note:** You can also compile the test manually for debugging:
```bash
cd Applications/CausalLM/models/t5gemma2
g++ -std=c++17 -O2 -I../.. -I../../.. test_image_preprocess.cpp \
    t5gemma2_image_preprocess.cpp ../../llm_util.cpp -o test_image_preprocess
./test_image_preprocess [image_path]
```

The program will:
- Preprocess the image using standard T5Gemma2 parameters
- Save the output to `t5gemma2_preprocessed_output.txt`
- Print statistics (size, min, max, mean, first 5 values)
- Verify consistency with custom parameter function

### 2. Compare with PyTorch reference

```bash
cd Applications/CausalLM/models/t5gemma2
python3 test_pytorch_comparison.py [image_path]
```

This will:
- Load and preprocess the image using PyTorch's Gemma3Processor
- Load the C++ output from the previous step
- Compare the two outputs
- Report differences and whether they match within tolerance

### Expected Results

The C++ implementation should produce identical results to PyTorch within floating-point precision (tolerance: 1e-5).

**Expected statistics for normalized output:**
- Size: 2,411,328 (3 * 896 * 896)
- Range: approximately [-1.0, 1.0]
- Mean: approximately 0.0 (depending on image content)

## Implementation Details

### Image Loading

The implementation uses `loadAndPreprocessImage()` from `llm_util.cpp`:
- Loads image using stb_image
- Resizes to target dimensions (896x896) using bilinear interpolation
- Converts to RGB format (handles grayscale and RGBA)
- Returns data in CHW format (Channels, Height, Width)

### Normalization Process

```cpp
// Step 1: Rescale from [0, 255] to [0, 1]
float rescaled = pixel_value / 255.0f;

// Step 2: Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
float normalized = (rescaled - 0.5f) / 0.5f;
```

This is equivalent to:
```cpp
float normalized = pixel_value / 127.5f - 1.0f;
```

### Data Format

- **Input**: Image file (PNG, JPG, etc.)
- **Output**: `std::vector<float>` in CHW format
  - Channel order: R, G, B
  - Index: `c * height * width + h * width + w`
  - Total size: 3 * 896 * 896 = 2,411,328 floats

## PyTorch Reference Configuration

The implementation matches the configuration in `res/t5gemma2/processor_config.json`:

```json
{
  "image_processor": {
    "do_resize": true,
    "do_rescale": true,
    "do_normalize": true,
    "size": {"height": 896, "width": 896},
    "rescale_factor": 0.00392156862745098,  // 1/255
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 2  // BILINEAR
  }
}
```

## Usage Example in Model

```cpp
#include "t5gemma2_image_preprocess.h"

void processImageForModel(const std::string &image_path) {
    // Preprocess image
    std::vector<float> pixel_values = preprocessT5Gemma2Image(image_path);
    
    // Create tensor from pixel values
    // Shape: [1, 3, 896, 896] (batch, channels, height, width)
    int batch_size = 1;
    int channels = 3;
    int height = 896;
    int width = 896;
    
    // Use the pixel values for model inference
    // ...
}
```

## Troubleshooting

### Build Errors

- **`llm_util.hpp` not found**: Ensure the include path points to `Applications/CausalLM/`
- **STB image errors**: STB_IMAGE_IMPLEMENTATION should be defined in exactly one .cpp file (already done in `llm_util.cpp`)

### Runtime Errors

- **"Failed to load image"**: Check that the image file exists and is readable
- **"Unsupported number of channels"**: The implementation supports 1 (grayscale), 3 (RGB), and 4 (RGBA) channels

### Validation Errors

- **Output doesn't match PyTorch**: 
  1. Verify the image is the same
  2. Check the normalization parameters in `processor_config.json`
  3. Ensure PyTorch is using the same Gemma3Processor version
  4. Compare bilinear interpolation implementation

## References

- PyTorch implementation: `res/gemma3/image_processing_gemma3.py`
- Configuration: `res/t5gemma2/processor_config.json`
- Utility functions: `Applications/CausalLM/llm_util.cpp`
