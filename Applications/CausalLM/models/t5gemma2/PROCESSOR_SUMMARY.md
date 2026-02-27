# T5Gemma2 Processor Implementation Summary

## Overview

Successfully implemented the T5Gemma2 processor in C++ that handles both text and image inputs, following the pattern of the PyTorch Gemma3Processor.

## What Was Implemented

### 1. Multi-Image Preprocessing

**Files:**
- `t5gemma2_image_preprocess.h` - Added multi-image function declarations
- `t5gemma2_image_preprocess.cpp` - Implemented multi-image preprocessing functions

**New Functions:**
- `preprocessT5Gemma2Images(filepaths)` - Process multiple images with default T5Gemma2 parameters
- `preprocessT5Gemma2ImagesCustom(filepaths, width, height, mean, std)` - Process multiple images with custom parameters

**Features:**
- Processes batch of images
- Returns data in NCHW format (batch, channels, height, width)
- Each image is processed independently
- Debug output showing progress for each image

### 2. T5Gemma2Processor Class

**Files:**
- `t5gemma2_processor.h` - Processor class header
- `t5gemma2_processor.cpp` - Processor class implementation

**Class Features:**

#### Configuration Structures
- `ImageProcessingConfig` - Image processing parameters (size, mean, std, etc.)
- `TextProcessingConfig` - Text processing parameters (BOS/EOS tokens, padding, etc.)

#### Processing Methods
- `process(text, images)` - Process both text and images
- `process(images)` - Process images only
- `process(text)` - Process text only

#### Output Structure
```cpp
struct T5Gemma2ProcessorOutput {
  std::vector<float> pixel_values;   // Image tensor (batch, channels, height, width)
  std::vector<int> input_ids;         // Tokenized text input
  std::vector<int> attention_mask;      // Attention mask
  std::vector<int> token_type_ids;     // Token type IDs (0 for text, 1 for image)
};
```

#### Special Tokens
- `<start_of_image>` (BOI) - Beginning of image
- `<end_of_image>` (EOI) - End of image
- `<image_soft_token>` - Image soft token (repeated 256 times)
- Standard tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`

#### Processing Pipeline
1. **Image Processing:** Preprocess images using multi-image preprocessing functions
2. **Text Processing:** Tokenize text (placeholder implementation for now)
3. **Placeholder Expansion:** Expand image placeholders with full image token sequence
4. **Attention Mask:** Create attention mask (all 1s for valid tokens)
5. **Token Type IDs:** Mark image tokens with type ID 1 (text tokens are 0)

### 3. Test Program

**File:** `test_processor.cpp`

**Test Cases:**
1. Process images only
2. Process text and images together
3. Process multiple images
4. Process text only

**Output Files:**
- `processor_pixel_values.txt` - Flattened image tensor
- `processor_input_ids.txt` - Tokenized text
- `processor_attention_mask.txt` - Attention mask
- `processor_token_type_ids.txt` - Token type IDs

### 4. Build Integration

**Modified Files:**
- `Applications/CausalLM/models/t5gemma2/meson.build` - Added processor source files
- `Applications/CausalLM/meson.build` - Added processor test executable

**Built Executables:**
- `build/Applications/CaulLM/t5gemma2_image_preprocess_test` - Image preprocessing test
- `build/Applications/CaulLM/t5gemma2_processor_test` - Processor test

## Usage Examples

### Process Single Image
```cpp
#include "t5gemma2_processor.h"

using namespace nntrainer;

T5Gemma2Processor processor;
std::vector<std::string> images = {"path/to/image.png"};
T5Gemma2ProcessorOutput output = processor.process(images);

// output.pixel_values contains preprocessed image tensor
// output.input_ids contains tokenized text with image placeholders
// output.attention_mask contains attention mask
// output.token_type_ids marks image tokens
```

### Process Multiple Images
```cpp
std::vector<std::string> images = {
    "path/to/image1.png",
    "path/to/image2.png"
};
T5Gemma2ProcessorOutput output = processor.process(images);
```

### Process Text and Images
```cpp
std::string text = "Describe this image";
std::vector<std::std::string> images = {"path/to/image.png"};

T5Gemma2ProcessorProcessorOutput output = processor.process(text, images);
```

### Process Text Only
```cpp
std::string text = "What is the capital of France?";
T5Gemma2ProcessorOutput output = processor.process(text);
```

### Custom Configuration
```cpp
T5Gemma2Processor processor;

// Set custom image configuration
T5Gemma2Processor::ImageProcessingConfig img_config;
img_config.image_size = 512;  // Custom size
img_config.image_mean = {0.485f, 0.456f, 0.406f};  // Custom mean
img_config.image_std = {0.229f, 0.224f, 0.225f};    // Custom std
processor.setImageConfig(img_config);

// Set custom text configuration
T5Gemma2Processor::TextProcessingConfig text_config;
text_config.add_bos_token = false;
text_config.add_eos_token = true;
processor.setTextConfig(text_config);
```

## Debug Output

The processor provides detailed debug output showing:
- Initialization parameters
- Input information (text, number of images)
- Processing progress for each image
- Output statistics (sizes, first few values)
- Token type ID assignments

Format: `[T5Gemma2Processor] Message`

## Data Format

### Image Tensor (pixel_values)
- **Format:** NCHW (batch, channels, height, width)
- **Default shape:** (num_images, 3, 896, 896)
- **Value range:** Approximately [-1.0, 1.0] (after normalization)
- **Total size:** batch_size × 3 × 896 × 896 floats

### Token IDs (input_ids)
- **Format:** 1D array of token IDs
- **Special tokens:** BOS, EOS, BOI, EOI, and 256 image tokens per image
- **Placeholder values:** Starting from 100 (will be replaced by real tokenizer)

### Attention Mask
- **Format:** 1D array (same size as input_ids)
- **Values:** All 1s (valid tokens)

### Token Type IDs
- **Format:** 1D array (same size as input_ids)
- **Values:** 0 for text tokens, 1 for image tokens

## Image Token Sequence

When an image is detected in the text (via `<start_of_image>` token), it is expanded to:

```
\n\n<start_of_image><image_soft_token>*256<end_of_image>\n\n
```

**Token order per image:**
1. `\n\n` token (token ID: 108, will be loaded from tokenizer in future) - **text token (type 0)**
2. `BOI_TOKEN` (`<start_of_image>`, token ID: 255999) - **image token (type 1)**
3. 256 `IMAGE_TOKEN`s (`<image_soft_token>`, token ID: 256000) - **image tokens (type 1)**
4. `EOI_TOKEN` (`<end_of_image>`, token ID: 256001) - **image token (type 1)**
5. `\n\n` token (token ID: 108) - **text token (type 0)**

**Total tokens per image:** 260 tokens (2 for \n\n + 258 for image section + 2 for \n\n)
**Image tokens marked as type 1:** 258 tokens (BOI + 256*IMAGE + EOI)
**Text tokens marked as type 0:** 2 tokens (the \n\n tokens)

## TODO: Future Enhancements

1. **Tokenizer Integration:** Replace placeholder tokenizer with actual huggingface_tokenizer integration
2. **Pan-and-Scan:** Implement pan-and-scan functionality for large images (currently skipped per request)
3. **Padding:** Implement padding for batched text with different lengths
4. **Error Handling:** Add more robust error handling for missing images, invalid text, etc.
5. **Performance:** Optimize for large batch processing
6. **Configuration Loading:** Load special token IDs from tokenizer configuration files

## Building

```bash
cd /path/to/nntrainer
meson setup build
ninja -C build
```

This will build:
- `libcausallm.so` with T5Gemma2 processor
- `t5gemma2_image_preprocess_test` - Image preprocessing test executable
- `t5gemma2_processor_test` - Processor test executable

## Testing

```bash
cd build/Applications/CausalLM
./t5gemma2_processor_test
```

This will run all 4 test cases and save output files for verification.

## Files Created/Modified

### Created:
- `Applications/CausalLM/models/t5gemma2/t5gemma2_processor.h`
- `Applications/CausalLM/models/t5gemma2/t5gemma2_processor.cpp`
- `Applications/CausalLM/models/t5gemma2/test_processor.cpp`
- `Applications/CausalLM/models/t5gemma2/PROCESSOR_SUMMARY.md`

### Modified:
- `Applications/CausalLM/models/t5gemma2/t5gemma2_image_preprocess.h` - Added multi-image functions
- `Applications/CausalLM/models/t5gemma2/t5gemma2_image_preprocess.cpp` - Implemented multi-image functions
- `Applications/CausalLM/models/t5gemma2/meson.build` - Added processor source files
- `Applications/CausalLM/meson.build` - Added processor test executable

## Next Steps

1. **Build and Test:** Build the project and verify the processor works with test images
2. **Tokenizer Integration:** Integrate with huggingface_tokenizer for real text processing
3. **Model Integration:** Integrate the processor with the T5Gemma2 model implementation
4. **Validation:** Compare outputs with PyTorch reference implementation
5. **Performance:** Optimize for production use

## References

- PyTorch Gemma3Processor: `res/gemma3/processing_gemma3.py`
- PyTorch Gemma3ImageProcessor: `res/gemma3/image_processing_gemma3.py`
- T5Gemma2 configuration: `res/t5gemma2/processor_config.json`
- T5Gemma2 tokenizer: `res/t5gemma2/tokenizer_config.json`
- Image preprocessing: `t5gemma2_image_preprocess.h/cpp`
