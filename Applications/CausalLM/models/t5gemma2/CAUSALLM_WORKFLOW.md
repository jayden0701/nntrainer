# CausalLM Workflow in nntrainer

## Overview

This document explains how Causal Language Models (CausalLM) work in nntrainer's Applications/CausalLM directory.

## Class Hierarchy

```
Transformer (base class)
    ↓
CausalLM (inherits from Transformer)
    ↓
Specific Models (e.g., Gemma3CausalLM, Qwen3CausalLM)
    - Inherits from both CausalLM and specific transformer variant
    - Example: Gemma3CausalLM inherits from CausalLM and Gemma3Transformer
```

## Architecture

### Model Structure

```
[Input Tokens]
       ↓
[Embedding Layer]
       ↓
[Transformer Decoder Blocks] × NUM_LAYERS
  ├── RMSNorm
  ├── Multi-Head Attention
  ├── Addition (Residual)
  ├── RMSNorm
  ├── Feed Forward Network (SwiGLU)
  └── Addition (Residual)
       ↓
[Output RMSNorm]
       ↓
[LM Head] (or TieWordEmbedding)
       ↓
[Output Logits]
```

## Initialization Workflow

### 1. Constructor

**File:** `causal_lm.cpp`, `transformer.cpp`

```cpp
// 1. Create model instance with configurations
CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg)
  : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM)
{
  setupParameters(cfg, generation_cfg, nntr_cfg);
}
```

**Configuration Files:**
- `config.json` - Model architecture (hidden_size, num_layers, etc.)
- `generation_config.json` - Generation parameters (temperature, top_k, top_p)
- `nntrainer_config.json` - Runtime configuration (batch_size, max_seq_len, paths)

### 2. setupParameters()

**File:** `causal_lm.cpp`, `transformer.cpp`

Loads parameters from configuration files:
- Model architecture: `NUM_VOCAB`, `DIM`, `NUM_LAYERS`, `NUM_HEADS`, etc.
- Generation parameters: `EOS_TOKEN_ID`, `BOS_TOKEN_ID`, `TEMPERATURE`, `TOP_K`, `TOP_P`
- Runtime config: `BATCH_SIZE`, `MAX_SEQ_LEN`, `NUM_TO_GENERATE`
- KV cache settings (optional): `USE_KVCACHE`, `PRE_COMPUTED_CACHE_PATH`

### 3. initialize()

**File:** `transformer.cpp`

```cpp
void Transformer::initialize() {
  // Step 1: Register custom layers
  registerCustomLayers();
  
  // Step 2: Construct model layers
  constructModel();
  
  // Step 3: Set model properties
  model->setProperty(model_props);
  
  // Step 4: Compile model
  model->compile(ExecutionMode::INFERENCE);
  
  // Step 5: Initialize model
  model->initialize(ExecutionMode::INFERENCE);
}
```

**Custom Layers Registered:**
- `RMSNormLayer` - Root Mean Square Normalization
- `MHACoreLayer` - Multi-Head Attention Core with KV cache
- `SwiGLULayer` - Swish-Gated Linear Unit activation
- `EmbeddingLayer` - Token embedding
- `TieWordEmbedding` - Shared embedding/LM head
- `LmHeadLayer` - Language model head (output projection)

### 4. constructModel()

**File:** `transformer.cpp`

Builds the neural network architecture:
1. **Input Layer** - Accepts token IDs
2. **Embedding Layer** - Maps tokens to embeddings
3. **Transformer Blocks** - `NUM_LAYERS` decoder blocks
4. **Output RMSNorm** - Final normalization
5. **LM Head** - Projects to vocabulary logits

Each transformer decoder block contains:
- Pre-norm RMSNorm
- Multi-Head Self-Attention (with KV cache)
- Residual connection
- Pre-norm RMSNorm
- Feed Forward Network (SwiGLU)
- Residual connection

### 5. load_weight()

**File:** `transformer.cpp`

Loads pre-trained weights from a binary file:
```cpp
model->load(weight_path, ModelFormat::MODEL_FORMAT_BIN);
```

## Inference Workflow

### 1. run() - Main Entry Point

**File:** `causal_lm.cpp`

```cpp
void CausalLM::run(const WSTR prompt, bool do_sample, 
                   const WSTR system_prompt, const WSTR tail_prompt)
```

### 2. Input Preparation

**Steps:**
1. Print input text (system_prompt + prompt + tail_prompt)
2. Tokenize the input using huggingface_tokenizer
3. Truncate if too long: `max(MAX_SEQ_LEN - NUM_TO_GENERATE)`
4. Prepare input tensor: `BATCH_SIZE × MAX_SEQ_LEN`

**KV Cache Mode:**
- If `USE_KVCACHE` and pre-computed cache exists: Load KV cache for system prompt
- If `USE_KVCACHE` and cache doesn't exist: Will save after prefill

### 3. Prefill Phase

**Purpose:** Process all input tokens at once using parallel attention.

**File:** `causal_lm.cpp`

```cpp
// Load pre-computed KV cache if available
if (USE_KVCACHE) {
  load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
}

// Prefill: process all input tokens
output = model->incremental_inference(
  BATCH_SIZE,           // batch size
  input,               // input tensor
  label,               // (empty)
  init_len,            // sequence length
  SYS_PROMP_LEN,       // from position
  SYS_PROMP_LEN + input_len,  // to position
  false                // is_decoding
);

// Generate first token after prefill
id_list = generate_multi_tokens(output[0], NUM_VOCAB, BATCH_SIZE, 1, ids_history, _len);
```

**Incremental Inference:**
- Processes tokens in parallel for efficient prefill
- Builds KV cache for each attention head
- KV cache stores Key and Value tensors for all processed tokens

### 4. Generation Phase

**Purpose:** Generate tokens one at a time using autoregressive decoding.

**File:** `causal_lm.cpp`

```cpp
for (token_generation_idx = input_len + 1;
     token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
     ++token_generation_idx) {
  
  // Generate next token
  auto output = model->incremental_inference(
    BATCH_SIZE,
    input,           // contains only previous generated token
    label,
    input_len,       // sequence length (always 1 during generation)
    token_generation_idx - 1 + global_token_len,  // from position
    token_generation_idx + global_token_len       // to position
  );
  
  // Sample from logits
  ids_list = generate(output[0], do_sample);
  
  // Update input with new token
  input_sample[b * MAX_SEQ_LEN] = static_cast<float>(ids_list[b]);
  
  // Register output for display
  registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
  
  // Check for EOS token
  if (ids_list[b] == EOS_TOKEN_ID) {
    eos_list[b] = true;
  }
  
  // Stop if all batches finished
  if (all(eos_list)) {
    break;
  }
}
```

### 5. generate() - Token Sampling

**File:** `causal_lm.cpp`

```cpp
std::vector<unsigned int> generate(float *logits, bool do_sample) {
  if (do_sample == false) {
    // Greedy decoding: pick argmax
    argmax_idx = argmax(logits);
  } else {
    // Apply temperature scaling
    logits = logits / TEMPERATURE;
    
    // Apply top-k filtering
    logits = top_k_filter(logits, TOP_K);
    
    // Apply top-p (nucleus) sampling
    logits = top_p_filter(logits, TOP_P);
    
    // Apply softmax
    probs = softmax(logits);
    
    // Sample from distribution
    token_idx = sample(probs);
  }
}
```

**Sampling Strategies:**
- **Greedy (do_sample=false):** Always pick highest probability token
- **Temperature:** Control randomness (higher = more random)
- **Top-K:** Keep only K most likely tokens
- **Top-P:** Keep tokens with cumulative probability ≤ P

### 6. registerOutputs() - Display Generated Text

**File:** `causal_lm.cpp

```cpp
void CausalLM::registerOutputs(tokenizer, ids, pos, eos_list) {
  for (batch b = 0; b < BATCH_SIZE; ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(ids[b]);
      decoded_str = tokenizer->Decode(pending_ids_);
      
      // Display only when we have a complete token
      // (handle incomplete tokens and punctuation)
      if (is_complete_token(decoded_str)) {
        std::cout << decoded_str;
        output_list[b].append(decoded_str);
        pending_ids_.clear();
      }
    }
  }
}
```

## Key Components

### 1. Tokenizer Integration

**File:** `transformer.cpp`

```cpp
// Load tokenizer from huggingface format
tokenizer = tokenizers::Tokenizer::FromBlobJSON(
  LoadBytesFromFile(nntr_cfg["tokenizer_file"])
);

// Encode text to token IDs
auto _input = tokenizer->Encode(prompt);

// Decode token IDs to text
std::string decoded_str = tokenizer->Decode(pending_ids_);
```

### 2. KV Cache

**Purpose:** Cache Key and Value tensors to avoid recomputing attention for all previous tokens.

**File:** `causal_lm.cpp`

```cpp
void save_kvcache(std::string path, int to) {
  // Iterate through all MHACore layers
  model->forEachLayer([](Layer &l, RunLayerContext &context, void *to) {
    if (l.getType() == MHACoreLayer::type) {
      auto k_cache = context.getTensor(0);  // Key cache
      auto v_cache = context.getTensor(1);  // Value cache
      
      // Save only up to position 'to'
      k_cache_prompt = k_cache.getSharedDataTensor(dim, 0, true);
      k_cache_prompt.save(f);
      v_cache_prompt.save(f);
    }
  });
}

void load_kvcache(std::string path, int to) {
  // Load KV cache from file
  // Similar to save_kvcache but reads from file
}
```

**Use Cases:**
- System prompt caching: Pre-compute and cache system prompt KV cache
- Faster inference: Avoid recomputing attention for repeated prompts

### 3. Incremental Inference

**File:** nntrainer core (not in Applications/)

Two modes:
- **Prefill mode:** Process multiple tokens in parallel
- **Decoding mode:** Process one token at a time

During decoding, each new token only attends to:
- Previous tokens via KV cache
- Current token via standard attention

## Example: Gemma3CausalLM

**File:** `gemma3/gemma3_causallm.h`

```cpp
class Gemma3CausalLM : public CausalLM, public Gemma3Transformer {
public:
  Gemma3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) 
    : Transformer(sanitizeConfig(cfg), ...),
      CausalLM(sanitizeConfig(cfg), ...),
      Gemma3Transformer(sanitizeConfig(cfg), ...)
  {}
  
  // Override specific methods if needed
  void setupParameters(...) override;
  void registerCustomLayers() override;
};
```

**Gemma3-Specific Features:**
- Custom embedding scaling: `EMBEDDING_SCALE = sqrt(hidden_size)`
- Layer type configuration (mix of different block types)
- Custom attention/MLP implementations

## T5Gemma2 Integration Points

For T5Gemma2 implementation, you need to:

### 1. Create T5Gemma2CausalLM Class

```cpp
class T5Gemma2CausalLM : public CausalLM, public T5Gemma2Transformer {
  // Similar to Gemma3CausalLM pattern
};
```

### 2. Handle Image Inputs

**Current CausalLM workflow assumes text-only. For T5Gemma2:**

1. **Extend run() method:**
```cpp
void T5Gemma2CausalLM::run(
  const WSTR prompt, 
  const std::vector<std::string> &images,  // New parameter
  bool do_sample = false,
  const WSTR system_prompt = "", 
  const WSTR tail_prompt = "")
{
  // Process images with T5Gemma2Processor
  auto processor_output = processor.process(prompt, images);
  
  // processor_output contains:
  // - pixel_values: image tensor (batch, 3, 896, 896)
  // - input_ids: token IDs with image placeholders
  // - attention_mask: attention mask
  // - token_type_ids: token type IDs (0=text, 1=image)
  
  // Use processor_output.input_ids for tokenization
  // Feed processor_output.pixel_values to vision encoder
}
```

2. **Add Vision Encoder:**
   - Create vision encoder similar to `timm_vit_transformer`
   - Integrate with text encoder via cross-attention
   - May need to extend `Transformer` base class

3. **Modify Input Preparation:**
```cpp
// Instead of:
input_sample[batch][pos] = token_id;

// Use processor output:
auto processor_output = processor.process(prompt, images);
input_ids = processor_output.input_ids;
pixel_values = processor_output.pixel_values;
token_type_ids = processor_output.token_type_ids;
```

### 3. Modify Model Construction

Add vision encoder layers:
```
[Input Images] → [Vision Encoder] → [Image Embeddings]
                                         ↓
[Input Tokens] → [Text Encoder] → [Token Embeddings] → [Cross-Attention]
```

## Performance Optimizations

### 1. KV Cache
- Pre-compute system prompt KV cache once
- Reuse across multiple generations with same system prompt
- Reduces prefill time for repeated queries

### 2. Memory Swap (FSU)
- Offload KV cache to disk when memory is limited
- Controlled by `fsu` and `fsu_lookahead` parameters

### 3. Sliding Window Attention
- Limit attention to recent N tokens
- Configurable via `sliding_window` parameter
- Useful for long sequences

### 4. Quantization
- Control tensor types via:
  - `model_tensor_type`: Main model tensors
  - `embedding_dtype`: Embedding layer
  - `lmhead_dtype`: Output projection
  - `fc_layer_dtype`: Feed-forward layers

## Configuration Files

### config.json
```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128000,
  "intermediate_size": 27392,
  "max_position_embeddings": 8192,
  "rope_theta": 10000,
  "rms_norm_eps": 1e-5,
  "tie_word_embeddings": true
}
```

### generation_config.json
```json
{
  "bos_token_id": 1,
  "eos_token_id": [128001, 128009],
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.95
}
```

### nntrainer_config.json
```json
{
  "batch_size": 1,
  "model_tensor_type": "FP16",
  "embedding_dtype": "FP32",
  "init_seq_len": 512,
  "max_seq_len": 8192,
  "num_to_generate": 512,
  "tokenizer_file": "tokenizer.json"
}
```

## Using CausalLM from main.cpp

**File:** `Applications/CausalLM/main.cpp`

The main.cpp executable demonstrates how to use CausalLM models end-to-end.

### Complete Workflow

#### 1. Register Models to Factory

```cpp
// Register all supported models with their architecture names
causallm::Factory::Instance().registerModel(
  "Gemma3ForCausalLM",
  [](json cfg, json generation_cfg, json nntr_cfg) {
    return std::make_unique<causallm::Gemma3CausalLM>(
      cfg, generation_cfg, nntr_cfg);
  }
);

// Register more models: Qwen2, Qwen3, GptOss, Llama, etc.
```

**Registered Models:**
- `LlamaForCausalLM` - Uses base CausalLM class
- `Qwen2ForCausalLM` - Qwen2-specific implementation
- `Qwen3ForCausalLM` - Qwen3-specific implementation
- `Qwen3MoeForCausalLM` - Qwen3 with Mixture of Experts
- `GptOssForCausalLM` - GPT Open Source implementation
- `Gemma3ForCausalLM` - Gemma3-specific implementation
- `TimmViT` - Vision Transformer (for image models)

#### 2. Parse Command Line Arguments

```cpp
int main(int argc, char *argv[]) {
  // Usage: ./nntr_causallm <model_path> [input_prompt]
  const std::string model_path = argv[1];
  std::string input_text;
  
  if (argc >= 3) {
    input_text = argv[2];  // Use provided prompt
  } else {
    input_text = nntr_cfg["sample_input"];  // Use sample from config
  }
}
```

#### 3. Load Configuration Files

```cpp
// Load model configuration
json cfg = LoadJsonFile(model_path + "/config.json");
json generation_cfg = LoadJsonFile(model_path + "/generation_config.json");
json nntr_cfg = LoadJsonFile(model_path + "/nntr_config.json");

// Get system prompts if available
if (nntr_cfg.contains("system_prompt")) {
  system_head_prompt = nntr_cfg["system_prompt"]["head_prompt"];
  system_tail_prompt = nntr_cfg["system_prompt"]["tail_prompt"];
}
```

**Configuration Files:**
- `config.json` - Model architecture and parameters
- `generation_config.json` - Generation parameters (temperature, top_k, top_p)
- `nntrainer_config.json` - Runtime configuration (paths, sample_input, system_prompt)

#### 4. Resolve Architecture

```cpp
std::string architecture = cfg["architectures"][0];

// Handle special model types (e.g., embedding models)
if (nntr_cfg.contains("model_type")) {
  std::string model_type = nntr_cfg["model_type"];
  architecture = resolve_architecture(model_type, architecture);
}

// Example mappings:
// - model_type="embedding", architecture="Gemma3ForCausalLM" → "EmbeddingGemma"
// - model_type="embedding", architecture="Qwen3ForCausalLM" → "Qwen3Embedding"
```

#### 5. Create Model Instance

```cpp
// Use factory to create model instance based on architecture
auto model = causallm::Factory::Instance().create(
  architecture,  // e.g., "Gemma3ForCausalLM"
  cfg,
  generation_cfg,
  nntr_cfg
);
```

The factory pattern allows:
- Easy addition of new models
- Runtime model selection
- Clean separation of model creation logic

#### 6. Initialize Model

```cpp
model->initialize();
```

This calls the initialization sequence:
1. Register custom layers
2. Construct model architecture
3. Set model properties
4. Compile model
5. Initialize model
6. Set `is_initialized = true`

#### 7. Load Weights

```cpp
const std::string weight_file = model_path + "/" + nntr_cfg["model_file_name"];
model->load_weight(weight_file);
```

Weights are loaded from a binary file in `.bin` format.

#### 8. Run Inference

```cpp
// Determine sampling mode
bool do_sample = generation_cfg.value("do_sample", false);

// Run the model
model->run(
  input_text,           // User prompt
  do_sample,            // Whether to use sampling or greedy decoding
  system_head_prompt,   // System prompt prefix (before user input)
  system_tail_prompt    // System prompt suffix (after user input)
);
```

**System Prompt Pattern:**
```
<system_head_prompt> + <user_prompt> + <system_tail_prompt>

Example (chat format):
"You are a helpful assistant.\n\n" + "What is the capital of France?" + "\n\nAnswer:"
```

#### 9. Monitor Performance

```cpp
#ifdef PROFILE
  start_peak_tracker();  // Track peak memory usage
#endif

model->run(...);

#ifdef PROFILE
  stop_and_print_peak();  // Print peak memory
#endif

// Print timing
auto e2e_duration = finish_time - start_time;
std::cout << "[e2e time]: " << e2e_duration.count() << " ms\n";
printMemoryUsage();
```

**Performance Metrics:**
- End-to-end time (including initialization, weight loading, inference)
- Peak memory usage (VmRSS)
- Per-token generation statistics (printed by CausalLM::run())

### Complete Example

```bash
# Build the executable
ninja -C build/Applications/CausalLM nntr_causallm

# Run with custom prompt
./nntr_causallm /path/to/model "Describe this image in detail"

# Run with default prompt from config
./nntr_causallm /path/to/model

# Expected output:
/path/to/model
/path/to/model/model.bin
You are a helpful assistant.

Describe this image in detail

Answer: The image shows a beautiful landscape...

=================[ LLM with NNTrainer ]===================
prefill: 20 tokens, 150 ms, 133 TPS
generation: 50 tokens, 200 ms, 250 TPS
[e2e time]: 450 ms
Max Resident Set Size: 1234567 KB
```

### Model Directory Structure

```
/path/to/model/
├── config.json              # Model architecture
├── generation_config.json   # Generation parameters
├── nntr_config.json         # Runtime configuration
├── model.bin                # Pre-trained weights
├── tokenizer.json           # Tokenizer (huggingface format)
└── (optional) sample_input.txt
```

### Adding T5Gemma2 to main.cpp

To add T5Gemma2 support to main.cpp:

```cpp
// Register T5Gemma2 model
causallm::Factory::Instance().registerModel(
  "T5Gemma2ForCausalLM",
  [](json cfg, json generation_cfg, json nntr_cfg) {
    return std::make_unique<causallm::T5Gemma2CausalLM>(
      cfg, generation_cfg, nntr_cfg);
  }
);
```

Then ensure:
1. `T5Gemma2CausalLM` class is defined
2. Header file is included: `#include "t5gemma2_causallm.h"`
3. Model directory has `config.json` with `"architectures": ["T5Gemma2ForCausalLM"]`

### Memory Tracking

main.cpp includes memory tracking utilities:

```cpp
void start_peak_tracker() {
  // Start background thread to track peak RSS
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      peak_rss_kb.store(std::max(peak_rss_kb.load(), current));
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB\n";
}
```

Enable with `-DPROFILE` compilation flag.

## Summary

The CausalLM workflow in nntrainer follows this pattern:

### From main.cpp (Application Level)
1. Register models to factory
2. Parse command line arguments
3. Load configuration files (config.json, generation_config.json, nntr_config.json)
4. Resolve architecture name
5. Create model instance via factory
6. Initialize model
7. Load weights
8. Run inference with prompt
9. Monitor performance

### From CausalLM Class (Model Level)
1. **Initialization:** Load configs → Setup parameters → Register custom layers → Construct model → Compile → Initialize
2. **Load Weights:** Load pre-trained weights from binary file
3. **Input Preparation:** Tokenize prompt → Prepare input tensor
4. **Prefill:** Process all input tokens in parallel, build KV cache
5. **Generation:** Generate tokens one-by-one using KV cache and sampling
6. **Output:** Display generated text as complete tokens are produced

For T5Gemma2, you'll need to extend this workflow to handle image inputs by:
- Integrating the T5Gemma2Processor in input preparation
- Adding a vision encoder to the model architecture
- Implementing cross-attention between text and image embeddings
- Modifying run() to accept image paths
- Registering T5Gemma2ForCausalLM in factory

## References

- Base classes: `causal_lm.h/cpp`, `transformer.h/cpp`
- Main executable: `Applications/CausalLM/main.cpp`
- Factory pattern: `Applications/CausalLM/factory.h`
- Example implementation: `gemma3/gemma3_causallm.h/cpp`
- Custom layers: `Applications/CausalLM/layers/`
- Layer implementations: `nntrainer/layers/`
- Processor integration: `t5gemma2/t5gemma2_processor.h/cpp`
