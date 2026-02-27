# NNTrainer에서 두 모델 결합하기

## 개요

NNTrainer에서 두 개 이상의 모델(Vision Encoder + Text Encoder 등)을 결합하는 방법을 설명합니다.

## 방법 1: 단일 모델 내에서 모든 레이어 추가 (권장)

### 개념

하나의 `ModelHandle` 안에 Vision Encoder와 Text Encoder의 모든 레이어를 순차적으로 추가합니다.

### 장점
- 구현이 간단
- 한 번의 `initialize()`와 `compile()` 호출
- 한 번의 `incremental_inference()`로 전체 파이프라인 실행

### 구현 예시

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2Transformer::constructModel() {
  std::vector<LayerHandle> layers;
  
  // ========== VISION ENCODER ==========
  
  // Vision input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "vision_input0"),
              withKey("input_shape", "1:3:896:896")}));  // NCHW format
  
  // Vision embedding (Patch embedding + Positional embedding)
  layers.push_back(createLayer(
    "conv2d",
    {withKey("name", "vision_patch_embedding"),
     withKey("input_layers", "vision_input0"),
     withKey("filters", DIM),
     withKey("kernel_size", "14,14"),
     withKey("stride", "14,14"),
     withKey("padding", "0,0"),
     withKey("bias", "true")}));
  
  // Vision transformer blocks
  for (int i = 0; i < NUM_VISION_LAYERS; ++i) {
    std::vector<LayerHandle> vision_block =
      createVisionTransformerBlock(i, 
        i == 0 ? "vision_patch_embedding" 
                : "vision_layer" + std::to_string(i-1) + "_output");
    layers.insert(layers.end(), vision_block.begin(), vision_block.end());
  }
  
  // Vision output normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "vision_output_norm"),
     withKey("input_layers", "vision_layer" + 
             std::to_string(NUM_VISION_LAYERS-1) + "_output"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // ========== TEXT ENCODER ==========
  
  // Text input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "text_input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  
  // Text embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  layers.push_back(createLayer(
    embedding_type,
    {"name=text_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  
  // ========== FUSION: Vision + Text ==========
  
  // Image token embedding (from vision output)
  // vision_output_norm의 shape: [BATCH, NUM_PATCHES, DIM]
  // 이를 flatten해서 [BATCH, NUM_PATCHES*DIM]으로 변환
  layers.push_back(createLayer(
    "reshape",
    {withKey("name", "vision_reshape"),
     withKey("input_layers", "vision_output_norm"),
     withKey("target_shape", "1:1:" + std::to_string(NUM_PATCHES*DIM))}));
  
  // Vision embeddings를 text embeddings에 concat
  // text_embedding0: [BATCH, TEXT_SEQ_LEN, DIM]
  // vision_reshape: [BATCH, NUM_IMAGE_TOKENS, DIM]
  // concat: [BATCH, TEXT_SEQ_LEN+NUM_IMAGE_TOKENS, DIM]
  layers.push_back(createLayer(
    "concat",
    {withKey("name", "vision_text_fusion"),
     withKey("input_layers", "text_embedding0,vision_reshape"),
     withKey("axis", "2")}));
  
  // ========== ENCODER BLOCKS (with fused input) ==========
  
  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
    std::vector<LayerHandle> encoder_block =
      createTransformerEncoderBlock(i,
        i == 0 ? "vision_text_fusion"
                : "encoder_layer" + std::to_string(i-1) + "_output");
    layers.insert(layers.end(), encoder_block.begin(), encoder_block.end());
  }
  
  // Encoder output normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "encoder_output_norm"),
     withKey("input_layers", "encoder_layer" + 
             std::to_string(NUM_ENCODER_LAYERS-1) + "_output"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // ========== DECODER ==========
  
  // Decoder input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "decoder_input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  
  // Decoder embedding layer (shared with text)
  layers.push_back(createLayer(
    embedding_type,
    {"name=decoder_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE),
     withKey("shared_from", "text_embedding0")}));
  
  // Decoder transformer blocks with cross-attention
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    std::vector<LayerHandle> decoder_block =
      createTransformerDecoderBlock(i,
        i == 0 ? "decoder_embedding0"
                : "decoder_layer" + std::to_string(i-1) + "_output",
        "encoder_output_norm");
    layers.insert(layers.end(), decoder_block.begin(), decoder_block.end());
  }
  
  // Decoder output normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "decoder_output_norm"),
     withKey("input_layers", "decoder_layer" + 
             std::to_string(NUM_DECODER_LAYERS-1) + "_output"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // LM Head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_t5gemma2"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "decoder_output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "text_embedding0"));
  
  layers.push_back(createLayer(lmhead_type, lmhead_prop));
  
  // Add all layers to model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}
```

### 여러 Input 받기: 가능합니다!

**incremental_inference 시그니처 (nntrainer/models/neuralnet.h):**

```cpp
std::vector<float *>
incremental_inference(unsigned int batch, const std::vector<float *> &input,
                     const std::vector<float *> &label,
                     unsigned int init_seq_len, unsigned int from,
                     unsigned int to,
                     bool output_hidden_state = false);
```

**중요:** `input` 파라미터가 `const std::vector<float *> &` 타입입니다!

이것은 **여러 개의 input tensor를 받을 수 있다**는 것을 의미합니다.

### T5Gemma2 예시: 3개의 Input

```cpp
void T5Gemma2ForConditionalGeneration::run(...) {
  // 1. Process inputs with T5Gemma2Processor
  auto processor_output = processor.process(prompt, images);
  
  // 2. Prepare multiple inputs
  std::vector<float *> inputs;
  
  // Input 1: Vision input [BATCH, 3, 896, 896]
  float *vision_input = processor_output.pixel_values.data();
  inputs.push_back(vision_input);  // → vision_input0으로 들어감
  
  // Input 2: Text input [BATCH, TEXT_SEQ_LEN]
  float *text_input = processor_output.input_ids.data();
  inputs.push_back(text_input);  // → text_input0으로 들어감
  
  // Input 3: Decoder input [BATCH, 1] (generation phase only)
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  inputs.push_back(decoder_input);  // → decoder_input0으로 들어감
```

### 추론 방법 (여러 Input)

**단계 1: Vision Encoder Prefill**
```cpp
// Vision input만 사용
std::vector<float *> vision_inputs = {vision_input};
auto vision_output = model->incremental_inference(
  BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
```

**단계 2: Text Encoder Prefill (Vision Output 포함)**
```cpp
// Text input + Vision output
std::vector<float *> text_inputs = {text_input, vision_output[0]};
auto encoder_output = model->incremental_inference(
  BATCH_SIZE, text_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
```

**단계 3: Decoder Generation**
```cpp
// Decoder input + Encoder output
for (int i = 0; i < NUM_TO_GENERATE; ++i) {
  std::vector<float *> decoder_inputs = {decoder_input, encoder_output[0]};
  auto decoder_output = model->incremental_inference(
    BATCH_SIZE, decoder_inputs, {}, 1, i, i+1, true);
  
  // Sample token
  unsigned int token = generate(decoder_output[0], do_sample);
  
  // Update input
  decoder_input[0] = static_cast<float>(token);
  
  // Check EOS
  if (token == EOS_TOKEN_ID) break;
}
```

### Input 순서 중요

**Model에 추가한 Input Layer 순서와 일치해야 합니다:**

```cpp
void T5Gemma2Transformer::constructModel() {
  // Input Layer 추가 순서:
  // 1. vision_input0
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  // 2. text_input0
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  // 3. decoder_input0
  model->addLayer(createLayer("input", {
    withKey("name", "decoder_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  // ... 레이어들 ...
}

// inference 시 input 순서:
std::vector<float *> inputs = {
  vision_input,  // → vision_input0
  text_input,     // → text_input0
  decoder_input    // → decoder_input0
};
auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
```

### 실제 사용 예시

**CausalLM (단일 Input):**
```cpp
std::vector<float *> inputs = {input_ids};
auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
```

**T5Gemma2 (3개 Input - Vision + Text + Decoder):**
```cpp
std::vector<float *> inputs = {
  vision_input,  // vision_input0
  text_input,     // text_input0
  decoder_input    // decoder_input0
};
auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
```

**특정 단계에서는 일부 Input만 사용:**
```cpp
// Vision encoder만 실행
std::vector<float *> vision_inputs = {vision_input};
auto vision_output = model->incremental_inference(BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);

// Text encoder만 실행 (vision output는 concat 레이어에서 처리)
std::vector<float *> text_inputs = {text_input};
auto text_output = model->incremental_inference(BATCH_SIZE, text_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);

// Decoder만 실행 (encoder output는 cross-attention에서 사용)
std::vector<float *> decoder_inputs = {decoder_input, encoder_output[0]};
auto decoder_output = model->incremental_inference(BATCH_SIZE, decoder_inputs, {}, 1, pos, pos+1, true);
```

### 결론

**네, 여러 input을 받을 수 있습니다!**

`incremental_inference` 함수의 `input` 파라미터가 `std::vector<float *>` 타입이므로, 원하는 만큼의 input tensor를 전달할 수 있습니다.

**주의사항:**
1. Input Layer 추가 순서와 input 전달 순서가 일치해야 함
2. 각 input의 shape이 해당 Input Layer의 `input_shape`와 일치해야 함
3. 필요한 단계에서는 일부 input만 전달 가능 (나머지는 0 또는 nullptr)

이 기능 덕분에 T5Gemma2와 같은 멀티모달 모델을 효율적으로 구현할 수 있습니다!

```cpp
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,
    bool do_sample,
    const WSTR system_prompt,
    const WSTR tail_prompt) {
  
  // 1. Process images with T5Gemma2Processor
  auto processor_output = processor.process(prompt, images);
  
  // 2. Prepare inputs
  std::vector<float *> inputs;
  
  // Vision input: [BATCH, 3, 896, 896]
  float *vision_input = processor_output.pixel_values.data();
  inputs.push_back(vision_input);
  
  // Text input: [BATCH, TEXT_SEQ_LEN]
  float *text_input = processor_output.input_ids.data();
  inputs.push_back(text_input);
  
  // 3. Vision encoder prefill
  auto vision_output = model->incremental_inference(
    BATCH_SIZE, {vision_input}, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
  
  // 4. Text encoder prefill (with vision embeddings)
  auto encoder_output = model->incremental_inference(
    BATCH_SIZE, {text_input, vision_output[0]}, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  
  // 5. Decoder generation
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  
  for (int i = 0; i < NUM_TO_GENERATE; ++i) {
    auto decoder_output = model->incremental_inference(
      BATCH_SIZE, {decoder_input}, {}, 1, i, i+1, true, encoder_output[0]);
    
    // Sample token
    unsigned int token = generate(decoder_output[0], do_sample);
    
    // Update input
    decoder_input[0] = static_cast<float>(token);
    
    // Check EOS
    if (token == EOS_TOKEN_ID) break;
  }
  
  free(decoder_input);
}
```

## 방법 2: 두 개의 ModelHandle 사용

### 개념

Vision Encoder와 Text Encoder를 각각 별도의 ModelHandle로 생성하고, 두 모델을 수동으로 연결합니다.

### 장점
- 모듈성이 높음
- 각 모델을 독립적으로 테스트 가능
- 메모리 효율적 (필요한 경우에만 실행)

### 구현 예시

```cpp
// t5gemma2_causallm.cpp
class T5Gemma2Transformer : virtual public Transformer {
private:
  ModelHandle vision_model;
  ModelHandle text_model;
  ModelHandle decoder_model;
  
public:
  void constructModel() override {
    // 1. Create vision model
    vision_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    constructVisionEncoder(vision_model);
    
    // 2. Create text model
    text_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    constructTextEncoder(text_model);
    
    // 3. Create decoder model
    decoder_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    constructDecoder(decoder_model);
    
    // Set properties for each model
    auto vision_props = getVisionModelProperties();
    vision_model->setProperty(vision_props);
    vision_model->compile(ExecutionMode::INFERENCE);
    vision_model->initialize(ExecutionMode::INFERENCE);
    
    auto text_props = getTextModelProperties();
    text_model->setProperty(text_props);
    text_model->compile(ExecutionMode::INFERENCE);
    text_model->initialize(ExecutionMode::INFERENCE);
    
    auto decoder_props = getDecoderModelProperties();
    decoder_model->setProperty(decoder_props);
    decoder_model->compile(ExecutionMode::INFERENCE);
    decoder_model->initialize(ExecutionMode::INFERENCE);
  }
  
private:
  void constructVisionEncoder(ModelHandle &model) {
    std::vector<LayerHandle> layers;
    
    layers.push_back(createLayer(
      "input", {withKey("name", "vision_input0"),
                withKey("input_shape", "1:3:896:896")}));
    
    layers.push_back(createLayer(
      "conv2d",
      {withKey("name", "vision_patch_embedding"),
       withKey("input_layers", "vision_input0"),
       withKey("filters", DIM),
       withKey("kernel_size", "14,14"),
       withKey("stride", "14,14")}));
    
    // Add vision transformer blocks...
    
    for (auto &layer : layers) {
      model->addLayer(layer);
    }
  }
  
  void constructTextEncoder(ModelHandle &model) {
    std::vector<LayerHandle> layers;
    
    layers.push_back(createLayer(
      "input", {withKey("name", "text_input0"),
                withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
    
    layers.push_back(createLayer(
      "embedding_layer",
      {"name=text_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
       "out_dim=" + std::to_string(DIM)}));
    
    // Add text transformer blocks...
    
    for (auto &layer : layers) {
      model->addLayer(layer);
    }
  }
  
  void constructDecoder(ModelHandle &model) {
    std::vector<LayerHandle> layers;
    
    layers.push_back(createLayer(
      "input", {withKey("name", "decoder_input0"),
                withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
    
    layers.push_back(createLayer(
      "embedding_layer",
      {"name=decoder_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
       "out_dim=" + std::to_string(DIM)}));
    
    // Add decoder transformer blocks with cross-attention...
    
    for (auto &layer : layers) {
      model->addLayer(layer);
    }
  }
};

void T5Gemma2ForConditionalGeneration::run(...) {
  // 1. Run vision encoder
  std::vector<float *> vision_inputs = {pixel_values};
  auto vision_output = vision_model->incremental_inference(
    BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
  
  // 2. Run text encoder with vision embeddings
  std::vector<float *> text_inputs = {input_ids, vision_output[0]};
  auto encoder_output = text_model->incremental_inference(
    BATCH_SIZE, text_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  
  // 3. Run decoder
  std::vector<float *> decoder_inputs = {decoder_input};
  auto decoder_output = decoder_model->incremental_inference(
    BATCH_SIZE, decoder_inputs, {}, 1, pos, pos+1, true, encoder_output[0]);
}
```

### 문제점
- 두 모델 사이의 메모리 공유가 어려움
- 추론 시 메모리 복사 필요
- KV cache 관리가 복잡

## 방법 3: 여러 개의 Input Layer 사용

### 개념

하나의 ModelHandle 안에서 여러 개의 Input Layer를 사용하고, Concat 레이어로 결합합니다.

### 구현 예시

```cpp
void T5Gemma2Transformer::constructModel() {
  // Vision input
  model->addLayer(createLayer(
    "input", {withKey("name", "vision_input"),
              withKey("input_shape", "1:3:896:896")}));
  
  // Text input
  model->addLayer(createLayer(
    "input", {withKey("name", "text_input"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)}}));
  
  // Vision encoder
  constructVisionEncoder("vision_input", "vision_output");
  
  // Text encoder
  constructTextEncoder("text_input", "text_output");
  
  // Fusion
  model->addLayer(createLayer(
    "concat",
    {withKey("name", "vision_text_fusion"),
     withKey("input_layers", "vision_output,text_output"),
     withKey("axis", "2")}));
  
  // Combined encoder
  constructCombinedEncoder("vision_text_fusion", "encoder_output");
  
  // Decoder
  constructDecoder("encoder_output", "final_output");
}
```

### 추론

```cpp
void run(...) {
  // 여러 개의 input
  std::vector<float *> inputs = {
    vision_input,  // vision_input0으로 들어감
    text_input     // text_input0으로 들어감
  };
  
  // 한 번의 inference
  auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
}
```

## 권장 접근 방식: 방법 1 (단일 모델)

### 이유
1. **구현이 간단:** 한 번의 `constructModel()` 호출
2. **자동 연결:** nntrainer가 레이어 연결 자동 처리
3. **KV Cache:** 쉽게 관리 가능
4. **성능:** 메모리 복사 없이 직접 연결

### 구조

```
[vision_input0] → [Vision Encoder] → [vision_output_norm]
                                                      ↓
[text_input0] → [Text Embedding] → [concat] → [Encoder] → [encoder_output_norm]
                                                      ↓
[decoder_input0] → [Decoder Embedding] → [Decoder (cross-attn)] → [Output]
```

### 레이어 연결

nntrainer는 `input_layers` 파라미터를 통해 레이어를 자동으로 연결합니다:

```cpp
// input_layers를 사용하여 레이어 연결
model->addLayer(createLayer(
  "rms_norm",
  {withKey("name", "vision_output_norm"),
   withKey("input_layers", "vision_layer11_output"),  // 이 레이어의 출력을 입력으로 사용
   withKey("epsilon", std::to_string(NORM_EPS))}));

// Concat으로 두 개의 출력을 결합
model->addLayer(createLayer(
  "concat",
  {withKey("name", "vision_text_fusion"),
   withKey("input_layers", "text_embedding0,vision_reshape"),  // 콤마로 구분
   withKey("axis", "2")}));
```

## T5Gemma2 완전 구조 예시

```cpp
void T5Gemma2Transformer::constructModel() {
  // ========== Vision Encoder ==========
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  model->addLayer(createLayer("conv2d", {
    withKey("name", "vision_patch_embedding"),
    withKey("input_layers", "vision_input0"),
    withKey("filters", DIM),
    withKey("kernel_size", "14,14"),
    withKey("stride", "14,14")
  }));
  
  // Vision transformer blocks (NUM_VISION_LAYERS)
  for (int i = 0; i < NUM_VISION_LAYERS; ++i) {
    addVisionTransformerBlock(i);
  }
  
  model->addLayer(createLayer("rms_norm", {
    withKey("name", "vision_output_norm"),
    withKey("input_layers", "vision_layer" + std::to_string(NUM_VISION_LAYERS-1) + "_output"),
    withKey("epsilon", std::to_string(NORM_EPS))
  }));
  
  // ========== Text Encoder ==========
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM)
  }));
  
  // ========== Fusion ==========
  model->addLayer(createLayer("reshape", {
    withKey("name", "vision_reshape"),
    withKey("input_layers", "vision_output_norm"),
    withKey("target_shape", "1:1:" + std::to_string(NUM_PATCHES*DIM)
  }));
  
  model->addLayer(createLayer("concat", {
    withKey("name", "vision_text_fusion"),
    withKey("input_layers", "text_embedding0,vision_reshape"),
    withKey("axis", "2")
  }));
  
  // ========== Encoder Blocks (with fused input) ==========
  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
    addEncoderBlock(i);
  }
  
  model->addLayer(createLayer("rms_norm", {
    withKey("name", "encoder_output_norm"),
    withKey("input_layers", "encoder_layer" + std::to_string(NUM_ENCODER_LAYERS-1) + "_output"),
    withKey("epsilon", std::to_string(NORM_EPS))
  }));
  
  // ========== Decoder ==========
  model->addLayer(createLayer("input", {
    withKey("name", "decoder_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "decoder_embedding0"),
    withKey("input_layers", "decoder_input0"),
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM),
    withKey("shared_from", "text_embedding0")
  }));
  
  // Decoder blocks (NUM_DECODER_LAYERS)
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    addDecoderBlock(i, "encoder_output_norm");
  }
  
  model->addLayer(createLayer("rms_norm", {
    withKey("name", "decoder_output_norm"),
    withKey("input_layers", "decoder_layer" + std::to_string(NUM_DECODER_LAYERS-1) + "_output"),
    withKey("epsilon", std::to_string(NORM_EPS))
  }));
  
  // LM Head
  model->addLayer(createLayer("lm_head", {
    withKey("name", "output_of_t5gemma2"),
    withKey("input_layers", "decoder_output_norm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("shared_from", "text_embedding0")
  }));
}
```

## 결론

**권장 방법:** 방법 1 (단일 모델 내에서 모든 레이어 추가)

### 이유
1. 구현이 가장 간단
2. nntrainer의 레이어 연결 기능을 최대한 활용
3. KV cache 관리가 쉬움
4. 성능 최적화 (메모리 복사 없음)

### 핵심 레이어
- **Concat:** Vision과 Text 출력을 결합
- **Reshape:** 차원 조정
- **Cross-Attention:** Encoder output을 Decoder가 활용

이 방식으로 T5Gemma2와 같은 멀티모달 Encoder-Decoder 모델을 효율적으로 구현할 수 있습니다.

## 2단계 구조: Processor + Vision Encoder (Optional) // Encoder + Decoder

이것은 T5Gemma2와 같은 멀티모달 모델에서 매우 일반적이고 효율적인 구조입니다.

### 구조

```
STEP 1: 전처리 (Processor)
────────────────────────────
[이미지] → [T5Gemma2Processor] → [pixel_values, input_ids, attention_mask, token_type_ids]
                                  ↓
STEP 2: Vision Encoder (Optional - 이미지가 있을 때만 실행)
───────────────────────────────────────────────
[pixel_values] → [Vision Encoder] → [vision_embeddings]
                                    ↓
STEP 3: Text Encoder
────────────────────────
[input_ids, vision_embeddings] → [Text Encoder] → [encoder_output]
                                    ↓
STEP 4: Decoder
────────────────
[decoder_input, encoder_output] → [Decoder (cross-attn)] → [output]
```

### 이 구조의 장점

1. **유연성:** 이미지가 없는 경우 Vision Encoder를 건너뛸 수 있음
2. **모듈성:** 각 단계가 독립적
3. **효율성:** 필요한 부분만 실행
4. **유지보수성:** T5 같은 텍스트 전용 모델과 유사한 구조 유지

### 구현 예시

#### 구조

```cpp
class T5Gemma2ForConditionalGeneration {
private:
  std::unique_ptr<T5Gemma2Processor> processor;
  std::unique_ptr<Transformer> model;
  
public:
  void run(const WSTR prompt, const std::vector<std::string> &images,
           bool do_sample = false) {
    
    // ========== STEP 1: Processor ==========
    auto processor_output = processor.process(prompt, images);
    
    // processor_output에 포함된 내용:
    // - pixel_values: 이미지 텐서 [BATCH, 3, 896, 896]
    // - input_ids: 텍스트 토큰 아이디 [BATCH, SEQ_LEN]
    // - attention_mask: 어텐션 마스크
    // - token_type_ids: 토큰 타입 (0=text, 1=image)
    
    // ========== STEP 2: Vision Encoder (Optional) ==========
    std::vector<float *> vision_output;
    
    if (!processor_output.pixel_values.empty()) {
      // 이미지가 있으면 Vision Encoder 실행
      std::vector<float *> vision_inputs = {processor_output.pixel_values.data()};
      vision_output = model->incremental_inference(
        BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
    } else {
      // 이미지가 없으면 Vision Encoder 건너뜀기
      // Text Encoder는 text-only로 작동
    }
    
    // ========== STEP 3: Text Encoder ==========
    std::vector<float *> encoder_inputs;
    
    // Text input은 항상 있음
    encoder_inputs.push_back(processor_output.input_ids.data());
    
    // Vision output가 있으면 추가
    if (!vision_output.empty()) {
      encoder_inputs.push_back(vision_output[0]);
    }
    
    auto encoder_output = model->incremental_inference(
      BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
    
    // ========== STEP 4: Decoder Generation ==========
    float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
    decoder_input[0] = BOS_TOKEN_ID;
    
    for (int i = 0; i < NUM_TO_GENERATE; ++i) {
      // Decoder input + Encoder output
      std::vector<float *> decoder_inputs = {decoder_input, encoder_output[0]};
      auto decoder_output = model->incremental_inference(
        BATCH_SIZE, decoder_inputs, {}, 1, i, i+1, true);
      
      // 토큰 샘플링
      unsigned int token = generate(decoder_output[0], do_sample);
      
      // 입력 업데이트
      decoder_input[0] = static_cast<float>(token);
      
      // EOS 체크
      if (token == EOS_TOKEN_ID) break;
    }
    
    free(decoder_input);
  }
};
```

### Vision Encoder Optional 처리의 장점

#### 1. 텍스트 전용 쿼리 (Text-only Query)

```cpp
// 이미지 없이 텍스트 쿼리
run("What is the capital of France?", {}  // images: 빈 벡터

// 내부적으로:
// 1. Processor: 텍스트만 처리 (pixel_values.empty() == true)
// 2. Vision Encoder: 건너뜀기 (!images.empty() == false)
// 3. Text Encoder: 텍스트만 처리
// 4. Decoder: 텍스트만 생성

// 결과: T5와 같은 텍스트 전용 모델과 동일한 방식으로 작동
```

#### 2. 멀티모달 쿼리 (Multimodal Query)

```cpp
// 이미지와 텍스트 쿼리
run("Describe this image.", {"cat.jpg"})

// 내부적으로:
// 1. Processor: 이미지+텍스트 처리
// 2. Vision Encoder: 이미지 인코딩
// 3. Text Encoder: 텍스트+vision_embeddings 처리
// 4. Decoder: 이미지를 참고하여 텍스트 생성
```

#### 3. 메모리 효율

```cpp
// 이미지가 없는 경우
// Vision Encoder에 메모리 할당 안 됨
// KV cache도 생성되지 않음
// → 전체 메모리 사용량 감소
```

### Processor의 역할

Processor는 "유연성 계층(Adaptor)" 역할을 합니다:

```cpp
// T5Gemma2Processor
ProcessorOutput process(const std::string &text, const std::vector<std::string> &images) {
  ProcessorOutput output;
  
  // 1. 텍스트 토큰화
  output.input_ids = tokenizer->Encode(text);
  
  // 2. 이미지가 있으면 처리
  if (!images.empty()) {
    for (const auto &image_path : images) {
      auto pixel_values = loadAndPreprocessImage(image_path);
      output.pixel_values.insert(
        output.pixel_values.end(),
        pixel_values.begin(), pixel_values.end()
      );
    }
    
    // 3. 이미지 토큰 플레이스홀더 추가
    output.input_ids = insertImageTokens(output.input_ids, images.size());
  }
  
  // 4. attention_mask, token_type_ids 생성
  output.attention_mask = createAttentionMask(output.input_ids);
  output.token_type_ids = createTokenTypeIds(output.input_ids, images.size());
  
  return output;
}
```

### 모델 구조 (Vision Encoder Optional)

```cpp
void T5Gemma2Transformer::constructModel() {
  // ========== VISION ENCODER (OPTIONAL) ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  model->addLayer(createLayer("conv2d", {
    withKey("name", "vision_patch_embedding"),
    withKey("input_layers", "vision_input0"),
    withKey("filters", DIM),
    withKey("kernel_size", "14,14"),
    withKey("stride", "14,14")
  }));
  
  // Vision transformer blocks...
  
  model->addLayer(createLayer("rms_norm", {
    withKey("name", "vision_output_norm"),
    withKey("input_layers", "vision_layer11_output"),
    withKey("epsilon", std::to_string(NORM_EPS))
  }));
  
  // ========== TEXT ENCODER ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM)
  }));
  
  // ========== FUSION (CONDITIONAL) ==========
  
  // Vision embeddings와 Text embeddings 병합
  // Vision이 없으면 이 부분은 text_embedding0만 사용
  
  model->addLayer(createLayer("concat", {
    withKey("name", "vision_text_fusion"),
    withKey("input_layers", "text_embedding0,vision_reshape"),  // vision_reshape는 skip될 수 있음
    withKey("axis", "2")
  }));
  
  // ========== ENCODER BLOCKS ==========
  
  // Fused input으로 처리 (text-only 또는 text+vision)
  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
    addEncoderBlock(i);
  }
  
  model->addLayer(createLayer("rms_norm", {
    withKey("name", "encoder_output_norm"),
    withKey("input_layers", "encoder_layer" + std::to_string(NUM_ENCODER_LAYERS-1) + "_output"),
    withKey("epsilon", std::to_string(NORM_EPS))
  }));
  
  // ========== DECODER ==========
  // (이전과 동일)
}
```

### conditional_fusion 레이어 구현

nntrainer에 없는 레이어 타입이므로, 두 가지 방식으로 구현:

#### 방법 1: Concat 레이어와 placeholder

```cpp
void T5Gemma2Transformer::constructModel() {
  // Vision encoder 생략...
  
  // Text encoder
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM)
  }));
  
  // Conditional fusion: 항상 concat (vision input은 dummy tensor)
  // Vision이 없으면 dummy_vision을 0으로 채워서 전달
  model->addLayer(createLayer("concat", {
    withKey("name", "vision_text_fusion"),
    withKey("input_layers", "text_embedding0,dummy_vision"),
    withKey("axis", "2")
  }));
}
```

**추론 시:**
```cpp
// 이미지 있는 경우
inputs = {text_input, vision_input};

// 이미지 없는 경우
std::vector<float> dummy_vision(NUM_PATCHES * DIM, 0.0f);
inputs = {text_input, dummy_vision.data()};
```

#### 방법 2: 두 개의 경로 (추천)

```cpp
void T5Gemma2Transformer::constructModel() {
  // ========== TEXT-ONLY PATH ==========
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
  }));
  
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM)
  }));
  
  // ========== MULTIMODAL PATH ==========
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  // Vision encoder...
  
  model->addLayer(createLayer("concat", {
    withKey("name", "vision_text_fusion"),
    withKey("input_layers", "text_embedding0,vision_output_norm"),
    withKey("axis", "2")
  }));
  
  // Encoder blocks (fused input으로 처리)
}
```

**추론 시:**
```cpp
// 이미지 있는 경우: multimodal path 사용
inputs = {text_input, vision_input};
auto encoder_output = model->incremental_inference(BATCH_SIZE, inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);

// 이미지 없는 경우: text-only path 사용
inputs = {text_input};
auto encoder_output = model->incremental_inference(BATCH_SIZE, inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
```

### 실제 T5Gemma2 추론 플로우

```cpp
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,
    bool do_sample) {
  
  // ========== STEP 1: Processor ==========
  auto output = processor.process(prompt, images);
  
  bool has_vision = !output.pixel_values.empty();
  
  // ========== STEP 2: Vision Encoder (Optional) ==========
  std::vector<float *> vision_output;
  
  if (has_vision) {
    std::vector<float *> vision_inputs = {output.pixel_values.data()};
    vision_output = model->incremental_inference(
      BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
  }
  
  // ========== STEP 3: Text Encoder ==========
  std::vector<float *> encoder_inputs = {output.input_ids.data()};
  
  if (has_vision) {
    encoder_inputs.push_back(vision_output[0]);
  }
  
  auto encoder_output = model->incremental_inference(
    BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  
  // ========== STEP 4: Decoder Generation ==========
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  
  for (int i = 0; i < NUM_TO_GENERATE; ++i) {
    std::vector<float *> decoder_inputs = {decoder_input, encoder_output[0]};
    auto decoder_output = model->incremental_inference(
      BATCH_SIZE, decoder_inputs, {}, 1, i, i+1, true);
    
    unsigned int token = generate(decoder_output[0], do_sample);
    decoder_input[0] = static_cast<float>(token);
    
    if (token == EOS_TOKEN_ID) break;
  }
  
  free(decoder_input);
}
```

### 이 구조의 장점 요약

1. **유연성:** 이미지 있음/없음 유연하게 처리
2. **효율성:** 불필요한 연산 건너뜀기
3. **유지보수성:** T5와 유사한 구조 유지
4. **모듈성:** 각 단계 독립적으로 테스트 가능
5. **사용자 경험:** 단순한 인터페이스 (이미지 있으면 전달, 없으면 안 전달)

이 2단계 구조는 T5Gemma2와 같은 멀티모달 모델에서 가장 일반적이고 효율적인 접근 방식입니다!
