# Model의 서로 다른 Layer에 서로 다른 Input 주기

## 질문

"model의 서로 다른 layer에 서로 다른 input을 줄 수 있어?"

## 답변

**네, 가능합니다!** NNTrainer는 한 모델 내의 여러 Input Layer에 서로 다른 input을 전달할 수 있습니다.

## 방법: 여러 Input Layer 사용

### 구조

```
ModelHandle:
├── Input Layer 1 (vision_input0) → [Vision Encoder] → [vision_output]
│
├── Input Layer 2 (text_input0) → [Text Encoder] → [text_output]
│
└── Input Layer 3 (decoder_input0) → [Decoder] → [final_output]
```

### 구현 예시

```cpp
void T5Gemma2Transformer::constructModel() {
  // ========== Input Layer 1: Vision ==========
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")  // BATCH, Channels, Height, Width
  }));
  
  // Vision Encoder 레이어들...
  model->addLayer(createLayer("conv2d", {
    withKey("name", "vision_patch_embedding"),
    withKey("input_layers", "vision_input0"),  // vision_input0에서 입력받음
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
  
  // ========== Input Layer 2: Text ==========
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))  // BATCH, 1, SEQ_LEN
  }));
  
  // Text Encoder 레이어들...
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),  // text_input0에서 입력받음
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM)
  }));
  
  // ========== Input Layer 3: Decoder ==========
  model->addLayer(createLayer("input", {
    withKey("name", "decoder_input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))  // BATCH, 1, SEQ_LEN
  }));
  
  // Decoder 레이어들...
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "decoder_embedding0"),
    withKey("input_layers", "decoder_input0"),  // decoder_input0에서 입력받음
    withKey("in_dim", NUM_VOCAB),
    withKey("out_dim", DIM),
    withKey("shared_from", "text_embedding0")
  }));
}
```

### 추론 시 여러 Input 전달

```cpp
void T5Gemma2ForConditionalGeneration::run(...) {
  // ========== Input 1: Vision ==========
  float *vision_input = (float *)malloc(BATCH_SIZE * 3 * 896 * 896 * sizeof(float));
  // vision_input에 이미지 데이터 채우기...
  
  // ========== Input 2: Text ==========
  float *text_input = (float *)malloc(BATCH_SIZE * INIT_SEQ_LEN * sizeof(float));
  for (int i = 0; i < TEXT_SEQ_LEN; ++i) {
    text_input[i] = static_cast<float>(text_tokens[i]);
  }
  
  // ========== Input 3: Decoder ==========
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  
  // ========== 여러 Input 전달 ==========
  std::vector<float *> inputs = {
    vision_input,  // → vision_input0
    text_input,     // → text_input0
    decoder_input    // → decoder_input0
  };
  
  // 한 번의 inference로 모든 input 처리
  auto output = model->incremental_inference(
    BATCH_SIZE, inputs, {}, 
    seq_len, 0, seq_len, false);
}
```

## Input Layer와 Input의 매핑

**중요:** `incremental_inference(input)`에 전달하는 input의 순서와 `constructModel()`에서 추가한 Input Layer의 순서가 일치해야 합니다!

```cpp
// constructModel()에서 Input Layer 추가 순서:
model->addLayer(createLayer("input", {"name=vision_input0", ...}));  // 1번째
model->addLayer(createLayer("input", {"name=text_input0", ...}));     // 2번째
model->addLayer(createLayer("input", {"name=decoder_input0", ...}));  // 3번째

// inference 시 input 전달 순서:
std::vector<float *> inputs = {
  vision_input,  // → vision_input0 (1번째)
  text_input,     // → text_input0 (2번째)
  decoder_input    // → decoder_input0 (3번째)
};
```

## 단계별 Input 전달

### 방법 1: 모든 Input을 한 번에 전달

```cpp
// 모든 input을 한 번에 전달
std::vector<float *> inputs = {
  vision_input,
  text_input,
  decoder_input
};
auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
```

### 방법 2: 특정 단계에서만 일부 Input 전달

```cpp
// Vision encoder만 실행
std::vector<float *> vision_inputs = {vision_input};
auto vision_output = model->incremental_inference(
  BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);

// Text encoder만 실행 (vision output는 concat 레이어에서 처리)
std::vector<float *> text_inputs = {text_input};
auto text_output = model->incremental_inference(
  BATCH_SIZE, text_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);

// Decoder만 실행 (encoder output는 cross-attention에서 사용)
std::vector<float *> decoder_inputs = {decoder_input, encoder_output[0]};
auto decoder_output = model->incremental_inference(
  BATCH_SIZE, decoder_inputs, {}, 1, pos, pos+1, true);
```

## 실제 사용 예시: T5Gemma2

### 전체 추론 플로우

```cpp
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,
    bool do_sample) {
  
  // ========== Processor ==========
  auto processor_output = processor.process(prompt, images);
  
  bool has_vision = !processor_output.pixel_values.empty();
  
  // ========== Vision Encoder (Optional) ==========
  std::vector<float *> vision_output;
  
  if (has_vision) {
    // Vision input만 전달
    std::vector<float *> vision_inputs = {processor_output.pixel_values.data()};
    vision_output = model->incremental_inference(
      BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
  }
  
  // ========== Text Encoder ==========
  std::vector<float *> encoder_inputs;
  encoder_inputs.push_back(processor_output.input_ids.data());
  
  if (has_vision) {
    encoder_inputs.push_back(vision_output[0]);
  }
  
  auto encoder_output = model->incremental_inference(
    BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  
  // ========== Decoder Generation ==========
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  
  for (int i = 0; i < NUM_TO_GENERATE; ++i) {
    // Decoder input + Encoder output
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

## Input Layer의 특징

### 1. 독립적인 입력 Shape

각 Input Layer는 자신만의 `input_shape`을 가집니다:

```cpp
// Vision input: [BATCH, 3, 896, 896]
model->addLayer(createLayer("input", {
  withKey("name", "vision_input0"),
  withKey("input_shape", "1:3:896:896")
}));

// Text input: [BATCH, 1, SEQ_LEN]
model->addLayer(createLayer("input", {
  withKey("name", "text_input0"),
  withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
}));

// Decoder input: [BATCH, 1, SEQ_LEN]
model->addLayer(createLayer("input", {
  withKey("name", "decoder_input0"),
  withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN)
}));
```

### 2. 레이어 연결

`input_layers` 파라미터를 사용하여 레이어를 연결합니다:

```cpp
// vision_input0 → vision_patch_embedding
model->addLayer(createLayer("conv2d", {
  withKey("name", "vision_patch_embedding"),
  withKey("input_layers", "vision_input0"),  // vision_input0에서 입력받음
  ...
}));

// text_input0 → text_embedding0
model->addLayer(createLayer("embedding_layer", {
  withKey("name", "text_embedding0"),
  withKey("input_layers", "text_input0"),  // text_input0에서 입력받음
  ...
}));
```

### 3. Fusion (다른 Input 결합)

Concat 레이어로 여러 Input을 결합할 수 있습니다:

```cpp
// text_embedding0 + vision_reshape → vision_text_fusion
model->addLayer(createLayer("concat", {
  withKey("name", "vision_text_fusion"),
  withKey("input_layers", "text_embedding0,vision_reshape"),  // 두 개의 입력 결합
  withKey("axis", "2")
}));
```

## 조건부 Input (Optional Input)

### 시나리오: 이미지가 있을 때만 Vision Encoder 실행

```cpp
void run(const WSTR prompt, const std::vector<std::string> &images) {
  auto processor_output = processor.process(prompt, images);
  
  bool has_vision = !processor_output.pixel_values.empty();
  
  // Vision input만 줄지 말지 결정
  std::vector<float *> encoder_inputs;
  encoder_inputs.push_back(processor_output.input_ids.data());
  
  if (has_vision) {
    // Vision Encoder 실행
    std::vector<float *> vision_inputs = {processor_output.pixel_values.data()};
    auto vision_output = model->incremental_inference(
      BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
    
    // Vision output를 encoder_inputs에 추가
    encoder_inputs.push_back(vision_output[0]);
  }
  
  // Text Encoder 실행 (vision 있으면 함께 처리, 없으면 text만)
  auto encoder_output = model->incremental_inference(
    BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
}
```

## 요약

### 질문: 서로 다른 layer에 서로 다른 input을 줄 수 있어?

**답변: 네, 가능합니다!**

### 방법

1. **여러 Input Layer 생성:** `constructModel()`에서 여러 Input Layer 추가
2. **input_layers로 연결:** 각 레이어에서 `input_layers` 파라미터로 입력 레이어 지정
3. **incremental_inference에 여러 input 전달:** `std::vector<float *>`로 여러 input 전달

### 장점

1. **유연성:** 서로 다른 타입의 input (이미지, 텍스트 등)을 동시에 처리
2. **효율성:** 필요한 input만 전달하여 불필요한 연산 건너뜀기
3. **모듈성:** 각 input path를 독립적으로 설계

### 실제 예시: T5Gemma2

```
Input Layer 1 (vision_input0) → [Vision Encoder] → [vision_output]
                                                  ↓
Input Layer 2 (text_input0) → [Text Encoder] → [text_output] → [Fusion] → [Encoder]
                                                  ↓
Input Layer 3 (decoder_input0) → [Decoder (cross-attn)] → [Output]
```

이것이 NNTrainer에서 서로 다른 layer에 서로 다른 input을 주는 방법입니다!
