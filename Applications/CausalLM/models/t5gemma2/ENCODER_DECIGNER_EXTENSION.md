# CausalLM을 Encoder-Decoder 구조로 확장

## 개요

현재 CausalLM은 Decoder-only 구조입니다. Encoder-Decoder 구조(예: T5, BART)를 지원하도록 확장하는 방법을 설명합니다.

## 현재 CausalLM 구조 (Decoder-only)

```
[Input Tokens]
    ↓
[Embedding Layer]
    ↓
[Transformer Decoder Blocks] × NUM_LAYERS
    ├── Self-Attention (causal mask)
    ├── Addition
    ├── RMSNorm
    ├── Feed Forward
    └── Addition
    ↓
[RMSNorm]
    ↓
[LM Head]
    ↓
[Output Logits]
```

## Encoder-Decoder 구조

```
ENCODER SIDE:
[Input Tokens]
    ↓
[Embedding Layer]
    ↓
[Transformer Encoder Blocks] × NUM_ENCODER_LAYERS
    ├── Self-Attention (bidirectional mask)
    ├── Addition
    ├── RMSNorm
    ├── Feed Forward
    └── Addition
    ↓
[Encoder Hidden States] (BATCH, SEQ_LEN, HIDDEN_DIM)
         ↓ (cross-attention via)
         ↓
DECODER SIDE:
[Output Tokens (shifted right)]
    ↓
[Embedding Layer]
    ↓
[Transformer Decoder Blocks] × NUM_DECODER_LAYERS
    ├── Self-Attention (causal mask)
    ├── Addition
    ├── RMSNorm
    ├── Cross-Attention (encoder output as K, V)  ← 새로 추가
    │   ├── Q: Decoder hidden states
    │   ├── K: Encoder hidden states
    │   └── V: Encoder hidden states
    ├── Addition
    ├── RMSNorm
    ├── Feed Forward
    └── Addition
    ↓
[RMSNorm]
    ↓
[LM Head]
    ↓
[Output Logits]
```

## 확장 방법

### 방법 1: 새로운 기본 클래스 생성 (권장)

```
Transformer (base class)
    ↓
    ├── CausalLM (Decoder-only) - 기존
    └── EncoderDecoder (Encoder-Decoder) - 새로 추가
            ├── T5 (Encoder-Decoder)
            └── T5Gemma2 (Encoder-Decoder with vision)
```

### 방법 2: CausalLM을 수정하여 Encoder-Decoder 지원 (비권장)

- 기존 코드와 호환성 문제
- 복잡도 증가

## 구현 계획: 방법 1 (새로운 기본 클래스)

### 1. EncoderDecoder 기본 클래스 생성

**파일:** `Applications/CausalLM/models/encoder_decoder.h`

```cpp
#ifndef __ENCODER_DECODER_H__
#define __ENCODER_DECODER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief EncoderDecoder Class
 * Base class for Encoder-Decoder models (T5, BART, etc.)
 */
class EncoderDecoder : virtual public Transformer {
public:
  EncoderDecoder(json &cfg, json &generation_cfg, json &nntr_cfg)
    : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }
  
  virtual ~EncoderDecoder() = default;
  
protected:
  // Encoder parameters
  int NUM_ENCODER_LAYERS;
  int NUM_DECODER_LAYERS;
  
  void setupParameters(json &cfg, json &generation_cfg,
                      json &nntr_cfg) override;
  
  void constructModel() override;
  
  void registerCustomLayers() override;
  
  // Create encoder block
  virtual std::vector<LayerHandle>
  createTransformerEncoderBlock(const int layer_id, std::string input_name);
  
  // Create decoder block with cross-attention
  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name,
                                std::string encoder_output_name);
};

} // namespace causallm

#endif
```

### 2. EncoderDecoder 구현

**파일:** `Applications/CausalLM/models/encoder_decoder.cpp`

```cpp
#include "encoder_decoder.h"

void EncoderDecoder::setupParameters(json &cfg, json &generation_cfg,
                                    json &nntr_cfg) {
  // 기본 Transformer 파라미터 설정
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  
  // Encoder-Decoder 전용 파라미터
  NUM_ENCODER_LAYERS = cfg.contains("num_encoder_layers")
                        ? cfg["num_encoder_layers"].get<int>()
                        : NUM_LAYERS;
  NUM_DECODER_LAYERS = cfg.contains("num_decoder_layers")
                        ? cfg["num_decoder_layers"].get<int>()
                        : NUM_LAYERS;
}

void EncoderDecoder::constructModel() {
  std::vector<LayerHandle> layers;
  
  // ========== ENCODER ==========
  
  // Encoder input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "encoder_input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  
  // Encoder embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  layers.push_back(createLayer(
    embedding_type,
    {"name=encoder_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  
  // Encoder transformer blocks
  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
    std::vector<LayerHandle> encoder_block;
    if (i == 0)
      encoder_block = createTransformerEncoderBlock(0, "encoder_embedding0");
    else
      encoder_block = createTransformerEncoderBlock(
        i, "encoder_layer" + std::to_string(i - 1) + "_encoder_output");
    layers.insert(layers.end(), encoder_block.begin(), encoder_block.end());
  }
  
  // Encoder output normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "encoder_output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "encoder_layer" + std::to_string(NUM_ENCODER_LAYERS - 1) +
               "_encoder_output"),
     withKey("packed", "false")}));
  
  // ========== DECODER ==========
  
  // Decoder input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "decoder_input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  
  // Decoder embedding layer (shared with encoder)
  layers.push_back(createLayer(
    embedding_type,
    {"name=decoder_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  
  // Decoder transformer blocks with cross-attention
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    std::vector<LayerHandle> decoder_block;
    if (i == 0)
      decoder_block = createTransformerDecoderBlock(
        0, "decoder_embedding0", "encoder_output_norm");
    else
      decoder_block = createTransformerDecoderBlock(
        i, "decoder_layer" + std::to_string(i - 1) + "_decoder_output",
        "encoder_output_norm");
    layers.insert(layers.end(), decoder_block.begin(), decoder_block.end());
  }
  
  // Decoder output normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "decoder_output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "decoder_layer" + std::to_string(NUM_DECODER_LAYERS - 1) +
               "_decoder_output"),
     withKey("packed", "false")}));
  
  // LM Head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_encoder_decoder"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "decoder_output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "encoder_embedding0"));
  
  layers.push_back(createLayer(lmhead_type, lmhead_prop));
  
  // Add all layers to model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle>
EncoderDecoder::createTransformerEncoderBlock(const int layer_id,
                                              std::string input_name) {
  std::vector<LayerHandle> layers;
  
  // RMSNorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "encoder_layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // Self-Attention (bidirectional - no causal mask)
  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "encoder_layer" + std::to_string(layer_id) + "_attention_norm",
                    "encoder_layer" + std::to_string(layer_id) + "_attention_norm",
                    "encoder_layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());
  
  // Addition
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "encoder_layer" + std::to_string(layer_id) + "_encoder_add"),
     withKey("input_layers", input_name + ",encoder_layer" +
                               std::to_string(layer_id) + "_attention_out")}));
  
  // RMSNorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "encoder_layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "encoder_layer" + std::to_string(layer_id) + "_encoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // Feed Forward
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "encoder_layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());
  
  // Addition
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "encoder_layer" + std::to_string(layer_id) + "_encoder_output"),
     withKey("input_layers", "encoder_layer" + std::to_string(layer_id) +
                               "_encoder_add,encoder_layer" + std::to_string(layer_id) +
                               "_ffn_down")}));
  
  return layers;
}

std::vector<LayerHandle>
EncoderDecoder::createTransformerDecoderBlock(const int layer_id,
                                              std::string input_name,
                                              std::string encoder_output_name) {
  std::vector<LayerHandle> layers;
  
  // Self-Attention RMSNorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_self_attn_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // Self-Attention (causal mask)
  auto self_att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "decoder_layer" + std::to_string(layer_id) + "_self_attn_norm",
                    "decoder_layer" + std::to_string(layer_id) + "_self_attn_norm",
                    "decoder_layer" + std::to_string(layer_id) + "_self_attn_norm");
  layers.insert(layers.end(), self_att_layer.begin(), self_att_layer.end());
  
  // Self-Attention Addition
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_self_attn_add"),
     withKey("input_layers", input_name + ",decoder_layer" +
                               std::to_string(layer_id) + "_self_attn_out")}));
  
  // Cross-Attention RMSNorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_cross_attn_norm"),
     withKey("input_layers",
             "decoder_layer" + std::to_string(layer_id) + "_self_attn_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  
  // ========== Cross-Attention (새로 추가) ==========
  // Q: Decoder hidden states
  // K: Encoder hidden states
  // V: Encoder hidden states
  
  auto Q = "decoder_layer" + std::to_string(layer_id) + "_wq";
  auto K = "decoder_layer" + std::to_string(layer_id) + "_wk_cross";
  auto V = "decoder_layer" + std::to_string(layer_id) + "_wv_cross";
  auto A = "decoder_layer" + std::to_string(layer_id) + "_cross_attention";
  auto O = "decoder_layer" + std::to_string(layer_id) + "_cross_attention_out";
  
  // Q projection (from decoder)
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", "decoder_layer" + std::to_string(layer_id) + "_cross_attn_norm"),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));
  
  // K projection (from encoder output)
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", encoder_output_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));
  
  // V projection (from encoder output)
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", encoder_output_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));
  
  // Cross-Attention core
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("is_causal", "false"),  // Cross-attention은 bidirectional
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));
  
  // Cross-Attention output projection
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));
  
  // Cross-Attention Addition
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_cross_attn_add"),
     withKey("input_layers", "decoder_layer" + std::to_string(layer_id) +
                               "_self_attn_add,decoder_layer" + std::to_string(layer_id) +
                               "_cross_attention_out")}));
  
  // Feed Forward RMSNorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_ffn_norm"),
    withKey("input_layers",
            "decoder_layer" + std::to_string(layer_id) + "_cross_attn_add"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("packed", "false")}));
  
  // Feed Forward
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "decoder_layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());
  
  // Feed Forward Addition
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "decoder_layer" + std::to_string(layer_id) + "_decoder_output"),
    withKey("input_layers", "decoder_layer" + std::to_string(layer_id) +
                               "_cross_attn_add,decoder_layer" + std::to_string(layer_id) +
                               "_ffn_down")}));
  
  return layers;
}

void EncoderDecoder::registerCustomLayers() {
  Transformer::registerCustomLayers();
  // Cross-attention은 기존 MHA 레이어로 처리 가능
}

```

### 3. T5 구현

**파일:** `Applications/CausalLM/models/t5/t5.h`

```cpp
#ifndef __T5_H__
#define __T5_H__

#include <encoder_decoder.h>

namespace causallm {

class T5 : virtual public EncoderDecoder {
public:
  static constexpr const char *architectures = "T5ForConditionalGeneration";
  
  T5(json &cfg, json &generation_cfg, json &nntr_cfg)
    : Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg),
      EncoderDecoder(cfg, generation_cfg, nntr_cfg) {
    // T5 특화 파라미터
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  }
  
  virtual ~T5() = default;
  
protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);
};

class T5ForConditionalGeneration : public T5 {
public:
  T5ForConditionalGeneration(json &cfg, json &generation_cfg, json &nntr_cfg)
    : T5(cfg, generation_cfg, nntr_cfg) {}
  
  void setupParameters(...) override {
    T5::setupParameters(cfg, generation_cfg, nntr_cfg);
    EncoderDecoder::setupParameters(cfg, generation_cfg, nntr_cfg);
  }
};

} // namespace causallm

#endif
```

### 4. T5Gemma2 구현

**파일:** `Applications/CausalLM/models/t5gemma2/t5gemma2_causallm.h`

```cpp
#ifndef __T5GEMMA2_CAUSAL_LM_H__
#define __T5GEMMA2_CAUSAL_LM_H__

#include <encoder_decoder.h>
#include <gemma3/gemma3_causallm.h>

namespace causallm {

class T5Gemma2Transformer : virtual public EncoderDecoder {
public:
  static constexpr const char *architectures = "T5Gemma2Transformer";
  
  T5Gemma2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg)
    : EncoderDecoder(cfg, generation_cfg, nntr_cfg) {
    // Vision encoder 추가
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  }
  
  virtual ~T5Gemma2Transformer() = default;
  
  void constructModel() override {
    // Vision encoder 추가
    EncoderDecoder::constructModel();
    // ... vision encoder 레이어 추가
  }
  
protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);
};

class T5Gemma2ForConditionalGeneration : public T5Gemma2Transformer {
public:
  static constexpr const char *architectures = "T5Gemma2ForConditionalGeneration";
  
  T5Gemma2ForConditionalGeneration(json &cfg, json &generation_cfg, json &nntr_cfg)
    : T5Gemma2Transformer(cfg, generation_cfg, nntr_cfg) {}
  
  void setupParameters(...) override {
    T5Gemma2Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }
  
  void run(const WSTR prompt, const std::vector<std::string> &images,
           bool do_sample = false, const WSTR system_prompt = "",
           const WSTR tail_prompt = "") override {
    // T5Gemma2Processor 사용해서 이미지+텍스트 처리
  }
};

} // namespace causallm

#endif
```

### 5. main.cpp에 등록

```cpp
// main.cpp
causallm::Factory::Instance().registerModel(
  "T5ForConditionalGeneration",
  [](json cfg, json generation_cfg, json nntr_cfg) {
    return std::make_unique<causallm::T5ForConditionalGeneration>(
      cfg, generation_cfg, nntr_cfg);
  }
);

causallm::Factory::Instance().registerModel(
  "T5Gemma2ForConditionalGeneration",
  [](json cfg, json generation_cfg, json nntr_cfg) {
    return std::make_unique<causallm::T5Gemma2ForConditionalGeneration>(
      cfg, generation_cfg, nntr_cfg);
  }
);
```

## 추론 흐름

### 1. Encoder Prefill

```cpp
// Encoder 전체 입력 처리
encoder_output = model->incremental_inference(
  BATCH_SIZE, encoder_input, label, encoder_seq_len,
  0, encoder_seq_len, false);
```

### 2. Decoder Prefill

```cpp
// Decoder 입력 처리 (encoder_output 활용)
decoder_output = model->incremental_inference(
  BATCH_SIZE, decoder_input, label, decoder_seq_len,
  0, decoder_seq_len, false,
  encoder_output);  // encoder_output을 cross-attention에 전달
```

### 3. Decoder Generation

```cpp
// 한 토큰씩 생성
for (int i = 0; i < NUM_TO_GENERATE; ++i) {
  decoder_output = model->incremental_inference(
    BATCH_SIZE, decoder_input, label, 1,
    pos, pos+1, true,
    encoder_output);  // encoder_output 계속 활용
  
  // 토큰 샘플링
  token = sample(decoder_output);
}
```

## config.json 설정

```json
{
  "hidden_size": 3072,
  "num_hidden_layers": 24,
  "num_encoder_layers": 12,
  "num_decoder_layers": 12,
  "num_attention_heads": 24,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "intermediate_size": 24576,
  "max_position_embeddings": 4096,
  "rope_theta": 10000,
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": true
}
```

## 주요 변경 사항 요약

### 1. 새로운 클래스 추가
- `EncoderDecoder` 기본 클래스
- `createTransformerEncoderBlock()` 메서드
- `createTransformerDecoderBlock()` 메서드 (cross-attention 포함)

### 2. Cross-Attention 추가
- Q: Decoder hidden states
- K: Encoder hidden states
- V: Encoder hidden states
- `is_causal`: false (bidirectional)

### 3. 두 개의 입력
- Encoder input
- Decoder input (shifted right)

### 4. 파라미터 분리
- `NUM_ENCODER_LAYERS`
- `NUM_DECODER_LAYERS`

## 참고 사항

### 공유 임베딩
- Encoder와 Decoder가 임베딩 레이어 공유 가능
- `TIE_WORD_EMBEDDINGS = true`

### KV Cache
- Encoder KV cache: 생성 과정에서 재사용
- Decoder KV cache: 디코딩 과정에서 재사용

### Vision Encoder (T5Gemma2)
- 이미지를 encoder input에 통합
- Vision encoder → image embeddings → transformer encoder
- T5Gemma2Processor 사용

## T5Gemma2 특이사항

### 구조
```
[Input Images] → [Vision Encoder] → [Image Embeddings]
                                                    ↓
[Input Text] → [Text Encoder] → [Text Embeddings] → [T5 Encoder]
                                                    ↓
                                              [Encoder Hidden States]
                                                    ↓
[Output Tokens] → [T5 Decoder (with cross-attention)] → [Output]
```

### Processor 통합
- T5Gemma2Processor에서 이미지와 텍스트 처리
- Pixel values를 vision encoder에 전달
- Input tokens를 text encoder에 전달
- Encoder output을 decoder의 cross-attention에 전달

## 결론

Encoder-Decoder 구조 지원을 위해:

1. **새로운 기본 클래스 생성** (권장)
   - `EncoderDecoder` 클래스
   - CausalLM과 독립적
   - 호환성 유지

2. **주요 추가 사항**
   - Encoder 블록 (bidirectional attention)
   - Decoder 블록 (cross-attention 추가)
   - 두 개의 입력 처리

3. **T5Gemma2 구현**
   - `EncoderDecoder` 상속
   - Vision encoder 추가
   - T5Gemma2Processor 통합

이 방식은 기존 CausalLM 코드에 영향을 주지 않으면서, Encoder-Decoder 구조를 깔끔하게 지원할 수 있습니다.
