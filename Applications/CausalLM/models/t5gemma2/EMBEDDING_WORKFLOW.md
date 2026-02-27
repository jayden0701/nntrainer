# Embedding 모델 작동 원리

## 개요

`embedding.cpp`는 Sentence Transformers 스타일의 embedding 모델을 구현합니다. 텍스트를 고정된 차원의 embedding vector로 변환하는 인코더 모델입니다.

## 클래스 상속 구조

```
Transformer (base class)
    ↓ (virtual 상속)
Embedding (ModelType::EMBEDDING)
    ├── encode() - 텍스트를 embedding vector로 변환
    ├── run() - embedding 결과 출력
    ├── constructModel() - Transformer + 추가 모듈 구성
    ├── setupParameters() - modules.json 로드
    └── addModule() - 동적 모듈 추가
```

## 주요 차이점: CausalLM vs Embedding

| 특징 | CausalLM | Embedding |
|------|----------|-----------|
| **목적** | 텍스트 생성 (Generation) | 텍스트 임베딩 (Encoding) |
| **출력** | 토큰 시퀀스 | 고정 차원 벡터 (DIM) |
| **추론 방식** | Incremental inference (한 토큰씩) | Single forward pass (전체 한 번에) |
| **LM Head** | 있음 (vocab 크기) | 없음 |
| **추가 레이어** | 없음 | Pooling, Normalize, Dense 등 |
| **사용 사례** | 챗봇, 번역, 요약 | 의미 검색, 유사도 계산, 클러스터링 |

## 작동 흐름

### 1. 초기화 (setupParameters)

```cpp
void Embedding::setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) {
  // 1. 기본 Transformer 파라미터 설정
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  
  // 2. modules.json 로드
  std::string modules_config_path = nntr_cfg["module_config_path"];
  json modules_json = LoadJsonFile(modules_config_path);
  modules = modules_json.get<std::vector<json>>();
  
  // 3. 각 모듈의 config.json 로드
  for (auto &module : modules) {
    if (module.contains("path")) {
      std::filesystem::path module_dir = base_dir / module["path"];
      json module_config = LoadJsonFile(module_dir / "config.json");
      
      int idx = module["idx"];
      module_configs[idx] = module_config;
    }
  }
}
```

**modules.json 예시:**
```json
[
  {
    "type": "sentence_transformers.models.Transformer",
    "idx": 0
  },
  {
    "type": "sentence_transformers.models.Pooling",
    "idx": 1,
    "path": "1_Pooling"
  },
  {
    "type": "sentence_transformers.models.Normalize",
    "idx": 2,
    "path": "2_Normalize"
  }
]
```

### 2. 모델 구성 (constructModel)

```cpp
void Embedding::constructModel() {
  for (auto &module : modules) {
    std::string type = module["type"];
    std::string component = getLastComponent(type);  // "Pooling", "Normalize" 등
    
    if (component == "Transformer") {
      // Transformer 기본 구조 (embedding + decoder blocks)
      Transformer::constructModel();
    } else {
      // 추가 모듈 (Pooling, Normalize, Dense 등)
      int idx = module["idx"];
      addModule(type, idx);
    }
  }
}
```

**최종 모델 구조:**
```
[Input Tokens]
    ↓
[Embedding Layer]
    ↓
[Transformer Blocks] × NUM_LAYERS
    ↓
[Output RMSNorm]
    ↓
[Pooling Layer]        ← Embedding 전용
    ↓
[Normalize Layer]      ← Embedding 전용
    ↓
[Output Embedding Vector] (DIM 차원)
```

### 3. 모듈 추가 (addModule)

```cpp
void Embedding::addModule(const std::string &type, int idx) {
  // 1. 모듈 config 로드
  json config = module_configs[idx];
  
  // 2. 타입 매핑
  // "Pooling" → "embedding_pooling"
  // "Normalize" → "embedding_normalize"
  // "Dense" → "fully_connected"
  std::string component = getLastComponent(type);
  std::string layer_name = layer_map[component];
  
  // 3. JSON config를 nntrainer 속성으로 변환
  std::vector<std::string> props;
  for (auto &el : config.items()) {
    std::string key = el.key();
    std::string val = el.value().dump();
    
    if (key == "out_features") {
      props.push_back("unit=" + val);
    } else if (key == "bias" && val == "false") {
      props.push_back("disable_bias=true");
    } else {
      props.push_back(key + "=" + val);
    }
  }
  
  // 4. 레이어 생성 및 추가
  LayerHandle layer = ml::train::createLayer(layer_name, props);
  model->addLayer(layer);
}
```

**Pooling Layer 예시 (config.json):**
```json
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false
}
```

**변환된 nntrainer 속성:**
```
word_embedding_dimension=768
pooling_mode_cls_token=false
pooling_mode_mean_tokens=true
pooling_mode_max_tokens=false
```

### 4. 커스텀 레이어 등록

```cpp
void Embedding::registerCustomLayers() {
  // 기본 Transformer 레이어 등록
  Transformer::registerCustomLayers();
  
  // Embedding 전용 레이어 등록
  app_context->registerFactory(
    nntrainer::createLayer<causallm::EmbeddingPoolingLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::EmbeddingNormalizeLayer>);
}
```

### 5. 텍스트 인코딩 (encode)

```cpp
std::vector<float *> Embedding::encode(const WSTR prompt,
                                       const WSTR system_prompt,
                                       const WSTR tail_prompt) {
  // 1. 프롬프트 결합
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  
  // 2. 토큰화
  auto _input = tokenizer->Encode(prompt_, true);
  
  // 3. 입력 텐서 준비
  unsigned int input_len = std::min(_input.size(), MAX_SEQ_LEN);
  float *input_sample = malloc(BATCH_SIZE * MAX_SEQ_LEN * sizeof(float));
  
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[b * MAX_SEQ_LEN + i] = static_cast<float>(_input[i]);
    }
  }
  
  // 4. Single forward pass (전체 시퀀스 한 번에 처리)
  // start: 0, end: input_len, is_decoding: false
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, input_len, 0, input_len, false);
  
  free(input_sample);
  return output;
}
```

**중요:** CausalLM과 달리 **한 번의 forward pass**로 전체 시퀀스를 처리합니다.

### 6. 실행 (run)

```cpp
void Embedding::run(const WSTR prompt, bool do_sample,
                    const WSTR system_prompt, const WSTR tail_prompt) {
  // 1. 텍스트 인코딩
  std::vector<float *> results = encode(prompt, system_prompt, tail_prompt);
  
  // 2. 결과 출력
  std::cout << "Embedding Result (" << BATCH_SIZE << " batch(es)):" << std::endl;
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    std::cout << "Batch " << b << ": [";
    
    // 처음 10개 요소만 샘플로 출력
    int print_dim = (DIM > 10) ? 10 : DIM;
    for (int i = 0; i < print_dim; ++i) {
      std::cout << results[0][b * DIM + i];
      if (i < print_dim - 1) std::cout << ", ";
    }
    
    if (DIM > 10) std::cout << ", ...";
    std::cout << "] (Total DIM: " << DIM << ")" << std::endl;
  }
}
```

## 주요 레이어 설명

### 1. Pooling Layer

**목적:** 시퀀스의 다양한 토큰 임베딩을 하나의 벡터로 압축

**Pool 전략:**
- **CLS Token Pooling:** `[CLS]` 토큰의 임베딩만 사용
- **Mean Pooling:** 모든 토큰 임베딩의 평균
- **Max Pooling:** 각 차원의 최댓값

**예시 (Mean Pooling):**
```
입력: [CLS] The cat sat on the mat [SEP]
임베딩: [768차원] × 8개 토큰

Mean Pooling: (768차원 벡터 8개의 평균) → [768차원]
```

### 2. Normalize Layer

**목적:** 임베딩 벡터를 단위 길이로 정규화

**수학적 표현:**
```
normalized_embedding = embedding / ||embedding||
```

**이유:**
- 코사인 유사도 계산 용이
- 서로 다른 벡터 길이 효과 제거
- 안정적인 유사도 비교

### 3. Dense Layer

**목적:** 차원 축소 또는 투영

**예시:**
```
입력: 768차원 → Dense(256) → 출력: 256차원
```

## 전체 파이프라인 예시

```
입력: "The cat sat on the mat"
    ↓
Tokenizer: [2023, 389, 4231, 418, 277, 389, 1620] (7개 토큰)
    ↓
Embedding Layer: [768차원] × 7개
    ↓
Transformer Blocks (12 layers): [768차원] × 7개
    ↓
RMSNorm: [768차원] × 7개
    ↓
Pooling Layer (Mean): [768차원] × 1개
    ↓
Normalize Layer: [768차원] (단위 길이)
    ↓
출력: [0.0123, -0.0456, 0.0789, ..., -0.0234] (768차원 벡터)
```

## 사용 사례

### 1. 의미 검색 (Semantic Search)

```cpp
// 쿼리 임베딩
auto query_embedding = embedding_model.encode("machine learning");

// 문서 임베딩
auto doc_embeddings = embedding_model.encode({
  "Deep learning is a subset of machine learning",
  "Natural language processing uses neural networks",
  "Machine learning algorithms learn from data"
});

// 코사인 유사도 계산
for (auto &doc_emb : doc_embeddings) {
  float similarity = cosine_similarity(query_embedding, doc_emb);
  std::cout << "Similarity: " << similarity << std::endl;
}
```

### 2. 텍스트 클러스터링

```cpp
// 모든 문서 임베딩
std::vector<std::vector<float>> embeddings;
for (auto &text : documents) {
  embeddings.push_back(embedding_model.encode(text));
}

// K-means 클러스터링
auto clusters = kmeans(embeddings, k=5);
```

### 3. 유사도 측정

```cpp
auto emb1 = embedding_model.encode("The car is red");
auto emb2 = embedding_model.encode("The vehicle is crimson");

float similarity = cosine_similarity(emb1, emb2);
// 결과: ~0.85 (매우 유사함)
```

## nntrainer_config.json 설정

```json
{
  "batch_size": 1,
  "model_tensor_type": "FP32",
  "embedding_dtype": "FP32",
  "init_seq_len": 512,
  "max_seq_len": 512,
  "num_to_generate": 0,
  "tokenizer_file": "tokenizer.json",
  "module_config_path": "modules.json",
  "model_type": "embedding"
}
```

**주요 차이점:**
- `num_to_generate`: 0 (생성 안 함)
- `module_config_path`: modules.json 경로
- `model_type`: "embedding"

## CausalLM과의 비교

### CausalLM 추론

```cpp
// Prefill phase
output = model->incremental_inference(BATCH_SIZE, input, label, init_len, 0, init_len, false);

// Generation phase (loop)
for (int i = 0; i < NUM_TO_GENERATE; ++i) {
  output = model->incremental_inference(BATCH_SIZE, input, label, 1, pos, pos+1, true);
  // ... 토큰 샘플링 ...
}
```

### Embedding 추론

```cpp
// Single forward pass only
output = model->incremental_inference(BATCH_SIZE, input, label, input_len, 0, input_len, false);
// 완료! (반복 없음)
```

## 요약

### Embedding 모델의 핵심 특징

1. **모듈식 구조:** modules.json으로 동적 모듈 구성
2. **Single Forward Pass:** 전체 시퀀스 한 번에 처리
3. **Pooling:** 시퀀스를 고정 차원 벡터로 압축
4. **Normalization:** 유사도 계산을 위한 단위 길이 정규화
5. **출력:** 고정 차원 embedding vector (DIM)

### 작동 순서

1. **setupParameters:** modules.json과 각 모듈의 config.json 로드
2. **constructModel:** Transformer + Pooling + Normalize 등 추가
3. **encode:** 텍스트 → 토큰 → 임베딩 벡터
4. **run:** embedding 결과 출력

### 사용 시점

- 텍스트의 의미적 표현이 필요할 때
- 유사도 계산, 검색, 클러스터링 등
- Generation이 필요 없을 때

## 참고

- **Header:** `embedding.h`
- **구현:** `embedding.cpp`
- **레이어:** `Applications/CausalLM/layers/embedding_*.cpp`
- **Config:** `modules.json`, `Module_name/config.json`
- **사용 예시:** Qwen3Embedding, EmbeddingGemma (gemma3)
