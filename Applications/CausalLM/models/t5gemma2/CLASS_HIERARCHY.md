# CausalLM Class Hierarchy and Method Resolution

## Class Hierarchy

```
Transformer (base class)
    ├── initialize()          - 구현됨 (transformer.cpp)
    ├── constructModel()       - virtual (기본 구현)
    ├── setupParameters()      - virtual (기본 구현)
    ├── registerCustomLayers() - virtual (기본 구현)
    └── run()                 - virtual (빈 스텁 - 아무것도 안 함)
         ↓ (virtual 상속)
CausalLM (virtual public Transformer)
    ├── initialize()          - override 하지 않음 (Transformer::initialize() 사용)
    ├── constructModel()       - override (LM Head 추가)
    ├── setupParameters()      - override (CausalLM 파라미터 추가)
    ├── registerCustomLayers() - override (LM Head 레이어 등록)
    ├── run()                 - override (실제 generation 구현)
    ├── generate()            - CausalLM 전용 (token sampling)
    ├── registerOutputs()      - CausalLM 전용 (출력 표시)
    ├── save_kvcache()        - CausalLM 전용 (KV cache 저장)
    └── load_kvcache()        - CausalLM 전용 (KV cache 로드)
         ↓ (virtual 상속)
Gemma3Transformer (virtual public Transformer)
    ├── constructModel()       - override (Gemma3 특화 구조)
    ├── createAttention()      - override (Gemma3 스타일)
    ├── createMlp()            - override (Gemma3 스타일)
    └── setupParameters()      - override (Gemma3 파라미터)
         ↓
Gemma3CausalLM (multiple inheritance - 다이아몬드 상속)
    ├── inherits from CausalLM
    └── inherits from Gemma3Transformer
```

## 다이아몬드 상속 문제 해결

### 실제 구조

```
        Transformer (base)
              ↓        ↓
        CausalLM    Gemma3Transformer
              ↓        ↓
          Gemma3CausalLM (다중 상속)
```

이것은 **다이아몬드 상속**입니다. 일반적으로 다이아몬드 상속에서는 다음 문제들이 발생할 수 있습니다:
1. 두 개의 Transformer 서브오브젝트 생성 (메모리 낭비) - virtual 상속으로 해결
2. 기본 클래스 멤버에 대한 모호한 참조
3. 중복 생성자 호출

### 그런데 왜 문제가 없나요?

#### 1. virtual 상속을 사용함 (중요!)

```cpp
class CausalLM : virtual public Transformer { ... };  // ✅ virtual 상속
class Gemma3Transformer : virtual public Transformer { ... };  // ✅ virtual 상속
```

**실제로 virtual 상속을 사용합니다!**

#### 2. virtual 상속의 효과

```
Gemma3CausalLM 객체:
┌─────────────────────────────────────┐
│        Gemma3CausalLM                │
└─────────────────────────────────────┘
         ↓          ↓
    CausalLM   Gemma3Transformer
         ↓          ↓
    ┌───────────────────────────┐
    │  Transformer (공통)        │
    │  - 단 하나만 생성됨!       │
    └───────────────────────────┘
```

**virtual 상속을 사용하면 다이아몬드 최상위의 기본 클래스(Transformer)가 단 하나만 생성됩니다.**
- 두 개의 서브오브젝트가 아닌, 하나의 공통 Transformer 서브오브젝트
- 메모리 효율적
- 중복 초기화 방지

#### 3. 메서드 충돌 해결: 두 번 override

**문제:** `setupParameters()`와 `registerCustomLayers()`는 두 부모 클래스가 모두 override 합니다. 충돌하지 않나요?

| 메서드 | CausalLM | Gemma3Transformer | Transformer |
|-------|----------|------------------|-------------|
| `run()` | ✅ override (텍스트 생성) | ❌ | ⭕ virtual 스텁 |
| `constructModel()` | ✅ override (LM Head 추가) | ❌ | ⭕ virtual |
| `setupParameters()` | ✅ override (gen 파라미터) | ✅ override (Gemma3 파라미터) | ⭕ virtual |
| `registerCustomLayers()` | ✅ override (LM Head 등록) | ✅ override (Gemma3 레이어 등록) | ⭕ virtual |
| `initialize()` | ❌ | ❌ | ✅ 구현 |
| `createAttention()` | ❌ | ✅ override (Gemma3 스타일) | ⭕ virtual |
| `createMlp()` | ❌ | ✅ override (Gemma3 스타일) | ⭕ virtual |

**해결 방법:** Gemma3CausalLM에서 두 메서드를 모두 override하고, 내부에서 각 부모의 메서드를 명시적으로 호출

```cpp
// gemma3_causallm.cpp
void Gemma3CausalLM::setupParameters(...) override {
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);    // CausalLM 파라미터
  Gemma3Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);  // Gemma3 파라미터
}

void Gemma3CausalLM::registerCustomLayers() override {
  CausalLM::registerCustomLayers();      // LM Head 레이어 등록
  Gemma3Transformer::registerCustomLayers();  // Gemma3 레이어 등록
}
```

**호출 순서:**
```cpp
// main.cpp
model->initialize()
  ↓ Transformer::initialize()  // (transformer.cpp)
    ↓ registerCustomLayers()  // virtual call
      ↓ Gemma3CausalLM::registerCustomLayers()  // override
        ├─ CausalLM::registerCustomLayers()  // 명시적 호출
        └─ Gemma3Transformer::registerCustomLayers()  // 명시적 호출
    
    ↓ constructModel()  // virtual call
      ↓ CausalLM::constructModel()  // override
        ├─ Transformer::constructModel()  // 명시적 호출 (embedding + decoder blocks)
        │   ├─ createTransformerDecoderBlock() [virtual call]
        │   │   ├─ createAttention() [virtual call]
        │   │   │   ↓ Gemma3Transformer::createAttention() [override]
        │   │   └─ createMlp() [virtual call]
        │   │       ↓ Gemma3Transformer::createMlp() [override]
        │   └─ ... (N layers)
        │
        └─ model->addLayer(LM Head)  // LM Head 추가
```

**중요:** Gemma3Transformer에서 override한 `createAttention()`와 `createMlp()`는 `constructModel()` 안에서 호출되므로, 자동으로 Gemma3 버전이 사용됩니다!

**중요:** 명시적으로 `CausalLM::`와 `Gemma3Transformer::`를 붙여서 호출해야 합니다. 그렇지 않으면:
```cpp
// ❌ 잘못된 방법
void Gemma3CausalLM::registerCustomLayers() override {
  // 아무것도 안 하면 무한 재귀 (stack overflow)
}
```

#### 4. CausalLM::constructModel()의 구현 방식

```cpp
// causal_lm.cpp
void CausalLM::constructModel() {
  // 명시적으로 Transformer::constructModel() 호출
  Transformer::constructModel();  // 공유 Transformer 서브오브젝트 사용
  
  // LM Head 추가
  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}
```

**virtual 상속이므로 `Transformer::constructModel()`는 공유 Transformer 서브오브젝트에 접근합니다.**

#### 5. Gemma3CausalLM의 생성자

```cpp
// gemma3_causallm.h
Gemma3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg)
  : Transformer(sanitizeConfig(cfg), ...),        // 첫 번째 Transformer 초기화
    CausalLM(sanitizeConfig(cfg), ...),           // CausalLM (Transformer 상속)
    Gemma3Transformer(sanitizeConfig(cfg), ...)   // Gemma3Transformer (Transformer 상속)
```

두 번 초기화되지만, 각 초기화는 다른 Transformer 서브오브젝트에 영향을 줍니다.

## main.cpp에서의 호출 분석

### 1. Factory에서 모델 생성

```cpp
// main.cpp
auto model = causallm::Factory::Instance().create(
  "Gemma3ForCausalLM",  // architecture 이름
  cfg,
  generation_cfg,
  nntr_cfg
);

// Factory::create() 내부:
return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg, nntr_cfg);
```

**생성된 객체:** `Gemma3CausalLM` 인스턴스

### 2. initialize() 호출

```cpp
// main.cpp
model->initialize();
```

**호출 순서:**
1. main.cpp는 `Gemma3CausalLM*` 포인터로 `initialize()` 호출
2. `Gemma3CausalLM`는 `initialize()`를 override 하지 않음
3. 따라서 부모 클래스인 `CausalLM`의 `initialize()`를 찾음
4. `CausalLM`도 `initialize()`를 override 하지 않음
5. 따라서 `Transformer::initialize()`가 호출됨

**실제 실행되는 코드:** `transformer.cpp`의 `Transformer::initialize()`

```cpp
// transformer.cpp
void Transformer::initialize() {
  // Step 1: Register custom layers
  registerCustomLayers();  // virtual call - 호출됨
  
  // Step 2: Construct model layers
  constructModel();         // virtual call - 호출됨
  
  // Step 3: Set model properties
  model->setProperty(model_props);
  
  // Step 4: Compile model
  model->compile(ExecutionMode::INFERENCE);
  
  // Step 5: Initialize model
  model->initialize(ExecutionMode::INFERENCE);
}
```

**여기서 virtual 함수 호출:**
- `registerCustomLayers()`:
  - `Gemma3CausalLM`가 override → `Gemma3CausalLM::registerCustomLayers()` 호출
  - 내부에서 `CausalLM::registerCustomLayers()`와 `Gemma3Transformer::registerCustomLayers()` 호출

- `constructModel()`:
  - `CausalLM`가 override → `CausalLM::constructModel()` 호출
  - 내부에서 `Transformer::constructModel()` 호출 후 LM Head 추가

### 3. run() 호출

```cpp
// main.cpp
model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
```

**호출 순서:**
1. main.cpp는 `Gemma3CausalLM*` 포인터로 `run()` 호출
2. `Gemma3CausalLM`는 `run()`을 override 하지 않음
3. 따라서 부모 클래스인 `CausalLM`의 `run()`를 찾음
4. `CausalLM`는 `run()`을 override 함 → `CausalLM::run()` 호출
5. `Transformer::run()`은 빈 스텁 (아무것도 안 함)

**실제 실행되는 코드:** `causal_lm.cpp`의 `CausalLM::run()`

```cpp
// causal_lm.cpp
void CausalLM::run(const WSTR prompt, bool do_sample, 
                   const WSTR system_prompt, const WSTR tail_prompt) {
  // Input preparation
  // Tokenize prompt
  // Prefill phase
  output = model->incremental_inference(...);
  
  // Generation phase
  for (token_generation_idx = input_len + 1; ...) {
    output = model->incremental_inference(...);
    ids_list = generate(output[0], do_sample);
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
  }
}
```

## causal_lm.cpp의 역할

`causal_lm.cpp`는 **CausalLM 전용 기능**을 구현합니다:

### 1. run() - Main Generation Logic
```cpp
void CausalLM::run(const WSTR prompt, bool do_sample, 
                   const WSTR system_prompt, const WSTR tail_prompt)
```
- 텍스트 생성 전체 프로세스 구현
- Input preparation → Prefill → Generation → Output display
- `Transformer::run()`은 빈 함수이므로, CausalLM가 override 해서 실제 로직 구현

### 2. constructModel() - Add LM Head
```cpp
void CausalLM::constructModel() {
  // 먼저 Transformer::constructModel() 호출 (embedding + transformer blocks)
  Transformer::constructModel();
  
  // 그 다음 LM Head 추가
  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}
```
- Transformer의 기본 아키텍처에 LM Head 추가
- LM Head는 토큰 확률 분류 (vocab 크기)

### 3. setupParameters() - CausalLM 파라미터
```cpp
void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Transformer 파라미터 설정
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  
  // CausalLM 전용 파라미터
  EOS_TOKEN_ID = generation_cfg["eos_token_id"];
  BOS_TOKEN_ID = generation_cfg["bos_token_id"];
  TEMPERATURE = generation_cfg["temperature"];
  TOP_K = generation_cfg["top_k"];
  TOP_P = generation_cfg["top_p"];
  BAD_WORD_IDS = nntr_cfg["bad_word_ids"];
  USE_KVCACHE = nntr_cfg["system_prompt"]["kvcache"];
}
```

### 4. generate() - Token Sampling
```cpp
std::vector<unsigned int> generate(float *logits, bool do_sample)
```
- 토큰 샘플링 로직
- Greedy decoding 또는 temperature + top-k + top-p sampling

### 5. registerOutputs() - Display Generated Text
```cpp
void registerOutputs(tokenizer, ids, pos, eos_list)
```
- 생성된 토큰을 디코딩해서 콘솔에 출력
- 불완전한 토큰 처리 (pending_ids)

### 6. save_kvcache() / load_kvcache() - KV Cache
```cpp
void save_kvcache(std::string path, int to)
void load_kvcache(std::string path, int to)
```
- 시스템 프롬프트의 KV cache를 저장/로드
- 반복되는 쿼리에서 성능 향상

### 7. registerCustomLayers() - LM Head 등록
```cpp
void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();  // 기본 레이어 등록
  
  // LM Head 레이어 등록
  app_context->registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
}
```

## 요약: 각 파일의 역할

### transformer.cpp
- **베이스 클래스 구현**
- `initialize()` - 모든 모델의 초기화
- `constructModel()` - 기본 트랜스포머 아키텍처 (embedding + decoder blocks)
- `setupParameters()` - 기본 파라미터 로드
- `registerCustomLayers()` - 기본 커스텀 레이어 등록 (RMSNorm, MHA, SwiGLU, Embedding)
- `run()` - 빈 함수 (virtual 스텁)

### causal_lm.cpp
- **CausalLM 전용 기능**
- `run()` - **텍스트 생성 전체 로직** (override Transformer::run())
- `constructModel()` - **LM Head 추가** (override Transformer::constructModel())
- `setupParameters()` - **generation 파라미터 로드** (override Transformer::setupParameters())
- `generate()` - 토큰 샘플링
- `registerOutputs()` - 출력 표시
- `save_kvcache()` / `load_kvcache()` - KV cache 관리

### gemma3/gemma3_causallm.cpp
- **Gemma3 전용 기능**
- Gemma3 특화 파라미터 설정
- Gemma3 특화 커스텀 레이어 등록
- CausalLM과 Gemma3Transformer의 기능 조합

## 실제 실행 흐름

```
main.cpp:
  model = Factory::create("Gemma3ForCausalLM", cfg, gen_cfg, nntr_cfg)
  ↓
  gemma3_causallm.cpp: Gemma3CausalLM constructor
    - Transformer constructor
    - CausalLM constructor
    - Gemma3Transformer constructor
  ↓
main.cpp:
  model->initialize()
  ↓
  transformer.cpp: Transformer::initialize() (CausalLM가 override 안 함)
    - registerCustomLayers() → gemma3_causallm.cpp: Gemma3CausalLM::registerCustomLayers()
    - constructModel() → causal_lm.cpp: CausalLM::constructModel()
                       → transformer.cpp: Transformer::constructModel() (내부 호출)
    - model->setProperty()
    - model->compile()
    - model->initialize()
  ↓
main.cpp:
  model->load_weight(path)
  ↓
main.cpp:
  model->run(prompt, do_sample, system_head_prompt, system_tail_prompt)
  ↓
  causal_lm.cpp: CausalLM::run() (override Transformer::run())
    - Input preparation
    - Prefill: model->incremental_inference(...)
    - Generation loop:
      - model->incremental_inference(...)
      - generate() → causal_lm.cpp
      - registerOutputs() → causal_lm.cpp
```

## 핵심 질문: Transformer::initialize()가 내 Override한 구조를 어떻게 반영하나요?

### 질문
"initialize()는 Transformer클래스에서 부르는 건데, 어떻게 내가 Override한 나의 LM(Gemma3Transformer)의 구조가 거기에 반영이 잘 돼?"

### 답변: Virtual 함수 다형성 (Polymorphism)

**핵심 개념:** virtual 함수는 **실제 객체 타입**에 따라 호출됩니다!

### 상세 호출 흐름

```cpp
// main.cpp
model->initialize()  // model은 Gemma3CausalLM* 타입
  ↓
// transformer.cpp
Transformer::initialize() {
  registerCustomLayers();  // ← virtual call! 실제 객체가 Gemma3CausalLM이므로
  constructModel();        // ← virtual call! override된 버전 호출
}
```

### constructModel()의 상세 흐름

```cpp
// transformer.cpp: Transformer::constructModel()
void Transformer::constructModel() {
  // Input layer
  model->addLayer(input_layer);
  
  // Embedding layer
  model->addLayer(embedding_layer);
  
  // Transformer blocks (NUM_LAYERS 만큼 반복)
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto blocks = createTransformerDecoderBlock(i, input_name);
    model->addLayer(blocks);
  }
  
  // Output RMSNorm
  model->addLayer(rms_norm_layer);
}
```

### createTransformerDecoderBlock()의 상세 흐름

```cpp
// transformer.cpp
std::vector<LayerHandle> Transformer::createTransformerDecoderBlock(
    const int layer_id, std::string input_name) {
  
  // 1. RMSNorm
  layers.push_back(rms_norm_layer);
  
  // 2. Attention ← virtual call!
  auto att_layer = createAttention(layer_id, ...);  // ← virtual call!
  // 실제 객체가 Gemma3CausalLM이므로 Gemma3Transformer::createAttention() 호출
  
  // 3. Addition
  layers.push_back(addition_layer);
  
  // 4. RMSNorm
  layers.push_back(rms_norm_layer);
  
  // 5. MLP ← virtual call!
  auto mlp_layer = createMlp(layer_id, ...);  // ← virtual call!
  // 실제 객체가 Gemma3CausalLM이므로 Gemma3Transformer::createMlp() 호출
  
  // 6. Addition
  layers.push_back(addition_layer);
  
  return layers;
}
```

### 왜 Gemma3 버전이 호출되나요?

**객체의 실제 타입:** `Gemma3CausalLM`

```cpp
// main.cpp
auto model = Factory::create("Gemma3ForCausalLM", cfg, gen_cfg, nntr_cfg);
// 실제 생성된 객체: Gemma3CausalLM 인스턴스

model->initialize();
// model의 정적 타입: Transformer* (또는 CausalLM*)
// model의 동적 타입: Gemma3CausalLM
```

**Virtual 함수 호출 규칙:**
```cpp
this->createAttention(...)  // virtual call
// this는 실제로 Gemma3CausalLM 객체를 가리킴
// Gemma3CausalLM은 createAttention()를 override 안 함
// 따라서 Gemma3Transformer::createAttention() 호출
```

### 완전한 호출 트리

```
model->initialize()
  ↓ Transformer::initialize() [transformer.cpp]
    ↓ registerCustomLayers() [virtual call]
      ↓ Gemma3CausalLM::registerCustomLayers() [gemma3_causallm.cpp - override]
        ├─ CausalLM::registerCustomLayers() [명시적 호출]
        └─ Gemma3Transformer::registerCustomLayers() [명시적 호출]
    
    ↓ constructModel() [virtual call]
      ↓ CausalLM::constructModel() [causal_lm.cpp - override]
        ├─ Transformer::constructModel() [명시적 호출]
        │   ├─ createInputLayer()
        │   ├─ createEmbeddingLayer()
        │   ├─ for (i = 0; i < NUM_LAYERS; ++i)
        │   │   └─ createTransformerDecoderBlock(i) [virtual call]
        │   │       ├─ createRMSNorm()
        │   │       ├─ createAttention() [virtual call]
        │   │       │   ↓ Gemma3Transformer::createAttention() [override]
        │   │       │   └─ Gemma3 스타일 attention 레이어 생성
        │   │       ├─ createAddition()
        │   │       ├─ createRMSNorm()
        │   │       ├─ createMlp() [virtual call]
        │   │       │   ↓ Gemma3Transformer::createMlp() [override]
        │   │       │   └─ Gemma3 스타일 MLP 레이어 생성
        │   │       └─ createAddition()
        │   └─ createRMSNorm() (output)
        │
        └─ model->addLayer(LM Head)  // CausalLM이 추가
```

### 중요한 점

1. **Transformer::initialize()**는 `constructModel()`을 호출하지만, 이는 virtual call입니다.
2. **Virtual call**은 실제 객체의 동적 타입(`Gemma3CausalLM`)을 봅니다.
3. `Gemma3CausalLM`은 `constructModel()`를 override 하지 않지만, 부모인 `CausalLM`가 override 했습니다.
4. 따라서 `CausalLM::constructModel()`가 호출됩니다.
5. `CausalLM::constructModel()` 내부에서 `Transformer::constructModel()`를 호출합니다.
6. `Transformer::constructModel()` 내부에서 `createTransformerDecoderBlock()`을 호출합니다.
7. `createTransformerDecoderBlock()` 내부에서 `createAttention()`과 `createMlp()`를 virtual call로 호출합니다.
8. `Gemma3CausalLM`은 이들을 override 하지 않지만, 또 다른 부모인 `Gemma3Transformer`가 override 했습니다.
9. 따라서 `Gemma3Transformer::createAttention()`과 `Gemma3Transformer::createMlp()`가 호출됩니다.

### 이것이 다형성 (Polymorphism)의 힘!

```cpp
// 예시 코드
void Transformer::createDecoderBlock() {
  // 기본 코드는 transformer.cpp에 있음
  auto att = createAttention(...);  // virtual call
  auto mlp = createMlp(...);        // virtual call
  
  // 실제로는 Gemma3Transformer::createAttention()와 
  // Gemma3Transformer::createMlp()가 호출됨!
}

// gemma3/gemma3_causallm.cpp
std::vector<LayerHandle> Gemma3Transformer::createAttention(...) {
  // Gemma3 특화 attention 구현
  // 예: 특수한 scaling, 다른 activation 등
}

std::vector<LayerHandle> Gemma3Transformer::createMlp(...) {
  // Gemma3 특화 MLP 구현
  // 예: GatedLinearUnit 대신 SwiGLU 등
}
```

### 결론

**Transformer::initialize()**는 기본 프레임워크를 제공하지만, **virtual 함수**를 통해 실제 구현은 자식 클래스에서 동적으로 결정됩니다.

- `createAttention()`과 `createMlp()`는 `Transformer::constructModel()` 안에서 호출되지만
- Virtual call이므로 `Gemma3Transformer`가 override한 버전이 자동으로 호출됨
- 따라서 Gemma3의 특화된 구조가 자동으로 반영됨!

이것이 C++의 **virtual 함수 다형성**을 활용한 전형적인 템플릿 메서드 패턴 (Template Method Pattern)입니다.

## 결론

**질문: main.cpp에서 model->run()이나 initialize()를 하는데, 이는 transformer을 부르는 거 아냐? 그러면 causal_lm.cpp는 어디에 쓰여?**

**답변:**

1. **initialize():**
   - `Transformer::initialize()`가 호출됨
   - 하지만 이 안에서 `registerCustomLayers()`, `constructModel()` 같은 virtual 함수들은 `CausalLM`의 버전이 호출됨
   - `causal_lm.cpp`의 `constructModel()`이 호출되어 LM Head가 추가됨

2. **run():**
   - `CausalLM::run()`이 호출됨 (override)
   - `Transformer::run()`은 빈 함수이므로, 실제 생성 로직은 `causal_lm.cpp`에 있음
   - 여기서 `generate()`, `registerOutputs()` 등 `causal_lm.cpp`의 메서드들이 사용됨

3. **causal_lm.cpp의 역할:**
   - 텍스트 생성 로직 (`run()`)
   - LM Head 추가 (`constructModel()`)
   - Generation 파라미터 설정 (`setupParameters()`)
   - 토큰 샘플링 (`generate()`)
   - 출력 표시 (`registerOutputs()`)
   - KV cache 관리

즉, **initialize()**는 `transformer.cpp`의 코드가 실행되지만, 내부의 virtual 함수들은 `causal_lm.cpp`의 것들이 호출됩니다. **run()**은 바로 `causal_lm.cpp`의 코드가 실행됩니다.
