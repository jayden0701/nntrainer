# T5Gemma2: Initialize 시점에서 Vision Encoder 유무 처리 방법

## 문제

**질문:** `/home/jayden/NNTrainer/t5gemma2/nntrainer/Applications/CausalLM/main.cpp`에서 `initialize()`에 매개변수가 없는데, T5Gemma2의 경우 image input이 있냐 없냐에 따라 모델 구조가 달라질텐데 (vision encoder), 이를 init 시점에서 어떻게 알려줄 수 있을까?

## 현황 분석

### main.cpp에서의 호출 순서

```cpp
// main.cpp
int main(int argc, char *argv[]) {
  // ... config loading ...
  
  // 모델 생성
  auto model = causallm::Factory::Instance().create(architecture, cfg, generation_cfg, nntr_cfg);
  
  // 초기화 (매개변수 없음!)
  model->initialize();
  
  // 가중치 로드
  model->load_weight(weight_file);
  
  // 실행 (이때 이미지 여부 결정)
  bool do_sample = generation_cfg.value("do_sample", false);
  model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
}
```

### CausalLM::run() 시그니처

```cpp
// causal_lm.h
void run(const WSTR prompt, bool do_sample = false,
         const WSTR system_prompt = "", const WSTR tail_prompt = "") override;
```

**문제:** `run()`은 텍스트만 받고, 이미지는 매개변수로 받지 않음!

## 해결 방법 4가지

### 방법 1: 항상 Vision Encoder 구성, Runtime에서 건너뜀기 (추천)

**개념:** Vision Encoder를 항상 모델에 포함시키고, runtime에 이미지가 없으면 건너뜀기

#### 구현

**1. constructModel()에서 Vision Encoder 항상 포함**

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2Transformer::constructModel() {
  // ========== VISION ENCODER (항상 포함) ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  model->addLayer(createLayer("conv2d", {
    withKey("name", "vision_patch_embedding"),
    withKey("input_layers", "vision_input0"),
    ...
  }));
  
  // ... Vision Encoder layers ...
  
  // ========== TEXT ENCODER ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    ...
  }));
  
  // ... Text Encoder layers ...
  
  // ========== DECODER ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "decoder_input0"),
    ...
  }));
  
  // ... Decoder layers ...
}
```

**2. run()에서 이미지 여부에 따라 conditional 실행**

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,  // ← images 매개변수 추가
    bool do_sample,
    const WSTR system_prompt,
    const WSTR tail_prompt) {
  
  // 이미지 유무 확인
  bool has_vision = !images.empty();
  
  if (has_vision) {
    // 이미지 있으면 Vision Encoder 실행
    auto processor_output = processor.process(prompt, images);
    
    std::vector<float *> vision_inputs = {processor_output.pixel_values.data()};
    auto vision_output = model->incremental_inference(
      BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
    
    // Text Encoder에 vision output 포함
    std::vector<float *> encoder_inputs = {
      processor_output.input_ids.data(),
      vision_output[0]
    };
    
    auto encoder_output = model->incremental_inference(
      BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  } else {
    // 이미지 없으면 Text Encoder만 실행
    auto processor_output = processor.process(prompt, images);
    
    std::vector<float *> encoder_inputs = {processor_output.input_ids.data()};
    
    // dummy vision tensor (0으로 채움)
    std::vector<float> dummy_vision(NUM_PATCHES * DIM, 0.0f);
    encoder_inputs.push_back(dummy_vision.data());
    
    auto encoder_output = model->incremental_inference(
      BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  }
  
  // ... Decoder generation ...
}
```

**3. main.cpp 수정**

```cpp
// main.cpp
int main(int argc, char *argv[]) {
  // ... config loading ...
  
  auto model = causallm::Factory::Instance().create(architecture, cfg, generation_cfg, nntr_cfg);
  model->initialize();
  model->load_weight(weight_file);
  
  bool do_sample = generation_cfg.value("do_sample", false);
  
  // T5Gemma2ForConditionalGeneration인 경우 images 전달
  auto t5gemma2_model = dynamic_cast<T5Gemma2ForConditionalGeneration*>(model.get());
  
  if (t5gemma2_model) {
    // 이미지 경로 처리 (예: 명령행 인자 또는 config)
    std::vector<std::string> images;
    if (argc >= 4) {
      // argv[3]에 이미지 경로
      images.push_back(argv[3]);
    }
    
    t5gemma2_model->run(input_text, images, do_sample, system_head_prompt, system_tail_prompt);
  } else {
    // CausalLM (text-only)
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
  }
}
```

#### 장점
- **간단:** `initialize()` 시그니처 변경 불필요
- **유연:** Runtime에 유연하게 결정
- **호환성:** 기존 CausalLM 인터페이스 유지

#### 단점
- **메모리:** Vision Encoder 가중치 항상 로드 (이미지 없을 때 낭비)
- **초기화:** Vision Encoder 레이어 항상 생성

---

### 방법 2: nntr_config.json에 플래그 추가

**개념:** nntr_config.json에 `"has_vision": true/false` 플래그 추가

#### 구현

**1. nntr_config.json 추가**

```json
{
  "model_file_name": "model.bin",
  "sample_input": "What is the capital of France?",
  "has_vision": true,  // ← 추가: Vision Encoder 포함 여부
  "system_prompt": {
    ...
  }
}
```

**2. setupParameters()에서 플래그 읽기**

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  
  // Vision Encoder 포함 여부 설정
  HAS_VISION = nntr_cfg.value("has_vision", true);  // 기본값: true
}

// t5gemma2_causallm.h
class T5Gemma2Transformer : virtual public Transformer {
protected:
  bool HAS_VISION;  // Vision Encoder 포함 여부
};
```

**3. constructModel()에서 조건부 레이어 추가**

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2Transformer::constructModel() {
  // ========== VISION ENCODER (조건부) ==========
  
  if (HAS_VISION) {
    model->addLayer(createLayer("input", {
      withKey("name", "vision_input0"),
      withKey("input_shape", "1:3:896:896")
    }));
    
    model->addLayer(createLayer("conv2d", {
      withKey("name", "vision_patch_embedding"),
      withKey("input_layers", "vision_input0"),
      ...
    }));
    
    // ... Vision Encoder layers ...
  }
  
  // ========== TEXT ENCODER (항상 포함) ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "text_input0"),
    ...
  }));
  
  // ... Text Encoder layers ...
  
  // ========== DECODER (항상 포함) ==========
  
  model->addLayer(createLayer("input", {
    withKey("name", "decoder_input0"),
    ...
  }));
  
  // ... Decoder layers ...
}
```

**4. run()에서 HAS_VISION 체크**

```cpp
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,
    bool do_sample,
    const WSTR system_prompt,
    const WSTR tail_prompt) {
  
  bool has_vision = HAS_VISION && !images.empty();
  
  if (has_vision) {
    // Vision Encoder 실행
    auto processor_output = processor.process(prompt, images);
    
    std::vector<float *> vision_inputs = {processor_output.pixel_values.data()};
    auto vision_output = model->incremental_inference(
      BATCH_SIZE, vision_inputs, {}, NUM_PATCHES, 0, NUM_PATCHES, false);
    
    // Text Encoder에 vision output 포함
    std::vector<float *> encoder_inputs = {
      processor_output.input_ids.data(),
      vision_output[0]
    };
    
    auto encoder_output = model->incremental_inference(
      BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  } else {
    // Text Encoder만 실행
    auto processor_output = processor.process(prompt, images);
    
    std::vector<float *> encoder_inputs = {processor_output.input_ids.data()};
    
    auto encoder_output = model->incremental_inference(
      BATCH_SIZE, encoder_inputs, {}, TEXT_SEQ_LEN, 0, TEXT_SEQ_LEN, false);
  }
  
  // ... Decoder generation ...
}
```

#### 장점
- **메모리 절약:** has_vision=false면 Vision Encoder 가중치 로드 안 함
- **명확:** Config 파일에서 명시적으로 설정

#### 단점
- **Config 관리:** has_vision 플래그 수동 설정 필요
- **두 가지 모델:** has_vision=true/false마다 다른 가중치 파일 필요

---

### 방법 3: 컴파일 타임 플래그 사용

**개념:** 컴파일 시 매크로로 Vision Encoder 포함 여부 결정

#### 구현

**1. 컴파일 플래그 정의**

```bash
# meson.build
if get_option('enable_vision')
  add_project_arguments('-DENABLE_VISION', language: 'cpp')
endif
```

**2. 코드에서 조건부 컴파일**

```cpp
// t5gemma2_causallm.cpp
void T5Gemma2Transformer::constructModel() {
#ifdef ENABLE_VISION
  // Vision Encoder 포함
  model->addLayer(createLayer("input", {
    withKey("name", "vision_input0"),
    withKey("input_shape", "1:3:896:896")
  }));
  
  // ... Vision Encoder layers ...
#endif
  
  // Text Encoder
  // ... Text Encoder layers ...
}
```

#### 장점
- **최적화:** 컴파일 시 불필요한 코드 제거
- **성능:** has_vision=false면 완전히 제거

#### 단점
- **빌드 필요:** 플래그 변경 시 재빌드 필요
- **유연성 부족:** Runtime에 결정 불가

---

### 방법 4: 레지스트리에 두 가지 모델 등록

**개념:** T5Gemma2ForConditionalGeneration과 T5Gemma2TextOnlyForConditionalGeneration 두 가지 모델 등록

#### 구현

**1. main.cpp에 두 가지 모델 등록**

```cpp
// main.cpp
int main(int argc, char *argv[]) {
  // ... 기존 등록 ...
  
  // T5Gemma2 (with Vision)
  causallm::Factory::Instance().registerModel(
    "T5Gemma2ForConditionalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::T5Gemma2ForConditionalGeneration>(cfg, generation_cfg, nntr_cfg);
    });
  
  // T5Gemma2 (text-only)
  causallm::Factory::Instance().registerModel(
    "T5Gemma2TextOnlyForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::T5Gemma2TextOnlyForConditionalGeneration>(cfg, generation_cfg, nntr_cfg);
    });
  
  // ...
}
```

**2. config.json에서 architecture 선택**

```json
{
  "architectures": ["T5Gemma2ForConditionalLM"],  // 또는 "T5Gemma2TextOnlyForCausalLM"
  ...
}
```

**3. 두 가지 클래스 구현**

```cpp
// t5gemma2_causallm.h

// Vision 포함 버전
class T5Gemma2ForConditionalGeneration : public CausalLM, public T5Gemma2Transformer {
public:
  static constexpr const char *architectures = "T5Gemma2ForConditionalLM";
  
  T5Gemma2ForConditionalGeneration(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    T5Gemma2Transformer(cfg, generation_cfg, nntr_cfg, true) {  // has_vision=true
  }
  
  void run(const WSTR prompt, const std::vector<std::string> &images,
           bool do_sample, const WSTR system_prompt, const WSTR tail_prompt) override {
    // Vision Encoder 사용
  }
};

// Text-only 버전
class T5Gemma2TextOnlyForConditionalGeneration : public CausalLM, public T5Gemma2Transformer {
public:
  static constexpr const char *architectures = "T5Gemma2TextOnlyForCausalLM";
  
  T5Gemma2TextOnlyForConditionalGeneration(json &cfg, json &generation_cfg, json nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    T5Gemma2Transformer(cfg, generation_cfg, nntr_cfg, false) {  // has_vision=false
  }
  
  void run(const WSTR prompt, bool do_sample,
           const WSTR system_prompt, const WSTR tail_prompt) override {
    // Text-only
  }
};
```

#### 장점
- **명확:** 두 가지 모델로 명확히 분리
- **최적화:** 각 모델에 최적화된 가중치

#### 단점
- **코드 중복:** 두 가지 클래스 유지 필요
- **복잡성:** 관리 포인트 증가

---

## 추천 방법: 방법 1 (항상 구성, Runtime 건너뜀기)

### 이유

1. **간단함:** `initialize()` 시그니처 변경 불필요
2. **유연성:** Runtime에 유연하게 결정
3. **호환성:** 기존 CausalLM 인터페이스 유지
4. **실용성:** T5Gemma2의 Vision Encoder는 상대적으로 작음 (Gemma3 비전에 비해)
5. **UX:** 사용자가 config를 수정할 필요 없음

### 구현 요약

```cpp
// 1. constructModel(): Vision Encoder 항상 포함
void T5Gemma2Transformer::constructModel() {
  // Vision Encoder (항상)
  addVisionEncoder();
  
  // Text Encoder (항상)
  addTextEncoder();
  
  // Decoder (항상)
  addDecoder();
}

// 2. run(): Runtime에 이미지 유무 체크
void T5Gemma2ForConditionalGeneration::run(
    const WSTR prompt,
    const std::vector<std::string> &images,
    bool do_sample,
    const WSTR system_prompt,
    const WSTR tail_prompt) {
  
  bool has_vision = !images.empty();
  
  if (has_vision) {
    // Vision Encoder 실행 + Text Encoder (vision 포함)
  } else {
    // Text Encoder만 실행 (vision 없음)
  }
  
  // Decoder 실행
}

// 3. main.cpp: 동적 캐스팅
int main(int argc, char *argv[]) {
  auto model = causallm::Factory::Instance().create(architecture, cfg, generation_cfg, nntr_cfg);
  model->initialize();
  model->load_weight(weight_file);
  
  auto t5gemma2_model = dynamic_cast<T5Gemma2ForConditionalGeneration*>(model.get());
  
  if (t5gemma2_model) {
    // 이미지 처리
    std::vector<std::string> images;
    if (argc >= 4) images.push_back(argv[3]);
    
    t5gemma2_model->run(input_text, images, do_sample, system_head_prompt, system_tail_prompt);
  } else {
    // 일반 CausalLM
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
  }
}
```

### 메모리 고려사항

T5Gemma2의 Vision Encoder 크기:
- Gemma3 비전: ~4GB
- T5Gemma2 비전: ~500MB (추정, 더 작음)

Vision Encoder가 작으므로, 항상 로드해도 큰 부담이 아님.

### 가중치 로드

Vision Encoder가 항상 포함되므로, 가중치 파일에 Vision Encoder 가중치도 포함:

```
model.bin:
├── [Text Encoder weights]
├── [Vision Encoder weights]  // 항상 포함
└── [Decoder weights]
```

이미지가 없는 경우:
- Vision Encoder 가중치는 로드되지만 사용 안 함
- 메모리 낭비: ~500MB (상대적으로 작음)

### 대안: 가중치 파일 분리

메모리 최적화가 중요하다면, 가중치 파일 분리 가능:

```
# Text-only 버전
model_text_only.bin:
├── [Text Encoder weights]
└── [Decoder weights]

# Vision 포함 버전
model_with_vision.bin:
├── [Text Encoder weights]
├── [Vision Encoder weights]
└── [Decoder weights]
```

이 경우, 방법 4 (두 가지 모델 등록) 사용 권장.

## 결론

### 추천: 방법 1 (항상 구성, Runtime 건너뜀기)

**사용 사례:**
- 일반적인 T5Gemma2 사용
- 메모리 여유가 있음
- 간단한 구현 원함

**구현:**
```cpp
// constructModel(): Vision Encoder 항상 포함
// run(): Runtime에 이미지 유무 체크
// main.cpp: 동적 캐스팅으로 T5Gemma2인지 확인
```

### 대안: 방법 2 (nntr_config.json 플래그)

**사용 사례:**
- 메모리 최적화 중요
- has_vision=true/false 두 가지 시나리오 명확히 분리 필요

**구현:**
```cpp
// nntr_config.json: "has_vision": true/false
// setupParameters(): HAS_VISION 플래그 설정
// constructModel(): HAS_VISION에 따라 조건부 레이어 추가
```

### 최종 추천

T5Gemma2의 경우, **방법 1**을 추천합니다:

1. 구현이 가장 간단
2. Runtime 유연성 최고
3. Vision Encoder가 상대적으로 작아 메모리 부담 적음
4. 사용자 경험 최상 (config 수정 불필요)