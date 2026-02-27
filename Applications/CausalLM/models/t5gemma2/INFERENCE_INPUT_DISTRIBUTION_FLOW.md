# Incremental_inference에서 여러 Input을 여러 Layer로 분배하는 과정

## 개요

`incremental_inference(input)`에 여러 input을 전달했을 때, NNTrainer가 이를 각 레이어로 분배하는 과정을 상세히 설명합니다.

## 전체 플로우

```
사용자 코드
  ↓
incremental_inference(std::vector<float *> input)
  ↓
input_tensors로 변환 (Tensor::Map)
  ↓
getInputDimension()으로 in_dim 조회
  ↓
validateInput()로 검증
  ↓
setInputsLabels()로 model_graph에 설정
  ↓
incremental_forwarding()로 전파
  ↓
각 레이어의 forward 호출
  ↓
출력 반환
```

## 상세 코드 분석

### 1. 사용자 코드에서의 호출

```cpp
// 사용자 코드 (causal_lm.cpp 또는 t5gemma2_causallm.cpp)
std::vector<float *> inputs = {
  vision_input,   // [BATCH, 3, 896, 896]
  text_input,      // [BATCH, SEQ_LEN]
  decoder_input    // [BATCH, SEQ_LEN]
};

auto output = model->incremental_inference(
  BATCH_SIZE, inputs, {},  // label은 비어있음
  init_seq_len, 0, seq_len, false);
```

### 2. incremental_inference 시작 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1858-1920
std::vector<float *> NeuralNetwork::incremental_inference(
  unsigned int batch_size,
  const std::vector<float *> &input,           // ← 사용자가 전달한 여러 input
  const std::vector<float *> &label,
  unsigned int init_seq_len,
  unsigned int from,
  unsigned int to,
  bool output_hidden_state) {

  // ========== STEP 1: sharedConstTensors로 변환 ==========
  
  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();  // ← 중요! Input Layer들의 dimension 조회

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);  // batch size 설정
    
    // input[idx] → Tensor::Map → input_tensors[idx]
    input_tensors.emplace_back(MAKE_SHARED_TENSOR(Tensor::Map(
      input[idx],                              // raw 포인터
      in_dim[idx].getDataLen() * sizeof(float),  // 데이터 크기
      in_dim[idx],                             // TensorDim (shape 포함)
      0)));                                    // offset
  }
  
  // input_tensors[0] → Tensor::Map(vision_input, [1, 3, 896, 896])
  // input_tensors[1] → Tensor::Map(text_input, [1, 1, SEQ_LEN])
  // input_tensors[2] → Tensor::Map(decoder_input, [1, 1, SEQ_LEN])
  
  // ========== STEP 2: label 처리 (비어있으면 생략) ==========
  
  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    // ... label도 input_tensors처럼 변환 ...
    output_tensors = incremental_inference(input_tensors, label_tensors,
                                           init_seq_len, from, to);
  } else {
    // label이 비어있으면 input만 전달
    output_tensors = incremental_inference(input_tensors, init_seq_len, from, to);
  }
  
  // ========== STEP 3: 출력 처리 ==========
  std::vector<float *> output;
  output.reserve(output_tensors.size());
  
  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    // ... 출력 데이터 처리 ...
    output.push_back(out_t.getData());
  }
  
  return output;
}
```

### 3. getInputDimension() 호출 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1478
std::vector<TensorDim> NeuralNetwork::getInputDimension() {
  if (!compiled) {
    throw std::logic_error("model should be compiled before get dimension");
  }
  return model_graph.getInputDimension();  // NetworkGraph에 위임
}
```

**NetworkGraph::getInputDimension()**에서는 compile 시점에 등록된 Input Layer들의 dimension을 반환합니다:

```cpp
// nntrainer/graph/network_graph.h (추정)
std::vector<TensorDim> getInputDimension() {
  std::vector<TensorDim> dims;
  
  // compile 시점에 등록된 Input Layer들을 순회
  for (auto &node : getLayerNodes()) {
    if (node->getType() == "input") {
      // 각 Input Layer의 input_shape 속성에서 dimension 추출
      dims.push_back(node->getInputDimension());
    }
  }
  
  return dims;
}
```

**예시:**
```cpp
// constructModel()에서 추가된 Input Layer 순서:
// 1. vision_input0: input_shape="1:3:896:896"
// 2. text_input0: input_shape="1:1:512"
// 3. decoder_input0: input_shape="1:1:512"

// getInputDimension() 반환:
dims = [
  TensorDim(1, 3, 896, 896),  // vision_input0
  TensorDim(1, 1, 512),        // text_input0
  TensorDim(1, 1, 512)         // decoder_input0
]

// input.size() == 3 (사용자가 전달한 input 개수)
// in_dim.size() == 3 (getInputDimension()이 반환한 dimension 개수)
// input_tensors.reserve(input.size());  // 크기 일치 확인
```

### 4. validateInput() 검증 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1394-1416
bool NeuralNetwork::validateInput(sharedConstTensors X) {
  auto input_dim = getInputDimension();
  
  // 입력 개수 검증
  if (X.size() != input_dim.size()) {
    ml_loge("Error: provided number of inputs %d, required %d",
            (int)X.size(), (int)input_dim.size());
    return false;
  }
  
  // 각 입력의 shape 검증
  for (unsigned int dim = 0; dim < input_dim.size(); dim++) {
    if (input_dim[dim] != X[dim]->getDim()) {
      ml_loge("Error: provided input shape does not match required shape");
      std::stringstream ss;
      ss << X[dim]->getDim();
      ml_loge("Provided tensor summary : %s", ss.str().c_str());
      
      ss.str(std::string());
      ss << input_dim[dim];
      ml_loge("Required tensor summary : %s", ss.str().c_str());
      return false;
    }
  }
  
  return true;
}
```

**검증 예시:**
```cpp
// 사용자가 전달한 input:
X = [
  Tensor::Map(vision_input, [1, 3, 896, 896]),  // X[0]
  Tensor::Map(text_input, [1, 1, 512]),        // X[1]
  Tensor::Map(decoder_input, [1, 1, 512])     // X[2]
]

// getInputDimension()에서 가져온 dimension:
input_dim = [
  TensorDim(1, 3, 896, 896),  // input_dim[0]
  TensorDim(1, 1, 512),        // input_dim[1]
  TensorDim(1, 1, 512)         // input_dim[2]
]

// 검증:
X[0]->getDim() == input_dim[0]  // [1, 3, 896, 896] == [1, 3, 896, 896] ✓
X[1]->getDim() == input_dim[1]  // [1, 1, 512] == [1, 1, 512] ✓
X[2]->getDim() == input_dim[2]  // [1, 1, 512] == [1, 1, 512] ✓

// 모두 일치하면 true 반환
```

### 5. setInputsLabels() 호출 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1817
model_graph.setInputsLabels(input_tensors, label);

// nntrainer/graph/network_graph.cpp (추정)
void NetworkGraph::setInputsLabels(
  sharedConstTensors input_tensors,
  sharedConstTensors label_tensors) {
  
  // Input tensors를 모델 그래프의 Input Layer에 설정
  for (unsigned int idx = 0; idx < input_tensors.size(); idx++) {
    // idx 순서대로 Input Layer 찾기
    auto input_layer = getLayerNodes()[idx];  // 첫 번째는 vision_input0
    
    if (input_layer->getType() == "input") {
      // Input Layer의 input tensor 설정
      input_layer->getRunContext().setInput(input_tensors[idx]);
    }
  }
  
  // Label tensors도 비슷하게 설정
  for (unsigned int idx = 0; idx < label_tensors.size(); idx++) {
    // Label Layer에 label tensor 설정
  }
}
```

**중요:** Input Layer의 추가 순서와 input의 순서가 일치해야 합니다!

```cpp
// constructModel()에서 추가한 순서:
// model->addLayer(createLayer("input", {"name=vision_input0", ...}));  // 1번째
// model->addLayer(createLayer("input", {"name=text_input0", ...}));     // 2번째
// model->addLayer(createLayer("input", {"name=decoder_input0", ...}));  // 3번째

// inference 시 전달한 순서:
std::vector<float *> inputs = {
  vision_input,   // → vision_input0 (1번째 Input Layer)
  text_input,     // → text_input0 (2번째 Input Layer)
  decoder_input    // → decoder_input0 (3번째 Input Layer)
};

// setInputsLabels() 내부:
// inputs[0] → vision_input0
// inputs[1] → text_input0
// inputs[2] → decoder_input0
```

### 6. incremental_forwarding() 호출 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1800-1815
sharedConstTensors NeuralNetwork::incremental_inference(
  sharedConstTensors X, sharedConstTensors label, unsigned int init_seq_len,
  unsigned int from, unsigned int to) {
  
  // batch size 검증
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }
  
  // validateInput()로 shape 검증
  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");
  
  // tensor 할당 (처음에는 only)
  if (!from) {
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
  }
  
  // incremental_forwarding()로 전파
  out = incremental_forwarding(from, to, X, label, false);
  
  // 입력/레이블 초기화
  model_graph.setInputsLabels({}, {});
  
  return out;
}
```

### 7. incremental_forwarding() 레이어 순회 (neuralnet.cpp)

```cpp
// nntrainer/models/neuralnet.cpp: 1680-1705
sharedConstTensors NeuralNetwork::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {

  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, from, to, stop_cb](std::shared_ptr<LayerNode> node, bool training) -> void {
    
    // 각 레이어의 incremental_forwarding 호출
    node->incremental_forwarding(from, to, training);
  };
  
  // NetworkGraph가 레이어를 순회하며 forwarding_op 호출
  return model_graph.incremental_forwarding(from, to, training, forwarding_op,
                                            stop_cb, userdata);
}
```

## NetworkGraph에서의 레이어 순회 및 분배

### 1. compile 시점의 Input Layer 등록

```cpp
// nntrainer/models/neuralnet.cpp: 165-196 (compile)
int NeuralNetwork::compile(ExecutionMode mode) {
  // ...
  
  auto &input_conn = std::get<std::vector<props::InputConnection>>(model_props);
  
  std::vector<std::unique_ptr<GraphRealizer>> realizers;
  
  // Input Layer 연결 설정
  realizers.emplace_back(new PreviousInputRealizer(
    std::vector<Connection>(input_conn.begin(), input_conn.end())));
  
  // Graph representation 변환
  for (auto &realizer : realizers) {
    graph_representation = realizer->realize(graph_representation);
  }
  
  // NetworkGraph 생성 및 레이어 추가
  model_graph = NetworkGraph(...);
  for (auto &node : graph_representation) {
    model_graph.addLayer(node);
  }
  
  // compile: 레이어 연결 관계 확정, Input Layer 순서 고정
  int status = model_graph.compile(loss_type);
  
  compiled = true;
  return status;
}
```

### 2. Input Layer에서의 forward 호출

```cpp
// nntrainer/layers/input_layer.cpp (추정)
void InputLayer::incremental_forwarding(unsigned int from, unsigned int to, bool training) {
  // Input Layer는 그냥 입력 텐서를 그대로 출력으로 전달
  // 연결된 레이어에서 이 입력을 사용
}
```

**실제 forward 과정:**

```cpp
// nntrainer/graph/network_graph.cpp (추정)
sharedConstTensors NetworkGraph::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  
  // 레이어를 순회하며 forward 호출
  for (auto iter = cbegin(); iter != cend(); ++iter) {
    auto &node = *iter;
    
    // forwarding_op: node->incremental_forwarding(from, to, training)
    forwarding_op(node, training);
    
    // node->incremental_forwarding() 내부:
    // - Input Layer: 입력을 그대로 출력
    // - 다른 Layer: input_layers로 지정된 레이어의 출력을 가져옴
  }
  
  return output_tensors;
}
```

## 전체 플로우 다이어그램

```
사용자 코드
  │
  ├─ std::vector<float *> inputs = {
  │    vision_input,
  │    text_input,
  │    decoder_input
  │  };
  │
  ↓
  model->incremental_inference(BATCH_SIZE, inputs, {}, ...)
  │
  ↓ NeuralNetwork::incremental_inference()
  │
  ├─ 1. getInputDimension() 호출
  │   │
  │   └─ model_graph.getInputDimension()
  │       │
  │       └─ compile 시점에 등록된 Input Layer 순회
  │           ├─ vision_input0: [1, 3, 896, 896]
  │           ├─ text_input0: [1, 1, 512]
  │           └─ decoder_input0: [1, 1, 512]
  │       │
  │       └─ std::vector<TensorDim> 반환
  │
  ├─ 2. input_tensors로 변환 (Tensor::Map)
  │   │
  │   ├─ for idx in 0..2:
  │   │   │
  │   │   ├─ in_dim[idx].batch(batch_size)
  │   │   │
  │   │   └─ input_tensors[idx] = Tensor::Map(input[idx], in_dim[idx])
  │   │       │
  │   │       ├─ input_tensors[0] = Tensor::Map(vision_input, [1, 3, 896, 896])
  │   │       ├─ input_tensors[1] = Tensor::Map(text_input, [1, 1, 512])
  │   │       └─ input_tensors[2] = Tensor::Map(decoder_input, [1, 1, 512])
  │   │
  │   └─ sharedConstTensors input_tensors 생성 완료
  │
  ├─ 3. validateInput() 호출
  │   │
  │   ├─ X.size() == input_dim.size() 검증 (3 == 3) ✓
  │   │
  │   └─ X[i]->getDim() == input_dim[i] 검증
  │       ├─ X[0]->getDim() == [1, 3, 896, 896] ✓
  │       ├─ X[1]->getDim() == [1, 1, 512] ✓
  │       └─ X[2]->getDim() == [1, 1, 512] ✓
  │
  ├─ 4. setInputsLabels(input_tensors, {}) 호출
  │   │
  │   └─ model_graph.setInputsLabels()
  │       │
  │       └─ for idx in 0..2:
  │           │
  │           ├─ getLayerNodes()[idx] → Input Layer 찾기
  │           │   ├─ idx=0: vision_input0
  │           │   ├─ idx=1: text_input0
  │           │   └─ idx=2: decoder_input0
  │           │
  │           └─ input_layer->getRunContext().setInput(input_tensors[idx])
  │               ├─ vision_input0 → input_tensors[0]
  │               ├─ text_input0 → input_tensors[1]
  │               └─ decoder_input0 → input_tensors[2]
  │
  ├─ 5. incremental_forwarding(input_tensors, init_seq_len, from, to) 호출
  │   │
  │   └─ model_graph.incremental_forwarding()
  │       │
  │       └─ for each layer in model_graph (topological order):
  │           │
  │           ├─ layer->incremental_forwarding(from, to, training)
  │           │
  │           └─ 레이어 별 forward 수행:
  │               ├─ vision_input0 (Input Layer)
  │               │   └─ 입력을 그대로 출력
  │               │
  │               ├─ vision_patch_embedding (Conv2D)
  │               │   └─ input_layers="vision_input0" → vision_input0의 출력 사용
  │               │
  │               ├─ vision_layer0 (Transformer Block)
  │               │   └─ input_layers="vision_patch_embedding" → 이전 레이어 출력 사용
  │               │
  │               ├─ ...
  │               │
  │               ├─ vision_output_norm (RMSNorm)
  │               │   └─ vision path 마지막
  │               │
  │               ├─ text_input0 (Input Layer)
  │               │   └─ 입력을 그대로 출력
  │               │
  │               ├─ text_embedding0 (Embedding)
  │               │   └─ input_layers="text_input0" → text_input0의 출력 사용
  │               │
  │               ├─ ...
  │               │
  │               ├─ vision_text_fusion (Concat)
  │               │   └─ input_layers="text_embedding0,vision_reshape"
  │               │       └─ 두 개의 입력 결합
  │               │
  │               ├─ encoder_layer0 (Transformer Block)
  │               │   └─ input_layers="vision_text_fusion" → 결합된 출력 사용
  │               │
  │               ├─ ...
  │               │
  │               ├─ encoder_output_norm (RMSNorm)
  │               │   └─ encoder path 마지막
  │               │
  │               ├─ decoder_input0 (Input Layer)
  │               │   └─ 입력을 그대로 출력
  │               │
  │               ├─ decoder_embedding0 (Embedding)
  │               │   └─ input_layers="decoder_input0" → decoder_input0의 출력 사용
  │               │
  │               ├─ ...
  │               │
  │               ├─ decoder_layer0 (Transformer Block)
  │               │   ├─ input_layers="decoder_embedding0" (self-attention)
  │               │   └─ input_layers="encoder_output_norm" (cross-attention의 K, V)
  │               │
  │               └─ final_output (LM Head)
  │
  └─ 6. output_tensors 반환
      │
      └─ 각 레이어의 output에서 최종 출력 추출
```

## 핵심 포인트

### 1. 순서 기반 매핑

**Input Layer 추가 순서와 input 전달 순서가 일치해야 합니다:**

```cpp
// constructModel()
model->addLayer(createLayer("input", {"name=vision_input0", ...}));  // 1번째
model->addLayer(createLayer("input", {"name=text_input0", ...}));     // 2번째
model->addLayer(createLayer("input", {"name=decoder_input0", ...}));  // 3번째

// inference
std::vector<float *> inputs = {
  vision_input,   // → vision_input0 (1번째)
  text_input,     // → text_input0 (2번째)
  decoder_input    // → decoder_input0 (3번째)
};
```

### 2. input_layers로 레이어 연결

각 레이어는 `input_layers` 파라미터로 어디에서 입력을 받을지 지정합니다:

```cpp
// Conv2D layer
model->addLayer(createLayer("conv2d", {
  withKey("name", "vision_patch_embedding"),
  withKey("input_layers", "vision_input0"),  // ← vision_input0에서 입력받음
  ...
}));

// Concat layer
model->addLayer(createLayer("concat", {
  withKey("name", "vision_text_fusion"),
  withKey("input_layers", "text_embedding0,vision_reshape"),  // ← 두 개의 입력
  ...
}));
```

### 3. Topological Order 순회

NetworkGraph는 레이어를 topological order로 순회합니다:

```cpp
// Topological order:
1. vision_input0 (Input)
2. vision_patch_embedding (Conv2D)
3. vision_layer0 (Transformer Block)
...
10. vision_output_norm (RMSNorm)
11. text_input0 (Input)
12. text_embedding0 (Embedding)
...
20. text_output_norm (RMSNorm)
21. decoder_input0 (Input)
22. decoder_embedding0 (Embedding)
...
30. final_output (LM Head)

// 순회하며 forward 호출
for (auto iter = cbegin(); iter != cend(); ++iter) {
  iter->incremental_forwarding(from, to, training);
}
```

### 4. Tensor::Map으로 메모리 공유

`Tensor::Map`을 사용하여 raw 포인터를 Tensor로 래핑합니다 (메모리 복사 없음):

```cpp
input_tensors.emplace_back(MAKE_SHARED_TENSOR(Tensor::Map(
  input[idx],                              // raw 포인터
  in_dim[idx].getDataLen() * sizeof(float),  // 데이터 크기
  in_dim[idx],                             // TensorDim (shape 포함)
  0)));                                    // offset

// 장점:
// - 메모리 복사 없음 (zero-copy)
// - raw 포인터로 직접 접근
// - 효율적
```

## 실제 예시: T5Gemma2

### constructModel()

```cpp
void T5Gemma2Transformer::constructModel() {
  // Input Layer 1
  model->addLayer(createLayer("input", {"name=vision_input0", ...}));
  
  // Vision Encoder
  model->addLayer(createLayer("conv2d", {
    withKey("name", "vision_patch_embedding"),
    withKey("input_layers", "vision_input0"),  // ← vision_input0 사용
    ...
  }));
  
  // ... vision layers ...
  
  // Input Layer 2
  model->addLayer(createLayer("input", {"name=text_input0", ...}));
  
  // Text Encoder
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "text_embedding0"),
    withKey("input_layers", "text_input0"),  // ← text_input0 사용
    ...
  }));
  
  // Fusion
  model->addLayer(createLayer("concat", {
    withKey("name", "vision_text_fusion"),
    withKey("input_layers", "text_embedding0,vision_reshape"),  // ← 두 개의 입력
    ...
  }));
  
  // ... encoder layers ...
  
  // Input Layer 3
  model->addLayer(createLayer("input", {"name=decoder_input0", ...}));
  
  // Decoder
  model->addLayer(createLayer("embedding_layer", {
    withKey("name", "decoder_embedding0"),
    withKey("input_layers", "decoder_input0"),  // ← decoder_input0 사용
    ...
  }));
  
  // ... decoder layers ...
}
```

### run()

```cpp
void T5Gemma2ForConditionalGeneration::run(...) {
  // Input 준비
  float *vision_input = processor_output.pixel_values.data();
  float *text_input = processor_output.input_ids.data();
  float *decoder_input = (float *)malloc(BATCH_SIZE * sizeof(float));
  decoder_input[0] = BOS_TOKEN_ID;
  
  // 여러 input 전달
  std::vector<float *> inputs = {
    vision_input,   // → vision_input0
    text_input,     // → text_input0
    decoder_input    // → decoder_input0
  };
  
  // inference
  auto output = model->incremental_inference(
    BATCH_SIZE, inputs, {}, 
    seq_len, 0, seq_len, false);
  
  // 내부 과정:
  // 1. inputs[0] → vision_input0 → Vision Encoder
  // 2. inputs[1] → text_input0 → Text Encoder (Fusion에서 vision_output와 결합)
  // 3. inputs[2] → decoder_input0 → Decoder (encoder_output를 cross-attention에서 사용)
}
```

## 요약

### 질문: 여러 input이 들어갔을 때 incremental_inference에서 어떻게 여러 layer로 보내는가?

**답변:**

1. **Input Layer 순서 기반 매핑:**
   - `constructModel()`에서 추가한 Input Layer 순서대로 `getInputDimension()`가 dimension 반환
   - 사용자가 전달한 input 순서와 Input Layer 순서가 일치해야 함

2. **Tensor::Map으로 변환:**
   - `input[idx]` → `Tensor::Map(input[idx], in_dim[idx])`
   - 메모리 복사 없이 raw 포인터 래핑

3. **validateInput()로 검증:**
   - input 개수와 dimension 개수 일치 검증
   - 각 input의 shape 검증

4. **setInputsLabels()로 설정:**
   - `inputs[0]` → 1번째 Input Layer (vision_input0)
   - `inputs[1]` → 2번째 Input Layer (text_input0)
   - `inputs[2]` → 3번째 Input Layer (decoder_input0)

5. **incremental_forwarding()로 전파:**
   - Topological order로 레이어 순회
   - 각 레이어에서 `input_layers`로 지정된 레이어의 출력 사용

### 핵심 메커니즘

- **순서 기반:** Input Layer 추가 순서와 input 전달 순서가 일치
- **input_layers 파라미터:** 각 레이어가 어디서 입력을 받을지 지정
- **Topological 순회:** 레이어를 순서대로 forward 호출

이것이 NNTrainer에서 여러 input을 여러 layer로 분배하는 방법입니다!
