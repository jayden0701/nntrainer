좀 다르긴 한데...일단 넘어갈까 그냥?

* 현재 image processor을 JPG로 하면 값은 유사함. (똑같지는 않음)
PNG는 값이 좀 다른데, 비교는 나중에 해보면 될 거 같음.
아마 A(투명도)를 PIL에서 to_rgb하는 과정에서 문제가 되는 것 같음
=> 아닌 거 같은데...


일단 encoder-decoder을 만들어 보자~


할일
1. Text(Start of image토큰 있음) + 여러 이미지 받을 수 있는 형태 제작
2. encoder에서 이를 잘 받자(896x896을 256으로 받는지)
3. 실제로는 encoder에서 잘 받는지 확인 후, 걍 pytorch의 값을 받아서 넣자 (이후 layer 값 확인용)
4. 
5. 


할일 1의 세부사항 :
pytorch의
start_of_image 여러개 넣고, 그러면 그 자리에 256개의 image 공간 놓고, token종류 map만들고, attention map만드는 저 원리를 그대로 구현했는지 확인


* 애초에 pytorch버전의 t5gemma2도 이미지 여러개 넣었을 때 좀 삐리하다


Tokenizer(Processor)에서 만드는 Attention Mask는 걍 다 1 주면 됨 (padding이 있으면 해당 padding만큼 가리고 그러는데 그런게 없으니)


NNTrainer쪽 구현 (CLINE)
- 뭔가 글에 <image_soft_token>을 다 박는데, 이게 필요한가? 필요할 거 같음. (tokenizer로 나중에 한 번에 tokenize한다는 가정하에ㅇㅇ) 그러면 /n/n도 미리 글에 넣을까?

다른 CausalLM이 Tokenizer을 어떻게 불러오는 지 함 보자

근데 굳이 260000으로 칠하고 나중에 다시 다 넣는 이유가 뭐지?
-> text는 260000으로 칠해진 상태로 encoder을 통과한다.
image는 아예 따로 통과한다...그것이 문제임

[pytorch]
    self.text_model = T5Gemma2TextEncoder._from_config(config.text_config, eoi_token_index=eoi_token_index)
    self.vision_tower = AutoModel.from_config(config=config.vision_config)
    self.multi_modal_projector = T5Gemma2MultiModalProjector(config)

이듯이 vision tower을 siglip_vision_model에서 들고오는데, 이것도 implement해야함
config대로 되어야 함~



108(\n\n)을 더하는 거는 image이면 자동인건가?
-> 네

        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"
식으로 Processor에서 만듬


(0226 구조짜기)
Transformer을 상속한 T5Gemma 하나에 다 넣는 것이 맞아 보임.
CausalLM에서 제공하는 system prompt의 KV cache화등은 여기에서 일단은 필요가 없어 보임

여기에서
initialize() 시점에서 image input이 있는지 파악, 
있을 시 flag를 통해
SigLip Encoder과 multimodal projector을 constructModel() 시점에 모델에 적절히 붙인다.

이후 해당 flag를 이용해 load_weight()에서 weight을 loading하는데, 당연히 weight이 차이가 나니, weight파일도 버전이 2개여야 할 것.

이후 run() 시점에서 processor로 적절히 처리를 해야 하는데,

image 사용 여부 flag에 따라 생각보다 구조를 바꿔서 잘 붙여야 할 것.

나중에 모델의 부분부분을 모듈화/클래스화 할 수 있을지 생각해 보자



서로 다른 layer에 서로 다른 input을 줄 수 있습니다!

## DIFFERENT_INPUTS_TO_DIFFERENT_LAYERS.md 작성 완료

### 핵심 답변

__질문:__ model의 서로 다른 layer에 서로 다른 input을 줄 수 있어?

__답변:__ 네, 가능합니다!

### 방법

1. __여러 Input Layer 생성:__ `constructModel()`에서 여러 Input Layer 추가
2. __input_layers로 연결:__ 각 레이어에서 `input_layers` 파라미터로 입력 레이어 지정
3. __incremental_inference에 여러 input 전달:__ `std::vector<float *>`로 여러 input 전달

### 구조 예시

```javascript
ModelHandle:
├── Input Layer 1 (vision_input0) → [Vision Encoder] → [vision_output]
│
├── Input Layer 2 (text_input0) → [Text Encoder] → [text_output]
│
└── Input Layer 3 (decoder_input0) → [Decoder] → [final_output]
```

### 코드 예시

```cpp
// constructModel() - 여러 Input Layer 추가
model->addLayer(createLayer("input", {"name=vision_input0", ...}));
model->addLayer(createLayer("input", {"name=text_input0", ...}));
model->addLayer(createLayer("input", {"name=decoder_input0", ...}));

// inference 시 - 여러 input 전달
std::vector<float *> inputs = {
  vision_input,  // → vision_input0
  text_input,     // → text_input0
  decoder_input    // → decoder_input0
};
auto output = model->incremental_inference(BATCH_SIZE, inputs, {}, seq_len, 0, seq_len, false);
```

### 중요: Input 순서 매핑

`incremental_inference(input)`에 전달하는 input의 순서와 `constructModel()`에서 추가한 Input Layer의 순서가 일치해야 합니다!

### 장점

1. __유연성:__ 서로 다른 타입의 input (이미지, 텍스트 등)을 동시에 처리
2. __효율성:__ 필요한 input만 전달하여 불필요한 연산 건너뜀기
3. __모듈성:__ 각 input path를 독립적으로 설계
