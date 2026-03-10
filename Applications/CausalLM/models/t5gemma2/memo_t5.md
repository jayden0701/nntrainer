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


멀티모달의 경우 만약 image계열 input이 없다면 모델에 vision encoder등을 load하지 않는 식으로 최적화가 가능함으로, model construct / model weight load전에 이를 알려줘야 함.

weight load가 lazy임으로 나중에 결정해도 될것처럼 보이지만
load_weight시점에서 구조에 맞는 weight를 다 로딩함으로, 그 전에 구조를 결정해놓아야 함


그렇기에 initialize() 시점에 알려주는 로직을 추가하자...

모델(클래스)의 생성자 시점에 config가 들어감으로 거기서 단어키가 뭔지는 알 수 있음


(0304 퇴근전) Encoder의 구조를 대충 CLINE으로 짜는중. 내일와서 어떻게 구현되었는지 확인해 보자

모델이 compile단계 (그 중 output setting)에서 막힘. 제대로 구성이 안되어서 그런 거 같으니 걍 코드쪽에서 한번 더 보자

(0305 퇴근전) Encoder일단 구현 후 weight loading까지는 됨 (config기반 수치들도 일단 넣긴 함)
gemma3의 구조를 많이 재활용함...미리 알았으면 좋았을 듯
내일 와서는 
  1) Weight Loading잘 되었는지 비교
  2) Run()을 텍스트 용으로 만들어서 encoder값이 잘 통과되는지 확인

(0306 점심전) Weight loading을 확인해 봤는데, 아예 다름. converter에서 잘못된 순서로 준 거 같은데, converter의 디버깅 과정과, t5gemma의 layer순서 (compile이랑 순서 확인 완료 후) 를 찍어봐서 제대로 로딩하는지 확인해보자. 순서는 일단 처음이 일치하는데, load시 들어가는 순서가 맞는지 보자(그 load하는 file offset 찍는 gauss3n에 넣은 디버거로 )

(0306 퇴근전) weight별 loading값, offset등 넣었는데, 섞여서 나옴(멀티스레드). 단일 스레드로 할지나 프린트문을 스레드와 상관없이 잘나오게 하는 방법 생각해보자


pre_attn_layernorm은 NNTrainer측 weight가 1 큼 (변환기에서 알아서 했네ㅋㅋ 이게 맞는건지는 나중에 생각해 보자)
q_norm도 마찬가지. 모든 RMS가 이리 했네ㅋㅋ

layer2가 결과가 좀 안정적이라 여기서 확인 ㄱㄱ
-> 대충 다 맞는 거 같음

CLINE으로 text쪽 모델에 돌리게 할 것

norm후로 차이가 나는데, 아무래도 값 이슈가 있는듯. 이제 이걸 확인해보자

다시보니깐 걍 차이가 나는데..?
아! 이거는 print설정의 차이일수도 있겠군
-> 아닌거 같음

이것은...임베딩부터 틀어졌다면. 임베딩 자체의 설정 이슈일수도있다! 임베딩에 어떤 설정을 줘서 어떻게 불렀는지 찾아보자!
embed_scale의 값 자체는 같은데, 처리하는 방식이 다르려나?

Pytorch쪽에서 일단 scale값을 상당히 다르게 들고왔음.
이게 bfloat의 매력이구나...저거 걍 float32로 돌릴 수 있는지 확인해보자(파이토치 쪽에서)

float32로 돌리니 상당히 유사해짐^^

attention(qkv들고가는거)통과후 조금 달라짐. rope때문일까? 그랬으면 더 달랐을 듯. 일단 이 점 확인

어떤시점에서 결국 겁나 갈라짐. 그게 어디인지를 확인해야 함. 진짜 attention때문인가?

(0309 퇴근전) 보니깐 attn(mha_core)에서 달라지는 값을 잡아야 encoder이 같아질것

decoder의 경우 attn 구현이 그리 어렵지는 않을 것 같음(퇴근 직전 그림 그려봄). 다만 encoder쪽 caching구현을 위해, 기존에 어떻게 캐싱했는지 확인할 것.
 
Q: 한 파일에서 여러 weight가 같은걸 참조하고 싶을 때 쓰는 기술? (share하는법)


Q K^T 부터 값이 틀어지네, 이후 softmax해도 값이 잘 돌아오진 않음
(그 전은? 알아보자)
Q-Q_RoPE는 동일, K는 RoPE적용하면 달라지는데, UINT16의 오차인것 같음.

이정도 오차는 사실 당연한데...뒤쪽 레이어에 구현상 문제가 사실은 존재하여 그러는 것일수도 있고,
이전 layer도 중간에 오차가 있었을 수 있음.

cosine유사도? 같은거를 이용하여 검사해보라고 하심

RoPE를 끌 수 있겠네 생각해보니깐...끄고 한번 해보자
*준봉님의 메모리 잘못잡는패치도 확인

RoPE를 끄니 : 여전히 attn이후 값은 3~4자리 정도에서 틀림. 그래도 최종 결과도 norm전후로 4자리정도에서 틀림 (ㄱㅊ은 듯)

**아마 이거였던 듯 : RoPE가 sliding말고 global인 경우 rope_type이 "linear"로 다른데, (factor도 존재) 이게 반영이 안되었던 듯!






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
