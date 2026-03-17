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

**현재 유력후보 : RoPE가 sliding말고 global인 경우 rope_type이 "linear"로 다른데, (factor도 존재) 이게 반영이 안되었던 듯!

(0310 퇴근전) decoder 구조가 아직 안잡힘. decoder의 converter을 만들었는데, 
  [내일 출근후 할일]
  (decoder)converted weight == pytorch 실행중 weight 인지 먼저 확인하고,
  이후 NNTrainer에 제대로 loading되는지만 가볍게 확인하자
  그 다음에 decoder 의 constructModel을 다시 짜고
  merged attn구조를 만들어 보자

[바로가기]
model.encoder.text_model.norm

(0311 퇴근전)
Decoder의 cross attention의 from-to를 잘 잡기 위해 승희님의 PR을 이용하여 일단 work around를 하자
fc_layer자체의 from-to가 좀 이상하긴한데 확인해봐
=> 알아버렸다. 걍 맨앞에서부터 to-from개 만큼 들고 오는 거임!

그냥 layer의 input으로 들어온 얘들의 앞에서부터 (to-from)*WIDTH 를 들고 옴
실제 to부터 확인하는 거는 KV cache용이었다!

(0312 운동전)
승희님 custom from to들고옴. layer이름 검색 매커니즘 함 더 보고, 적용해서 어떻게 할지 생각해보자
(cache는 생각x)
승백님에게 왜 cache없어도 될거라고 생각하셨는지 물어보기
+k/v_proj layer의 weight sharing기능도 알아보기


(0312 퇴근전)
custom to를 썼는데, from을 바꾸지 않음
일단 decoder측은 height를 1로 고정함(concat했을 때 붙게) 이게 맞는지는 모르겠음.

이후 mha_core에서 from-to를 어떻게 해야할지(to를 custom으로 뒤로 늘려도 from은 계속 다가오고, cache관리는 어떻게하고...etc)
고민이 되어, 내일은 승희님 버전 말고 현석님 버전으로 한번 고쳐볼 것

승희님거 이름 map으로 줬는데 애초에 왜 못찾지...


[현석님 pr 이용 버전]
cache되는 FC_Layer을 만들거임(norm도 가능) + 현석님 restInput반영되었다 가정

1) cache_type prop을 one-time / incremental 두 종류 중 선택 가능하게 하여,
one_time이면 걍 한 번 하고, 크기 저장한 후 계속 cache 내보내기, (flag 이용하면 간단?)
incremental이면 들어오는대로 계속 cache쌓기

sliding생각하면 원형 cache? (window가 5면 5개 넘으면 맨 뒤에거 지우고 새로운거 더하고 이런식)도 생각은 해야 할듯 (사실 incremental의 경우 이거를 무조건 쓰게 하면 될 거 같은데?)

2) 근데 input/output size reset이 언제더라...아무튼 잘 되었다는 가정하에 가능

3) norm켜져있으면 norm하고 저장



[승희님 pr 이용 버전]

cache되는 FC_Layer을 만들거임(norm도 가능)
이때, input으로는 input중 맨 앞 to-from(사실 1개)을 들고 오게 하고,
norm켜져있으면 norm하고,
from 위치에 해당 input의 proj를 cache하면 그게 사실상 kv cache
대신, output의 크기가 0~to까지 전부 매번 보내줘야 함. 이는 애초에 input tensor을 지금처럼 MAX_SEQ로 잡으면 쉬울듯


encoder용 FC_LAYER도 동일한 구조인데, 승희님의 to-from고정으로 걍 고정된 ENC_SEQ_LEN값을 임시로 넘겨준다든가 이런식으로 구현 가능

이후 concat해서 넘겨주는데, mha_core입장에선 이제 이게 to + ENC_SEQ_LEN 만큼 계산해야 하 지 만?
걍 현석님의 PR 반영된 cross attn이 잘 된다면 상관 없음.


rope를 linear구현을 했는데, 차이가 남...

1) linear 구현이 틀림
2) RoPE 자체가 틀어짐을 만든다.
=> 생각해보니 걍 layer0부터 Rope하면 값이 달랐음. Rope자체에 문제가 있는 것.
...걍 기존 rope에 틀린 구현이 있는건가?

(0316 퇴근전) RoPE에서 default/linear을 번갈아 써야 하는데, 현재 매커니즘은 하나의 cos/sin/theta를 만들고, 있으면 그걸 재활용
그래서 default/linear 등 모드당 1개를 들도록 만들어야 하는데, 
CLINE / Codex를 이용해서 각각 코딩했지만, 안됨. 
현재 CLINE버전은 버렸고, 일단 Codex도 안되지만, Codex도 아직 print debug를 안했기에 내일 와서 하ㅣ자



rope의 값 만 출력해서 비교해 보자!




(0313 심야의 고민)
승희님 custom to + self attn으로 검증가능한 법

prefill 처럼해서 확인해보자(단계 1만 ㅇㅇ)

대신 이러려면 길이가 K,V = 1+ENC_SEQ_LEN 인데 Q=1임
Q에 0으로 padding을 하는게 가능한지 생각해보자




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
