
1. Processor
     * Image 존재 여부를 현재 iamges가 비었는지 확인하는데, model은 이걸 값으로 들고 있으니(init 시점에서 넘겨주자)
    1) TextProcessor - 완료
    2) ImageProcessor - 완료


Tokenizer은 Transformer이 들고 있으니, 여기까지만 하고 값을 모델쪽으로 돌아오자.


3월 1주차 출근시 할 것 : (주말에 가능하면 codex로 뼈대 만들어 놓자)

0) 모든 시나리오 (text only / image only / multiimage / text + multi image)에 대해 Processor이 TextProcessor/ImageProcessor잘하는지 확인. (확장text / processed image pixels / image mask)를 리턴해야 함.
-> processor을 바꿔서 input_prompt를 잘 처리하게 바꿈
- meson.build에서 processor test부분은 제거해야 함


1) t5gemma2.cpp 만들어서 뼈대 만들고(가능하면 주말 codex) Processor 불러서 text(아직 text, token만 추가됨)에 tokenzier 적용하자
-> Transformer::setupParameters() 가 현재 걍 주석처리임. 이후 수정 필
- checkImageInput()이 있으면 안됨. 그냥 nntr_config에서 image 처리 필요한지만 적어놓고 그걸로 모델 구조 만들자


2) encoder 구현
- MAX_SEQ_LEN어떻게 잡을지 생각해보자
- Encoder에는 KV cache를 꺼도 될 것 같음

3) decoder 구현
- weight경우 일단은 lm_head따로 쓰자 (tie-embedding nono). 나중에 맨 앞이 embedding아니라서 문제생길 수 있음.

- cache없는 버전 (매번 {encoder결과값, generated tokens} 2개를 input으로 주고, encoder 결과 값도 매번 새로 계산하는 버전)

- cached_fc를 만들어서, 사실 encoder쪽 embedding의 projection은 1회만 계산되고 추후 cache기반으로 pass하면 되기 때문에, 여기서 계산하고 각 layer에 뿌려주는 식으로 해보자
(이후 구현. 일단은 걍 무지성 no cache로 구현)



4) Enc - Dec 구조 고도화 (순서대로 ㄱㄱ)
- NUM TO GENERATE, MAX_SEQ_LEN 같은 거는 사실 encoder decoder마다 좀 달라야 함
- Encoder측에선 KV cache를 꺼도 될 듯
- EOS나 생성 완료 시점에 대한 logic을 causalLM에서 들고 와야 함
- generate잘 되는지 확인


5) 작동 순서 구현
- 대부분 CLINE 구현. 직접 쭉 보긴 해야 함


