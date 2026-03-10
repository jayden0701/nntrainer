
1. Processor
     * Image 존재 여부를 현재 iamges가 비었는지 확인하는데, model은 이걸 값으로 들고 있으니(init 시점에서 넘겨주자)
    1) TextProcessor - 완료
    2) ImageProcessor - 완료


Tokenizer은 Transformer이 들고 있으니, 여기까지만 하고 값을 모델쪽으로 돌아오자.


3월 1주차 출근시 할 것 : (주말에 가능하면 codex로 뼈대 만들어 놓자)

0) 모든 시나리오 (text only / image only / multiimage / text + multi image)에 대해 Processor이 TextProcessor/ImageProcessor잘하는지 확인. (확장text / processed image pixels / image mask)를 리턴해야 함.
-> processor을 바꿔서 input_prompt를 잘 처리하게 바꿈

-> 이 시점에서 PR(DRAFT)만들고 Processor 만들고 올리기 (확장된 거만 올림)


1) t5gemma2.cpp 만들어서 뼈대 만들고(가능하면 주말 codex) Processor 불러서 text(아직 text, token만 추가됨)에 tokenzier 적용하자
-> Transformer::setupParameters() 가 현재 걍 주석처리임. 이후 수정 필
-> T5Gemma2::setupParameters() 에서 Config 값 저장하는 거 제대로 구현 안됨. Encoder쪽은 완료. Decoder수정필요

init()까지 되려면 

-> PR에 commit

2) encoder 구현
- 현재 RoPE 관련 에러인걸로 추측

3) decoder 구현
- config값 받아와야 함
- 일단은 1개 그래프에서 받아오는 구조로 ㄱㄱ

4) Enc - Dec 구조 고도화 (순서대로 ㄱㄱ)
- 방식1 : 그래프 1개, 


5) 작동 순서 구현
- run()에서 처음에 encoder넣고 decoder부르고...이런게 아직 안짜짐
- run()이 걍 CLINE으로 짜짐. 추가적으로 확인해봐야 함.


