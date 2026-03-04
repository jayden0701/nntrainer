
1. Processor
     * Image 존재 여부를 현재 iamges가 비었는지 확인하는데, model은 이걸 값으로 들고 있으니(init 시점에서 넘겨주자)
    1) TextProcessor - 완료
    2) ImageProcessor - 완료


Tokenizer은 Transformer이 들고 있으니, 여기까지만 하고 값을 모델쪽으로 돌아오자.


3월 1주차 출근시 할 것 : (주말에 가능하면 codex로 뼈대 만들어 놓자)

0) 모든 시나리오 (text only / image only / multiimage / text + multi image)에 대해 Processor이 TextProcessor/ImageProcessor잘하는지 확인. (확장text / processed image pixels / image mask)를 리턴해야 함.
-> processor을 바꿔서 input_prompt를 잘 처리하게 바꿈

-> 이 시점에서 PR(DRAFT)만들고 Processor 만들고 올리기


1) t5gemma2.cpp 만들어서 뼈대 만들고(가능하면 주말 codex) Processor 불러서 text(아직 text, token만 추가됨)에 tokenzier 적용하자
- 현재 processor config가 processor에 struct로 정의되어 있는데, 이도 전체 T5 config 호출 시 부를 수 있게 변경해 보자.
-> Transformer::setupParameters() 가 현재 걍 주석처리임. 이후 수정 필
-> T5Gemma2::setupParameters() 에서 Config 값 저장하는 거 제대로 구현 안됨. 이는 모델 만들면서 필요한 거 있으면 다 추가해 보자

init()까지 되려면 

-> PR에 commit

2) tokenize 된거 확인하면 text쪽 encoder를 미리 다 만들어 놓자
...
완성시 PR에 commit

3) 


