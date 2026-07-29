# CPU Sliding Attention KV Cache 쉽게 이해하기

## 한 문장 요약

sliding attention layer가 실제로 참고하는 최근 token 수만큼만 CPU의
K/V cache를 보관하도록 바꿨다. 오래된 K/V는 버리고 새 K/V를 넣되,
token의 원래 순서는 계속 기억한다.

이 변경의 목표는 **모델의 동작 범위를 바꾸지 않고 CPU memory 낭비를
줄이는 것**이다. 실제 출력이 기존 구현과 같은지는 원격 runtime
검증으로 최종 확인해야 한다.

## 먼저 알아둘 세 가지

### Attention

모델이 현재 token을 처리할 때 다른 token 중 무엇을 참고할지 계산하는
기능이다. causal attention에서는 현재와 이전 token만 참고한다.

### KV cache

이미 처리한 token의 Key와 Value 계산 결과를 저장해 두는 공간이다. 매번
과거 token을 처음부터 다시 계산하지 않게 해 준다.

K와 V를 각각 저장하므로 cache buffer도 두 개다.

### Sliding attention

모든 과거 token을 보는 대신 최근 일정 범위만 보는 attention이다.

예를 들어 window가 4라면 현재 token은 최근 4개의 K/V만 사용한다.

| 종류 | 현재 token이 참고하는 범위 |
|---|---|
| Full causal attention | 지금까지의 모든 token |
| Sliding causal attention | 최근 `window`개 token |
| Non-causal attention | 현재 token의 뒤쪽까지 포함할 수 있음 |

## 기존에는 무엇이 아쉬웠나?

sliding attention은 계산할 때 최근 window만 사용하고 있었지만, CPU의
K/V cache는 전체 `MAX_SEQ_LEN` 크기로 잡혀 있었다.

예를 들어 다음과 같은 설정을 생각할 수 있다.

```text
MAX_SEQ_LEN = 4096
sliding_window = 512
```

이 layer가 실제로 보는 것은 최근 512개뿐인데도 4096개 분량의 K와 V를
모두 저장했다. sliding layer 하나만 놓고 보면 필요한 capacity의 8배를
할당한 셈이다.

정확한 byte 수는 batch, KV head 수, head dimension, dtype에 따라
달라지지만 기본 관계는 다음과 같다.

```text
KV cache memory
  = K와 V 두 개
  × batch
  × capacity
  × KV width
  × element byte 수
```

따라서 다른 조건이 같다면 capacity를 4096에서 512로 줄였을 때 해당
sliding layer의 K/V payload는 1/8이 된다. 고정 metadata와 allocator
overhead를 포함한 실제 process memory 비율은 조금 다를 수 있다. 모델
전체 절감량도 full attention layer가 얼마나 섞여 있는지에 따라
달라진다.

## 이제 어떻게 저장하나?

window가 4이고 token이 차례대로 들어온다고 하자. 아래 숫자는 설명을
쉽게 하기 위한 token 번호다.

처음 네 token까지는 빈 칸을 순서대로 채운다.

```text
token 1 입력: [1, _, _, _]
token 2 입력: [1, 2, _, _]
token 3 입력: [1, 2, 3, _]
token 4 입력: [1, 2, 3, 4]
```

다섯 번째 token이 오면 가장 오래된 1을 버리고 나머지를 왼쪽으로 한 칸
옮긴 뒤, 마지막 칸에 5를 쓴다.

```text
token 5 입력: [2, 3, 4, 5]
token 6 입력: [3, 4, 5, 6]
token 7 입력: [4, 5, 6, 7]
```

K와 V buffer를 항상 함께 이렇게 이동한다. 그래서 cache 안에는 언제나
`[가장 오래된 값 ... 가장 최신 값]` 순서가 유지된다.

이 문서에서 편의상 rolling buffer라고 부르지만, 구현은 index만 돌려
쓰는 전형적인 원형 ring buffer가 아니다. overflow 때 `memmove`로 실제
여러 row 전체를 한 row 위치만큼 왼쪽으로 옮기는 방식이다.

## 왜 굳이 data를 옮기나?

진짜 ring buffer라면 data 이동은 줄일 수 있지만, 물리적인 시작 위치가
계속 바뀐다. 그러면 기존 CPU attention kernel 여러 곳에 wrap index를
이해하는 코드를 추가해야 한다.

이번 구현은 기존 kernel이 기대하는 연속된 시간 순서를 그대로 유지하는
쪽을 선택했다.

```text
항상 cache[0] = 현재 보유한 값 중 가장 오래된 K/V
항상 cache[used_len - 1] = 가장 최근 K/V
window가 다 찬 뒤에는 cache[last] = 가장 최근 K/V
```

장점은 기존 계산 경로와 연결하기 단순하다는 것이다. 단점은 window가
찬 뒤 매 token마다 data 이동 비용이 생긴다는 것이다.

## “token의 원래 위치”와 “cache 안 위치”는 다르다

이 변경에서 가장 중요한 부분이다.

code가 0부터 위치를 센다고 하고 window가 4라고 하자.

```text
absolute position 0 -> physical cache row 0
absolute position 1 -> physical cache row 1
absolute position 2 -> physical cache row 2
absolute position 3 -> physical cache row 3

absolute position 4:
  기존 row 1, 2, 3을 row 0, 1, 2로 이동
  새 값을 physical row 3에 기록
```

absolute position 4의 token이 cache에서는 row 3에 들어가더라도, 모델이
그 token을 “문장 전체의 4번 위치”라고 아는 사실은 바뀌면 안 된다.

그래서 두 종류의 위치를 분리했다.

- RoPE에는 원래 문장 위치인 absolute position을 사용한다.
- cache write와 attention 계산에는 compact cache의 physical position을
  사용한다.

둘을 섞으면 window가 한 번 찬 뒤부터 position encoding이나 attention
범위가 잘못될 수 있다.

## 어떤 layer의 cache가 줄어드나?

다음 조건을 모두 만족하는 layer만 compact cache를 쓴다.

- causal attention이다.
- sliding window가 0보다 크다.
- sliding window가 유한하다.
- sliding window가 `MAX_SEQ_LEN`보다 작다.

다음 경우에는 기존처럼 full capacity를 유지한다.

- full attention layer
- non-causal attention layer
- window가 0이거나 `UINT_MAX`인 경우
- window가 `MAX_SEQ_LEN` 이상인 경우

Gemma3, Gemma4, GPT-OSS처럼 sliding layer와 full layer가 섞인 모델은
layer마다 다른 capacity를 가진다. 예를 들어 어떤 layer는 512만
보관하고 다음 layer는 4096 전체를 보관할 수 있다.

Embedding model처럼 non-causal인 경우에는 설정에 sliding window가
있더라도 full cache를 유지한다. non-causal attention을 token 단위로
잘라 처리하면 미래 token을 보지 못해 원래 의미가 달라질 수 있기
때문이다.

## 긴 prompt도 처리할 수 있나?

window보다 긴 prompt가 한 번에 들어오는 경우가 있다. 작은 cache에 긴
prompt 전체를 바로 쓰면 범위를 넘게 된다.

compact 경로에서는 긴 prompt를 내부적으로 token 단위로 처리한다.

```text
prompt token 하나 처리
-> 필요하면 K/V shift
-> 새 K/V 기록
-> 다음 prompt token 처리
```

따라서 prompt 길이가 window보다 커도 마지막 window만 남기며 진행한다.
이 경로가 기존 batched sliding attention과 수치적으로 같은지는 원격
runtime test에서 반드시 확인해야 한다.

## 전체 연결은 어떻게 맞췄나?

cache 크기를 정하는 곳과 실제 사용하는 곳이 다르면 shape mismatch나
out-of-bounds가 생길 수 있다. 이번 변경은 다음 흐름을 하나로 맞췄다.

```text
각 model이 layer의 실제 attention window를 알림
                    |
                    v
Transformer가 layer별 physical capacity를 계산
                    |
          +---------+---------+
          |                   |
          v                   v
graph cache placeholder   KVCacheManager allocation
          |                   |
          +---------+---------+
                    |
       dtype과 전체 shape 확인 후 binding
                    |
                    v
           CPU mha_core가 rolling
```

`KVCacheManager`가 기억하는 현재 위치는 physical row가 아니라 전체
문장에서의 logical absolute position이다. 각 MHA layer가 기억하는
position도 prefill, generation, save/load 후 같은 값이 되도록
동기화했다.

## Cache 파일 저장과 불러오기는?

memory 안의 cache는 작아졌지만 지원되는 cache dtype에서 기존 cache
file과의 호환성을 유지하기 위해 file format과 가상 layout은 그대로
두었다.

compact cache에 이미 없는 오래된 prefix는 저장할 때 zero로 채운다.
batch가 1이면 현재 보유한 최근 K/V를 기존 full-cache file의 논리적 tail
위치에 기록한다. multi-batch partial-save에서는 기존 flattened layout이
표현할 수 있는 범위와 compact cache가 보유한 범위의 교집합만 기록한다.

따라서 중요한 차이는 다음과 같다.

- runtime K/V memory는 줄어든다.
- 같은 `seq_len`을 저장할 때 cache file 크기는 줄어들지 않는다.
- 지원되는 dtype의 기존 header와 trailing data framing을 유지한다.
- 이미 eviction된 과거 K/V를 다시 만들어 낼 수는 없다.

기존 multi-batch partial-save format에는 모든 batch의 prefix를 자연스럽게
표현하지 못하는 제약이 있었다. 이번 변경은 새 versioned format을
만들지 않고 기존 byte layout을 보존했다.

production cache에서 쓰는 FP16/UINT16과 test에서 쓰는 FP32는 compact
persistence가 지원한다. 별도 metadata memory를 갖는 그 밖의 일부 dtype은
잘못된 호환을 가장하지 않고 명시적으로 거부한다.

## 안전을 위해 추가한 확인

- layer별 capacity 개수와 값이 올바른지 검사한다.
- layer, batch, read/write 범위를 검사한다.
- logical position이 `MAX_SEQ_LEN`을 넘지 않게 검사한다.
- graph placeholder와 manager cache의 dtype 및 전체 shape를 비교한다.
- K와 V cache의 높이가 서로 같은지 확인한다.
- load 후 manager와 MHA의 absolute position을 맞춘다.
- compact cache에서 이미 버린 과거 position을 저장하려 하면 거부한다.

## 이 방식의 trade-off

### 좋아진 점

- causal sliding layer의 CPU K/V memory가 window 크기에 비례한다.
- sliding/full이 섞인 모델도 layer별로 필요한 만큼만 할당한다.
- 기존 CPU kernel이 기대하는 연속된 chronological layout을 유지한다.
- 지원되는 cache dtype에서 기존 serialization framing과 호환된다.

### 비용 또는 남은 한계

- window가 찬 뒤 token마다 K/V `memmove` 비용이 발생한다.
- 긴 prompt는 compact layer에서 token 단위로 처리하므로 성능 확인이
  필요하다.
- memory는 줄지만 cache file 크기는 줄지 않는다.
- logical sequence limit인 `MAX_SEQ_LEN` 자체가 늘어나는 것은 아니다.
- LFM2의 conv-only layer에는 사용하지 않는 full manager allocation이
  아직 남아 있다.
- compact 여부를 property와 cache shape가 맞는지 보고 판단하므로,
  예상하지 못한 shape mismatch가 생기면 조용히 noncompact 경로로 갈
  가능성을 리뷰해야 한다.

## 지금까지 무엇을 검증했나?

로컬에서는 요청 범위에 맞춰 정적 검증만 수행했다.

- 변경한 C/C++ 18개 파일의 clang-format 14 확인
- whitespace와 diff 정적 검사
- manager와 관련 test의 syntax-only 검사
- Transformer, LFM2, Gemma4 등 일부 production translation unit의
  syntax-only 검사
- 선언과 production callsite 대응 확인

`mha_core.cpp` 전체 translation unit은 upstream에도 있는 Windows
`M_PI`와 export 경고 문제 때문에 clean syntax-only pass를 얻지 못했다.
compact-cache 변경 줄을 직접 지목한 새 진단은 없었지만, 이것은 실제
build 성공을 대신하지 않는다.

test code에는 다음 항목을 추가했다.

- layer별 compact/full capacity
- logical position과 physical capacity의 분리
- 범위 오류
- wrap 이후 mixed-capacity save/load
- legacy multi-batch layout
- UINT16 serialization framing
- window보다 긴 Gemma3 prompt
- non-causal EmbeddingGemma의 full capacity

다만 실제 build, link, unit test, model inference는 아직 실행하지 않았다.
Gemma3 test도 capacity와 no-throw를 확인하는 smoke test이며 실제 K/V
내용이나 reference logit을 비교하지 않는다.

## 원격 검증에서 가장 중요한 세 가지

1. **정확성:** 같은 입력에서 compact 구현의 결과가 기존 full-storage
   sliding attention 또는 reference와 수치적으로 같은가?
2. **memory:** 실제 process memory와 layer별 tensor shape가 기대만큼
   줄었는가?
3. **성능:** 매 token `memmove`와 tokenized long prefill 비용이 허용
   가능한가?

그다음에는 window 경계, batch 2 이상, GPT-OSS sink, FP16 Android,
UINT16 CPU, cache load 직후 decode를 중점적으로 확인하는 것이 좋다.

## 관련 문서와 commit

- 기능 commit:
  `8d4afb4fbaa7936d670fed8212ce7c84a1b5a8e1`
- reviewer용 요약:
  [CPU sliding KV-cache review guide](cpu-sliding-kv-cache-review-guide.md)
- 상세 구현 기록:
  [CPU sliding KV-cache implementation log](cpu-sliding-kv-cache-implementation-log.md)
