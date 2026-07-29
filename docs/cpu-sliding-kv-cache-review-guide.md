# CPU Sliding-Attention KV Cache Review Guide

이 문서는 `feature/cpu-sliding-kv-cache` 브랜치의 코드 리뷰를 위한
진입점이다. 구현 과정과 정적 검증의 상세 기록은
[implementation log](cpu-sliding-kv-cache-implementation-log.md)에 있다.

## 1. 리뷰 대상

- 기준 upstream commit:
  `308bdd4f3b87a9dd5d5899104be32e6518efd637`
- 기능 구현 commit:
  `8d4afb4fbaa7936d670fed8212ce7c84a1b5a8e1`
- 기능 commit만 확인하는 명령:

```bash
git diff 308bdd4f3b87a9dd5d5899104be32e6518efd637 \
  8d4afb4fbaa7936d670fed8212ce7c84a1b5a8e1
```

코드 변경은 18개 파일이며 `subprojects/`는 건드리지 않았다. 이 문서
commit은 설명만 추가하고 기능 코드는 변경하지 않는다.

## 2. 목표 동작

causal sliding-attention layer의 K/V cache는 logical sequence limit인
`MAX_SEQ_LEN` 전체가 아니라 다음 크기만 물리적으로 보유한다.

```text
physical_capacity = min(MAX_SEQ_LEN, sliding_window)
```

다음 경우에는 기존 full capacity를 유지한다.

- full attention
- non-causal attention
- `sliding_window == 0`
- `sliding_window == UINT_MAX`
- sliding window가 `MAX_SEQ_LEN` 이상인 경우

logical position은 계속 absolute sequence position이며 최대
`MAX_SEQ_LEN`까지 증가한다. 이번 변경은 logical sequence limit을
늘리는 기능이 아니다.

## 3. 데이터 흐름

```text
model-specific attention window
        |
        v
Transformer::resolveKVCacheCapacity()
        |
        +--> graph cache placeholder height
        |
        +--> KV_CACHE_CAPACITIES[layer]
                    |
                    v
          KVCacheManager allocation
                    |
                    v
          mha_core external cache input
                    |
                    v
   chronological compact K/V rows on CPU
```

placeholder와 manager allocation은 같은 per-layer capacity vector를
사용한다. binding 시 dtype뿐 아니라 전체 shape도 비교한다.

## 4. 핵심 불변식

리뷰 시 아래 조건이 모든 경로에서 유지되는지 우선 확인한다.

1. `cache_pos_`, `cache_index`, RoPE position은 absolute logical position이다.
2. compact cache row는 항상 `[oldest ... newest]` 순서로 연속 배치된다.
3. wrap 이후 새 K/V는 마지막 physical row에 기록된다.
4. QK, softmax, V-cache 계산에는 physical position을 전달한다.
5. RoPE에는 physical position이 아니라 absolute position을 전달한다.
6. batch별 cache view의 batch dimension은 반드시 1이다.
7. 한 번의 prefill 호출에 대한 `skip_prefill` 판정은 token loop 도중
   바뀌지 않는다.
8. non-causal attention은 미래 K/V를 동시에 볼 수 있어야 하므로
   compact token-by-token 경로에 진입하지 않는다.
9. manager logical position과 각 `mha_core`의 `cache_index`가 prefill,
   generation, save/load 이후 동일하게 유지된다.
10. absolute position `p`에서 attention context는 현재 token을 포함한
    `min(p + 1, window)`개이며 K와 V는 항상 함께 shift된다.

## 5. 구현 지도

| 영역 | 주요 파일 | 검토할 내용 |
|---|---|---|
| Cache ownership | [`kv_cache_manager.cpp`](../Applications/CausalLM/kv_cache_manager.cpp), [`kv_cache_manager.h`](../Applications/CausalLM/kv_cache_manager.h) | per-layer capacity, bounds, logical position, persistence |
| CPU attention | [`mha_core.cpp`](../Applications/CausalLM/layers/mha_core.cpp), [`mha_core.h`](../Applications/CausalLM/layers/mha_core.h) | compact detection, shift, long prefill, RoPE/physical position |
| Base propagation | [`transformer.cpp`](../Applications/CausalLM/models/transformer.cpp), [`transformer.h`](../Applications/CausalLM/models/transformer.h) | capacity resolution/recording, base layer pattern |
| Host synchronization | [`causal_lm.cpp`](../Applications/CausalLM/models/causal_lm.cpp) | allocate/bind shape, position advance |
| Model overrides | `models/gemma3`, `gemma4`, `gpt_oss`, `gpt_oss_cached_slim`, `lfm2`, `qwen2`, `qwen3` | 실제 layer window 전달 여부 |
| Non-causal host | [`sentence_transformer.cpp`](../Applications/CausalLM/models/sentence_transformer.cpp) | full capacity 유지와 shape binding |
| Manager tests | [`unittest_kv_cache_manager.cpp`](../test/unittest/layers/unittest_kv_cache_manager.cpp) | capacity, bounds, persistence framing |
| Model test | [`unittest_causallm_gemma3.cpp`](../test/unittest/models/unittest_causallm_gemma3.cpp) | mixed sliding/full 및 long prefill |

## 6. CPU rolling 경로

`MHACoreLayer::usesCompactCache()`는 다음 조건을 모두 만족할 때만
compact 경로를 선택한다.

- causal attention
- finite positive sliding window
- window가 `max_timestep`보다 작음
- 실제 cache height가 sliding window와 같음

`compactCacheForwarding()`은 긴 prefill/chunk도 token 단위로 처리한다.
각 token의 위치는 다음처럼 나뉜다.

```text
absolute_position = call_from + token_index
physical_from = min(absolute_position, capacity - 1)
physical_to = physical_from + 1
```

`absolute_position >= capacity`이면 K/V의 각 batch row를 `memmove`로 한 칸
왼쪽 이동한 뒤 마지막 row를 덮어쓴다. 이 방식은 modulo ring보다
비용이 크지만 기존 CPU QK/V kernel이 요구하는 chronological contiguous
layout을 그대로 제공한다.

특히 다음 두 overload를 함께 확인해야 한다.

- sink가 없는 `one_batch_incremental_forwarding()`
- GPT-OSS가 사용하는 sink overload

두 경로 모두 shift, absolute RoPE, physical QK/softmax/V position 및
`skip_prefill` 순서가 같아야 한다.

## 7. 모델별 capacity 선택

| 모델 경로 | sliding/full 판정 |
|---|---|
| Base Transformer | `sliding_window_pattern` |
| Gemma3 | `layer_types`에서 계산한 실제 `window_size` |
| Gemma4 | `isSlidingAttentionLayer()`와 layer별 KV width |
| GPT-OSS / cached-slim | `LAYER_TYPES[layer_id]` |
| Qwen2 / Qwen3 | `SLIDING_WINDOW`; non-causal이면 resolver가 full로 복원 |
| LFM2 | `full_attention`은 `UINT_MAX`, 나머지 attention은 `SLIDING_WINDOW` |
| LFM2 conv-only layer | cache placeholder가 없어 binding은 건너뛰지만 manager의 미사용 full allocation은 남음 |

리뷰 시 placeholder에 전달한 window와 같은 `sliding_window`가
`mha_core` property에도 전달되는지 대조한다.

## 8. Persistence 호환 계약

기존 파일은 version/header가 없는 Tensor serialization stream이다. 기존
manager는 `(B, 1, seq_len, width)` shared slice를 offset 0에서 저장했기
때문에 `B > 1 && seq_len < max_seq_len`일 때 각 batch의 prefix가 아니라
full cache를 flatten한 첫 `B * seq_len` physical row가 기록된다. 이 한계는
기존 파일과의 byte-layout 호환을 위해 유지했다.

### Full-capacity layer

기존 shared Tensor slice의 `save/read`를 그대로 호출한다. 따라서
dtype-specific framing까지 upstream 동작과 동일하다.

### Compact layer

- retained logical rows를 같은 virtual full-cache 위치에 투영한다.
- 이미 eviction된 prefix는 zero bytes로 기록한다.
- load 전에 compact tensor를 zero로 초기화한다.
- legacy stream과 compact retained window가 겹치는 row만 읽는다.
- `UINT8/UINT16/UINT32`는 기존 2-byte quantization scheme header와
  trailing-memory segment 크기를 유지한다.
- legacy trailing bytes는 실제 metadata가 아닐 수 있어 checked
  read-and-discard하며 compact tensor의 초기 metadata는 보존한다.
- eviction 이후 과거 position은 복원할 수 없으므로 현재 logical
  position과 다른 `save(seq_len)` 요청을 거부한다.

compact persistence에서 부가 memory를 갖는 다른 quantized dtype은
명시적으로 거부한다. production cache 경로에서 사용하는 FP16,
UINT16과 테스트용 FP32는 지원된다.

## 9. 추가된 테스트

`KVCacheManagerTest`에는 다음 범주가 추가되었다.

- per-layer capacity 및 invalid count/value
- logical position과 physical capacity 분리
- layer/batch/view bounds
- legacy multi-batch flattened byte layout
- mixed full/compact save-load
- compact multi-batch intersection
- eviction 이후 unavailable save position 거부
- UINT16 legacy Tensor framing
- UINT16 legacy-to-compact save/reload

`Gemma3KVCacheTest.SlidingLayerKeepsOnlyWindowDuringLongPrefill`은
`MAX_SEQ_LEN=8`, window 4, prompt 길이 6을 사용한다. layer 0은 capacity
4, layer 1은 capacity 8인지 확인하고 generation이 예외 없이 두 output
ID를 반환하는지 확인한다. 실제 K/V row나 reference logit을 비교하는
수치 정확성 테스트는 아니다.

EmbeddingGemma test는 non-causal model이 sliding config를 가져도 두
layer 모두 full capacity 8을 유지하는지 확인한다.

## 10. 우선 리뷰 체크리스트

- [ ] `absolute_from + token`과 `cache_index + step_size`에 overflow 또는
      logical bound 우회가 없는가?
- [ ] capacity 1, `position == capacity - 1`, 첫 wrap, 반복 wrap이 안전한가?
- [ ] batch 2 이상에서 shift와 shared-tensor offset이 다른 batch를
      침범하지 않는가?
- [ ] FP32, Android FP16 conversion, native FP16 경로가 같은 position을
      사용하는가?
- [ ] `shift_cache_left()`의 raw row addressing이 지원하는 모든 Tensor
      format/channel layout에서 유효한가?
- [ ] sink/non-sink 및 internal/external cache 경로의 의미가 같은가?
- [ ] long prefill에서 모든 token의 K/V가 기록되고 마지막 window만
      남는가?
- [ ] `skip_prefill`이 K/V 기록은 유지하면서 attention output 계산만
      건너뛰는가?
- [ ] model별 placeholder shape와 manager shape가 항상 같은가?
- [ ] CausalLM save, normal prefill, skip-prefill, generation과 LFM2
      generation에서 manager position이 정확히 한 번만 증가하는가?
- [ ] load된 position이 window보다 클 때 첫 decode가 올바르게
      shift/write하는가?
- [ ] model별 manager `MAX_SEQ_LEN`과 `mha_core`의 `max_timestep`이
      다를 수 있는 경로에서도 logical bound가 일관적인가?
- [ ] finite sliding 설정인데 cache shape가 예상과 다르면
      `usesCompactCache()`가 noncompact path로 조용히 fallback하는 것이
      의도된 동작인가?
- [ ] legacy persistence의 K/V 및 layer segment 경계가 dtype별로
      유지되는가?
- [ ] non-causal 모델이 compact 경로에 들어갈 가능성이 없는가?
- [ ] 함께 변경된 noncompact batch-view, sink FP32 dispatch 및
      sink `skip_prefill` 동작에 회귀가 없는가?

## 11. 알려진 제한과 non-goal

- wrap token마다 `O(batch * capacity * kv_width)` `memmove`가 발생한다.
  이번 변경은 memory 절감을 우선하며 circular-kernel 최적화는 범위 밖이다.
- persistence file은 호환을 위해 virtual full-cache 크기를 유지하므로
  파일 크기는 줄지 않는다.
- legacy multi-batch partial-save layout의 기존 한계는 수정하지 않았다.
- manager의 contiguous write-view helper는 rolling을 수행하지 않는다.
  compact cache가 capacity를 넘은 뒤에는 예외를 내며 production
  `mha_core`가 cache tensor를 직접 rolling한다.
- LFM2 conv-only layer는 graph cache를 binding하지 않지만 manager에
  미사용 full-capacity K/V allocation이 남아 있다.
- 추가된 model test는 capacity와 no-throw smoke를 검증할 뿐 compact
  attention의 K/V content 또는 수치 출력을 reference와 비교하지 않는다.
- 로컬에서는 실제 build, link, test를 실행하지 않았다.

## 12. 원격 검증 제안

환경에 맞게 build directory와 Meson 옵션을 조정한 뒤 최소한 다음을
실행하는 것을 권장한다.

```bash
meson setup build -Denable-transformer=true -Denable-fp16=false
ninja -C build unittest_kv_cache_manager unittest_causallm_models
meson test -C build unittest_kv_cache_manager --print-errorlogs
meson test -C build unittest_causallm_models \
  --test-args='--gtest_filter=Gemma3KVCacheTest.*' \
  --print-errorlogs
```

추가 runtime matrix 권장 항목:

- prompt/decode position `W-1`, `W`, `W+1`, `2W+1`
- batch 2 이상
- GPT-OSS sink attention
- Android `ENABLE_FP16`
- precomputed cache load 후 첫 wrap
- full/sliding mixed layer output을 기존 full-storage 구현 또는 reference와
  비교
- known Q/K/V를 사용한 compact output과 full-storage sliding reference의
  수치 비교
- malformed/truncated persistence stream
