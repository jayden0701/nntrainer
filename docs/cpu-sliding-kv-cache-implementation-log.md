# CPU Sliding-Attention KV Cache Implementation Log

## 1. 요청과 기준점

- 목표: CPU causal sliding-attention layer가 `MAX_SEQ_LEN` 전체가 아니라
  sliding window만큼의 K/V cache만 물리적으로 유지하도록 구현
- 공식 upstream: `https://github.com/nntrainer/nntrainer.git`
- 기준 commit:
  `308bdd4f3b87a9dd5d5899104be32e6518efd637`
- 기능 branch: `feature/cpu-sliding-kv-cache`
- 기능 commit:
  `8d4afb4fbaa7936d670fed8212ce7c84a1b5a8e1`
- 실제 build와 runtime test는 원격에서 수행하기로 했으므로 로컬에서는
  정적 검증만 수행

## 2. 변경 전 구조

- `KVCacheManager`는 모든 layer에 동일한 `MAX_SEQ_LEN` 높이를 할당했다.
- base Transformer와 Gemma4 cache placeholder도 모두 full height였다.
- production external-cache 경로는 manager view helper가 아니라
  `mha_core.cpp`가 input 3/4의 cache tensor를 직접 slice한다.
- CPU attention kernel은 sliding window만 계산했지만 K/V storage는 full
  sequence였다.
- Gemma3, Gemma4, GPT-OSS처럼 sliding/full layer가 섞인 모델이 있어
  단일 global capacity로는 요구사항을 만족할 수 없었다.
- cache height만 줄이면 `step_size > window`인 prefill과 wrap 이후
  decode가 out-of-bounds가 된다.

## 3. 설계 결정

### Logical position과 physical capacity 분리

`KVCacheManager::cache_pos_`와 `mha_core::cache_index`는 계속 absolute
logical position을 나타낸다. 새 `cache_capacities_` vector만 layer별
physical tensor height를 나타낸다.

### Causal finite sliding layer만 축소

capacity resolver는 non-causal, zero window, `UINT_MAX` window를 full
capacity로 유지한다. non-causal prefill을 token 단위로 바꾸면 현재
query가 미래 K/V를 보지 못해 attention 의미가 바뀌므로 compact 경로에서
제외했다.

### Chronological shift 사용

모듈로 ring index를 CPU QK/V kernel 전반에 전파하는 대신 wrap 때 K/V를
한 row 왼쪽으로 이동한다. 계산 kernel이 기존 contiguous chronological
layout을 그대로 사용하므로 변경 범위와 semantic risk가 작다.

대가로 wrap token마다 window 크기에 비례하는 memory copy가 발생한다.

### Long prefill은 compact 경로에서 token 단위 처리

한 번의 prefill call을 token loop로 나누되 call-level `is_prefill` 값을
모든 token에 전달한다. 따라서 `skip_prefill` 의미가 token마다 decode로
바뀌지 않는다.

### Absolute RoPE와 physical cache index 분리

- K/Q RoPE: absolute sequence position
- cache write, QK, softmax, V aggregation: compact physical position

sink attention overload도 같은 규칙을 사용한다.

### Persistence는 virtual full layout 유지

파일 크기를 줄이는 새 format을 만들지 않고 기존 stream layout에 compact
cache를 투영한다. 이 선택은 기존 precomputed cache를 읽을 수 있게 하지만
저장 파일 크기는 줄이지 않는다.

## 4. 구현 내용

### KVCacheManager

- uniform KV width + per-layer capacity overload
- per-layer KV width + per-layer capacity overload
- `getCacheCapacity(layer_idx)`
- capacity count/value 검증
- layer, batch, read/write view bounds 검증
- overflow-safe `advance()`
- full cache와 compact cache가 섞인 allocation
- legacy/full/compact persistence 처리

write/read view helper는 contiguous view만 제공한다. compact cache가 wrap된
뒤에는 해당 helper가 rolling을 시도하지 않고 예외를 낸다.

### Transformer와 model propagation

`Transformer::setupParameters()`가 `KV_CACHE_CAPACITIES`를 full capacity로
초기화한다. graph construction 중 각 attention layer가 실제 window를
`createKVCachePlaceholders()`에 전달하면 resolver가 capacity를 계산하고
vector에 기록한다.

적용한 override:

- Gemma3
- Gemma4와 shared attention
- GPT-OSS
- GPT-OSS cached-slim
- Qwen2
- Qwen3
- LFM2

LFM2의 `full_attention`은 명시적으로 `UINT_MAX`를 전달한다. conv-only
layer는 cache placeholder가 없으므로 binding 단계에서 건너뛴다. 다만
manager의 layer별 기본값은 full capacity이므로 해당 layer의 미사용
allocation은 남아 있다.

### mha_core

- internal cache allocation도 causal finite window에서는 compact height 사용
- external/internal cache의 compact 여부를 tensor shape와 property로 판정
- batch별 K/V shift helper
- compact long prefill/chunk token loop
- FP32, Android FP32-to-FP16, native FP16 분기 유지
- sink/non-sink overload 모두 rolling 지원
- cached K/V shared view의 batch dimension을 1로 고정
- external forwarding에서 logical `max_timestep` range를 mutation 전에 검증
- external cache 사용 시 dynamic input-dimension update가 internal tensor
  index를 수정하지 않도록 보호

### Host position synchronization

base CausalLM에서 다음 호출 후 manager logical position을 증가시킨다.

- KV cache save-mode prefill
- normal prefill
- skip-prefill path
- token generation

LFM2 embedding path는 prefill에 이미 있던 advance를 유지하고 generation
advance를 추가했다.

## 5. Persistence 상세

### Legacy flattening

기존 구현은 cache tensor에서 `(B, 1, seq_len, width)` shared view를 offset
0으로 만들어 저장했다. 따라서 `B > 1 && seq_len < max_seq_len`이면 각
batch의 `seq_len` row가 아니라 full tensor의 첫 `B * seq_len` row가
저장된다.

이 동작은 이상적이지 않지만 기존 파일의 segment layout을 바꾸지 않기
위해 유지했다.

### Full-capacity layer

기존 shared slice의 `Tensor::save/read`를 그대로 사용한다.

### Compact layer

- `retained_len = min(seq_len, capacity)`
- `retained_from = seq_len - retained_len`
- retained row를 legacy virtual flattened range와 교차시켜 stream에 기록
- discard된 prefix는 zero로 기록
- load 전에 tensor를 zero로 초기화
- stream에 없는 batch/row는 zero 유지

eviction 전에는 현재 position 이하의 prefix를 저장할 수 있다. eviction
후에는 과거 prefix가 이미 없어졌으므로 `seq_len == cache_pos_`만 허용한다.

### UINT Tensor framing

최종 감사에서 non-`ENABLE_FP16` CPU cache가 `UINT16`을 사용하며 기존
`Tensor::save/read`가 각 K/V segment마다 다음을 기록한다는 점을
확인했다.

```text
2-byte qscheme + payload + scale/zero-point trailing memory
```

이에 따라:

- full layer는 기존 Tensor serialization을 그대로 사용한다.
- compact `UINT8/UINT16/UINT32` layer는 같은 header와 trailing segment
  크기를 유지한다.
- legacy partial slice의 trailing bytes는 실제 metadata가 아니라 다음
  raw cache row일 수 있으므로 compact load에서는 checked
  read-and-discard한다.
- compact tensor의 초기 scale/zero-point memory는 덮어쓰지 않는다.

## 6. 리뷰 중 발견해 수정한 항목

- batch별 cached K/V view의 batch dimension이 full batch로 남던 문제
- full-cache multi-batch persistence의 legacy byte layout 회귀
- compact eviction 이후 보유하지 않은 과거 position 저장
- compact load에서 stream에 없는 retained row가 stale 값으로 남는 문제
- base CausalLM manager position advance 누락
- LFM2 generation manager position advance 누락
- sink overload가 absolute position을 QK physical offset으로 사용하던 문제
- external forwarding range의 unsigned overflow 가능성
- non-FP16 `UINT16` Tensor framing 누락
- legacy trailing bytes로 compact quantization metadata가 오염될 가능성

## 7. 추가 테스트

Manager test:

- `allocate_per_layer_capacities`
- `logical_position_independent_of_physical_capacity`
- `allocate_invalid_cache_capacity_count`
- `allocate_invalid_cache_capacity_values`
- `invalid_batch_idx`
- `save_preserves_legacy_multibatch_layout`
- `uint16_save_preserves_legacy_tensor_framing`
- `uint16_compact_load_save_preserves_framing`
- `save_load_mixed_capacities_after_wrap`
- `compact_multibatch_uses_legacy_layout_intersection`
- `compact_save_rejects_unavailable_position`
- `compact_cache_view_rejects_logical_position`

Model test:

- `Gemma3KVCacheTest.SlidingLayerKeepsOnlyWindowDuringLongPrefill`
- 기존 EmbeddingGemma test에 non-causal full-capacity assertion 추가

Gemma3 test는 mixed capacity `4/8`, window보다 긴 prompt 및 generation
no-throw를 확인한다. 실제 compact K/V row나 reference logit을 비교하지
않으므로 수치 동등성은 원격 runtime 검증 항목으로 남아 있다.

## 8. 수행한 정적 검증

성공:

- clang-format 14를 변경 C/C++ 18개 파일에 적용
- `clang-format-14 --dry-run --Werror`
- `git diff --check`
- block-comment integrity 검사
- 변경 파일 집합 18개 확인
- `subprojects/` 무변경 확인
- `kv_cache_manager.cpp` syntax-only
- `unittest_kv_cache_manager.cpp` syntax-only
- `unittest_causallm_gemma3.cpp` syntax-only
- `transformer.cpp`, `lfm2_causallm.cpp`, `gemma4_causallm.cpp`
  syntax-only
- 전체 production callsite/declaration 대응 확인

제약:

- `mha_core.cpp` 전체 syntax-only는 upstream에도 존재하는 Windows
  `M_PI` 미정의와 `WIN_EXPORT class` warning-as-error 때문에 실패했다.
  compact-cache 변경 줄을 지목한 신규 진단은 없었다.
- `causal_lm.cpp` full-flags syntax-only는 제한 시간 내 완료되지 않았다.
- GPT-OSS cached-slim TU는 사용한 compile database에 entry가 없었다.
- 실제 build, link, unit test, model inference는 실행하지 않았다.

## 9. 리뷰에 남은 고위험 항목

- `shift_cache_left()`의 raw row/channel addressing이 지원 Tensor format
  전체에서 유효한지
- batch 2 이상에서 MHA의 실제 K/V shift가 batch별로 독립적인지
- known Q/K/V 기준으로 tokenized compact prefill과 기존 batched sliding
  attention의 출력이 수치적으로 같은지
- GPT-OSS sink, Android FP16 conversion, non-FP16 UINT16 runtime
- internal 3-input cache와 external 5-input cache 양쪽의 wrap 동작
- precomputed cache를 `position > window`에서 load한 직후 첫 decode
- Gemma4 shared KV와 heterogeneous width binding
- model별 manager `MAX_SEQ_LEN`과 `mha_core max_timestep` 차이
- shape heuristic 기반 compact 판정이 mismatch를 조용히 fallback하는 점
- 함께 수정된 noncompact batch view, sink dispatch 및 `skip_prefill` 회귀
- persistence malformed/truncated stream 및 실제 MHA roll 후 round trip
- `memmove`와 tokenized prefill의 성능 비용 및 실제 memory 감소량

## 10. 전달 상태

- 기능 commit에는 DCO sign-off와 agent co-author trailer가 포함되어 있다.
- commit:
  `8d4afb4fbaa7936d670fed8212ce7c84a1b5a8e1`
- branch:
  `origin/feature/cpu-sliding-kv-cache`
- 기능 commit 직후 local/remote SHA 일치를 확인했다.

리뷰는 먼저
[review guide](cpu-sliding-kv-cache-review-guide.md)의 우선 체크리스트를
따르는 것을 권장한다.
