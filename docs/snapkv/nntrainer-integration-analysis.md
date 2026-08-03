# SnapKV의 nntrainer CPU CausalLM 통합 경로 분석

> 조사 기준: `main` 브랜치, 2026-08-03 워크트리. 이 문서는 현재 코드의
> KV-cache 생성/갱신/소비 경로와 SnapKV 삽입 후보를 정적 분석한 결과다.
> `subprojects/`는 조사·수정 대상에서 제외했다.

## 1. 결론 요약

현재 CausalLM의 실제 CPU 추론 KV-cache 경로는
`Applications/CausalLM/kv_cache_manager.*`와
`Applications/CausalLM/layers/mha_core.*`에 집중되어 있다.

- `KVCacheManager`가 레이어별 K/V 저장소를
  `[batch, 1, max_seq_len, num_kv_heads * head_dim]`으로 소유한다.
- 각 모델이 `cache_k_l<N>`, `cache_v_l<N>` 입력 placeholder를 만들고,
  `CausalLM::allocateAndBindKVCache()`가 manager의 메모리를 placeholder에
  바인딩한다.
- prefill 및 decode 때 `MHACoreLayer`가 외부 cache에 rotary-embedded K와 V를
  직접 쓰고, 동일 함수 안에서 QK score를 softmax한 뒤 V cache를 소비한다.
- SnapKV가 필요로 하는 **prefill의 observation-window attention probability**는
  `MHACoreLayer::one_batch_incremental_forwarding()` 안의 `out_`에만 존재한다.
  특히 `softmax_triangle()` 다음, value-cache 곱 이전/직후가 유일하게 자연스러운
  선택 점수 추출 위치다.

따라서 정확한 SnapKV를 manager-only 변경으로 구현할 수는 없다. manager에는
attention score가 없기 때문이다. 최소 침습적인 첫 구현 후보는 다음과 같다.

1. `mha_core`에서 prefill attention probability로 레이어/배치/KV-head별 보존
   인덱스를 계산한다.
2. attention output 계산이 끝난 직후 해당 K/V head slice를 cache 앞쪽으로
   stable gather한다.
3. 현재 하나의 `cache_index`가 겸하고 있는 **논리적 토큰 위치(RoPE 위치)**와
   **물리적 cache 유효 길이/쓰기 슬롯**을 반드시 분리한다.
4. 기본값은 완전 비활성화하여 기존 경로를 byte-identical하게 유지한다.

단, 현재 placeholder와 manager가 늘 `MAX_SEQ_LEN` 전체를 할당하므로, 단순 in-place
eviction만 구현하면 decode 연산량과 논리적 cache 길이는 줄어도 실제 예약 메모리는
줄지 않는다. 지속 메모리까지 줄이는 완성형은 별도의 prefill 임시 cache 또는
작은 external cache 설계가 필요하다. 이 차이를 명세와 테스트에서 숨기면 안 된다.

## 2. 현재 실행 경로

### 2.1 구성에서 graph 생성까지

1. `Transformer::setupParameters()`가 `nntr_config.json`에서
   `INIT_SEQ_LEN`, `MAX_SEQ_LEN`, `NUM_TO_GENERATE`를 읽고
   (`Applications/CausalLM/models/transformer.cpp:129-146`), 모델 config에서
   Q/KV head 수, head dim, sliding window, max position을 읽는다
   (`transformer.cpp:162-199`). 대응 상태는
   `transformer.h:361-391`에 저장된다.
2. `Transformer::createAttention()`은 Q/K/V projection을 만든다
   (`transformer.cpp:478-504`). K/V width는
   `head_dim * n_heads / GQA_SIZE`, 즉 `num_kv_heads * head_dim`이다.
3. `createKVCachePlaceholders()`가 레이어별 `cache_k_l<N>`와
   `cache_v_l<N>` 입력 layer를 만든다 (`transformer.cpp:440-472`). shape는
   `BATCH_SIZE:1:MAX_SEQ_LEN:kv_width`이며, `ENABLE_FP16`이면 `FP16`, 아니면
   `UINT16`이다.
4. Q/K/V와 두 cache 입력이 5-input `mha_core`에 연결된다
   (`transformer.cpp:510-523`). 5-input이라는 사실로 `mha_core`가 external cache
   mode를 선택한다 (`mha_core.cpp:135-145`).
5. custom `mha_core`는 CPU app context에 등록된다
   (`transformer.cpp:575-587`). 따라서 이 문서의 주 경로는 일반
   `nntrainer/layers/multi_head_attention_layer.*`가 아니라 CausalLM 전용
   `Applications/CausalLM/layers/mha_core.*`다.

### 2.2 cache 할당과 graph 입력 바인딩

1. `CausalLM::run()`은 추론 시작 시 `allocateAndBindKVCache()`를 호출한다
   (`causal_lm.cpp:362-380`).
2. 일반 CausalLM은 모든 레이어에 동일한
   `NUM_KEY_VALUE_HEADS * HEAD_DIM` width로 manager를 할당한다
   (`causal_lm.cpp:136-152`). 실제 Tensor 생성은
   `kv_cache_manager.cpp:37-71`이다.
3. cache dtype은 desktop FP32-activation build에서도 기본적으로 `UINT16`이다.
   이는 FP16 bit-pattern 저장용이며, `ENABLE_FP16` build만 Tensor dtype 자체가
   `FP16`이다 (`causal_lm.cpp:138-144`, `mha_core.cpp:229-246`).
4. 모델은 먼저 `layer<N>_attention:input3/input4`를 찾고, 실패하면
   `cache_k_l<N>`/`cache_v_l<N>`의 여러 suffix를 찾는다
   (`causal_lm.cpp:164-193`). 찾은 graph Tensor의 backing memory를 manager Tensor로
   바꾼다 (`causal_lm.cpp:194-201`).
5. 매 inference 호출의 raw input vector에도 cache pointer를 이름순으로 넣는다
   (`causal_lm.cpp:489-512`). `NeuralNetwork::incremental_inference()`는 graph의
   전체 input dimension으로 raw pointer를 external Tensor에 매핑한다
   (`nntrainer/models/neuralnet.cpp:1557-1572`).
6. input layer는 input/output 실제 포인터가 다를 때만 전체 Tensor를 copy한다
   (`nntrainer/layers/input_layer.cpp:45-68`). graph는 external placeholder output을
   직접 바인딩하는 용도를 위해 input in-place 동작을 유지한다
   (`nntrainer/graph/network_graph.cpp:696-703`). 이 때문에 cache allocation 크기를
   placeholder 선언 크기보다 작게 만드는 것은 별도 검증 없이 안전하다고 볼 수 없다.

### 2.3 prefill과 decode 위치 전파

- prefill 시작 위치는 `SYS_PROMP_LEN + global_token_len`이다
  (`causal_lm.cpp:559-565`).
- `setKVCachePosition(prefill_from)`은 manager의 `cache_pos_`와 모든
  `mha_core.cache_index`를 같은 절대 위치로 맞춘다
  (`causal_lm.cpp:207-215`, `mha_core.cpp:1552-1567`).
- prefill은 `[prefill_from, prefill_to)` 범위의 한 번의
  `incremental_inference()`로 실행된다 (`causal_lm.cpp:568-590`).
- decode는 보통 한 토큰 범위로 반복 호출된다
  (`causal_lm.cpp:620-629`). external mode의
  `MHACoreLayer::incremental_forwarding()`은 호출 때마다 `_from`을
  `cache_index`에 다시 대입한다 (`mha_core.cpp:406-417`). 즉 현재 구현은 호출자가
  주는 절대 토큰 위치와 물리 cache 위치가 같다는 전제를 갖는다.
- 각 attention layer는 forward 끝에서 자체 `cache_index`를 step size만큼
  증가시킨다 (`mha_core.cpp:398`, legacy path는 `:562-564`).

주의: 일반 `CausalLM::run()`은 manager의 `advanceKVCachePosition()`을 prefill/decode
후 호출하지 않는다. 이 메서드는 현재 LFM2의 embedding 입력 경로에서 prefill 뒤에만
사용된다 (`lfm2_causallm.cpp:565-574`). 따라서 일반 추론 중
`KVCacheManager::getPosition()`은 authoritative한 유효 길이가 아니다.

## 3. `mha_core` 내부 Tensor와 CPU 소비 경로

### 3.1 shape 및 layout

| Tensor | 논리 shape (NCHW) | 저장/소비 의미 |
|---|---|---|
| Q | `[B, 1, step, Hq * D]` | head-major feature, activation dtype |
| K, V input | `[B, 1, step, Hkv * D]` | projection 결과 |
| K/V cache | `[B, 1, capacity, Hkv * D]` | row마다 KV head가 interleave됨 |
| QK `out_` | `[1, 1, flattened_query_key_pairs, Hq]` | key-position-major, Q-head interleave |
| attention output | `[B, 1, step, Hq * D]` | 이후 O projection 입력 |

`finalize()`가 `D = query_width / Hq`를 계산하고 K의 head dim과 같은지 확인한다
(`mha_core.cpp:176-198`). external cache가 아니면 내부 cache를
`[B,1,max_timestep,Hkv*D]`로 요청한다 (`mha_core.cpp:229-253`). external cache에서는
input[3]/input[4]가 cache다 (`mha_core.cpp:294-300`).

CPU cache 주소 공식은 다음과 같다.

```text
element(row, kv_head, d)
  = base + (row * Hkv + kv_head) * D + d
```

이는 CausalLM의 FP32 reference kernel (`mha_core.cpp:42-102`)과 x86 UINT16/FP16-bit
kernel (`nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp:1610-1687`,
`nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp:1691-1727`)에서 동일하다. 따라서 KV
head마다 서로 다른 원 토큰을 선택해도,
각 물리 row의 head slice에 선택 결과를 gather하면 기존 QK/V kernel layout을 바꾸지
않고 사용할 수 있다.

### 3.2 한 batch의 append와 attention

일반(no sink) 경로는 `mha_core.cpp:693-782`다.

1. `cache_index * cache.width()` offset으로 새 K/V write view를 만든다
   (`:713-722`).
2. K를 절대 `cache_index` 위치의 RoPE로 회전해 cache에 쓰고, V는 변환/복사한다
   (`:724-738`).
3. query에도 같은 위치 기준 RoPE를 적용한다 (`:740-749`).
4. cache `[0, cache_index + step)` view를 만든다 (`:751-763`).
5. QK score용 `out_`를 만들고 `compute_kcaches()`를 호출한다 (`:765-775`).
6. `softmax_triangle()`이 `out_`를 attention probability로 in-place 변환한다
   (`:777`, 구현 `:1176-1273`).
7. 해당 probability와 V cache를 곱한다 (`:779-781`, 구현 `:1381-1495`).

attention sink가 있는 GPT-OSS 계열은 거의 같은 별도 경로
(`mha_core.cpp:784-863`)를 사용한다. sink-aware softmax는 `:1275-1379`다.

SnapKV 점수는 6번 뒤의 `out_`에서 추출해야 한다. V cache를 gather하기 전에 7번의
기존 attention output을 먼저 계산해야 하므로, 실제 cache compaction은 7번 뒤가
안전하다. score 추출 helper는 6번 뒤 호출해도 되지만 선택 인덱스만 보관하고 실제
K/V 이동은 7번 뒤 수행해야 한다.

### 3.3 CPU dtype 분기

- activation FP32 + cache FP32: C++ reference kernel
  (`mha_core.cpp:597-604`, `:631-635`).
- activation FP32 + cache UINT16/FP16 bit storage: CPU backend kernel
  (`mha_core.cpp:605-613`, `:636-641`). 기본 desktop CPU CausalLM이 이 경로다.
- activation FP16: `ENABLE_FP16`에서만 컴파일되는 분기
  (`mha_core.cpp:645-689`, `:1446-1493`).
- Android에서 activation이 FP32여도 `ENABLE_FP16`이면 임시 FP16 Q/K/V/O로
  변환하는 별도 분기가 있다 (`mha_core.cpp:345-389`).

첫 CPU 구현의 cache gather는 float 연산을 할 필요가 없다. Tensor dtype의 element
size만 사용해 K/V bit pattern을 그대로 이동해야 FP32, UINT16, FP16을 같은 코드로
지원할 수 있다. 선택 점수만 `out_` dtype을 float로 읽거나 명시적으로 변환한다.

## 4. 파일별 line-level 역할 지도

### 4.1 `KVCacheManager`

| 위치 | 현재 역할 | SnapKV 관련 의미 |
|---|---|---|
| `kv_cache_manager.h:27-45` | 외부 cache 관리자 책임과 eviction future extension 선언 | 정책 소유 후보임을 명시하지만 현재 score 입력은 없음 |
| `:67-86` / `.cpp:19-71` | 균일/레이어별 width 할당 | Gemma4는 레이어별 width overload 사용 |
| `:94-113` / `.cpp:74-90` | 단일 전역 `cache_pos_` 관리 | 레이어별 eviction 및 logical/physical 분리에 부족 |
| `:120-174` / `.cpp:92-187` | 원본 Tensor와 read/write view 제공 | view offset은 element 단위이며 batch stride는 full capacity |
| `:180-194` / `.cpp:189-249` | raw Tensor save/load | 압축 cache metadata가 없고 logical/physical 길이를 구분 못 함 |
| `:222-238` | `[B,1,max_seq,kv_width]` 저장 및 메타데이터 | 레이어별 valid length/head geometry 없음 |

정적 분석에서 확인한 기존 제약도 구현 전에 처리해야 한다.

- view 함수는 layer와 capacity는 검사하지만 `batch < batch_size_`는 검사하지 않는다
  (`kv_cache_manager.cpp:106-186`). 새 compaction API는 반드시 batch bounds를
  검사해야 한다.
- per-layer-width overload는 `num_heads_kv_ = head_dim_ = 0`으로 두지만,
  `getKVWidth()`는 `kv_width_`가 아니라 두 값의 곱을 반환한다
  (`kv_cache_manager.h:213-216`, `.cpp:55-60`). 따라서 Gemma4에서 이 accessor는 0을
  반환한다. SnapKV는 이 값을 head geometry로 사용하면 안 된다.
- save/load는 batch 전체 dim의 `height`만 줄인 contiguous view를 만든다
  (`kv_cache_manager.cpp:207-218`, `:235-246`). 원 저장소의 batch stride는
  `max_seq_len * kv_width`인데 slice Tensor는 `seq_len * kv_width` stride를 가정하므로
  batch > 1에서 올바른 batch별 prefix를 직렬화하지 않는다.

### 4.2 `Transformer` / `CausalLM`

| 위치 | 현재 역할 | SnapKV 삽입 후보 |
|---|---|---|
| `transformer.cpp:129-199` | JSON config 파싱 | runtime `snapkv` object 파싱 위치 |
| `transformer.h:361-391` | graph 구성에 쓰는 모델 상태 | enable/capacity/window/kernel 기본값 저장 후보 |
| `transformer.cpp:440-472` | 일반 cache placeholder 생성 | 실제 작은 persistent cache를 원할 때 capacity 변경 지점 |
| `transformer.cpp:478-523` | 기본 attention graph | MHA property 전달 지점 |
| `causal_lm.cpp:76-110` | CausalLM runtime/system cache 설정 | precomputed cache와 SnapKV 조합 검증 위치 |
| `causal_lm.cpp:136-205` | manager 할당 및 graph binding | manager capacity/state 구성 위치 |
| `causal_lm.cpp:207-220` | 모든 MHA position 동기화 | logical position과 per-layer physical length 분리 필요 |
| `causal_lm.cpp:524-590` | system cache save/load 및 prefill | prefill 후 eviction 완료/sync hook 후보 |
| `causal_lm.cpp:620-677` | decode 및 multi-turn 길이 누적 | 압축 후 logical position 보존 검증 핵심 |

### 4.3 모델 변형

모든 CausalLM이 base `createAttention()`을 쓰지는 않는다. MHA property를 graph 생성
시 전달하는 설계를 택하면 아래 경로를 빠짐없이 갱신해야 한다.

| 모델 계열 | attention/cache 연결 위치 | 특이점 |
|---|---|---|
| base Transformer | `transformer.cpp:478-523` | 표준 GQA |
| Qwen2 | `models/qwen2/qwen2_causallm.cpp:22-66` | 표준 external cache |
| Qwen3 및 MoE/Slim 파생 | `models/qwen3/qwen3_causallm.cpp:33-93` | Q/K norm 후 MHA; 파생 모델이 이를 상속 |
| GPT-OSS | `models/gpt_oss/gptoss_causallm.cpp:34-86` | attention sink, 레이어별 sliding/full |
| GPT-OSS cached-slim | `models/gpt_oss_cached_slim/gptoss_cached_slim_causallm.cpp:34-86` | 위와 별도 override |
| Gemma3 | `models/gemma3/gemma3_causallm.cpp:126-208` | 레이어별 sliding/full, logit softcap |
| LFM2 | `models/lfm2/lfm2_causallm.cpp:45-104` | conv-only block 가능; cache binding에서 없는 layer를 skip |
| Gemma4 | `models/gemma4/gemma4_causallm.cpp:497-587,600-724` | 레이어별 head dim/KV head 수, shared KV, skip-prefill |

Gemma4는 일반 placeholder helper를 쓰지 않고
`createGemma4KVCachePlaceholders()`를 쓴다 (`gemma4_causallm.cpp:217-244`). 또한
레이어별 `getAttentionHeadDim()`, `getKVHeadCount()`, `getKVCacheWidth()`가 다를 수 있다
(`:42-55`)고, manager도 `kv_widths` overload로 할당한다 (`:846-861`). 일반 모델만
가정한 전역 `NUM_KEY_VALUE_HEADS` 기반 compaction은 Gemma4 cache를 손상시킨다.

Gemma4 shared-KV layer에는 `skip_prefill`이 전달될 수 있다
(`gemma4_causallm.cpp:570-587`). `mha_core`는 이 경우 K/V를 쓴 직후 attention score를
만들기 전에 반환한다 (`mha_core.cpp:740-743`). 따라서 최초 CPU 범위를 Qwen2/Qwen3
같은 표준 full-attention CausalLM으로 제한하거나, shared source layer의 선택 plan을
명시적으로 재사용하는 설계가 필요하다.

### 4.4 build/test 등록

- CausalLM은 `enable-transformer` 및 application build 아래에서 포함된다
  (`meson.build:61-63`, `:748-758`, `Applications/meson.build:53-56`).
- `kv_cache_manager.cpp`는 `Applications/CausalLM/meson.build:1-10`의 CausalLM
  library source다.
- `mha_core.cpp`는 별도 shared layer library다
  (`Applications/CausalLM/layers/meson.build:88-99`).
- 기존 manager test target은 `Applications/CausalLM/meson.build:212-246`, 소스는
  `test/unittest/layers/unittest_kv_cache_manager.cpp`다.
- 모델 differential test는 `Applications/CausalLM/meson.build:310-362`에 있고,
  현재 `enable-fp16=false`인 none/windows/android 플랫폼에서만 Meson target이
  활성화된다 (`:310-317`).
- Android ndk-build는 source를 두 목록에 직접 나열한다. core 목록은
  `Applications/CausalLM/jni/Android.mk:65-113`, quantize executable 목록은
  `:183-229`다. 새 `.cpp`를 만들면 Meson만 갱신해서는 안 된다.

가능하면 첫 구현은 기존 `mha_core.*`, `kv_cache_manager.*`, `causal_lm.*` 안에서
끝내 새 build source 등록을 피한다. 새 policy `.cpp`가 필요하면 그것을 소비하는
library가 `mha_core_layer`인지 `causallm`인지 먼저 결정해야 한다. `mha_core_layer`가
상위 `causallm` library의 `KVCacheManager` symbol에 의존하도록 만들면 현재 dependency
방향과 순환될 수 있다.

## 5. SnapKV 통합을 막는 핵심 결합

### 5.1 `cache_index`가 세 역할을 동시에 수행한다

현재 `cache_index`는 다음 세 의미가 항상 같다고 가정한다.

1. 새 K/V를 쓸 물리 row (`mha_core.cpp:714-722`)
2. K/Q에 적용할 절대 RoPE position (`:724-731`, `:745-748`)
3. attention이 읽을 유효 row 수 (`:751-775`)

SnapKV로 prompt `P`개를 `C`개로 줄이면 다음 decode token에서 물리 write row는
`C`지만 RoPE position은 `P`다. 이 둘을 분리하지 않은 구현은 압축 cache 뒤에
`P` row로 쓰거나, decode query에 `C` 위치의 잘못된 RoPE를 적용한다.

권장 상태 이름은 다음처럼 의미가 드러나야 한다.

```text
logical_position  := 다음 token의 절대 RoPE 위치 (기존 _from)
cache_length      := 이 layer/cache의 물리 유효 row 수 및 다음 write row
cache_capacity    := 실제 backing buffer가 수용하는 row 수
```

external `incremental_forwarding()`의 `_from/_to`는 logical position으로만 쓰고,
압축 이후 `cache_length = _from`으로 덮어쓰면 안 된다. 비압축 경로에서는 두 값이
같으므로 기존 결과가 유지된다.

### 5.2 manager의 단일 position도 충분하지 않다

SnapKV 선택은 레이어/배치/head별로 다르다. 보존 개수는 보통 같게 맞출 수 있지만,
sliding layer를 압축하지 않거나 prompt가 capacity 이하인 경우 레이어별 valid length가
달라질 수 있다. `KVCacheManager::cache_pos_` 하나만으로는 이를 표현하지 못한다.

최소한 아래 둘 중 하나가 필요하다.

- manager에 `logical_position_`과 `std::vector<unsigned int> valid_lengths_`를 둔다.
- 또는 물리 valid length는 각 `MHACoreLayer`가 소유하고 manager는 storage만 소유한다.
  이 경우 save/load 및 host introspection API는 MHA state를 함께 받아야 한다.

### 5.3 GQA score와 KV head 수가 다르다

`out_`의 마지막 축은 `num_heads_Q`지만 cache는 `num_heads_KV`다. 현재 kernel도
`gqa_size = num_heads_Q / num_heads_KV`로 Q head 그룹을 하나의 KV head에 매핑한다
(`mha_core.cpp:772-775`). SnapKV selection도 같은 그룹의 Q-head observation score를
명시적으로 reduce하여 **KV head마다 하나의 index set**을 만들어야 한다. Q-head별
index를 그대로 Hkv cache에 gather하면 shape가 맞지 않거나 임의 head의 선택만 남는다.

권장 reduce는 명세에서 고정해야 한다(예: 각 observation query에 대해 softmax 후,
동일 KV group의 Q-head 및 observation row 평균/합). 합과 평균은 고정된 개수에서는
순위가 같지만, mask/가변 row를 도입하면 같지 않을 수 있다.

또한 `num_heads_Q % num_heads_KV == 0`을 명시적으로 검증해야 한다. 현재 코드는 정수
나눗셈 결과를 사용하지만 이 divisibility를 직접 검사하지 않는다.

### 5.4 sliding-window attention과 SnapKV는 같은 정책이 아니다

`compute_kcaches()`와 V-cache kernel은 `local_window_size`만큼의 마지막 물리 row만
읽는다 (`mha_core.cpp:51-68`, `:589-613`, `:650-684`, `:1385-1444`). sliding layer의
prefill score에는 window 밖의 오래된 prefix가 애초에 없으므로 SnapKV importance를
계산할 수 없다.

첫 구현은 다음 중 하나를 명시해야 한다.

- `sliding_window == UINT_MAX`인 full-attention layer에만 SnapKV 적용, sliding layer는
  기존 유지.
- sliding layer는 별도의 단순 ring/window eviction으로 마지막 W개만 유지.

두 정책을 같은 `snapkv_enabled` 분기 안에서 암묵적으로 섞지 않는 것이 안전하다.

### 5.5 precomputed cache, multi-turn, save/load

현재 cache 파일에는 header나 logical position metadata가 없고 K/V Tensor bytes만
저장된다 (`kv_cache_manager.cpp:189-249`). 호출자가 넘기는 `seq_len`이 저장 row 수,
로드 row 수, 다음 절대 position을 모두 뜻한다 (`causal_lm.cpp:283-298`). 압축 후에는
이 세 값이 달라진다.

첫 버전은 `USE_KVCACHE`/save/load와 SnapKV 조합을 명시적으로 거부하는 편이 안전하다.
지원하려면 versioned header에 적어도 아래가 필요하다.

- logical next position
- 레이어별 physical valid length와 width/head geometry
- dtype/format/batch/layer count
- SnapKV capacity/window/pooling version
- 필요 시 prompt-tail 경계 또는 retained original positions

multi-turn도 같은 문제를 가진다. `global_token_len`은 절대 위치를 누적한다
(`causal_lm.cpp:565`, `:620-629`, `:677`). 다음 `run()`에서
`setKVCachePosition(global_token_len)`으로 물리 위치까지 되돌리면 압축 효과와 cache
내용이 깨진다.

## 6. 구현 후보 비교

### 후보 A: manager-only, prefill 후 host compaction

`CausalLM`이 prefill 반환 후 manager에 보존 인덱스를 넘겨 compaction하는 방식이다.

- 장점: storage mutation과 serialization 책임이 manager에 모인다.
- 단점: manager에는 attention score가 없다. MHA가 레이어별 selection plan을 보관하고
  host가 `forEachLayer()`로 회수해야 한다. layer 이름→manager index 매핑, hybrid
  conv-only layer, batch/Gemma4 geometry 처리가 추가된다.
- 단점: 모든 MHA의 물리 길이를 다시 설정하는 동기화 단계가 필요하다.

정책/저장 책임을 장기적으로 manager에 두고 싶을 때는 적합하지만 첫 패치로는 연결
코드가 많다.

### 후보 B: `mha_core`가 score 산출과 in-place compaction 수행 (권장 1단계)

`out_`를 이미 소유한 MHA가 attention output 계산 직후 자신에게 바인딩된 K/V cache를
직접 gather한다.

- 장점: score와 정확한 head geometry가 같은 객체에 있고, layer index 매핑이 없다.
- 장점: 기존 CPU QK/V kernels와 Tensor layout을 바꾸지 않는다.
- 장점: no-sink/sink 두 경로에서 공용 helper로 구현 가능하다.
- 단점: manager가 실제 physical length를 모르게 된다. save/load와 manager position을
  함께 정리해야 한다.
- 단점: backing allocation은 여전히 `MAX_SEQ_LEN`; persistent memory 절감은 아직 없다.

CPU correctness를 먼저 확보하는 최소 침습 단계로 가장 현실적이다. 문서/metric에는
이를 “in-place logical eviction / decode working-set compression”으로 명확히 표시해야
한다.

### 후보 C: 작은 persistent cache + 레이어별 full-prefill 임시 cache (권장 완성형)

external cache capacity를 `max_capacity_prompt + max_decode_tokens` 정도로 줄이고,
prefill 때 한 attention layer만 full K/V 임시 Tensor를 만들어 기존 output과 SnapKV
score를 계산한 뒤 선택된 K/V만 persistent cache에 쓴다.

- 장점: 레이어 전체의 full prompt cache를 동시에 보유하지 않아 실제 지속 메모리를
  SnapKV 목표에 맞게 줄일 수 있다.
- 장점: graph placeholder도 실제 capacity와 일치시킬 수 있다.
- 단점: prefill 경로가 기존 “external cache에 먼저 append 후 읽기”에서 분기되어 코드
  양과 peak transient memory 검증이 늘어난다.
- 단점: chunked prefill, precomputed cache, multi-turn 재압축 설계가 필요하다.

후보 B로 selection/gather와 logical/physical 위치 모델을 먼저 검증한 뒤 후보 C로
allocation을 줄이는 2단계가 위험이 가장 낮다.

### 비권장: placeholder dim은 full인데 backing buffer만 작게 바인딩

graph input dimension과 external Tensor mapping은 여전히 full cache 크기를 전제로 한다
(`neuralnet.cpp:1564-1572`). 포인터 동일성 때문에 현재 input copy가 우연히 생략될 수
있지만, framework validation/향후 copy/resize가 작은 실제 allocation을 넘지 않는다는
보장이 없다. 이 방식은 out-of-bounds를 숨기는 구현이므로 피해야 한다.

## 7. 권장 1단계 변경 지점

### 7.1 `mha_core.h/.cpp`

1. property 또는 runtime state로 다음을 추가한다.

   ```text
   snapkv_enabled=false
   snapkv_max_capacity_prompt=<positive integer>
   snapkv_window_size=<positive integer>
   snapkv_pooling_kernel_size=<positive odd integer>
   ```

   기본 disabled가 중요하다. `capacity > window`, kernel 홀수, overflow 없는
   `capacity + max_new_tokens`를 검증한다.

2. `cache_index`를 logical position과 physical cache length로 분리한다. 기존
   `cache_index` property는 backward compatibility를 위해 둘을 함께 설정하되,
   압축 활성 상태에서는 새 명시적 setter/property를 사용한다.
3. `softmax_triangle()` 후의 `out_`에서 observation score를 추출한다. initial full
   prefill (`from == 0`, `step_size > 1`, full attention)만 1차 지원 조건으로 두면
   indexing과 상태가 명확하다.
4. 동일 KV group의 Q heads를 reduce하고 prefix 축에 padding-preserving pooling을
   적용한 뒤 deterministic top-k를 고른다. tie는 `(score desc, original_index asc)`로
   고정하고, 최종 index는 원 토큰 순서로 정렬한다.
5. `compute_fp16vcache_transposed()`가 끝난 뒤 K/V를 함께 gather한다. tail observation
   window는 항상 원 순서 그대로 뒤에 붙인다.
6. cache capacity bounds를 forwarding 진입 시 명시적으로 검사한다. 현재 external
   경로에는 legacy path의 `to > max_timestep` 검사와 동등한 명시적 guard가 없다.

prefill triangular `out_`는 key-position-major/Q-head-interleaved다. full attention,
`from=0`일 때 query `q`의 key `k` score offset은 개념적으로 다음과 같다.

```text
flat_pair = q * (q + 1) / 2 + k
score_index = flat_pair * num_heads_Q + q_head
```

구현은 `calc_windowed_attn_index()`와 현재 `from`을 사용해 기존 layout과 한 곳에서
공식을 공유해야 하며, 별도 삼각 index 공식을 중복해 sliding/chunked 확장 때 어긋나지
않도록 한다.

### 7.2 `KVCacheManager`

1단계에서 MHA가 직접 compaction하더라도 manager state는 정리해야 한다.

- `cache_pos_`의 의미를 문서화하고 logical/physical 상태를 분리한다.
- 레이어별 valid length accessor를 두거나, 적어도 압축 후 공통 physical length를
  설정하는 API를 둔다.
- per-layer width/head geometry를 안전하게 조회할 API를 둔다. Gemma4 overload에서도
  0을 반환하지 않아야 한다.
- batch bounds와 `size_t` 곱셈 overflow를 검사한다.
- save/load는 SnapKV 활성 시 명시적으로 거부하거나 versioned format으로 바꾼다.

K/V gather는 dtype을 수치 변환하지 말고 byte-copy해야 한다. NCHW CausalLM에서 head
slice의 byte offset은 다음과 같다.

```text
(((batch * max_seq_len + row) * num_kv_heads + kv_head) * head_dim)
  * element_size
```

선택 원본과 목적지가 겹치므로 scratch row/head buffer 또는 `memmove`와 정렬된 index
불변식을 사용해야 한다. 읽을 원본이 앞선 write로 덮이지 않는다는 것을 테스트 없이
가정하지 않는 편이 안전하다.

### 7.3 config 전파

SnapKV는 모델 architecture가 아니라 runtime/cache 정책이므로 file-based 모델에서는
`nntr_config.json`의 nested object가 자연스럽다.

```json
{
  "snapkv": {
    "enabled": true,
    "max_capacity_prompt": 256,
    "window_size": 32,
    "pooling_kernel_size": 5
  }
}
```

graph 생성 시 MHA property가 필요하면 모든 override 표(4.3절)를 수정하거나,
`Transformer`에 공용 `appendSnapKVProperties()` helper를 만들어 각 attention builder가
호출하도록 해야 한다. 단순 runtime state만 필요하면 model initialize 후 prefill 전에
`forEachLayer()` + layer property로 전달하면 변형 모델의 중복을 줄일 수 있다.

## 8. API/ABI 영향

### 피해야 할 첫 단계 변경

- public C API의 `Config`는 값으로 전달되는 struct다
  (`Applications/CausalLM/api/causal_lm_api.h:75-90`). 여기에 필드를 추가하면 기존
  caller와 크기/호출 ABI가 달라진다. 첫 구현은 public `setOptions()`를 바꾸지 않는 것이
  좋다.
- `ModelRuntimeConfig`는 internal header에 있지만 등록 모델 경로 전체가 같은 layout을
  전제로 한다 (`api/model_config_internal.h:50-77`). file JSON만 먼저 지원하면 이
  struct 변경을 피할 수 있다.
- 새 virtual method를 `Transformer`/`CausalLM`에 추가하면 vtable ABI가 변한다.
  protected non-virtual helper 또는 기존 property/`forEachLayer()` 경로를 우선한다.

### 이미 존재하는 C++ ABI 위험

- `MHACoreLayer`는 `WIN_EXPORT` class다 (`mha_core.h:214-237`). private member를 추가해도
  class size가 변하므로 외부에서 객체를 직접 생성하는 binary에는 ABI 영향이 있다.
- `CausalLM`도 exported class이고 `KVCacheManager`를 값 멤버로 가진다
  (`causal_lm.h:57`, `:210-217`). manager에 member를 추가하면 CausalLM layout도
  바뀐다.
- Windows에서 CausalLM 본체는 static library지만
  (`Applications/CausalLM/meson.build:127-143`), `mha_core_layer`는 shared library다
  (`Applications/CausalLM/layers/meson.build:88-99`). 플랫폼별
  결과가 다르므로 “private member이므로 ABI 무관”이라고 볼 수 없다.

저장소가 CausalLM C++ ABI 안정성을 보장하지 않는다면 전체 rebuild를 전제로 변경할 수
있지만, PR 설명에 이를 기록해야 한다. ABI 보존이 요구되면 새 상태를 pImpl로 모으는
별도 작업이 필요하다.

## 9. cross-platform 주의점

- 첫 CPU target의 기본 cache dtype은 FP32가 아니라 `UINT16` FP16 bit-pattern이다.
  `reinterpret_cast<float *>`는 raw inference API의 pointer carrier일 뿐, cache를
  float 배열로 역참조해도 된다는 뜻이 아니다 (`causal_lm.cpp:493-498`).
- `_FP16`/`__fp16`은 compiler/architecture guard 밖에서 사용하지 않는다.
  `meson.build:162-224`에 x86/ARM/Android별 FP16 조건이 다르다.
- AVX/NEON 전용 gather를 첫 구현에 넣지 말고 표준 C++ byte movement로 정확성을 먼저
  확보한다. 기존 QK/V SIMD kernel layout은 그대로 유지할 수 있다.
- POSIX API, VLA, `/tmp` 고정 경로를 쓰지 않는다. 기존 manager test의
  `/tmp/test_kv_cache.bin` (`unittest_kv_cache_manager.cpp:228-260`)은 Windows 친화적
  선례가 아니므로 새 테스트는 `std::filesystem::temp_directory_path()`를 쓴다.
- `UINT_MAX`는 full sliding-window sentinel이다. capacity/window 산술에서 이를 더하거나
  곱하지 않는다 (`mha_core.cpp:1572-1583`).
- `unsigned int`의 `capacity + max_new_tokens`, Tensor element/byte offset은 overflow를
  검사하고 실제 offset 계산은 `size_t`로 한다.
- batch별 선택은 달라도 physical valid length는 같아야 현재 dense Tensor shape를
  유지할 수 있다.
- pooling/top-k tie 결과를 deterministic하게 만들어 x86/ARM과 thread scheduling에
  따른 cache 선택 변동을 막는다. ranking 중 NaN은 최하위로 정의한다.
- 현재 Tensor code는 NCHW의 height=sequence, width=features 전제를 광범위하게 쓴다.
  generic manager `format` 인자를 이유로 NHWC까지 지원한다고 선언하지 말고, 첫
  CausalLM SnapKV는 NCHW를 검증하거나 명시적으로 제한한다.
- `ThreadManager::parallel_for()`로 QK/softmax/V 작업이 끝난 뒤에만 cache를 이동한다.
  실행 중인 head worker와 gather가 같은 cache를 동시에 읽고 쓰면 data race다.

## 10. 권장 테스트 위치와 케이스

### 10.1 순수 selection/compaction 단위 테스트

기존 `test/unittest/layers/unittest_kv_cache_manager.cpp` 또는 별도
`unittest_snapkv.cpp`에 작은 deterministic Tensor로 다음을 검증한다.

1. disabled일 때 byte-for-byte no-op.
2. prompt length `<= capacity`일 때 no-op.
3. `capacity > window`, odd pooling kernel 등 config validation.
4. 단일 head에서 알려진 observation score의 pooled top-k와 stable tie 순서.
5. GQA에서 Q-head group reduce 후 KV-head별 서로 다른 index 선택.
6. tail window가 항상 보존되고 원 순서가 유지됨.
7. K와 V가 정확히 같은 index로 gather됨.
8. batch 2, layer 2에서 batch/layer memory가 서로 오염되지 않음.
9. FP32와 UINT16에서 bitwise 동일한 이동. FP16 build는 compile/run 가능할 때 별도.
10. capacity 0/overflow/out-of-range batch/head/row 예외.
11. in-place overlap에서 원본이 덮이지 않음.
12. per-layer width overload 및 Gemma4형 서로 다른 width.

selection helper가 `mha_core` private에만 있으면 단위 테스트가 어렵다. head-agnostic한
score selection과 byte gather를 작은 internal utility로 분리하되, 새 library dependency
순환을 만들지 않는 위치를 선택하는 것이 좋다.

### 10.2 `mha_core` 통합 테스트

현재 전용 `mha_core` gtest가 없으므로 다음 중 하나를 추가한다.

- custom layer를 작은 graph에 넣고 5-input external cache로 prefill→decode를 직접
  실행하는 test target.
- 기존 tiny Qwen3 adapter에 cache/state inspection hook을 추가하는 모델 test.

필수 검증은 다음과 같다.

- prefill output은 eviction을 하지 않은 기존 output과 동일하다. compaction은 현재
  prefill output 계산 뒤에 일어나야 하기 때문이다.
- cache physical length는 capacity로 줄고 logical next position은 prompt length다.
- 다음 decode K/Q의 RoPE position은 prompt length를 사용하지만 write row는 physical
  length를 사용한다.
- 각 head의 compact cache로 계산한 decode logits가 독립 reference 구현과 tolerance
  내에서 같다.
- `snapkv_enabled=false`와 `capacity >= prompt`는 기존 Qwen2/Qwen3 reference fixture와
  동일하다.
- batch > 1, GQA, attention sink를 각각 커버한다.
- sliding/full 혼합 모델은 명세대로 skip 또는 별도 window eviction한다.
- multi-turn, precomputed save/load는 지원한다면 correctness, 지원하지 않으면 명시적
  error를 검증한다.

기존 generic adapter는 decode마다 `setKVCachePosition(from)`을 호출한다
(`test/unittest/models/causallm_test_utils.h:322-347`). 이 helper도 logical/physical
분리 API를 사용하도록 바꾸지 않으면 SnapKV 테스트가 실제 run 경로와 다르게 실패한다.

### 10.3 정적/빌드 검증 매트릭스

현재 환경에서 실행 build가 불가능하더라도 최소한 아래를 PR 체크리스트에 둔다.

```text
clang-format-14: 변경된 모든 .h/.cpp
git diff --check
desktop x86_64, enable-transformer=true, enable-fp16=false
desktop x86_64, enable-transformer=true, enable-fp16=true (지원 compiler)
Windows Meson/MSVC CPU build
Android NDK ARM64 compile (기능 disabled 포함)
unittest_kv_cache_manager
unittest_causallm_models (Qwen2/Qwen3 disabled parity + SnapKV fixture)
ASan/UBSan 가능한 host에서 batch/overlap/offset tests
```

새 test 수가 늘면 저장소의 `check_count` CI가 반응할 수 있다는 `AGENTS.md` 규칙도
확인해야 한다.

## 11. 구현 순서 제안

1. **불변식 고정**: config validation, full-attention initial-prefill 범위, GQA reduce,
   pooling/tie semantics, unsupported 조합을 명세한다.
2. **순수 CPU helper**: post-softmax score→KV-head별 indices와 dtype-agnostic K/V gather를
   독립 deterministic 테스트로 검증한다.
3. **위치 분리**: SnapKV가 꺼진 상태에서 logical position/cache length 분리 refactor를
   먼저 하고 기존 reference parity를 확인한다.
4. **MHA hook**: `out_`가 살아 있는 두 one-batch 경로(no sink/sink)에 선택/compaction을
   넣는다.
5. **host state 동기화**: manager, CausalLM run, test adapter의 position API를 맞춘다.
6. **지원 범위 확대**: Qwen2/Qwen3 full attention에서 시작해 GPT-OSS sink, sliding 혼합,
   Gemma3, LFM2, Gemma4 shared-KV 순으로 확장한다.
7. **실제 메모리 축소**: correctness가 고정된 뒤 작은 persistent external cache +
   per-layer transient prefill cache로 넘어간다.
8. **serialization/public API**: versioned cache format과 stable public option이 실제로
   필요할 때 별도 topic/commit으로 진행한다.

이 순서는 “한 commit = 한 topic” 규칙에도 맞고, 위치 refactor와 알고리즘 오차를 한
번에 디버깅하는 위험을 줄인다.

## 12. 구현 전 반드시 결정할 의심점

- 논문의 pooling이 avg인지 max인지, padding/경계 semantics와 기본 kernel 값은 무엇인가?
- GQA에서 동일 KV head를 공유하는 Q heads를 sum, mean, max 중 무엇으로 reduce할 것인가?
- `max_capacity_prompt`가 observation window를 포함한 총 prompt capacity인지 확인할 것.
- eviction을 최초 prefill 한 번만 하는지, chunked/multi-turn prefill마다 재적용할지?
- sliding layer에는 SnapKV를 끌지, 별도 last-W eviction을 할지?
- 1단계 목표가 decode working-set 축소인지, 실제 persistent allocation 축소까지인지?
- batch마다 다른 retained index를 허용하되 valid length는 공통으로 둘 것인지?
- precomputed cache/save/load를 1차 범위에서 금지할지, 즉시 versioning할지?
- Gemma4 shared-KV/skip-prefill layer의 selection plan을 어느 source layer에서 가져올지?

이 결정 없이 `cache_index`만 capacity로 줄이는 구현은 RoPE 위치 오류를 만들고,
manager cache만 앞쪽으로 복사하는 구현은 attention 기반 SnapKV가 아니라 임의 eviction이
된다.
