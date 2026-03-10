# MHACoreLayer Documentation

## 개요

`mha_core.cpp`는 T5Gemma2 모델의 Multi-Head Attention (MHA) core 기능을 구현한 C++ 파일입니다. 이 문서는 코드의 구조, 작동 방식, 그리고 주요 최적화 기술을 설명합니다.

---

## 목차

1. [개요](#개요)
2. [아키텍처와 주요 컴포넌트](#아키텍처와-주요-컴포넌트)
3. [핵심 클래스 구조](#핵심-클래스-구조)
4. [주요 메서드 분석](#주요-메서드-분석)
5. [작동 방식](#작동-방식)
6. [최적화 기술](#최적화-기술)
7. [데이터 흐름](#데이터-흐름)
8. [구현 특이점](#구현-특이점)

---

## 아키텍처와 주요 컴포넌트

### 주요 컴포넌트

| 컴포넌트 | 설명 | 기본값 |
|--------|------|--------|
| `num_heads_Q` | Query의 attention head 개수 | config에서 설정 |
| `num_heads_KV` | Key/Value의 attention head 개수 (GQA) | config에서 설정 (num_heads_Q / group_size) |
| `head_dim` | 각 head의 차원 | `query_width / num_heads_Q` |
| `group_size` | GQA 그룹 크기 | `num_heads_Q / num_heads_KV` |
| `theta` | RoPE theta parameter | 10000 (default) or 1000000 (full_attention) |
| `rope_scaling_type` | RoPE scaling type | "default" or "yarn" |
| `scale` | RoPE scaling factor | 8.0 (linear RoPE) |
| `local_window_size` | Sliding window 크기 | 4096 |
| `max_timestep` | KV cache 최대 시퀀스 | config에서 설정 |
| `is_causal` | Causal mask 여부 | decoder에서 True |
| `use_rope` | RoPE 사용 여부 | config에서 설정 |
| `use_sink` | Sink token 사용 여부 | config에서 설정 |
| `attn_logit_softcapping` | Attention logit softcapping | config에서 설정 |

---

## 핵심 클래스 구조

### MHACoreLayer

```cpp
class MHACoreLayer {
private:
  // Properties
  nntrainer::LayerNode mha_core_props;  // Layer properties
  nntrainer::ActivationType sm;  // Softmax activation
  float epsilon;  // Numerical stability
  
  // Configuration
  unsigned int num_heads_Q;     // Query head count
  unsigned int num_heads_KV;    // Key/Value head count (GQA)
  unsigned int head_dim;       // Head dimension
  unsigned int gqa_size;       // Group Query Attention group size
  
  // RoPE
  float theta;                // RoPE base theta
  std::vector<float> thetas;   // Precomputed frequencies
  float attention_scaling;    // RoPE scaling factor
  std::vector<std::vector<float>> *freqs_cos;  // Precomputed cos values
  std::vector<std::vector<float>> *freqs_sin;  // Precomputed sin values
  std::vector<std::vector<_FP16>> *freqs_cos_fp16;  // FP16 version
  std::vector<std::vector<_FP16>> *freqs_sin_fp16;  // FP16 version
  
  // Cache
  unsigned int cache_index;        // Cache position
  bool cache_shift;             // Cache shift flag
  
  // Masking
  unsigned int local_window_size; // Sliding window size
  bool is_causal;             // Causal mask
  
  // Regularization
  float attn_logit_softcapping; // Attention logit softcapping
  bool use_sink;                // Sink token usage
};
```

---

## 주요 메서드 분석

### 1. `finalize()`

**목적**: Layer 초기화 및 텐서/캐시 할당

**주요 작업**:
1. 입력/출력 차원 확인 및 검증
2. Attention head 수 및 head_dim 계산
3. GQA (Grouped Query Attention) 설정
4. KV Cache tensor 생성
5. RoPE theta 파라미터 설정
6. Sink token 생성 (사용 시)
7. Softcapping 파라미터 설정

**코드 분석**:
```cpp
// Head dimension 계산
num_heads_Q = std::get<nntrainer::props::NumHeads>(mha_core_props).get();
num_heads_KV = num_heads_Q or std::get<props::NumHeads_KV>(mha_core_props).get();
head_dim = static_cast<size_t>(query_width) / num_heads_Q;

// GQA group size 계산
gqa_size = num_heads_Q / num_heads_KV;

// RoPE theta 설정
theta = (float)std::get<props::RopeTheta>(mha_core_props).get();
```

---

### 2. `incremental_forwarding()`

**목적**: Inference 모드에서의 단계별 forward pass

**핵심 메커니즘**:
- Incremental inference 지원 (한 토큰씩 처리)
- KV Cache 업데이트
- Cache overflow 시 자동 shift

**작동 흐름**:

```
입력: query_step, key_step, value_step [from:to]
      ↓
    ┌─────────────────────┐
    │  RoPE 적용 (optional)  │
    │  query_step에만 적용     │
    │  b_cache_key_step에는     │
    │    이미 적용됨            │
    └─────────────────────┘
      ↓
    ┌─────────────────────┐
    │   Cache Update        │
    │   b_cached_key에       │
    │   새로운 key/value 저장    │
    └─────────────────────┘
      ↓
    ┌─────────────────────┐
    │   Attention 계산      │
    │   compute_kcaches()    │
    │   - Attention scores │
    │   - Softmax          │
    │   - Attention output  │
    └─────────────────────┘
      ↓
    ┌─────────────────────┐
    │   Cache Update        │
    │   b_cached_value에   │
    │   새로운 value 저장    │
    └─────────────────────┘
      ↓
출력: attention_output_step
```

**Cache Handling**:

```cpp
// Cache overflow 처리
if (to >= max_timestep) {
    cache_shift = true;
    from = max_timestep - 1;
    to = max_timestep;
    // 가장 오래된 cache 항목 삭제
}
```

---

### 3. `compute_kcaches()`

**목적**: Attention score 계산 (Query × Key^T)

**signature** (nntrainer core library):
```cpp
template<typename T>
void compute_kcaches(
    const T* in,           // Query: [1, 1, seq_len, num_heads_Q * head_dim]
    const KV* cache,       // Key cache: [1, 1, max_seq_len, num_heads_KV * head_dim]
    T* out,                // Output: [1, 1, seq_len, num_heads_Q * (context_len)]
    unsigned int row_to_compute,  // 계산할 행 수
    unsigned int num_cache_head,     // KV cache head 수
    unsigned int head_dim,          // Head dimension
    unsigned int group_size,       // GQA group size
    unsigned int tile_size,         // Tiling factor
    unsigned int sliding_window,     // Sliding window 크기
    unsigned int head_kv_start,    // KV head 시작 인덱스
    unsigned int head_kv_end         // KV head 끝 인덱스
);
```

**최적화 기술**:

#### 1. **Grouped Query Attention (GQA)**

```cpp
// num_heads_Q = 8, num_heads_KV = 4
// GQA 그룹 크기 = 8 / 4 = 2

// KV head 0,1 → Query head 0,1,2,3 (repeating)
unsigned int gqa_size = num_heads_Q / num_heads_KV;

// nntrainer core library (추정)
for (unsigned int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
    compute_kcaches(..., head_kv, head_kv + 1);
    // head_kv=0,1 covers query heads 0,1,2,3
    // head_kv=2,3 covers query heads 4,5,6,7
}
```

#### 2. **Sliding Window Attention**

```cpp
// 로컬 윈도(local_window_size) 내에서만 attention 계산
if (sequence_len < local_window_size) {
    // 전체 시퀀스 사용
    row_to_compute = sequence_len;
} else {
    // Sliding window만 사용
    row_to_compute = local_window_size;
}
```

#### 3. **Single Token Optimization**

```cpp
if (sequence_len == 1) {
    // Single token generation (decoding)
    // KV head에 대해 병렬 처리
    int row_to_compute = is_causal ? from + 1 : from + sequence_len;
    unsigned int num_cache_head = num_head / group_size;
    
#pragma omp parallel for schedule(static)
    for (unsigned int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        nntrainer::compute_kcaches(
            in_data, cache_data, out_data, row_to_compute,
            num_cache_head, head_dim, group_size, tile_size, 
            local_window_size, head_kv, head_kv + 1
        );
    }
}
```

#### 4. **Multi-threading for Prefill**

```cpp
if (sequence_len > 1) {
    std::vector<std::future<void>> futures;
    unsigned int seq = sequence_len < local_window_size ? sequence_len : local_window_size;
    
    for (int i = 0; i < seq; ++i) {
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        size_t out_start_row = i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;
        
        futures.emplace_back(pool.submit_task([=]() {
            nntrainer::compute_kcaches(input_addr, cache_addr, output_addr, ...);
        }));
    }
    for (auto &fut : futures) {
        fut.get();
    }
}
```

---

### 4. `softmax_triangle()`

**목적**: Attention scores에 Softmax 적용

**특징**:
- **Triangle 최적화**: Causal mask에서 불필요한 계산 건너뜀기
- **Softcap**: `attn_logit_softcapping`으로 attention logits 제한
- **Sink token**: Sink score 빼기기 (사용 시)

**Causal Mask Optimization**:

```cpp
// Triangle index 계산
size_t calc_attn_index(size_t i) {
    return (i * (i + 1)) / 2;  // i=0→0, i=1→1, i=2→3, i=3→6
}

// 예: from=5, to=8 (sequence_len=3)
// Query positions: 5, 6, 7
// row_to_compute: 6, 7, 8
// Triangle optimization:
//   - Position 5 attends to: 0-5 (6개)
//   - Position 6 attends to: 0-6 (7개)
//   - Position 7 attends to: 0-7 (8개)
//   - 총 계산: 6+7+8 = 21회
//   - Full square: 8×8 = 64회 (대신 ~3배 효율)
```

**Softcap 적용**:

```cpp
if (attn_logit_softcapping > 0.0f) {
  size_t len = qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
  float inv_softcapping = 1.0f / attn_logit_softcapping;
  for (size_t i = 0; i < len; ++i) {
    // Softcap: tanh(scaling * score) * softcap
    qk_out_[i] = std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
  }
}
```

**Sink Token 처리**:

```cpp
if (use_sink) {
    // Sink score를 모든 attention score에 더함
    // nntrainer::softmax_row_inplace(out_data, start_row, end_row, num_head, sink_step);
}
```

---

### 5. `compute_fp16vcache_transposed()`

**목적**: Attention score × Value cache (Value projection)

**최적화**:

#### 1. **FP16 Cache** (Memory 효율)

```cpp
// KV cache를 FP16으로 저장
ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16}
);

// FP16 quantized value 사용
const uint16_t *vcache_data = vcache.getData<uint16_t>();
```

#### 2. **Transposed Memory Access** (캐시 효율)

```cpp
// nntrainer core library (추정)
void compute_fp16vcache_fp32_transposed(
    int row_num,                  // 현재 행
    const float *input,        // Attention scores
    const _FP16 *vcache,       // FP16 value cache
    float *out,                // Output
    unsigned int num_cache_head,
    unsigned int gqa_size,
    unsigned int head_dim,
    unsigned int sliding_window
) {
    // Transposed memory access for better cache locality
    // Key insight: Contiguous memory access for values
    
    for (unsigned int i = 0; i < num_cache_head; ++i) {
        // Value cache에서 해당 head의 values 추출
        // Transpose: [batch, seq, num_cache_head * head_dim]
        //          → [batch, num_cache_head, seq, head_dim]
        
        // Attention score × Value
        // Score: [1, seq_len, num_cache_head * head_dim]
        // Value:  [batch, num_cache_head, seq, head_dim] (transposed)
    }
}
```

#### 3. **Multi-threading**

```cpp
if ((to - from) != 1) {
    std::vector<std::future<void>> futures;
    int seq = (to - from) < local_window_size ? to - from : local_window_size;
    
    for (int i = 0; i < seq; ++i) {
        futures.emplace_back(pool.submit_task([=]() {
          // 각 position별로 병렬 처리
          compute_fp16vcache_transposed(row_num, input, vcache_data, ...);
        }));
    }
}
} else {
    // Single token: KV head 별 병렬
#pragma omp parallel for schedule(static)
    for (int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        compute_fp16vcache_transposed(..., head_kv, head_kv + 1);
    }
}
```

---

### 6. `precompute_freqs()` & RoPE 메서드들

#### `precompute_freqs()`

**목적**: RoPE cos/sin 값 미리 계산

**최적화**: Startup 시 한 번만 계산

```cpp
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len, float theta, bool is_fp16) {
  // 이미 계산되어 있으면 스킵
#ifdef ENABLE_FP16
  if (freqs_cos_fp16 != nullptr && freqs_cos_fp16->size() == seq_len)
    return;
#else
  if (freqs_cos != nullptr && freqs_cos->size() == seq_len)
    return;
#endif
  
  // RoPE parameter 계산
  if (thetas.empty()) {
    if (rope_scaling_type == "default")
      _compute_default_parameters(head_dim, theta);  // 고정된 theta
    else if (rope_scaling_type == "yarn")
      _compute_yarn_parameters(head_dim, theta);  // Dynamic scaling
  }
  
  // Cos/Sin values 계산
  for (unsigned int i = 0; i < seq_len; ++i) {
    for (unsigned int j = 0; j < half_; ++j) {
      float angle = i * thetas[j];
      (*cos)[i][j] = std::cos(angle) * attention_scaling;
      (*cos)[i][j + half_] = std::cos(angle) * attention_scaling;  // duplicate
      (*sin)[i][j] = std::sin(angle) * attention_scaling;
      (*sin)[i][j + half_] = std::sin(angle) * attention_scaling;  // duplicate
    }
  }
}
```

#### `_compute_default_parameters()` (Default RoPE)

```cpp
void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {
  // no attention scaling
  attention_scaling = 1.0f;
  
  // θ_i = 10000^(-2(i-1)/dim) for i = [1, 2, ..., dim/2]
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    // Inverse frequency 계산
    thetas.push_back(1.0 / (
      std::pow(theta, (2 * i) / static_cast<float>(head_dim))
    ));
  }
}
```

#### `_compute_yarn_parameters()` (YARN RoPE - Long Context)

```cpp
void MHACMeLayer::_compute_yarn_parameters(int head_dim, float theta) {
  // Attention scaling calculation
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };
  attention_scaling = get_mscale(scale);
  
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;
  
  // Find correction dimension
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings / 
           (num_rotations * 2 * M_PI))) / (2 * std::log(base));
  };
  
  // Interpolation vs extrapolation
  std::vector<float> inv_freq_interpolation;  // 학습된 길이 범위
  std::vector<float> inv_freq_extrapolation;    // 그 이상 긴 시퀀스 범위
  
  // Linear ramp between extrapolation and interpolation
  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  
  // Combine frequencies
  for (size_t i = 0; i < dim / 2; ++i) {
    thetas[i] = inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
                 inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}
```

---

### 7. `apply_rotary_emb_tensor_v2()`

**목적**: Query/Key/Value에 RoPE 적용

**최적화**:

#### Precomputed RoPE Lookup

```cpp
void MHACoreLayer::apply_rotary_emb_tensor_v2(
    nntrainer::Tensor &in,  // Input tensor
    nntrainer::Tensor &out, // Output tensor (same as in if convert_only=True)
    unsigned int dim,
    unsigned int from,      // Current position
    bool convert_only      // If true, out=in (in-place modification)
) {
  // 미리 계산된 cos/sin 값 사용
  std::vector<std::vector<float>> *cos_;
  std::vector<std::vector<float>> *sin_;
  
  for (unsigned int b = 0; b < in.batch(); b++) {
    for (unsigned int c = 0; c < in.channel(); c++) {
      for (unsigned int h = 0; h < in.height(); h++) {
        if (from < max_timestep) {
          cos_ = &(*freqs_cos)[from + h];
          sin_ = &(*freqs_sin)[from + h];
        }
        
        float *in_ptr = in.getData<float>() + ...;
        // RoPE: output = rotate_half(x) * cos + x * sin
        nntrainer::compute_rotary_emb_value(
          in.width(), dim, half_, in_ptr, nullptr,
          cos_->data(), sin_->data(), convert_only
        );
      }
    }
  }
}
```

#### RoPE 공식

```python
# rotate_half(x)
# x1 = x[..., :mid]
# x2 = x[..., mid:]
# return [-x2, x1]

# output = x * cos + rotate_half(x) * sin
```

---

### 8. `one_batch_incremental_forwarding()`

**목적**: Single batch의 전체 forward pass

**작동 순서**:

```
1. Query/Key/Value 준비
   ↓
2. RoPE 적용 (Query만, Key는 이미 cache에 적용됨)
   ↓
3. Cache에서 현재까지의 key/value 로드
   ↓
4. Cache에 새로운 key/value 저장
   ↓
5. Attention score 계산 (compute_kcaches)
   ↓
6. Softmax 적용 (softmax_triangle)
   ↓
7. Value projection (compute_fp16vcache_transposed)
```

**코드 구조**:

```cpp
void MHACoreLayer::one_batch_incremental_forwarding(
    const unsigned int batch,
    const unsigned int _from,  // Start position
    const unsigned int from,   // Current position (same as _from)
    const unsigned int to,     // End position
    nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step,
    nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step,
    nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value,
    ...
) {
  // 1. Cache에서 현재까지의 key/value 로드
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(...);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(...);
  
  // 2. RoPE 적용
  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, from, false);
    apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, from, false);
  }
  
  // 3. Cache 업데이트 (새로운 key/value 추가)
  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true
  );
  
  // 4. Attention score 계산
  nntrainer::Tensor out_(1, 1, ...);
  compute_kcaches(query_step, b_cached_key, out_, ...);
  
  // 5. Softmax
  softmax_triangle(out_, to - from, num_heads_Q, from, pool);
  
  // 6. Value projection
  compute_fp16vcache_transposed(out_, b_cached_value, ...);
}
```

---

## 작동 방식

### 전체 Forward Pass 흐름

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Input Loading                                               │
│   query_step, key_step, value_step                        │
│   Shape: [batch=1, seq_len, num_heads_Q * head_dim]       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Rotary Position Embedding (if use_rope)                    │
│   apply_rotary_emb_tensor_v2(query_step, ...)                │
│   apply_rotary_emb_tensor_v2(key_step, ...)                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Cache Loading & Update                                      │
│   b_cached_key = cache_key[from:to]                          │
│   b_cached_value = cache_value[from:to]                        │
│   → cache[from:to]에 key_step, value_step 추가               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Attention Score Computation                                 │
│   compute_kcaches(query_step, b_cached_key, out_)              │
│   → Query × Key^T                                        │
│   → Output: [batch, 1, seq_len, num_heads_Q * context_len]   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Softmax Activation                                            │
│   softmax_triangle(out_, to - from, num_heads_Q, from, pool)   │
│   → Attention probabilities                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Value Projection                                            │
│   compute_fp16vcache_transposed(out_, b_cached_value, ...)      │
│   → Attention weights × Value cache                      │
│   → Output: [batch, 1, seq_len, num_heads_Q * head_dim]    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Output                                                     │
│   attention_output_step (projected value projection result)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 최적화 기술

### 1. **Grouped Query Attention (GQA)**

**개념**: Query 헤드를 그룹으로 묶어 KV 헤드 개수 감소

**효과**:
- **메모리 사용량 감소**: KV cache 크기 줄어듦
- **메모리 대역폭 효율**: 연속된 메모리 접근

```cpp
// 예: num_heads_Q=8, num_heads_KV=4
// GQA group_size = 8 / 4 = 2

// Query heads: Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7
// KV heads:    K0, K1, K2, K3

// Attention pattern:
// Q0, Q1 → K0, K1 (group 0)
// Q2, Q3 → K2, K3 (group 1)
// Q4, Q5 → K0, K1 (group 0, reuse)
// Q6, Q7 → K2, K3 (group 1, reuse)
```

### 2. **Sliding Window Attention**

**개념**: 로컬 윈도 내에서만 attention 계산

**효과**:
- **긴 시퀀스 지원**: O(seq_len) → O(window_size)
- **계산 복잡도 감소**

```cpp
// Full attention: 모든 position attend to all positions
// O(seq_len²)

// Sliding window: 각 position이 로컬 window만 attend
// O(seq_len × window_size)
```

**구현**:
```cpp
if (sequence_len < local_window_size) {
  // 짧은 시퀀스: 전체 attention
  row_to_compute = sequence_len;
} else {
  // 긴 시퀀스: sliding window
  row_to_compute = local_window_size;
}
```

### 3. **Triangle Optimization (Causal Mask)**

**개념**: Causal mask에서 불필요한 계산 건너뜀기

**효과**:
- **계산량 감소**: O(seq_len²) → O(seq_len²/2)
- **Memory bandwidth 절약**: 불필요한 softmax 계산 회피

```cpp
// Causal mask 예시 (seq_len=4)
// Position 0: attends to 0 (1계산)
// Position 1: attends to 0,1 (3계산)
// Position 2: attends to 0,1,2 (6계산)
// Position 3: attends to 0,1,2,3 (10계산)
// 총: 1+3+6+10 = 20계산
// Full square: 4×4 = 16계산

// Triangle index: (i*(i+1))/2
// 0→0, 1→1, 2→3, 3→6
```

### 4. **FP16 Quantization**

**개념**: KV cache를 FP16으로 저장

**효과**:
- **메모리 절약**: 절반 메모리 사용
- **Cache hit율 향상**: 더 많은 context를 캐시에 저장 가능

**구현**:
```cpp
// KV cache를 FP16으로 저장
ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TenzorDim::DataType::FP16}
);

// Value projection에서 FP16 cache 사용
compute_fp16vcache_transposed(..., vcache.getData<uint16_t>(), ...);
```

### 5. **Multi-threading**

**개념**: 병렬 처리로 계산 속도 향상

**전략략 전략**:
- Single token decoding: KV head 병렬
- Prefill (multiple tokens): Sequence 별 병렬
- Softmax: Row 별 병렬

**구현**:
```cpp
// Single token: KV head 병렬
#pragma omp parallel for schedule(static)
for (unsigned int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
    compute_kcaches(..., head_kv, head_kv + 1);
}

// Multiple tokens: Sequence 별 병렬
for (int i = 0; i < seq; ++i) {
    futures.push_back(pool.submit_task([=]() {
        compute_kcaches(...);
    }));
}
```

### 6. **Precomputed RoPE**

**개념**: Cos/Sin 값을 미리 계산해 runtime 계산 제거

**효과**:
- **Startup time 감소**: 한 번만 계산
- **Runtime speed 향상**: Lookup table에서 cos/sin 값 가져옴기

**구현**:
```cpp
// Initialize: freqs_cos, freqs_sin 계산 (한 번만)
precompute_freqs(head_dim, max_position_embeddings, theta, false);

// Runtime: precomputed 값 사용
cos_ = &(*freqs_cos)[from + h];
sin_ = &(*freqs_sin)[from + h];
```

### 7. **Softcap (Attention Logit Softcapping)**

**개념**: Attention logits를 제한하여 안정성 향상

**효과**:
- **Numerical stability**: 큰 attention score 방지
- **학습 안정성:** 더 안정된 gradient

**수식**:
```cpp
// score → tanh(score / softcap) * softcap
inv_softcapping = 1.0 / attn_logit_softcapping;
softcapped_score = tanh(score * inv_softcapping) * attn_logit_softcapping;
```

**구현**:
```cpp
if (attn_logit_softcapping > 0.0f) {
  for (size_t i = 0; i < len; ++i) {
    qk_out_[i] = std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
  }
}
```

---

## 데이터 흐름

### Tensor Shapes

```
Input Tensors:
├── query_step: [batch=1, seq_len, num_heads_Q * head_dim]
│   예: [1, 3, 8 * 256] = [1, 3, 2048]
├── key_step: [batch=1, seq_len, num_heads_KV * head_dim]
│   예: [1, 3, 4 * 256] = [1, 3, 1024]
└── value_step: [batch=1, seq_len, num_heads_KV * head_dim]
    예: [1, 3, 4 * 256] = [1, 3, 1024]

Cache Tensors:
├── cache_key: [batch, 1, max_timestep, num_heads_KV * head_dim]
│   예: [1, 1, 8192, 4 * 256] = [1, 1, 8192, 1024]
└── cache_value: [batch, 1, max_timestep, num_heads_KV * head_dim]
    예: [1, 1, 8192, 4 * 256] = [1, 1, 8192, 1024]

Attention Score:
├── out_: [batch, 1, num_heads_Q * context_len]
│   예: [1, 1, 8 * (from + to)] = [1, 1, 8 * 3] (for from=5, to=8)
│
└── Softmax 후:
    └── out_: [batch, 1, num_heads_Q * context_len]

Output Tensor:
└── attention_output_step: [batch, 1, seq_len, num_heads_Q * head_dim]
    예: [1, 1, 3, 8 * 256] = [1, 1, 3, 2048]
```

---

## 구현 특이점

### 1. **Incremental Inference 지원**

```cpp
void MHACoreLayer::incremental_forwarding(...) {
  unsigned int from, to;  // 현재 처리할 범위
  if (to >= max_timestep) {
    cache_shift = true;  // Cache overflow 처리
    from = max_timestep - 1;
    to = max_timestep;
  }
  // from:to 범위만 처리
}
```

### 2. **KV Cache 자동 관리**

```cpp
// Cache update가 자동으로 수행됨
nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(...);
// → key_step, value_step가 cache에 자동 저장됨
```

### 3. **Causal Mask 자동 계산**

```cpp
// Triangle index: (i*(i+1))/2
size_t MHACoreLayer::calc_attn_index(size_t i) {
  return (i * (i + 1)) / 2);
}
```

### 4. **RoPE Type 자동 선택**

```cpp
if (rope_scaling_type == "default")
  _compute_default_parameters(head_dim, theta);  // 기본 RoPE
else if (rope_scaling_type == "yarn")
  _compute_yarn_parameters(head_dim, theta);  // YARN RoPE (long context)
```

### 5. **Data Type 지원 (FP32/FP16)**

```cpp
if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
  // FP32 computation
  compute_kcaches<float>(...);
} else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
  // FP16 computation
  compute_kcaches<_FP16>(...);
#else
  NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
}
```

### 6. **Sliding Window Auto-Cutoff**

```cpp
int seq = sequence_len < local_window_size ? sequence_len : local_window_size;
// 시퀀스가 window보다 길면 window 크기로 제한
```

---

## 비교: PyTorch vs NNTrainer

| 항목 | PyTorch T5Gemma2 | NNTrainer T5Gemma2 |
|------|------------------|---------------------|
| **Attention Interface** | ALL_ATTENTION_FUNCTIONS | compute_kcaches |
| **Implementation** | Python (SDPA/Fallback) | C++ (최적화됨) |
| **Attention Scaling** | 명시적 `self.scaling` | 암묵적 (코드/weight) |
| **RoPE Implementation** | Python (torch.cos/sin) | C++ (Precomputed lookup) |
| **Softcap** | `attn_logit_softcapping` | `attn_logit_softcapping` |
| **KV Cache** | EncoderDecoderCache | FP16 Tensor |
| **GQA** | num_key_value_groups | gqa_size + repeat_kv |
| **Sliding Window** | Sliding window mask | local_window_size |
| **Triangle Optimization** | Causal mask | Triangle index |
| **Multi-threading** | PyTorch parallel | OpenMP + Thread pool |
| **Quantization** | Auto (torch.float16/bfloat16) | Explicit FP16 cache |

---

## 성능 최적화 요약

| 최적화 기술 | 구현 위치 | 효과 |
|-------------|----------|------|
| **GQA** | `compute_kcaches` | KV cache 크기 절약 |
| **Sliding Window** | `compute_kcaches` | 긴 시퀀스 지원 |
| **Triangle Opt** | `softmax_triangle` | Causal mask 계산 절반 |
| **FP16 Cache** | KV cache 저장 | 메모리 절약 |
| **Precomputed RoPE** | `precompute_freqs` | Runtime 계산 제거 |
| **Multi-threading** | `compute_kcaches`, `softmax_triangle``, `compute_fp16vcache_transposed` | 병렬 처리 |
| **Softcap** | `softmax_triangle` | Numerical stability |

---

## 주요 사용 시나리오

### 1. **Decoding (Single Token Generation)**

```
from=5, to=6 (6번째 토큰)
batch_size=1

단계:
1. apply_rotary_emb_tensor_v2(query_step)      → RoPE 적용
2. b_cached_key[from:6] 로드                 → Cache lookup
3. compute_kcaches(query, b_cached_key, out_)   → Attention score
   - Parallel: KV heads (OpenMP)
   - Causal: 6개 position (from=5, row_to_compute=6)
4. softmax_triangle(out_, ...)                  → Softmax
5. compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step)
```

### 2. **Prefill (Multiple Tokens)**

```
from=0, to=3 (3개 톁)
batch_size=1

단계:
1. apply_rotary_emb_tensor_v2(query_step)      → RoPE 적용
2. b_cached_key[0:3] 로드                 → Cache lookup
3. compute_kcaches(query, b_cached_key, out_, ...)
   - Parallel: Sequence positions
   - Row 0: attends to positions 0
   - Row 1: attends to positions 0-1
   - Row 2: attends to positions 0-2
4. softmax_triangle(out_, ...)
5. compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step)
```

### 3. **Cache Overflow 처리**

```
from=5, to=8193 (max_timestep=8192)
cache_shift = true
from = 8191 (마지막 유효 cache 항목)
to = 8192

Process:
1. 가장 오래된 항목 삭제 (cache_shift)
2. 새로운 항목만 처리
3. Cache 재정렬
```

---

## 요약

`mha_core.cpp`는 T5Gemma2의 MHA core를 C++로 최적화한 구현입니다. 주요 특징은:

1. **Incremental Inference**: 효율적인 decoding 지원
2. **GQA (Grouped Query Attention)**: KV cache 메모리 절약
3. **Sliding Window Attention**: 긴 시퀀스 지원
4. **Triangle Optimization**: Causal mask 계산 최적화
5. **FP16 Quantization**: KV cache FP16 저장
6. **Multi-threading**: OpenMP + Thread pool 병렬 처리
7. **RoPE**: Precomputed cos/sin lookup (default/YARN 지원)
8. **Softcap**: Attention logit 제한으로 numerical stability

이 구현은 PyTorch의 T5Gemma2와 동일한 기능을 C++로 구현하여 더 나은 최적화와 성능을 달성합니다.