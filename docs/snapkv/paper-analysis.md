# SnapKV 논문 분석과 구현 해석

이 문서는 `SnapKV.pdf`를 nntrainer의 CPU KV cache eviction 기능으로 옮기기 전에 필요한 논문 사실, 저자 구현의 구체적 동작, 논문이 답하지 않는 사항을 분리해 기록한다. 코드 명세나 구현 계획의 입력 문서이며, 이 문서 자체가 nntrainer API를 확정하지는 않는다.

## 1. 조사 범위와 근거 수준

### 1.1 읽은 자료

- 로컬 주 근거: `C:\nntrainer\SnapKV.pdf`, 17쪽. 표지에 `arXiv:2404.14469v2`, 2024-06-17 개정판으로 표시되어 있다.
- 온라인 원문 교차검증: [arXiv 2404.14469](https://arxiv.org/abs/2404.14469), v2는 2024-06-17 개정.
- 출판판 교차검증: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html) 및 [24쪽 출판 PDF](https://papers.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf). 로컬 v2 이후 hyperparameter sensitivity와 prefill overhead 부록이 추가되어 있다.
- 메타데이터 및 기계 판독 교차검증: [Hugging Face paper page](https://huggingface.co/papers/2404.14469)와 paper API/Markdown.
- 저자 공개 구현: [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV), 조사 시점 `main` SHA `e216ddc84c5bd210378cbdbbba12ba02102aa640`.
- 독립 구현 교차검증: [NVIDIA/kvpress의 SnapKVPress](https://github.com/NVIDIA/kvpress/blob/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a/kvpress/presses/snapkv_press.py), 특히 GQA score aggregation.
- GQA 미해결 사항: 저자 저장소의 [issue #22](https://github.com/FasterDecoding/SnapKV/issues/22).

로컬 PDF의 17쪽을 모두 텍스트 추출하고 110 DPI로 렌더링해 표, 수식, 그림, 부록을 시각 검토했다. 알고리즘의 직접 근거는 로컬 PDF 4쪽의 식 (1)-(8), 5쪽의 Listing 1, 6쪽의 Section 4.3이다.

### 1.2 이 문서의 표기

- **[논문]**: 로컬 arXiv v2 또는 NeurIPS 출판판이 직접 명시한다.
- **[저자 구현]**: FasterDecoding 공개 코드가 실제로 수행한다. 논문 명세와 동일하다는 뜻은 아니다.
- **[독립 구현]**: NVIDIA kvpress처럼 저자가 아닌 구현의 해석이다.
- **[해석]**: 수식과 코드에서 논리적으로 도출되지만 논문이 문장으로 명시하지 않는다.
- **[권고]**: nntrainer 명세에서 명시적으로 선택할 동작이다.

이 구분은 중요하다. 특히 GQA/MQA, padding mask, top-k tie, cache position bookkeeping은 논문만으로 결정할 수 없다.

## 2. 한 문장 요약과 범위

SnapKV는 **prompt의 마지막 `W`개 query가 과거 prefix token에 주는 attention을 head별로 합산하고, token 축 pooling 후 상위 `k`개 prefix KV를 고른 뒤, 마지막 `W`개 prompt KV를 무조건 보존하는 one-shot prompt-cache 압축법**이다.

핵심 범위는 다음과 같다.

- 압축 대상은 prompt KV cache이다.
- 압축 시점은 전체 prompt encoding이 끝난 직후, generation 전에 한 번이다.
- decode 중에는 score를 다시 계산하거나 기존 prompt/generated KV를 다시 evict하지 않는다.
- 보존되는 prompt cache 크기만 입력 길이와 무관하게 고정된다. generated KV는 계속 append되므로 전체 cache가 영원히 상수 크기인 것은 아니다.
- full prompt를 먼저 처리해야 하므로 모델의 본래 context 한계를 늘리지 않으며, prompt prefill 자체와 그 순간의 full-context 요구량을 제거하지 않는다.
- 학습이나 fine-tuning은 필요 없다.

근거: 로컬 PDF 1-2쪽 요약 및 Figure 1, 4-6쪽 Section 4, 7쪽 Section 5.1.2, 12쪽 Section 6.

## 3. 논문이 출발한 관찰

### 3.1 generation 전에 중요한 prefix 위치를 예측할 수 있다

[논문] UltraChat에서 prompt 길이 3K 초과, response 길이 512 초과인 sample을 사용했다. prompt attention을 128-token window로 나누고 마지막 20개 window가 고른 중요한 prompt 위치와 실제 generation이 사용한 위치의 overlap을 비교했다. 마지막 prompt window가 generation의 선택과 가장 높은 유사도를 보였다.

- Figure 2의 표기: sample 3,050개, 평균 prompt 길이 3,263.80, 평균 turn 4.13, 평균 context 길이 955.78.
- 근거: 로컬 PDF 3-4쪽, Section 3과 Figure 2.

### 3.2 선택 패턴은 generation 동안 비교적 안정적이다

[논문] 생성된 512 token을 128-token window 4개로 나누고, 각 generation window가 중요하다고 본 prompt 위치와 마지막 prompt window가 고른 위치의 overlap을 layer별로 비교했다. 대체로 높은 overlap을 보였다.

- 근거: 로컬 PDF 3-4쪽, Figure 3.

### 3.3 중요 위치는 질문/문맥에 따라 달라지지만 질문 위치에는 비교적 둔감하다

- 같은 문서라도 질문이 달라지면 선택되는 prefix 위치가 달라진다. 따라서 고정 attention sink나 corpus 공통 중요도 같은 context-independent 정책으로 대체할 수 없다.
- 질문이 긴 context 앞에 있든 뒤에 있든 hit rate는 높게 유지되었다.
- 근거: 로컬 PDF 5-6쪽, Section 4.2와 Figures 4-5.

### 3.4 관찰 실험은 완전한 재현 명세가 아니다

중요 위치를 정의하는 hit-rate 식에는 threshold `theta`가 등장하지만 그 값은 제시되지 않는다. Figure 2-5를 재현하는 top-k 수, threshold, 세부 mask도 충분히 명시되지 않는다. `theta`는 SnapKV runtime 알고리즘의 parameter가 아니라 관찰을 평가하기 위한 parameter이다.

또한 Section 4.2.1의 robustness 분석은 서로 다른 "instruction-response pair"가 고른 위치를 비교하며 observation window가 instruction과 해당 response를 함께 포함한다고 서술한다. 이는 이미 정답 response를 가진 분석용 구성이고 runtime SnapKV의 입력 계약이 아니다. 실제 알고리즘, Figure 1, Listing 1, 공개 구현은 **response 생성 전 prompt의 마지막 `W`개 token만** observation으로 사용한다. 구현이 미래 response를 요구하거나 teacher-forced response를 observation에 넣어서는 안 된다.

## 4. 용어와 정확한 크기 관계

단일 layer와 단일 sample부터 정의한다. batch 축은 뒤에서 확장한다.

| 기호 | 의미 |
|---|---|
| `L` 또는 `L_prompt` | 전체 prompt token 수 |
| `W` 또는 `L_obs` | prompt 끝의 observation window token 수 |
| `P` 또는 `L_prefix` | observation 이전 prefix 길이, `P = L - W` |
| `D` | attention head dimension |
| `H_q` | query/attention head 수 |
| `H_kv` | key/value head 수. MHA에서는 `H_q = H_kv` |
| `C` | 압축 후 **전체 prompt KV capacity**, observation을 포함 |
| `k` | 선택할 prefix 위치 수. capacity 방식에서는 `k = C - W` |
| `G` | 1D pooling kernel size |
| `p` | 논문 식 (3)의 prefix retention fraction으로 해석되는 값 |

[논문] 식 (1)은 `L = P + W`이다. 식 (3)은 `k = floor(p * P)`라고 설명한다. Listing 1과 공개 구현은 비율 대신 absolute capacity를 받아 `k = C - W`를 사용한다.

따라서 두 parameterization의 관계는 다음과 같다.

```text
C = floor(p * P) + W
p ~= (C - W) / P
```

논문은 `p`를 "compression rate"라 부르지만 식에서는 남겨 둘 prefix 비율로 동작한다. 반면 실험 본문은 `(L - C) / L`을 92% compression처럼 제거율로 부른다. 구현 API에서는 혼동을 피하기 위해 `max_prompt_cache_tokens = C` 같은 absolute capacity를 기본 계약으로 삼는 것이 안전하다.

주의할 문구 불일치도 있다. 로컬 PDF 7쪽은 capacity 1024에서 "가장 중요한 1024 attention features를 선택"한다고 쓰지만 Listing 1의 실제 결과는 `1024 - W`개 prefix와 `W`개 observation, 합계 1024개이다.

## 5. MHA에서의 엄밀한 알고리즘

### 5.1 입력 상태

MHA를 먼저 가정하면 per-layer tensor는 다음 논리 shape를 갖는다.

```text
Q, K, V: [B, H, L, D]
```

여기서 `Q`와 `K`는 모델이 실제 attention에 사용하는 positional encoding 적용 후 상태여야 한다. 저자 Llama/Mistral 코드는 RoPE 적용 후 SnapKV score를 계산한다.

각 layer, batch item, head는 독립적으로 다른 token index 집합을 고른다. "attention feature"는 channel dimension이 아니라 **그 head의 prompt token/KV position**을 뜻한다.

### 5.2 압축하지 않는 경우

[논문/저자 구현] Listing 1과 공개 코드는 `L < C`이면 입력 K/V를 그대로 반환한다. `L == C`이면 압축 경로로 들어가 전체 prefix를 top-k로 다시 모으므로 크기는 줄지 않지만 prefix 물리 순서는 바뀔 수 있다.

[권고] `L <= C`이면 그대로 반환하는 편이 낫다. 보존 집합과 attention 수학은 같고, 불필요한 score 계산과 순서 변경을 피한다. 저자 코드와 byte-level cache order를 그대로 재현해야 하는 테스트에서만 `L < C` 경계를 사용한다.

### 5.3 observation query와 full-prompt key의 score

마지막 `W`개 query를 취한다.

```text
Q_obs = Q[:, :, P:L, :]                    # [B, H, W, D]
S = Q_obs @ transpose(K, token, D) / sqrt(D)  # [B, H, W, L]
```

observation 내부의 query `i`는 absolute prompt position `P + i`에 해당한다. causal model이라면 key `j <= P + i`만 허용한다.

```text
S[b,h,i,j] = -inf  if j > P + i
```

모델의 padding, packed-sequence, block-sparse 등 추가 attention mask가 있으면 그것도 softmax 전에 동일하게 적용해야 한다.

그 다음 허용된 **전체 prompt key 축 `L`**에 대해 softmax한다.

```text
A = softmax(S, axis=key)                   # [B, H, W, L]
```

이후 prefix 열만 취해 observation query 축을 합한다.

```text
vote[b,h,j] = sum(i=0..W-1, A[b,h,i,j]),  0 <= j < P
vote: [B, H, P]
```

중요한 세부사항은 다음과 같다.

- softmax를 prefix에 대해서만 새로 normalize하지 않는다.
- 허용된 observation-window key가 차지하는 확률 질량이 prefix vote에 영향을 준다.
- sum과 mean은 모든 sample/head에서 유효 query 수가 같은 경우 top-k ranking이 같다. 논문/저자 구현은 sum, NVIDIA kvpress는 mean을 사용한다.
- 논문 식 (2)는 `i=0`부터 `L_obs`까지 합한다고 인쇄되어 있어 문자 그대로면 `W+1`개인 off-by-one이다. tensor shape와 Listing 1에 따라 `0..W-1`, 정확히 `W`개 query로 해석해야 한다.
- 논문 문장의 "across all heads"는 head 간 score 합산을 뜻하지 않는다. 식 (3), Listing 1, 공개 구현 모두 각 head에서 독립 top-k를 수행한다.

근거: 로컬 PDF 4쪽 식 (2)-(3), 5쪽 Listing 1의 lines 8-11. 구체 mask/float32 softmax는 [저자 구현 lines 45-55](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L45-L55).

### 5.4 1D pooling이 의미하는 "clustering"

vote의 token 축에 stride 1, same padding의 1D pooling을 적용한다.

홀수 kernel `G = 2r + 1`일 때:

```text
avg:  pooled[j] = (1 / G) * sum(delta=-r..r, vote[j + delta])
max:  pooled[j] = max(delta=-r..r, vote[j + delta])
```

범위 밖은 PyTorch 호환 avg-pool에서는 0으로 padding하고 분모 `G`에 포함하며, max-pool에서는 음의 무한대에 해당하는 값으로 취급한다.

이 연산은 cluster를 검출하고 cluster 단위로 하나만 남기는 알고리즘이 아니다. 높은 vote를 이웃으로 퍼뜨린 뒤 여전히 개별 position top-k를 한다. 결과적으로 중요한 token 주변의 연속 token이 함께 뽑힐 확률이 커진다.

특성:

- 선택 수는 중복 없이 정확히 `k`개이다.
- 여러 중요 위치의 pooling 범위가 겹칠 수 있다.
- 이웃 cluster를 위한 별도 union/dedup 또는 예산 재분배는 없다.
- `G=1`은 pooling 없는 SnapKV와 같다.
- `padding = floor(G/2)`로 출력 길이를 `P`와 같게 유지하려면 `G`가 홀수여야 한다. 공개 코드는 odd 여부를 검사하지 않지만 even `G`는 PyTorch식 출력 길이가 `P+1`이 될 수 있어 out-of-range gather 위험이 있다.

[권고] CPU 첫 구현은 `G >= 1`인 홀수만 허용하고, avg/max edge padding semantics를 테스트로 고정한다.

근거: 로컬 PDF 5쪽 Listing 1 line 13, 6쪽 Section 4.3, 8쪽 Section 5.2. 저자 코드 [lines 56-61](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L56-L61).

### 5.5 top-k와 K/V gather

각 batch/head에서 prefix 후보 `P`개 중 pooled score 상위 `k = C - W`개 index를 고른다.

```text
I = top_k_indices(pooled, k)               # [B, H, k]
K_keep = gather(K[:, :, 0:P, :], I)        # [B, H, k, D]
V_keep = gather(V[:, :, 0:P, :], I)
K_obs = K[:, :, P:L, :]                    # [B, H, W, D]
V_obs = V[:, :, P:L, :]
K_cache = concat(K_keep, K_obs, token_axis)  # [B, H, C, D]
V_cache = concat(V_keep, V_obs, token_axis)
```

반드시 같은 `I`를 K와 V 양쪽에 적용한다. observation `W`개는 score와 무관하게 전부 보존한다.

#### 실제 물리 순서

PyTorch `topk`의 기본 `sorted=True`를 사용하는 저자 구현은 선택 prefix를 원래 token 순서가 아니라 **pooled 중요도 내림차순**으로 gather한다. 그 뒤 observation을 원래 시간 순서로 붙인다.

```text
[selected prefix in per-head score order] [observation chronological]
```

generation token은 이후 뒤에 시간 순서로 append된다.

```text
[selected prefix by score] [prompt observation] [generated tokens]
```

따라서 cache slot `s`가 모든 head에서 같은 original token을 뜻하지 않는다. 같은 slot의 original position은 layer/head/sample마다 다를 수 있다.

[해석] 이미 absolute RoPE가 적용된 K/V를 함께 같은 permutation으로 옮기고 decode attention이 모든 cache entry를 과거로 취급한다면, attention의 집합 연산은 순열에 불변이다. prefix index를 오름차순으로 다시 정렬해도 이상적 실수 연산 결과는 같지만 floating reduction order는 달라질 수 있고 저자 구현과 cache order parity는 잃는다. ALiBi나 cache slot으로 relative bias를 재계산하는 구현은 original position metadata 없이 순서를 바꾸면 안 된다.

top-k 동점 처리 방식은 논문에 없다. PyTorch도 device에 따라 tie ordering을 보장하지 않는다. CPU 구현의 재현성을 위해 `(score descending, original index ascending)` 같은 tie-break를 명세에 넣을지 결정해야 한다.

근거: 로컬 PDF 5쪽 Listing 1 lines 14-24, 저자 코드 [lines 62-70](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L62-L70).

## 6. prompt encoding과 decoding의 정확한 시점

### 6.1 prefill은 full prompt로 정상 계산해야 한다

SnapKV는 prompt attention 결과 자체를 근사하는 방법이 아니다. 각 layer의 prompt forward는 full K/V를 사용해 원래 hidden state를 계산하고, **미래 decode를 위해 저장할 past cache만** 압축해야 한다.

저자 Llama 구현도 prefill에서 다음 순서를 갖는다.

1. full prompt Q/K/V와 RoPE를 계산한다.
2. K/V를 압축해 `past_key_value`에는 compressed cache를 저장한다.
3. 현재 prefill attention output은 지역 변수에 남아 있는 full prompt K/V로 계산한다.

관련 코드는 [llama_hijack lines 73-96](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L73-L96)이다.

이 불변식을 깨고 prefill의 마지막 `W` query output까지 compressed K/V로 다시 계산하면 저자 구현과 다른 알고리즘이 된다.

### 6.2 첫 생성 token과 이후 token

일반적인 causal generation에서 첫 생성 token은 full prompt forward의 마지막 logit에서 sample된다. 따라서 첫 token 선택은 압축의 영향을 받지 않는다. 그 token을 다음 forward의 input으로 넣는 시점부터 query가 compressed prompt cache를 본다.

decode forward마다:

- 새 token의 K/V를 cache 뒤에 append한다.
- query는 선택된 prefix, 전체 prompt observation, 앞서 생성된 모든 token을 본다.
- prompt prefix를 다시 scoring하지 않는다.
- observation window를 slide하지 않는다.
- generated token을 evict하지 않는다.

### 6.3 물리 길이와 논리 position은 다르다

`L > C`인 prompt를 압축했다고 하자. prefill 직후:

```text
physical cache length = C
logical next position = L
```

그 후 `t`개의 generated token이 decode forward를 거쳐 cache에 들어갔다면:

```text
physical cache length = C + t
logical next position = L + t
```

RoPE position, input token trimming, stop condition 등에 physical cache length `C+t`를 사용하면 position이 왼쪽으로 당겨지는 오류가 난다. original prompt length와 generated count를 별도 logical length로 추적해야 한다. 저자 구현도 `kv_seq_len`을 별도로 유지하려고 한다.

이미 RoPE가 적용된 selected K는 원래 prompt absolute position의 회전을 그대로 보존한다. decode query도 `L+t`라는 원래 논리 위치로 회전해야 한다.

### 6.4 "constant cache"의 정확한 뜻

논문이 constant라고 부르는 것은 **prompt에서 유래한 cache 부분 `C`**이다. 전체 cache는 `C+t`로 자란다. 논문 7쪽도 "compressed KV cache size of prompt stays the same"이라고 한정하고 "inference 동안 extra update가 없다"고 설명한다.

그러므로 max generation length까지 memory를 계획할 때는 `C + max_new_tokens`를 고려해야 한다.

### 6.5 one-shot과 chunked prefill 제약

원 알고리즘은 전체 prompt의 마지막 `W` query와 전체 prefix K가 동시에 필요하다. 따라서 기본 계약은:

- prompt boundary를 알고,
- full prompt KV를 끝까지 유지하고,
- prompt가 완전히 encode된 뒤 한 번만 압축한다.

저자 integration도 full-prompt prefill과 subsequent decode를 sequence-length equality로 구분한다. chunked prefill에서 첫 chunk를 prompt 전체로 오인해 조기 압축하면 이후 chunk가 decode token처럼 단순 append되어 잘못된 선택이 된다.

nntrainer가 prompt를 token-by-token 또는 chunk로 처리한다면 가능한 정확한 구현은 두 가지다.

- prompt 전체 종료까지 eviction을 미루고 마지막 `W` query의 실제 attention row를 보관한 뒤 압축한다.
- full KV와 마지막 `W` Q를 보관하고 prompt 종료 시 score를 재계산한다.

어느 경우든 prompt 처리 중 peak KV 요구량은 full prompt와 같다. chunk 중간 eviction은 원 논문 범위 밖이다.

## 7. mask, softmax, padding의 세부 계약

### 7.1 causal mask

마지막 `W` query 대 전체 K score에는 다음 mask가 필요하다.

- 모든 prefix key `0..P-1`: 모든 observation query에 허용.
- observation key `P+j`: query `P+i`에 대해 `j <= i`일 때만 허용.
- 미래 observation key `j > i`: 금지.

저자 구현은 정확히 observation-by-observation `W x W` lower triangle을 만들고 전체 score의 오른쪽 아래 block에 더한다.

### 7.2 softmax 범위와 정밀도

저자 구현은 scaled dot product를 만든 뒤 float32로 softmax하고 Q dtype으로 다시 cast한다. CPU 구현에서 모델의 기존 attention softmax 정밀도와 일치시키는 것이 가장 안전하다.

score를 위해 기존 attention probability를 재사용할 수 있다면 그 값을 쓰는 것이 mask/scaling 일치를 보장하고 중복 matmul을 줄인다. FlashAttention처럼 attention matrix를 반환하지 않으면 `W x L` score만 별도 계산한다.

### 7.3 padding과 variable-length batch

논문 수식은 "single batch of sequence"를 설명하고 padding 처리를 정의하지 않는다. 저자 `update_kv`는 인자로 받은 `attention_mask`를 새 causal mask로 덮어쓰므로 padding mask를 score에 반영하지 않는다. 따라서 공개 코드가 일반적인 variable-length padded batch에 대한 정확한 근거가 되지는 못한다.

정확한 batch 확장에는 sample별 valid prompt length가 필요하다.

- observation은 각 sample의 마지막 `W`개 **유효 token**이어야 한다.
- masked/padding key는 softmax와 top-k 후보 양쪽에서 제외해야 한다.
- sample마다 `L`, `P`, 필요하면 `k`가 달라질 수 있다.
- 고정 dense output을 원하면 per-sample cache length metadata 또는 padding된 compressed cache가 필요하다.

decode attention mask도 원래 길이 `L+t`의 token mask를 그대로 physical key 축 `C+t`에 적용할 수 없다. 선택된 prefix/observation/generated slot에 대응하는 compressed valid mask를 만들거나, batch size 1에서 모든 stored slot이 유효하다는 전제로 key padding mask를 생략해야 한다. position ID 계산에는 반대로 원래 logical attention/token history가 필요하므로, "decode용 physical key-valid mask"와 "logical position 계산용 token history"를 하나의 길이 값으로 합치지 않는다.

[권고] CPU 1차 범위를 batch size 1로 제한하거나, 처음부터 per-sample valid length를 명시적 입력으로 받는다. padding을 조용히 허용하면서 저자 mask를 복제하면 잘못된 KV가 선택될 수 있다.

## 8. layer/head별 선택과 cache 표현

### 8.1 selection granularity

SnapKV 선택은 다음 모두에 독립적이다.

- layer
- batch sample
- attention head

즉 `I[layer,batch,head,slot]`가 필요하다고 생각할 수 있다. 실제 구현은 압축된 K/V 자체를 저장하면 `I`를 decode 동안 유지할 필요는 없지만, debugging과 position-dependent bias에는 original position metadata가 필요할 수 있다.

### 8.2 head별로 token 축 의미가 달라진다

head 0의 compressed slot 3과 head 1의 compressed slot 3이 서로 다른 original prompt position일 수 있다. 따라서 다음 표현은 논문과 맞지 않는다.

- 모든 head에 하나의 공통 global token index list를 적용하는 cache.
- token-major 구조에서 한 token row를 모든 head에 대해 통째로 유지/삭제하는 eviction.

논문 그대로 구현하려면 head별 gather가 가능한 `[head][compressed_slot][D]` 또는 동등한 layout이 필요하다.

## 9. GQA와 MQA: 논문이 비워 둔 핵심

### 9.1 논문은 GQA/MQA를 정의하지 않는다

논문 수식과 Listing 1은 `Q`, `K`, `V`의 head 축이 모두 `N`이라고 가정한다. GQA의 `H_q > H_kv`, MQA의 `H_kv = 1`일 때 여러 query head가 하나의 K/V head를 공유하는 상황을 설명하지 않는다. NeurIPS 출판판에도 GQA/MQA 언급이 없다.

Mistral, Mixtral, Command-R처럼 실험 모델에 GQA 계열이 포함되지만 논문만으로 "공유 KV에 연결된 query head들의 서로 다른 score를 어떻게 하나로 합칠지" 알 수 없다.

### 9.2 저자 구현의 실제 동작

저자 Llama/Mistral integration은 prefill K/V를 `repeat_kv`하여 `[B,H_kv,L,D]`에서 `[B,H_q,L,D]`로 복제한 다음 query-head별 SnapKV top-k를 수행하고, 복제된 head별 cache를 저장한다.

- 장점: paper의 per-attention-head 선택을 문자 그대로 흉내 낸다.
- 단점: 원래 공유하던 K/V가 query head 수만큼 물리적으로 갈라져 GQA/MQA의 cache memory 절감이 사라진다.
- 같은 source KV head에서 출발했더라도 query head별 selected index가 다르므로 압축 후 다시 하나의 공유 KV cache로 합칠 수 없다.
- `num_key_value_groups`가 `update_kv`에 전달되지만 실제로 사용되지 않는다.

이 문제는 2024-10-03 열린 저자 저장소 issue #22에서 지적되었고 조사 시점에도 답변 없이 open이다.

근거: [저자 `repeat_kv`와 head expansion](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L73-L87), [`snapkv_utils.py`의 unused group 인자](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L38-L41).

### 9.3 native GQA/MQA cache를 유지하는 독립 구현 해법

NVIDIA kvpress는 query head별 score를 먼저 계산한 뒤, 같은 KV head를 공유하는 group 안에서 score를 평균하고 KV-head별 top-k를 수행한다.

```text
scores_q:  [B, H_q, P]
reshape:   [B, H_kv, groups, P]
scores_kv = mean(scores_q, axis=groups)     # [B, H_kv, P]
I_kv = top_k(scores_kv, k)
```

group 크기가 모두 같으면 sum과 mean은 ranking 관점에서 동일하다. MQA에서는 모든 query head score를 하나의 KV head score로 평균한다.

- 장점: native `[B,H_kv,C,D]` cache와 GQA/MQA memory 절감을 유지한다.
- 단점: 논문과 저자 구현이 실험한 정확한 semantics라고 입증되지 않은 확장이다.

다른 가능한 정책은 group max, query-head index union, 대표 query head 등이지만 논문 근거가 없다. union은 예산 `k`를 쉽게 초과한다.

### 9.4 nntrainer에 대한 권고

- MHA는 paper-exact per-head top-k로 구현한다.
- GQA/MQA를 1차 범위에 포함한다면 native KV cache를 유지하고 group-mean score 후 KV-head별 top-k를 사용한다.
- 이 동작을 "SnapKV GQA aggregation"으로 명세하고 paper-exact MHA와 구분한다.
- 저자 구현 parity가 반드시 필요하면 `expand_kv_to_query_heads` 같은 별도 mode로 제공하되 memory trade-off를 문서화한다.
- 아무 정책도 확정하지 않은 채 `H_q` index를 `H_kv` cache에 적용해서는 안 된다.

## 10. pooling/default/hyperparameter 근거표

논문에는 모든 모델에 적용되는 단일 default가 없다.

| 출처/실험 | `C` prompt capacity | `W` observation | `G` pool kernel | pooling | 기타 |
|---|---:|---:|---:|---|---|
| LWM Needle-in-a-Haystack, 로컬 7쪽 | 1024 | 16 | 5 | max | 380K prompt pressure test |
| LWM decode speed, 로컬 7쪽 | 2048 | 미기재 | 미기재 | 미기재 | generation length 512 |
| Mistral LongEval-Lines, 로컬 8쪽 | 미기재 | 16 | 5 | max | pooling ablation |
| LongBench 4 models, 로컬 9쪽 | 1024/2048/4096 | 32 | 7 | max | 16 dataset |
| Command-R 전체 실험, 로컬 10쪽 | 4096 | 64 | 13 | 문맥상 pooling 사용, 종류 미기재 | 2x-32x 압축 |
| 저자 공개 integration 기본 | 2048 | 32 | 5 | avg | `init_snapkv` default |
| 저자 `SnapKVCluster` 생성자 자체 기본 | 320 | 64 | 5 | avg | integration default에 보통 덮임 |

저자 integration 기본 근거는 [`init_snapkv` lines 72-87](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py#L72-L87)이다. 생성자 기본과 실제 model config 기본이 서로 다르므로 "공식 default"를 말할 때는 integration default `(C=2048,W=32,G=5,avg)`를 뜻한다고 명확히 해야 한다.

추가 근거:

- 로컬 PDF 9쪽은 max와 average pooling 사이 유의미한 성능 차이를 관찰하지 못했다고 한다.
- NeurIPS 최종판 9쪽 Table 2는 Mistral, `C=1024`에서 `W={16,32,64}`, `G={5,7,9}`, 그리고 no-pool `G=1`을 비교한다. 모든 task에서 하나의 설정이 우세하지 않았다.
- 출판판 baseline은 `W=32,G=7`이고, pooling 사용 설정이 no-pool보다 평가 가능한 9개 task 중 8개에서 높았다.
- 따라서 `W`, `G`, pool type은 model/workload별 tunable이며 논문이 universal optimum을 제공하지 않는다.

[권고] 초기 correctness profile은 논문 LongBench 설정과 가까운 `C=1024 또는 2048`, `W=32`, `G=7`, `max`를 별도 preset으로 둘 수 있다. 저자 코드 parity profile은 `C=2048`, `W=32`, `G=5`, `avg`이다. 둘을 하나의 "논문 기본값"이라고 섞지 않는다.

## 11. 효과와 한계: CPU 구현에 과장 없이 적용하기

### 11.1 논문이 보고한 효과

- 16K input, batch 2, A100-80GB에서 약 3.6x decode speedup.
- 같은 조건의 최대 처리 길이를 16K에서 131K로 늘린 약 8.2x memory-bound 개선.
- LWM-Text-Chat-1M, `C=1024,W=16,G=5`로 single A100-80GB에서 최대 380K NIAH 수행.
- LongBench 16개 dataset에서 full cache와 대체로 비슷한 성능. 1024 capacity에서도 평균 성능 하락이 작았고 H2O 4096보다 여러 task에서 우수.
- Command-R 128K NIAH에서 4096 capacity, 약 32x 압축 시 -0.5% score difference.

근거: 로컬 PDF 1, 7-10쪽, Figures 6-8와 Tables 1-3.

보고 수치에는 문서 내부의 작은 불일치도 있다. LongBench 절은 4개 model의 평균 input 길이가 약 13K라서 `C=1024`가 92%, `C=4096`가 68% 제거율이라고 계산한다. NeurIPS 최종판은 같은 절에서 평균 prompt 길이가 5K-7K라고 쓰고 Appendix D의 dataset 평균도 대체로 4.3K-7.1K를 제시하면서, 13K와 92%/68% 문장도 그대로 유지한다. model별 tokenization/truncation 차이인지 설명이 없다. 따라서 이 비율은 대략적 논문 보고값으로만 인용하고 nntrainer memory 산식은 실제 `L`, `C`, `W`로 계산해야 한다.

### 11.2 CPU에 직접 일반화할 수 없는 것

속도와 memory 수치는 GPU, FlashAttention/Hugging Face 구현, 특정 model/batch에 대한 결과다. CPU에서 같은 배수를 보장하지 않는다. CPU에서는 다음이 별도로 측정되어야 한다.

- 마지막 `W x L` score 계산 비용 또는 기존 attention score 재사용 비용.
- pooling/top-k/gather 비용.
- head별 비연속 gather가 cache locality에 주는 영향.
- 줄어든 decode dot-product와 memory bandwidth의 이득.
- generated length가 짧을 때 one-shot 압축 비용을 상쇄하는지.

### 11.3 prompt 처리의 근본 한계

로컬 PDF 12쪽 Section 6은 SnapKV가 generation 측 KV cache만 겨냥하고 prompt inference 처리를 해결하지 않는다고 명시한다.

- 모델이 원래 긴 prompt를 encode하지 못하면 SnapKV도 못 한다.
- full prompt K/V와 attention이 compression 전까지 필요하다.
- prompt prefill latency를 본질적으로 없애지 않는다.
- full prompt의 일시적 memory allocation이 필요하므로 peak memory가 항상 `C` 수준으로 줄어드는 것은 아니다.

NeurIPS 최종판 13쪽 Appendix B는 H100, 5K-45K Mistral 실험에서 측정 가능한 prefill time/max-memory overhead가 없었다고 보고한다. 이는 GPU 실험 관찰이며, 알고리즘상 full prompt requirement가 사라진다는 뜻이 아니다. 저자 공개 FlashAttention patch는 attention probability가 없어 `Q_obs @ K_full`을 별도 계산하므로 CPU에서는 overhead를 직접 검증해야 한다.

## 12. 논문/공개 구현의 모호점 목록과 제안 해석

| ID | 모호점 | 근거 상태 | 구현 명세에서 권할 해석 |
|---|---|---|---|
| A1 | GQA/MQA group score | 논문 미기재, 저자 코드는 KV 복제, issue #22 open | MHA exact; GQA/MQA는 group-mean per-KV-head를 명시적 확장으로 채택 |
| A2 | `p`가 retention인지 eviction인지 | 식은 retention, 본문 "compression"은 제거율에도 사용 | public API는 absolute `C`; 비율이면 이름을 `prefix_keep_ratio`로 제한 |
| A3 | `pool1d` 종류 | Listing 미기재, 실험 주로 max, 공식 default avg | enum으로 노출하고 preset을 구분 |
| A4 | pool edge padding | 논문 미기재 | PyTorch 호환: avg zero/count-pad, max negative infinity |
| A5 | even kernel | 논문/코드 validation 없음 | odd positive만 허용 |
| A6 | top-k tie | 미기재 | CPU deterministic tie-break를 정하거나 nondeterminism을 허용한다고 명시 |
| A7 | selected prefix order | 논문은 gather만 제시, PyTorch topk는 score order | parity는 score-descending; chronological sort는 별도 선택 |
| A8 | `L == C` | 논문/코드는 compression path | `L <= C` no-op 권고 |
| A9 | padding/variable batch | 논문 미기재, 저자 score mask가 padding 무시 | batch 1 제한 또는 valid-length-aware per-sample 구현 |
| A10 | chunked prefill | 논문 미기재, 원 구조는 full prompt one-shot | explicit prompt-finalize 단계 전에는 압축 금지 |
| A11 | generated KV eviction | 논문은 update 없음 | generated token 모두 append, cache `C+t` |
| A12 | logical position after compaction | 논문 의사코드 미기재, 저자 integration 별도 추적 | logical `L+t`와 physical `C+t`를 분리 |
| A13 | positional encoding 시점 | 의사코드 미기재, 저자 코드는 RoPE 후 | 실제 attention에 쓰는 post-position Q/K로 score; original key position 보존 |
| A14 | attention softmax 범위 | 식은 prompt-softmax subset, 저자 코드는 full prompt softmax 후 prefix slice | full allowed prompt key에 softmax 후 prefix vote |
| A15 | 관찰 식의 합 범위 | 식 (2)가 `0..L_obs`로 off-by-one | 정확히 `W` rows, `0..W-1` |
| A16 | hit threshold `theta` | 값 미기재 | runtime parameter로 만들지 않음; paper observation 재현은 별도 과제 |
| A17 | attention score 재사용 여부 | 논문 `compute_attn`, 출판 부록은 overhead 작다고만 함 | 정확한 기존 score가 있으면 재사용, 없으면 `W x L`만 재계산 |
| A18 | non-causal/packed attention | 논문 causal generation 전제 | 모델의 실제 mask를 그대로 적용하지 못하면 unsupported |
| A19 | robustness 실험의 response 포함 observation | 분석 실험에는 response 포함 문구, runtime은 pre-generation prompt-only | runtime observation에는 prompt token만 사용 |

## 13. 구현 가능한 reference pseudocode

다음은 논문 MHA semantics를 명확히 만든 CPU 지향 pseudocode이다. nntrainer API/자료형은 확정하지 않는다.

```text
compress_prompt_kv(Q, K, V, valid_mask, C, W, G, pool_type):
    # Q,K,V logical shape: [B,H,L,D], already position-encoded.
    require C > W > 0
    require G >= 1 and G is odd
    require Q.shape == K.shape == V.shape for paper-exact MHA

    L = Q.token_length
    if L <= C:
        return K, V

    P = L - W
    k = C - W
    require 0 < k <= P

    for each batch b, head h:
        for i in [0, W):
            qpos = P + i
            for j in [0, L):
                if j > qpos or not valid_mask[b,qpos,j]:
                    logits[i,j] = -infinity
                else:
                    logits[i,j] = dot(Q[b,h,qpos], K[b,h,j]) / sqrt(D)
            prob[i,:] = stable_softmax(logits[i,:])

        for j in [0, P):
            vote[j] = sum_i prob[i,j]

        pooled = pool1d_same(vote, G, pool_type)
        indices = deterministic_top_k(pooled, k)

        for s in [0, k):
            K_out[b,h,s,:] = K[b,h,indices[s],:]
            V_out[b,h,s,:] = V[b,h,indices[s],:]
        for s in [0, W):
            K_out[b,h,k+s,:] = K[b,h,P+s,:]
            V_out[b,h,k+s,:] = V[b,h,P+s,:]

    return K_out, V_out
```

GQA 확장에서는 query-head별 `vote/pooled`을 계산한 뒤 KV group 안에서 pooled score를 평균하고 KV-head별 top-k/gather를 한다. pooling 전 평균과 pooling 후 평균은 avg-pool에서는 선형이라 같지만 max-pool에서는 같지 않다. NVIDIA 구현은 query-head별 avg-pooling 후 group 평균이다. GQA + max-pool의 aggregation 순서는 별도 명세가 필요하며, 독립 구현 precedent를 따르려면 **query-head별 pool 후 group 평균**으로 고정한다.

## 14. 정적 검증과 unit-test oracle로 쓸 불변식

빌드/실행이 불가능한 환경에서도 다음은 코드 리뷰와 작은 tensor reference test로 검증할 수 있다.

### 14.1 입력 validation

- `C > W > 0`.
- `G`는 양의 홀수.
- compression path에서 `L > C`, 따라서 `P >= k > 0`.
- K/V shape, dtype, head dimension이 일치.
- MHA exact mode에서 `H_q == H_kv`; GQA mode에서는 `H_q % H_kv == 0`.

### 14.2 selection 결과

- no-op path의 output은 input과 동일한 shape/content/order.
- compression path output prompt 길이는 정확히 `C`.
- 각 head의 selected prefix index는 `[0,P)` 안에 있고 중복이 없으며 정확히 `k`개.
- 마지막 `W` output slot은 original `[P,L)` K/V와 bitwise 동일.
- K와 V는 동일 index와 동일 순서로 gather.
- masked/padding position은 선택되지 않음.

### 14.3 mask/score

- observation query `i`는 observation key `j <= i`만 볼 수 있음.
- softmax row 합은 유효 key에 대해 1, masked key는 0.
- prefix vote는 softmax 후 prefix slice이며 prefix-only renormalization이 아님.
- avg/max pool output 길이는 `P`.
- `G=1`에서 pooled score는 vote와 동일.

### 14.4 cache lifecycle

- prompt output/logit 계산에는 full prompt K/V가 사용됨.
- prefill 직후 physical prompt cache length는 `C`, logical sequence length는 `L`.
- `t` decode append 뒤 physical length는 `C+t`, logical length는 `L+t`.
- decode 중 SnapKV top-k/pooling 호출 횟수는 0.
- 각 layer/head가 독립 selection을 가질 수 있음.

### 14.5 작은 hand-check 예

`L=8`, `W=2`, `C=5`이면 `P=6`, `k=3`이다.

- score shape: `[H,2,8]`.
- query position 6은 key 0..6만, position 7은 key 0..7을 허용.
- softmax 후 prefix 0..5의 두 row를 합해 길이 6 vote 생성.
- pooled top-3 prefix + original position 6,7 전체를 보존.
- compressed per-head cache length는 5.
- next generated query의 logical position은 8이지 5가 아니다.

이 예는 causal mask off-by-one, capacity가 observation을 포함하는지, logical/physical length 혼동을 동시에 잡는다.

## 15. 결론: nntrainer 명세에 반드시 들어갈 핵심

1. SnapKV는 prompt-finalize 시점의 one-shot, per-layer/per-head prompt cache 압축이다.
2. 마지막 `W` prompt token은 항상 보존하고, prefix에서는 `C-W`개만 고른다.
3. score는 post-position observation Q 대 full prompt K의 causal masked attention이며, full prompt key softmax 후 prefix 열을 합한다.
4. pooling은 token 축 same 1D smoothing이지 명시적 cluster union이 아니다.
5. cache 순서는 저자 parity 기준 `[score-order selected prefix][chronological observation][chronological generated]`이다.
6. prompt의 논리 길이와 압축 cache의 물리 길이를 분리해야 한다.
7. decode 중 prompt를 다시 evict하지 않고 generated KV도 evict하지 않으므로 전체 cache는 `C+t`이다.
8. full prompt prefill output은 full K/V로 계산해야 하며, SnapKV는 prompt 처리 한계나 peak full-context 필요를 없애지 않는다.
9. GQA/MQA는 논문 미정의다. 저자 코드의 KV-head 복제와 native-GQA group aggregation 중 하나를 명시적으로 선택해야 한다.
10. padding, chunked prefill, top-k tie/order, pool edge semantics은 논문 밖의 구현 계약이므로 조용히 추정하지 말고 API와 test에 고정해야 한다.

## 16. 출처별 핵심 위치

### 로컬 `SnapKV.pdf` arXiv v2

- 1-2쪽: 문제 정의, 주요 성능 주장, Figure 1 workflow.
- 3-4쪽: observation 실험과 Figures 2-3.
- 4쪽: 용어, 식 (1)-(8), voting/top-k/hit rate.
- 5쪽: Section 4.1과 Listing 1 전체.
- 6쪽: Section 4.3 pooling 동기.
- 7-8쪽: LWM/LongEval 설정, speed/memory/pooling ablation.
- 9쪽: LongBench capacity, `W=32`, max-pool `G=7`, max-vs-avg 언급.
- 10쪽: Command-R `C=4096,W=64,G=13`.
- 12쪽: generation-only 범위와 prompt inference 한계.
- 16쪽: generation time breakdown.
- 17쪽: capacity 1024/2048/4096 generation 사례.

### NeurIPS 2024 최종판에서 추가로 확인한 사항

- 9쪽 Table 2/Section 5.4: `W`와 `G` sensitivity, universal optimum 없음, pooling의 전반적 이점.
- 13쪽 Appendix B: H100 5K-45K prefill time/max-memory overhead 측정.
- 13쪽 Appendix C: prompt 길이에 따른 generation time breakdown.

### 공개 코드

- [알고리즘과 두 종류 default](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py).
- [Llama prefill/decode cache integration](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py).
- [FlashAttention2-only monkey patch와 Transformers 4.37 대상](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/monkeypatch.py).
- [NVIDIA의 native-GQA score aggregation](https://github.com/NVIDIA/kvpress/blob/8bb3315aa552d2d0b33f38ef0835e68cfa49a11a/kvpress/presses/snapkv_press.py#L95-L103).

마지막 확인일: 2026-08-03.
