# `if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)` 의미 설명

## 코드

```cpp
// causal_lm.cpp
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();
```

## 각 조건의 의미

### 1. `USE_KVCACHE` - KV Cache 기능 사용 여부

**의미:** KV Cache 기능이 활성화되어 있는지 확인

```cpp
USE_KVCACHE = false;  // 기본값

if (nntr_cfg.contains("system_prompt") &&
    nntr_cfg["system_prompt"].contains("kvcache")) {
  USE_KVCACHE = true;
  PRE_COMPUTED_CACHE_PATH = nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
  if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
    SYS_PROMP_LEN = nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"].get<unsigned int>();
}
```

**nntr_config.json 예시:**
```json
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin",
      "sys_prompt_token_size": 128
    }
  }
}
```

### 2. `!SAVE_KVCACHE` - KV Cache 저장 모드가 아닌지

**의미:** 현재 KV Cache를 저장하는 중인지 확인

```cpp
SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));
```

**설명:**
- `SAVE_KVCACHE = true`: 캐시 파일이 없으므로 생성 후 저장
- `SAVE_KVCACHE = false`: 캐시 파일이 있으므로 로드해서 사용

### 3. `SYS_PROMP_LEN == 0` - 시스템 프롬프트 길이가 설정되지 않았는지

**의미:** config 파일에 `sys_prompt_token_size`가 설정되지 않았는지 확인

```cpp
SYS_PROMP_LEN = 0;  // 기본값

// config 파일에서 읽어옴
if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
  SYS_PROMP_LEN = nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"].get<unsigned int>();
```

## 전체 조건의 의미

```cpp
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();
```

**의미:**
1. KV Cache 기능을 사용하고 (`USE_KVCACHE == true`)
2. 캐시를 저장하는 중이 아니라 (`!SAVE_KVCACHE == true`, 즉 로드하는 중)
3. config 파일에 시스템 프롬프트 길이가 설정되지 않았으면 (`SYS_PROMP_LEN == 0`)

**동작:** 시스템 프롬프트를 토큰화해서 그 길이를 `SYS_PROMP_LEN`에 저장

## 왜 필요한가요?

### KV Cache 기능

**KV Cache**는 Attention 메커니즘에서 Key와 Value 텐서를 캐싱하는 기술입니다:

```cpp
// Self-Attention
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
//                    ↑   ↑
//                    K   V: 캐싱해서 재사용
```

### 시스템 프롬프트 최적화

**문제:** 시스템 프롬프트가 매우 긴 경우 (예: 128개 토큰)

```
시스템 프롬프트: "You are a helpful assistant. You should answer questions in a friendly manner..." (128 토큰)
사용자 프롬프트: "What is the capital of France?" (7 토큰)
```

**각 쿼리에서 시스템 프롬프트를 매번 처리하면 비효율적:**
```
쿼리 1: [128 토큰 시스템] + [7 토큰 질문] → [생성]
쿼리 2: [128 토큰 시스템] + [5 토큰 질문] → [생성]  // 시스템 프롬프트 다시 계산!
쿼리 3: [128 토큰 시스템] + [8 토큰 질문] → [생성]  // 시스템 프롬프트 다시 계산!
```

**KV Cache를 사용하면:**
```
1. 처음: 시스템 프롬프트 처리 → KV Cache 저장
2. 이후: 시스템 프롬프트의 KV Cache 로드 → 사용자 프롬프트만 처리

쿼리 1: 시스템 프롬프트 처리 → KV Cache 저장 (초기 실행)
쿼리 2: KV Cache 로드 + [5 토큰 질문] 처리 → [생성]  // 훨씬 빠름!
쿼리 3: KV Cache 로드 + [8 토큰 질문] 처리 → [생성]  // 훨씬 빠름!
```

### SYS_PROMP_LEN의 역할

`SYS_PROMP_LEN`은 시스템 프롬프트의 토큰 길이를 나타냅니다:

```cpp
// incremental_inference 호출
output = model->incremental_inference(
  BATCH_SIZE, input, label, init_len,
  SYS_PROMP_LEN,                    // ← 시작 위치 (시스템 프롬프트 길이)
  SYS_PROMP_LEN + input_len, false);
```

**시퀀스 구조:**
```
|<---------------- MAX_SEQ_LEN ------------------>|
||              ||              ||
||<-- 시스템 -->||<-- 입력 -->||<-- 생성 -->||
||  프롬프트  ||  토큰    ||  토큰  ||
||              ||              ||
||   128개    ||   7개     ||  100개  ||
||              ||              ||
||  (캐싱)    ||  (계산)   ||  (계산) ||
||              ||              ||
0              SYS_PROMP_LEN  input_len
               (128)           (135)
```

## 작동 흐름

### 시나리오 1: 처음 실행 (캐시 파일 없음)

```cpp
// 1. config.json
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin"
      // sys_prompt_token_size 없음
    }
  }
}

// 2. 초기화
USE_KVCACHE = true;
SAVE_KVCACHE = true;  // 캐시 파일 없음 → 저장 모드
SYS_PROMP_LEN = 0;    // config에 설정 없음

// 3. 조건 체크
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  // !SAVE_KVCACHE == false 이므로 이 조건 통과 안 함
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();  // 실행 안 됨

// 4. KV Cache 저장
output = model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                      0, input_len, false);
SYS_PROMP_LEN = input_len;  // 시스템 프롬프트 길이 저장
save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

// 5. 결과
// 캐시 파일 생성: kvcache.bin
// 사용자 메시지: "sys_prompt_token_size를 128로 설정하세요"
```

### 시나리오 2: 두 번째 실행 (캐시 파일 있음, config에 길이 설정됨)

```cpp
// 1. config.json
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin",
      "sys_prompt_token_size": 128
    }
  }
}

// 2. 초기화
USE_KVCACHE = true;
SAVE_KVCACHE = false;  // 캐시 파일 있음 → 로드 모드
SYS_PROMP_LEN = 128;   // config에서 읽어옴

// 3. 조건 체크
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  // SYS_PROMP_LEN == 0이 false 이므로 이 조건 통과 안 함
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();  // 실행 안 됨

// 4. KV Cache 로드
load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

// 5. 추론
output = model->incremental_inference(BATCH_SIZE, input, label, init_len,
                                      SYS_PROMP_LEN,  // 128부터 시작
                                      SYS_PROMP_LEN + input_len, false);
```

### 시나리오 3: 두 번째 실행 (캐시 파일 있음, config에 길이 설정 안 됨)

```cpp
// 1. config.json
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin"
      // sys_prompt_token_size 없음
    }
  }
}

// 2. 초기화
USE_KVCACHE = true;
SAVE_KVCACHE = false;  // 캐시 파일 있음 → 로드 모드
SYS_PROMP_LEN = 0;     // config에 설정 없음

// 3. 조건 체크
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  // 모든 조건 true!
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();  // 실행됨!

// 4. 시스템 프롬프트 토큰화
// system_prompt: "You are a helpful assistant..."
// tokenizer->Encode(system_prompt): [1234, 5678, 9012, ...]  // 128개 토큰
// SYS_PROMP_LEN = 128;

// 5. KV Cache 로드
load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);  // 128

// 6. 추론
output = model->incremental_inference(BATCH_SIZE, input, label, init_len,
                                      SYS_PROMP_LEN,  // 128부터 시작
                                      SYS_PROMP_LEN + input_len, false);
```

## 왜 이런 복잡한 조건이 필요한가요?

### 목적: 자동으로 SYS_PROMP_LEN 결정

**이 조건의 목적:**
1. 사용자가 매번 config 파일에 `sys_prompt_token_size`를 수동으로 설정하지 않아도 됨
2. 첫 번째 실행 후에는 캐시 파일이 있으므로 `!SAVE_KVCACHE`가 false가 됨
3. 두 번째 실행부터는 이 조건을 통해 자동으로 시스템 프롬프트 길이를 계산

### 진리표

| USE_KVCACHE | SAVE_KVCACHE | SYS_PROMP_LEN == 0 | 조건 결과 | 동작 |
|------------|-------------|-------------------|---------|------|
| false | any | any | false | KV Cache 사용 안 함 |
| true | true | any | false | 캐시 저장 모드 (첫 실행) |
| true | false | false | false | config에 길이 이미 설정됨 |
| true | false | true | **true** | **시스템 프롬프트 길이 자동 계산** |

## 실제 사용 예시

### 1. 처음 실행 (캐시 생성)

```bash
# config.json
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin"
    }
  }
}

# 실행
./main

# 출력
==============
[KV CACHE SAVE MODE]
===============

kv caches are saved in kvcache.bin
and the size of prompt is 128.
You may need this prompt lenth to set the "sys_prompt_token_size"
==================================================

# 사용자가 config.json 수정 필요:
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin",
      "sys_prompt_token_size": 128  // ← 추가
    }
  }
}
```

### 2. 이후 실행 (캐시 로드)

```bash
# config.json (이미 sys_prompt_token_size 설정됨)
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin",
      "sys_prompt_token_size": 128
    }
  }
}

# 실행
./main

# 조건: USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0
# true && true && false = false
# → SYS_PROMP_LEN 자동 계산 안 함 (이미 128로 설정되어 있으므로)

# KV Cache 로드
load_kvcache("kvcache.bin", 128);

# 추론
output = model->incremental_inference(..., 128, 128 + input_len, false);
```

### 3. 이후 실행 (config 수정 안 함)

```bash
# config.json (sys_prompt_token_size 설정 안 됨)
{
  "system_prompt": {
    "kvcache": {
      "pre_computed_cache_path": "kvcache.bin"
    }
  }
}

# 실행
./main

# 조건: USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0
# true && true && true = **true**
# → SYS_PROMP_LEN 자동 계산!

SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();
// system_prompt: "You are a helpful assistant..."
// 결과: SYS_PROMP_LEN = 128

# KV Cache 로드
load_kvcache("kvcache.bin", 128);

# 추론
output = model->incremental_inference(..., 128, 128 + input_len, false);
```

## 요약

### 이 조건의 핵심 역할

```cpp
if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
  SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();
```

**역할:** KV Cache 기능을 사용할 때, config 파일에 `sys_prompt_token_size`가 설정되지 않은 경우 자동으로 시스템 프롬프트의 토큰 길이를 계산

**언제 실행됨:**
1. KV Cache 기능 사용 (`USE_KVCACHE == true`)
2. 캐시 로드 모드 (`!SAVE_KVCACHE == true`, 캐시 파일 있음)
3. config에 길이 설정 안 됨 (`SYS_PROMP_LEN == 0`)

**왜 필요한가:**
- 사용자 편의성: 매번 config에 길이를 수동으로 설정하지 않아도 됨
- 자동화: 첫 실행 후에는 자동으로 길이를 계산
- 유연성: 시스템 프롬프트가 변경되어도 자동으로 길이 업데이트

### 최적화 효과

**KV Cache 사용 전:**
```
각 쿼리마다 시스템 프롬프트 처리 (128 토큰 × N 레이어)
총 시간: 100ms (프리필) + 10ms × 10 (생성) = 200ms
```

**KV Cache 사용 후:**
```
첫 쿼리: 시스템 프롬프트 처리 (128 토큰 × N 레이어)
이후 쿼리: 시스템 프롬프트의 KV Cache 로드 (즉시)
총 시간: 100ms (프리필) + 1ms (캐시 로드) + 10ms × 10 (생성) = 110ms
성능 향상: 45% (90ms 절약)
```

이 조건은 KV Cache 기능을 효율적으로 사용하기 위한 중요한 최적화 기술입니다!
