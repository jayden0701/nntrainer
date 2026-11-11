# HTP KV Manager / Sparse KV reuse: 정상 성공 경로의 비용 검토

기준: `baek2sm/nntrainer`, `kv/sparse-prefix-reuse-20261110`,
커밋 `73f8feec47e2bab4d7ac39a1f7d46e3f3b501b20`.

이번 검토는 손상, 파일 읽기 실패, 메모리 부족을 주입하지 않았다. 정상 source에서 sparse
reuse가 성공할 때 수행하는 할당·초기화·복사·반복과 소유권 수명을 검토했다.
손상 발생률은 측정하지 않았으며, 앞선 실패 주입 검사를 빈도 자료로 해석하면 안 된다.

## 먼저 판단

가장 명확한 수정 후보는 **회전할 K의 사전 memcpy 제거**, **sliding 범위 밖 행의 반복 제거**,
**destination pages의 reserve**다. 모두 변경 범위가 작다. 그다음은 **변경 없는 전체 page 공유**,
**복원이 끝난 임시 snapshot의 조기 해제**, **page 단위 다중 할당과 0 초기화 개선**이다.
두 번째 전체 복사는 현재 버퍼 구조를 바꿔야 제거할 수 있다.

| 우선도 | 정상 경로에서 확인한 작업 | 판정 |
|---|---|---|
| 높음 | K를 복사한 직후 읽어서 회전하고 전부 덮어씀 | 해당 K memcpy는 제거 가능 |
| 높음 | gather의 모든 payload를 0으로 채운 뒤 전체 overwrite | 초기값은 쓰이지 않음. 저장 형식 변경 필요 |
| 높음 | 변경 없는 선두의 완전한 page까지 새로 할당·복사 | 정렬/소유권 조건이 맞으면 immutable page 공유 가능 |
| 높음 | gather snapshot → live host cache로 전체 복사, 이후 snapshot 보관 | 복사는 현재 구조상 역할이 있음. 보관 기간은 별도 축소 가능 |
| 중간 | sliding 범위 밖 행을 하나씩 순회하며 continue | 루프 시작점을 옮겨 제거 가능 |
| 중간 | page마다 객체 1개와 K/V vector 2개 할당, pages도 반복 증설 | reserve는 즉시 가능. arena/pool은 후속 설계 |
| 중간 | 같은 길이 sparse 후보의 run 수/실제 준비 비용 무시 | 더 싼 후보를 우선할 여지가 있음 |
| 낮음~중간 | publish의 추가 깊은 복사와 full-capacity 때 두 번 rebuild | source 게시 비용. 큰 KV 복사보다 후순위 |
| 낮음 | 고정 RoPE 정보 재생성, token vector 왕복 복사 | 제거 가능하나 우선순위를 부풀리면 안 됨 |

## 실행해서 확인한 범위

실제 `Applications/CausalLM/kv_cache_manager.cpp`의 `gatherSparsePrefix()`를 그대로 빌드하고,
그 함수 본문에서 생성한 계측본 및 네 가지 누적 변경본과 비교했다. production 파일은 수정하지 않았다.

- 작은 입력: FP16/FP32 × split/interleaved/no-RoPE × 여섯 run 배치 × 다섯 함수,
  총 180회 바이트 비교. 부분 page와 정렬되지 않은 삭제 경계를 포함한다.
- 큰 입력: source 24,576행에서 `[4,096, 8,192)`를 삭제하고 20,480행을 재사용한다.
  30 layers, global 5개, sliding 25개, window 1,024행.
- 메모리 사용을 줄이기 위해 큰 입력의 모든 layer는 D64, KV head 1개, FP16로 축소했다.
  31.25MiB destination이다. 실제 Gemma4 전체 모델 실행이 아니다.
- 각 변경본은 원래 함수와 결과 KV가 바이트 단위로 같았고 source checksum도 유지했다.
- 같은 비교는 독립 함수의 ASan/UBSan 빌드에서도 통과했다. sanitizer 빌드는 무관한 engine
  의존성을 제외하기 위해 원래 메서드 본문을 변경 없이 별도 translation unit에 추출했다.
- 계측은 `operator new` 요청 수/크기, `vector::resize`의 값 초기화 크기,
  지정한 memcpy의 크기와 행 루프 실행 수다. DRAM 트래픽, RSS, HTP 지연시간의 측정치가 아니다.

누적 변경이므로 각 행의 효과를 독립적인 개선율처럼 합산하면 안 된다.

| 누적 변경 | 전체 할당 요청 | K 사전 복사 | payload 0 초기화 | 행 루프 실행 | 버린 행 반복 |
|---|---:|---:|---:|---:|---:|
| 기존 함수의 계측본 | 6,237 | 15.625MiB | 31.25MiB | 512,000 | 384,000 |
| 회전할 K 사전 복사 제거 | 6,237 | 2.5MiB | 31.25MiB | 512,000 | 384,000 |
| + destination pages reserve | 6,092 | 2.5MiB | 31.25MiB | 512,000 | 384,000 |
| + sliding 시작점 조정 | 6,092 | 2.5MiB | 31.25MiB | 128,000 | 0 |
| + 선두 exact 전체 page 공유 | 5,132 | 0 | 26.25MiB | 107,840 | 0 |

마지막 행은 전체 K 복사를 없앴다는 뜻이 아니다. 이동한 K는 원본을 읽고 보정 결과를 출력에
기록한다. 없어진 것은 그 연산 전에 하는 별도 memcpy다. 마지막 변경의 행 루프 횟수는
공유 page마다 한 번만 반복하기 때문에 64행씩 처리한 부분과 구분해야 한다.

## 1. 회전할 K의 사전 memcpy는 실제로 중복이다

위치: `Applications/CausalLM/kv_cache_manager.cpp:952–983`.

현재 순서는 다음과 같다.

```text
source K → memcpy → destination K
                        ↓ 다시 읽음
                    float scratch → RoPE → destination K 전체 덮어쓰기
```

source는 immutable이므로 scratch를 source K에서 직접 읽을 수 있다. FP16→FP32 변환,
RoPE 연산, 유한값 검사, FP16 재변환의 순서는 그대로 유지한다. V의 복사와 위치가 변하지
않은 K의 복사는 여전히 필요하다.

```text
source K → float scratch → RoPE → destination K
```

위 호스트 입력에서는 K 사전 복사 15.625MiB 중 13.125MiB를 제거했다.
동일 입력의 실제 Gemma4 geometry로 계산하면, global moved K 160MiB와 sliding moved K
100MiB, 합계 **260MiB의 memcpy payload**에 해당한다. 읽기·쓰기를 각각 세면 520MiB의
소스 코드상 접근량이지만, 이를 실제 DRAM 트래픽 절감량으로 단정하지 않는다.

이 변경은 연산 순서를 바꾸지 않아 비교한 FP16/FP32 사례에서 bitwise 결과를 유지했다.

## 2. 새 payload는 전부 0 초기화한 뒤 덮어쓴다

위치: `kv_cache_manager.cpp:944–950`.

`page->key.resize(bytes)`와 `page->value.resize(bytes)`는 새 unsigned-char 원소를 0으로
값 초기화한다. 성공한 gather는 이 payload의 모든 바이트를 뒤에서 채운다. 부분 마지막 page도
필요한 크기만 할당하므로 별도로 남겨야 할 padding은 없다.

실제 geometry의 20K destination은 K+V **600MiB**다. 이 600MiB를 0으로 채우는 초기값은
성공한 결과에 기여하지 않는다. 다만 단순히 `resize`를 `reserve`로 바꾸고 `data()`에 쓰는 것은
vector의 유효 크기/객체 수명 규칙을 어기므로 해결책이 아니다.

권장 방향은 초기화하지 않는 소유 byte buffer, 또는 destination 전용 arena에 page view를
배치하는 것이다. C++17 및 소유권/예외 처리 조건을 지켜야 한다. 모든 KV slab의 0 초기화를
한꺼번에 없애자는 주장은 아니다. 다른 slab에는 미사용 tail이나 backend의 접근 규칙이 있다.

## 3. 변경 없는 선두 page도 새로 만든다

위치: `kv_cache_manager.cpp:942–955`.

삭제 전의 선두 구간은 `source == destination`이고 K/V가 바뀌지 않는다. 그런데 현재 함수는
그 구간도 새로운 Page, K vector, V vector를 만들고 복사한다.

source와 destination의 page 경계가 맞고, run 안에 완전한 64행이 있으며, source page가
실제로 그 행을 제공한다면 `shared_ptr<const Page>`를 결과의 pages에 넣을 수 있다.
옮겨진 page와 경계 page는 새로 만든다. source를 덮어쓰는 방식과는 다르다.

위 입력의 global 선두 4K는 320 pages다. 호스트 검사에서 page 객체 320개와 payload buffer
640개, 합계 **960회 할당 요청**이 줄었으며 새 payload는 31.25→26.25MiB였다.
실제 geometry에서는 **80MiB의 새로운 K+V payload**를 만들지 않아도 되는 계산이다.

이 공유는 source의 수명을 연장할 수 있다. 현재 sparse source capture가 `owned=true`로
작은 page를 보관하도록 고친 부분은 유지해야 한다. 큰 RPC slab view를 다시 붙잡는 구현으로
되돌리면 page 공유로 줄인 양보다 더 큰 메모리를 묶을 수 있다.

## 4. gather→restore의 두 번째 전체 복사와 임시본 수명

위치: `kv_cache_manager.cpp:989–1037`, `models/causal_lm.cpp:1971–1987,2250–2264`.

```text
saved source pages
    → 새 gathered snapshot
    → live host Tensor slab
    → 이후 HTP prefill/decode의 KV 처리
```

`restorePrefix()`는 gathered snapshot의 K/V를 live host slab에 다시 memcpy한다.
600MiB destination 예에서는 600MiB payload의 추가 복사다. 현재 구조는 page 형태의 immutable
결과와 연속 host Tensor를 분리하며, 검증 전에 live state를 변경하지 않도록 한다.
따라서 이 memcpy 한 줄만 삭제할 수 있는 것은 아니다.

장기적으로는 **최종 live destination이 될 별도 staging slab에 직접 gather한 뒤 commit**하는
방식을 검토할 만하다. 정렬, RPC 등록, Tensor binding, 실패 시 복구와 이전 source의 불변성을
함께 설계해야 한다. 현재 active cache에 곧바로 써 버리는 수정은 피한다.

더 작은 수정은 **복원 완료 뒤 gathered snapshot 해제**다. sparse 결과는 approximate라
`captureDevicePrefill()`에서 새 exact source capture가 차단되는데도,
`device_prefill_history_`가 이 결과를 계속 소유한다. bind와 calibration 복원 이후 마지막 사용을
확정하고 해제할 수 있다. 같은 600MiB가 generation 중 남아 있을 수 있으며, 이를 해제해도
gather 순간의 peak까지 없어지지는 않는다.

기존 `benchmark_htp_sparse_benefit.cpp:54–58`은 `freezeProducer()` 이후
`beforePrefillBlock()`에서 이 history를 지운다. 일반 경로의 보관 시간과 다르므로 메모리 결과를
해석할 때 구분해야 한다.

## 5. sliding에서 버릴 행을 반복문으로 하나씩 버린다

위치: `kv_cache_manager.cpp:918–928`.

전체 run이 불필요하면 미리 건너뛰지만, 일부만 필요하면 `t=0`부터 시작하고
`pos < dst.base`일 때마다 continue한다. 필요한 시작점은 이미 알고 있다.

```cpp
const size_t begin =
  dst.base > run.destination ? dst.base - run.destination : 0;
for (size_t t = begin; t < run.count; ++t) {
  // ...
}
```

24K→20K 예에서 sliding layer마다 16,384행을 방문하지만 필요한 것은 마지막 1,024행이다.
25개 layer에서 버리는 반복이 384,000회다. global을 포함한 총 루프 실행은
512,000→128,000회로 줄었고 KV 결과는 같았다. 이것은 **해당 행 루프의 75% 감소**이며
전체 gather나 TTFT가 75% 빨라진다는 뜻은 아니다.

## 6. 수천 번의 작은 할당과 pages 증설

위치: `kv_cache_manager.cpp:914–916,944–950`.

위 geometry에서는 global 1,600개와 sliding 400개, 합계 2,000 Page가 만들어진다.
통상 `make_shared<Page>` 한 번, K vector 한 번, V vector 한 번으로 **6,000회**다.
여기에 pages 메타데이터의 증설, layer scratch와 회전 계수 등이 더해져 호스트 계측은
총 6,237회였다. allocator/표준 라이브러리가 달라지면 정확한 요청 수는 달라질 수 있다.

`dst.pages.reserve((length - dst.base + 63) / 64)`만으로 이번 환경에서 145회가 줄었다.
즉시 할 만한 수정이지만, 절감한 메타데이터는 약 61KiB여서 수백 MiB payload 복사와
동급 효과로 소개하면 안 된다.

arena나 reusable page pool을 쓰면 payload의 수천 번 할당을 더 줄일 수 있다. cached source는
오래 살아 있으므로 단순한 전역 scratch 재사용은 안 된다. 결과 snapshot의 소유권 단위로
묶거나 실제로 참조가 사라진 page만 재사용해야 한다.

## 7. 부분 RoPE인데 모든 차원을 변환하고 계산한다

위치: `kv_cache_manager.cpp:958–983`, `nntrainer/kv_cache/sparse_prefix.h:97–107`,
`nntrainer/layers/llm/fp16_attention.cpp:1256–1270`.

Gemma4 global layer는 D512의 25%만 rotary이고, 나머지 주파수는 0이다. 현재는 전체
512차원을 FP16→FP32로 풀고 전체 pair에 `c=1,s=0` 연산까지 적용한 뒤 다시 저장한다.
global의 moved K에 한해서 pair 계산/변환의 75%는 수학적으로 identity 부분이다.
sliding layer와 전체 모델 연산에 그대로 75%를 적용할 수 없다.

활성 rotary index만 변환·회전하고 나머지는 원본 비트를 복사할 여지가 있다. 다만 identity
산술도 signed zero 등의 비트를 바꿀 수 있으므로 “기존 구현과 무조건 bitwise 동일”이라고
약속하면 안 된다. 이 후보는 위 네 가지 누적 변경에 포함하지 않았다.

HTP의 sin/cos는 이미 layer/run당 한 번 계산된다. CPU core의 토큰마다 sin/cos를 계산하는
별도 문제를 이 경로에 다시 적용하면 안 된다. 각 restore 때 layer 순회와 주파수 재생성을
반복하는 것은 작은 후속 후보이며, 모델 설정이 고정되는 시점에 캐시할 수 있다.

## 8. KV Manager 메타데이터: 확실하지만 크기는 작다

### publish에서 한 번 더 깊은 복사

위치: `Applications/CausalLM/device_kv_manager.h:319–326`.

`Prefix p{tokens, ...}`가 소유 token 사본을 만든 뒤 `prefixes_.emplace(p.handle, p)`가
lvalue인 p를 다시 복사한다. tokens와 vocabulary 모두 새 vector가 된다. immutable source를
위한 첫 token 소유본은 필요하지만 map으로 넘기는 두 번째 복사는 move로 줄일 수 있다.
실패 정리를 위해 handle은 move 전에 별도로 보관해야 한다.

주의할 점은 vocabulary다. unique 후 size가 줄어도 capacity는 원본 토큰 수만큼 남는다.
현재 추가 copy는 우연히 이 capacity를 size 수준으로 줄인다. p 전체를 단순 move하면 최종
vocabulary가 커질 수 있다. vocabulary의 용량 축소와 token move를 함께 설계한다.
24K token vector 한 개는 96KiB이므로 600MiB KV 문제와 우선순위를 구분한다.

### prefix 개수 한도에 도달한 정상 게시에서 radix 두 번 재구축

위치: `device_kv_manager.h:306–326,421–442`.

victim 제거 뒤 rebuild하고, 새 prefix 삽입 뒤 또 rebuild한다. 실패가 없는 요청에서도
기존 모든 prefix의 radix를 두 번 만든다. 성공 경로에서 한 번만 publication하고, capture나
삽입 실패 때 유효 owner를 복구하는 쪽으로 재구성할 수 있다. 첫 rebuild를 무조건 지우면
예외 후 owner 복구 성질이 달라지므로 순서만 삭제해서는 안 된다.

rebuild 자체도 매 boundary마다 root에서 같은 prefix를 다시 비교한다. boundary 간격이 P,
source 길이가 N인 단일 chain은 비교량이 대략 O(N²/P)다. 이것은 **DeviceKVManager의
게시 시 radix 재구축**이고, CPU core의 lookup 경로 벡터 복사 문제와는 다르다.

### 토큰 왕복 복사

`models/causal_lm.cpp:2212–2227`에서 전체 입력 vector 생성, prefix token adoption,
device_run_tokens 생성, suffix insert와 재할당이 이어진다. ID 타입이 같으면 참조/view를
받고, 최종 크기를 reserve해 작은 비용을 줄일 수 있다. 현재 채택 token과 최종 setTokens에
있는 move는 이미 적절하다. 수십~수백 KiB 수준이므로 먼저 KV payload를 다루는 것이 맞다.

## 9. 준비 비용을 무시하는 선택과 정상 Flash fallback

`device_kv_manager.h:248–252`는 재사용 길이만으로 stable_sort한다. 같은 길이에 1 run과
192 runs가 있어도 더 싼 후보가 우선하지 않는다. 기존 호스트 검사에서 실제로 더 잘게
쪼개진 후보가 선택됐다. 작은 최소 gain 64행도 실제 생략할 block 수, gather bytes,
Flash read bytes, rotation rows와 함께 평가할 필요가 있다.

`models/causal_lm.cpp:1930–1951`은 ending checkpoint 하나로 충분하면 그것만 읽는다.
이 최적화는 이미 있다. 부족할 때는 필요한 source 구간을 고르기 전에 앞의 checkpoint들을
읽는 fallback이 있다. 삭제 위치 때문에 정상 snapshot 여러 개가 필요한 경우에도 중복 global
payload의 Flash/baked 읽기가 생길 수 있다. 구간 metadata로 필요한 checkpoint를 정한 뒤
읽는 순서로 바꾸는 것이 후속 후보다. resident shared page와 Flash 재읽기의 메모리 비용을
동일하게 계산하면 안 된다.

## 구현 순서와 기기에서 확인할 지표

1. K 사전 copy 제거, sliding 루프 시작점 조정, pages reserve. 기존 연산을 유지하는 작은 변경.
2. exact 전체 page 공유와 복원 완료 후 approximate destination 해제. unique bytes와 수명 검증.
3. arena/초기화 없는 payload 및 최종 host staging slab로 직접 gather. 구조 변경.
4. active rotary index 처리와 비용 기반 후보 선택. bitwise/품질 및 실기기 성능 검증.
5. publish 복사·radix rebuild·고정 geometry 캐싱. source 준비 비용과 작은 메타데이터 비용 정리.

기기에서는 같은 source/request에 대해 external TTFT, gather/restore/bind 각 시간,
unique source/destination/live-host/device bytes, high-water memory, allocation 수,
Flash read bytes를 함께 기록해야 한다. 내부 prefill 시간만으로는 이 수정의 효과를 볼 수 없다.
순수 복사/수명 변경은 동일 KV 또는 동일 출력 ID를 대조하고, 부분 RoPE 산술을 생략하는
변경은 signed zero를 포함한 수치 대조와 기존 품질 검사를 추가한다.

## 결과와 재실행 코드

- [계측 결과 JSON](../tools/kv_cache/sparse_prefix_normal_path_audit/result.json)
- [ASan/UBSan 결과 JSON](../tools/kv_cache/sparse_prefix_normal_path_audit/result-asan.json)
- [독립 검사 안내](../tools/kv_cache/sparse_prefix_normal_path_audit/README.ko.md)
- [빌드/본문 생성 스크립트](../tools/kv_cache/sparse_prefix_normal_path_audit/run.py)
- [검사 fixture와 바이트 대조](../tools/kv_cache/sparse_prefix_normal_path_audit/audit.cpp)

실행 결과는 계측 환경의 성공 경로를 검증한 것이다. HTP 실기기 성능, 전체 모델 품질,
전체 플랫폼 CI 통과를 주장하지 않는다. production 구현은 수정하지 않았다.
