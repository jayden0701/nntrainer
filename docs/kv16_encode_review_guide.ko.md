# KV16 encode 최적화 리뷰 가이드

이 문서는 `pr/4481` 브랜치의 KV16 prefix codec 최적화 스택을 검토하는
에이전트/리뷰어를 위한 안내서다. 무엇이 바뀌었는지, 어떤 불변식이 지켜져야
하는지, 바뀐 내용을 독립적으로 재검증하는 방법, 과거에 발견된 버그 유형을
담는다.

## 1. 범위와 커밋 지도

| 커밋 | 내용 |
| --- | --- |
| `5ededcb00` | KV16 lossless 호스트 prefix codec (포맷 v1) — 최적화 전 기준선 |
| `dc50ebb9d` | encode를 prefill 임계 경로에서 비동기 PrefixCodecPool(워커 4, nice 19)로 이동. cold capture TTFT +22~27% → +2.0% |
| `4a4fbb30b` | 후보 선제 추정(승자 블록만 materialize) + `bits()` → `__builtin_clzll` + gamma 분해 호이스팅. 65.7ms → 42.4ms (1.55x) |
| `08ce259c0` | predict 고속 경로 + 포인터 패킹 + slice-by-8 CRC. 42.4ms → ~25ms (누적 ~2.6x) |

핵심 파일: `Applications/CausalLM/kv16_lossless.cpp/.h`,
`Applications/CausalLM/kv_cache_manager.cpp` (capturePrefix/materializePrefix),
검증 레코드 `docs/validation/kv16-npu-device-20261111.{json,md}`.

## 2. 리뷰 전 필독: 지켜야 할 불변식

1. **바이트 동일성** — 최적화 전후 `encode()` 출력이 완전히 같아야 한다.
   컨테이너 CRC 값까지 포함. 포맷 v1과 디코더는 무변경.
2. **이중 반올림 의미론** — `predict(value, gamma)`는 FP32-RNE → FP16-RNE
   이중 반올림을 정수로 에뮬레이트한다. 임의의 FP16 비트 패턴 × 임의의
   FP32 비트 패턴에서 generic 경로와 완전히 같은 결과를 내야 한다
   (subnormal/inf/nan/round-to-even 포함).
3. **타이브레이크 순서** — 블록 크기 비교는 "fixed가 raw를 먼저 이기고,
   sparse가 현재 승자를 이겨야" 하는 순서를 유지한다. 전체 materialize 시와
   같은 승자가 나와야 한다.
4. **CRC 등가성** — slice-by-8 CRC32는 바이트 단위 테이블 방식과 같은
   0xedb88320 reflected 값을 내야 하고, [48,52) CRC 필드 자체는 스킵한다.

## 3. 커밋별 리뷰 포인트 (위험도 순)

### predict 고속 경로 (`predict_with_gamma`, 08ce259c0)

- fast path 진입 조건: `!nan && !inf && v.mantissa >= 1<<10 &&
  g.mantissa >= 1<<23` — 즉 양쪽 다 finite normal. mantissa 임계로 normal을
  판별하는 이유는 `parts()`가 subnormal/zero의 exponent를 -24/-149로
  정규화하기 때문 (exp field 0과 1이 같은 exponent로 정규화됨).
- 지수 유도: FP32 단계의 actual exponent = `bits(product)-1 + v.exponent +
  g.exponent`. **여기서 `parts()`가 반환하는 exponent에는 fraction_bits가
  이미 빠려 있다** (`exp_field - bias - fraction_bits`), 즉 value = mantissa ×
  2^exponent 관계. FP16 단계의 rounding shift는 r1이 항상 [2^23, 2^24)이므로
  상수 13.
- 폴백 조건 3개: FP32 overflow (`exponent > 127` → generic은 inf를 내고 fast
  path도 inf를 돌려야 함), FP16 subnormal (`exponent < -14` → generic으로
  재계산), FP16 overflow (두 단계의 carry 반영 후 `> 15` → inf).
- 두 단계 각각 round-to-even tie 처리(`rem == half && (q & 1)`)와 carry
  (`rounded == 1<<(bits)` → 절반으로 내리고 지수 +1)를 generic과 같은 순서로
  적용해야 한다.
- **가장 흔한 실수**: stage-2 입력 지수를 product의 지수(e1)로 착각하는 것.
  실제로는 FP32 actual(`bits(product)-1+e1`)이고 carry마다 +1된다.

### pass-1 residuals과 pass-2 패킹 (08ce259c0)

- pass-1: residual 버퍼를 **크기를 지정해 생성**한다
  (`std::vector<uint16_t> residuals(count * plain.size())`).
  `reserve()` 후 raw 포인터로 쓰고 `resize()`로 확장하면 value-init이 쓴
  데이터를 0으로 덮어쓴다 — 실제로 발생해 A/B가 잡은 버그.
- pass-2: 마찬가지로 **resize(확장, 0 채움)를 먼저 하고 포인터로 덮어쓴다**.
  순서를 반대로 하면 같은 버그.
- `width == 16`일 때 residual 배열을 memcpy하는 것은 little-endian 가정이다.
  대상 플랫폼(x86-64, arm64)은 전부 LE라 안전하지만, 빅엔디안 포팅 시
  수정 필요.
- sparse 경로는 그대로 put()을 쓴다 (빈도가 낮아 병목 아님).

### slice-by-8 CRC32 (`checksum`, 08ce259c0)

- 테이블 점화식: `t[j][i] = (t[j-1][i] >> 8) ^ t[0][t[j-1][i] & 255]`.
- 8바이트 루프: `lo = c ^ (p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24)` —
  **괄호 필수**. `c ^ p0 | p1<<8 | ...`로 쓰면 `^`가 `|`보다 우선순위가
  높아 상위 바이트가 XOR이 아닌 OR로 계산된다 (실제로 발생해 크래시 전에
  A/B가 잡은 버그).
- [48, 52) 구간(CRC 필드) 스킵은 두 구간 순회로 분리 — 바이트당 분기 제거가
  핵심 이득.

### 후보 선제 추정 (4a4fbb30b)

- pass 1에서 residual/width/nonzero만 계산하고, 닫힌 공식
  (raw = 4n, fixed = base + ceil(n·width/8), sparse = base + bitmap + 2·nonzero)
  로 크기를 비교한 뒤 승자만 materialize한다.
- 1차 구현이 fixed/sparse 승자의 **mode 바이트를 0(Raw)으로 남긴** 버그가
  있었다 — 크기 비교와 직렬화가 분리되면서 생기는 전형적 실수. A/B가 기기
  실행 전에 잡았다.

### 비동기화 (dc50ebb9d)

- capturePrefix는 shared-slab view를 owned RAW로 freeze + codec hint 기록 +
  풀 큐잉만 하고, encode/VERIFY/RAW release는 워커(nice 19)가 수행.
- **모든 page backing reader**(capturePrefix 재사용 확인, materializePrefix,
  checkpoint trace, Flash/writer 제출, footprint 통계)는 접근 전에 풀을
  drain해야 한다 — 이 drain 누락이 이 디자인의 최대 위험. encode 실패/
  no-gain/VERIFY 불일치 시 owned RAW를 유지하므로 codec 문제가 추론을 바꾸지
  못한다.

## 4. 독립 재검증 방법

### 호스트 바이트 동일 A/B (가장 강한 검증)

`tools/kv16/kv16_encode_ab_harness.cpp` 헤더 주석의 명령 그대로:

```sh
OLD_REV=5ededcb00   # 최적화 이전 기준선 (개별 커밋 검증 시 그 이전 커밋)
git show $OLD_REV:Applications/CausalLM/kv16_lossless.cpp |
  sed -e 's/^namespace causallm {/namespace causalleg {/' \
      -e 's/^} \/\/ namespace causallm$/} \/\/ namespace causalleg/' \
      -e '13a using causallm::kv16::Geometry;\nusing causallm::kv16::Decoded;\nusing causallm::kv16::max_decoded_bytes;' \
  > /tmp/kv16_old_renamed.cpp
g++ -O2 -std=c++17 -I. -Iapi/ccapi/include -IApplications/CausalLM \
    -IApplications/CausalLM/app_config \
    tools/kv16/kv16_encode_ab_harness.cpp /tmp/kv16_old_renamed.cpp -o /tmp/kv16_ab
/tmp/kv16_ab            # 2,048 케이스 A/B + decode 왕복 → "A/B BYTE-EXACT PASS"
/tmp/kv16_ab predict    # 65,536 FP16 × 4,096 gamma 전수 비교 → "PASS"
/tmp/kv16_ab bench      # encode throughput old vs new
```

예상 결과: `cases=2048 failures=0 modes: m0=27983 m1=56080 m2=5777`,
`checked=268435456 failures=0 -> PASS`. 기준선 rev을 바꿔가면 각 커밋 단계를
개별 검증할 수 있다.

### 장치 검증 (SM-S948U 또는 동급 v81 기기)

- **푸시 전 SHA 대조 필수**: 장치 `/data/local/tmp/Quick.AI/`의 바이너리
  SHA256이 로컬 빌드와 다르면 stale push로 세그폴트처럼 보인다.
- 체커 4종: `htp_kv16_codec_check`(self-check), `htp_kv16_prefix_check`,
  `htp_kv16_flash_check`, `htp_calibration_check` — 전부 exit 0.
- full-scope evidence: 모델 팩의 `nntr_config.json`을
  `nntr_config.kv16_checkpoint.json`으로 교체하고
  `htp_kv16_evidence_check`를 baseline(0)/codec(1+VERIFY=1) 각각 실행 후
  `tools/kv16/compare_device_evidence.py base.json codec.json --scope full
  --output out.json` → PASS (exit 0). **끝나면 반드시 pf2k config로 복원.**
- 검증 실행은 모델 디렉토리에 cd한 상태로 하되 MODEL_DIR은 절대경로로
  전달한다 (wrapper가 CWD를 바꿈). adb 세션 종료와 함께 체커가 죽는 문제는
  `setsid` 분리 실행으로 회피.

## 5. 참고 자료

- 검증 레코드: `docs/validation/kv16-npu-device-20261111.{json,md}`
  (`encode_optimization` / `follow_up_20261113` 섹션에 측정값 전체)
- 산출물: `docs/validation/kv16-npu-device-20261111/encode-opt/`
  (evidence, trace, compare 결과)
- 성능 요약: encode 65.7ms → 42.4ms → ~25.3ms (2,154토큰 플레인, 누적
  ~2.6x). TTFT는 encode가 임계 경로 밖이라 +2.5%/+3.8% 수준 유지(thermal
  편차가 런 간 편차 지배). warm restore/TPOT 동등.

## 6. 열린 항목 (리뷰 시 함께 볼 것)

- residual scan의 NEON 벡터화 (predict가 남은 시간의 ~63%) — 미착수
- handoff 시나리오 1(suffix 분리), 3(Prefix Flash), 4(손상 fallback)
  명시적 검증 — 미수행
- 장기 상주(Flash 저장) 시의 실측 RSS — pinned RPC slab view 때문에 저장
  절감(6.42%, 세트당 104.2MiB)이 시스템 메모리에 1:1 매핑되지 않아 별도
  실험 필요
- n=3 인터리브 perf 페어 (단일 페어는 thermal 편차 지배)

