# KV16 NPU 장치 검증 기록 (2026-11-11)

인계서 `kv16-handoff-bundle/docs/kv16_npu_handoff.ko.md`의 "장치 연결 전에
준비할 것"과 "기기에서 순서대로 확인할 시나리오"의 1~6번 중 실측 가능한
범위를 실행한 기록이다. 기계 가독 기록은
[kv16-npu-device-20261111.json](kv16-npu-device-20261111.json)이다.

## 환경

- 장치: SM-S948U (Galaxy S26 Ultra), SM8850, Android 16, DSP v81
- NDK r28c, Hexagon SDK 6.4.0.2 (HEXAGON_Tools 19.0.04), HexKL 6.4.0.2
- 팩: `/sdcard/Download/aistudio-mobile/models/gemma4-26b-a4b-htp-w4a32`
- ARM/DSP는 `Applications/CausalLM/build_android.sh --htp --install`로
  새로 빌드·배포한 matching 세트(해시는 JSON 기록 참조)를 사용했다.

## 네이티브 체커: 7종 모두 PASS

`htp_calibration_check`, `htp_prefill_check`, `htp_prefix_flash_check`,
`htp_decode_kv_convert_check`, `htp_kv16_codec_check`,
`htp_kv16_prefix_check`, `htp_kv16_flash_check` — 모두 exit 0.

실행은 `LD_LIBRARY_PATH=/data/local/tmp/Quick.AI`만 설정해야 한다.
`/vendor/lib64`를 추가하면 시스템 `libandroidfw.so` 심볼 해석이 깨진다.

## 세션 검증 (htp_sessions_check, radix prefix fixture)

이전 장치 검증(`decode-prefix-batch-20261105`)의 nntr_config를 팩 경로로
재지정해 사용했다: q8, skip_prefill=true, init_seq_len=4096,
`htp.kv_session/kv_paged/kv_prefix_cache=true`, `kv_seed_q=false`.

- baseline (`NNTR_HTP_PREFIX_KV16=0`)과 codec (`=1`, `VERIFY=1`) 모두
  4 prefix / 4096 reused tokens / 5 sessions로 동일한 radix 동작.
- `tools/kv16/compare_checkpoint_traces.py` **checkpoint scope PASS**.
- seed trace(JSONL)가 baseline/codec 간 **바이트 단위로 동일**.
- switched 세션의 token IDs, position, kv_bytes 모두 동일.

## codec 압축 결과

| 항목 | 값 |
| --- | ---: |
| checkpoint 이벤트 | 6 |
| compressed pages | 1,700 |
| raw bytes | 1,701,683,200 |
| stored bytes | 1,592,384,318 |
| 절감 | **109,298,882 bytes (6.42%)** |

작은 page 일부는 metadata가 이득을 상쇄해 RAW로 유지되었다(문서대로 정상).
`compressed_pages=1700 > 0`이므로 codec 경로가 실제로 동작했다.

## 성능 (TRACE/VERIFY off, 단일 실행)

| 구간 | prefill | TTFT baseline | TTFT codec | TPOT b→c |
| --- | ---: | ---: | ---: | --- |
| cold capture (sess 0, turn 0) | 2,079 | 3,941 ms | 4,820 ms | 1.86→1.99 |
| cold capture (sess 1, turn 0) | 4,607 | 9,122 ms | 11,144 ms | 1.82→1.78 |
| warm restore (turn 1/2) | 11 | 736–793 ms | 738–834 ms | 1.57–1.66→1.65–1.86 |

cold capture에 encode 비용이 더해져 TTFT가 약 +22% 늘었다. warm restore은
단일 실행 분산 범위 내이거나 소폭 느리다. **codec은 TTFT/TPS 개선 장치가
아니라 host FP16 prefix 보관 메모리 절감 장치다.** 반복·thermal 통제 측정은
별도 과제로 남는다.

## 반복·thermal 통제 성능 (n=3 교차, 2026-11-11 추가)

3라운드 × (OFF,ON) 교차(순서 로테이션), 8초 무부한 게이트, 실행 전후
thermal status 기록, TRACE/VERIFY off, 2초 주기 RSS 샘플링. 장치는 USB
연결 상태에서 연속 실행이라 thermal status가 0→1→2→3으로 상승했고 교차
순서로 상쇄했다.

| 항목 | baseline (OFF) | codec (ON) |
| --- | --- | --- |
| TTFT cold capture 중앙값 (2079 tok / 4607 tok) | 3,889 / 9,286 ms | 4,954 / 11,541 ms (+24~27%) |
| TTFT warm restore 중앙값 | 779–838 ms | 753–843 ms (동등) |
| TPOT 중앙값 | 1.79–1.98 ms | 1.79–1.98 ms (동등) |
| load+teardown 중앙값 | 78.6 s | 82.7 s (+4.1 s) |
| RSS peak | 966 MiB | 993 MiB (+27 MiB) |

- RSS peak이 codec에서 27 MiB 높다: capture 시 encode scratch와 compressed
  payload가 live cache와 공존하기 때문. 저장 바이트 절감(-104 MiB/스냅샷)이
  실행 중 peak RSS 감소로 이어지지 않는 것은 인계서 경고대로다. 장기
  상주(prefix를 여러 턴에 걸쳐 유지) 실험은 별도로 필요하다.
- 결론: warm restore 성능은 동등, cold capture에만 비용. 이득은 host FP16
  prefix 보관 바이트(스냅샷 세트당 109.3 MB)다.

## full-scope 증거 비교 (PASS, 2026-11-11 추가)

`test/htp/host/kv16_evidence_check.cpp`(신규, 엔진 수정 없음)를 baseline/codec
두 모드로 실기 실행해 `tools/kv16/compare_device_evidence.py --scope full`로
비교했다 — **PASS (exit 0)**.

- 플로우: warm-up + 시드 캡처 2회 + query 연속. `enableDeviceKVManager(8,8)`로
  device prefix capture를 켜고, 캡처가 일어난 런마다 `saveDeviceKVCache`로
  상주 DSP INT8 체크포인트를 덤프했다.
- 관측값: token IDs, position, call-plan, calibration(trace u32 bit words 해시),
  전체 logits(LogitsProcessor, 첫 decode step), k/v/seed_kv/seed_scales/
  decode_state(설계상 동일 덤프 바이트).
- 덤프 프레이밍의 두 필드는 실행마다 달라진다(snapshot cookie = 헤더 [8,16)과
  그를 덮는 마지막 FNV trailer 8바이트). 체커는 이 두 범위를 제외하고 해시했다.
  원본 덤프 파일의 SHA는 `evidence-full/kv16-dump-sha256.txt`에 남긴다.
- 결과: baseline/codec 양쪽 모든 관측값 동일, codec 모드 compressed pages > 0.
- 산출물: `docs/validation/kv16-npu-device-20261111/evidence-full/`
  (evidence.json, trace.jsonl.gz, run.log ×2, compare-evidence-full.json).
  덤프(.bin)와 logits(.f32)는 용량 문제로 해시만 기록했다.

## 아직 입증하지 못한 것

- RSS/RPC slab/system peak의 실제 절감, Flash v2 저장 경로의 장치 실측.
- handoff 시나리오 1(suffix 분리), 3(Prefix Flash), 4(손상 fallback)의
  명시적 검증.
- 장치 팩의 원래 `nntr_config.json`은 백업 전에 덮어써져 있었다(12387 bytes).
  `nntr_config.gemma4_pf2k.json`(동일 12387 bytes)으로 복원했고, 검증 config는
  `nntr_config.kv16_checkpoint.json`으로 보존했다. 원본이 pf2k와 같은
  내용이라는 것은 크기 일치 추정이며 바이트 증명은 아니다.

## 원문 산출물

`docs/validation/kv16-npu-device-20261111/` 아래에 baseline/codec 세션 JSON,
seed trace, 전체 로그, compare-checkpoint.json, compare-evidence.json,
성능 세션 JSON과 gzip 압축 trace를 보존했다.
