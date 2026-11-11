# 정상 sparse gather 비용 확인 코드

대상 commit: `73f8feec47e2bab4d7ac39a1f7d46e3f3b501b20`.
실패나 손상을 주입하지 않고 성공한 gather의 작업량을 비교한다.

## 실행

이 폴더를 복사한 뒤 Linux/WSL의 C++17 컴파일러와 Python 3으로 실행한다.
checkout에는 nntrainer의 정상 빌드 의존 헤더가 필요하다. nlohmann의 실제 single-header
`json.hpp`가 별도 폴더에 있으면 `--json-include`를 지정한다.

```sh
python3 run.py /path/to/nntrainer --json-include /path/to/real-json-header
python3 run.py /path/to/nntrainer --json-include /path/to/real-json-header \
  --sanitize --output result-asan.json
```

Windows에서 만든 git worktree를 WSL에서 읽을 때 `.git`의 경로를 Linux git이 이해하지
못할 수 있다. 그 경우 `--commit 73f8feec47e2bab4d7ac39a1f7d46e3f3b501b20`으로 기준
commit을 기록할 수 있다. 파일/함수 SHA256도 결과에 별도로 기록한다.

## 무엇을 빌드하는가

- `audit.cpp`: 유효한 source pages 생성, 할당 계측, 결과 바이트 비교와 source 불변성 확인.
- `audit_support.h`: 별도 memcpy/resize/행 반복 계측.
- `generated.cpp`: run.py가 **현재 checkout의 실제 gather 본문**을 읽어 생성.
- production `kv_cache_manager.cpp`, `fp16.cpp`: 원래 함수를 그대로 빌드해 대조군으로 사용.

수정 대상 문장이 정확히 한 번 나타나지 않으면 본문 변환을 중단한다. 다른 commit에서 실행할
때 조용히 엉뚱한 함수를 검사하지 않도록 하기 위한 조건이다. 본래 checkout은 수정하지 않는다.

네 변경은 누적된다: 회전할 K 사전 copy 제거 → pages reserve → sliding 시작점 조정 →
정렬된 exact 선두 전체 page 공유. 결과가 원래 함수와 바이트 단위로 다르거나 source가
바뀌면 비정상 종료한다. 원래 계측본도 원래 함수와 비교한다.

작은 입력은 FP16/FP32, split/interleaved/no-RoPE, 여섯 run 배치를 비교한다. 부분 page,
정렬되지 않은 삭제 경계와 signed zero가 포함된다. 큰 입력은 30 layers의 24K→20K 삭제
사례이며 host payload를 D64/KV head 1개로 축소한다.

## 결과 해석

`result.json`은 일반 빌드, `result-asan.json`은 sanitizer 빌드의 결과다.
일반 빌드는 production cpp 전체를 수정 없이 컴파일한다. ASan/UBSan 빌드는 무관한 allocator의
vtable을 참조하지 않도록 원래 gather 메서드를 본문 변경 없이 `production_gather.cpp`에
추출해 컴파일한다. engine 동작을 stub으로 바꾸지는 않는다. 이 독립 함수의 메모리·범위/산술
오류 검사를 전체 HTP engine 검증으로 해석하면 안 된다.

할당 수는 해당 환경의 `operator new` 요청 수다. payload_zero_bytes는 vector 값 초기화
크기, key_initial_copy_bytes/value_copy_bytes는 지정한 memcpy의 payload 크기다.
DRAM 트래픽, RSS, Android allocator, QNN/HTP 실행시간을 측정하지 않는다.

이 코드는 최적화 검토용이다. 생성된 함수에는 계측이 들어 있으므로 그대로 production에
복사하지 않는다. arena/초기화 제거, 최종 slab 직접 gather, snapshot 조기 해제 및 부분
RoPE 활성 차원 최적화는 이 검사에서 구현하지 않았다.

전체 검토는 [상세 보고서](../../../docs/sparse_prefix_reuse_normal_path_review.ko.md)에 있다.
