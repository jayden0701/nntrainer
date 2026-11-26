# CPU KV의 총예산과 동적 롤링 기기 실험

이 실험은 같은 KV 메모리 상한에서 상주 공간을 선읽기에 빌려주는 편이 더 빠른지 판별한다. 대상은 Android의 CPU FSU다. 기기, 저장소, 모델, 문맥 길이가 바뀔 때마다 별도로 측정한다.

## 구현과 비교 대상

| 설정 | 동작 |
|---|---|
| pack의 `kv_flash_fsu.lookahead=N` | 파일 읽기가 필요한 서로 다른 owner를 최대 N개 선읽는다. 한 reader가 제출 순서대로 처리하고, attention은 자신이 쓸 owner의 완료만 기다린다. 서로 다른 owner를 N개 동시에 읽는 값으로 해석하지 않는다. |
| `NNTR_KV_TOTAL_BYTES=C` | 고유 sliding K/V 텐서, 페이지 단위 풀의 현재·상주·선읽기 K/V, 비동기 쓰기 staging을 합쳐 C 바이트 이내로 제한한다. 현재 스텝이 쓸 끝 행까지 attention 진입 전에 예약한다. 가중치, 활성화, 프로세스 RSS, 파일 페이지 캐시 및 별도 snapshot 임시 텐서는 이 지표 밖이다. |
| `NNTR_KV_RESIDENT_BYTES=R` | 토큰 사이에 유지할 owner의 추가 상한이다. 총예산을 설정하고 R을 생략하면 C까지 유지할 수 있다. R=0이면 유지할 owner가 없어 롤링이 빌릴 공간도 없다. 두 환경변수 모두 십진 바이트 정수를 받는다. |
| `NNTR_KV_ROLLING=0` | 선읽기는 빈 공간을 쓴다. 현재 attention의 필수 데이터가 자라면 상주 공간을 회수할 수 있다. 기본 비교군이다. |
| `NNTR_KV_ROLLING=1` | 선읽기 공간이 부족하면 대상보다 나중에 다시 쓸 상주 owner부터 회수한다. 현재 연산 중인 owner와 더 일찍 쓸 alias는 보호한다. 회수 가능한 공간이 충분할 때만 dirty 범위를 파일에 기록하고 슬롯을 빌린다. 현재 레이어가 끝나 공간이 생기는 시점에도 다시 제출한다. C를 지정해야 한다. |

예를 들어 A(global), B·C(sliding DRAM), D·E(global) 순서라면 A에서 D와 E를 예약할 수 있다. D를 쓸 시점에 E 읽기가 남아 있어도 D 연산을 시작한다. 깊이와 공간이 허용해야 두 owner를 예약한다. 다음 토큰에 보존할 owner는 현재 토큰에서 빌린 공간, 이후의 사용 순서와 R에 따라 달라진다. 토큰 경계에서는 읽기와 쓰기를 drain한다. 현재 정책은 기기별 시간 예측이나 다음 토큰을 가로지르는 선읽기를 수행하지 않는다.

## 기기에서 실행

1. 같은 커밋으로 CPU FP16 external KV를 지원하는 Android 실행 파일과 라이브러리를 배포한다. 동일 가중치·토크나이저·생성 시드로 resident pack, FSU 깊이 1 pack, 더 깊은 FSU pack을 준비한다. FSU pack끼리는 `kv_flash_fsu.lookahead`만 바꾼다. resident pack에는 `kv_flash_fsu`를 넣지 않고 `NNTR_KV_WINDOW_CAP=1`로 sliding 창을 맞춘다.
2. global 전용 모델과 global/sliding 혼합 모델을 각각 실행한다. 혼합 모델의 로그에서 `mixed storage`가 실제로 발생하고 여러 global owner가 있는지 확인한다. global 하나뿐인 모델에서는 여러 global 읽기의 효과를 검증하기 어렵다. 짧은 요청, 긴 decode, 이어쓰기, reset 후 새 요청을 포함하는 동일한 `--requests` 입력을 쓴다.
3. 먼저 총예산을 생략해 resident 기준과 FSU의 `kv_payload_peak_bytes`를 수집한다. 기기에서 허용하는 메모리와 관측 범위로 C 후보를 정한다. sliding과 현재 owner 한 쌍도 들지 않는 C에서는 오류가 정상이다. 같은 C에서 R을 여러 값으로 바꾼다. 일부 owner만 유지할 수 있는 구간을 포함한다.
4. 같은 C·R에서 **고정 깊이 1 → 고정 깊이 N → 롤링 깊이 N**을 비교한다. 가운데 실행으로 깊이의 효과와 상주 공간 차용 효과를 구분한다. 반복마다 실행 순서를 회전하고 전원·열 상태, 빌드 SHA, pack 설정, direct/buffered 모드를 남긴다. 후보를 고른 뒤에는 독립된 반복에서 재확인한다.

다음은 호스트의 POSIX 셸에서 한 설정을 실행하는 예다. `adb`가 선택한 기기에 `/data/local/tmp/nntr-cpu`가 있다고 가정한다. C와 R은 앞 단계에서 고른 정수다. PACK과 REQ는 기기 작업 디렉터리에 대한 상대 경로다. 그룹마다 별도 로그 디렉터리를 쓰고 반복마다 새 파일을 만든다.

```sh
: "${C:?set total bytes}" "${R:?set retention bytes}" "${MODE:?set 0 or 1}"
: "${PACK:?set FSU pack}" "${REQ:?set requests file}" "${LOGDIR:?set host log directory}"
mkdir -p "$LOGDIR"
adb shell "cd /data/local/tmp/nntr-cpu && LD_LIBRARY_PATH=. NNTR_ENGINE=cpu NNTR_KV_TIMING=1 NNTR_KV_TOTAL_BYTES=$C NNTR_KV_RESIDENT_BYTES=$R NNTR_KV_ROLLING=$MODE ./nntr_causallm '$PACK' --requests '$REQ'" > "$LOGDIR/b${R}_r${REP:-1}.log" 2>&1
```

같은 요청으로 resident를 반복 실행해 각 그룹에 `resident_r1.log` 등의 이름으로 넣는다. resident 실행에도 `NNTR_KV_TIMING=1`을 준다. 파일 이름, pack 이름과 디렉터리 경로는 실제 배포에 맞춘다. 각 그룹은 동일한 C·MODE·깊이를 사용한다. ROWS는 비교할 요청의 실제 `cached_after`다.

```sh
python3 tools/kv_cache/kv_budget_predict.py calibrate --logs "$LOGDIR" --rows "$ROWS" --device "$DEVICE" --model "$MODEL" --out "$LOGDIR/profile.json"
python3 tools/kv_cache/kv_budget_predict.py solve --profile "$LOGDIR/profile.json" --rows "$ROWS" --memory kv-peak --cap "$C" --metric p95
python3 tools/kv_cache/kv_budget_predict.py solve --profile "$LOGDIR/profile.json" --rows "$ROWS" --memory kv-peak --metric p95 --target "$TARGET_MS"
```

## 판정

- 먼저 매 요청의 `output_ids`, `cached_before`, `cached_after`를 resident와 비교한다. 모든 capped 실행에서 `owner_lookahead=2`, `capped=1`, 의도한 `rolling`과 `total_budget_bytes`, `kv_payload_peak_bytes <= total_budget_bytes`를 확인한다.
- 고정 메모리 실험에서는 같은 C에서 관측한 steady p50·p95가 더 낮은 설정을 고른다. 고정 지연 실험에서는 목표를 만족하는 최소 관측 KV peak를 고른다. 실제 배포에서 보장할 상한이 기준이면 목표를 통과한 최소 **설정 C**도 함께 기록한다. TTFT·prefill 회귀는 별도로 비교한다.
- `consumer_wait_us`는 소비하려는 owner의 선읽기 완료 대기, `demand_read_us`는 동기 miss 읽기다. `read_bytes`, `backing_write_bytes`와 두 시간은 누적치이므로 요청 전후 차이를 같은 토큰 수로 나눈다. `borrowed_owners`는 선읽기를 위해 회수한 owner 횟수, `budget_blocks`는 공간 부족으로 보류한 선읽기 시도, `discarded_prefetch`는 필수 데이터 성장 때문에 반납한 선읽기 횟수다. 차용이 0이면 해당 실행은 차용의 성능 효과를 측정하지 못한 것이다.
- 읽기 대기가 줄어도 dirty 쓰기나 재읽기가 늘어 p95가 나빠질 수 있다. 이 경우 해당 조건에서는 고정 정책을 유지한다. R=0과 양수를 비교할 때 쓰기 정책까지 분리하려면 별도 실험에서 모든 실행에 `NNTR_KV_WRITE_THROUGH=1`을 적용한다.
- HTP로 옮길 때는 DSP의 마지막 버퍼 사용이 끝난 시각도 슬롯 재사용 조건에 포함한다. CPU의 점유 시간과 예산 결과를 HTP에 적용하려면 HTP에서 별도 스윕이 필요하다.
