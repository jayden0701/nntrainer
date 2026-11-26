# compaction × KV FSU 물리 레이아웃 통합 설계 — base 이동과 물리 행 수식의 화해

> 작성 배경: `kv-cache-fsu` 브랜치(가드 커밋 `850e0b21e` 포함, 문서 작성 시점 HEAD `060a0f24a`)에서 CPU KV flash streaming(`KVFlashStorage`)의 물리 행 관리와 windowed 레이어 `compact()`의 행 이동이 조합될 수 없음을 확인하고, 그 화해 설계를 정리한 문서다. 코드 수정은 없으며 설계 결정을 위한 근거 정리다.
> 참조: [기기 에이전트용 실험·이식 순서](../kv-cache-fsu-htp-handoff.ko.md) · [활성 KV의 Flash 스트리밍 분석](../kv-cache-fsu-analysis.ko.md) · [CPU 구현·기기 테스트 기록](../kv-cache-fsu-cpu-device-log.ko.md) · [계획·이행 요약](../kv-cache-fsu-status.ko.md)

## 1. 문제 정의 — 같은 평면에 두 개의 "행 → 절대 위치" 사상

### 1.1 두 레이아웃 계약

| | windowed(compacting) 레이어 — memmove 방식 | FSU(flash) 소유 레이어 — 물리 행 고정 방식 |
|---|---|---|
| 물리 행 p의 의미 | 절대 위치 `p + base` (base는 compact()가 옮긴다) | 절대 위치 `p` (full layer) 또는 `p + k·cap` (ring layer, `setLayerCaps`) |
| live 행의 위치 | 평면 **앞쪽에 연속** `[0, pos − base)` | 쓰인 자리 그대로. full layer는 `[0, pos)`, ring은 `[0, cap)`을 wrap |
| 행 이동 | `compact()`가 `std::memmove`로 앞당기고 `bases_[l] = from` | 없음. `finishFlashLayer()`가 `from % cap`로 물리 위치를 계산해 저장 |
| 읽기 | 앞쪽 `[0, pos − base)` 연속 뷰 | `load()`가 `[0, saved)` 전체를 풀 버퍼로 읽고 binding 교체 |

windowed 계약의 근거는 `compact()`의 memmove와 base 갱신 [kv_cache_manager.cpp:1705–1748](../Applications/CausalLM/kv_cache_manager.cpp#L1705), `getBase()`의 주석 [kv_cache_manager.h:155–168](../Applications/CausalLM/kv_cache_manager.h#L155), 스냅샷의 `save_off = (from − base)·width` [kv_cache_manager.cpp:2048–2051](../Applications/CausalLM/kv_cache_manager.cpp#L2048)이다. FSU 계약의 근거는 `finishFlashLayer()`의 `start = from % cap` [kv_cache_manager.cpp:804–817](../Applications/CausalLM/kv_cache_manager.cpp#L804)과, 실행 중에만 pool 버퍼를 binding에 연결하고 `evict()` 후 원래 매핑으로 되돌리는 `load()/evict()` [kv_cache_manager.cpp:215–233](../Applications/CausalLM/kv_cache_manager.cpp#L215) · [kv_cache_manager.cpp:293–330](../Applications/CausalLM/kv_cache_manager.cpp#L293)이다.

충돌의 본질은 두 가지다.

1. **사상 불일치**: windowed 계약은 물리 주소의 기준점(base)이 매 compaction마다 움직이고, FSU 계약은 기준점이 0(또는 cap 주기)으로 고정돼 있다. 같은 평면에 두 사상이 공존할 수 없다.
2. **데이터 위치 불일치**: flash 캐시에서 compact()가 만질 `layer_caches_[l].key_cache`의 데이터는, 소유자가 퇴피한 뒤에는 **텅 빈 원래 매핑**(익명 mmap)이다. 실제 live 데이터는 pool 버퍼(유지 중) 또는 파일(퇴피 후)에 있다. memmove는 텅 빈 매핑의 쓰레기를 앞당길 뿐 실데이터는 뒤에 남는다 — 가드 주석이 정확히 이 지점을 말한다 [kv_cache_manager.cpp:1718–1727](../Applications/CausalLM/kv_cache_manager.cpp#L1718).

### 1.2 충돌 지점 세 곳 (코드 준거)

**(a) `finishFlashLayer()`의 `start = from % cap`** — [kv_cache_manager.cpp:809](../Applications/CausalLM/kv_cache_manager.cpp#L809). 절대 위치 `from`을 base 없이 물리 행으로 바꾼다. base가 0이 아닌 windowed 레이어에서는 쓰기 위치가 틀어진다. 구체 예: cap=64, keep=32인 windowed 레이어에서 `compact(70)` 후 base=38, live 행 `[38,70)`은 물리 `[0,32)`에 있다. 다음 단계가 `[70,71)`을 쓸 때 올바른 물리 행은 `70 − 38 = 32`지만 수식은 `70 % 64 = 6`을 내놓는다 — 절대 44의 live 행을 덮어쓰고 절대 70은 기록되지 않는다. 또한 windowed 계약은 평면이 wrap하지 않는다(창이 앞으로 이동하므로)는 전제인데 이 수식은 ring 용량을 넘겼다고 가정하고 두 구간으로 쪼갠다.

**(b) `KVFlashStorage::load()`/`readPlanes()`의 `[0, saved)` 전체 읽기** — [kv_cache_manager.cpp:215–233](../Applications/CausalLM/kv_cache_manager.cpp#L215) · [kv_cache_manager.cpp:571–594](../Applications/CausalLM/kv_cache_manager.cpp#L571) (`transfer(entry, 0, 0, entry.saved, false)`, L576·L583). `saved`는 `store()`가 유지하는 물리 바이트 상한이다(`entry.saved = max(entry.saved, last)`, [kv_cache_manager.cpp:250](../Applications/CausalLM/kv_cache_manager.cpp#L250) · [kv_cache_manager.cpp:289](../Applications/CausalLM/kv_cache_manager.cpp#L289)). 이 읽기가 "물리 0부터 저장된 전부"라는 뜻을 갖려면 **물리 0이 가장 오래된 유효 행**이어야 한다. base가 이동한 뒤에는 물리 0의 절대 위치가 달라지므로, `[0, saved)`는 죽은 행·창 밖 행·live 행을 읽는자가 해석할 수 없는 형태로 섞는다. store도 같은 좌표계로 쓰므로 이동 이후의 파일 내용은 어떤 규약으로도 기술되지 않는다.

**(c) save/load 스냅샷의 `rowsToPersist()`·`liveFrom()` 수식** — [kv_cache_manager.h:538–545](../Applications/CausalLM/kv_cache_manager.h#L538) · [kv_cache_manager.h:664–667](../Applications/CausalLM/kv_cache_manager.h#L664). save는 windowed 레이어의 `save_off = (from − bases_[li]) · kv_widths_[li]`로 텐서 앞쪽에서 창 행을 읽는다([kv_cache_manager.cpp:2046–2059](../Applications/CausalLM/kv_cache_manager.cpp#L2046), `FlashSnapshotLease` [kv_cache_manager.cpp:768–783](../Applications/CausalLM/kv_cache_manager.cpp#L768)). load는 페이로드를 행 0에 적고 `bases_[i] = liveFrom(i, seq_len)`으로 base를 재설정한 뒤 `flash_->store(i, 0, bytes, /*replace=*/true)`로 파일 물리 `[0, …)`에 기록한다([kv_cache_manager.cpp:2205–2257](../Applications/CausalLM/kv_cache_manager.cpp#L2205)). 둘 다 memmove 계약(창이 앞에 연속)을 전제한다. 참고로 비windowed ring 레이어의 스냅샷은 이미 다른 규약(`rowsToPersist`가 평면 전체를 쓰고 헤더의 절대 위치가 사상을 재유도)을 쓰므로, 파일 안에 두 규약이 공존한다 — 화해 작업은 이 공존을 의도된 것으로 만드는 일이기도 하다.

### 1.3 현행 가드 (커밋 `850e0b21e`)

| 가드 | 위치 | 내용 |
|---|---|---|
| allocate 거부 | [kv_cache_manager.cpp:1013–1020](../Applications/CausalLM/kv_cache_manager.cpp#L1013) | flash 저장소는 `!rows.empty()`(rows/keep)를 shared allocator·core shared와 함께 `std::invalid_argument`로 거부. flash 캐시에 windowed 레이어가 아예 존재할 수 없다 |
| compact 거부 | [kv_cache_manager.cpp:1721–1732](../Applications/CausalLM/kv_cache_manager.cpp#L1721) | flash가 켜진 캐시에서 windowed 레이어의 이동을 만나면 `std::logic_error`. allocate 가드 덕에 현재 도달 불가한 belt |
| 회귀 테스트 | [unittest_kv_cache_manager.cpp:1076](../test/unittest/layers/unittest_kv_cache_manager.cpp#L1076) | `flash_storage_and_windowed_rows_are_exclusive` — 양방향 상호배제 고정 |
| 기록 | [cpu-device-log §9.2](../kv-cache-fsu-cpu-device-log.ko.md) | 4–8장의 기기 실행에 조용한 데이터 손실이 없었음을 확인하고, 장기 과제를 핸드오프 4단계 0항으로 이동 |

`validateRange()`의 host-addressable 검사([kv_cache_manager.cpp:1508–1524](../Applications/CausalLM/kv_cache_manager.cpp#L1508))만으로는 이 충돌을 잡지 못한다. FSU의 원래 매핑이 익명 mmap이라 host addressable이기 때문이다 — 그래서 명시적 `logic_error` 가드가 필요했다.

### 1.4 왜 지금 화해해야 하나

- 핸드오프 문서의 1순위 최적화 후보가 "읽는 데이터 줄이기"이고, 그 안에서 **"CPU에서 sliding 설정만으로 ring 용량이 줄지 않는다"**가 확인된 원인이다([analysis §2 표 1순위 행](../kv-cache-fsu-analysis.ko.md)). sliding 레이어의 물리 용량을 줄이는 길은 (i) windowed compaction(memmove — flash와 충돌) 또는 (ii) ring modulo(CPU attention이 modulo 읽기를 못 한다 — `kvRingEngineEligible()`이 CPU를 제외, [kv_ring.h:307](../Applications/CausalLM/kv_ring.h#L307)) 둘뿐이다.
- 핸드오프 4단계 0항이 **"(선결) compaction과 FSU 물리 레이아웃을 먼저 화해시킨다"**를 HTP 이식의 첫 항으로 요구한다. CPU sliding ring 축소와 HTP 이식이 모두 이 조합을 연다.

## 2. 설계 대안 비교

### 2.1 공통 평가 기준 — 다섯 기존 기능과의 상호작용

레이아웃 화해의 성패는 새 수식 자체보다 **기존 기능을 깨뜨리지 않는가**에서 갈린다. 평가 기준은 아래 다섯 가지다.

| 기능 | 현재 계약(근거) | 레이아웃 화해가 만지는 지점 |
|---|---|---|
| alias (kv-share) | sharer는 소유자의 `MemoryData`를 공유([aliasLayerCache](../Applications/CausalLM/kv_cache_manager.cpp#L1349)), FSU alias는 소유자 엔트리로 라우팅([alias](../Applications/CausalLM/kv_cache_manager.cpp#L189)). compact/prepareDecodeView는 aliased 레이어를 스킵 | 이동/재배치가 공유 평면에서 일어날 때 sharer가 같은 사상 변화를 따라가야 한다 |
| retention (`NNTR_KV_RESIDENT_BYTES`) | 예산 안의 소유자를 pool에 유지해 전체 이력 재읽기를 건너뜀([evict 유지 경로](../Applications/CausalLM/kv_cache_manager.cpp#L303)), `retained_resident` 회계([7장 기록](../kv-cache-fsu-cpu-device-log.ko.md)) | 이동이 유지 버퍼 안에서 일어나는가, 회계 범위(전체 평면 vs 창)가 바뀌는가 |
| write-back dirty | `store()`가 dirty 범위만 적고 퇴피 때 `flushDirty()`가 합쳐 씀([store](../Applications/CausalLM/kv_cache_manager.cpp#L248) · [flushDirty](../Applications/CausalLM/kv_cache_manager.cpp#L515)). 장부는 기록 성공 후에 지움 | 행이 이동하면 dirty 좌표도 이동해야 한다. 이동이 eviction flush를 전체 live prefix 재기록으로 증폭시키는가 |
| lookahead prefetch | `begin()`이 다음 레이어들을 워커로 선읽기([prefetch](../Applications/CausalLM/kv_cache_manager.cpp#L559)), `wait()`가 완료 대기 | 이동이 선읽기 중인 버퍼의 좌표를 무효화하는가. 대기(drain)가 겹침을 깨는가 |
| snapshot save/load | `FlashSnapshotLease`(load→evict)로 임대해 저장/복원, windowed는 창 행만 페이로드(rowsToPersist), load는 행 0 적치 + base 재설정 | 파일 규약이 바뀌는가. v1 헤더·geometry digest와의 호환이 유지되는가 |

### 2.2 대안 A — FSU 인지 compaction: 이동을 pool 버퍼와 파일 양쪽에 반영

memmove 계약을 유지하고("물리 0 = base"), FSU가 이동을 실행·반영한다. 핸드오프 4단계 0항의 (a) "FSU가 base 이동을 따라가도록 오프셋 산식을 바꾼다(재배시 비용·dirty 범위 이관 포함)"에 해당한다.

- **동작**: `compact()`의 flash 분기가 (1) 진행 중 I/O drain(`wait()`/`waitWrite()`), (2) 소유자 버퍼 확보(퇴피 상태면 load), (3) 버퍼 안 memmove, (4) dirty 장부 좌표 이관(−skip)과 `[0, live)` 재표시, (5) `bases_[l] = from`. `finishFlashLayer()`의 windowed 분기는 `start = from − base`로 교체(ring 분기의 `% cap`은 유지).
- **변경 범위**: `KVFlashStorage`에 이동·좌표 이관 API 추가, `compact()`/`finishFlashLayer()` 분기, allocate 가드 해제. 스냅샷 수식은 불변(규약이 같으므로).
- **성능 비용**: compaction마다 창 K행 이동. 유지 소유자는 DRAM memmove만(SRAM 비용), 퇴피 소유자는 읽기 2×A(K·r) + 재기록 2×A(K·r). 유지 소유자조차 이동으로 `[0, live)`가 dirty가 되어 **eviction flush가 전체 창 재기록으로 증폭**된다. decode에서 토큰 단위 compaction이면 이동·재기록이 매 토큰 반복된다.
- **리스크**: 진행 중 I/O와의 충돌 처리가 가장 크다. 선읽기가 `[0, saved)`를 읽는 도중의 이동은 버퍼 좌표를 무효화하므로 해당 레이어의 pending drain이 강제되고, lookahead 겹침과 비동기 쓰기(staging 복사본도 좌표가 무효화됨)가 직렬화된다 — 3.3/3.4가 만든 겹침 이득의 회귀. dirty 좌표 이관과 장부 순서 보장(`850e0b21e`의 수정)의 재검증이 필요하다.
- **다섯 기능 상호작용**: alias — 소유자 이동을 sharer가 따라가야 함(같은 MemoryData라 버퍼 바이트는 공유되지만 attention이 새 base로 읽어야 하고, compactKVCache의 cache_base 푸시로 커버 가능). retention — 유지 경로는 memmove만으로 저비용, 퇴피 경로가 비쌈. dirty — 좌표 이관 + 증폭. prefetch — drain 강제로 겹침 손실. snapshot — 불변(최대 장점).

### 2.3 대안 B — 세대 기반 base 상대 물리 주소: compaction은 무이동, 이동은 세대 재배치로 회수

데이터는 움직이지 않고 **물리 주소의 기준점을 "세대 원점 G"으로 옮긴다**: 물리 행 = 논리행 − G. G는 세대가 바뀔 때만 움직이고, compaction은 창 시작점 base만 앞당긴다(행은 버린다). 이동은 평면이 찰 때 한 번(rebase)으로 회수한다.

- **동작**: 레이어별 세대 원점 `gens_[l]`(초기 0)을 둔다. 평면은 세대 안에서 append-only(물리 행 p ↔ 절대 p + G)이고, `compact()`는 `bases_[l] = from` + 버려진 행의 amax fold만 한다(데이터 이동·I/O 없음 — 버려진 행 `[old_base, from)`은 세대 안에 죽은 공간으로 남는다). `finishFlashLayer()`의 windowed 분기는 `start = from − G`(단일 구간, wrap 없음 — 세대 안에서 frontier가 단조 증가하므로). frontier(`pos − G`)가 cap에 닿으면 **rebase**: live 창 `[base, pos)`를 새 세대의 `[0, …)`로 이동(버퍼+파일, 또는 파일에 새 세대 영역 할당)하고 `G = base`. 진행 중 I/O는 rebase 시점에만 drain한다.
- **binding 흡수**: 세대 안에서 창은 물리 `[base − G, pos − G)`에 있다. FSU의 `setAddr`이 binding을 `버퍼 + (base − G)·r`에 두면 host attention의 기존 수식이 그대로 성립한다 — 쓰기 행 `cache_row = cache_index − cache_base`([fp16_attention.cpp:796–806](../nntrainer/layers/llm/fp16_attention.cpp#L796))는 버퍼 절대 행 `(base−G) + (index−base) = index − G`로, 과거 컨텍스트 읽기 `[0, cache_to)`([fp16_attention.cpp:870–880](../nntrainer/layers/llm/fp16_attention.cpp#L870))는 버퍼 행 `[(base−G), (to−G))`로 정확히 창을 가리킨다. 즉 **mha_core/커널 변경 0**, (base − G) 병역은 FSU의 binding이 진다. 단, `cache_base` 속성 푸시가 현재 `HtpMHACoreLayer`만 대상이다([causal_lm.cpp:1106](../Applications/CausalLM/models/causal_lm.cpp#L1106)) — CPU `MHACoreLayer`에도 같은 푸시가 필요하다(속성 파서는 이미 존재, [fp16_attention.cpp:1774](../nntrainer/layers/llm/fp16_attention.cpp#L1774)).
- **load**: 현재 코드가 이미 (b)의 세대 리셋이다 — 행 0 적치 + `bases_[i] = liveFrom(...)` + `store(…, 0, …, true)`는 "G = liveFrom, 데이터는 물리 [0, …)" 그 자체다([kv_cache_manager.cpp:2205–2257](../Applications/CausalLM/kv_cache_manager.cpp#L2205)). `[0, saved)` 전체 읽기(1.2b)도 세대 안에서는 여전히 "유효 전부"를 뜻한다. 다만 창 밖 죽은 행 `[G, base)`까지 읽으므로, 개선으로 load를 창 범위 `[(base−G)·r, saved)`로 좁히고 회계도 그 범위로 잡을 수 있다(죽은 접두 페이지는 매핑만 있고 DRAM을 안 쓴다).
- **스냅샷**: `save_off`만 `(from − base)` → `(from − G)`로 바뀌고 페이로드(창 행 수, 행 순서)는 동일 — **v1 파일과 바이트 호환**. geometry digest에는 rows/keep이 이미 들어 있다([geometryDigest](../Applications/CausalLM/kv_cache_manager.cpp#L1931)).
- **변경 범위**: `gens_` 상태 추가, `compact()`의 flash 분기(무이동화 + rebase 검사), `finishFlashLayer`의 windowed 수식, FSU에 세대/창 원점 전달과 binding 오프셋, allocate 가드 해제, save의 오프셋 한 항.
- **성능 비용**: compaction 자체는 **0 I/O**. rebase가 세대 길이(cap − K행)마다 2×A(K·r) — 행당 이동 비용 K/(cap − K)행. 단점으로 세대 안 창 밖 죽은 공간(최대 cap − K행)이 파일과 `saved`에 남고, 퇴피 소유자의 load가 죽은 접두를 읽을 수 있다(위의 창 범위 읽기로 제거 가능). 버려진 행의 amax fold는 퇴피 소유자에서 작은 파일 읽기를 요구한다(또는 `finishFlashLayer` 시점에 행이 버퍼에 있을 때 점진 fold).
- **리스크**: 기준점이 두 개(base, G)가 되는 인지 부담. rebase 경계를 step이 가로지르지 못하게 하는 강제(ring seam throw와 같은 규칙, [mha_core.cpp:2192](../Applications/CausalLM/layers/mha_core.cpp#L2192) 참조)와 rebase 시 dirty/유지 회계 이관이 구현 리스크.
- **다섯 기능 상호작용**: alias — 이동이 세대 경계에만 있고 소유자 평면 좌표는 불변이므로 안전(sharer의 base는 소유자와 동일하게 진행). retention — 유지 버퍼가 그대로 작업 복사본이고 compaction이 무이동이라 예산 산식이 창 기준으로 단순해짐(rebase 시에만 이관). dirty — **좌표 불변**(세대 안에서 행이 안 움직이므로 장부 이관 불필요), rebase 시에만 이관. prefetch — compaction이 선읽기를 무효화하지 않음(rebase에서만 drain). snapshot — 바이트 호환 유지.

### 2.4 대안 C — 무이동 ring: compaction을 읽기 범위(wrap)로만 표현

물리 행 = 절대 위치 % cap(ring)을 windowed 레이어에도 적용하고, compaction은 읽기 시작 오프셋만 앞당긴다. 창 `[base, pos)`는 물리에서 **wrap한 두 구간**이 되고, attention이 wrap을 읽도록 바꾼다.

- **동작**: `finishFlashLayer()`의 `% cap` 유지, load/store가 두 구간(기존 ring store의 두 구간 store를 읽기로도), 창 경계의 두 구간 읽기를 커널이 처리.
- **변경 범위**: 가장 넓다. host attention(`b_cached_key`의 연속 뷰 전제), accelerated/quantized attention 경계, (HTP 이식 시) DSP 커널, FSU의 load 두 구간화, prepareDecodeView의 오프셋 합성.
- **성능 비용**: 이동 0, 읽기 = wrap 두 구간 합 = 창 K행으로 최소. 다만 풀 버퍼가 링 평면 전체(cap행)를 미러링해야 하는 경우 retention 회계가 cap행 기준이 된다.
- **리스크**: MHA 커널 영향 범위가 본질 리스크다. CPU는 ring ineligible(kv_ring.h:307)이라 "CPU ring을 줄인다"의 목적 달성에 필요한 커널 작업이 가장 크고, 스냅샷도 windowed 규약(창 행만)을 ring 규약(평면 전체 + 위치 재유도)으로 바꿔야 파일이 커지거나 포맷이 갈라진다.
- **다섯 기능 상호작용**: alias — 읽기 규약 변화가 sharer에 동일 적용돼야 함. retention — cap행 미러 부담. dirty — 좌표 불변이나 flush 범위가 두 구간으로 갈라짐(store는 이미 두 구간 지원). prefetch — 두 구간 선읽기로 변경 필요. snapshot — 규약 전환 필요(호환성 깨짐 또는 포맷 확장).

### 2.5 대안 D — 화해 보류: windowed 레이어는 DRAM 상주, FSU는 나머지 레이어 (상호배제의 정책화)

레이아웃을 화해하지 않는 대신, 조합을 **부분적으로** 연다: 한 allocate() 안에서 windowed 레이어는 flash 소유자가 아니게(호스트 텐서로 K행 상주) 하고, 나머지(global) 레이어만 FSU로 스트리밍한다. 현재 가드는 rows/keep이 하나라 있으면 flash를 전부 거부하므로([kv_cache_manager.cpp:1015](../Applications/CausalLM/kv_cache_manager.cpp#L1015)), 이 대안은 가드를 "windowed flash 소유자만 금지"로 좁히는 것이다.

- **동작**: allocate의 flash 분기에서 windowed 레이어는 `addLayer`하지 않고 호스트 텐서 유지, 나머지는 기존대로. `compact()`는 기존 memmove 경로(DRAM)를 그대로 쓰고 flash 가드는 windowed flash 소유자에만 발화(사실상 불변 belt). `bindFlashKV` 연결에서 windowed 레이어 제외([causal_lm.cpp:619–623](../Applications/CausalLM/models/causal_lm.cpp#L619)). save/load의 `FlashSnapshotLease`는 레이어별 flash 소유자 판정 필요(load는 이미 `isLayerKVAliased(i) ? nullptr : flash_.get()`로 판정, save는 무조건 임대 — [kv_cache_manager.cpp:2046](../Applications/CausalLM/kv_cache_manager.cpp#L2046)).
- **변경 범위**: 가장 작다 — allocate 분기, lease 판정, 바인딩 루프, 테스트. 레이아웃 수식 변경 0.
- **성능 비용**: windowed(sliding) 레이어가 FSU 이점(파일 스트리밍)을 못 받는다. 대신 K행만 상주하므로 메모리 목표 자체는 달성한다 — gemma4 26B 기준 sliding 전 층 논리 창 합은 FP16 약 200 MiB(INT8 100 MiB, [analysis §2](../kv-cache-fsu-analysis.ko.md))로, analysis 3.2가 제안한 "sliding의 실제 참조 구간 상주"와 같은 결과다. global 이력은 FSU가 계속 흡수한다.
- **리스크**: 혼재 구성의 검증 부담(같은 모델에 두 저장소 경로). sliding 창이 예산을 넘는 모델(창×층수가 큰 경우)에서는 절감 상한에 도달.
- **다섯 기능 상호작용**: 두 경로가 레이어별로 분리되므로 다섯 기능 모두 기존 계약 그대로 — 화해 대안 중 유일하게 상호작용이 새로 생기지 않는다. 단, windowed 레이어가 FSU를 쓰지 못하므로 이후 요구(창까지 파일에 두는 구성)가 생기면 B/C가 필요하다.

### 2.6 종합 비교

| | A. 이동 반영 | B. 세대/base 상대 | C. 무이동 ring | D. 혼재(DRAM 상주) |
|---|---|---|---|---|
| compaction I/O | 이동 2×A(K·r) (퇴피 시) + eviction flush 증폭 | 0 (rebase에서만) | 0 | 해당 없음(DRAM memmove) |
| steady-state 읽기 | 창 K행 | 창 K행(접두 죽은 행 제외 시), 미개선 시 ≤ cap행 | 창 K행(두 구간) | 0(상주) |
| 커널/attention 변경 | 없음 | 없음(binding 오프셋 흡수) | **큼**(wrap 읽기) | 없음 |
| 스냅샷 호환 | 유지 | **유지(save_off 한 항)** | 깨짐 또는 포맷 확장 | 유지 |
| dirty 장부 | 좌표 이관 + 증폭 | 불변(rebase 시만) | 불변(두 구간 flush) | 불변 |
| prefetch 겹침 | drain 강제로 손실 | 유지(rebase에서만 drain) | 두 구간화 필요 | 유지 |
| retention | 유지 버퍼 memmove, 회계 불변 | 창 기준 산식으로 단순화 | cap행 미러 부담 | 기존(비-flash 경로) |
| alias | 소유자 이동 추적 필요 | 좌표 불변이라 안전 | 읽기 규약 동기화 필요 | 분리라 안전 |
| 주요 리스크 | I/O 회귀 + 직렬화 | 기준점 2개의 인지 부담, rebase 경계 규칙 | 커널 영향 범위 | 절감 상한, 혼재 검증 |

## 3. 권장안과 단계적 이식

### 3.1 권장안

**D를 선행 정책으로, B를 레이아웃 계약으로 채택한다.**

1. CPU sliding ring 축소(1순위 목표)는 **D(혼재 구성)** 로 먼저 달성한다 — sliding은 창 K행 DRAM 상주, global은 FSU. 레이아웃 수식 변경 없이 목표를 검증할 수 있고, 실패해도 되돌림 비용이 거의 없다.
2. windowed 레이어도 파일 스트리밍이 필요해지는 시점(DRAM 예산이 창 합을 넘기거나, 저장소 균일화가 요구될 때)에 **B(세대 기반 base 상대 주소)** 를 채택한다. 근거: compaction이 무이동·무I/O로 다섯 기능과 가장 적게 충돌하고, 현재 `load()` 코드가 이미 세대 리셋 의미를 갖며, 스냅샷이 바이트 호환으로 유지되고, host attention 변경이 FSU binding 오프셋 하나로 흡수된다(2.3).
3. **A는 기본 권장에서 제외**한다 — 이동·재기록 비용이 3.2/3.4가 제거한 I/O를 회귀시키고 진행 중 I/O 직렬화가 lookahead/async-write를 무력화한다. 단 A의 이동 구현(버퍼 memmove + 파일 재기록)은 B의 rebase에 그대로 재사용되므로 폐기가 아니다.
4. **C는 보류**한다 — HTP DSP가 이미 paged window(64행 타일)를 자체 소유권으로 다루므로(4단계 5번), DSP 쪽 wrap 읽기와 함께 재평가한다. host CPU에서는 커널 영향 범위가 목적 대비 과도하다.

### 3.2 단계별 이식과 검증

| 단계 | 내용 | 검증 방법 |
|---|---|---|
| 0. 계약 문서화 (코드 무변경) | 1.1의 두 계약과 가드의 의미를 코드 주석·이 문서로 고정. 기존 40/40 유닛테스트 회귀 확인 | `meson test -C build-fsu unittest_kv_cache_manager`; 기존 `KVCacheManagerFlash.*` 전부 통과 유지 |
| 1. D — 혼재 구성 | allocate flash 분기에서 windowed 레이어를 비-flash 소유자로, 가드를 "windowed flash 소유자 금지"로 좁힘. save의 lease에 flash 소유자 판정. `bindFlashKV` 연결에서 windowed 제외 | 신규 테스트: 같은 manager에 windowed(비-flash) + flash 레이어 혼재 — compaction 후 두 종류 레이어 모두 DRAM 참조와 비트 일치, save/load 왕복, compaction 경계(창을 넘는 첫 토큰)에서 read-back |
| 2. B 기반 마련 | `gens_` 상태 추가(초기 0, 동작 불변), FSU에 세대 원점 전달 경로, windowed flash allocate 허용(아직 compaction은 가드) | 기존 전 테스트 회귀 + windowed flash 레이어의 단순 append(store/load)가 DRAM 참조와 일치 |
| 3. 무이동 compaction | `compact()` flash 분기: fold(base−G 오프셋으로 버려진 행 접기) + `bases_` 진행 + rebase 검사. `finishFlashLayer` windowed 수식 `from − G`. binding 오프셋 `(base−G)·r` | 아래 3.3의 시나리오 1–3 |
| 4. rebase(세대 재배치) | frontier 고갈 시 live 창 이동(버퍼+파일 또는 신규 영역), dirty/유지 회계 이관, I/O drain. step이 세대 경계를 가로지르면 거부 | 아래 시나리오 1·4 |
| 5. 스냅샷·회계 마감 | save_off의 G 항, 창 범위 load + 회계, 예산과의 정합 | 아래 시나리오 3·5, tiny fixture CLI 동등성 |
| 6. 노출·HTP | CPU 실그래프에서 windowed flash 활성화, 4단계 실험 행렬에 "compaction 경계" 행 추가 | 기기 매트릭스(핸드오프 문서의 판정 절차) |

**진행 기록 (2026-11, kv-cache-fsu 브랜치):**

- **1단계 완료 (fc2fec519)** — allocate flash 분기가 windowed 레이어를 host DRAM 비-flash 소유자로 두고 나머지를 스트리밍하는 혼재 구성이 실렸다. windowed가 flash plane을 alias하는 조합은 sharer 연결 전에 거부한다(아래 §4-6 결정). 혼재 3건의 신규 유닛테스트(DRAM 참조 비트 일치, compaction 경계 read-back, 스냅샷 왕복)를 포함해 42/42. 알려진 비용: lookahead 1에서 flash 레이어 뒤에 windowed 레이어가 오면 한 칸 prefetch 겹침이 사라진다(정확도 무관).
- **6단계의 CPU 반쪽 완료 (본 변경)** — CPU 실그래프가 sliding 레이어에 HTP와 동일한 windowed 상한(`keep = window + max(init_seq_len, num_to_generate)` 64행 반올림, `rows = min(max_seq, keep)`)을 쓰도록 활성화했다. `kv_flash_fsu` 구성에서 자동(`CausalLM::kvFsuWindowCap()`), 상주 실행에서는 `NNTR_KV_WINDOW_CAP=1`로 강제해 동일 상한의 A/B 기준을 잡는다. windowed 레이어의 compaction memmove가 legacy host 경로의 물리 행 계산(`mha_core::cacheRow`)과 `cache_base` 속성으로 흡수되고, `compactKVCache()`가 CPU `MHACoreLayer`에도 base를 밀어준다. `num_to_generate <= 0`(무제한 생성, `setupParameters`가 0으로 정규화)일 때는 headroom이 `max_seq - window`가 되어 수식이 전체 높이로 퇴화한다(`windowedKVRows`, 디코드가 컨텍스트 끝까지 가도록 — 유한 평면으로는 무제한 꼬리를 못 담는다). 검증과 리뷰 결함 3건의 수정은 아래 항목과 기기 로그 §11 참조. **남은 반쪽**: HTP 기기 매트릭스의 compaction 경계 행 — 기기 측 검증은 여전히 미수행.
- **리뷰 결함 3건 수정 (본 변경)** — (1) 리셋 후 재요청 FATAL: 첫 요청의 compact가 남긴 `bases_`가 `setSessionTokenPosition(0)`에서 지워지지 않아 다음 요청의 `compact(0)`이 "does not hold rows [0, 0)"로 죽었다. 새 프롬프트 경로에서 `KVCacheManager::reset()`(bases_/decode offsets/dropped maxima/flash 소유권을 원점으로)을 호출하고, 바로 이어 `compactKVCache(0, /*force=*/true)`로 `cache_base=0`을 재푸시한다 — manager reset과 base 재푸시는 한 쌍이고, `compactKVCache`의 no-op 조기 리턴은 이 쌍이 이미 맞춰진 경우에만 지름길이 된다. (2) 무제한 생성 FATAL: 위 `windowedKVRows`의 무제한 분기(전체 높이 fall-back)로 수정. (3) windowed→windowed alias의 base stale: 아래 §4-6 결정대로 `getBase`/`getDecodeOffset`가 alias 소스를 resolve한다. 검증: `unittest_kv_cache_manager` 46/46(build-fsu, fp16 — 리셋 원점 복원, rows 수식, windowed alias base 추적, 창 불일치 거부 4건 신규), `unittest_causallm_models` 124/125(builddir_x86_test, fp32, 1건 플랫폼 스킵), CLI: 리셋 2요청(상주 windowed·FSU) FATAL 없이 출력 일치, 3-way(전체평면=windowed=FSU, 단일+연속, cached_after 81→89) 일치, qwen3_tiny·원본 gemma4_tiny 회귀 유지, 무제한 생성(g4t_512, num_to_generate 없음)이 세 구성 모두 cached_after 511까지 완주하고 `output_ids`가 전체평면 컨트롤과 비트 일치.
- **§4-6 결정됨 (fc2fec519, 본 변경에서 세분화)** — windowed 소유자의 alias를 유형별로 나눠 결정했다. (i) windowed 레이어가 **flash 스트리밍 plane**을 alias하려는 조합은 지금처럼 `allocate()`에서 거부한다(windowed plane은 host DRAM 텐서여야 한다는 §2.5 D 계약의 연장 — compact의 memmove를 flash 소유자의 pool 버퍼는 받을 수 없다). (ii) **windowed→windowed alias**(gemma4 `num_kv_shared_layers`가 sliding 타입 공유 레이어를 만드는 조합)는 허용하되, sharer의 base는 소유자를 따라간다: `getBase`/`getDecodeOffset`가 alias 소스를 resolve해 `cache_base` 푸시·validateRange·save 오프셋이 모두 소유자 사상을 쓰고, `validateKVSources`가 alias 가족의 창(keep) 동일성을 강제한다. sharer의 mha_core는 소유자의 k/v norm으로 그래프가 연결되므로 같은 행에 byte-identical 값을 쓰는 redundant write가 되어, 소유자 사상 위에서는 무해하다. compact가 aliased 레이어를 스키핑하는 동작은 그대로 유지된다(이동은 소유자만).
- **실팩 검증 보류 사유**: 로컬 gemma4 실팩 전체가 `tie_word_embeddings=true` + `skip_prefill=true` 구성인데, HEAD의 `buildLmHeadOutput()`이 tied lm_head(`tie_word_embeddings` 레이어)에 `plan_last_row_only=true`를 넣고 이 속성을 그 레이어가 받지 못해 그래프 구성 단계에서 FATAL난다. 본 변경과 무관한 기존 불일치(변경 코드가 전부 dormant인 상주 경로에서도 동일 재현, 원인은 [causal_lm.cpp:1052](../Applications/CausalLM/models/causal_lm.cpp#L1052)의 속성 주입과 [tie_word_embedding.h](../nntrainer/layers/llm/tie_word_embedding.h)의 props 선언 불일치)라서 여기서 넓히지 않고 기록만 남긴다. 실규모 측정은 이 선결과제 후에 한다.

### 3.3 검증 시나리오 (유닛테스트)

기존 flash 테스트의 골격 — DRAM 참조 레이어와 비트 비교, `output_ids` 대체 가능한 스냅샷 바이트 비교, `ScopedEnv` — 를 재사용한다([unittest_kv_cache_manager.cpp:935](../test/unittest/layers/unittest_kv_cache_manager.cpp#L935) · [1116](../test/unittest/layers/unittest_kv_cache_manager.cpp#L1116) · [1265](../test/unittest/layers/unittest_kv_cache_manager.cpp#L1265)).

1. **세대 경계를 지나는 compaction (65행 링 테스트의 windowed 버전)**: cap=65, keep=32인 windowed flash 레이어를 40 → 65 → 71로 전진. frontier가 cap(65)에 닿아 rebase가 발생하는 경계와, rebase 직후 첫 store·첫 load를 포함한다. DRAM windowed 참조(memmove 경로)와 전 행 비트 일치, `droppedAmaxK/V`도 fold 값 일치. 기존 비windowed 65행 ring 테스트(`persists_ring_alias_and_existing_snapshot_format`)는 회귀로 유지 — 두 규약의 공존을 동시에 고정한다.
2. **retention 상주 중 compaction**: `NNTR_KV_RESIDENT_BYTES`로 소유자를 유지한 채 compaction 여러 번 — `read_bytes` 증가 0(무이동 확인), evict 시 flush 바이트가 창 크기로 수렴, `retained_bytes` 회계가 창 기준 정확값(기존 정확값 검증 스타일 유지). 예산 부족으로 퇴피한 소유자의 compaction(작은 파일 읽기로 fold)도 분리 검증.
3. **snapshot 경계**: rebase 직전 save와 직후 save가 DRAM 참조와 같은 바이트(v1 호환), rebase 경계에서 load 후 이어쓰기·이어읽기, 기존 windowed 스냅샷 파일의 로드 호환(`valid_window_snapshot_round_trip` · `dense_files_remain_compatible_in_both_directions`의 flash 버전).
4. **I/O 경합**: `NNTR_KV_ASYNC_WRITE=1` + lookahead 1/3 조합에서 rebase·compaction이 진행 중 쓰기/선읽기를 drain한 뒤 진행하는지 — 미완료 전송 슬롯 재사용 금지, 실패 주입 시 이전 정상본 보존 또는 명시적 요청 실패(핸드오프 3단계 테스트 행 규칙 준용).
5. **예산·회계 상호작용**: 창 기준 working set으로 예산을 고정했을 때 `read MB/token = 0` 수렴(8장의 산식이 windowed 레이어에서도 성립하는지), `peak_kv_bytes`/`retained_bytes`가 창 범위 회계와 일치.
6. **동등성**: tiny fixture CLI(qwen3_tiny/gemma4_tiny)로 resident vs D(혼재) vs B(windowed flash)의 `output_ids`·위치 일치 — 호스트 핸드오프 문서의 비교 절차 재사용.

## 4. 열린 질문 (결정 필요 사항)

1. **예산 정책과의 상호작용**: `NNTR_KV_RESIDENT_BYTES`는 owner 단위·`buffer.resident` 기준이다. B에서 windowed 소유자의 작업 집합은 창 K행(죽은 접두는 미페일 페이지)이 되므로, 예산의 단위(전체 평면 vs 창 범위)와 채움 순서(레이어 인덱스 우선)를 windowed에 맞게 재정의할지. 유지 중 소유자의 rebase가 예산을 초과할 때 즉시 퇴피할지 다음 evict로 미룰지.
2. **HTP 64행 페이지와의 단위 정렬**: DSP q8 페이지는 한 레이어·한 KV head·64토큰이다. compaction/rebase 단위를 64행 격자에 정렬하면 타일 분할이 없다(부분 타일은 DSP에 유지 — 4단계 5번과 일치). rebase 경계를 64의 배수로 강제할지, 미완성 타일의 host 반올림 규칙.
3. **rebase 트리거와 파일 영역**: frontier 고갈 시 필수 rebase만 할지, 쓰기 증폭 평준화를 위한 선제 rebase를 둘지. 파일 영역은 제자리 이동(용량 불변)과 세대별 신규 영역(`next_offset_` 진행 — 세션 로컬 파일이라 성장 한도 관리 필요) 중 택일.
4. **`cache_base` 속성 의미의 확정**: B는 binding 오프셋으로 흡수해 mha_core 무변경이 가능하지만, `prepareDecodeView()`의 `decode_offsets_`와 (base − G) 오프셋의 합성 규칙, 그리고 HTP seed가 기대하는 원점(G 또는 base)을 문서·코드로 고정해야 한다.
5. **스냅샷 호환 범위**: v1 헤더 유지(바이트 호환)로 갈지, 세대 정보를 헤더에 올려 재개 외 시나리오(크래시 후 임시 파일 재해석 등 — 현재는 지원 안 함)까지 열지.
6. **alias된 windowed 소유자** — **결정됨 (본 변경, §3.2 진행 기록 참조)**: windowed→flash 소유자 alias는 거부(상호배타 유지), windowed→windowed alias는 허용하고 `getBase`/`getDecodeOffset`가 소스를 resolve해 sharer가 소유자의 base를 따라간다. 소유자 이동을 sharer의 `bases_`에 복사하는 방안은 채택하지 않았다 — base 변경 지점(compact·prepareDecodeView·restore·reset)마다 복사를 맞춰야 하는 반해, resolve는 읽는 곳이 한 곳이라 상태가 분열할 수 없다. 유일한 전제 조건인 "alias 가족은 같은 창"은 `validateKVSources`가 allocate 시점에 강제한다.
7. **DRAM windowed 경로의 통일 여부**: B 채택 후에도 DRAM 경로는 memmove를 유지할지(호스트 memmove는 저렴), 같은 세대 계약으로 통일해 경로를 하나로 줄일지. 통일 시 기존 windowed 유닛테스트와 HTP window 스냅샷의 회귀 범위.
