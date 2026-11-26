# Engineering handoff for CPU and HTP KV offloading

Start with two independent tasks. On CPU, locate the 3–5 ms per generated token still missing when every Qwen KV owner fits in memory. On Qualcomm's Hexagon Tensor Processor (HTP), make one completed KV tile survive eviction and restoration during a live decode. CPU diagnosis and the HTP resident reference run can proceed in parallel. [Results and definitions](kv-fsu-results.md)

## Reproduce the CPU reference

Use one build, model, tokenizer, request sequence, generation settings, thread count and I/O mode per comparison. Keep the four operations in their original order: short request, long prompt, continuation, reset and new request. For the historical Qwen workload the continuation starts at cache position 944, ends at 1,020 and generates 59 tokens. Gemma starts its continuation at 984, ends at 1,207 and generates 192 tokens. On a different workload, record the real positions. Build the Android CPU application with FP16 support. Use the same device and record build SHA, model and pack settings with each run.

| Control | What to set or verify |
|---|---|
| Resident reference | Omit `kv_flash_fsu` from the pack. Use `NNTR_KV_WINDOW_CAP=1` when comparing compact sliding windows. Check the actual run had no FSU report. |
| Depth | Set `kv_flash_fsu.lookahead` in the pack. Read the logged `lookahead`. The report prints a constant `owner_lookahead=2` across different depths. |
| C and R | Set `NNTR_KV_TOTAL_BYTES` and `NNTR_KV_RESIDENT_BYTES` in decimal bytes. With a cap, omitted R defaults to C. Set R explicitly when comparing policies. R = 0 keeps the sliding windows required by the model. |
| Rolling | Set `NNTR_KV_ROLLING` to 0 or 1. Rolling requires C. Check the reported cap, retention, depth, policy, I/O mode and KV peak for every request. |
| Counters | Set `NNTR_KV_TIMING=1`. Subtract consecutive totals for per-request reads, bytes, wait times, backing writes and borrowed owners. The process-lifetime peak is a high-water mark. |
| Timing | Use the generation line for decode tokens/s and `e2e_ms` for request time. `steady_ms` is a median. `demand_read_us` counts reads begun at consumption, while `consumer_wait_us` counts time waiting for an already queued read. |

Example for a host POSIX shell with the CPU binary deployed in `/data/local/tmp/nntr-cpu`. Set `PACK` and `REQ` to device-relative paths, and `C`, `R`, `MODE` and `RUN_LOG` for one run. Use a new local log file for each process. Group files by model, workload, C, depth, rolling and I/O policy, with several R values and matching `resident*.log` references in each group.

```sh
adb shell "cd /data/local/tmp/nntr-cpu && LD_LIBRARY_PATH=. NNTR_ENGINE=cpu NNTR_NUM_THREADS=7 NNTR_KV_TIMING=1 NNTR_KV_TOTAL_BYTES=$C NNTR_KV_RESIDENT_BYTES=$R NNTR_KV_ROLLING=$MODE ./nntr_causallm '$PACK' --requests '$REQ'" > "$RUN_LOG" 2>&1

python3 tools/kv_cache/kv_budget_predict.py calibrate --logs "$GROUP" --rows "$ROWS" --device "$DEVICE" --model "$MODEL" --out "$PROFILE"
python3 tools/kv_cache/kv_budget_predict.py solve --profile "$PROFILE" --rows "$ROWS" --memory kv-peak --cap "$C" --metric p95
python3 tools/kv_cache/kv_budget_predict.py solve --profile "$PROFILE" --rows "$ROWS" --memory kv-peak --metric p95 --target "$TARGET_MS"
```

Use the observed `cached_after` for `ROWS`. The first solve selects the fastest measured p95 within an observed KV peak allowance. The second selects the smallest measured KV peak meeting a p95 target. A run measured at a 112 MiB cap with a 44 MiB peak still needs a rerun at the intended deployment cap. Select candidates on one set of runs and validate them on fresh repeats. The historical `parse_kv_roll_logs.py` helper is useful for navigating files. Its summary labels steady p50 as a mean, sums cumulative I/O counters, and can omit first-request bytes from bytes/token. Use raw request deltas and compare complete token IDs and cache positions for sign-off. [Budget selector](../tools/kv_cache/kv_budget_predict.py) · [Selector tests](../tools/kv_cache/test_kv_budget_predict.py)

## CPU work in priority order

1. **Reproduce the all-resident gap.** Alternate resident / FSU 160/128 / FSU / resident and reverse the order on the next block. In the historical run, generation took 2.231 s versus 1.954–2.055 s. KV file I/O was zero. The logged drain was 18.4 microseconds for the entire continuation. Resident controls ran earlier in the device session. Record paired differences in ms/token and their spread, plus clocks or thermal state when available.
2. **Attribute the time.** Measure layer entry management, the common attention body, layer exit management and end-of-token owner traversal separately. `finishStep()` stops the current drain timer *before* traversing the owners. Accumulate timings in memory, report per request and rerun with timing disabled. If management explains the gap, change only the measured hot path and retest full retention, partial retention and reset. If the gap moves with run order, collect matched runs before assigning an implementation cost. [KV lifecycle](../Applications/CausalLM/kv_cache_manager.cpp) · [Attention hooks](../Applications/CausalLM/layers/mha_core.cpp)
3. **Find the borrowing boundary.** At the original Qwen context, hold R at 6 MiB and try C of 10, 12 and 16 MiB. For Gemma, compare fixed/rolling at 32/5 and 24/5 MiB. Its observed sliding footprint was 18 MiB and its three global owners together reached 7.08 MiB. Record which settings actually increment `borrowed_owners`, peak memory, bytes reread, demand/consumer waits, tokens/s and p95. Use more than one context length and alternate run order. For Qwen 112/32, complete the depth 2/4 × fixed/rolling comparison. Keep write policy fixed in a separate experiment when isolating retention from write-back.

## Port to HTP in independently testable steps

The Android CPU is the host. The DSP is the accelerator-side processor. UFS is the phone's storage device. A shared buffer is memory mapped for CPU and DSP access. VTCM is limited DSP-local scratch memory. The existing optional paged Gemma4 HTP decode path stores a 64-token page of K and V per layer/head, with INT8 values and quantization state. The existing session checkpoint saves and restores whole states. The new task concerns tiles evicted while the same request is still decoding.

| Milestone | Implementation and evidence |
|---|---|
| **1. Establish a device reference** | Build matching ARM application/libraries and DSP skeleton. Start with the supported paged Gemma4 HTP profile. Run resident prefill, continuation and reset. Save the full output IDs, cache positions, sliding bases, native KV state and peak allocations. Use this same HTP configuration for correctness comparisons. [Configuration and page semantics](gemma4-htp-paged-sessions.md) |
| **2. Round-trip a closed global tile** | Start synchronously. Export both K and V, the valid row count, session/owner/head identity, logical position and the quantization state needed to interpret the bytes. Keep the open tail and sliding windows resident. Write successfully, relinquish the original physical page, restore it and continue. Compare K/V bytes, scales, positions and subsequent outputs. Demonstrate reclaimed capacity or physical page reuse. Read the session restore code for validation and failure isolation before designing a live tile interface. [DSP restore control](../nntrainer/tensor/htp_backend/dsp/nntr_hvx_attn_m1.c) · [Checkpoint validation](gemma4-htp-kv-manager-validation.md) |
| **3. Enforce a fixed budget** | Count concurrent DSP pages, open tails, scales, page tables, host staging and in-flight transfers. Count a shared physical allocation once. All valid historical global tiles must be available when consumed. Begin with a cap that can hold the active owner's complete working set. Report the minimum required memory if it cannot fit. Measure KV payload and process/device memory separately. |
| **4. Overlap I/O and compute** | Start one prefetch ahead. Measure host file submission/completion, availability to the DSP and first/last DSP use. A fused graph may execute multiple layers in one RPC; identify an observable completion boundary and measure any graph-split or notification cost. Confirm cache visibility under the chosen shared-memory interface. Reuse a slot only after all DSP and file users finish. Compare fixed and rolling at equal C, R and depth; add a deeper queue only if throughput or tail latency improves at the same memory limit. |

### Failure cases to inject

| Test | Expected outcome |
|---|---|
| Save and continue at 63, 64, 65, 127, 128 and 129 valid rows, then cross a sliding-window rollover. | The tail appends correctly, K and V remain distinct, quantization scales and the first valid position match resident, and all subsequent IDs match. |
| Delay a read while resetting, switching sessions or writing to a page shared by two branches. | A stale completion cannot publish into the new session. A shared immutable page remains valid for its other user. |
| Interrupt after writing K, truncate V, corrupt a tile ID or make restore fail after allocation. | Publish a replacement only after validating the complete tile. Preserve the last valid state, release temporary allocations, return an error and verify a healthy retry. |
| Fill the cap with active work and delay the last write or DSP reader. | Speculative reads yield capacity. Required work either progresses or reports insufficient memory. No slot is reused early and memory returns to its expected level after reset. |

**Deliverables for the next review.** Bring a CPU attribution table in ms/token with paired repeats, and a DSP tile round-trip with output equivalence and evidence of page release. For the first overlap result, include the measured intervals, throughput, p95 and memory peak. A zero demand-read counter alone cannot establish how much I/O overlapped computation.
