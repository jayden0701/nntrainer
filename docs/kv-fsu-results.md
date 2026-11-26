# CPU KV cache offloading on Android

On Android, full streaming beat partial retention on both long-prompt Qwen3-1.7B requests. In the 700-token run it used **20.20 vs 42.01 MiB** of KV and finished in **75.44 vs 157.38 seconds**. Gemma4-E2B behaved differently. Keeping 8 MiB available for retention cut the same-length request from **104.30 to 54.11 seconds** at the same **28.10 MiB** KV peak. Small retention limits gave Gemma little benefit. The useful comparison depends on the model, context, memory ceiling and latency metric.

## Terms and scope

| Term | Meaning |
|---|---|
| KV cache and owner | Attention reuses keys (K) and values (V) computed for earlier tokens. An owner holds one distinct K/V pair. Several layers can share it. |
| Resident and FSU | The resident reference keeps all required KV in DRAM. FSU can write KV to a file in phone storage and read it back. A resident *owner* is one pair kept across tokens inside an FSU run. |
| Prefetch and depth | Prefetch starts a file read before attention needs the data. Depth limits the number of distinct upcoming owners queued for reading. A queued read may still be in flight when needed. |
| Total cap C and retention limit R | C bounds KV payload for windows, active and prefetched owners, retained owners and write staging. R bounds the portion retained across tokens. A setting of 16/8 means C = 16 MiB and R = 8 MiB. Process memory also includes weights and other allocations. |
| Fixed and rolling | Fixed prefetch uses free space. Rolling may reclaim a later-use retained owner and retries prefetch when a layer finishes. |
| Decode speed and request time | Tokens/s covers generation. Request time includes initial prompt processing (prefill). Steady p95 is the 95th percentile of token latency after the first decode step. |

The baseline is the CPU prototype at `13cbb99ea`. The measurements were collected on SM-S947N, with seven CPU threads and direct I/O for the reported streaming comparisons. The source branch contains both a baseline comparison and separate policy sweeps. Each answers a different question. [Baseline comparison records](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/cmp) · [Policy measurements](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs)

## What changed

| Prototype behavior | Revised behavior |
|---|---|
| The prototype already overlapped some reads with compute and supported asynchronous large prefill writes. At the next layer it waited for the previously submitted layer range. Shared and inactive layers could consume lookahead distance. | Queue reads by distinct owner and wait for the owner about to be consumed. DRAM-only sliding layers can compute while a later global owner is being read. |
| Releasing an owner after attention forced its saved history to be read again on the next token. Small decode appends could rewrite the same file page. | Keep selected owners under R. Merge adjacent dirty ranges while retained and flush on eviction. Account for growth when reserving memory under C. |
| A uniform storage policy poorly fit models mixing global attention, which needs all valid history, and sliding attention, which needs a recent window. | Keep compact sliding windows in DRAM, stream global owners and reuse aliases. Restore the live suffix from older dense snapshots. Accept the tied output-head planning property used by the Gemma pack. |
| The original counters reported aggregate I/O. | Record KV payload peaks, demand reads, prefetch waits, borrowing and steady token latency. The budget tool supports a fixed memory allowance or a target latency. |

[CPU KV manager](../Applications/CausalLM/kv_cache_manager.cpp) · [Model integration](../Applications/CausalLM/models/causal_lm.cpp) · [Budget tool](../tools/kv_cache/kv_budget_predict.py)

## Same policy, before and after

Qwen3-1.7B used identical requests with depth 1, full streaming and no cross-token retention. All four generated token sequences and cache positions matched the current resident reference.

| Request | Generated tokens | Prototype tokens/s | Revised tokens/s |
|---|---:|---:|---:|
| Short prompt | 192 | 18.25 | 16.66 |
| Long prompt and generation | 192 | 12.39 | 12.35 |
| Continue the existing context | 59 | 11.26 | 11.18 |
| New request after reset | 192 | 17.31 | 18.13 |

Both runs read **30.53 GB** across the four requests. Continuation times were 5.436 and 5.472 seconds. A single pair gives limited evidence about smaller speed differences. On the revised build, the largest reported process-memory figure across requests was 1,165,216 KB for full streaming and 1,381,988 KB for resident, a difference of about 212 MiB. The baseline Qwen resident raw log is absent from this comparison. The baseline failed to load the tested Gemma4-E2B pack with an unknown layer-property error. The revised resident build completed all four requests and generated the continuation at 16.34 tokens/s. [Same-session runs](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/cmp)

At depth 2, waiting for the next owner individually improved Qwen's continuation median from 90.86 to 83.64 ms/token in the build comparison, averaging two runs per build. Summing all four request times per run gave 46.03 s for the prototype and 46.94 s for the revised build. At depth 4 those totals were 54.85 and 49.85 s. The two repetitions support a continuation improvement at depth 2 and leave the overall winner at the best observed depth uncertain. [Depth-matched prototype and revised logs](https://github.com/jayden0701/nntrainer/tree/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/opt)

## Same budget, different policy

The following continuation measurements compare configurations within the revised implementation.

| Model and budget (C/R) | Fixed → rolling, tokens/s | Request time | What the I/O counters showed |
|---|---:|---:|---|
| Qwen3-1.7B, 10/6 MiB | 3.12 → **11.74** | 19.60 → **5.22 s** | Demand reads fell from 11.95 s to zero. Rolling borrowed space and read slightly more data. Waiting for already submitted reads still took 1.28 s. |
| Qwen3-1.7B, 16/8 MiB | **12.70** → 11.66 | **4.84** → 5.29 s | Both saw zero demand reads. Fixed still waited 1.38 s for prefetch completion. A separate fixed repeat had a substantially worse p95. |
| Gemma4-E2B, 32/4 MiB | 6.94 → **12.11** | 27.97 → **16.15 s** | Both read 2.61 GB in 1,147 owner reads. Borrowing stayed at zero. Rolling submitted reads earlier and demand reads fell from 6.94 s to zero. Read service time also differed. |

At 1,207 tokens Gemma's three global owners together occupied 7.08 MiB, in addition to its sliding windows. At 40/8 MiB all three stayed in memory and file reads fell to zero. Qwen's 28 global owners occupied 111.56 MiB at 1,020 tokens. At 160/128 MiB every owner stayed resident, yet generation still took 2.231 s for 59 tokens, versus 1.954–2.055 s in three resident runs. This **8–12% throughput gap** remains unexplained. Neither the zero I/O counters nor the timing of the runs establishes its cause. [Qwen measurements](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/qw) · [Gemma measurements](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/g4)

## Choosing a budget

### Long prompts and the cost of partial retention

Both scenarios started with an empty cache. The recorded prefills were 1,024 tokens for Qwen and 1,023 for Gemma. A generated 48 tokens and B generated 700. Every FSU row used fixed scheduling, depth 2 and direct I/O. Full streaming had no total cap and R = 0. Positive-R runs used a cap and enabled write-back, so their differences include buffer and write policy. Each row gives the original single runs; the Gemma repeats are identified below. The time columns use the logged steady **median** and whole-request wall time. Retained owners are counted at the end of B.

| Model | C/R MiB | Steady ms/token A / B | Request seconds A / B | KV peak MiB A / B | B file reads, GB | B retained owners |
|---|---|---:|---:|---:|---:|---:|
| Qwen | uncapped/0 | 51.94 / 90.46 | 11.90 / 75.44 | 12.56 / 20.20 | 110.31 | 0/28 |
| Qwen | 48/24 | 73.81 / 219.11 | 13.01 / 157.38 | 33.50 / 42.01 | 94.85 | 3/28 |
| Gemma | uncapped/0 | 112.41 / 143.73 | 12.00 / 104.30 | 24.28 / 28.10 | 13.80 | 0/3 |
| Gemma | 32/2 | 127.97 / 162.36 | 12.21 / 122.83 | 24.28 / 28.10 | 13.80 | 0/3 |
| Gemma | 32/4 | 132.17 / 146.63 | 12.20 / 103.14 | 24.28 / 28.10 | 11.83 | 1/3 |
| Gemma | 32/8 | 51.46 / 55.55 | 8.06 / 48.38 | 24.28 / 28.10 | 1.13 | 2/3 |
| Gemma | 48/8 | 60.35 / 64.26 | 9.74 / 54.11 | 24.28 / 28.10 | 1.13 | 2/3 |

**Qwen.** On B, 48/24 reduced owner reads from 19,600 to 16,781. Mean elapsed time per owner read rose from 2.93 to 7.94 ms, and consumer wait rose from 17.86 to 88.55 s. Both runs had zero demand-read time. This locates an exposed read delay; separating reader scheduling, destination-page preparation and storage service requires tracing. In the four-request workload at the same C = 48 and depth 2, switching R from 0 to 24 reduced E2E from 55.31 to 36.08 s. Two short requests ending at 210 tokens fitted all 28 owners in 24 MiB. Those two requests alone accounted for 13.96 s of the 19.23 s saved, including one after reset. [Long-prompt logs](https://github.com/jayden0701/nntrainer/tree/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/scen) · [Matched 48/0](https://github.com/jayden0701/nntrainer/blob/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/qw/sweepA_C50331648_R0_la2_r0_rep1.log) · [Matched 48/24](https://github.com/jayden0701/nntrainer/blob/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/qw/sweepB_C50331648_R25165824_la2_r0_rep1.log)

**Gemma.** At R = 4 on B, fewer bytes bought little elapsed-time improvement and median token latency was slightly higher than streaming. At R = 8, all three global owners fit on A and two fit at the end of B. The separate 32/8 and 48/8 B runs read exactly 1,132,773,376 bytes each and both recorded zero budget blocks. Their time difference cannot establish a benefit from lowering C. In the corrected A repeats, resident and 32/8 had paired steady medians of 51.43 vs 51.43 and 51.63 vs 53.35 ms/token. Use the `y1/y2` runs. The earlier `mixed_x1/x2` used the resident path and generated 192 tokens. [Original workloads and corrected repeats](https://github.com/jayden0701/nntrainer/tree/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/scen)

**Gemma memory accounting.** At 1,723 tokens the 12 unique sliding caches hold 6.00 MiB of useful recent KV and reserve 18.00 MiB with prefill headroom. Each of three global owners needs 3.37 MiB. At the end of B, 48/8 retained two global owners, or 6.73 MiB. During attention, the active and prefetched owners also occupy space. Counting distinct buffers once, one, two and three global owners bring the footprint including the sliding reservation to **21.37, 24.73 and 28.10 MiB**. A 32 MiB cap accommodates all three. A and B resident logs report process memory but lack a separate measured resident KV peak. A percentage based on theoretical payload or capacity must identify that denominator. [Gemma resident B](https://github.com/jayden0701/nntrainer/blob/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/scen/g4_B_res.log) · [Gemma 32/8 B](https://github.com/jayden0701/nntrainer/blob/51515d4efbd042bff98b1cdd15a8e7302117e9f9/tools/kv_cache/roll74_device_logs/scen/g4_B_mixed.log)

### Four-request budget sweep

A larger R reliably reduced rereads in the Qwen 48 MiB sweep, but the fastest setting depended on the request. At the same C, R = 8 MiB gave the lowest measured p95 for the continuation. R = 32 MiB gave the lowest measured p95 for the preceding long request and completed the four-request set in 34.72 s, versus 41.81 s at R = 8 MiB. Most settings were measured once. An independent 16/8 repeat crossed a 100 ms p95 target that the first run had met. Across the 55 capped runs in the generous-budget sweep there were **zero prefetch borrow events**. Those runs measure retention, scheduling and run-to-run variation; the earlier tight-budget Qwen runs exercised borrowing.

These are Android CPU results. A device-specific budget needs matched workloads and independent repeats. The next measurements should explain the all-resident CPU gap and establish an output-equivalent HTP tile round trip before tuning its budget. [Engineering handoff](kv-fsu-handoff.md)
