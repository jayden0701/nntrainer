# CPU KV cache offloading on Android

Qwen can keep generating with a small KV memory allowance by moving attention history between a file and DRAM. With a 10 MiB allowance, changing when the runtime fetched that history cut one 59-token continuation from **19.60 to 5.22 seconds**. Running the old full-streaming policy on the revised code produced nearly the same continuation speed as the prototype. The gain depended on which policy the memory allowance made possible.

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

## Same budget, different policy

The following continuation measurements compare configurations within the revised implementation.

| Model and budget (C/R) | Fixed → rolling, tokens/s | Request time | What the I/O counters showed |
|---|---:|---:|---|
| Qwen3-1.7B, 10/6 MiB | 3.12 → **11.74** | 19.60 → **5.22 s** | Demand reads fell from 11.95 s to zero. Rolling borrowed space and read slightly more data. Waiting for already submitted reads still took 1.28 s. |
| Qwen3-1.7B, 16/8 MiB | **12.70** → 11.66 | **4.84** → 5.29 s | Both saw zero demand reads. Fixed still waited 1.38 s for prefetch completion. A separate fixed repeat had a substantially worse p95. |
| Gemma4-E2B, 32/4 MiB | 6.94 → **12.11** | 27.97 → **16.15 s** | Both read 2.61 GB in 1,147 owner reads. Borrowing stayed at zero. Rolling submitted reads earlier and demand reads fell from 6.94 s to zero. Read service time also differed. |

At 1,207 tokens Gemma's three global owners together occupied 7.08 MiB, in addition to its sliding windows. At 40/8 MiB all three stayed in memory and file reads fell to zero. Qwen's 28 global owners occupied 111.56 MiB at 1,020 tokens. At 160/128 MiB every owner stayed resident, yet generation still took 2.231 s for 59 tokens, versus 1.954–2.055 s in three resident runs. This **8–12% throughput gap** remains unexplained. Neither the zero I/O counters nor the timing of the runs establishes its cause. [Qwen measurements](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/qw) · [Gemma measurements](https://github.com/jayden0701/nntrainer/tree/16ef928445e76be4ac36bcd7a5fbcab528e21fac/tools/kv_cache/roll74_device_logs/g4)

## Choosing a budget

A larger R reliably reduced rereads in the Qwen 48 MiB sweep, but the fastest setting depended on the request. At the same C, R = 8 MiB gave the lowest measured p95 for the continuation. R = 32 MiB gave the lowest measured p95 for the preceding long request and completed the four-request set in 34.72 s, versus 41.81 s at R = 8 MiB. Most settings were measured once. An independent 16/8 repeat crossed a 100 ms p95 target that the first run had met. Across the 55 capped runs in the generous-budget sweep there were **zero prefetch borrow events**. Those runs measure retention, scheduling and run-to-run variation; the earlier tight-budget Qwen runs exercised borrowing.

These are Android CPU results. A device-specific budget needs matched workloads and independent repeats. The next measurements should explain the all-resident CPU gap and establish an output-equivalent HTP tile round trip before tuning its budget. [Engineering handoff](kv-fsu-handoff.md)
