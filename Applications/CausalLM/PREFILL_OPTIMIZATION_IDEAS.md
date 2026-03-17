# CausalLM Prefill Optimization Ideas

This note is a **brainstorming backlog** for improving **prefill latency / throughput** in `Applications/CausalLM`.

> Scope: prompt processing path before token-by-token decode dominates.

## 1) Graph / schedule level ideas

1. **Split execution mode explicitly into `prefill` vs `decode` graph plans** so kernel choices, threading, and memory policies are specialized per phase.
2. **Build static shape-specialized subgraphs for common prompt lengths** (e.g., 128/256/512/1024) to avoid runtime shape branching overhead.
3. **Use two-stage prefill scheduling**: (A) embedding+QKV projection burst, (B) attention+MLP burst; tune threads separately.
4. **Enable layer pipelining across CPU cores** for long prompt chunks (producer-consumer between QKV and attention kernels).
5. **Add prefill micro-batching scheduler** that merges multiple user prompts with similar lengths when interactive SLA allows.
6. **Introduce asynchronous prefill queue** with deadline-aware admission control (short prompts prioritized).
7. **Token length bucketing** to reduce padding waste when batched prefill is used.
8. **Adaptive chunk size prefill**: choose chunk length based on cache size and memory bandwidth.
9. **Pipeline parallel prefill across NUMA nodes** with per-node cache shard ownership.
10. **Early-stop prefill for system prompt cache hit** at exact token boundary to skip duplicate compute.

## 2) KV cache and memory ideas

11. **KV cache page allocator** (fixed-size pages) to reduce large contiguous allocations and fragmentation.
12. **Use huge pages for KV cache** when available to reduce TLB misses on long prompts.
13. **Cache-friendly KV layout variants** selectable by kernel (`[head][seq][dim]` vs `[seq][head][dim]`).
14. **On-write transposed KV format during prefill** so decode reads are contiguous without extra transform.
15. **Layer-wise KV precision policy**: early layers FP8/FP16, deeper layers FP16/BF16 depending on quality budget.
16. **Quantized KV cache with per-head scale** (int8/int4) only for long-context sections.
17. **Hybrid KV residency**: keep recent tokens in DRAM and cold prefix in mapped storage with async prefetch.
18. **Ring-buffer KV mode for sliding-window models** to avoid full copy/shift operations.
19. **Double-buffer KV write path** to overlap compute and memcpy/store.
20. **NUMA-aware KV placement** binding each layer's cache near its worker threads.

## 3) Attention kernel ideas

21. **Dedicated prefill attention kernels** optimized for `Q_len >> 1` (unlike decode kernels tuned for `Q_len=1`).
22. **FlashAttention-style tiled softmax** for long prompt chunks to reduce memory traffic.
23. **Fused RoPE + QK matmul prep** so rotated tensors are never materialized separately.
24. **GQA/MQA specialization path** that avoids repeated expansion work per query head.
25. **Block-sparse prefill attention option** for prompts with known sparse patterns.
26. **Causal mask generation on-the-fly in kernel** instead of materializing large mask tensors.
27. **Kernel autotuner cache** keyed by `(seq_len, head_dim, num_heads, dtype, cpu_features)`.
28. **Vectorized softmax with compensated reduction** to improve both speed and numerical stability.
29. **Prefetch hints / software pipelining** inside attention inner loops.
30. **Attention compute/IO overlap**: async load next K/V tile while current tile computes.

## 4) MLP and projection ideas

31. **Fuse RMSNorm + matmul input cast** (especially for fp16/bf16 input paths).
32. **Fuse gate/up projections for SwiGLU** into one packed GEMM where possible.
33. **Prepack linear weights for prefill GEMMs** with architecture-specific blocked format.
34. **Persistent threadpool for GEMM-heavy layers** to remove per-layer launch overhead.
35. **Dynamic thread capping per GEMM size** to avoid oversubscription for small prompt chunks.
36. **Weight streaming order tuned for LLC reuse** across adjacent layers.
37. **Quantized weight dequant + matmul fusion** (int4/int8 weights) to reduce memory bandwidth pressure.
38. **Bias/activation fusion in linear epilogue** where model architecture allows.

## 5) Tokenization and input pipeline ideas

39. **Parallel tokenizer path** for batched user requests.
40. **Pinned reusable token buffers** to avoid repeated allocations for `input_ids`.
41. **Prompt canonicalization cache** (normalized system prompt + template) to improve prefix reuse hit-rate.
42. **Fast-path for already-tokenized inputs** in benchmark/service mode.
43. **Streaming prefill ingestion**: begin model prefill before full user prompt tokenization completes.

## 6) Prefix/prompt caching ideas

44. **Prefix hash index with LRU** for reusable prompt segments (beyond static system prompt).
45. **Hierarchical prefix cache**: exact token match first, then longest common prefix fallback.
46. **Persistent KV snapshot format versioning** so precomputed caches survive app restarts safely.
47. **Cross-session shared readonly prefix cache** for common instruction templates.
48. **Partial-layer prefix cache reuse** (reuse early layers for fuzzy prefix matches where exact reuse fails).
49. **Cache admission policy based on future reuse score** rather than first-seen insert.
50. **Background compaction/eviction thread** to keep cache lookups predictable.

## 7) MoE-specific prefill ideas

51. **Expert routing warmup pass** over prompt blocks to prefetch likely experts asynchronously.
52. **Route-consistent chunking**: chunk prompt so expert sets are stable and cache hits increase.
53. **Expert prefill cache residency hints** (keep top-N frequently activated experts hot during prefill window).
54. **Batch-by-route prefill for MoE** to execute same-expert tokens together when latency budget allows.
55. **Compressed expert weight staging** with just-in-time decode into compute buffer.
56. **Router threshold tuning per prompt length** to reduce expensive tail experts during long prefill.

## 8) Runtime / system ideas

57. **CPU affinity presets** (`throughput`, `latency`, `balanced`) for prefill.
58. **OpenMP / thread backend auto-tuning** at startup with lightweight calibration.
59. **Memory bandwidth governor integration** (where supported) to lock high-performance state during prefill.
60. **Asynchronous disk I/O for FSU** with deeper readahead tuned for sequential prompt traversal.
61. **IO scheduler hinting** for model-weight and cache file access patterns.
62. **Use `madvise` / `posix_fadvise` patterns** for predictable prefill scan regions.
63. **Background page fault pre-touching** for model segments needed in first decoder blocks.

## 9) Algorithmic ideas (quality/speed trade-off knobs)

64. **Prompt truncation heuristics by semantic salience** instead of naive tail truncation.
65. **Layer dropping during prefill only** (for draft/warm start), followed by full decode from checkpoint.
66. **Speculative prefill** with smaller draft model to propose KV approximations, then verify/correct.
67. **Adaptive RoPE scaling shortcuts** for very long prompts if quality target permits.
68. **Mixed precision schedule by token position** (older prompt tokens more aggressively compressed).
69. **Selective attention heads at prefill** for low-importance segments.
70. **Chunk-level early exit** when downstream confidence metric suggests diminishing returns.

## 10) Observability and tuning ideas

71. **Per-layer prefill timeline tracing** (compute, wait, memory stall, io stall).
72. **KV cache hit/miss and reuse metrics** split by system/user prompt portions.
73. **Prompt-length latency percentiles dashboard** to expose non-linear slow zones.
74. **Kernel-level roofline stats** (flops utilization vs bandwidth bound) for attention/MLP.
75. **Online autotuning with safe rollback** and persistence of best settings per device.
76. **A/B switch framework** in config for each optimization (easy benchmarking and bisecting).
77. **Golden quality regression suite** (perplexity / task metrics) tied to each speed optimization.

## 11) Code-local opportunities in current tree

78. Extend the existing **system prompt KV cache** path in `models/causal_lm.cpp` to support multi-prefix and partial-prefix reuse.
79. Add runtime switches in `Applications/CausalLM/README.md` + model configs for separate prefill/decode thread settings.
80. Use the documented attention internals in `layers/mha_core_documentation.md` to introduce a prefill-specialized kernel path without affecting decode fast-path.
81. Revisit `cached_fc_layer` usage to ensure repeated prefill GEMMs avoid redundant weight layout transforms.
82. Introduce a prefill benchmarking mode in the CausalLM app that reports TTFT decomposition (tokenize / prefill / sample).

---

## Prioritized “first 10 to try” (practical)

1. Prefill/decode split execution plans.
2. Flash/tiled prefill attention kernel.
3. KV cache layout + write-once transposed format.
4. Prefix cache generalization beyond fixed system prompt.
5. GEMM prepack + fused SwiGLU projections.
6. Async MoE expert prefetch during prefill.
7. Token length bucketing + micro-batched prefill.
8. NUMA-aware thread/cache placement.
9. Per-layer prefill tracing to find real bottlenecks.
10. Autotuner for kernel/thread/chunk configs with persisted best profile.

These ten usually give the largest practical wins before advanced algorithmic approximations are attempted.
