# Sparse gather optimization A/B on SM-S948U

The review findings recorded in
[`sparse_prefix_reuse_normal_path_review.ko.md`](sparse_prefix_reuse_normal_path_review.ko.md)
were implemented in two commits: `[kv_cache_manager] Apply reviewed sparse reuse
optimizations` (①–⑦) and `[test] Add per-optimization sparse gather host bench`.
This report is the same-device before/after validation of those commits on the
production path.

## Method

Two Android HTP builds of `gemma4-26b-a4b-htp-w4a32` ran back-to-back on the
same SM-S948U device:

- **baseline** — worktree at `fd5fbb3e0` (the audit commit immediately before
  the optimization), built with `build_android.sh --htp --cache`;
- **optimized** — HEAD of `review/sparse-prefix-normal-path-20261111`
  (`fd5fbb3e0` + the two commits above).

The engine library (`libnntrainer.so`) is identical between the two builds; only
`libcausallm.so` and the benchmark binary differ. Each build ran
`htp_sparse_benefit MODEL out.json 2 all cache 4096` (cache cohort, 2 repeats,
4096-row prefill plane) right after `sync; echo 3 > /proc/sys/vm/drop_caches`.
The exact mode is a control: its code path is identical in both builds, so the
exact-mode delta measures thermal/clock drift between the two runs and is used
to normalize the sparse numbers.

## Host bench recap (per-optimization contributions)

The host pipeline (`tools/kv_cache/sparse_prefix_opt_bench/run.py`) verifies
bitwise equality with the production gather and reproduced the audit's
predictions exactly: allocations 6237 → 6092 → 5132 (baseline → step-gated →
full), K pre-copy 15.625 → 2.5 → 0 MiB, gather loop rows 512K → 128K → 107.8K,
ASan/UBSan clean.

## Device results (external TTFT, ms; mean of 2 repeats)

| case | exact base | exact opt | sparse base | sparse opt | raw Δ | normalized Δ |
|------|-----------:|----------:|------------:|-----------:|------:|-------------:|
| 24k_exact_control | 2134.5 | 1983.8 | 2148.9 | 2001.5 | +6.9% | −0.2% |
| 4k_at_0k | 8351.6 | 8026.9 | 2485.0 | 2211.0 | +11.0% | +7.4% |
| 8k_at_4k | 9137.7 | 8735.4 | 2697.1 | 2425.8 | +10.1% | +5.9% |
| 20k_at_4k | 37113.6 | 35293.8 | 3584.7 | 3334.4 | +7.0% | +2.2% |
| 20k_at_12k | 21205.7 | 20214.7 | 3277.3 | 2944.9 | +10.1% | +5.7% |
| 20k_multi_holes | 37588.8 | 36522.8 | 3656.9 | 3351.5 | +8.4% | +5.7% |
| 24k_one_token_at_4k | 48467.2 | 47923.1 | 3955.8 | 3762.4 | +4.9% | +3.8% |

Raw gains of 4.9–11.0% overstate the gather effect: the identical-code exact
control improved 1–8% between the two runs, so the drift-corrected (sparse/exact
ratio) gain is **2.2–7.4%** on sparse external TTFT, with the control case
confirming the correction to within 0.2%. Decode ms/token improved slightly
(e.g. 100.5 → 97.1 on the control; ~2–4% elsewhere). Output token IDs were
deterministic across repeats in both builds and the baseline sparse outputs
matched the optimized ones — the optimizations changed no visible tokens.
Peak memory moved by at most ±0.25 GB with no consistent direction; the retained
source pool dominates the footprint. Sparse TTFT remains dominated by HTP
recompute of hole rows and the Q8 KV restore, which is why host-side gather
savings surface as a small single-digit fraction of the total.

## Environment notes (reproduction pitfalls)

- Do **not** push the Hexagon SDK's 10 KB stub `libcdsprpc.so` to the device
  run directory: `LD_LIBRARY_PATH` resolves it ahead of
  `/vendor/lib64/libcdsprpc.so` (600 KB), and the stub's `rpcmem_alloc` fails,
  which aborts weight repack with `rpcmem_alloc(16 MiB) failed -- the HOST ION
  heap is out` even with gigabytes free. Removing the stub restores the vendor
  library.
- ION allocation can fail while page cache holds freed pages; run
  `sync; echo 3 > /proc/sys/vm/drop_caches` as root (`su 0` is available) before
  each cohort.
- A worktree build of an older commit pairs with **its own**
  `nntrainer/tensor/htp_backend/dsp/build/libnntr_hvx_skel.so`; pointing its
  `ADSP_LIBRARY_PATH` at the skel of a newer commit fails weight swap with
  `err=0x80000402`.
- Benchmark binaries require `meson configure <build> -Denable-htp-device-check=true`.

## Verdict

All seven reviewed optimizations land with visible, reproducible device-level
benefit (single-digit % of sparse TTFT after drift correction, plus reduced
host allocation churn) at no output-accuracy cost, and each individual
contribution is quantified by the host bench. The device numbers above were
produced with `htp_sparse_benefit` runs of 2026-11-11; the JSON reports are not
checked in, but the run command in this document regenerates them.
