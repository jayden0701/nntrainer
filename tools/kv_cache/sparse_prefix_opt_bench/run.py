#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and run the per-optimization sparse prefix gather benchmark.

The benchmark compiles the unmodified production gatherSparsePrefix from the
working tree as the bitwise reference (golden.cpp), then measures cumulative
optimization variants against it. Host-only: it neither changes the checkout
nor exercises HTP hardware. Requires a C++17 compiler.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    here = Path(__file__).resolve().parent
    output = args.output or here / "result.json"
    source = repo / "Applications/CausalLM/kv_cache_manager.cpp"
    full = source.read_text(encoding="utf-8")
    start = full.index("std::shared_ptr<const KVCacheManager::PrefixSnapshot>\n"
                       "KVCacheManager::gatherSparsePrefix(")
    end = full.index("\nvoid KVCacheManager::restorePrefix(", start)
    original = full[start:end]
    golden = here / "golden.cpp"
    golden.write_text(
        '#include <kv_cache_manager.h>\n#include <fp16.h>\n#include <limits>\n'
        'namespace causallm {\n' + original + '\n}\n', encoding="utf-8")
    includes = [here] + [repo / p for p in (
        "Applications/CausalLM", "nntrainer", "nntrainer/tensor",
        "nntrainer/utils", "nntrainer/kv_cache", "api", "api/ccapi/include",
        "api/capi/include")]
    includes += sorted({p.parent for p in (repo / "nntrainer").rglob("*.h")})
    flags = ["-std=c++17", "-O2", "-fno-fast-math", "-ffunction-sections",
             "-fdata-sections"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                  "-fno-omit-frame-pointer"]
    flags += [v for p in dict.fromkeys(includes) for v in ("-I", str(p))]
    sources = [here / "bench.cpp", golden,
               repo / "nntrainer/utils/fp16.cpp"]
    with tempfile.TemporaryDirectory(prefix="nntr-sparse-opt-bench-") as folder:
        build = Path(folder)

        def compile_one(item):
            index, path = item
            obj = build / f"{index}.o"
            subprocess.run([args.cxx, *flags, "-c", str(path), "-o", str(obj)],
                           check=True)
            return obj

        with ThreadPoolExecutor(max_workers=4) as pool:
            objects = list(pool.map(compile_one, enumerate(sources)))
        binary = build / "sparse-opt-bench"
        link_flags = ["-fsanitize=address,undefined"] if args.sanitize else []
        subprocess.run([args.cxx, *link_flags, "-Wl,--gc-sections",
                        *map(str, objects), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary), str(args.reps), str(output)],
                                text=True, capture_output=True)
        output.with_suffix(".stderr.txt").write_text(result.stderr,
                                                     encoding="utf-8")
        if result.returncode or (args.sanitize and result.stderr.strip()):
            raise RuntimeError(f"Bench failed ({result.returncode}):\n"
                               f"{result.stdout}\n{result.stderr}")
    report = json.loads(output.read_text(encoding="utf-8"))
    report["source_sha256"] = hashlib.sha256(full.encode()).hexdigest()
    report["function_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         text=True, capture_output=True)
    report["checkout_commit"] = (git.stdout.strip()
                                 if git.returncode == 0 else None)
    report["compiler"] = subprocess.check_output(
        [args.cxx, "--version"], text=True).splitlines()[0]
    report["sanitizers"] = args.sanitize
    report["repetitions"] = args.reps
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Per-optimization attribution: cumulative deltas against the previous
    variants = report["variants"]


    fields = ["payload_zero_bytes", "key_initial_copy_bytes", "visited_rows",
              "skipped_rows", "allocations", "gather_ms_median"]
    print(f"{'variant':<24} {'zeroinit MiB':>12} {'Kprecopy MiB':>13} "
          f"{'loop rows':>10} {'skipped':>9} {'allocs':>8} {'median ms':>10}")
    for v in variants:
        c = v["counters"]
        iters = c["visited_rows"] + c["skipped_rows"]
        print(f"{v['name']:<24} {c['payload_zero_bytes'] / 2**20:>12.3f} "
              f"{c['key_initial_copy_bytes'] / 2**20:>13.3f} "
              f"{iters:>10} {c['skipped_rows']:>9} {c['allocations']:>8} "
              f"{v['gather_ms_median']:>10.2f}")
    print("\nPer-optimization deltas (cumulative; do not sum):")
    for v in variants[1:]:
        prev = variants[variants.index(v) - 1]
        parts = []
        for f_ in fields:
            cur = (v["counters"][f_] if f_ != "gather_ms_median"
                   else v["gather_ms_median"])
            old = (prev["counters"][f_] if f_ != "gather_ms_median"
                   else prev["gather_ms_median"])
            if old:
                parts.append(f"{f_}: {old} -> {cur} ({100.0 * (cur - old) / old:+.1f}%)")
        print(f"  {v['name']}: " + "; ".join(parts))
    print(f"\nControls: {report['control_comparisons']} bitwise comparisons, "
          f"large fixture matches production: "
          f"{report['large_bitwise_matches_production']}")
    print(f"JSON written to {output}")


if __name__ == "__main__":
    main()
