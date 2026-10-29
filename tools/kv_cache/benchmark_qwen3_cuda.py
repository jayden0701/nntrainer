#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time fixed CUDA Qwen workloads using matching legacy/candidate observers.

No downloads or builds. Correctness is validated separately. Trace is disabled;
each profile runs in a fresh process, in ABBA order. Replay mode compares the
same observer and slot arithmetic with graph execution disabled/enabled.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time

from validate_qwen3_cuda import digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument(
    "--baseline-probe",
    type=Path,
    help="legacy observer; replay mode uses candidate for both arms",
)
parser.add_argument("--candidate-probe", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--nvidia-smi", default="/usr/lib/wsl/lib/nvidia-smi")
parser.add_argument("--mode", choices=("legacy", "replay"), default="legacy")
parser.add_argument("--contexts", type=int, nargs="+", default=(1024, 4096))
parser.add_argument("--decode-tokens", type=int)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--repetitions", type=int, default=3)
args = parser.parse_args()
if args.mode == "legacy" and args.baseline_probe is None:
    parser.error("--baseline-probe is required for legacy mode")
decode_tokens = args.decode_tokens or (32 if args.mode == "replay" else 8)
if decode_tokens < 4 or args.warmup < 1 or args.repetitions < 1:
    parser.error("use decode>=4, warmup>=1, repetitions>=1")
if any(context < 1 or context + decode_tokens >= 8192 for context in args.contexts):
    parser.error("contexts plus decode must fit below 8192")
profiles = ("ordered", "replay") if args.mode == "replay" else ("baseline", "direct")
root = Path(__file__).resolve().parents[2]
base = (args.baseline_probe or args.candidate_probe).resolve()
candidate = args.candidate_probe.resolve()
model = args.model.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
if (out / "summary.json").exists():
    parser.error("use a new output directory to preserve previous results")

stage = out / "model"
stage.mkdir(exist_ok=True)
for asset in model.iterdir():
    if asset.name in ("nntr_config.json", "generation_config.json"):
        continue
    link = stage / asset.name
    if not link.exists():
        link.symlink_to(asset)
cfg = json.loads((model / "nntr_config.json").read_text())
cfg.update(max_seq_len=8192, init_seq_len=128)
(stage / "nntr_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
(stage / "generation_config.json").write_bytes(
    (model / "generation_config.json").read_bytes()
)
env = {
    k: v for k, v in os.environ.items() if not k.startswith(("NNTR_", "VJEPA_", "LD_"))
}
env.update(
    NNTR_ENGINE="cuda",
    NNTR_CUDA_KV_DEV="1",
    NNTR_CUDA_HOST_MAPPED="1",
    NNTR_CUDA_DEV_ACT="0",
    NNTR_CUDA_ATTN="1",
    NNTR_CUDA_ROPE="1",
    NNTR_CUDA_QKNORM="1",
    NNTR_CUDA_ELTWISE="1",
    NNTR_CUDA_ASYNC="0",
    NNTR_CUDA_DECODE_GRAPH="0",
    NNTR_CUDA_GRAPH="0",
    NNTR_CUDA_M2B="0",
    NNTR_CUDA_PREFILL_GRAPH="0",
    NNTR_PREFILL_CHUNK="128",
    NNTR_INIT_SEQ_LEN="128",
    NNTR_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="4",
    NNTR_TOKENIZER_CACHE="0",
)


def gpu():
    result = subprocess.run(
        [
            args.nvidia_smi,
            "--query-gpu=timestamp,name,uuid,temperature.gpu,clocks.current.sm,pstate,memory.used,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def save(value):
    (out / "summary.json").write_text(json.dumps(value, indent=2) + "\n")


manifest = {
    "environment": {k: v for k, v in env.items() if k.startswith("NNTR_")},
    "model_config": cfg,
    "source_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "source_diff_sha256": hashlib.sha256(
        subprocess.check_output(["git", "diff"], cwd=root)
    ).hexdigest(),
    "scope": "dense FP16; mapped activations; device KV; chunk128",
    "mode": args.mode,
    "profiles": profiles,
    "environment_by_profile": {},
    "runs": [],
    "comparison": [],
}
for name, probe in (("baseline", base), ("candidate", candidate)):
    libraries = {}
    linked = subprocess.check_output(["ldd", str(probe)], text=True)
    (out / (name + "-ldd.txt")).write_text(linked)
    for line in linked.splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[1] == "=>" and fields[2].startswith("/"):
            path = Path(fields[2])
            if root in path.parents:
                libraries[str(path)] = digest(path)
    manifest[name] = {
        "path": str(probe),
        "sha256": digest(probe),
        "libraries": libraries,
    }
if args.mode == "replay":
    env.update(
        NNTR_CUDA_KV_CORE="1",
        NNTR_CUDA_KV_GRAPH="1",
        NNTR_CUDA_KV_DIRECT="1",
        NNTR_CUDA_ASYNC="1",
    )
manifest["environment"] = {k: v for k, v in env.items() if k.startswith("NNTR_")}
manifest["weights_sha256"] = digest(model / cfg["model_file_name"])
manifest["runner_sha256"] = digest(Path(__file__))
for context in args.contexts:
    request = {
        "context_tokens": context,
        "decode_tokens": decode_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "expected_plane": 128,
        "expected_prefill_chunk": 128,
    }
    request_file = out / f"request-{context}.json"
    request_file.write_text(json.dumps(request, indent=2) + "\n")
    for index, profile in enumerate(
        (profiles[0], profiles[1], profiles[1], profiles[0])
    ):
        prefix = out / f"{context}-{index}-{profile}"
        run = {
            "context": context,
            "order": index,
            "profile": profile,
            "gpu_before": gpu(),
        }
        current = dict(
            env,
            NNTR_CUDA_KV_CORE="0" if profile == "baseline" else "1",
            NNTR_CUDA_KV_DIRECT="1",
        )
        if args.mode == "replay":
            on = "1" if profile == "replay" else "0"
            current.update(
                NNTR_CUDA_KV_CORE="1",
                NNTR_CUDA_KV_GRAPH="1",
                NNTR_CUDA_GRAPH=on,
                NNTR_CUDA_M2B=on,
                NNTR_CUDA_DECODE_GRAPH=on,
            )
        manifest["environment_by_profile"][profile] = {
            k: v for k, v in current.items() if k.startswith("NNTR_")
        }
        request["require_replay"] = profile == "replay"
        request["require_direct"] = args.mode == "replay"
        request_file.write_text(json.dumps(request, indent=2) + "\n")
        probe = (
            candidate
            if args.mode == "replay"
            else (base if profile == "baseline" else candidate)
        )
        start = time.monotonic()
        with (
            prefix.with_suffix(".stdout.txt").open("w") as stdout,
            prefix.with_suffix(".stderr.txt").open("w") as stderr,
        ):
            result = subprocess.run(
                [
                    str(probe),
                    str(stage),
                    str(request_file),
                    str(prefix.with_suffix(".json")),
                ],
                env=current,
                stdout=stdout,
                stderr=stderr,
                timeout=1800,
            )
        run.update(
            exit_code=result.returncode,
            elapsed_seconds=time.monotonic() - start,
            gpu_after=gpu(),
        )
        manifest["runs"].append(run)
        save(manifest)
        if result.returncode:
            raise SystemExit("Benchmark failed: " + str(prefix))
        report = json.loads(prefix.with_suffix(".json").read_text())
        run["report"] = report
        save(manifest)
        print(
            context,
            index,
            profile,
            [round(t["run_wall_ms"], 2) for t in report["trials"] if not t["warmup"]],
            flush=True,
        )
    rows = [r for r in manifest["runs"] if r["context"] == context]
    comparison = {"context": context, "metrics": {}}
    for metric in (
        "run_wall_ms",
        "prefill_compute_ms",
        "decode_api_ms",
        "decode_steady_median_ms",
    ):
        samples = {
            profile: [
                t[metric]
                for run in rows
                if run["profile"] == profile
                for t in run["report"]["trials"]
                if not t["warmup"]
            ]
            for profile in profiles
        }
        baseline = statistics.median(samples[profiles[0]])
        direct = statistics.median(samples[profiles[1]])
        comparison["metrics"][metric] = {
            profiles[0] + "_median": baseline,
            profiles[1] + "_median": direct,
            "change_percent": 100 * (direct / baseline - 1),
            profiles[0] + "_p95": sorted(samples[profiles[0]])[
                math.ceil(0.95 * len(samples[profiles[0]])) - 1
            ],
            profiles[1] + "_p95": sorted(samples[profiles[1]])[
                math.ceil(0.95 * len(samples[profiles[1]])) - 1
            ],
            "samples_per_profile": len(samples[profiles[0]]),
        }
    comparison["steady_inter_token_ms"] = {}
    for profile in profiles:
        intervals = [
            value
            for run in rows
            if run["profile"] == profile
            for trial in run["report"]["trials"]
            if not trial["warmup"]
            for value in trial["decode_logits_intervals_ms"][2:]
        ]
        comparison["steady_inter_token_ms"][profile] = {
            "median": statistics.median(intervals),
            "p95": sorted(intervals)[math.ceil(0.95 * len(intervals)) - 1],
            "samples": len(intervals),
            "excludes": "first two decode intervals of each measured trial",
        }
    comparison["budget_5_percent_pass"] = all(
        value["change_percent"] <= 5
        for name, value in comparison["metrics"].items()
        if name in ("run_wall_ms", "prefill_compute_ms", "decode_api_ms")
    )
    manifest["comparison"].append(comparison)
    save(manifest)
manifest["pass"] = all(row["budget_5_percent_pass"] for row in manifest["comparison"])
save(manifest)
print(json.dumps(manifest["comparison"], indent=2), flush=True)

if not manifest["pass"]:
    raise SystemExit(1)
