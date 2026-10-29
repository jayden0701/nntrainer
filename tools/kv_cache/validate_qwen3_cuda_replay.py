#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare core eager/ordered/replay Qwen logits, thresholds and restored turns.

Requires an explicitly prepared model and matching qwen_cpu_probe with optional
fixed-token requests. Does not download, build, or alter model weights.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from validate_qwen3_cuda import compare, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--requests",
        type=Path,
        required=True,
        help="passed G5 validation directory containing rendered requests",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    output = args.output.resolve()
    if (output / "manifest.json").exists():
        parser.error("use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    stage = output / "model"
    stage.mkdir()
    cfg = json.loads((model / "nntr_config.json").read_text())
    cfg.update(num_to_generate=43, max_seq_len=256)
    for asset in model.iterdir():
        if asset.name not in ("nntr_config.json", "generation_config.json"):
            (stage / asset.name).symlink_to(asset)
    save(stage / "nntr_config.json", cfg)
    save(
        stage / "generation_config.json",
        dict(
            json.loads((model / "generation_config.json").read_text()), do_sample=False
        ),
    )
    request = json.loads((args.requests / "arithmetic-request.json").read_text())
    forced = [32, 33, 34, 35]
    request.update(forced_ids=forced)
    request_file = output / "request.json"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NNTR_", "LD_", "VJEPA_"))
    }
    env.update(
        NNTR_ENGINE="cuda",
        NNTR_CUDA_KV_CORE="1",
        NNTR_CUDA_KV_DIRECT="1",
        NNTR_CUDA_KV_DEV="1",
        NNTR_CUDA_HOST_MAPPED="1",
        NNTR_CUDA_DEV_ACT="0",
        NNTR_CUDA_ATTN="1",
        NNTR_CUDA_ROPE="1",
        NNTR_CUDA_QKNORM="1",
        NNTR_CUDA_ELTWISE="1",
        NNTR_CUDA_PREFILL_GRAPH="0",
        NNTR_PREFILL_CHUNK="8",
        NNTR_KV_GPU_TRACE="1",
        NNTR_CUDA_GRAPH_DBG="1",
        NNTR_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        NNTR_TOKENIZER_CACHE="0",
    )
    root = Path(__file__).resolve().parents[2]
    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-m",
            "-o",
            "--exclude-standard",
            "nntrainer",
            "Applications",
            "test",
            "tools",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    manifest = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_files_sha256": {
            name: digest(root / name) for name in files if (root / name).is_file()
        },
        "probe_sha256": digest(args.probe.resolve()),
        "weights_sha256": digest(model / cfg["model_file_name"]),
        "environment": {k: v for k, v in env.items() if k.startswith("NNTR_")},
        "model_config": cfg,
        "runs": [],
        "comparisons": [],
        "scope": "44 fixed tokens; chunk8; threshold64; two live/restored turns",
    }
    linked = subprocess.check_output(["ldd", str(args.probe.resolve())], text=True)
    (output / "ldd.txt").write_text(linked)
    manifest["libraries_sha256"] = {}
    for line in linked.splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[1] == "=>" and fields[2].startswith("/"):
            path = Path(fields[2])
            if root in path.parents:
                manifest["libraries_sha256"][str(path)] = digest(path)
    reports = {}
    passed = True
    for profile in ("classic", "ordered", "replay"):
        run_dir = output / profile
        run_dir.mkdir()
        current = dict(
            env,
            NNTR_CUDA_KV_GRAPH="0" if profile == "classic" else "1",
            NNTR_CUDA_ASYNC="0" if profile == "classic" else "1",
            **{
                name: "1" if profile == "replay" else "0"
                for name in (
                    "NNTR_CUDA_GRAPH",
                    "NNTR_CUDA_M2B",
                    "NNTR_CUDA_DECODE_GRAPH",
                )
            },
        )
        this_request = dict(request)
        if profile != "classic":
            this_request.update(
                snapshot_resume=True,
                continuations=[
                    {"rendered_prompt": "\nContinue.\n", "forced_ids": forced},
                    {"rendered_prompt": "\nContinue again.\n", "forced_ids": forced},
                ],
            )
        save(request_file, this_request)
        report_file = run_dir / "report.json"
        with (
            (run_dir / "stdout.txt").open("w") as stdout,
            (run_dir / "stderr.txt").open("w") as stderr,
        ):
            result = subprocess.run(
                [
                    str(args.probe.resolve()),
                    str(stage),
                    str(request_file),
                    str(report_file),
                ],
                env=current,
                stdout=stdout,
                stderr=stderr,
                timeout=600,
            )
        row = {"profile": profile, "exit_code": result.returncode}
        manifest["runs"].append(row)
        save(output / "manifest.json", manifest)
        if result.returncode:
            raise SystemExit("Probe failed: " + profile)
        report = json.loads(report_file.read_text())
        reports[profile] = report
        row["initial_position"] = report["position_after"]
        row["turn_positions"] = [
            t["position_after"] for t in report.get("continuations", [])
        ]
        trace = (run_dir / "stderr.txt").read_text()
        rows = re.findall(
            r"append_copy_bytes=(\d+) captures=(\d+) replays=(\d+)", trace
        )
        row["trace"] = rows
        trace_pass = bool(rows) and all(int(r[0]) == 0 for r in rows)
        if profile == "replay":
            trace_pass &= any(int(r[1]) > 1 and int(r[2]) > 0 for r in rows)
            row["threshold_recapture"] = "retired at kv=65" in trace
            trace_pass &= row["threshold_recapture"]
        row["trace_pass"] = bool(trace_pass)
        passed &= bool(trace_pass)
        for index, (live, restored) in enumerate(
            zip(
                report.get("continuations", []), report.get("resumed_continuations", [])
            )
        ):
            item = compare(live, restored, np)
            item.update(profile=profile, turn=index + 1, against="restored")
            item["pass"] &= item["bitwise"] and (
                live["position_before"] == restored["position_before"]
                and live["position_after"] == restored["position_after"]
            )
            manifest["comparisons"].append(item)
            passed &= item["pass"]
        print(profile, row["initial_position"], bool(trace_pass), flush=True)
        save(output / "manifest.json", manifest)

    for name in ("classic", "ordered"):
        left, right = reports[name], reports["replay"]
        pairs = [(left, right)]
        if name == "ordered":
            pairs += list(zip(left["continuations"], right["continuations"]))
        for turn, (a, b) in enumerate(pairs):
            item = compare(a, b, np)
            item.update(profile=name, against="replay", turn=turn)
            if name == "ordered":
                item["pass"] &= item["bitwise"]
            else:
                # V slot projection uses FP32 FMA instead of cuBLAS reduction.
                item["pass"] &= all(
                    s["max_abs"] <= 0.15 and s["mean_abs"] <= 0.02
                    for s in item["steps"]
                )
            manifest["comparisons"].append(item)
            passed &= item["pass"]
    # Frozen G5 HF rows also cover actual greedy generation.
    cfg["num_to_generate"] = 7
    save(stage / "nntr_config.json", cfg)
    for prompt in ("arithmetic", "geography"):
        source_request = json.loads(
            (args.requests / (prompt + "-request.json")).read_text()
        )
        hf = json.loads((args.requests / (prompt + "-hf.json")).read_text())
        greedy_reports = {}
        for profile in ("ordered", "replay"):
            run_dir = output / (prompt + "-greedy-" + profile)
            run_dir.mkdir()
            save(request_file, source_request)
            current = dict(
                env,
                NNTR_CUDA_KV_GRAPH="1",
                NNTR_CUDA_ASYNC="1",
                **{
                    name: "1" if profile == "replay" else "0"
                    for name in (
                        "NNTR_CUDA_GRAPH",
                        "NNTR_CUDA_M2B",
                        "NNTR_CUDA_DECODE_GRAPH",
                    )
                },
            )
            report_file = run_dir / "report.json"
            with (
                (run_dir / "stdout.txt").open("w") as stdout,
                (run_dir / "stderr.txt").open("w") as stderr,
            ):
                result = subprocess.run(
                    [
                        str(args.probe.resolve()),
                        str(stage),
                        str(request_file),
                        str(report_file),
                    ],
                    env=current,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=300,
                )
            if result.returncode:
                raise SystemExit("Greedy probe failed: " + str(run_dir))
            report = json.loads(report_file.read_text())
            greedy_reports[profile] = report
            item = compare(hf, report, np)
            item.update(prompt=prompt, profile=profile, against="HF FP32 greedy")
            item["pass"] &= all(
                row["max_abs"] <= 0.15 and row["mean_abs"] <= 0.02
                for row in item["steps"]
            )
            manifest["comparisons"].append(item)
            passed &= item["pass"]
        item = compare(greedy_reports["ordered"], greedy_reports["replay"], np)
        item.update(prompt=prompt, against="ordered/replay greedy")
        item["pass"] &= item["bitwise"]
        manifest["comparisons"].append(item)
        passed &= item["pass"]
        print(prompt, "greedy", item["pass"], flush=True)

    manifest["snapshot_bytes_equal"] = digest(
        Path(reports["ordered"]["snapshot"]["path"])
    ) == digest(Path(reports["replay"]["snapshot"]["path"]))
    passed &= manifest["snapshot_bytes_equal"]
    manifest["pass"] = bool(passed)
    save(output / "manifest.json", manifest)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
