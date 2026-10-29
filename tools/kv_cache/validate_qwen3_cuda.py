#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate pinned Qwen3 CUDA KV against HF and a matching legacy runtime.

Model preparation is explicit and separate. No downloads or system changes are
performed. Each profile runs in a fresh process with capture disabled.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

PROMPTS = {
    "arithmetic": "Compute 2 + 2. Reply with the number only.",
    "geography": "What is the capital of France? Reply with one word.",
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compare(left, right, np):
    if len(left["steps"]) != len(right["steps"]):
        return {"pass": False, "reason": "step count differs"}
    metrics = []
    for a, b in zip(left["steps"], right["steps"]):
        x = np.fromfile(a["logits_file"], dtype=np.float32)
        y = np.fromfile(b["logits_file"], dtype=np.float32)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            return {"pass": False, "reason": "shape or finite check failed"}
        delta = x.astype(np.float64) - y.astype(np.float64)
        metrics.append(
            {
                "max_abs": float(np.abs(delta).max()),
                "mean_abs": float(np.abs(delta).mean()),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "bitwise": x.tobytes() == y.tobytes(),
            }
        )
    return {
        "pass": left["token_ids"] == right["token_ids"],
        "tokens_equal": left["token_ids"] == right["token_ids"],
        "steps": metrics,
        "bitwise": all(row["bitwise"] for row in metrics),
    }


def validate_sessions(args, output, runtime_model, env, np):
    """Compare live and restored sessions using the same observer and runtime."""
    request = json.loads((output / "arithmetic-request.json").read_text())
    request.update(
        snapshot_resume=True,
        continuations=[
            {"rendered_prompt": "\nCompute 3 + 3. Reply with the number only.\n"},
            {"rendered_prompt": "\nCompute 4 + 4. Reply with the number only.\n"},
        ],
    )
    request_file = output / "session-request.json"
    save(request_file, request)
    result = {"runs": [], "comparisons": []}
    reports = {}
    passed = True
    for profile, probe in (
        ("baseline", args.baseline_probe),
        ("direct", args.candidate_probe),
    ):
        stage = output / ("session-" + profile)
        stage.mkdir(exist_ok=True)
        report_file = stage / "report.json"
        current = dict(
            env,
            NNTR_PREFILL_CHUNK="8",
            NNTR_CUDA_KV_CORE="0" if profile == "baseline" else "1",
            NNTR_CUDA_KV_DIRECT="1",
        )
        with (
            (stage / "stdout.txt").open("w") as stdout,
            (stage / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.run(
                [
                    str(probe.resolve()),
                    str(runtime_model),
                    str(request_file),
                    str(report_file),
                ],
                env=current,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout,
            )
        run = {"profile": profile, "exit_code": process.returncode}
        result["runs"].append(run)
        if process.returncode:
            passed = False
            continue
        report = json.loads(report_file.read_text())
        reports[profile] = report
        snapshot = report["snapshot"]
        run["snapshot"] = dict(snapshot, sha256=digest(Path(snapshot["path"])))
        run["position_pass"] = (
            snapshot["saved_position"]
            == snapshot["restored_position"]
            == report["position_after"]
        )
        passed &= run["position_pass"]
        expected_position = snapshot["saved_position"]
        turns = report["continuations"]
        restored = report["resumed_continuations"]
        if len(turns) != len(request["continuations"]) or len(restored) != len(turns):
            passed = False
        for index, (live, resumed) in enumerate(zip(turns, restored)):
            comparison = compare(live, resumed, np)
            comparison.update(profile=profile, turn=index + 1, against="restored")
            comparison["position_pass"] = (
                live["position_before"]
                == resumed["position_before"]
                == expected_position
                and live["position_after"] == resumed["position_after"]
                and live["position_after"] > expected_position
            )
            comparison["pass"] &= comparison["bitwise"] and comparison["position_pass"]
            expected_position = live["position_after"]
            result["comparisons"].append(comparison)
            passed &= comparison["pass"]
    if len(reports) == 2:
        baseline, direct = reports["baseline"], reports["direct"]
        for index, (left, right) in enumerate(
            zip(
                [baseline] + baseline["continuations"],
                [direct] + direct["continuations"],
            )
        ):
            comparison = compare(left, right, np)
            comparison.update(turn=index, against="legacy CUDA same arithmetic")
            comparison["position_pass"] = (
                left["position_before"] == right["position_before"]
                and left["position_after"] == right["position_after"]
            )
            comparison["pass"] &= comparison["bitwise"] and comparison["position_pass"]
            result["comparisons"].append(comparison)
            passed &= comparison["pass"]
        result["snapshot_bytes_equal"] = (
            result["runs"][0]["snapshot"]["sha256"]
            == result["runs"][1]["snapshot"]["sha256"]
        )
        passed &= result["snapshot_bytes_equal"]
    result["pass"] = bool(passed)
    save(output / "sessions.json", result)
    print("multi-turn and snapshot", bool(passed), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-probe", type=Path, required=True)
    parser.add_argument("--candidate-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--max-abs", type=float, default=0.15)
    parser.add_argument("--mean-abs", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--snapshot-resume",
        action="store_true",
        help="also compare two live/restored continuation turns",
    )
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        parser.error("use a new output directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    nntr = json.loads((model / "nntr_config.json").read_text())
    if args.tokens < 2 or nntr.get("skip_prefill", False):
        parser.error("use at least two total tokens and skip_prefill=false")
    # run() emits a prefill token before its num_to_generate decode iterations.
    effective = dict(nntr, num_to_generate=args.tokens - 1)
    runtime_model = output / "runtime-model"
    runtime_model.mkdir(exist_ok=True)
    for asset in model.iterdir():
        if asset.name in ("nntr_config.json", "generation_config.json"):
            continue
        link = runtime_model / asset.name
        if not link.exists():
            link.symlink_to(asset)
    save(runtime_model / "nntr_config.json", effective)
    generation = json.loads((model / "generation_config.json").read_text())
    generation["do_sample"] = False
    save(runtime_model / "generation_config.json", generation)
    if nntr["model_tensor_type"] != "FP16-FP16":
        parser.error("this validation profile requires FP16 activations/weights")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NNTR_", "VJEPA_", "LD_"))
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
        NNTR_KV_GPU_TRACE="1",
        NNTR_TOKENIZER_CACHE="0",
        NNTR_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="4",
    )
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "model_config": nntr,
        "effective_model_config": effective,
        "weights_sha256": digest(model / nntr["model_file_name"]),
        "environment": {k: v for k, v in env.items() if k.startswith("NNTR_")},
        "reference": "HF eager FP32 weights/activations/native FP32 KV",
        "limits": {"max_abs": args.max_abs, "mean_abs": args.mean_abs},
        "torch": torch.__version__,
        "runs": [],
        "comparisons": [],
    }
    for name, probe in (
        ("baseline", args.baseline_probe),
        ("candidate", args.candidate_probe),
    ):
        probe = probe.resolve()
        manifest[name] = {"path": str(probe), "sha256": digest(probe)}
        abi = probe.with_suffix(".abi.json")
        if abi.exists():
            manifest[name]["abi"] = json.loads(abi.read_text())
        linked = subprocess.check_output(["ldd", str(probe)], text=True)
        (output / (name + "-ldd.txt")).write_text(linked)
        libraries = {}
        for line in linked.splitlines():
            fields = line.split()
            if len(fields) > 2 and fields[1] == "=>" and fields[2].startswith("/"):
                path = Path(fields[2])
                if root in path.parents:
                    libraries[str(path)] = digest(path)
        manifest[name]["libraries"] = libraries

    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=root, text=True
    ).splitlines()
    changed += subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "nntrainer",
            "Applications",
            "test",
            "tools",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    manifest["source_files_sha256"] = {
        name: digest(root / name)
        for name in sorted(set(changed))
        if (root / name).is_file()
    }

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(
        args.hf, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    passed = True
    for name, prompt in PROMPTS.items():
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        request = {
            "rendered_prompt": rendered,
            "input_ids": inputs.input_ids[0].tolist(),
        }
        request_file = output / (name + "-request.json")
        save(request_file, request)
        with torch.inference_mode():
            generated = reference.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        hf = {
            "token_ids": generated.sequences[0, inputs.input_ids.shape[-1] :].tolist(),
            "steps": [],
        }
        for index, scores in enumerate(generated.scores):
            path = output / f"{name}-hf.step{index}.f32"
            scores[0].float().numpy().tofile(path)
            hf["steps"].append({"logits_file": str(path)})
        save(output / (name + "-hf.json"), hf)
        reports = {}
        for chunk in (0, 8):
            for profile in ("baseline", "copy", "direct"):
                stage = output / f"{name}-{profile}-chunk{chunk}"
                stage.mkdir(exist_ok=True)
                probe = (
                    args.baseline_probe
                    if profile == "baseline"
                    else args.candidate_probe
                ).resolve()
                current = dict(
                    env,
                    NNTR_PREFILL_CHUNK=str(chunk),
                    NNTR_CUDA_KV_CORE="0" if profile == "baseline" else "1",
                    NNTR_CUDA_KV_DIRECT="1" if profile == "direct" else "0",
                )
                report_file = stage / "report.json"
                start = time.monotonic()
                with (
                    (stage / "stdout.txt").open("w") as stdout,
                    (stage / "stderr.txt").open("w") as stderr,
                ):
                    result = subprocess.run(
                        [
                            str(probe),
                            str(runtime_model),
                            str(request_file),
                            str(report_file),
                        ],
                        env=current,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=args.timeout,
                    )
                run = {
                    "prompt": name,
                    "profile": profile,
                    "chunk": chunk,
                    "exit_code": result.returncode,
                    "elapsed_with_load_seconds": time.monotonic() - start,
                }
                manifest["runs"].append(run)
                if result.returncode:
                    passed = False
                    save(output / "manifest.json", manifest)
                    continue
                report = json.loads(report_file.read_text())
                reports[(profile, chunk)] = report
                run["token_ids"] = report["token_ids"]
                if profile != "baseline":
                    trace = (stage / "stderr.txt").read_text()
                    rows = re.findall(
                        r"\[kv-gpu-step\].*device_reads=(\d+) "
                        r"direct_v=(\d+) append_copy_bytes=(\d+)",
                        trace,
                    )
                    run["kv_trace"] = rows
                    trace_pass = bool(rows) and int(rows[-1][0]) > 0
                    if profile == "direct":
                        trace_pass &= int(rows[-1][1]) > 0 and all(
                            int(row[2]) == 0 for row in rows
                        )
                    else:
                        trace_pass &= int(rows[-1][2]) > 0
                    run["trace_pass"] = bool(trace_pass)
                    passed &= bool(trace_pass)
                comparison = compare(hf, report, np)
                comparison.update(
                    prompt=name, profile=profile, chunk=chunk, against="HF FP32"
                )
                close = comparison["pass"] and all(
                    row["max_abs"] <= args.max_abs and row["mean_abs"] <= args.mean_abs
                    for row in comparison.get("steps", [])
                )
                comparison["pass"] = bool(close)
                manifest["comparisons"].append(comparison)
                passed &= bool(close)
                print(name, profile, chunk, "HF", bool(close), flush=True)
                save(output / "manifest.json", manifest)
        for chunk in (0, 8):
            baseline = reports.get(("baseline", chunk))
            for profile in ("copy", "direct"):
                candidate = reports.get((profile, chunk))
                if not baseline or not candidate:
                    continue
                comparison = compare(baseline, candidate, np)
                comparison.update(
                    prompt=name,
                    profile=profile,
                    chunk=chunk,
                    against="legacy CUDA same arithmetic",
                )
                comparison["pass"] &= comparison["bitwise"]
                manifest["comparisons"].append(comparison)
                passed &= comparison["pass"]
        if ("direct", 0) in reports and ("direct", 8) in reports:
            comparison = compare(reports[("direct", 0)], reports[("direct", 8)], np)
            comparison.update(prompt=name, against="direct chunk 0 vs 8")
            comparison["pass"] &= all(
                row["max_abs"] <= args.max_abs and row["mean_abs"] <= args.mean_abs
                for row in comparison.get("steps", [])
            )
            manifest["comparisons"].append(comparison)
            passed &= comparison["pass"]
    if args.snapshot_resume:
        manifest["sessions"] = validate_sessions(args, output, runtime_model, env, np)
        passed &= manifest["sessions"]["pass"]
    manifest["pass"] = bool(passed)
    save(output / "manifest.json", manifest)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
