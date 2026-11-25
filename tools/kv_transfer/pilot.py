#!/usr/bin/env python3
"""Collect native CPU KV pairs, fit transfer, and check a held-out continuation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dataset import prompt_digest
from snapshot import PLANES, Snapshot, new_directory, read_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("target_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--target-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("pilot_prompts.json"))
    p.add_argument("--layout", choices=["headwise", "all-head"], default="headwise")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--generation-tokens", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.threads < 1 or args.generation_tokens < 0:
        p.error("threads must be positive and generation tokens must be nonnegative")
    return args


def configure(model, location, prefix, generate, cache=None, tokens=None):
    location.mkdir()
    config = read_json(model / "nntr_config.json")
    generation = read_json(model / "generation_config.json")
    for item in model.iterdir():
        if item.is_file() and item.name not in {"nntr_config.json", "generation_config.json"}:
            (location / item.name).symlink_to(item.resolve())
    config["num_to_generate"] = generate
    generation["do_sample"] = False
    config["system_prompt"] = {"head_prompt": prefix, "tail_prompt": ""}
    if cache is not None:
        config["system_prompt"]["kvcache"] = {"pre_computed_cache_path": str(cache)}
        if tokens is not None:
            config["system_prompt"]["kvcache"]["sys_prompt_token_size"] = tokens
    write_json(location / "nntr_config.json", config)
    write_json(location / "generation_config.json", generation)


def native(args, model_dir, model_id, config, continuation, log, **extra):
    theta = read_json(model_dir / "config.json")["rope_theta"]
    env = {**os.environ, "NNTR_ENGINE": "cpu", "NNTR_KV_CORE_CPU": "1",
           "NNTR_NUM_THREADS": str(args.threads), "NNTR_KV_TRANSFER_MODEL_ID": model_id,
           "NNTR_KV_TRANSFER_ROPE_THETA": str(theta)}
    for name in ("NNTR_KV_TRANSFER_IMPORT", "NNTR_KV_TRANSFER_EXPORT",
                 "NNTR_GENERATED_IDS", "NNTR_FORCED_DECODE_INPUTS"):
        env.pop(name, None)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([str(args.executable.resolve()), str(config), continuation],
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 env={**env, **extra})
    elapsed = time.monotonic() - start
    print(f"{log.name}: exit={process.returncode} seconds={elapsed:.3f}", flush=True)
    if process.returncode:
        raise RuntimeError(f"native inference failed; see {log}")
    return round(elapsed, 3)


def mapper(*args):
    command = Path(__file__).with_name("kv_transfer.py")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": os.environ.get("KV_PILOT_BLAS_THREADS", "1")}
    start = time.monotonic()
    output = subprocess.check_output([sys.executable, str(command), *map(str, args)],
                                     text=True, env=env)
    return json.loads(output), round(time.monotonic() - start, 3)


def collect(args, prompts, out):
    manifest = {"version": 1, "pairs": []}
    captures = []
    for index, prompt in enumerate(prompts):
        row = {}
        model_runs = {}
        for label in ("source", "target"):
            model_dir = getattr(args, f"{label}_dir")
            model_id = getattr(args, f"{label}_id")
            name = f"{label}-{index:02d}"
            snapshot = out / name
            cache = out / f"native-{name}"
            config = out / f"config-{name}"
            configure(model_dir, config, prompt["prefix"], 1, cache)
            elapsed = native(args, model_dir, model_id, config, "", out / f"{name}.log",
                             NNTR_KV_TRANSFER_EXPORT=str(snapshot))
            row[label] = name
            model_runs[label] = {"seconds": elapsed, "tokens": Snapshot(snapshot).tokens}
        if "group" in prompt:
            row["group"] = prompt["group"]
        manifest["pairs"].append(row)
        captures.append({"prompt": prompt["id"], **row, "capture": model_runs})
    dataset = out / "dataset.json"
    write_json(dataset, manifest)
    return dataset, captures


def direct_copy(source, target, destination):
    """Create the explicitly labeled source-plane transplant control."""
    shape = source.spec.shape(source.tokens)
    if (shape != target.spec.shape(target.tokens)
            or source.token_ids != target.token_ids or source.positions != target.positions):
        return None
    with new_directory(destination) as staging:
        for plane in PLANES.values():
            shutil.copyfile(source.path / plane, staging / plane)
        metadata = read_json(target.path / "metadata.json")
        metadata["baseline"] = {"method": "direct_copy", "source_model": source.spec.model_id}
        write_json(staging / "metadata.json", metadata)
    return destination


def continue_target(args, out, prompt, native_snapshot, predictions, baseline):
    cache = out / ("native-" + native_snapshot.path.name)
    reused = out / "continuation-import"
    cold = out / "continuation-cold"
    configure(args.target_dir, reused, prompt["prefix"], args.generation_tokens,
              cache, native_snapshot.tokens)
    configure(args.target_dir, cold, prompt["prefix"], args.generation_tokens)
    result = {}
    runs = [("cold", cold, None), ("exact", reused, native_snapshot.path),
            ("mapped", reused, predictions)]
    if baseline is not None:
        runs.append(("direct-copy", reused, baseline))
    for label, config, snapshot in runs:
        ids = out / f"continuation-{label}.ids"
        env = {"NNTR_GENERATED_IDS": str(ids)}
        if snapshot is not None:
            env["NNTR_KV_TRANSFER_IMPORT"] = str(snapshot)
        seconds = native(args, args.target_dir, args.target_id, config,
                         prompt["continuation"], out / f"continuation-{label}.log", **env)
        result[label] = {"seconds": seconds, "ids": [int(id) for id in ids.read_text().split()]}
    return result


def main():
    args = parse_args()
    prompts = read_json(args.prompts)["prompts"]
    if len(prompts) < 2:
        raise ValueError("pilot needs at least two prompts")
    out = args.output.resolve()
    out.mkdir(parents=True)
    dataset, captures = collect(args, prompts, out)
    mapping = out / "mapping"
    fit, fit_seconds = mapper("fit", dataset, mapping, "--layout", args.layout,
                              "--k", args.k, "--alpha", args.alpha, "--seed", args.seed)
    validation = set(fit["validation"]["prompt_digests"])
    result = {"fit": fit, "fit_seconds": fit_seconds, "threads": args.threads,
              "generation_tokens": args.generation_tokens, "prompts": captures}
    first = True
    for prompt, record in zip(prompts, captures):
        source = Snapshot(out / record["source"])
        digest = prompt_digest(source.token_ids)
        record["digest"] = digest
        record["split"] = "validation" if digest in validation else "train"
        if digest not in validation:
            continue
        prediction = out / ("predicted-" + record["source"])
        _, record["apply_seconds"] = mapper("apply", mapping, source.path, prediction)
        reference = Snapshot(out / record["target"])
        record["cache_metrics"], _ = mapper("compare", prediction, reference.path)
        baseline = direct_copy(source, reference, out / ("direct-copy-" + record["source"]))
        if baseline is not None:
            record["direct_copy_metrics"], _ = mapper("compare", baseline, reference.path)
        if first and args.generation_tokens:
            continuation = continue_target(args, out, prompt, reference, prediction, baseline)
            continuation["cold_exact_equal"] = continuation["cold"]["ids"] == continuation["exact"]["ids"]
            record["continuation"] = continuation
            first = False
    write_json(out / "result.json", result)
    print(out / "result.json")
    for record in captures:
        if "continuation" in record and not record["continuation"]["cold_exact_equal"]:
            raise RuntimeError("native target import changed generated IDs; see result.json")


if __name__ == "__main__":
    main()
