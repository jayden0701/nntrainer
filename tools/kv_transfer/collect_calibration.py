#!/usr/bin/env python3
"""Capture paired exact-token CPU caches for an independent calibration corpus."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.calibration.read_text())["sequences"][:args.limit]
    manifest = {"version": 1, "pairs": []}
    for index, record in enumerate(records):
        ids = record["token_ids"]
        ids_path = out / f"input-{index:03d}.ids"
        ids_path.write_text("\n".join(map(str, ids)) + "\n")
        pair = {}
        for role, model_id in (("source", "Qwen/Qwen3-0.6B"), ("target", "Qwen/Qwen3-1.7B")):
            name = f"{role}-{index:03d}"
            pair[role] = name
            snapshot = out / name
            model = getattr(args, role).resolve()
            if (snapshot / "metadata.json").is_file():
                meta = json.loads((snapshot / "metadata.json").read_text())
                if meta["token_ids"] != ids or meta["model_id"] != model_id:
                    raise ValueError("Resume snapshot does not match requested input/model")
                continue
            config_path = out / f"config-{name}"
            config_path.mkdir(exist_ok=True)
            for item in model.iterdir():
                if item.is_file() and item.name != "nntr_config.json" and not (config_path / item.name).exists():
                    (config_path / item.name).symlink_to(item.resolve())
            config = json.loads((model / "nntr_config.json").read_text())
            config.update(init_seq_len=len(ids), max_seq_len=max(2048, len(ids) + 1), num_to_generate=1)
            config["system_prompt"] = {"head_prompt": "raw calibration prefix", "tail_prompt": "",
                                       "kvcache": {"pre_computed_cache_path": str(out / f"native-{name}")}}
            (config_path / "nntr_config.json").write_text(json.dumps(config, indent=2) + "\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("NNTR_SCORE_", "NNTR_KV_TRANSFER_", "NNTR_FORCED_"))}
            env.update(NNTR_ENGINE="cpu", NNTR_KV_CORE_CPU="1", NNTR_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
                       NNTR_KV_TRANSFER_EXPORT=str(snapshot), NNTR_KV_TRANSFER_MODEL_ID=model_id,
                       NNTR_KV_TRANSFER_ROPE_THETA="1000000", NNTR_KV_TRANSFER_EXPORT_INPUT_IDS=str(ids_path))
            start = time.monotonic()
            with (out / f"{name}.log").open("w") as log:
                subprocess.run([str(args.executable.resolve()), str(config_path), "ignored"], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            meta = json.loads((snapshot / "metadata.json").read_text())
            if meta["token_ids"] != ids:
                raise ValueError("Exporter changed calibration token sequence")
            result = {"index": index, "role": role, "tokens": len(ids), "seconds": time.monotonic() - start}
            with (out / "capture-times.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
        manifest["pairs"].append(pair)
        (out / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
