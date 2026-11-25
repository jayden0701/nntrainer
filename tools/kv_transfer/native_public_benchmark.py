#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run nntrainer's CPU KV manager on token streams from the public Ridge evaluation."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from benchmark import NativeScorer, file_digest, model_spec, pin_json, save_json, verify_prefix
from hf_public_benchmark import PUBLIC_REPORT_SHA256, json_digest, load_cases, measure, read_json
from mapper import Mapping, apply
from snapshot import Snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source", "target", "mapper", "data", "public-report",
                 "hf-cases", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--scratch", type=Path)
    args = p.parse_args()
    if (not 0 <= args.start < 1000 or not 1 <= args.limit <= 1000 - args.start
            or min(args.threads, args.batch) < 1):
        p.error("invalid case range, batch size or thread count")
    if file_digest(args.public_report) != PUBLIC_REPORT_SHA256:
        p.error("unexpected public report")
    cases = load_cases(args.data, read_json(args.public_report)["hellaswag"],
                       AutoTokenizer.from_pretrained(args.target, local_files_only=True))
    reference = read_json(args.hf_cases)
    if [c.index for c in cases] != [r["index"] for r in reference]:
        raise ValueError("different HF case order")
    for c, r in zip(cases, reference):
        if (r["context_tokens_sha256"] != json_digest(c.context)
                or r["ending_tokens_sha256"] != [json_digest(x) for x in c.endings]
                or r["imported_tokens"] != c.prefix_length or r["label"] != c.label):
            raise ValueError(f"HF input mismatch for {c.index}")
    cases = cases[args.start:args.start + args.limit]
    mapping = Mapping(args.mapper)
    def scorer(directory, spec):
        if spec != model_spec(directory, spec.model_id):
            raise ValueError(f"checkpoint geometry mismatch for {directory}")
        return NativeScorer(args.binary, directory, spec, args.threads, 64)
    source = scorer(args.source, mapping.geometry.source)
    target = scorer(args.target, mapping.geometry.target)
    args.out.mkdir(parents=True, exist_ok=True)
    files = [args.binary, args.binary.parent / "libcausallm.so", args.data,
             args.public_report, args.hf_cases,
             *(args.mapper / f for f in ("metadata.json", "weights.f32", "bias.f32"))]
    for directory in (args.source, args.target):
        config = read_json(directory / "nntr_config.json")
        files.extend(directory / f for f in ("config.json", "nntr_config.json",
                                              "tokenizer.json", config["model_file_name"]))
    pin_json(args.out / "inputs.json", {
        "files": {str(f): file_digest(f) for f in files},
        "case_ids": [c.index for c in cases], "threads": args.threads, "batch": args.batch,
        "runner": file_digest(Path(__file__)), "scorer": file_digest(Path(__file__).with_name("benchmark.py")),
    })
    checkpoint = args.out / "cases.json"
    rows = read_json(checkpoint) if checkpoint.exists() else []
    if [r["index"] for r in rows] != [c.index for c in cases[:len(rows)]]:
        raise ValueError("checkpoint case order mismatch")
    for start in range(len(rows), len(cases), args.batch):
        batch = cases[start:start + args.batch]
        with TemporaryDirectory(dir=args.scratch) as temporary:
            work = Path(temporary)
            jobs = []
            for c in batch:
                directory = work / str(c.index)
                directory.mkdir()
                snapshots = {"cold": None}
                if c.prefix_length:
                    prefix = c.context[:c.prefix_length]
                    source.export(directory, directory / "export.log", prefix, directory / "source")
                    exported = Snapshot(directory / "source")
                    verify_prefix(exported, prefix, source.spec)
                    apply(mapping, exported, directory / "mapped", block_size=128, memory_mib=1024)
                    snapshots["transfer"] = Snapshot(directory / "mapped")
                example = {"id": c.index, "context_ids": c.context,
                           "candidates": [{"ids": ids} for ids in c.endings]}
                jobs.append((example, directory, snapshots))
            log = args.out / f"score-{start}.log"
            scores, _ = target.score(work, log, jobs, persistent=True)
        for c in batch:
            result = scores[c.index]
            rows.append({"index": c.index, "label": c.label, "context_tokens": len(c.context),
                         "imported_tokens": c.prefix_length, "ending_tokens": list(map(len, c.endings)),
                         "standalone_logprobs": [x["sum_logprob"] for x in result["cold"]],
                         "mapped_logprobs": [x["sum_logprob"] for x in result.get("transfer", result["cold"])]})
        save_json(checkpoint, rows)
        print(f"completed {len(rows)}/{len(cases)}", flush=True)
    save_json(args.out / "summary.json", {"all": measure(rows),
                                         "imported": measure([r for r in rows if r["imported_tokens"]])})


if __name__ == "__main__":
    main()
