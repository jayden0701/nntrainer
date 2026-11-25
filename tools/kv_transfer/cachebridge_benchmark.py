#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare resident CPU KV handoffs using the native CausalLM CLI."""

import argparse
import json
import os
import statistics
from itertools import groupby
import subprocess
from pathlib import Path

from benchmark import file_digest, pin_json, save_json
from snapshot import read_json


def hellaswag(args):
    from transformers import AutoTokenizer

    from hf_public_benchmark import PUBLIC_REPORT_SHA256, json_digest, load_cases

    if not all((args.public_report, args.hf_cases, args.data)):
        raise ValueError("HellaSwag needs the dataset, public report and reference token hashes")
    if file_digest(args.public_report) != PUBLIC_REPORT_SHA256:
        raise ValueError("Unexpected public Ridge report")
    raw = read_json(args.public_report)["hellaswag"]
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    cases = load_cases(args.data, raw, tokenizer)
    with args.hf_cases.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    if len(cases) != len(reference):
        raise ValueError("Unexpected public case count")
    for case, check in zip(cases, reference):
        if (case.index != check["index"] or case.label != check["label"]
                or case.prefix_length != check["imported_tokens"]
                or json_digest(case.context) != check["context_tokens_sha256"]
                or [json_digest(ids) for ids in case.endings] != check["ending_tokens_sha256"]):
            raise ValueError(f"Public token or handoff mismatch for case {case.index}")
        yield {"id": case.index, "history": case.context[:case.prefix_length],
               "query": case.context[case.prefix_length:], "options": case.endings,
               "gold": case.label}


def quality(args):
    if not args.data or not args.calibration:
        raise ValueError("QuALITY evaluation needs evaluation and calibration manifests")
    documents = read_json(args.data)["documents"]
    fit_ids = {row["id"] for row in read_json(args.calibration)["documents"]}
    if fit_ids & {row["id"] for row in documents}:
        raise ValueError("QuALITY evaluation overlaps calibration documents")
    for doc in documents:
        for query in doc["queries"]:
            yield {"id": query["id"], "history": doc["history"], "query": query["token_ids"],
                   "options": query["answer_ids"], "gold": query["gold"],
                   "document_id": doc["id"]}


def latency(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    if not args.data:
        raise ValueError("Latency benchmark requires a QuALITY manifest")
    documents = read_json(args.data)["documents"]
    query = tokenizer.encode("\n\nSummarize the main idea in one sentence.\nAnswer:",
                             add_special_tokens=False)
    for length in (512, 1024, 2048, 4096):
        history = max((doc["history"] for doc in documents), key=len)
        # Repeat the long document to reach the requested context on the same token stream.
        history = (history * ((length + len(history) - 1) // len(history)))[:length]
        for repeat in range(args.repeats + 1):
            yield {"id": f"{length}-{repeat}", "history": history, "query": query,
                   "warmup": repeat == 0}


def summarize(cases, modes):
    summary = {}
    for mode in modes:
        scored = [row for row in cases if "gold" in row and mode in row]
        hits = sum(max(range(len(row[mode]["scores"])), key=row[mode]["scores"].__getitem__)
                   == row["gold"] for row in scored)
        entries = [row[mode] for row in cases if mode in row and not row.get("warmup")]
        lengths = {}
        for length in sorted({row["history_tokens"] for row in cases if "gold" not in row}):
            subset = [row[mode] for row in cases
                      if row["history_tokens"] == length and not row.get("warmup") and mode in row]
            if subset:
                fields = ("ttft_ms", "target_query_ms", "cache_prepare_ms",
                          "gather_rope_ms", "affine_ms", "install_ms", "mapper_scratch_bytes")
                lengths[length] = {field: statistics.median(row[field] for row in subset)
                                   for field in fields}
        report = {"cases": len(entries), "fallbacks": sum("fallback" in row for row in entries)}
        if scored:
            report.update(correct=hits, scored=len(scored), accuracy=hits / len(scored))
            transferred = [row for row in scored if row["history_tokens"]]
            report["eligible"] = len(transferred)
        if lengths:
            report["median_by_history_tokens"] = lengths
        summary[mode] = report
    return summary


def run(args):
    requests = list({"hellaswag": hellaswag, "quality": quality, "latency": latency}[args.task](args))
    requests = requests[args.start:None if args.limit is None else args.start + args.limit]
    if not requests:
        raise ValueError("No selected cases")
    mappers = {name: str(Path(path).resolve()) for name, path in
               (entry.split("=", 1) for entry in args.mapper)}
    if args.skip_cold and not mappers and not args.exact:
        raise ValueError("The benchmark needs at least one mode")
    modes = [*([] if args.skip_cold else ["cold"]), *(["exact"] if args.exact else []), *mappers]
    if args.reuse_history and args.task != "quality":
        raise ValueError("History reuse is available for QuALITY scoring")
    for i, request in enumerate(requests):
        order = i % len(modes)
        request["modes"] = modes[order:] + modes[:order] if args.task == "latency" else modes
        request["reuse_history"] = args.reuse_history

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = [args.binary, args.binary.parent / "libcausallm.so",
              Path(__file__), *(path for path in (args.data, args.calibration,
                                                args.public_report, args.hf_cases) if path)]
    for directory in (args.source, args.target):
        config = directory / "nntr_config.json"
        inputs.extend([config, directory / "config.json", directory / "tokenizer.json",
                       directory / read_json(config)["model_file_name"]])
    for path in mappers.values():
        inputs.extend(Path(path) / name for name in ("metadata.json", "weights.f32", "bias.f32"))
    pin_json(output / "inputs.json", {
        "files": {str(path.resolve()): file_digest(path) for path in inputs},
        "task": args.task, "case_ids": [request["id"] for request in requests], "modes": modes,
        "threads": args.threads, "block_tokens": args.block_tokens,
        "reuse_history": args.reuse_history,
    })
    results = output / "cases.jsonl"
    with results.open("a+", encoding="utf-8") as stream:
        stream.seek(0)
        rows = [json.loads(line) for line in stream]
        if [row["id"] for row in rows] != [req["id"] for req in requests[:len(rows)]]:
            raise ValueError("Native checkpoint case order mismatch")
        remainder = requests[len(rows):]
        if remainder:
            native_requests = remainder
            if args.reuse_history:
                # Complete all queries in a mode before overwriting target KV.
                # Reuse the same committed source for every mode of a document.
                native_requests = []
                for _, items in groupby(remainder, key=lambda req: req["document_id"]):
                    document = list(items)
                    native_requests.extend({**req, "modes": [mode]}
                                           for mode in modes for req in document)
            manifest = output / "requests.json"
            spec = {"source_model": str(args.source.resolve()), "source_id": "Qwen/Qwen3-0.6B",
                    "target_id": "Qwen/Qwen3-1.7B", "block_tokens": args.block_tokens,
                    "mappers": mappers, "requests": native_requests}
            save_json(manifest, spec)
            env = {**os.environ, **{key: str(args.threads) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NNTR_NUM_THREADS")},
                   "NNTR_ENGINE": "cpu"}
            command = [str(args.binary.resolve()), str(args.target.resolve()),
                       "--cachebridge", str(manifest)]
            with (output / "native.log").open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace") as process:
                    pending = {}
                    processed = 0
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if not line.startswith("[CACHEBRIDGE] "):
                            continue
                        row = json.loads(line.removeprefix("[CACHEBRIDGE] "))
                        expected = native_requests[processed]
                        processed += 1
                        if row["id"] != expected["id"] or any(mode not in row for mode in expected["modes"]):
                            raise ValueError("Unexpected result from native benchmark")
                        merged = pending.setdefault(row["id"], row)
                        merged.update({mode: row[mode] for mode in expected["modes"]})
                        while len(rows) < len(requests):
                            request = requests[len(rows)]
                            candidate = pending.get(request["id"], {})
                            if any(mode not in candidate for mode in modes):
                                break
                            row = pending.pop(request["id"])
                            row.update({key: request[key] for key in ("gold", "warmup", "document_id")
                                        if key in request})
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            rows.append(row)
                            if len(rows) % 10 == 0 or len(rows) == len(requests):
                                print(f"Completed {len(rows)}/{len(requests)}", flush=True)
                    if process.wait() or len(rows) != len(requests):
                        raise RuntimeError(f"Native benchmark incomplete. See {output / 'native.log'}")
    report = summarize(rows, modes)
    save_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("hellaswag", "quality", "latency"))
    for name in ("binary", "source", "target", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("data", "public-report", "hf-cases", "calibration"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mapper", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--reuse-history", action="store_true")
    args = parser.parse_args()
    if (min(args.threads, args.block_tokens, args.repeats) < 1 or args.start < 0
            or (args.limit is not None and args.limit < 1)):
        parser.error("Threads, blocks, repeats and case range must be positive")
    run(args)


if __name__ == "__main__":
    main()
