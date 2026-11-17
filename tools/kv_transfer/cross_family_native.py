"""Run the fixed C2C-style held-out sample through nntrainer's CPU KV manager."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .cross_family.fuse import CacheFuser, SOURCE, RECEIVER, read_cache, write_cache
from .cross_family.train import Aligner, wiki_examples, hellaswag_examples
from .snapshot import Snapshot, new_directory, write_json

# Existing benchmark helpers support both script and module invocation this way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import NativeScorer


def run(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1:
        raise ValueError("unsupported checkpoint version")
    fuser = CacheFuser(checkpoint["width"])
    fuser.load_state_dict(checkpoint["state"])
    fuser.eval()
    source_tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    receiver_tok = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    aligner = Aligner(source_tok, receiver_tok)
    source = NativeScorer(args.executable, args.source, SOURCE, 4)
    receiver = NativeScorer(args.executable, args.receiver, RECEIVER, 4)
    examples = [(task, e) for task, iterator in (
        ("wikitext2", wiki_examples(args.data, receiver_tok, args.count)),
        ("hellaswag", hellaswag_examples(args.data, receiver_tok, args.count)),
    ) for e in iterator]
    hf = json.loads(args.hf_results.read_text())
    with new_directory(args.output.resolve()) as output:
        jobs = []
        timings = Counter()
        for index, (task, e) in enumerate(examples):
            name = f"{task}-{e.row}"
            work = output / name
            work.mkdir()
            sids = aligner(e.prefix)
            timings["source_export_including_startup"] += source.export(work, work / "source.log", sids, work / "source")
            timings["receiver_export_including_startup"] += receiver.export(work, work / "receiver.log", e.prefix, work / "receiver")
            start = time.perf_counter()
            src, dst = Snapshot(work / "source"), Snapshot(work / "receiver")
            with torch.no_grad():
                fused = fuser(read_cache(src, SOURCE), read_cache(dst, RECEIVER))
                write_cache(work / "fused", fused, RECEIVER, dst.token_ids, dst.positions)
            timings["fusion_with_snapshot_io"] += time.perf_counter() - start
            modes = {"cold": None, "fused": Snapshot(work / "fused")}
            if index < 3 or args.count <= index < args.count + 3:
                modes["exact"] = dst
            example = {"id": name, "context_ids": e.prefix + [e.anchor], "candidates": [{"ids": ids} for ids in e.candidates]}
            jobs.append((example, work, modes))
        scores, seconds = receiver.score(output, output / "scores.log", jobs, persistent=True)
        timings["all_scores_including_startup"] = seconds
        rows = []
        for task, e in examples:
            name = f"{task}-{e.row}"
            modes = scores[name]
            nll = {mode: [-np.mean(score["token_logprobs"]) for score in candidates] for mode, candidates in modes.items()}
            reference = next(row for row in hf["examples"][task] if row["row"] == e.row)
            rows.append({"task": task, "row": e.row, "gold": e.gold, "nll": nll,
                         "hf_nll": reference["nll"], "native_token_logprobs": modes})
            if "exact" in modes:
                for cold, exact in zip(modes["cold"], modes["exact"]):
                    if cold["token_logprobs"] != exact["token_logprobs"]:
                        raise AssertionError(f"exact cache failed: {name}")
        write_json(output / "results.json", {"rows": rows, "timings_seconds": dict(timings),
                   "count_per_task": args.count, "exact_controls": 6,
                   "scope": "FP32 weights/activation, FP16 physical KV, native source and receiver prefill; learned fusion offline"})
        print(json.dumps(dict(timings)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("executable", "source", "receiver", "checkpoint", "data", "hf-results", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count < 3:
        parser.error("need at least 3 cases per task for the exact-cache controls")
    run(args)
