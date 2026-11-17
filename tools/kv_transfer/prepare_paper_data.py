#!/usr/bin/env python3
"""Download pinned evaluation data and select independent calibration examples."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SOURCES = {
    "fineweb": ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "sample/350BT/016_00000.parquet"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge/test-00000-of-00001.parquet"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/validation-00000-of-00001.parquet"),
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    provenance = {}
    for label, (repo, revision, filename) in SOURCES.items():
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        rows = pq.read_table(path).to_pylist()
        provenance[label] = {"repository": repo, "revision": revision, "file": filename,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
        if label == "fineweb":
            order = list(range(len(rows)))
            random.Random(args.seed).shuffle(order)
            selected = []
            for index in order:
                ids = tokenizer.encode(rows[index]["text"], add_special_tokens=False).ids
                if len(ids) < 1024:
                    continue
                selected.append({"row_index": index, "document_id": rows[index].get("id"),
                                 "token_ids": ids[:1024]})
                if len(selected) == args.calibration_count:
                    break
            if len(selected) != args.calibration_count:
                raise ValueError("Not enough eligible calibration documents")
            payload = {"seed": args.seed, "tokens_per_sequence": 1024, "sequences": selected}
            (args.output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
        else:
            with (args.output / f"{label}.jsonl").open("w") as stream:
                for index, row in enumerate(rows):
                    row["_row_index"] = index
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(label, len(rows), "ready", flush=True)
    provenance["tokenizer_sha256"] = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    provenance["calibration_seed"] = args.seed
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
