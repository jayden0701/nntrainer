#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact-token QuALITY handoffs and capture calibration histories."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark import NativeScorer, file_digest, model_spec, pin_json, verify_prefix
from snapshot import Snapshot, read_json


def prepare(dataset, tokenizer, limit, max_tokens):
    documents = {}
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError("QuALITY requires single-token answer letters")
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            document_id = row["article_id"]
            group = documents.setdefault(document_id, {"article": row["article"], "questions": {}})
            if group["article"] != row["article"]:
                raise ValueError(f"Inconsistent article text for {document_id}")
            for question in row["questions"]:
                if question.get("gold_label") in (1, 2, 3, 4):
                    group["questions"][question["question_unique_id"]] = question

    result = []
    ordered = sorted(documents, key=lambda key: hashlib.sha256(str(key).encode()).digest())
    for document_id in ordered:
        doc = documents[document_id]
        prefix = "Read the article and answer with the letter of the best option.\n\n" + doc["article"]
        history = tokenizer.encode(prefix, add_special_tokens=False)
        if len(history) >= max_tokens or len(doc["questions"]) < 4:
            continue
        queries = []
        for question_id, question in sorted(doc["questions"].items()):
            options = "\n".join(f"{key}. {option}" for key, option in zip("ABCD", question["options"]))
            query = f"\n\n{question['question']}\n{options}\nAnswer:"
            tokens = tokenizer.encode(query, add_special_tokens=False)
            if len(history) + len(tokens) > max_tokens:
                continue
            queries.append({"id": question_id, "token_ids": tokens,
                            "answer_ids": answer_ids, "gold": question["gold_label"] - 1})
            if len(queries) == 4:
                break
        if len(queries) == 4:
            result.append({"id": document_id, "history": history, "queries": queries})
        if len(result) == limit:
            break
    if len(result) != limit:
        raise ValueError(f"{dataset}: only {len(result)} documents fit the requested input limits")
    return result


def capture(manifest, binary, source, target, out, threads, role, start=0, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    documents = read_json(manifest)["documents"]
    selected = documents[start:None if limit is None else start + limit]
    if not selected:
        raise ValueError("No selected calibration documents")
    modes = [(name, path, tag) for name, path, tag in (
        ("source", source, "Qwen/Qwen3-0.6B"), ("target", target, "Qwen/Qwen3-1.7B"),
    ) if role in ("both", name)]
    for name, directory, tag in modes:
        scorer = NativeScorer(binary, directory, model_spec(directory, tag), threads)
        for index, doc in enumerate(selected, start):
            name_id = f"{name}-{index:03d}"
            destination = out / name_id
            if not destination.is_dir():
                scorer.export(out, out / f"{name_id}.log", doc["history"], destination)
            verify_prefix(Snapshot(destination), doc["history"], scorer.spec)
            print(f"{name_id}: {len(doc['history'])} history tokens", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("dataset", "tokenizer", "output"):
        prep.add_argument(name, type=Path)
    prep.add_argument("--documents", type=int, default=24)
    prep.add_argument("--max-tokens", type=int, default=2048)
    export = commands.add_parser("capture")
    for name in ("manifest", "binary", "source", "target", "output"):
        export.add_argument(name, type=Path)
    export.add_argument("--threads", type=int, default=4)
    export.add_argument("--start", type=int, default=0)
    export.add_argument("--limit", type=int)
    export.add_argument("--role", choices=("both", "source", "target"), default="both")
    args = parser.parse_args()
    if args.command == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        documents = prepare(args.dataset, tokenizer, args.documents, args.max_tokens)
        pin_json(args.output, {"dataset_sha256": file_digest(args.dataset), "documents": documents})
    else:
        if args.threads < 1 or args.start < 0 or (args.limit is not None and args.limit < 1):
            parser.error("Capture threads and document range must be positive")
        capture(args.manifest, args.binary, args.source, args.target,
                args.output.resolve(), args.threads, args.role, args.start, args.limit)


if __name__ == "__main__":
    main()
