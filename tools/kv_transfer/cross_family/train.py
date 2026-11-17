"""Train and evaluate a small, cross-family KV fuser on an x86 CPU."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import transformers

from ..snapshot import new_directory, write_json
from .fuse import CacheFuser, SOURCE, RECEIVER, write_cache


SEED = 4217
PREFIX = 95
TARGETS = 16
WIDTH = 32
SOURCE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
RECEIVER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


@dataclass(frozen=True)
class Example:
    row: int
    prefix: list
    anchor: int
    candidates: list
    gold: int = 0


def json_lines(path):
    with path.open(encoding="utf-8") as file:
        yield from map(json.loads, file)


def sample_rank(task, row):
    return sha256(f"{SEED}:{task}:{row}".encode()).digest()


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def window(row, ids):
    return Example(row, ids[:PREFIX], ids[PREFIX], [ids[PREFIX + 1:PREFIX + 1 + TARGETS]])


def fineweb_examples(root, source_tok, receiver_tok):
    data = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    examples = []
    seen = set()
    for row in data["sequences"]:
        text = source_tok.decode(row["token_ids"][:2 * (PREFIX + TARGETS)],
                                 clean_up_tokenization_spaces=False)
        ids = encode(receiver_tok, text)
        key = tuple(ids[:PREFIX + 1 + TARGETS])
        if len(ids) <= PREFIX + TARGETS or key in seen:
            continue
        seen.add(key)
        examples.append(window(row["row_index"], ids))
    if len(examples) < 40:
        raise ValueError("need 40 distinct FineWeb calibration rows")
    return examples[:32], examples[32:40]


def wiki_examples(root, tokenizer, count):
    rows = sorted(json_lines(root / "wikitext.jsonl"),
                  key=lambda row: sample_rank("wikitext2", row["_row_index"]))
    seen = set()
    for row in rows:
        ids = encode(tokenizer, row["text"])
        key = tuple(ids[:PREFIX + 1 + TARGETS])
        if len(ids) > PREFIX + TARGETS and key not in seen:
            seen.add(key)
            yield window(row["_row_index"], ids)
            if len(seen) == count:
                return
    raise ValueError("too few eligible WikiText-2 rows")


def hellaswag_examples(root, tokenizer, count):
    def preprocess(text):
        text = text.strip().replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", text).replace("  ", " ").strip()

    rows = sorted(json_lines(root / "hellaswag.jsonl"),
                  key=lambda row: sample_rank("hellaswag", row["_row_index"]))
    selected = 0
    for row in rows:
        context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        context = preprocess(context)
        ids = encode(tokenizer, context)
        if not 2 <= len(ids) <= PREFIX + 1:
            continue
        candidates = []
        for ending in row["endings"]:
            joined = encode(tokenizer, f"{context} {preprocess(ending)}")
            if joined[:len(ids)] != ids or not 1 <= len(joined) - len(ids) <= 64:
                break
            candidates.append(joined[len(ids):])
        if len(candidates) != len(row["endings"]):
            continue
        yield Example(row["_row_index"], ids[:-1], ids[-1], candidates, int(row["label"]))
        selected += 1
        if selected == count:
            return
    raise ValueError("too few eligible HellaSwag rows")


class Aligner:
    """One source ID per receiver ID, following the C2C longest-string rule."""

    def __init__(self, source, receiver):
        self.source = source
        self.receiver = receiver
        self.table = {}
        self.counts = Counter()

    def token(self, idx):
        if idx not in self.table:
            text = self.receiver.decode([idx], clean_up_tokenization_spaces=False)
            if idx in self.receiver.all_special_ids:
                kind = "special"
                choice = self.source.get_added_vocab().get(text)
                if choice is None:
                    choice = self.source.unk_token_id or self.source.eos_token_id
            else:
                candidates = encode(self.source, text)
                kind = ("empty" if not candidates else
                        "multiple" if len(candidates) > 1 else "single")
                choice = (max(candidates, key=lambda token: len(self.source.decode(
                    [token], clean_up_tokenization_spaces=False
                ))) if candidates else self.source.unk_token_id or self.source.eos_token_id)
            self.table[idx] = choice, kind
        choice, kind = self.table[idx]
        self.counts[kind] += 1
        return choice

    def __call__(self, ids):
        return [self.token(idx) for idx in ids]


def load_model(directory, kind, spec):
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa"
    ).requires_grad_(False).eval()
    cfg = model.config
    actual = (cfg.model_type, cfg.num_hidden_layers, cfg.num_key_value_heads,
              cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    expected = (kind, spec.layers, spec.kv_heads, spec.head_dim, spec.theta, None)
    if actual != expected:
        raise ValueError(f"unexpected geometry at {directory}: {actual} != {expected}")
    return model


@torch.no_grad()
def prefill(model, ids, stats, key):
    start = perf_counter()
    cache = model.model(input_ids=torch.tensor(ids), use_cache=True).past_key_values
    stats[key] += perf_counter() - start
    return cache.to_legacy_cache()


def suffix_logits(model, receiver, inputs):
    return model(
        input_ids=torch.tensor(inputs), past_key_values=DynamicCache(receiver),
        use_cache=True
    ).logits


def token_loss(logits, labels):
    return F.cross_entropy(logits.transpose(1, 2), torch.tensor(labels), reduction="none")


def score_nll(logits, labels):
    # Evaluate in float64 because vocabulary-wide FP32 reductions on the
    # strided tensor can obscure the small paired NLL differences here.
    scores = logits.to(torch.float64)
    targets = torch.tensor(labels).unsqueeze(-1)
    return scores.logsumexp(dim=-1) - scores.gather(-1, targets).squeeze(-1)


def corresponding_inputs(example, candidate):
    return [example.anchor, *candidate[:-1]]


def group(examples, batch_size):
    for index in range(0, len(examples), batch_size):
        yield examples[index:index + batch_size]


def make_batch(examples, aligner, source, receiver, stats):
    receiver_ids = [example.prefix for example in examples]
    start = perf_counter()
    source_ids = [aligner(ids) for ids in receiver_ids]
    stats["alignment"] += perf_counter() - start
    s = prefill(source, source_ids, stats, "source_prefill")
    t = prefill(receiver, receiver_ids, stats, "receiver_prefill")
    return s, t, source_ids


def prepare_batches(examples, batch_size, aligner, source, receiver, stats):
    batches = []
    for batch in group(examples, batch_size):
        s, t, _ = make_batch(batch, aligner, source, receiver, stats)
        labels = [example.candidates[0] for example in batch]
        inputs = [corresponding_inputs(example, y) for example, y in zip(batch, labels)]
        batches.append((s, t, inputs, labels))
    return batches


@torch.no_grad()
def validation_loss(fuser, receiver, batches):
    total = sum(
        token_loss(suffix_logits(receiver, fuser(s, t), x), y).sum().item()
        for s, t, x, y in batches
    )
    return total / sum(len(x) * len(x[0]) for _, _, x, _ in batches)


def fit(fuser, receiver, train, valid, epochs, output):
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(fuser.parameters(), lr=3e-4, weight_decay=0.01)
    history = []
    reference = validation_loss(fuser, receiver, valid)
    best = math.inf
    best_epoch = None
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        total = tokens = 0
        for idx in rng.permutation(len(train)):
            s, t, x, y = train[idx]
            opt.zero_grad(set_to_none=True)
            losses = token_loss(suffix_logits(receiver, fuser(s, t), x), y)
            loss = losses.mean()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(fuser.parameters(), 1.0)
            if not torch.isfinite(gradient) or gradient.item() == 0:
                raise ValueError("expected a finite, nonzero adapter gradient")
            opt.step()
            total += losses.sum().item()
            tokens += losses.numel()
        score = validation_loss(fuser, receiver, valid)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"version": 1, "width": WIDTH, "state": fuser.state_dict()},
                       output / "fuser.pt")
        record = {"epoch": epoch, "train_nll": total / tokens, "val_nll": score}
        history.append(record)
        print(json.dumps(record), flush=True)
    fuser.load_state_dict(torch.load(output / "fuser.pt", weights_only=True)["state"])
    return {"initial_val_nll": reference, "epochs": history,
            "selected_epoch": best_epoch, "training_seconds": perf_counter() - start}


def zero_cache(cache):
    return tuple(tuple(torch.zeros_like(plane) for plane in layer) for layer in cache)


def relative_delta(native, fused):
    numerator = sum((x - y).square().sum().item()
                    for a, b in zip(native, fused) for x, y in zip(a, b))
    denominator = sum(x.square().sum().item() for a in native for x in a)
    return math.sqrt(numerator / denominator)


def timed_call(fn, stats, key, *args):
    start = perf_counter()
    result = fn(*args)
    stats[key] += perf_counter() - start
    return result


def write_bridge(output, example, source_ids, source, native, fused, logits):
    labels = example.candidates[0]
    case = {
        "row": example.row, "receiver_prefix_ids": example.prefix,
        "source_prefix_ids": source_ids[0],
        "continuation_input_ids": corresponding_inputs(example, labels),
        "labels": labels,
        "cache_relative_l2": relative_delta(native, fused),
    }
    bridge_dir = output / "bridge"
    bridge_dir.mkdir()
    for name, cache, spec, ids in (
        ("source_hf", source, SOURCE, case["source_prefix_ids"]),
        ("receiver_hf", native, RECEIVER, case["receiver_prefix_ids"]),
        ("fused_hf", fused, RECEIVER, case["receiver_prefix_ids"]),
    ):
        write_cache(bridge_dir / name, cache, spec, ids, range(len(ids)))
    for mode, predictions in logits.items():
        case[mode] = {
            "loss": score_nll(predictions, [labels])[0].tolist(),
            "argmax": predictions[0].argmax(dim=-1).tolist(),
        }
        predictions[0].numpy().astype("<f4").tofile(bridge_dir / f"{mode}_logits.f32")
    write_json(bridge_dir / "case.json", case)


@torch.no_grad()
def evaluate(fuser, source, receiver, aligner, wiki, hellaswag, output):
    stats = Counter()
    result = {"wikitext2": [], "hellaswag": []}
    for task, examples in (("wikitext2", wiki), ("hellaswag", hellaswag)):
        # The WikiText windows have equal length. Rotate to a disjoint row
        # for a source control with the same position and nonzero features.
        wrong = None
        if task == "wikitext2":
            wrong = prefill(source, [aligner(examples[-1].prefix)], Counter(), "unused")
        for index, example in enumerate(examples):
            s, t, source_ids = make_batch([example], aligner, source, receiver, stats)
            caches = {
                "native": t,
                "fused": timed_call(fuser, stats, "fusion", s, t),
                "zero_source": fuser(zero_cache(s), t),
            }
            if wrong is not None:
                caches["wrong_source"] = fuser(wrong, t)
                wrong = s
            record = {"row": example.row, "gold": example.gold, "nll": {},
                      "cache_relative_l2": relative_delta(t, caches["fused"])}
            bridge_logits = {}
            for mode, cache in caches.items():
                losses = []
                for candidate in example.candidates:
                    x = corresponding_inputs(example, candidate)
                    logits = timed_call(suffix_logits, stats, f"{mode}_suffix",
                                        receiver, cache, [x])
                    losses.append(score_nll(logits, [candidate]).mean().item())
                    if index == 0 and task == "wikitext2" and mode in ("native", "fused"):
                        bridge_logits[mode] = logits
                        if mode == "native":
                            full = receiver(input_ids=torch.tensor([example.prefix + x]),
                                            use_cache=False, logits_to_keep=len(x)).logits
                            diff = (full - logits).abs().max().item()
                            if diff > 3e-4:
                                raise ValueError(f"split receiver differs from whole pass: {diff}")
                            record["full_split_logit_max_abs"] = diff
                record["nll"][mode] = losses
            if index == 0 and task == "wikitext2":
                write_bridge(output, example, source_ids, s, t, caches["fused"], bridge_logits)
            result[task].append(record)
    return result, dict(stats)


def summarize(results):
    reports = {}
    for task, rows in results.items():
        modes = rows[0]["nll"]
        table = {}
        native = np.array([row["nll"]["native"][row["gold"]] for row in rows])
        indices = np.random.default_rng(SEED).integers(0, len(rows), size=(10000, len(rows)))
        for mode in modes:
            losses = np.array([row["nll"][mode][row["gold"]] for row in rows])
            diff = losses - native
            table[mode] = {
                "gold_mean_nll": float(losses.mean()),
                "difference_from_native": float(diff.mean()),
                "difference_ci95": np.quantile(diff[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
            if task == "hellaswag":
                table[mode]["correct"] = sum(
                    int(np.argmin(row["nll"][mode])) == row["gold"] for row in rows
                )
        reports[task] = {"rows": len(rows), "modes": table,
                         "mean_cache_relative_l2": float(np.mean(
                             [row["cache_relative_l2"] for row in rows]))}
    return reports


def digest(path):
    checksum = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=32, help="rows per held-out task")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.test_count < 2 or not 1 <= args.threads <= 4:
        parser.error("need positive epochs, at least two test rows, and 1–4 threads")

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source_tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    receiver_tok = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    train, valid = fineweb_examples(args.data, source_tok, receiver_tok)
    wiki = list(wiki_examples(args.data, receiver_tok, args.test_count))
    hella = list(hellaswag_examples(args.data, receiver_tok, args.test_count))
    aligner = Aligner(source_tok, receiver_tok)
    with new_directory(args.output) as output:
        source = load_model(args.source, "qwen3", SOURCE)
        receiver = load_model(args.receiver, "llama", RECEIVER)
        fuser = CacheFuser(WIDTH)
        prefill_seconds = Counter()
        train_batches = prepare_batches(train, 4, aligner, source, receiver, prefill_seconds)
        val_batches = prepare_batches(valid, 4, aligner, source, receiver, prefill_seconds)
        s, t, _, _ = train_batches[0]
        with torch.no_grad():
            if relative_delta(t, fuser(s, t)) != 0:
                raise ValueError("zero-initialized residual must preserve receiver cache exactly")
        print(f"Calibration cache timings: {dict(prefill_seconds)}", flush=True)
        training = fit(fuser, receiver, train_batches, val_batches, args.epochs, output)
        del train_batches, val_batches, s, t
        results, eval_seconds = evaluate(fuser, source, receiver, aligner, wiki, hella, output)
        report = {
            "settings": {"arch": platform.machine(), "torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "threads": args.threads, "seed": SEED, "width": WIDTH,
                         "parameters": sum(p.numel() for p in fuser.parameters()),
                         "batch_size": 4, "cached_context": PREFIX,
                         "suffix_targets": TARGETS,
                         "aligner_counts": dict(aligner.counts)},
            "models": {
                spec.model_id: {"revision": revision, "sha256": {
                    name: digest(directory / name)
                    for name in ("config.json", "tokenizer.json", "model.safetensors")
                }}
                for spec, directory, revision in (
                    (SOURCE, args.source, SOURCE_REVISION),
                    (RECEIVER, args.receiver, RECEIVER_REVISION),
                )
            },
            "data": {name: digest(args.data / name) for name in (
                "calibration.json", "wikitext.jsonl", "hellaswag.jsonl", "provenance.json"
            )},
            "train_rows": [example.row for example in train],
            "val_rows": [example.row for example in valid],
            "prefill_seconds": dict(prefill_seconds), "training": training,
            "eval_seconds": eval_seconds, "summary": summarize(results), "examples": results,
        }
        write_json(output / "results.json", report)
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
