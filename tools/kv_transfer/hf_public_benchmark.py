#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score public n200 HellaSwag indices with local Transformers Qwen3 models.

The scoring and mapper conventions follow alesha-pro/vllm-cross-model-kv
at 2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc. This runner uses BF16 CPU eager
attention, local checkpoints, and an in-memory Transformers cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PUBLIC_COMMIT = "2e27fb4bcadcbe07136a79480d50ef7d81a2e1dc"
PUBLIC_REPORT_SHA256 = "ff92075191e985d8940c26639fdec514227775419e0c7b82a2af7c82eb958ae6"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Case:
    index: int
    label: int
    context: list[int]
    endings: list[list[int]]

    @property
    def prefix_length(self) -> int:
        return (len(self.context) - 1) // BLOCK_SIZE * BLOCK_SIZE

    def tokens_digest(self) -> str:
        return json_digest({"context": self.context, "endings": self.endings})


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_digest(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", text)).strip()


def load_cases(path: Path, public: dict, tokenizer) -> list[Case]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if any(row["_row_index"] != index for index, row in enumerate(rows)):
        raise ValueError("HellaSwag JSONL is not in validation row order")
    shuffled = list(range(len(rows)))
    random.Random(0).shuffle(shuffled)
    indices = public["indices"]
    if len(indices) != 1000 or indices != shuffled[:1000]:
        raise ValueError("Validation order differs from the public seed-0 index list")

    cases = []
    for index in indices:
        row = rows[index]
        query = clean(
            f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        )
        context = tokenizer.encode(query, add_special_tokens=False)
        endings = [
            tokenizer.encode(" " + clean(ending), add_special_tokens=False)
            for ending in row["endings"]
        ]
        label = int(row["label"])
        if not context or len(endings) != 4 or not all(endings) or not 0 <= label < 4:
            raise ValueError(f"Invalid tokenized HellaSwag case {index}")
        cases.append(Case(index, label, context, endings))

    short_prompts = [
        case.index for case in cases if not case.prefix_length for _ in case.endings
    ]
    if short_prompts != public["short_prompts"]:
        raise ValueError("Tokenized short prompts differ from the public report")
    return cases


def model_identity(path: Path, manifest_identity: dict) -> dict:
    hashes = {}
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        digest = sha256(path / filename)
        field = {"config.json": "config_sha256", "tokenizer.json": "tokenizer_json_sha256", "tokenizer_config.json": "tokenizer_config_sha256"}[filename]
        if digest != manifest_identity[field]:
            raise ValueError(f"Checkpoint file hash mismatch for {path / filename}")
        hashes[filename] = digest
    for filename in ("nntr_source_revision.json", "download_revision.json"):
        stamp = path / filename
        if stamp.is_file() and read_json(stamp)["revision"] == manifest_identity["revision"]:
            break
    else:
        raise ValueError(f"Checkpoint revision stamp mismatch for {path}")
    return {
        "model_id": manifest_identity["model_id"],
        "revision": manifest_identity["revision"],
        "files_sha256": hashes,
    }


def load_model(path: Path):
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=DTYPE, attn_implementation="eager"
    ).eval()


def pad_tokens(
    sequences: list[list[int]], *, left: bool, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        if sequence:
            start = width - len(sequence) if left else 0
            end = start + len(sequence)
            ids[row, start:end] = torch.tensor(sequence)
            mask[row, start:end] = 1
    return ids, mask


def forward(model, ids, new_mask, *, past=None, old_mask=None):
    mask = new_mask if old_mask is None else torch.cat((old_mask, new_mask), dim=1)
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)[:, -ids.shape[1] :]
    result = model.model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=past,
        use_cache=True,
    )
    return result.last_hidden_state, result.past_key_values, mask


def half_rope_factors(positions: torch.Tensor, theta: float, dim: int):
    inverse_frequency = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)
    angles = positions.float().unsqueeze(-1) * inverse_frequency
    return angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)


def rotate(keys: torch.Tensor, factors, *, inverse: bool) -> torch.Tensor:
    first, second = keys.float().chunk(2, dim=-1)
    cosine, sine = factors
    if inverse:
        sine = -sine
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    ).to(keys.dtype)


class RidgeMapper:
    def __init__(self, manifest: dict, weights: Path):
        if manifest["kind"] != "ridge" or manifest["rope_mode"] != "stripped":
            raise ValueError("Expected a content-space ridge mapper")
        self.source = manifest["source_model"]
        self.target = manifest["target_model"]
        self.target_shape = (self.target["num_kv_heads"], self.target["head_dim"])
        with safe_open(weights, framework="pt", device="cpu") as tensors:
            self.picks = tensors.get_tensor("source_layers").tolist()
            self.parameters = {
                name: tensors.get_tensor(name).to(DTYPE)
                for name in ("W_K", "b_K", "W_V", "b_V")
            }
        target_layers = self.target["num_layers"]
        out_features = math.prod(self.target_shape)
        in_features = (
            manifest["k"] * self.source["num_kv_heads"] * self.source["head_dim"]
        )
        if len(self.picks) != target_layers or any(
            len(picks) != manifest["k"]
            or len(set(picks)) != len(picks)
            or any(not 0 <= layer < self.source["num_layers"] for layer in picks)
            for picks in self.picks
        ):
            raise ValueError("Invalid source-layer selection")
        for name, tensor in self.parameters.items():
            shape = (target_layers, out_features)
            if name.startswith("W"):
                shape = (target_layers, in_features, out_features)
            if tensor.shape != shape:
                raise ValueError(f"Invalid mapper shape for {name}: {tensor.shape}")

    def apply(self, cache: DynamicCache, mask: torch.Tensor) -> DynamicCache:
        positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
        source_factors = half_rope_factors(
            positions, self.source["rope_theta"], self.source["head_dim"]
        )
        target_factors = half_rope_factors(
            positions, self.target["rope_theta"], self.target["head_dim"]
        )
        keys, values = [], []
        for key, value in cache.to_legacy_cache():
            # HF cache is [batch, head, token, dim]. Mapper uses token-major rows.
            content_key = rotate(key.transpose(1, 2), source_factors, inverse=True)
            keys.append(content_key.flatten(0, 1).flatten(1))
            values.append(value.transpose(1, 2).flatten(0, 1).flatten(1))

        mapped = []
        for layer, picks in enumerate(self.picks):
            outputs = []
            for name, source in (("K", keys), ("V", values)):
                features = torch.cat([source[index] for index in picks], dim=-1)
                result = torch.addmm(
                    self.parameters[f"b_{name}"][layer],
                    features,
                    self.parameters[f"W_{name}"][layer],
                ).reshape(*mask.shape, *self.target_shape)
                outputs.append(result)
            key = rotate(outputs[0], target_factors, inverse=False)
            mapped.append((key.transpose(1, 2), outputs[1].transpose(1, 2)))
        return DynamicCache.from_legacy_cache(tuple(mapped))


def context_state(cases, target, pad_id, *, source=None, mapper=None):
    contexts = [case.context for case in cases]
    if source is None:
        ids, mask = pad_tokens(contexts, left=True, pad_id=pad_id)
        return forward(target, ids, mask)

    prefix_ids, prefix_mask = pad_tokens(
        [case.context[: case.prefix_length] for case in cases],
        left=True,
        pad_id=pad_id,
    )
    if not prefix_ids.shape[1]:
        raise ValueError("Cached scoring requires at least one importable prefix")
    _, cache, _ = forward(source, prefix_ids, prefix_mask)
    if mapper is not None:
        cache = mapper.apply(cache, prefix_mask)
    tail_ids, tail_mask = pad_tokens(
        [case.context[case.prefix_length :] for case in cases],
        left=True,
        pad_id=pad_id,
    )
    return forward(target, tail_ids, tail_mask, past=cache, old_mask=prefix_mask)


def ending_logprobs(model, states, labels):
    # Use full-vocabulary FP32 softmax and project only real scored positions.
    for batch, tokens in zip(states.split(128), labels.split(128)):
        logits = model.lm_head(batch).float()
        yield logits.log_softmax(dim=-1).gather(-1, tokens[:, None]).squeeze(-1)


def score_shared(cases, target, pad_id, *, source=None, mapper=None):
    hidden, cache, context_mask = context_state(
        cases, target, pad_id, source=source, mapper=mapper
    )
    endings = [ending for case in cases for ending in case.endings]
    ids, mask = pad_tokens([ending[:-1] for ending in endings], left=False, pad_id=pad_id)
    first = torch.tensor([ending[0] for ending in endings]).view(-1, 4)
    first_lp = target.lm_head(hidden[:, -1]).float().log_softmax(dim=-1)
    totals = first_lp.gather(-1, first).flatten().double()
    if ids.shape[1]:
        cache.batch_repeat_interleave(4)
        suffix_hidden, _, _ = forward(
            target, ids, mask, past=cache, old_mask=context_mask.repeat_interleave(4, dim=0)
        )
        labels = torch.tensor([token for ending in endings for token in ending[1:]])
        probs = torch.cat(list(ending_logprobs(target, suffix_hidden[mask.bool()], labels)))
        per_ending = probs.double().split([len(ending) - 1 for ending in endings])
        totals += torch.stack([scores.sum() for scores in per_ending])
    return totals.view(-1, 4).tolist()


def score_serial(case, target, pad_id, *, source=None, mapper=None):
    # Independent teacher forcing: feed each candidate as a single sequence, and
    # select the logits immediately before the ending tokens by absolute offset.
    prefix_length = case.prefix_length if source is not None else 0
    scores = []
    for ending in case.endings:
        sequence = case.context + ending
        cache = prefix_mask = None
        if prefix_length:
            ids, prefix_mask = pad_tokens(
                [case.context[:prefix_length]], left=False, pad_id=pad_id
            )
            _, cache, _ = forward(source, ids, prefix_mask)
            if mapper is not None:
                cache = mapper.apply(cache, prefix_mask)
        ids, mask = pad_tokens([sequence[prefix_length:]], left=False, pad_id=pad_id)
        hidden, _, _ = forward(target, ids, mask, past=cache, old_mask=prefix_mask)
        offset = len(case.context) - prefix_length - 1
        labels = torch.tensor(ending)
        probs = ending_logprobs(target, hidden[0, offset : offset + len(ending)], labels)
        scores.append(torch.cat(list(probs)).double().sum().item())
    return scores


def results_for(cases, target, source, mapper, pad_id):
    cold = score_shared(cases, target, pad_id)
    eligible = [case for case in cases if case.prefix_length]
    mapped = (
        dict(zip(
            (case.index for case in eligible),
            score_shared(eligible, target, pad_id, source=source, mapper=mapper),
        ))
        if eligible else {}
    )
    return [
        {
            "index": case.index,
            "label": case.label,
            "context_tokens": len(case.context),
            "context_tokens_sha256": json_digest(case.context),
            "ending_tokens": [len(ending) for ending in case.endings],
            "ending_tokens_sha256": [json_digest(ending) for ending in case.endings],
            "imported_tokens": case.prefix_length,
            "recomputed_context_tokens": len(case.context) - case.prefix_length,
            "standalone_logprobs": scores,
            "mapped_logprobs": mapped.get(case.index, scores),
        }
        for case, scores in zip(cases, cold)
    ]


def prediction(scores, lengths=None):
    if lengths is not None:
        scores = [score / length for score, length in zip(scores, lengths)]
    return max(range(len(scores)), key=scores.__getitem__)


def measure(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    result = {"n": len(rows)}
    for mode in ("acc", "acc_norm"):
        correctness = {}
        predictions = {}
        for arm in ("standalone", "mapped"):
            predictions[arm] = [
                prediction(
                    row[f"{arm}_logprobs"],
                    row["ending_tokens"] if mode == "acc_norm" else None,
                )
                for row in rows
            ]
            correctness[arm] = np.array([
                guess == row["label"] for guess, row in zip(predictions[arm], rows)
            ])
        cold, mapped = correctness["standalone"], correctness["mapped"]
        cold_count, mapped_count = int(cold.sum()), int(mapped.sum())
        contingency = np.bincount(
            2 * cold.astype(np.int8) + mapped.astype(np.int8), minlength=4
        )
        # Joint correctness bootstrap via multinomial counts of paired outcomes.
        draws = np.random.default_rng(0).multinomial(
            len(rows), contingency / len(rows), size=10000
        )
        cold_draws = draws[:, 2] + draws[:, 3]
        mapped_draws = draws[:, 1] + draws[:, 3]
        valid = cold_draws > 0
        result[mode] = {
            "standalone_correct": cold_count,
            "mapped_correct": mapped_count,
            "standalone": cold_count / len(rows),
            "mapped": mapped_count / len(rows),
            "retention": mapped_count / cold_count if cold_count else None,
            "paired_delta": (mapped_count - cold_count) / len(rows),
            "paired_delta_ci95": np.quantile(
                (mapped_draws - cold_draws) / len(rows), [0.025, 0.975]
            ).tolist(),
            "paired_retention_ci95": np.quantile(
                mapped_draws[valid] / cold_draws[valid], [0.025, 0.975]
            ).tolist() if valid.any() else None,
            "wrong_to_correct": int(contingency[1]),
            "correct_to_wrong": int(contingency[2]),
            "changed_prediction": sum(
                cold_choice != mapped_choice for cold_choice, mapped_choice
                in zip(predictions["standalone"], predictions["mapped"])
            ),
        }
    return result


def pilot(cases, target, source, mapper, pad_id) -> list[dict]:
    # Compare the shortest and longest eligible prompts in a mixed padding batch.
    eligible = sorted(
        (case for case in cases if case.prefix_length), key=lambda case: len(case.context)
    )
    sample = [eligible[0], eligible[-1]]
    cold = score_shared(sample, target, pad_id)
    exact = score_shared(sample, target, pad_id, source=target)
    mapped = score_shared(sample, target, pad_id, source=source, mapper=mapper)
    checks = []
    for case, cold_scores, exact_scores, mapped_scores in zip(sample, cold, exact, mapped):
        cold_serial = score_serial(case, target, pad_id)
        mapped_serial = score_serial(case, target, pad_id, source=source, mapper=mapper)
        comparisons = (
            ("cold_batch_vs_serial", cold_scores, cold_serial),
            ("exact_cache_vs_cold_serial", exact_scores, cold_serial),
            ("mapped_batch_vs_serial", mapped_scores, mapped_serial),
        )
        lengths = [len(ending) for ending in case.endings]
        for name, actual, reference in comparisons:
            max_per_token_error = max(
                abs(left - right) / length
                for left, right, length in zip(actual, reference, lengths)
            )
            check = {
                "test": name,
                "index": case.index,
                "context_tokens": len(case.context),
                "imported_tokens": case.prefix_length,
                "max_mean_logprob_error": max_per_token_error,
                "same_acc_norm_prediction": (
                    prediction(actual, lengths) == prediction(reference, lengths)
                ),
            }
            print(json.dumps(check), flush=True)
            if not math.isfinite(max_per_token_error) or max_per_token_error > 0.15:
                raise RuntimeError(f"Pilot score mismatch: {check}")
            checks.append(check)
    return checks


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "mapper", "data", "public-report", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--pilot-check", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.limit <= 1000 and 1 <= args.threads <= 4
            and args.batch_size > 0 and args.minutes > 0):
        parser.error("Use limit 1–1000, threads 1–4, and positive batch size and minutes")

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    public_digest = sha256(args.public_report)
    if public_digest != PUBLIC_REPORT_SHA256:
        raise ValueError("Unexpected public n200 report SHA-256")
    public = read_json(args.public_report)["hellaswag"]
    manifest_path = args.mapper / "manifest.json"
    manifest = read_json(manifest_path)
    weights = args.mapper / "mapper-fp32.safetensors"
    if manifest["mapper_id"] != "ridge-qwen3-0.6b-to-1.7b-k4-n200":
        raise ValueError("Expected the public k4/n200 artifact")

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    cases = load_cases(args.data, public, tokenizer)[: args.limit]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Expected the Qwen3 padding token")
    metadata = {
        "runner_sha256": sha256(Path(__file__)),
        "public_source_commit": PUBLIC_COMMIT,
        "public_report_sha256": public_digest,
        "public_metrics": {key: public[key] for key in ("n", "standalone", "mapped", "retention_acc_norm")},
        "dataset": "Rowan/hellaswag",
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "validation",
        "dataset_jsonl_sha256": sha256(args.data),
        "requested_indices": [case.index for case in cases],
        "tokenized_cases_sha256": json_digest([case.tokens_digest() for case in cases]),
        "short_prompt_indices": [case.index for case in cases if not case.prefix_length],
        "source": model_identity(args.source, manifest["source_model"]),
        "target": model_identity(args.target, manifest["target_model"]),
        "mapper_id": manifest["mapper_id"],
        "mapper_code_commit": manifest["code_commit"],
        "mapper_manifest_sha256": sha256(manifest_path),
        "mapper_weights_sha256": sha256(weights),
        "mapper_inference_dtype": "bfloat16",
        "mapper_rope_compute_dtype": "float32",
        "model_dtype": "bfloat16",
        "attention": "eager",
        "device": "cpu",
        "pad_token_id": pad_id,
        "block_size": BLOCK_SIZE,
        "normalization": "sum ending logprobs / ending token count",
        "seed": 0,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "tokenizers", "safetensors", "numpy")
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    rows_path = args.out / "cases.json"
    if metadata_path.exists():
        if read_json(metadata_path) != metadata:
            raise ValueError(f"Resume metadata differs in {metadata_path}")
    else:
        write_json(metadata_path, metadata)

    rows = read_json(rows_path) if rows_path.exists() else []
    expected_indices = metadata["requested_indices"]
    if [row["index"] for row in rows] != expected_indices[:len(rows)]:
        raise ValueError("Checkpoint rows are not a prefix of the requested index order")
    print(f"Starting with {len(rows)}/{len(cases)} completed cases", flush=True)

    if len(rows) < len(cases) or args.pilot_check:
        source, target = load_model(args.source), load_model(args.target)
        mapper = RidgeMapper(manifest, weights)
        if args.pilot_check:
            write_json(args.out / "pilot-checks.json", pilot(cases, target, source, mapper, pad_id))
        scored_at = time.monotonic()
        initial_count = len(rows)
        # Sort small windows by length to limit padding while retaining an ordered
        # checkpoint. A partial window is saved before the time budget is enforced.
        window_size = args.batch_size * 4
        for start in range(initial_count, len(cases), window_size):
            window = cases[start : start + window_size]
            ordered = sorted(window, key=lambda case: (len(case.context), max(map(len, case.endings))))
            chunk = {}
            for offset in range(0, len(ordered), args.batch_size):
                batch = ordered[offset : offset + args.batch_size]
                for row in results_for(batch, target, source, mapper, pad_id):
                    chunk[row["index"]] = row
            rows.extend(chunk[case.index] for case in window)
            write_json(rows_path, rows)
            elapsed = time.monotonic() - scored_at
            print(
                f"{len(rows)}/{len(cases)} cases "
                f"({elapsed / (len(rows) - initial_count):.3f} s/case)",
                flush=True,
            )
            if time.monotonic() - started >= args.minutes * 60:
                print("Time budget reached; checkpoint is resumable", flush=True)
                break

    summary = {
        "complete": len(rows) == len(cases),
        "public_sample_complete": len(rows) == 1000,
        "n_requested": len(cases),
        "n_completed": len(rows),
        "n_short_cold": sum(not row["imported_tokens"] for row in rows),
        "all_cases": measure(rows),
        "actual_transfer_only": measure([row for row in rows if row["imported_tokens"]]),
        "bootstrap": {"draws": 10000, "seed": 0, "paired": True},
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
