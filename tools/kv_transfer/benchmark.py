
#!/usr/bin/env python3
"""Prepare and evaluate fixed, teacher-forced KV transfer benchmark samples."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from itertools import groupby, islice, zip_longest
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from dataset import prompt_digest
from mapper import Mapping, apply
from pilot import configure
from snapshot import ModelSpec, Snapshot, read_json, write_json

TASKS = ("arc_challenge", "hellaswag", "wikitext2")
MODES = ("cold", "exact", "transfer")
CHANCE = 0.25  # Paper §4.1 and Table 1.
WT2_PREFIX = 1024
WT2_CHUNK = 2 * WT2_PREFIX
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202608
SMALL_CHOICE_MARGIN = 0.002


def file_digest(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet

        for batch in pyarrow.parquet.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            yield from (json.loads(line) for line in file if line.strip())
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as file:
            yield from json.load(file)
    else:
        raise ValueError(f"expected a .parquet, .jsonl or .json dataset: {path}")


def preprocess_hellaswag(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def choice_example(task, row):
    if task == "arc_challenge":
        return (f"Question: {row['question']}\nAnswer:", row["choices"]["text"],
                row["choices"]["label"].index(row["answerKey"]))
    context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    prompt = preprocess_hellaswag(f"{row['activity_label']}: {context}")
    choices = [preprocess_hellaswag(choice) for choice in row["endings"]]
    return prompt, choices, int(row["label"])


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def encode_choice(tokenizer, prompt, choices, label):
    context = prompt.rstrip()
    delimiter = prompt[len(context):] + " "
    prefix_ids = encode(tokenizer, context)
    if len(prefix_ids) < 2:
        raise ValueError("evaluation requires a cache prefix and a context anchor token")
    if len(choices) < 2 or not 0 <= label < len(choices):
        raise ValueError("invalid multiple-choice label")
    continuations = []
    for choice in choices:
        joined = encode(tokenizer, context + delimiter + choice)
        if joined[:len(prefix_ids)] != prefix_ids:
            raise ValueError("a choice changes tokenization across the context boundary")
        targets = joined[len(prefix_ids):]
        if not choice or not targets:
            raise ValueError("each multiple-choice continuation needs text and token IDs")
        continuations.append({"text": choice, "characters": len(choice), "ids": targets})
    return {"context_ids": prefix_ids, "candidates": continuations, "gold": label}


def rank_sample(items, task, seed, limit):
    def rank(item):
        return hashlib.sha256(f"{seed}:{task}:{item[0]}".encode()).digest(), item[0]

    return sorted(items, key=rank)[:limit]


def prepare(args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    spec = read_json(args.datasets)
    items = []
    sources = {}
    limits = {"arc_challenge": args.arc, "hellaswag": args.hellaswag,
              "wikitext2": args.wikitext2}
    for task, limit in limits.items():
        if limit == 0:
            continue
        entry = spec[task]
        path = (args.datasets.parent / entry["file"]).resolve()
        sources[task] = {"path": str(path), "sha256": file_digest(path),
                         "revision": entry["revision"], "source": entry["source"],
                         "split": entry["split"]}
        if task == "wikitext2":
            # WikiText-2 test is a stream, including its empty source lines.
            text = "\n\n".join(row["text"] for row in rows(path))
            tokens = encode(tokenizer, text)
            chunks = len(tokens) // WT2_CHUNK
            if limit > chunks:
                raise ValueError(f"{task} has only {chunks} complete chunks")
            for index, _ in rank_sample(enumerate(range(chunks)), task, args.seed, limit):
                start = index * WT2_CHUNK
                context = tokens[start:start + WT2_PREFIX]
                target = tokens[start + WT2_PREFIX:start + WT2_CHUNK]
                items.append({"task": task, "row": index, "id": f"{task}-{index}",
                              "context_ids": context, "candidates": [{"ids": target}],
                              "stream_token_offset": start})
        else:
            selected = rank_sample(enumerate(rows(path)), task, args.seed, limit)
            if len(selected) != limit:
                raise ValueError(f"{task} has only {len(selected)} rows")
            for index, row in selected:
                example = encode_choice(tokenizer, *choice_example(task, row))
                items.append({"task": task, "row": index, "id": f"{task}-{index}",
                              "dataset_id": row.get("id", row.get("ind")), **example})
    if not items:
        raise ValueError("select at least one benchmark case")
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"version": 1, "seed": args.seed, "datasets": sources,
                   "tokenizer_sha256": file_digest(args.tokenizer), "items": items},
                  output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Prepared {len(items)} cases in {args.output}")


class NativeScorer:
    def __init__(self, executable, model_dir, spec, threads, score_block_rows=1):
        self.executable = executable.resolve()
        self.model_dir = model_dir.resolve()
        config = read_json(self.model_dir / "config.json")
        if config["rope_theta"] != spec.theta:
            raise ValueError(f"model RoPE theta disagrees with mapping: {model_dir}")
        nntr_config = read_json(self.model_dir / "nntr_config.json")
        self.max_seq_len = nntr_config["max_seq_len"]
        self.init_seq_len = nntr_config["init_seq_len"]
        self.spec = spec
        self.score_block_rows = score_block_rows
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("NNTR_SCORE_", "NNTR_KV_TRANSFER_"))
                    and key not in {"NNTR_GENERATED_IDS", "NNTR_FORCED_DECODE_INPUTS"}}
        self.env.update(NNTR_ENGINE="cpu", NNTR_KV_CORE_CPU="1",
                        NNTR_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                        OPENBLAS_NUM_THREADS=str(threads),
                        NNTR_KV_TRANSFER_MODEL_ID=spec.model_id,
                        NNTR_KV_TRANSFER_ROPE_THETA=str(config["rope_theta"]))

    def execute(self, model_dir, args, log, extra=None):
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run([str(self.executable), str(model_dir), *args],
                                       env={**self.env, **(extra or {})}, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"native {self.spec.model_id} exited {completed.returncode}; see {log}")
        return round(time.monotonic() - start, 3)

    def model_config(self, work, name, context_len, *, marker=None, prefix_len=None):
        directory = work / name
        configure(self.model_dir, directory, "Raw token benchmark", 1,
                  cache=marker, tokens=prefix_len)
        config_path = directory / "nntr_config.json"
        config = read_json(config_path)
        config["init_seq_len"] = max(config["init_seq_len"], context_len)
        write_json(config_path, config)
        return directory

    def export(self, work, log, ids, destination):
        model = self.model_config(work, log.stem + "-model", len(ids),
                                  marker=work / (log.stem + "-cache"))
        filename = work / (log.stem + ".ids")
        write_ids(filename, ids)
        seconds = self.execute(model, [""], log, {
            "NNTR_KV_TRANSFER_EXPORT": str(destination),
            "NNTR_KV_TRANSFER_EXPORT_INPUT_IDS": str(filename),
        })
        if not destination.is_dir():
            raise RuntimeError(f"native export omitted its snapshot; see {log}")
        return seconds

    def score(self, work, log, cases, *, persistent):
        jobs = []
        scores = {}
        for example, scratch, snapshots in cases:
            case_id = example["id"]
            context = example["context_ids"]
            scores[case_id] = {mode: [] for mode in snapshots}
            for mode, snapshot in snapshots.items():
                if snapshot is not None and snapshot.spec != self.spec:
                    raise ValueError("snapshot model spec disagrees with scorer model")
                feed = context if snapshot is None else context[-1:]
                for index, candidate in enumerate(example["candidates"]):
                    targets = candidate["ids"]
                    requested = len(context) + len(targets) - 1
                    if requested > self.max_seq_len:
                        raise ValueError(f"scoring needs {requested} model positions, "
                                         f"limit is {self.max_seq_len}")
                    inputs = scratch / f"{mode}-{index}-context.ids"
                    target_path = scratch / f"{mode}-{index}-target.ids"
                    output = scratch / f"{mode}-{index}-score.json"
                    write_ids(inputs, feed)
                    write_ids(target_path, targets)
                    job = {"input_ids": str(inputs), "target_ids": str(target_path),
                           "output": str(output)}
                    if snapshot is not None:
                        job.update(transfer_import=str(snapshot.path), prefix_tokens=snapshot.tokens)
                    jobs.append((case_id, mode, index, job, len(targets)))

        model = self.model_config(work, "score-model",
                                  max(len(example["context_ids"]) for example, _, _ in cases))
        if persistent:
            requests_path = work / "requests.json"
            requests_path.write_text(json.dumps(
                [{"prompt": "", "score": job} for _, _, _, job, _ in jobs]),
                encoding="utf-8")
            seconds = self.execute(model, ["--requests", str(requests_path)], log,
                                   self.score_environment())
            for case_id, mode, _, job, count in jobs:
                result = read_score(Path(job["output"]), count, self.score_block_rows)
                scores[case_id][mode].append(result)
        else:
            seconds = 0
            marker = work / "cache-load"
            for case_id, mode, index, job, count in jobs:
                extra = {**self.score_environment(), "NNTR_SCORE_INPUT_IDS": job["input_ids"],
                         "NNTR_SCORE_TARGET_IDS": job["target_ids"],
                         "NNTR_SCORE_OUTPUT": job["output"]}
                job_model = model
                if "transfer_import" in job:
                    job_model = work / f"{case_id}-{mode}-score-model"
                    if not job_model.exists():
                        marker.touch()
                        self.model_config(work, job_model.name, 1, marker=marker,
                                          prefix_len=job["prefix_tokens"])
                    extra["NNTR_KV_TRANSFER_IMPORT"] = job["transfer_import"]
                job_log = log.with_name(f"{case_id}-{mode}-{index}.log")
                duration = self.execute(job_model, [""], job_log, extra)
                score = read_score(Path(job["output"]), count, self.score_block_rows)
                scores[case_id][mode].append({**score, "seconds_with_process_startup": duration})
                seconds += duration
        return scores, round(seconds, 3)

    def score_environment(self):
        if self.score_block_rows == 1:
            return {}
        return {"NNTR_SCORE_BLOCK_ROWS": str(self.score_block_rows)}


def write_ids(path, ids):
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError(f"{path}: expected nonempty, nonnegative token IDs")
    path.write_text(" ".join(map(str, ids)) + "\n", encoding="ascii")


def read_score(path, count, expected_block_rows):
    score = read_json(path)
    if score.get("score_block_rows", 1) != expected_block_rows:
        raise ValueError(f"native scorer did not use the requested block size: {path}")
    probabilities = score["token_logprobs"]
    total = score["sum_logprob"]
    if (score["count"] != count or len(probabilities) != count
            or any(not math.isfinite(value) or value > 0 for value in probabilities)
            or not math.isclose(total, math.fsum(probabilities), abs_tol=1e-6)):
        raise ValueError(f"native score failed token count or likelihood checks: {path}")
    return score


def verify_prefix(snapshot, expected_ids, spec):
    if (snapshot.token_ids != tuple(expected_ids)
            or snapshot.positions != tuple(range(len(expected_ids)))
            or snapshot.spec != spec):
        raise ValueError(f"snapshot does not match expected tokens, positions and model: {snapshot.path}")


def prepare_case(example, log_dir, work, source, target, mapping, args, exact_control, prior):
    context = example["context_ids"]
    prefix = context[:-1]
    if not prefix:
        raise ValueError(f"{example['id']}: at least two context tokens needed")
    result = prior or {"id": example["id"], "task": example["task"],
                       "cache_tokens": len(prefix), "prefix_digest": prompt_digest(prefix),
                       "score_runs": [], "scores": {}}
    snapshots = {}
    if "cold" not in result["scores"]:
        snapshots["cold"] = None
    if exact_control and "exact" not in result["scores"]:
        exact_dir = work / "exact"
        result["exact_export_seconds"] = target.export(
            work, log_dir / "export-exact.log", prefix, exact_dir)
        native_snapshot = Snapshot(exact_dir)
        verify_prefix(native_snapshot, prefix, target.spec)
        snapshots["exact"] = native_snapshot
    if mapping is not None and "transfer" not in result["scores"]:
        source_dir = work / "source"
        transfer_dir = work / "transfer"
        result["source_export_seconds"] = source.export(
            work, log_dir / "export-source.log", prefix, source_dir)
        source_snapshot = Snapshot(source_dir)
        verify_prefix(source_snapshot, prefix, source.spec)
        start = time.monotonic()
        apply(mapping, source_snapshot, transfer_dir,
              block_size=args.block_tokens, memory_mib=args.memory_mib)
        result["map_seconds"] = round(time.monotonic() - start, 3)
        transfer_snapshot = Snapshot(transfer_dir)
        verify_prefix(transfer_snapshot, prefix, target.spec)
        snapshots["transfer"] = transfer_snapshot
    return result, snapshots


def record_scores(example, result, scores, elapsed, log, batch_size):
    result["score_runs"].append({"modes": list(scores), "seconds_with_process_startup": elapsed,
                                 "cases_in_process": batch_size, "native_log": str(log)})
    result["scores"].update(scores)
    scores = result["scores"]
    if "exact" in scores:
        result["exact_max_token_logprob_error"] = max(
            abs(cold - exact)
            for cold_choice, exact_choice in zip(scores["cold"], scores["exact"])
            for cold, exact in zip(cold_choice["token_logprobs"], exact_choice["token_logprobs"])
        )
    if "gold" in example:
        result["gold"] = example["gold"]
        result["predictions"] = {
            mode: max(range(len(choices)), key=lambda i: choices[i])
            for mode, choices in normalized_scores(example, result).items()
        }
    return result


def normalized_scores(example, case):
    return {
        mode: [score["sum_logprob"] / candidate["characters"]
               for score, candidate in zip(scores, example["candidates"])]
        for mode, scores in case["scores"].items()
    }


def choice_margins(examples, cases):
    close = {}
    for case in cases:
        for mode, values in normalized_scores(examples[case["id"]], case).items():
            best, runner_up = sorted(values, reverse=True)[:2]
            margin = best - runner_up
            if margin < SMALL_CHOICE_MARGIN:
                close.setdefault(mode, []).append({"id": case["id"], "margin_nat_per_char": margin})
    return close


def paired_accuracy_interval(cold_only, transfer_only, count):
    if count < 2:
        return None
    # Each resampled pair contributes a loss, no change, or a gain.
    probabilities = [cold_only / count, (count - cold_only - transfer_only) / count,
                     transfer_only / count]
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        count, probabilities, size=BOOTSTRAP_DRAWS
    )
    deltas = 100 * (draws[:, 2] - draws[:, 0]) / count
    return np.quantile(deltas, (0.025, 0.975)).tolist()


def summarize_choices(cases):
    by_mode = {mode: [case for case in cases if mode in case["predictions"]] for mode in MODES}
    correct = {
        mode: sum(case["predictions"][mode] == case["gold"] for case in subset)
        for mode, subset in by_mode.items() if subset
    }
    accuracy = {mode: value / len(by_mode[mode]) for mode, value in correct.items()}
    summary = {
        "metric": "acc_norm", "accuracy": accuracy, "correct": correct,
        "chance_floor": CHANCE, "cold_exact_prediction_disagreements": sum(
            case["predictions"]["cold"] != case["predictions"]["exact"]
            for case in by_mode["exact"]
        ),
    }
    paired_cases = by_mode["transfer"]
    if paired_cases:
        cold = sum(case["predictions"]["cold"] == case["gold"]
                   for case in paired_cases) / len(paired_cases)
        transferred = accuracy["transfer"]
        paired = {"both_correct": 0, "cold_only_correct": 0,
                  "transfer_only_correct": 0, "both_incorrect": 0}
        keys = {(True, True): "both_correct", (True, False): "cold_only_correct",
                (False, True): "transfer_only_correct", (False, False): "both_incorrect"}
        for case in paired_cases:
            predictions, gold = case["predictions"], case["gold"]
            paired[keys[predictions["cold"] == gold, predictions["transfer"] == gold]] += 1
        summary.update(
            cold_transfer_paired_counts=paired, paired_cold_accuracy=cold,
            transfer_minus_cold_pp=100 * (transferred - cold),
            paired_bootstrap_95_ci_pp=paired_accuracy_interval(
                paired["cold_only_correct"], paired["transfer_only_correct"], len(paired_cases)
            ),
            transfer_retention_pct=100 * transferred / cold if cold else None,
            transfer_floor_normalized_retention_pct=(
                100 * (transferred - CHANCE) / (cold - CHANCE) if cold > CHANCE else None
            ),
        )
    return summary


def summarize_wikitext(cases):
    scores = {mode: [case["scores"][mode][0] for case in cases if mode in case["scores"]]
              for mode in MODES}
    counts = {mode: sum(score["count"] for score in subset)
              for mode, subset in scores.items() if subset}
    nll = {mode: -math.fsum(score["sum_logprob"] for score in scores[mode]) / count
           for mode, count in counts.items()}
    summary = {"metric": "perplexity", "scored_tokens": counts, "mean_nll": nll,
               "perplexity": {mode: math.exp(loss) for mode, loss in nll.items()}}
    if "transfer" in nll:
        paired_cold = [case["scores"]["cold"][0] for case in cases
                       if "transfer" in case["scores"]]
        paired_nll = -math.fsum(score["sum_logprob"] for score in paired_cold) / counts["transfer"]
        summary["transfer_minus_cold_nll"] = nll["transfer"] - paired_nll
    return summary


def summarize(manifest, results, tolerance):
    report = {}
    for task in TASKS:
        cases = [case for case in results if case["task"] == task]
        planned = sum(example["task"] == task for example in manifest["items"])
        if not planned:
            continue
        entry = {"planned": planned, "scored": len(cases),
                 "scored_by_mode": {
                     mode: sum(mode in case["scores"] for case in cases)
                     for mode in MODES
                 }}
        if cases:
            errors = [case["exact_max_token_logprob_error"] for case in cases
                      if "exact_max_token_logprob_error" in case]
            if task == "wikitext2":
                entry.update(summarize_wikitext(cases))
            else:
                examples = {example["id"]: example for example in manifest["items"]
                            if example["task"] == task}
                entry.update(summarize_choices(cases))
                entry["small_choice_margins"] = choice_margins(examples, cases)
                entry["small_choice_margin_threshold_nat_per_char"] = SMALL_CHOICE_MARGIN
            entry["exact_controls_scored"] = len(errors)
            if errors:
                entry.update(exact_max_token_logprob_error=max(errors),
                             exact_error_tolerance=tolerance)
                entry["exact_import_check_passed"] = (
                    max(errors) <= tolerance and entry.get("cold_exact_prediction_disagreements", 0) == 0
                )
        report[task] = entry
    return report


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def model_spec(directory, model_id):
    config = read_json(directory / "config.json")
    return ModelSpec(model_id, config["num_hidden_layers"], config["num_key_value_heads"],
                     config["head_dim"], config["rope_theta"], config["head_dim"])


def pin_json(path, provenance):
    if path.exists():
        if read_json(path) != provenance:
            raise ValueError(f"configuration differs from existing results: {path}")
    else:
        save_json(path, provenance)


def ordered_cases(manifest):
    # Keep early progress visible in both accuracy tasks even on long CPU runs.
    by_task = [[example for example in manifest["items"] if example["task"] == task]
               for task in TASKS[:2]]
    for pair in zip_longest(*by_task):
        yield from (example for example in pair if example is not None)
    yield from (example for example in manifest["items"] if example["task"] == "wikitext2")


def run(args):
    manifest = read_json(args.manifest)
    if manifest["version"] != 1:
        raise ValueError("unsupported benchmark manifest version")
    mapping = None if args.mapping == Path("-") else Mapping(args.mapping)
    expected_tokenizer = manifest["tokenizer_sha256"]
    for directory in (args.source_dir, args.target_dir):
        if file_digest(directory / "tokenizer.json") != expected_tokenizer:
            raise ValueError(f"{directory}: tokenizer differs from the pinned benchmark tokenizer")
    source_spec = model_spec(args.source_dir, args.source_id)
    target_spec = model_spec(args.target_dir, args.target_id)
    if mapping is not None and (source_spec != mapping.geometry.source
                                or target_spec != mapping.geometry.target):
        raise ValueError("mapping model specs differ from the selected native models")
    source = NativeScorer(args.executable, args.source_dir, source_spec, args.threads)
    target = NativeScorer(args.executable, args.target_dir, target_spec, args.threads,
                          args.score_block_rows)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    libraries = ("libcausallm.so", "libcausallm.dylib", "causallm.dll", "libcausallm.dll")
    fingerprint = {
        name: file_digest(args.executable.parent / name)
        for name in libraries if (args.executable.parent / name).is_file()
    }
    pin_json(out / "run.json", {
        "manifest_sha256": file_digest(args.manifest),
        "executable_sha256": file_digest(args.executable), "native_libraries_sha256": fingerprint,
        "source_dir": str(source.model_dir), "target_dir": str(target.model_dir),
        "target_model_id": args.target_id, "source_model_id": args.source_id,
        "source_config_sha256": file_digest(source.model_dir / "nntr_config.json"),
        "target_config_sha256": file_digest(target.model_dir / "nntr_config.json"),
        "threads": args.threads, "exact_tolerance": args.exact_tolerance,
        "exact_controls": args.exact_controls, "single_request": args.single_request,
        "score_block_rows": args.score_block_rows,
    })
    if mapping is not None:
        pin_json(out / "mapping.json", {
            "metadata": read_json(args.mapping / "metadata.json"),
            "directory": str(args.mapping.resolve()),
            "weights_sha256": file_digest(args.mapping / "weights.f32"),
            "bias_sha256": file_digest(args.mapping / "bias.f32"),
            "block_tokens": args.block_tokens, "memory_mib": args.memory_mib,
        })
    report_path = out / "summary.json"
    results = {}
    for example in manifest["items"]:
        result_path = out / example["id"] / "result.json"
        if result_path.exists():
            results[example["id"]] = read_json(result_path)

    def save_report():
        report = summarize(manifest, list(results.values()), args.exact_tolerance)
        save_json(report_path, report)
        if any(entry.get("exact_import_check_passed") is False for entry in report.values()):
            raise RuntimeError(f"exact cache control failed; see {report_path}")
        return report

    save_report()
    per_task = dict.fromkeys(TASKS, 0)
    pending = []
    for example in ordered_cases(manifest):
        task = example["task"]
        exact_control = per_task[task] < args.exact_controls
        per_task[task] += 1
        prior = results.get(example["id"])
        required = {"cold"}
        if exact_control:
            required.add("exact")
        if mapping is not None:
            required.add("transfer")
        missing = required - (prior["scores"].keys() if prior else set())
        if missing and (not args.tasks or task in args.tasks):
            pending.append((example, exact_control, prior, missing))
    pending = pending[:args.limit]

    runs = out / "score-logs"
    runs.mkdir(exist_ok=True)

    def plane(case):
        # Keep each case on the same input plane it would use independently.
        return max(target.init_seq_len, len(case[0]["context_ids"]))

    for _, same_plane in groupby(pending, key=plane):
        while batch := list(islice(same_plane, args.batch_cases)):
            with tempfile.TemporaryDirectory(prefix=".cache-", dir=out) as directory:
                work = Path(directory)
                batch_inputs = []
                batch_results = []
                for example, exact_control, prior, missing in batch:
                    case_id = example["id"]
                    case_dir = out / case_id
                    case_dir.mkdir(exist_ok=True)
                    scratch = work / case_id
                    scratch.mkdir()
                    print(f"{case_id}: starting {','.join(sorted(missing))}", flush=True)
                    result, snapshots = prepare_case(example, case_dir, scratch, source, target,
                                                     mapping, args, exact_control, prior)
                    batch_inputs.append((example, scratch, snapshots))
                    batch_results.append((case_dir, example, result))
                first, last = batch[0][0]["id"], batch[-1][0]["id"]
                stage = "with-transfer" if mapping is not None else "baseline"
                log = runs / f"{first}--{last}-{stage}.log"
                scores, elapsed = target.score(work, log, batch_inputs,
                                               persistent=not args.single_request)
                for case_dir, example, result in batch_results:
                    case_id = example["id"]
                    record_scores(example, result, scores[case_id], elapsed, log.relative_to(out),
                                  len(batch))
                    save_json(case_dir / "result.json", result)
                    results[case_id] = result
            print(f"{first}--{last}: scored {len(batch)} cases in {elapsed:.3f}s", flush=True)
            save_report()
    print(json.dumps(save_report(), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="pin the sampled cases and exact token IDs")
    for name in ("datasets", "tokenizer", "output"):
        preparation.add_argument(name, type=Path)
    preparation.add_argument("--seed", type=int, default=202608)
    preparation.add_argument("--arc", type=int, default=32)
    preparation.add_argument("--hellaswag", type=int, default=32)
    preparation.add_argument("--wikitext2", type=int, default=1)

    evaluation = commands.add_parser("run", help="score the pinned cases with three cache modes")
    for name in ("executable", "source_dir", "target_dir"):
        evaluation.add_argument(name, type=Path)
    evaluation.add_argument("mapping", type=Path, help="learned mapping directory, or - for baseline only")
    for name in ("manifest", "output"):
        evaluation.add_argument(name, type=Path)
    evaluation.add_argument("--source-id", default="Qwen/Qwen3-0.6B")
    evaluation.add_argument("--target-id", default="Qwen/Qwen3-1.7B")
    evaluation.add_argument("--threads", type=int, default=8)
    evaluation.add_argument("--batch-cases", type=int, default=1,
                            help="cases to score per persistent native process")
    evaluation.add_argument("--score-block-rows", type=int, default=1,
                            help="teacher-forcing rows per native forward (1 to 64)")
    evaluation.add_argument("--block-tokens", type=int, default=128)
    evaluation.add_argument("--memory-mib", type=int, default=512)
    evaluation.add_argument("--exact-tolerance", type=float, default=0.01)
    evaluation.add_argument("--limit", type=int, help="maximum new cases in this invocation")
    evaluation.add_argument("--tasks", choices=TASKS, nargs="+", help="only start selected tasks")
    evaluation.add_argument("--exact-controls", type=int, default=2,
                            help="cases per task with native exact-cache import")
    evaluation.add_argument("--single-request", action="store_true",
                            help="invoke the native model separately for every score request")

    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.arc, args.hellaswag, args.wikitext2) < 0:
            parser.error("sample counts must be nonnegative")
        prepare(args)
    else:
        if min(args.threads, args.batch_cases, args.block_tokens, args.memory_mib) < 1:
            parser.error("thread, batch, block and memory limits must be positive")
        if args.single_request and args.batch_cases != 1:
            parser.error("single request mode requires a batch of one case")
        if args.limit is not None and args.limit < 1:
            parser.error("case limit must be positive")
        if not 1 <= args.score_block_rows <= 64:
            parser.error("score block size must be between 1 and 64")
        if args.exact_controls < 1:
            parser.error("at least one exact cache control per task is required")
        if not math.isfinite(args.exact_tolerance) or args.exact_tolerance < 0:
            parser.error("exact tolerance must be finite and nonnegative")
        run(args)


if __name__ == "__main__":
    main()
