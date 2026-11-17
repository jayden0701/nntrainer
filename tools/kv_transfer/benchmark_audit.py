
#!/usr/bin/env python3
"""Recheck numerically close benchmark choices using scalar teacher forcing."""

import argparse
import json
import tempfile
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

from benchmark import (
    NativeScorer, file_digest, model_spec, normalized_scores, prepare_case, save_json,
)
from mapper import Mapping
from snapshot import read_json


def audit(args):
    out = args.results.resolve()
    recorded_run = read_json(out / "run.json")
    if (file_digest(args.executable) != recorded_run["executable_sha256"]
            or file_digest(args.manifest) != recorded_run["manifest_sha256"]):
        raise ValueError("executable or manifest differs from the benchmark run")
    for library, digest in recorded_run["native_libraries_sha256"].items():
        if file_digest(args.executable.parent / library) != digest:
            raise ValueError(f"{library} differs from the benchmark run")

    recorded_mapping = read_json(out / "mapping.json")
    mapping_path = Path(recorded_mapping["directory"])
    for filename, key in (("weights.f32", "weights_sha256"), ("bias.f32", "bias_sha256")):
        if file_digest(mapping_path / filename) != recorded_mapping[key]:
            raise ValueError(f"{filename} differs from the benchmark run")
    mapper_args = SimpleNamespace(block_tokens=recorded_mapping["block_tokens"],
                                  memory_mib=recorded_mapping["memory_mib"])
    mapping = Mapping(mapping_path)
    models = {}
    for mode in ("source", "target"):
        directory = Path(recorded_run[f"{mode}_dir"])
        if file_digest(directory / "nntr_config.json") != recorded_run[f"{mode}_config_sha256"]:
            raise ValueError(f"{mode} config differs from the benchmark run")
        spec = model_spec(directory, recorded_run[f"{mode}_model_id"])
        models[mode] = NativeScorer(args.executable, directory, spec, recorded_run["threads"])
    target = models["target"]
    examples = {example["id"]: example for example in read_json(args.manifest)["items"]}

    audit_dir = out / "audit-scalar"
    audit_dir.mkdir(exist_ok=True)
    report_path = audit_dir / "summary.json"
    canonical_summary = read_json(out / "summary.json")
    flagged = {}
    for summary in canonical_summary.values():
        for mode, entries in summary.get("small_choice_margins", {}).items():
            if mode in {"cold", "transfer"}:
                for entry in entries:
                    flagged.setdefault(entry["id"], set()).add(mode)

    def input_plane(case_id):
        return max(target.init_seq_len, len(examples[case_id]["context_ids"]))

    groups = sorted(flagged, key=input_plane)
    results = {}
    for _, case_ids in groupby(groups, key=input_plane):
        with tempfile.TemporaryDirectory(prefix=".cache-", dir=audit_dir) as location:
            work = Path(location)
            pending = []
            originals = {}
            for case_id in case_ids:
                report_path_for_case = audit_dir / case_id / "result.json"
                previous = read_json(report_path_for_case) if report_path_for_case.exists() else {}
                checked = set(previous)
                requested = flagged[case_id] - checked
                results[case_id] = previous
                if not requested:
                    continue
                example = examples[case_id]
                case_work = work / case_id
                case_work.mkdir()
                case_logs = report_path_for_case.parent
                case_logs.mkdir(exist_ok=True)
                canonical = read_json(out / case_id / "result.json")
                prior = None if "cold" in requested else {"scores": {"cold": canonical["scores"]["cold"]}}
                _, snapshots = prepare_case(example, case_logs, case_work, models["source"], target,
                                             mapping if "transfer" in requested else None,
                                             mapper_args, False, prior)
                if snapshots.keys() != requested:
                    raise RuntimeError(f"{case_id}: audit is preparing unintended score modes")
                originals[case_id] = canonical
                pending.append((example, case_work, snapshots))
            if not pending:
                continue
            native_log = audit_dir / f"{pending[0][0]['id']}--{pending[-1][0]['id']}.log"
            print(f"Rechecking {sum(len(snapshots) for _, _, snapshots in pending)} close scores "
                  f"across {len(pending)} cases", flush=True)
            scores, elapsed = target.score(work, native_log, pending, persistent=True)
            for example, _, snapshots in pending:
                case_id = example["id"]
                block = originals[case_id]
                scalar = scores[case_id]
                scalar_norm = normalized_scores(example, {"scores": scalar})
                block_norm = normalized_scores(example, block)
                for mode in snapshots:
                    winner = max(range(len(scalar_norm[mode])), key=scalar_norm[mode].__getitem__)
                    token_errors = [
                        abs(a - b)
                        for scalar_choice, block_choice in zip(scalar[mode], block["scores"][mode])
                        for a, b in zip(scalar_choice["token_logprobs"], block_choice["token_logprobs"])
                    ]
                    results[case_id][mode] = {
                        "block_winner": block["predictions"][mode], "scalar_winner": winner,
                        "gold": example["gold"],
                        "block_norm_scores": block_norm[mode], "scalar_norm_scores": scalar_norm[mode],
                        "max_token_logprob_error": max(token_errors), "scalar_scores": scalar[mode],
                        "native_log": native_log.name, "seconds_for_process": elapsed,
                    }
                save_json(audit_dir / case_id / "result.json", results[case_id])

    summary = {"checked_cases": len(results),
               "checked_modes": sum(len(case) for case in results.values()),
               "winner_changes": [
                   {"id": case_id, "mode": mode, "block": item["block_winner"],
                    "scalar": item["scalar_winner"], "gold": item["gold"]}
                   for case_id, modes in results.items() for mode, item in modes.items()
                   if item["block_winner"] != item["scalar_winner"]
               ],
               "max_token_logprob_error": max(
                   (item["max_token_logprob_error"] for modes in results.values()
                    for item in modes.values()), default=None)}
    save_json(report_path, summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "manifest", "results"):
        parser.add_argument(name, type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
