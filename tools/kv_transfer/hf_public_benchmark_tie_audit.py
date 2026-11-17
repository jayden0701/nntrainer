#!/usr/bin/env python3
"""Re-score the closest normalized margins from the complete CPU HellaSwag run."""

import sys
from pathlib import Path

import hf_public_benchmark as bench


def margin(row, arm):
    normalized = sorted(
        (total / count for total, count in
         zip(row[f"{arm}_logprobs"], row["ending_tokens"])),
        reverse=True,
    )
    return normalized[0] - normalized[1]


@bench.torch.inference_mode()
def main():
    source_path, target_path, mapper_path, data_path, report_path, run_path = map(
        Path, sys.argv[1:]
    )
    rows = bench.read_json(run_path / "cases.json")
    summary = bench.read_json(run_path / "summary.json")
    public = bench.read_json(report_path)["hellaswag"]
    metadata = bench.read_json(run_path / "metadata.json")
    assert summary["public_sample_complete"]
    assert [row["index"] for row in rows] == public["indices"]
    assert metadata["runner_sha256"] == bench.sha256(Path(bench.__file__))
    bench.torch.set_num_threads(metadata["threads"])
    bench.torch.set_num_interop_threads(1)
    tokenizer = bench.AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    cases = {
        case.index: case for case in bench.load_cases(data_path, public, tokenizer)
    }
    manifest = bench.read_json(mapper_path / "manifest.json")
    source, target = bench.load_model(source_path), bench.load_model(target_path)
    mapper = bench.RidgeMapper(manifest, mapper_path / "mapper-fp32.safetensors")
    # Round up twice the largest mean-token error in the mixed-padding pilot.
    threshold = 0.016
    results = []
    for arm in ("standalone", "mapped"):
        closest = sorted(rows, key=lambda row: margin(row, arm))
        selected = [
            row for rank, row in enumerate(closest)
            if rank < 4 or margin(row, arm) <= threshold
        ]
        for row in selected:
            case = cases[row["index"]]
            arguments = (
                {"source": source, "mapper": mapper}
                if arm == "mapped" and case.prefix_length else {}
            )
            serial = bench.score_serial(case, target, tokenizer.pad_token_id, **arguments)
            batched = row[f"{arm}_logprobs"]
            lengths = row["ending_tokens"]
            serial_choice = bench.prediction(serial, lengths)
            batch_choice = bench.prediction(batched, lengths)
            check = {
                "index": case.index,
                "arm": arm,
                "batch_margin": margin(row, arm),
                "batch_choice": batch_choice,
                "serial_choice": serial_choice,
                "label": case.label,
                "serial_logprobs": serial,
                "max_mean_logprob_error": max(
                    abs(left - right) / length
                    for left, right, length in zip(serial, batched, lengths)
                ),
            }
            results.append(check)
            print(check, flush=True)
    bench.write_json(run_path / "near-ties.json", {
        "margin_threshold": threshold,
        "minimum_per_arm": 4,
        "candidate_passes": len(results) * 4,
        "batch_choice_changes": sum(
            row["batch_choice"] != row["serial_choice"] for row in results
        ),
        "checks": results,
    })


if __name__ == "__main__":
    main()
