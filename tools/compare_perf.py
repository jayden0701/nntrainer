#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_latest_history_entry(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    latest = None
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and "benchmarks" in record:
                latest = record
    return latest


def index_benchmarks(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for benchmark in summary.get("benchmarks", []):
        benchmark_id = benchmark.get("id")
        if benchmark_id:
            indexed[benchmark_id] = benchmark
    return indexed


def compare_summaries(
    baseline_summary: Dict[str, Any], current_summary: Dict[str, Any]
) -> Dict[str, Any]:
    baseline_cases = index_benchmarks(baseline_summary)
    current_cases = index_benchmarks(current_summary)

    shared_ids = sorted(set(baseline_cases) & set(current_cases))
    comparisons: List[Dict[str, Any]] = []

    for benchmark_id in shared_ids:
        baseline_case = baseline_cases[benchmark_id]
        current_case = current_cases[benchmark_id]

        baseline_median = float(baseline_case["median"])
        current_median = float(current_case["median"])
        delta_pct = None
        if baseline_median != 0.0:
            delta_pct = (
                (current_median - baseline_median) / baseline_median
            ) * 100.0

        comparisons.append(
            {
                "id": benchmark_id,
                "kernel": current_case.get("kernel"),
                "baseline_median": baseline_median,
                "current_median": current_median,
                "delta_pct": delta_pct,
            }
        )

    return {
        "baseline_cases": len(baseline_cases),
        "current_cases": len(current_cases),
        "comparisons": comparisons,
        "missing_in_baseline": sorted(set(current_cases) - set(baseline_cases)),
        "missing_in_current": sorted(set(baseline_cases) - set(current_cases)),
    }


def print_report(report: Dict[str, Any], baseline_label: str, current_label: str) -> None:
    comparisons = report["comparisons"]
    if not comparisons:
        print(f"No overlapping benchmark ids between {baseline_label} and {current_label}.")
        return

    print(f"Baseline: {baseline_label}")
    print(f"Current : {current_label}")
    print("id\tbaseline_ns\tcurrent_ns\tdelta_pct")
    for item in comparisons:
        delta_pct = item["delta_pct"]
        delta_str = "n/a" if delta_pct is None else f"{delta_pct:.2f}"
        print(
            f'{item["id"]}\t{item["baseline_median"]:.2f}\t'
            f'{item["current_median"]:.2f}\t{delta_str}'
        )

    if report["missing_in_baseline"]:
        print("Missing in baseline:", ", ".join(report["missing_in_baseline"]))
    if report["missing_in_current"]:
        print("Missing in current:", ", ".join(report["missing_in_current"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare nntrainer perf summaries.")
    parser.add_argument("--current", required=True, help="Current summary JSON")
    parser.add_argument("--baseline", help="Baseline summary JSON")
    parser.add_argument(
        "--history",
        help="perf_history.jsonl to use when --baseline is not provided",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    current_path = pathlib.Path(args.current)
    current_summary = load_json(current_path)

    baseline_summary = None
    baseline_label = ""

    if args.baseline:
        baseline_path = pathlib.Path(args.baseline)
        baseline_summary = load_json(baseline_path)
        baseline_label = str(baseline_path)
    elif args.history:
        history_path = pathlib.Path(args.history)
        baseline_summary = load_latest_history_entry(history_path)
        baseline_label = str(history_path)

    if baseline_summary is None:
        print("No baseline summary found. Wrote current summary only.")
        if args.output:
            output_path = pathlib.Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "baseline": None,
                        "current": str(current_path),
                        "comparisons": [],
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
        return 0

    report = compare_summaries(baseline_summary, current_summary)
    print_report(report, baseline_label, str(current_path))

    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "baseline": baseline_label,
                    "current": str(current_path),
                    **report,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
