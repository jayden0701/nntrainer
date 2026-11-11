#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare actual KV16 JSONL diagnostics and optional HTP session results.

FNV1a64 checkpoint hashes are diagnostic comparisons, not a cryptographic byte
proof. The default full scope remains INCOMPLETE because these traces do not
observe DSP seed bytes/scales, decode state or logits. --scope checkpoint can
PASS only the explicitly named checkpoint comparison.
"""

import argparse
import json
from pathlib import Path
import re


SCHEMA = "nntrainer.kv16.checkpoint-trace.v1"
CALIBRATION_WORDS = ("scale_k", "scale_v", "q_enc", "out_enc", "kmax", "vmax", "floor_k", "floor_v")
UNOBSERVED = ["seed INT8 bytes", "seed scales", "DSP decode state", "raw logits", "hardware execution provenance"]


def load_trace(path):
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return result


def _uint(value):
    return type(value) is int and 0 <= value <= 0xffffffff


def _words(value):
    return isinstance(value, list) and all(_uint(word) for word in value)


def validate_record(record):
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("missing or unsupported trace schema")
    if record.get("event") not in ("capture", "restore"):
        raise ValueError("unknown checkpoint event")
    for name in ("position", "from", "plane", "chunk"):
        if not _uint(record.get(name)):
            raise ValueError("missing or invalid " + name)
    if not record["position"] or record["from"] >= record["position"] or not record["plane"] or not record["chunk"]:
        raise ValueError("invalid original-call bounds")
    if record.get("element_bytes") != 2 or type(record.get("element_bytes")) is not int:
        raise ValueError("expected FP16 checkpoint storage")
    if not isinstance(record.get("model_tag"), str) or not record["model_tag"]:
        raise ValueError("missing model tag")
    layers, calibration = record.get("layers"), record.get("calibration")
    if not isinstance(layers, list) or not layers or not isinstance(calibration, list) or len(layers) != len(calibration):
        raise ValueError("missing layer/calibration observations")
    for layer in layers:
        if not isinstance(layer, dict) or not _uint(layer.get("base")) or layer["base"] > record["position"] or not _uint(layer.get("width")) or not layer["width"]:
            raise ValueError("invalid layer geometry")
        for name in ("k_fnv1a64", "v_fnv1a64"):
            if not isinstance(layer.get(name), str) or not re.fullmatch(r"[0-9a-f]{16}", layer[name]):
                raise ValueError("missing FNV1a64 observation")
        if not all(_words(layer.get(name)) for name in ("dropped_k_bits", "dropped_v_bits")):
            raise ValueError("missing dropped-amax bit observations")
    for state in calibration:
        if not isinstance(state, dict) or type(state.get("calibrated")) is not bool or not all(_words(state.get(name)) for name in CALIBRATION_WORDS):
            raise ValueError("missing calibration bit observations")
    codec = record.get("codec")
    if not isinstance(codec, dict) or not all(type(codec.get(name)) is int and codec[name] >= 0 for name in ("compressed_pages", "raw_pages", "raw_bytes", "stored_bytes")):
        raise ValueError("missing codec counters")
    return {name: record[name] for name in ("event", "position", "from", "plane", "chunk", "element_bytes", "model_tag", "layers", "calibration")}


def session_observations(report):
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("sessions checker did not report passed=true")
    fields = ("reference", "switched", "prefix", "prefix_reference", "flash_reference", "flash")
    result = []

    def collect(value, path):
        if isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")
        elif isinstance(value, dict) and "ids" in value:
            ids, position = value.get("ids"), value.get("position")
            if not isinstance(ids, list) or not ids or not all(_uint(token) for token in ids) or not _uint(position):
                raise ValueError("missing token/position result at " + path)
            result.append((path, ids, position))
        else:
            raise ValueError("missing sessions observation at " + path)

    for name in fields:
        if name not in report:
            raise ValueError("missing sessions section " + name)
        before = len(result)
        collect(report[name], name)
        if len(result) == before:
            raise ValueError("empty sessions section " + name)
    return result


def seed_input_observations(records):
    if not isinstance(records, list) or not records:
        raise ValueError("seed input traces must contain observed records")
    result = []
    for record in records:
        if not isinstance(record, dict) or record.get("schema") != "nntrainer.kv16.seed-input-trace.v1":
            raise ValueError("missing seed-input-trace.v1 schema")
        if not isinstance(record.get("layer"), str) or not record["layer"]:
            raise ValueError("missing seed layer identity")
        for name in ("position", "base", "rows", "width"):
            if not _uint(record.get(name)):
                raise ValueError("missing seed geometry " + name)
        if record["base"] > record["position"] or not record["rows"] or not record["width"]:
            raise ValueError("invalid seed row bounds")
        for name in ("k_element_bytes", "v_element_bytes"):
            if type(record.get(name)) is not int or record[name] not in (2, 4):
                raise ValueError("unsupported seed input element size")
        for name in ("k_fnv1a64", "v_fnv1a64", "floor_k_fnv1a64", "floor_v_fnv1a64"):
            if not isinstance(record.get(name), str) or not re.fullmatch(r"[0-9a-f]{16}", record[name]):
                raise ValueError("missing seed FNV1a64 observation")
        result.append(record)
    return result


def compare(left, right, scope="full", baseline_results=None, candidate_results=None,
            baseline_seed_trace=None, candidate_seed_trace=None):
    if scope not in ("checkpoint", "full"):
        raise ValueError("invalid scope")
    checks = []

    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": detail})

    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        add("trace", "INCOMPLETE", "both traces must contain checkpoint observations")
    else:
        add("event_count", "PASS" if len(left) == len(right) else "FAIL")
        capture_count = restore_count = compressed_count = 0
        for index, (baseline, candidate) in enumerate(zip(left, right)):
            try:
                expected = validate_record(baseline)
                actual = validate_record(candidate)
            except ValueError as error:
                add(f"event[{index}]", "INCOMPLETE", str(error))
                continue
            capture_count += candidate["event"] == "capture"
            restore_count += candidate["event"] == "restore"
            if baseline["codec"]["compressed_pages"]:
                add(f"event[{index}].baseline_codec", "FAIL", "baseline unexpectedly enabled compression")
            compressed_count += candidate["codec"]["compressed_pages"]
            for name in expected:
                add(f"event[{index}].{name}", "PASS" if expected[name] == actual[name] else "FAIL")
            stored, raw = candidate["codec"]["stored_bytes"], candidate["codec"]["raw_bytes"]
            add(f"event[{index}].storage", "PASS" if stored <= raw else "FAIL", "complete stored bytes must not exceed raw")
        add("capture_exercised", "PASS" if capture_count else "INCOMPLETE")
        add("restore_exercised", "PASS" if restore_count else "INCOMPLETE")
        add("compressed_page_exercised", "PASS" if compressed_count else "INCOMPLETE")
    if baseline_results is not None or candidate_results is not None:
        try:
            expected = session_observations(baseline_results)
            actual = session_observations(candidate_results)
        except ValueError as error:
            add("tokens_positions", "INCOMPLETE", str(error))
        else:
            add("tokens_positions", "PASS" if expected == actual else "FAIL")
    else:
        add("tokens_positions", "UNOBSERVED", "supply both sessions JSON files to compare output IDs/positions")
    if baseline_seed_trace is not None or candidate_seed_trace is not None:
        try:
            expected = seed_input_observations(baseline_seed_trace)
            actual = seed_input_observations(candidate_seed_trace)
        except ValueError as error:
            add("seed_inputs_before_rpc", "INCOMPLETE", str(error))
        else:
            add("seed_inputs_before_rpc", "PASS" if expected == actual else "FAIL", "input diagnostic; not DSP INT8 state readback")
    else:
        add("seed_inputs_before_rpc", "UNOBSERVED", "no seed-input trace supplied")
    statuses = {check["status"] for check in checks}
    checkpoint_status = "FAIL" if "FAIL" in statuses else "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS"
    status = checkpoint_status if scope == "checkpoint" or checkpoint_status == "FAIL" else "INCOMPLETE"
    return {"schema": "nntrainer.kv16.trace-comparison.v1", "scope": scope,
            "status": status, "checkpoint_status": checkpoint_status, "checks": checks,
            "hash_algorithm": "FNV1a64 diagnostic", "byte_comparison_proven": False,
            "hardware_validated": False, "performance_verified": False,
            "unobserved": UNOBSERVED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--baseline-seed-trace", type=Path)
    parser.add_argument("--candidate-seed-trace", type=Path)
    parser.add_argument("--scope", choices=("checkpoint", "full"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        left, right = load_trace(args.baseline), load_trace(args.candidate)
        read = lambda path: json.loads(path.read_text(encoding="utf-8-sig")) if path else None
        report = compare(left, right, args.scope, read(args.baseline_results), read(args.candidate_results),
                         load_trace(args.baseline_seed_trace) if args.baseline_seed_trace else None,
                         load_trace(args.candidate_seed_trace) if args.candidate_seed_trace else None)
    except (ValueError, OSError) as error:
        report = {"schema": "nntrainer.kv16.trace-comparison.v1", "scope": args.scope, "status": "INCOMPLETE", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']} ({args.scope}): {args.output}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
