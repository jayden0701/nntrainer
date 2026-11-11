#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare explicit device evidence; missing observations never count as PASS.

Exit 0: every required observation matched in the requested scope.
Exit 1: a mismatch or an invalid claimed codec result was observed.
Exit 2: required observations were absent, malformed, or not from hardware.
This tool does not collect observations or infer seed/logit equality from tokens.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "nntrainer.kv16.device-evidence.v1"
IDENTITY = (
    "model_sha256", "tokenizer_sha256", "numerical_config_sha256",
    "arm_sha256", "dsp_sha256", "code_revision", "device_fingerprint",
)
CHECKPOINT_HASHES = ("call_plan_sha256", "calibration_sha256", "k_sha256", "v_sha256")
FULL_HASHES = ("seed_kv_sha256", "seed_scales_sha256", "decode_state_sha256", "logits_sha256")


def sha256_observation(value, directory):
    """Accept a SHA256 or hash the explicitly named local evidence file."""
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        if value.lower() == hashlib.sha256(b"").hexdigest():
            raise ValueError("empty evidence cannot establish equality")
        return value.lower()
    if isinstance(value, dict) and isinstance(value.get("file"), str) and value["file"]:
        path = directory / value["file"]
        digest = hashlib.sha256()
        count = 0
        try:
            with path.open("rb") as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(part)
                    count += len(part)
        except OSError as error:
            raise ValueError(f"cannot read evidence file {path}: {error}") from error
        if not count:
            raise ValueError(f"empty evidence file: {path}")
        if "bytes" in value and (type(value["bytes"]) is not int or value["bytes"] != count):
            raise ValueError(f"evidence file size differs: {path}")
        return digest.hexdigest()
    raise ValueError("expected SHA256 or explicit {file, bytes?} evidence")


def compare(baseline, candidate, baseline_directory=Path("."), candidate_directory=Path("."), scope="full"):
    if scope not in ("checkpoint", "full"):
        raise ValueError("scope must be checkpoint or full")
    checks = []

    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": detail})

    def equal(name, left, right, validator=lambda value: value is not None):
        if not validator(left) or not validator(right):
            add(name, "INCOMPLETE", "missing or malformed observation")
        else:
            add(name, "PASS" if left == right else "FAIL", "" if left == right else "observations differ")

    def hashes(name, left, right):
        try:
            lhs = sha256_observation(left, baseline_directory)
            rhs = sha256_observation(right, candidate_directory)
        except ValueError as error:
            add(name, "INCOMPLETE", str(error))
        else:
            equal(name, lhs, rhs)

    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return {"schema": SCHEMA, "scope": scope, "status": "INCOMPLETE", "checks": [
            {"name": "manifest", "status": "INCOMPLETE", "detail": "both manifests must be objects"}]}
    for label, manifest, expected in (("baseline", baseline, "baseline"), ("candidate", candidate, "codec")):
        add(f"{label}.schema", "PASS" if manifest.get("schema") == SCHEMA else "INCOMPLETE", "expected " + SCHEMA)
        add(f"{label}.mode", "PASS" if manifest.get("mode") == expected else "INCOMPLETE", "expected " + expected)
        add(f"{label}.hardware", "PASS" if manifest.get("hardware_validated") is True else "INCOMPLETE", "requires actual device capture")

    left_identity, right_identity = baseline.get("identity"), candidate.get("identity")
    left_identity = left_identity if isinstance(left_identity, dict) else {}
    right_identity = right_identity if isinstance(right_identity, dict) else {}
    for name in IDENTITY:
        lhs, rhs = left_identity.get(name), right_identity.get(name)
        if name.endswith("_sha256"):
            hashes("identity." + name, lhs, rhs)
        else:
            equal("identity." + name, lhs, rhs, lambda value: isinstance(value, str) and bool(value))

    left_events, right_events = baseline.get("events"), candidate.get("events")
    if not isinstance(left_events, list) or not isinstance(right_events, list) or not left_events or not right_events:
        add("events", "INCOMPLETE", "both runs need nonempty observed event arrays")
    else:
        equal("event_count", len(left_events), len(right_events))
        seen = {"baseline": set(), "candidate": set()}
        for index, (lhs, rhs) in enumerate(zip(left_events, right_events)):
            prefix = f"events[{index}]"
            if not isinstance(lhs, dict) or not isinstance(rhs, dict):
                add(prefix, "INCOMPLETE", "event must be an object")
                continue
            equal(prefix + ".id", lhs.get("id"), rhs.get("id"), lambda value: isinstance(value, str) and bool(value))
            for label, event in (("baseline", lhs), ("candidate", rhs)):
                event_id = event.get("id")
                if isinstance(event_id, str):
                    if event_id in seen[label]:
                        add(prefix + "." + label + ".id", "INCOMPLETE", "duplicate event id")
                    seen[label].add(event_id)
            equal(prefix + ".position", lhs.get("position"), rhs.get("position"),
                  lambda value: type(value) is int and value >= 0)
            for name in CHECKPOINT_HASHES + (FULL_HASHES if scope == "full" else ()):
                hashes(prefix + "." + name, lhs.get(name), rhs.get(name))
            if scope == "full":
                equal(prefix + ".token_ids", lhs.get("token_ids"), rhs.get("token_ids"),
                      lambda value: isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value))

    codec = candidate.get("codec")
    if not isinstance(codec, dict):
        add("codec.used", "INCOMPLETE", "no codec accounting")
    else:
        compressed, raw, stored = (codec.get(name) for name in ("compressed_pages", "raw_bytes", "stored_bytes"))
        if type(compressed) is not int or type(raw) is not int or type(stored) is not int or min(compressed, raw, stored) < 0:
            add("codec.used", "INCOMPLETE", "nonnegative integer counters required")
        elif compressed == 0:
            add("codec.used", "INCOMPLETE", "codec mode did not publish any compressed page")
        elif not 0 < stored < raw:
            add("codec.used", "FAIL", "claimed compressed storage does not reduce complete bytes")
        else:
            add("codec.used", "PASS")
    statuses = {item["status"] for item in checks}
    status = "FAIL" if "FAIL" in statuses else "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS"
    return {"schema": SCHEMA, "scope": scope, "status": status, "checks": checks,
            "unverified_by_this_scope": list(FULL_HASHES) + ["token_ids"] if scope == "checkpoint" else [],
            "performance_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scope", choices=("checkpoint", "full"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        left = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
        right = json.loads(args.candidate.read_text(encoding="utf-8-sig"))
        report = compare(left, right, args.baseline.parent, args.candidate.parent, args.scope)
    except (ValueError, OSError) as error:
        report = {"schema": SCHEMA, "scope": args.scope, "status": "INCOMPLETE", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']} ({args.scope}): {args.output}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
