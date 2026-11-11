#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trace parser tests use the exact runtime JSON field names."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import compare_checkpoint_traces as trace


class TraceTests(unittest.TestCase):
    def setUp(self):
        record = {"schema": trace.SCHEMA, "event": "capture", "position": 4096,
                  "from": 0, "plane": 4096, "chunk": 4096, "element_bytes": 2,
                  "model_tag": "same-model-pack", "layers": [
                      {"base": 0, "width": 1024, "k_fnv1a64": "0123456789abcdef",
                       "v_fnv1a64": "fedcba9876543210", "dropped_k_bits": [], "dropped_v_bits": []}],
                  "calibration": [{"calibrated": True, **{name: [0x3f800000] for name in trace.CALIBRATION_WORDS}}],
                  "codec": {"compressed_pages": 0, "raw_pages": 64, "raw_bytes": 65536, "stored_bytes": 65536}}
        restore = dict(record, event="restore")
        self.left = copy.deepcopy([record, restore])
        self.right = copy.deepcopy(self.left)
        self.right[0]["codec"].update(compressed_pages=64, raw_pages=0, stored_bytes=48000)
        self.right[1]["codec"].update(compressed_pages=64, raw_pages=0, stored_bytes=48000)

    def test_matching_checkpoint_diagnostics_pass_only_explicit_scope(self):
        report = trace.compare(self.left, self.right, scope="checkpoint")
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["byte_comparison_proven"])
        self.assertFalse(report["hardware_validated"])
        report = trace.compare(self.left, self.right)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["checkpoint_status"], "PASS")

    def test_geometry_calibration_callplan_hash_mismatch_fails(self):
        for field, value in (("position", 4097), ("from", 1), ("chunk", 2048), ("model_tag", "wrong-pack")):
            right = copy.deepcopy(self.right)
            right[0][field] = value
            self.assertEqual(trace.compare(self.left, right)["status"], "FAIL")
        right = copy.deepcopy(self.right)
        right[0]["calibration"][0]["scale_k"][0] ^= 1
        self.assertEqual(trace.compare(self.left, right)["status"], "FAIL")
        right = copy.deepcopy(self.right)
        right[0]["layers"][0]["k_fnv1a64"] = "0" * 16
        self.assertEqual(trace.compare(self.left, right)["status"], "FAIL")

    def test_missing_restore_compression_or_state_is_incomplete(self):
        self.assertEqual(trace.compare([], [], scope="checkpoint")["status"], "INCOMPLETE")
        self.assertEqual(trace.compare(self.left[:1], self.right[:1], scope="checkpoint")["status"], "INCOMPLETE")
        right = copy.deepcopy(self.right)
        for item in right:
            item["codec"]["compressed_pages"] = 0
        self.assertEqual(trace.compare(self.left, right, scope="checkpoint")["status"], "INCOMPLETE")
        right = copy.deepcopy(self.right)
        del right[0]["calibration"][0]["q_enc"]
        self.assertEqual(trace.compare(self.left, right, scope="checkpoint")["status"], "INCOMPLETE")

    def test_fp32_bits_are_compared_without_numeric_tolerance(self):
        right = copy.deepcopy(self.right)
        right[1]["layers"][0]["dropped_k_bits"] = [0x80000000]
        self.left[1]["layers"][0]["dropped_k_bits"] = [0]
        self.assertEqual(trace.compare(self.left, right, scope="checkpoint")["status"], "FAIL")

    def test_existing_sessions_outputs_compare_tokens_and_positions_only(self):
        result = {"ids": [2, 19, 1], "position": 4100}
        sessions = {"passed": True, "reference": [[result]], "switched": [result],
                    "prefix": [result], "prefix_reference": [result], "flash_reference": result, "flash": result}
        report = trace.compare(self.left, self.right, "checkpoint", sessions, copy.deepcopy(sessions))
        self.assertEqual(report["status"], "PASS")
        different = copy.deepcopy(sessions)
        different["flash"]["ids"] = [2, 20, 1]
        self.assertEqual(trace.compare(self.left, self.right, "checkpoint", sessions, different)["status"], "FAIL")
        self.assertEqual(trace.compare(self.left, self.right, "full", sessions, sessions)["status"], "INCOMPLETE")

    def test_seed_inputs_require_both_nonempty_exact_streams_and_do_not_prove_dsp_output(self):
        seed = {"schema": "nntrainer.kv16.seed-input-trace.v1", "layer": "layer5_mha",
                "position": 4096, "base": 0, "rows": 4096, "width": 1024,
                "k_element_bytes": 2, "v_element_bytes": 2,
                **{name: "0123456789abcdef" for name in ("k_fnv1a64", "v_fnv1a64", "floor_k_fnv1a64", "floor_v_fnv1a64")}}
        args = dict(baseline_seed_trace=[seed], candidate_seed_trace=[copy.deepcopy(seed)])
        self.assertEqual(trace.compare(self.left, self.right, "checkpoint", **args)["status"], "PASS")
        self.assertEqual(trace.compare(self.left, self.right, "full", **args)["status"], "INCOMPLETE")
        args["candidate_seed_trace"][0]["floor_k_fnv1a64"] = "0" * 16
        self.assertEqual(trace.compare(self.left, self.right, "checkpoint", **args)["status"], "FAIL")
        for absent in (None, []):
            args["candidate_seed_trace"] = absent
            self.assertEqual(trace.compare(self.left, self.right, "checkpoint", **args)["status"], "INCOMPLETE")

    def test_cli_defaults_to_incomplete_full_and_rejects_truncated_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right, output = root / "baseline.jsonl", root / "codec.jsonl", root / "comparison.json"
            left.write_text("\n".join(map(json.dumps, self.left)) + "\n", encoding="utf-8")
            right.write_text("\n".join(map(json.dumps, self.right)) + "\n", encoding="utf-8")
            command = [sys.executable, str(Path(trace.__file__)), str(left), str(right), "--output", str(output)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())["checkpoint_status"], "PASS")
            result = subprocess.run(command + ["--scope", "checkpoint"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            right.write_text('{"schema":', encoding="utf-8")
            result = subprocess.run(command + ["--scope", "checkpoint"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
