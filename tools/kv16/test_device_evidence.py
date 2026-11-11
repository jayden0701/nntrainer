#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for incomplete or misleading device validation evidence."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import compare_device_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        digest = hashlib.sha256(b"actual observed data").hexdigest()
        self.left = {
            "schema": evidence.SCHEMA, "mode": "baseline", "hardware_validated": True,
            "identity": {name: digest if name.endswith("sha256") else "same-native-build-and-device"
                         for name in evidence.IDENTITY},
            "events": [{"id": "request-0/call-0", "position": 4096, "token_ids": [2, 901, 1],
                        **{name: digest for name in evidence.CHECKPOINT_HASHES + evidence.FULL_HASHES}}],
        }
        self.right = copy.deepcopy(self.left)
        self.right.update(mode="codec", codec={"compressed_pages": 3, "raw_bytes": 65536, "stored_bytes": 49000})

    def test_full_matching_evidence_passes(self):
        report = evidence.compare(self.left, self.right)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["performance_verified"])

    def test_matching_tokens_do_not_replace_seed_or_logits_evidence(self):
        for field in evidence.FULL_HASHES:
            right = copy.deepcopy(self.right)
            del right["events"][0][field]
            self.assertEqual(evidence.compare(self.left, right)["status"], "INCOMPLETE", field)
        self.assertEqual(evidence.compare(self.left, {"passed": True})["status"], "INCOMPLETE")

    def test_hardware_empty_events_and_no_codec_use_are_incomplete(self):
        mutations = ({"hardware_validated": False}, {"events": []}, {"codec": {"compressed_pages": 0, "raw_bytes": 3, "stored_bytes": 3}})
        for mutation in mutations:
            right = dict(self.right, **mutation)
            self.assertEqual(evidence.compare(self.left, right)["status"], "INCOMPLETE")

    def test_mismatches_and_nonbeneficial_claimed_compression_fail(self):
        for field in ("k_sha256", "calibration_sha256", "seed_scales_sha256", "logits_sha256"):
            right = copy.deepcopy(self.right)
            right["events"][0][field] = "0" * 64
            self.assertEqual(evidence.compare(self.left, right)["status"], "FAIL", field)
        right = copy.deepcopy(self.right)
        right["events"][0]["token_ids"] = [2, 1]
        self.assertEqual(evidence.compare(self.left, right)["status"], "FAIL")
        right["codec"]["stored_bytes"] = right["codec"]["raw_bytes"]
        self.assertEqual(evidence.compare(self.left, right)["status"], "FAIL")

    def test_checkpoint_scope_never_claims_seed_or_inference_validation(self):
        for field in evidence.FULL_HASHES + ("token_ids",):
            self.left["events"][0].pop(field)
            self.right["events"][0].pop(field)
        report = evidence.compare(self.left, self.right, scope="checkpoint")
        self.assertEqual(report["status"], "PASS")
        self.assertIn("seed_kv_sha256", report["unverified_by_this_scope"])
        self.assertEqual(evidence.compare(self.left, self.right)["status"], "INCOMPLETE")

    def test_explicit_binary_artifacts_and_wrong_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "k.bin").write_bytes(b"observed KV bytes\x00\xff")
            digest = hashlib.sha256((root / "k.bin").read_bytes()).hexdigest()
            self.left["events"][0]["k_sha256"] = digest
            self.right["events"][0]["k_sha256"] = {
                "file": "k.bin", "bytes": len((root / "k.bin").read_bytes())}
            self.assertEqual(evidence.compare(self.left, self.right, root, root)["status"], "PASS")
            self.right["events"][0]["k_sha256"]["bytes"] += 1
            self.assertEqual(evidence.compare(self.left, self.right, root, root)["status"], "INCOMPLETE")

    def test_empty_artifacts_cannot_establish_seed_or_logits_equality(self):
        empty = hashlib.sha256(b"").hexdigest()
        self.left["events"][0]["logits_sha256"] = empty
        self.right["events"][0]["logits_sha256"] = empty
        self.assertEqual(evidence.compare(self.left, self.right)["status"], "INCOMPLETE")

    def test_duplicate_event_ids_and_malformed_values_are_not_passes(self):
        self.left["events"].append(copy.deepcopy(self.left["events"][0]))
        self.right["events"].append(copy.deepcopy(self.right["events"][0]))
        self.assertEqual(evidence.compare(self.left, self.right)["status"], "INCOMPLETE")
        self.right["events"][0]["position"] = True
        self.right["events"][0]["token_ids"] = []
        self.assertEqual(evidence.compare(self.left, self.right)["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
