"""Regression checks for budget selection, log attribution and model fitting."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import kv_budget_predict as tool


class BudgetPolicyTest(unittest.TestCase):
    # Deliberately non-monotonic; memory high-water differs from retention.
    points = [
        {"budget": 0, "kv_peak": 120, "p50": 12, "p95": 23},
        {"budget": 20, "kv_peak": 135, "p50": 8, "p95": 11},
        {"budget": 40, "kv_peak": 145, "p50": 9, "p95": 13},
        {"budget": 60, "kv_peak": 160, "p50": 6, "p95": 9},
    ]

    def test_objectives(self):
        self.assertEqual(tool.solve(self.points, "p50", "budget", 40, None)["budget"], 20)
        self.assertEqual(tool.solve(self.points, "p95", "kv_peak", None, 12)["budget"], 20)
        self.assertIsNone(tool.solve(self.points, "p95", "kv_peak", 130, 12))
        self.assertIsNone(tool.solve(self.points, "p50", "budget", None, 1))

    def test_profile_not_transferable(self):
        args = ["solve", "--example", "--rows", "630", "--cap", "52MiB", "--device", "other"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(args)

    def test_same_constraint_tracks_each_devices_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.json"
            for device, latencies, expected in (("A", [12, 8, 9, 6], 20),
                                                ("B", [16, 13, 7, 6], 40)):
                points = [dict(p, p50=ms) for p, ms in zip(self.points, latencies)]
                path.write_text(json.dumps(dict(device=device, rows_cal=300, points=points)))
                for constraint in (["--cap", "40"], ["--target", "8"]):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = tool.main(["solve", "--profile", str(path), "--device", device,
                                          "--rows", "300", *constraint])
                    self.assertEqual(code, 0)
                    self.assertIn(f"NNTR_KV_RESIDENT_BYTES={expected} ", output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(["solve", "--profile", str(path), "--rows", "301", "--cap", "40"])

    def test_unmeasured_tail(self):
        with self.assertRaises(ValueError):
            tool.measured_points(tool.DEFAULT_PROFILE, "p95", "budget")

    def test_fit_requires_full_coverage_and_fit_error_is_per_point(self):
        data = [dict(k=k, p50=34 + (28-k)*0.5, zero_reads=(k == 28))
                for k in (0, 9, 16, 20, 28)]
        self.assertFalse(tool.fit_model(data[:-1], 32, 28))
        profile = tool.fit_model(data, 32, 28)
        self.assertLess(profile["streaming_fit_rmse_ms"], 1e-6)
        self.assertAlmostEqual(tool.model_decode(0, 28, profile), 48)

    def test_multiple_requests_and_zero_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def request(rows, ms):
                return "[REQUEST] " + json.dumps(dict(cached_before=0, cached_after=rows,
                    prefill_tokens=2, output_ids=[1, 2], decode_ms=ms)) + "\n"
            def report(b, retained, k):
                return (f"[kv-fsu] lookahead=1 direct_io=0 read_bytes=100 "
                        f"retained_bytes={retained} retained_owners={k} "
                        f"budget_bytes={b} kv_payload_peak_bytes=32768 "
                        "page_bytes=4096 layers=3 uniform_row_bytes=256\n")
            tail = "[kv-decode] steady_tokens=3 p95_ms=17\n"
            (root/"resident_r1.log").write_text(request(3, 1) + tail + request(6, 10))
            (root/"b0_r1.log").write_text(report(0, 0, 0) + request(3, 1) +
                                          report(0, 0, 0) + tail + request(6, 14))
            p = tool.calibrate(root, 6, None, None, {})
            self.assertEqual(p["points"][0], dict(budget=0, k=0, runs=1, p50=14,
                                p95=17, kv_peak=32768, zero_reads=True))
            self.assertEqual((p["owners"], p["row_bytes"]), (3, 256))
            self.assertNotIn("t_service", p)
            with self.assertRaises(ValueError):
                tool.calibrate(root, 6, None, None, {"b0": "1MiB"})
            mixed_policy = root / "b0_r2.log"
            mixed_policy.write_text(report(0, 0, 0).strip() + " owner_lookahead=1\n" +
                                    tail + request(6, 14))
            with self.assertRaisesRegex(ValueError, "lookahead"):
                tool.calibrate(root, 6, None, None, {})
            mixed_policy.unlink()
            # A sparse sampling of retention budgets must still use one
            # scheduler and one total cap. Reject a broken runtime peak too.
            base = report(0, 0, 0).strip()
            for extra, reason in [
                ("capped=1 total_budget_bytes=32768 rolling=1", "separate sweeps"),
                ("capped=1 total_budget_bytes=100 rolling=0", "over-budget"),
            ]:
                log = root / "b0_r1.log"
                log.write_text(base + " " + extra + "\n" + tail + request(6, 14))
                if reason == "separate sweeps":
                    mixed_policy.write_text(base + "\n" + tail + request(6, 14))
                with self.assertRaisesRegex(ValueError, reason):
                    tool.calibrate(root, 6, None, None, {})
                mixed_policy.unlink(missing_ok=True)
            log.write_text(base + "\n" + tail + request(6, 14))
            (root/"different.log").write_text(report(0, 0, 0) +
                                              request(6, 14).replace("[1, 2]", "[2, 1]"))
            with self.assertRaises(ValueError):
                tool.calibrate(root, 6, None, None, {})

    def test_budget_units_and_alignment(self):
        self.assertEqual(tool.parse_budget("1.5MiB"), 1572864)
        self.assertEqual(tool.parse_budget("1MB"), 1000000)
        self.assertEqual(tool.owner_size(3, 1024, 4096), 8192)
        for invalid in ("-1", "1Ki", "1.2", "inf"):
            with self.assertRaises(ValueError):
                tool.parse_budget(invalid)


if __name__ == "__main__":
    unittest.main()
