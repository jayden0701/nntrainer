"""Focused offline mapper tests with generated affine data."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import read_pairs, split_pairs
from mapper import Geometry, Mapping, apply, compare, fit, select_layers
from pilot import direct_copy
from selection import ProbeMoments
from snapshot import PLANES, ModelSpec, Snapshot, create_snapshot, rotate_keys, write_json


class TransferTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = Path(scratch.name)
        self.rng = np.random.default_rng(713)

    def make_pair(self, index, source, target, layers, layout, weights, bias,
                  token_ids=None, group=None):
        ids = token_ids if token_ids is not None else list(range(index * 100, index * 100 + 25))
        positions = [4 + 3*i for i in range(len(ids))]
        paths = {"source": f"s-{index}", "target": f"t-{index}"}
        if group is not None:
            paths["group"] = group
        with (create_snapshot(self.path / paths["source"], source, ids, positions) as src,
              create_snapshot(self.path / paths["target"], target, ids, positions) as dst):
            for kind_index, kind in enumerate(PLANES):
                data = self.rng.normal(loc=1.3, size=source.shape(len(ids))).astype(np.float32)
                for layer in range(source.layers):
                    src[kind][layer] = (rotate_keys(data[layer], positions, source)
                                        if kind == "keys" else data[layer])
                for layer, selected in enumerate(layers):
                    x = np.moveaxis(data[selected], 1, 0)
                    w, b = weights[kind_index, layer], bias[kind_index, layer]
                    if layout == "all-head":
                        y = x.reshape(len(ids), -1) @ w[0] + b[0]
                        y = y.reshape(len(ids), target.kv_heads, target.head_dim)
                    else:
                        y = np.stack([
                            x[:, :, h, :].reshape(len(ids), -1) @ w[h] + b[h]
                            for h in range(target.kv_heads)
                        ], axis=1)
                    dst[kind][layer] = (rotate_keys(y, positions, target)
                                        if kind == "keys" else y)
        return paths

    def manifest(self, pairs):
        path = self.path / "dataset.json"
        write_json(path, {"version": 1, "pairs": pairs})
        return path

    def test_split_half_rotation_nonconsecutive_positions(self):
        spec = ModelSpec("rope", 1, 1, 6, 10000, 4)
        key = np.array([[[1., 2., 3., 4., 5., 6.]]])
        result = rotate_keys(key, [1], spec)
        c, s = np.cos([1., 0.01]), np.sin([1., 0.01])
        want = [[[c[0] - 3*s[0], 2*c[1] - 4*s[1],
                  3*c[0] + s[0], 4*c[1] + 2*s[1], 5., 6.]]]
        np.testing.assert_allclose(result, want, atol=1e-12)
        keys = self.rng.normal(size=(4, 3, 6)).astype(np.float32)
        positions = [0, 3, 71, 1582]
        rotated = rotate_keys(keys, positions, spec)
        np.testing.assert_allclose(rotate_keys(rotated, positions, spec, inverse=True),
                                   keys, atol=5e-7)

    def test_headwise_identity_cli(self):
        spec = ModelSpec("same", 2, 2, 4, 10000, 4)
        layers = [[0], [1]]
        weights = np.broadcast_to(np.eye(4), (2, 2, 2, 4, 4))
        bias = np.zeros((2, 2, 2, 4))
        rows = [self.make_pair(i, spec, spec, layers, "headwise", weights, bias)
                for i in range(4)]
        manifest = self.manifest(rows)

        def cli(*args):
            command = Path(__file__).with_name("kv_transfer.py")
            output = subprocess.check_output([sys.executable, str(command), *map(str, args)],
                                             text=True)
            return json.loads(output)

        model_path = self.path / "map"
        result = cli("fit", manifest, model_path, "--alpha", "1e-6", "--block-tokens", 7)
        self.assertLess(result["held_out"]["keys"]["rmse"], 1e-6)
        _, val = split_pairs(read_pairs(manifest), 0.2, 0)
        src, ref = val[0].source.path, val[0].target.path
        predicted = self.path / "predicted"
        cli("apply", model_path, src, predicted, "--block-tokens", 8)
        scores = cli("compare", predicted, ref, "--block-tokens", 6)
        self.assertLess(scores["values"]["relative_l2"], 1e-6)
        self.assertEqual(scores["keys"]["elements"], 2 * 25 * 2 * 4)

    def test_all_head_unequal_depths_nonzero_intercept_and_holdout(self):
        source = ModelSpec("source", 3, 2, 4, 10000, 4)
        target = ModelSpec("target", 2, 1, 6, 1000000, 4)
        self.assertEqual(select_layers(3, 2, 2), [[0, 1], [1, 2]])
        layers = [[0, 2], [0, 1]]
        geometry = Geometry(source, target, "all-head", layers)
        weights = self.rng.normal(size=(2, 2, 1, 16, 6))
        bias = self.rng.normal(loc=3, size=(2, 2, 1, 6))
        pairs = [
            self.make_pair(i, source, target, layers, "all-head", weights, bias,
                           group="shared template" if i < 2 else None)
            for i in range(6)
        ]
        manifest = self.manifest(pairs)
        train, val = split_pairs(read_pairs(manifest), 0.25, 123)
        val_groups = {pair.group for pair in val}
        self.assertFalse(val_groups & {pair.group for pair in train})
        self.assertFalse({p.digest for p in train} & {p.digest for p in val})
        reversed_manifest = self.manifest(pairs[::-1])
        train2, val2 = split_pairs(read_pairs(reversed_manifest), 0.25, 123)
        self.assertEqual([p.digest for p in train], [p.digest for p in train2])
        self.assertEqual([p.digest for p in val], [p.digest for p in val2])
        model_path = self.path / "map"
        report = fit(model_path, train, val, geometry, alpha=1e-6,
                     block_size=9, memory_mib=1)
        mapping = Mapping(model_path)
        np.testing.assert_allclose(mapping.weights, weights, atol=2e-6)
        np.testing.assert_allclose(mapping.bias, bias, atol=3e-6)
        self.assertLess(report["held_out"]["keys"]["relative_l2"], 1e-6)
        self.assertLess(report["held_out"]["values"]["relative_l2"], 1e-6)
        output = self.path / "prediction"
        apply(mapping, val[0].source, output, block_size=6, memory_mib=1)
        predicted = Snapshot(output)
        self.assertEqual(predicted.token_ids, val[0].target.token_ids)
        self.assertEqual(predicted.positions, val[0].target.positions)
        self.assertLess(compare(predicted, val[0].target, 8)["keys"]["rmse"], 4e-6)
        with self.assertRaisesRegex(ValueError, "model or shape"):
            apply(mapping, val[0].target, self.path / "wrong",
                  block_size=6, memory_mib=1)
        with self.assertRaisesRegex(ValueError, "token-aligned"):
            compare(predicted, train[0].target, 8)

    def test_ols_probe_rank_deficiency_and_constant_target(self):
        source = ModelSpec("source", 2, 1, 4, 10000, 4)
        target = ModelSpec("target", 2, 1, 2, 10000, 2)
        signal = self.rng.normal(size=37)
        noise = self.rng.normal(size=37)
        x = np.column_stack((signal, signal, signal * 2, np.ones(37),
                             noise, noise * 2, noise, np.ones(37)))
        y = np.column_stack((3 + 2 * signal, np.ones(37),
                             np.zeros(37), np.zeros(37)))
        moments = ProbeMoments(source, target)
        for indexes in np.array_split(np.arange(37), 5):
            moments.update(x[indexes], y[indexes])
        scores, valid = moments.r2()
        expected_noise = np.corrcoef(signal, noise)[0, 1] ** 2
        np.testing.assert_allclose(scores, [[1, expected_noise], [0, 0]], atol=1e-13)
        np.testing.assert_array_equal(valid, [True, False])

    def test_ols_ridge_cli_uses_strided_calibration_only(self):
        source = ModelSpec("source", 3, 2, 4, 10000, 4)
        target = ModelSpec("target", 2, 2, 4, 1000000, 4)
        layers = [[2], [0]]
        # Swap the source heads to check that the production mapper sees all
        # source heads, while the OLS probe still has a same-head signal.
        weights = np.broadcast_to(np.eye(8) + np.roll(np.eye(8), 4, axis=0) * 2,
                                  (2, 2, 1, 8, 8))
        bias = np.zeros((2, 2, 1, 8))
        rows = [
            self.make_pair(i, source, target, layers, "all-head", weights, bias,
                           token_ids=list(range(i * 300, i * 300 + 197 + 3 * i)))
            for i in range(4)
        ]
        manifest = self.manifest(rows)
        train, validation = split_pairs(read_pairs(manifest), 0.25, 0)
        for pair in train:
            for filename in PLANES.values():
                plane = np.memmap(pair.target.path / filename, dtype="<f4", mode="r+",
                                  shape=target.shape(pair.target.tokens))
                for offset in (1, 2, 3):
                    plane[:, offset::4] += 100
                plane.flush()
        for pair in validation:
            for filename in PLANES.values():
                plane = np.memmap(pair.target.path / filename, dtype="<f4", mode="r+",
                                  shape=target.shape(pair.target.tokens))
                plane[:] = 1000
                plane.flush()
        del plane

        def fit_cli(model_path, fraction):
            command = Path(__file__).with_name("kv_transfer.py")
            output = subprocess.check_output([
                sys.executable, str(command), "fit", str(manifest), str(model_path),
                "--layout", "all-head", "--layer-selection", "ols-r2", "--k", "1",
                "--alpha", "1e-6", "--validation-fraction", fraction,
                "--token-stride", "4", "--block-tokens", "11", "--memory-mib", "1",
            ], text=True)
            return json.loads(output)

        model_path = self.path / "strided"
        report = fit_cli(model_path, "0.25")
        self.assertEqual(report["source_layers"], layers)
        self.assertEqual(report["token_stride"], 4)
        self.assertEqual(report["layer_selection"]["method"], "ols-r2")
        self.assertEqual(report["train"]["observations"],
                         sum(len(range(0, p.source.tokens, 4)) for p in train))
        self.assertEqual(report["validation"]["prompt_digests"], [validation[0].digest])
        self.assertGreater(report["held_out"]["values"]["rmse"], 100)
        np.testing.assert_allclose(Mapping(model_path).weights, weights, atol=2e-6)

        calibration_sources = {pair.source.path for pair in train}
        self.manifest([row for row in rows if self.path / row["source"] in calibration_sources])
        report = fit_cli(self.path / "calibration-only", "0")
        self.assertEqual(report["train"]["prompts"], len(train))
        self.assertEqual(report["validation"]["prompts"], 0)
        self.assertIsNone(report["held_out"])
        self.assertEqual(report["source_layers"], layers)

    def test_duplicate_unaligned_and_prefix_pairs(self):
        spec = ModelSpec("small", 1, 1, 2, 10000, 2)
        weights = np.broadcast_to(np.eye(2), (2, 1, 1, 2, 2))
        bias = np.zeros((2, 1, 1, 2))
        rows = [
            self.make_pair(i, spec, spec, [[0]], "headwise", weights, bias,
                           token_ids=ids, group=group)
            for i, (ids, group) in enumerate([
                ([11], "shared"), ([11, 14], "shared"), ([11, 20], "different"),
                ([7, 9], None),
            ])
        ]
        with self.assertRaisesRegex(ValueError, "duplicate token sequence"):
            read_pairs(self.manifest([rows[0], rows[0]]))
        with self.assertRaisesRegex(ValueError, "not token-aligned"):
            read_pairs(self.manifest([{"source": rows[0]["source"],
                                      "target": rows[1]["target"]}, rows[3]]))
        with self.assertRaisesRegex(ValueError, "prefix-related"):
            read_pairs(self.manifest(rows))
        train, val = split_pairs(read_pairs(self.manifest([rows[0], rows[1], rows[3]])),
                                 0.25, 0)
        self.assertEqual({len(train), len(val)}, {1, 2})

    def test_direct_copy_control_preserves_source_planes(self):
        source_model = ModelSpec("source", 1, 1, 2, 10000, 2)
        target_model = ModelSpec("target", 1, 1, 2, 10000, 2)
        weight = np.broadcast_to(np.eye(2), (2, 1, 1, 2, 2))
        bias = np.ones((2, 1, 1, 2))
        row = self.make_pair(0, source_model, target_model, [[0]], "headwise", weight, bias)
        source, target = (Snapshot(self.path / row[label]) for label in ("source", "target"))
        transplant = self.path / "direct-copy"
        self.assertEqual(direct_copy(source, target, transplant), transplant)
        copied = Snapshot(transplant)
        self.assertEqual(copied.spec, target.spec)
        self.assertEqual(copied.token_ids, target.token_ids)
        for plane in PLANES:
            np.testing.assert_array_equal(copied.plane(plane), source.plane(plane))
        self.assertGreater(compare(copied, target, 8)["values"]["rmse"], 0.99)

    def test_snapshot_size_and_same_shape_wrong_model(self):
        spec = ModelSpec("base", 1, 1, 2, 10000, 2)
        other = ModelSpec("other", 1, 1, 2, 10000, 2)
        for model in (spec, other):
            with create_snapshot(self.path / model.model_id, model, [1], [0]):
                pass
        with self.assertRaisesRegex(ValueError, "model or shape"):
            compare(Snapshot(self.path / "base"), Snapshot(self.path / "other"), 1)
        (self.path / "base" / "keys.f32").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "must hold exactly"):
            Snapshot(self.path / "base")


if __name__ == "__main__":
    unittest.main()
