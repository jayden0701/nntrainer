# SPDX-License-Identifier: Apache-2.0
"""Numerical regressions for CacheBridge's constrained affine update."""

import unittest
from types import SimpleNamespace

import numpy as np

from cachebridge_calibrate import energy_metric, sampling_plan, solve_correction, value_features


class ReadoutCalibrationTest(unittest.TestCase):
    def test_unconstrained_matches_ridge(self):
        gram = np.array([[2., 1.], [1., 4.]])
        cross = np.array([[1., .7], [2., -.8]])
        metric = np.array([[3., .4], [.4, 2.]])
        delta, energy = solve_correction(gram, cross, metric, 100.)
        np.testing.assert_allclose(delta, np.linalg.solve(gram + .03 * np.eye(2), cross))
        self.assertAlmostEqual(energy, np.sum(delta * (metric @ delta)))

    def test_constrained_solve_respects_energy_and_optimality(self):
        gram = np.array([[1., -.5], [-.5, 2.]])
        cross = np.array([[3., 1.], [-2., -4.]])
        metric = np.array([[2., 1.], [1., 2.]])
        limit = .05
        delta, energy = solve_correction(gram, cross, metric, limit)
        self.assertAlmostEqual(energy, limit)
        self.assertAlmostEqual(np.sum(delta * (metric @ delta)), limit)
        gradient = (gram + .015 * np.eye(2)) @ delta - cross
        direction = metric @ delta
        multiplier = -np.vdot(direction, gradient) / np.vdot(direction, direction)
        self.assertGreater(multiplier, 0.)
        np.testing.assert_allclose(gradient + multiplier * direction, 0., atol=1e-11)
        self.assertLess(np.sum(delta * ((gram + .015 * np.eye(2)) @ delta - 2 * cross)), 0.)

    def test_insufficient_budget_keeps_initial_map(self):
        update, energy = solve_correction(np.eye(2), np.ones((2, 1)), np.eye(2), 0.)
        np.testing.assert_array_equal(update, np.zeros((2, 1)))
        self.assertEqual(energy, 0.)

    def test_features_and_energy_use_head_major_selected_layers(self):
        value = np.arange(36, dtype=np.float32).reshape(3, 3, 2, 2)
        spec = SimpleNamespace(kv_heads=2, head_dim=2)
        snap = SimpleNamespace(spec=spec, tokens=3, plane=lambda _: value)
        pair = SimpleNamespace(source=snap, target=snap)
        features = value_features(snap, [2, 0], [1])
        np.testing.assert_array_equal(features[0, 1], np.r_[value[2, 1, 1], value[0, 1, 1]])
        samples = sampling_plan([pair], 2, 5, 2)
        np.testing.assert_array_equal(samples[0][1], [0, 2])
        weights = np.tile(np.eye(2), (2, 1, 1))
        metrics, budgets = energy_metric(samples, [1], 1, weights, np.zeros((2, 2)))
        np.testing.assert_array_equal(budgets, [0., 0.])
        np.testing.assert_allclose(metrics[:, -1, -1], [1.0001, 1.0001])


if __name__ == "__main__":
    unittest.main()
