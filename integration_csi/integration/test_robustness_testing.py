"""Unit tests for Robustness Testing module (Ashmita)."""
import unittest
from road_network import generate_road_network
import vehicle_observation as vo
from robustness_testing import (
    compute_mae,
    compute_rmse,
    evaluate_sparse_observations,
    identify_failure_conditions,
)


class TestRobustnessTesting(unittest.TestCase):
    def test_metrics(self):
        preds = {"A": 0.5, "B": 0.8}
        truth = {"A": 0.4, "B": 0.6}
        # Errors: |0.5 - 0.4| = 0.1, |0.8 - 0.6| = 0.2
        mae = compute_mae(preds, truth)
        self.assertAlmostEqual(mae, 0.15)

        # Sq errors: 0.01, 0.04 -> mean = 0.025 -> sqrt = 0.15811
        rmse = compute_rmse(preds, truth)
        self.assertAlmostEqual(rmse, (0.025)**0.5, places=4)

    def test_failure_conditions_structure(self):
        net = generate_road_network(num_nodes=15, num_segments=20, seed=42)
        records = identify_failure_conditions(net, seed=42)
        self.assertGreater(len(records), 0)
        for r in records:
            self.assertIn("scenario", r)
            self.assertIn("estimator_mae", r)
            self.assertIn("status", r)


if __name__ == "__main__":
    unittest.main()
