"""Unit tests for Iterative Road Quality Estimator (Shreya)."""
import unittest
from road_network import generate_road_network
from estimator import RoadQualityEstimator


class TestRoadQualityEstimator(unittest.TestCase):
    def setUp(self):
        self.net = generate_road_network(num_nodes=10, num_segments=15, seed=42)
        self.estimator = RoadQualityEstimator(self.net, update_rate=0.5, tolerance=1e-4)

    def test_initial_estimates(self):
        self.assertEqual(len(self.estimator.estimates), 15)
        for s_id, val in self.estimator.estimates.items():
            self.assertEqual(val, self.net.segments[s_id].initial_roughness)

    def test_ema_update_and_bounds(self):
        s_id = list(self.net.segments.keys())[0]
        initial = self.estimator.estimates[s_id]

        # Update towards 1.0 with alpha=0.5
        delta = self.estimator.update_batch({s_id: 1.0})
        expected = 0.5 * initial + 0.5 * 1.0
        self.assertAlmostEqual(self.estimator.estimates[s_id], expected)
        self.assertAlmostEqual(delta, abs(expected - initial))

        # Check bounds clipping [0, 1]
        self.estimator.update_batch({s_id: 2.5})
        self.assertLessEqual(self.estimator.estimates[s_id], 1.0)

    def test_convergence(self):
        s_id = list(self.net.segments.keys())[0]
        batch = {s_id: 0.5}
        iters = 0
        delta = 1.0
        while iters < 30 and delta >= self.estimator.tolerance:
            delta = self.estimator.update_batch(batch)
            iters += 1
        self.assertLess(delta, self.estimator.tolerance)


if __name__ == "__main__":
    unittest.main()
