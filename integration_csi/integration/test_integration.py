"""Automated end-to-end integration test suite."""
import unittest
from road_network import generate_road_network
import vehicle_observation as vo
from estimator import RoadQualityEstimator
from energy_model import EnergyPredictor
from energy_astar import EnergyAStarRouter


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.net = generate_road_network(num_nodes=30, num_segments=50, seed=7)
        self.cfg = vo.ObservationConfig()
        self.vehicles = vo.generate_vehicles(20, self.cfg, seed=7)

    def test_pipeline_execution(self):
        # 1. Simulate vehicle traversals
        obs = vo.simulate_observations(self.net, self.vehicles, trips_per_vehicle=2, seed=7)
        self.assertGreater(len(obs), 0)

        # 2. Train ML energy predictor
        ml_model = EnergyPredictor(random_state=7)
        ml_model.train_from_simulations(self.net, obs)
        self.assertTrue(ml_model.is_fitted)

        # 3. Refine roughness with iterative estimator
        estimator = RoadQualityEstimator(self.net, update_rate=0.5, tolerance=1e-3)
        sample_batch = {s_id: 0.4 for s_id in self.net.segments}
        delta = estimator.update_batch(sample_batch)
        self.assertLessEqual(delta, 1.0)

        # 4. Multi-Objective A* routing (safe node extraction)
        nodes = list(self.net.nodes.keys()) if isinstance(self.net.nodes, dict) else list(self.net.nodes)
        router = EnergyAStarRouter(self.net, ml_model)
        route = router.find_path(nodes[0], nodes[10], estimator.estimates, lambda_weight=0.5)

        self.assertIsNotNone(route)
        self.assertGreater(route["total_time_min"], 0.0)
        self.assertGreater(route["total_energy_kwh"], 0.0)
        self.assertEqual(route["nodes"][0], nodes[0])
        self.assertEqual(route["nodes"][-1], nodes[10])


if __name__ == "__main__":
    unittest.main()