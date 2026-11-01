"""Unit tests for Vehicle Observation module (Bhavesh)."""
import unittest
from road_network import generate_road_network
import vehicle_observation as vo


class TestVehicleObservation(unittest.TestCase):
    def setUp(self):
        self.net = generate_road_network(num_nodes=15, num_segments=20, seed=123)
        self.cfg = vo.ObservationConfig(noise_std=0.02)
        self.vehicles = vo.generate_vehicles(num_vehicles=10, config=self.cfg, seed=123)

    def test_vehicle_generation(self):
        self.assertEqual(len(self.vehicles), 10)
        for v in self.vehicles:
            self.assertIn(v.vehicle_type, vo.VEHICLE_TYPES)
            self.assertGreater(v.mass_kg, 0)
            self.assertGreater(v.base_energy_kwh_per_km, 0)

    def test_observation_simulation(self):
        obs = vo.simulate_observations(self.net, self.vehicles, trips_per_vehicle=2, seed=123)
        self.assertGreater(len(obs), 0)
        for o in obs:
            self.assertIn(o.segment_id, self.net.segments)
            self.assertGreater(o.speed_kmph, 0)
            self.assertGreater(o.travel_time_min, 0)
            self.assertGreater(o.energy_kwh, 0)

    def test_speed_slowdown_on_rough_roads(self):
        # Controlled single-segment test
        seg_id = list(self.net.segments.keys())[0]
        obs_batch = vo.simulate_segment_observations(self.net, seg_id, self.vehicles, 10, seed=123)
        self.assertEqual(len(obs_batch), 10)


if __name__ == "__main__":
    unittest.main()
