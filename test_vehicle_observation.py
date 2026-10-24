import unittest, math, statistics
from road_network import RoadNetwork, RoadSegment, generate_road_network
import vehicle_observation as vo


def chain_network(roughness_values, road_class="main_road", length=5.0):
    net = RoadNetwork()
    for i in range(len(roughness_values) + 1):
        net.add_node(f"N{i}")
    for i, r in enumerate(roughness_values):
        net.add_segment(RoadSegment(f"S{i}", f"N{i}", f"N{i+1}", road_class,
                                    length, 0.4, r, 50))
    return net


class TestVehicleObservation(unittest.TestCase):
    def setUp(self):
        self.net = generate_road_network(30, 50, seed=7)
        self.cfg = vo.ObservationConfig()
        self.vehicles = vo.generate_vehicles(50, self.cfg, seed=1)

    def test_reproducible(self):
        a = vo.simulate_observations(self.net, self.vehicles, 2, (3, 5), self.cfg, seed=3)
        b = vo.simulate_observations(self.net, self.vehicles, 2, (3, 5), self.cfg, seed=3)
        self.assertEqual(vo.observations_to_records(a), vo.observations_to_records(b))

    def test_no_true_roughness_leak(self):
        obs = vo.simulate_observations(self.net, self.vehicles, 1, (3, 3), self.cfg, seed=3)
        keys = vo.observations_to_records(obs)[0].keys()
        self.assertNotIn("true_roughness", keys)

    def test_valid_ids_and_range(self):
        obs = vo.simulate_observations(self.net, self.vehicles, 2, (3, 6), self.cfg, seed=3)
        vids = {v.vehicle_id for v in self.vehicles}
        for o in obs:
            self.assertIn(o.segment_id, self.net.segments)
            self.assertIn(o.vehicle_id, vids)
            self.assertTrue(0.0 <= o.calibrated_response <= 1.0)
            self.assertGreater(o.travel_time_min, 0)
            self.assertGreater(o.energy_kwh, 0)

    def test_routes_are_connected_walks(self):
        from random import Random
        route = vo.generate_route(self.net, Random(1), 6)
        for a, b in zip(route, route[1:]):
            sa, sb = self.net.get_segment(a), self.net.get_segment(b)
            self.assertTrue({sa.start_node, sa.end_node} & {sb.start_node, sb.end_node})

    def test_noise_free_equals_truth(self):
        cfg = vo.ObservationConfig(gain_std=0, bias_std=0, noise_std=0)
        net = chain_network([0.3, 0.7])
        veh = vo.generate_vehicles(5, cfg, seed=1)
        for sid, truth in (("S0", 0.3), ("S1", 0.7)):
            for o in vo.simulate_segment_observations(net, sid, veh, 20, cfg, seed=1):
                self.assertAlmostEqual(o.calibrated_response, truth, places=9)

    def _means(self, field):
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        net = chain_network(levels)
        veh = vo.generate_vehicles(100, self.cfg, seed=5)
        return [statistics.mean(getattr(o, field) for o in
                vo.simulate_segment_observations(net, f"S{i}", veh, 4000, self.cfg, seed=9))
                for i in range(len(levels))]

    def test_response_increases_with_roughness(self):
        m = self._means("calibrated_response")
        self.assertTrue(all(b > a for a, b in zip(m, m[1:])), m)

    def test_time_increases_with_roughness(self):
        m = self._means("travel_time_min")
        self.assertTrue(all(b > a for a, b in zip(m, m[1:])), m)

    def test_energy_increases_with_roughness(self):
        m = self._means("energy_kwh")
        self.assertTrue(all(b > a for a, b in zip(m, m[1:])), m)

    def test_more_noise_more_variance(self):
        net = chain_network([0.5])
        sds = []
        for ns in (0.02, 0.10):
            cfg = vo.ObservationConfig(noise_std=ns)
            veh = vo.generate_vehicles(100, cfg, seed=2)
            r = [o.calibrated_response for o in
                 vo.simulate_segment_observations(net, "S0", veh, 3000, cfg, seed=4)]
            sds.append(statistics.pstdev(r))
        self.assertLess(sds[0], sds[1])

    def test_vehicle_variation(self):
        gains = [v.gain for v in vo.generate_vehicles(200, self.cfg, seed=1)]
        self.assertGreater(statistics.pstdev(gains), 0.0)
        gains0 = [v.gain for v in vo.generate_vehicles(50, vo.ObservationConfig(gain_std=0), seed=1)]
        self.assertTrue(all(g == 1.0 for g in gains0))

    def test_missing_measurements(self):
        net = chain_network([0.5])
        veh = vo.generate_vehicles(20, self.cfg, seed=1)
        none = vo.simulate_segment_observations(net, "S0", veh, 500, vo.ObservationConfig(missing_prob=0.0), seed=1)
        allm = vo.simulate_segment_observations(net, "S0", veh, 500, vo.ObservationConfig(missing_prob=1.0), seed=1)
        some = vo.simulate_segment_observations(net, "S0", veh, 4000, vo.ObservationConfig(missing_prob=0.2), seed=1)
        self.assertFalse(any(math.isnan(o.calibrated_response) for o in none))
        self.assertTrue(all(math.isnan(o.calibrated_response) for o in allm))
        frac = sum(math.isnan(o.calibrated_response) for o in some) / len(some)
        self.assertAlmostEqual(frac, 0.2, delta=0.03)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): vo.ObservationConfig(noise_std=-1)
        with self.assertRaises(ValueError): vo.ObservationConfig(missing_prob=1.5)
        with self.assertRaises(ValueError): vo.generate_vehicles(0)
        with self.assertRaises(ValueError):
            vo.simulate_observations(self.net, self.vehicles, 0)


if __name__ == "__main__":
    unittest.main()
