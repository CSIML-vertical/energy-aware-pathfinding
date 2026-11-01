"""Unit tests for Road Network module (Raghavam)."""
import unittest
from road_network import generate_road_network, RoadNetwork, RoadSegment, ROAD_CLASSES


class TestRoadNetwork(unittest.TestCase):
    def setUp(self):
        self.net = generate_road_network(num_nodes=20, num_segments=30, seed=42)

    def test_network_structure(self):
        self.assertEqual(len(self.net.nodes), 20)
        self.assertEqual(len(self.net.segments), 30)
        self.assertTrue(self.net.is_connected())
        self.assertTrue(self.net.validate())

    def test_segment_properties_and_bounds(self):
        for seg in self.net.segments.values():
            self.assertIn(seg.road_class, ROAD_CLASSES)
            self.assertGreater(seg.length_km, 0)
            self.assertGreater(seg.speed_limit_kmph, 0)
            self.assertGreaterEqual(seg.initial_roughness, 0.0)
            self.assertLessEqual(seg.initial_roughness, 1.0)
            self.assertGreaterEqual(seg.true_roughness, 0.0)
            self.assertLessEqual(seg.true_roughness, 1.0)

    def test_observation_context_hides_ground_truth(self):
        for seg_id in list(self.net.segments.keys())[:5]:
            ctx = self.net.get_observation_context(seg_id)
            self.assertNotIn("true_roughness", ctx)
            self.assertIn("segment_id", ctx)
            self.assertIn("speed_limit_kmph", ctx)
            self.assertIn("length_km", ctx)


if __name__ == "__main__":
    unittest.main()
