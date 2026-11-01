"""Unit tests for Baseline Estimators module (Karan)."""
import unittest
import numpy as np
from road_network import ROAD_CLASSES
import baseline_estimators as be


class TestBaselineEstimators(unittest.TestCase):
    def test_fixed_road_class(self):
        classes = ["highway", "main_road", "small_road"]
        preds = be.predict_fixed_road_class(classes, ROAD_CLASSES)
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], ROAD_CLASSES["highway"]["initial_estimate"])
        self.assertAlmostEqual(preds[1], ROAD_CLASSES["main_road"]["initial_estimate"])
        self.assertAlmostEqual(preds[2], ROAD_CLASSES["small_road"]["initial_estimate"])

    def test_mean_observation_response_fallback(self):
        obs_means = [0.25, np.nan, 0.75]
        initial = [0.15, 0.40, 0.60]
        preds = be.predict_mean_observation_response(obs_means, initial)
        self.assertAlmostEqual(preds[0], 0.25)
        self.assertAlmostEqual(preds[1], 0.40)  # Fell back to initial
        self.assertAlmostEqual(preds[2], 0.75)


if __name__ == "__main__":
    unittest.main()
