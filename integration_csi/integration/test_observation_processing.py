"""Unit tests for Observation Processing module (Parth)."""
import unittest
from dataclasses import dataclass
from observation_processing import ObservationAggregator, aggregate_observations


@dataclass
class MockObs:
    segment_id: str
    calibrated_response: float
    speed_kmph: float = 50.0
    energy_kwh: float = 1.0


class TestObservationProcessing(unittest.TestCase):
    def test_aggregation_and_missing_values(self):
        agg = ObservationAggregator()
        agg.add_observation(MockObs("R001", 0.4))
        agg.add_observation(MockObs("R001", 0.6))
        agg.add_observation(MockObs("R001", float("nan")))  # Should be counted as missing
        agg.add_observation(MockObs("R002", 0.8))

        summary1 = agg.get_segment_summary("R001")
        self.assertEqual(summary1.sample_count, 2)
        self.assertEqual(summary1.missing_count, 1)
        self.assertAlmostEqual(summary1.mean_response, 0.5)

        summary2 = agg.get_segment_summary("R002")
        self.assertEqual(summary2.sample_count, 1)
        self.assertAlmostEqual(summary2.mean_response, 0.8)

    def test_confidence_aware_averaging(self):
        agg = ObservationAggregator()
        # R001 has 1 observation of 0.8; prior is 0.2
        agg.add_observation(MockObs("R001", 0.8))
        # R002 has 10 observations of 0.8; prior is 0.2
        for _ in range(10):
            agg.add_observation(MockObs("R002", 0.8))

        priors = {"R001": 0.2, "R002": 0.2, "R003": 0.5}
        estimates = agg.compute_confidence_aware_estimates(priors, prior_strength=2.0)

        # R001 with 1 obs: (1 * 0.8 + 2 * 0.2) / 3 = 1.2 / 3 = 0.40
        self.assertAlmostEqual(estimates["R001"], 0.40)

        # R002 with 10 obs: (10 * 0.8 + 2 * 0.2) / 12 = 8.4 / 12 = 0.70
        self.assertAlmostEqual(estimates["R002"], 0.70)

        # R003 with 0 obs: should equal prior (0.50)
        self.assertAlmostEqual(estimates["R003"], 0.50)

    def test_legacy_aggregate_observations(self):
        obs = [MockObs("R001", 0.3), MockObs("R001", 0.5)]
        res = aggregate_observations(obs)
        self.assertIn("R001", res)
        self.assertAlmostEqual(res["R001"][0], 0.4)
        self.assertEqual(res["R001"][1], 2)


if __name__ == "__main__":
    unittest.main()
