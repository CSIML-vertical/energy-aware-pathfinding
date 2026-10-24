import unittest
from road_network import generate_road_network
from estimator import RoadQualityEstimator

class TestRoadQualityEstimator(unittest.TestCase):

    def setUp(self):
        # Create a controlled network for testing[cite: 3]
        self.network = generate_road_network(num_nodes=20, num_segments=30, seed=42)
        self.estimator = RoadQualityEstimator(self.network, update_rate=0.2, tolerance=1e-4)

    def test_initialization(self):
        """Test that initial estimates are extracted correctly."""
        self.assertEqual(len(self.estimator.estimates), 30)
        
        # Verify values are between 0 and 1
        for val in self.estimator.estimates.values():
            self.assertTrue(0.0 <= val <= 1.0)

    def test_quality_bounds(self):
        """Test that updates strictly clamp between 0.0 and 1.0."""
        # Grab two valid segment IDs from our controlled network
        seg_ids = list(self.estimator.estimates.keys())
        seg_1, seg_2 = seg_ids[0], seg_ids[1]

        # Pass extreme observations that violate physical bounds
        extreme_batch = {
            seg_1: 5.0,  # Way too high
            seg_2: -2.0  # Way too low
        }
        
        self.estimator.update_batch(extreme_batch)
        
        # The update rate is 0.2. 
        # If seg_1 started at 0.5, formula gives: 0.8*0.5 + 0.2*5.0 = 1.4 -> clamped to 1.0
        self.assertLessEqual(self.estimator.estimates[seg_1], 1.0)
        self.assertGreaterEqual(self.estimator.estimates[seg_2], 0.0)

    def test_convergence_loop(self):
        """Test that repeated updates eventually trigger the convergence threshold."""
        # Create a static batch of target observations
        target_observations = {
            seg_id: 0.75 for seg_id in self.estimator.estimates.keys()
        }
        
        iterations = 0
        max_iterations = 100
        converged = False
        
        while iterations < max_iterations:
            max_delta = self.estimator.update_batch(target_observations)
            iterations += 1
            
            if max_delta < self.estimator.tolerance:
                converged = True
                break
                
        self.assertTrue(converged, "Estimator failed to converge within max iterations.")
        self.assertGreater(iterations, 1, "Convergence should take more than 1 iteration.")

if __name__ == '__main__':
    unittest.main()