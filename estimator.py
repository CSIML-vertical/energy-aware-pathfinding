"""
Module: estimator.py
Author: Shreya
Project Division: Iterative Road Quality Estimation

Description:
    Provides the core engine for dynamically updating road quality (roughness)
    estimates across successive simulation iterations based on aggregated vehicle 
    observations.

Dependencies:
    - road_network.py (Raghavam's module for RoadNetwork & RoadSegment structures)[cite: 3]

Inputs:
    - road_network: An instance of RoadNetwork containing initial segment states[cite: 3].
    - observations: A dictionary mapping segment_ids (str) to aggregated roughness values (float).
    - update_rate (float): Learning rate (alpha) for the exponential moving average.
    - tolerance (float): Convergence threshold measured against max delta.

Outputs:
    - max_delta (float): The largest absolute update step across all segments in a batch.
    - self.estimates (dict): State dictionary mapping segment_id to current estimated roughness[cite: 1].
"""

from road_network import generate_road_network


class RoadQualityEstimator:
    """
    Stateful estimator that iteratively refines segment-level road roughness
    using a weighted update rule[cite: 1]. Maintains quality bounds and checks
    for convergence across successive observation batches[cite: 1].
    """

    def __init__(self, road_network, update_rate: float = 0.1, tolerance: float = 1e-4):
        """
        Initialize the estimator and load baseline road quality estimates[cite: 1, 3].

        Args:
            road_network (RoadNetwork): The road network graph object[cite: 3].
            update_rate (float, optional): The weighting factor (alpha in [0, 1]) 
                applied to incoming observations. Defaults to 0.1[cite: 1].
            tolerance (float, optional): Maximum change threshold to trigger 
                convergence. Defaults to 1e-4[cite: 1].
        """
        self.update_rate = update_rate
        self.tolerance = tolerance
        
        # Extract initial estimates from Raghavam's network[cite: 3]
        self.estimates = {
            seg_id: segment.initial_roughness
            for seg_id, segment in road_network.segments.items()
        }

    def update_batch(self, observations: dict) -> float:
        """
        Process a batch of aggregated vehicle observations and update stored estimates[cite: 1].

        Applies the Exponential Moving Average (EMA) equation:
            new_estimate = (1 - alpha) * current_estimate + alpha * observed_value[cite: 1]

        Enforces quality bounds [0.0, 1.0] and tracks the maximum delta[cite: 1, 3].

        Args:
            observations (dict): Dictionary mapping segment IDs (str) to 
                aggregated observed roughness values (float)[cite: 1].

        Returns:
            float: The maximum absolute shift (max_delta) among all updated segments[cite: 1].
        """
        max_delta = 0.0

        for seg_id, observed_value in observations.items():
            if seg_id not in self.estimates:
                continue

            current_estimate = self.estimates[seg_id]
            
            # 1. Weighted update equation (Exponential Moving Average)[cite: 1]
            new_estimate = (1 - self.update_rate) * current_estimate + (self.update_rate * observed_value)
            
            # 2. Apply quality bounds [0.0, 1.0][cite: 1, 3]
            new_estimate = max(0.0, min(1.0, new_estimate))
            
            # 3. Compute delta for convergence checking[cite: 1]
            delta = abs(new_estimate - current_estimate)
            if delta > max_delta:
                max_delta = delta
                
            # 4. Save state for subsequent iterations[cite: 1]
            self.estimates[seg_id] = new_estimate

        return max_delta


if __name__ == "__main__":
    # Test script for standalone execution
    test_network = generate_road_network(num_nodes=50, num_segments=80, seed=42)
    estimator = RoadQualityEstimator(test_network)
    
    print(f"Estimator initialized with {len(estimator.estimates)} segments.")
    
    mock_batch = {"R00001": 0.85, "R00002": 0.20}
    delta = estimator.update_batch(mock_batch)
    print(f"Test batch processed. Maximum step delta: {delta:.6f}")