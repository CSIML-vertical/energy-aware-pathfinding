import pytest
from parameter_tuning import run_single_trial, get_best_params, aggregate_observations
from road_network import generate_road_network
from vehicle_observation import generate_vehicles, simulate_observations, VehicleObservation

def test_aggregate_observations():
    dummy_obs = [
        VehicleObservation("O1", "V1", "R1", 0.5, 50, 10, 1.0),
        VehicleObservation("O2", "V2", "R1", 0.7, 50, 10, 1.0),
        VehicleObservation("O3", "V3", "R1", float('nan'), 50, 10, 1.0), 
        VehicleObservation("O4", "V4", "R2", 0.9, 50, 10, 1.0)
    ]
    
    agg = aggregate_observations(dummy_obs)
    
    assert "R1" in agg
    assert agg["R1"][0] == pytest.approx(0.6) # (0.5 + 0.7) / 2
    assert agg["R1"][1] == 2 # count should be 2
    assert "R2" in agg

def test_run_single_trial():
    net = generate_road_network(num_nodes=10, num_segments=15, seed=1)
    cars = generate_vehicles(num_vehicles=10, seed=1)
    obs = simulate_observations(net, cars, trips_per_vehicle=2, seed=1)
    
    agg_data = aggregate_observations(obs)
    
    res = run_single_trial(net, agg_data, ur=0.1)
    
    assert "iterations" in res
    assert "mae" in res
    assert res["mae"] >= 0.0

def test_select_best_parameters():
    # mocking results (removed prior_strength)
    mock_results = [
        {"update_rate": 0.1, "iterations": 10, "mae": 0.15, "converged": True},
        {"update_rate": 0.2, "iterations": 15, "mae": 0.15, "converged": True}, # tie mae, worse iters
        {"update_rate": 0.5, "iterations": 8, "mae": 0.20, "converged": True},  # worse mae
        {"update_rate": 0.1, "iterations": 50, "mae": 0.10, "converged": False} # didn't converge, ignored
    ]
    
    best = get_best_params(mock_results)
    
    # should pick the first one
    assert best["update_rate"] == 0.1
    assert best["iterations"] == 10