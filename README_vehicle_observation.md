# Vehicle Observation Simulator (Bhavesh)

## Purpose
Simulate vehicles traversing road segments and generate noisy, calibrated roughness-related
observations, plus travel time and energy, for the Iterative Road Quality Estimation project.

## Assumptions
- Road quality is a normalized simulation score in [0,1], NOT measured IRI.
- `calibrated_response` is a calibrated roughness-related response (same 0-1 scale) with
  vehicle gain/bias error and noise. Raw energy alone cannot uniquely identify road quality, so
  `energy_kwh` and `travel_time_min` are side-channel outputs; estimators should use `calibrated_response`.
- Values are clipped to [0,1] (slight bias near the bounds).
- Energy/time constants are illustrative simulation values.

## Dependencies
- `road_network.py` (Raghavam): uses `RoadNetwork.get_observation_context`, `get_simulation_truth`,
  `get_neighbors`, `nodes`, `segments`. True roughness is read only inside the simulator.
- Python stdlib; pandas needed only for the `*_to_dataframe` helpers.

## Inputs
- `network`: RoadNetwork. `vehicles`: list of `Vehicle`. `ObservationConfig` (noise_std, gain_std,
  bias_std, missing_prob, speed/energy parameters). `seed` for reproducibility.

## Main functions
- `generate_vehicles(num_vehicles, config, seed) -> List[Vehicle]`
- `simulate_observations(network, vehicles, trips_per_vehicle, segments_per_trip, config, seed) -> List[VehicleObservation]`
- `simulate_segment_observations(network, segment_id, vehicles, num_observations, config, seed)` (controlled testing)
- `observations_to_records / observations_to_dataframe / vehicles_to_dataframe`

## Output: observation fields (observations.csv)
observation_id, vehicle_id, segment_id, calibrated_response (NaN if missing), speed_kmph,
travel_time_min, energy_kwh

## Vehicle fields (vehicles.csv)
vehicle_id, vehicle_type, mass_kg, base_energy_kwh_per_km, gain, bias, noise_std, driver_speed_factor

## Tests
`python -m unittest -v test_vehicle_observation` (results in test_results.txt)

## Experiments
1. Truth sweep (experiment1_sweep.csv/.png) 2. Noise vs accuracy (experiment2_noise.csv)
3. Energy vs calibrated response (experiment3_energy_vs_response.csv)
