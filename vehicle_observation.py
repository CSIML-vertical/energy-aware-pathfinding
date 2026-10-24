"""
Vehicle Observation Simulator (Bhavesh)
Generates simulated vehicles traversing road segments and produces noisy,
calibrated roughness-related observations plus travel time and energy.

Depends on: road_network.py (Raghavam), Python stdlib; pandas optional.
True roughness is accessed ONLY via network.get_simulation_truth().
"""
from dataclasses import dataclass, asdict
from random import Random
from typing import Dict, List, Optional, Tuple
import math

VEHICLE_TYPES: Dict[str, dict] = {
    "compact": {"mass_kg": 1100, "base_energy_kwh_per_km": 0.12, "weight": 0.30},
    "sedan":   {"mass_kg": 1500, "base_energy_kwh_per_km": 0.15, "weight": 0.35},
    "suv":     {"mass_kg": 2000, "base_energy_kwh_per_km": 0.20, "weight": 0.25},
    "van":     {"mass_kg": 2600, "base_energy_kwh_per_km": 0.25, "weight": 0.10},
}


@dataclass
class ObservationConfig:
    gain_std: float = 0.10           # vehicle-to-vehicle gain error (paper: params vary <= ~10%)
    bias_std: float = 0.03           # vehicle-to-vehicle residual calibration bias
    noise_std: float = 0.05          # mean per-observation measurement noise
    missing_prob: float = 0.0        # probability a response is missing (NaN)
    speed_roughness_slowdown: float = 0.35   # fractional speed drop at roughness = 1
    speed_noise_std: float = 0.05    # relative speed noise
    speed_exponent: float = 0.5      # speed effect on raw response (removed by calibration)
    min_speed_kmph: float = 5.0
    energy_roughness_coeff: float = 0.5      # extra energy fraction at roughness = 1
    energy_noise_frac: float = 0.03

    def __post_init__(self):
        for name in ("gain_std", "bias_std", "noise_std", "speed_noise_std",
                     "speed_exponent", "energy_roughness_coeff", "energy_noise_frac"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 <= self.missing_prob <= 1.0:
            raise ValueError("missing_prob must be in [0, 1].")
        if not 0.0 <= self.speed_roughness_slowdown < 1.0:
            raise ValueError("speed_roughness_slowdown must be in [0, 1).")
        if self.min_speed_kmph <= 0:
            raise ValueError("min_speed_kmph must be positive.")


@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_type: str
    mass_kg: float
    base_energy_kwh_per_km: float
    gain: float                 # multiplicative calibration error (ideal = 1)
    bias: float                 # additive calibration error (ideal = 0)
    noise_std: float            # this vehicle's measurement noise
    driver_speed_factor: float  # fraction of speed limit this driver typically drives


@dataclass
class VehicleObservation:
    observation_id: str
    vehicle_id: str
    segment_id: str
    calibrated_response: float  # roughness-related response in [0,1]; NaN if missing
    speed_kmph: float
    travel_time_min: float
    energy_kwh: float


def generate_vehicles(num_vehicles: int, config: Optional[ObservationConfig] = None,
                      seed: int = 0) -> List[Vehicle]:
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be at least 1.")
    config = config or ObservationConfig()
    rng = Random(seed)
    names = list(VEHICLE_TYPES)
    weights = [VEHICLE_TYPES[n]["weight"] for n in names]
    vehicles = []
    for i in range(1, num_vehicles + 1):
        vtype = rng.choices(names, weights=weights, k=1)[0]
        spec = VEHICLE_TYPES[vtype]
        vehicles.append(Vehicle(
            vehicle_id=f"V{i:05d}",
            vehicle_type=vtype,
            mass_kg=spec["mass_kg"],
            base_energy_kwh_per_km=spec["base_energy_kwh_per_km"],
            gain=max(0.5, rng.gauss(1.0, config.gain_std)),
            bias=rng.gauss(0.0, config.bias_std),
            noise_std=config.noise_std * rng.uniform(0.7, 1.3),
            driver_speed_factor=min(1.1, max(0.6, rng.gauss(0.9, 0.08))),
        ))
    return vehicles


def _observe(network, segment_id: str, vehicle: Vehicle, rng: Random,
             config: ObservationConfig, obs_id: str) -> VehicleObservation:
    ctx = network.get_observation_context(segment_id)          # no truth here
    true_r = network.get_simulation_truth(segment_id)          # simulator-only
    limit = ctx["speed_limit_kmph"]
    length = ctx["length_km"]

    # Actual speed: rougher road -> slower
    speed = (limit * vehicle.driver_speed_factor
             * (1.0 - config.speed_roughness_slowdown * true_r)
             * (1.0 + rng.gauss(0.0, config.speed_noise_std)))
    speed = max(config.min_speed_kmph, speed)

    # Raw response depends on speed; calibration divides the speed effect out
    speed_factor = (speed / limit) ** config.speed_exponent
    raw = vehicle.gain * true_r * speed_factor + vehicle.bias + rng.gauss(0.0, vehicle.noise_std)
    calibrated = min(1.0, max(0.0, raw / speed_factor))
    if rng.random() < config.missing_prob:
        calibrated = float("nan")

    travel_time_min = length / speed * 60.0
    speed_term = 0.7 + 0.3 * (speed / 50.0) ** 2
    energy = (length * vehicle.base_energy_kwh_per_km
              * (1.0 + config.energy_roughness_coeff * true_r) * speed_term
              * max(0.1, 1.0 + rng.gauss(0.0, config.energy_noise_frac)))

    return VehicleObservation(obs_id, vehicle.vehicle_id, segment_id,
                              calibrated, speed, travel_time_min, energy)


def generate_route(network, rng: Random, num_segments: int) -> List[str]:
    """Random walk on the network; avoids immediately reusing the last segment."""
    node = rng.choice(sorted(network.nodes))
    route, prev_seg = [], None
    for _ in range(num_segments):
        neighbors = network.get_neighbors(node)
        if not neighbors:
            break
        options = [n for n in neighbors if n[1] != prev_seg] or neighbors
        node, seg = rng.choice(options)
        route.append(seg)
        prev_seg = seg
    return route


def simulate_observations(network, vehicles: List[Vehicle], trips_per_vehicle: int = 3,
                          segments_per_trip: Tuple[int, int] = (3, 8),
                          config: Optional[ObservationConfig] = None,
                          seed: int = 0) -> List[VehicleObservation]:
    """Each vehicle makes several trips (random walks); one observation per traversed segment."""
    config = config or ObservationConfig()
    lo, hi = segments_per_trip
    if trips_per_vehicle < 1 or lo < 1 or hi < lo:
        raise ValueError("Invalid trips_per_vehicle or segments_per_trip.")
    rng = Random(seed)
    observations, counter = [], 1
    for vehicle in vehicles:
        for _ in range(trips_per_vehicle):
            for seg_id in generate_route(network, rng, rng.randint(lo, hi)):
                observations.append(_observe(network, seg_id, vehicle, rng, config,
                                             f"O{counter:07d}"))
                counter += 1
    return observations


def simulate_segment_observations(network, segment_id: str, vehicles: List[Vehicle],
                                  num_observations: int,
                                  config: Optional[ObservationConfig] = None,
                                  seed: int = 0) -> List[VehicleObservation]:
    """Controlled generator: many observations of ONE segment (for testing/experiments)."""
    config = config or ObservationConfig()
    if num_observations < 1:
        raise ValueError("num_observations must be at least 1.")
    rng = Random(seed)
    return [_observe(network, segment_id, rng.choice(vehicles), rng, config, f"O{i:07d}")
            for i in range(1, num_observations + 1)]


def observations_to_records(observations: List[VehicleObservation]) -> List[dict]:
    return [asdict(o) for o in observations]


def observations_to_dataframe(observations: List[VehicleObservation]):
    import pandas as pd
    return pd.DataFrame(observations_to_records(observations))


def vehicles_to_dataframe(vehicles: List[Vehicle]):
    import pandas as pd
    return pd.DataFrame([asdict(v) for v in vehicles])
