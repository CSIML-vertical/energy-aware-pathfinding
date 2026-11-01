
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class SegmentSummary:
    segment_id: str
    sample_count: int = 0
    missing_count: int = 0
    response_sum: float = 0.0
    response_sq_sum: float = 0.0
    speed_sum: float = 0.0
    energy_sum: float = 0.0

    @property
    def mean_response(self) -> Optional[float]:
        return (self.response_sum / self.sample_count) if self.sample_count > 0 else None

    @property
    def variance_response(self) -> float:
        if self.sample_count < 2:
            return 0.0
        mean = self.mean_response
        var = (self.response_sq_sum / self.sample_count) - (mean * mean)
        return max(0.0, var)

    @property
    def std_response(self) -> float:
        return math.sqrt(self.variance_response)

    @property
    def mean_speed_kmph(self) -> Optional[float]:
        return (self.speed_sum / self.sample_count) if self.sample_count > 0 else None

    @property
    def mean_energy_kwh(self) -> Optional[float]:
        return (self.energy_sum / self.sample_count) if self.sample_count > 0 else None


class ObservationAggregator:
    """
    Stores vehicle observations mapped to road-segment IDs, filters missing values,
    computes segment-level summary statistics, and provides confidence-aware averaging.
    """

    def __init__(self):
        self.segments: Dict[str, SegmentSummary] = {}
        self.raw_observations: Dict[str, List[Any]] = {}

    def add_observation(self, obs) -> None:
        """Add a single VehicleObservation record."""
        seg_id = obs.segment_id
        if seg_id not in self.segments:
            self.segments[seg_id] = SegmentSummary(segment_id=seg_id)
            self.raw_observations[seg_id] = []

        self.raw_observations[seg_id].append(obs)
        summary = self.segments[seg_id]

        response = getattr(obs, "calibrated_response", float("nan"))
        if response is None or math.isnan(response):
            summary.missing_count += 1
            return

        summary.sample_count += 1
        summary.response_sum += response
        summary.response_sq_sum += response * response
        if hasattr(obs, "speed_kmph"):
            summary.speed_sum += obs.speed_kmph
        if hasattr(obs, "energy_kwh"):
            summary.energy_sum += obs.energy_kwh

    def add_observations(self, observations: List[Any]) -> None:
        """Add a batch of VehicleObservation records."""
        for obs in observations:
            self.add_observation(obs)

    def get_sample_count(self, segment_id: str) -> int:
        """Return the number of valid (non-missing) observations for a segment."""
        if segment_id in self.segments:
            return self.segments[segment_id].sample_count
        return 0

    def get_mean_response(self, segment_id: str) -> Optional[float]:
        """Return the sample mean for a segment, or None if no valid observations exist."""
        if segment_id in self.segments:
            return self.segments[segment_id].mean_response
        return None

    def get_segment_summary(self, segment_id: str) -> Optional[SegmentSummary]:
        return self.segments.get(segment_id)

    def compute_confidence_aware_estimates(
        self,
        priors: Dict[str, float],
        prior_strength: float = 2.0,
    ) -> Dict[str, float]:
        """
        Calculates shrinkage / confidence-weighted estimates between the prior
        and the observed sample mean:

            estimate = (N * obs_mean + N0 * prior) / (N + N0)

        When a segment has few or zero observations (N -> 0), the estimate safely
        reverts to the prior. When observations are plentiful (N >> N0), the empirical
        mean dominates.
        """
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")

        estimates = {}
        all_segments = set(priors.keys()) | set(self.segments.keys())

        for seg_id in all_segments:
            prior = priors.get(seg_id, 0.5)
            summary = self.segments.get(seg_id)
            n = summary.sample_count if summary else 0
            obs_mean = summary.mean_response if (summary and summary.mean_response is not None) else prior

            total_weight = n + prior_strength
            if total_weight > 0:
                weighted_val = (n * obs_mean + prior_strength * prior) / total_weight
            else:
                weighted_val = prior

            estimates[seg_id] = max(0.0, min(1.0, weighted_val))

        return estimates

    def to_legacy_dict(self) -> Dict[str, Tuple[float, int]]:
        """
        Exports segment averages as a dict of {segment_id: (mean_response, count)}
        for backwards compatibility with existing pipelines.
        """
        res = {}
        for seg_id, summary in self.segments.items():
            if summary.sample_count > 0:
                res[seg_id] = (summary.mean_response, summary.sample_count)
        return res


def aggregate_observations(observations: List[Any]) -> Dict[str, Tuple[float, int]]:

    aggregator = ObservationAggregator()
    aggregator.add_observations(observations)
    return aggregator.to_legacy_dict()
