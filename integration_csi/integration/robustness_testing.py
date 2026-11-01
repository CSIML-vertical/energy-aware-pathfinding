"""
Module: robustness_testing.py
Author: Ashmita (Swapped with Shreya)
Project Division: Iterative Road Quality Estimation

Description:
    Comprehensive evaluation and robustness testing module for iterative road-quality
    estimation. Computes accuracy metrics (MAE, RMSE) against hidden ground truth,
    evaluates sensitivity across varying noise levels and observation densities,
    analyzes sparse/few-observation segments, and identifies estimator failure boundaries.

Dependencies:
    - road_network.py (Raghavam)
    - vehicle_observation.py (Bhavesh)
    - observation_processing.py (Parth)
    - estimator.py (Shreya)
    - baseline_estimators.py (Karan)
    - matplotlib, numpy, pandas

Inputs:
    - RoadNetwork instance with hidden ground truth
    - Configurable ObservationConfig parameters (noise_std, missing_prob, gain_std, bias_std)
    - Fleet sizes and iteration counts

Outputs:
    - Quantitative metrics: MAE, RMSE, class-stratified errors
    - Robustness analysis plots: robustness_analysis.png
    - CSV benchmark logs: robustness_results.csv
"""
import math
import csv
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from road_network import generate_road_network, RoadNetwork, ROAD_CLASSES
import vehicle_observation as vo
from observation_processing import ObservationAggregator, aggregate_observations
from estimator import RoadQualityEstimator
import baseline_estimators as be


def compute_mae(predictions: Dict[str, float], ground_truth: Dict[str, float]) -> float:
    """Calculate Mean Absolute Error between predicted estimates and ground truth."""
    common_keys = [k for k in predictions if k in ground_truth]
    if not common_keys:
        return 0.0
    errors = [abs(predictions[k] - ground_truth[k]) for k in common_keys]
    return float(np.mean(errors))


def compute_rmse(predictions: Dict[str, float], ground_truth: Dict[str, float]) -> float:
    """Calculate Root Mean Squared Error between predicted estimates and ground truth."""
    common_keys = [k for k in predictions if k in ground_truth]
    if not common_keys:
        return 0.0
    sq_errors = [(predictions[k] - ground_truth[k]) ** 2 for k in common_keys]
    return float(math.sqrt(np.mean(sq_errors)))


def evaluate_estimator_run(
    network: RoadNetwork,
    observations: List[Any],
    update_rate: float = 0.5,
    tolerance: float = 1e-4,
    max_iters: int = 50,
) -> Tuple[Dict[str, float], float, float, int]:
    """
    Runs the full aggregation and iterative estimation pipeline and evaluates against truth.
    Returns (final_estimates, mae, rmse, iters_to_converge).
    """
    truth = {s_id: seg.true_roughness for s_id, seg in network.segments.items()}
    agg_data = aggregate_observations(observations)
    batch_means = {s_id: agg_data[s_id][0] for s_id in agg_data}

    estimator = RoadQualityEstimator(network, update_rate=update_rate, tolerance=tolerance)
    iters = 0
    delta = 1.0
    while iters < max_iters and delta >= tolerance:
        delta = estimator.update_batch(batch_means)
        iters += 1

    estimates = estimator.estimates
    mae = compute_mae(estimates, truth)
    rmse = compute_rmse(estimates, truth)
    return estimates, mae, rmse, iters


def evaluate_noise_robustness(
    network: RoadNetwork,
    noise_levels: List[float] = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
    num_vehicles: int = 60,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates estimation accuracy as per-observation sensor noise (noise_std) increases.
    """
    truth = {s_id: seg.true_roughness for s_id, seg in network.segments.items()}
    road_classes = [seg.road_class for seg in network.segments.values()]
    fixed_preds = be.predict_fixed_road_class(road_classes, ROAD_CLASSES)
    baseline_mae = float(np.mean(np.abs(fixed_preds - list(truth.values()))))

    results = []
    for noise in noise_levels:
        cfg = vo.ObservationConfig(noise_std=noise)
        vehicles = vo.generate_vehicles(num_vehicles=num_vehicles, config=cfg, seed=seed)
        observations = vo.simulate_observations(network, vehicles, trips_per_vehicle=4, seed=seed)

        _, est_mae, est_rmse, iters = evaluate_estimator_run(network, observations)
        results.append({
            "test_type": "noise_sweep",
            "parameter": noise,
            "estimator_mae": est_mae,
            "estimator_rmse": est_rmse,
            "baseline_mae": baseline_mae,
            "iterations": iters,
            "total_observations": len(observations),
        })
    return results


def evaluate_fleet_size_robustness(
    network: RoadNetwork,
    fleet_sizes: List[int] = [5, 15, 30, 60, 100],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates estimation accuracy as the number of observing vehicles increases.
    """
    truth = {s_id: seg.true_roughness for s_id, seg in network.segments.items()}
    road_classes = [seg.road_class for seg in network.segments.values()]
    fixed_preds = be.predict_fixed_road_class(road_classes, ROAD_CLASSES)
    baseline_mae = float(np.mean(np.abs(fixed_preds - list(truth.values()))))

    results = []
    cfg = vo.ObservationConfig(noise_std=0.05)
    for num_v in fleet_sizes:
        vehicles = vo.generate_vehicles(num_vehicles=num_v, config=cfg, seed=seed)
        observations = vo.simulate_observations(network, vehicles, trips_per_vehicle=4, seed=seed)

        _, est_mae, est_rmse, iters = evaluate_estimator_run(network, observations)
        results.append({
            "test_type": "fleet_sweep",
            "parameter": num_v,
            "estimator_mae": est_mae,
            "estimator_rmse": est_rmse,
            "baseline_mae": baseline_mae,
            "iterations": iters,
            "total_observations": len(observations),
        })
    return results


def evaluate_sparse_observations(
    network: RoadNetwork,
    observations: List[Any],
) -> Dict[str, Dict[str, float]]:
    """
    Categorizes road segments by observation frequency (0, 1-2, 3-5, >5 observations)
    and computes MAE for each sparsity bucket.
    """
    truth = {s_id: seg.true_roughness for s_id, seg in network.segments.items()}
    aggregator = ObservationAggregator()
    aggregator.add_observations(observations)
    sample_counts = {s_id: aggregator.get_sample_count(s_id) for s_id in network.segments}

    # Run iterative estimator
    estimates, _, _, _ = evaluate_estimator_run(network, observations)

    buckets = {
        "0 (Unvisited)": [],
        "1 - 2 (Sparse)": [],
        "3 - 5 (Moderate)": [],
        "> 5 (Frequent)": [],
    }

    for s_id, count in sample_counts.items():
        err = abs(estimates[s_id] - truth[s_id])
        if count == 0:
            buckets["0 (Unvisited)"].append(err)
        elif 1 <= count <= 2:
            buckets["1 - 2 (Sparse)"].append(err)
        elif 3 <= count <= 5:
            buckets["3 - 5 (Moderate)"].append(err)
        else:
            buckets["> 5 (Frequent)"].append(err)

    summary = {}
    for bucket_name, errors in buckets.items():
        summary[bucket_name] = {
            "count": len(errors),
            "mae": float(np.mean(errors)) if errors else 0.0,
        }
    return summary


def identify_failure_conditions(
    network: RoadNetwork,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Tests stress scenarios to pinpoint conditions where the estimator breaks down
    (i.e., when error approaches or exceeds the initial class-based baseline).
    """
    truth = {s_id: seg.true_roughness for s_id, seg in network.segments.items()}
    road_classes = [seg.road_class for seg in network.segments.values()]
    fixed_preds = be.predict_fixed_road_class(road_classes, ROAD_CLASSES)
    baseline_mae = float(np.mean(np.abs(fixed_preds - list(truth.values()))))

    scenarios = [
        ("Nominal Baseline", vo.ObservationConfig(noise_std=0.05, missing_prob=0.0)),
        ("High Noise (0.25)", vo.ObservationConfig(noise_std=0.25, missing_prob=0.0)),
        ("Severe Noise (0.45)", vo.ObservationConfig(noise_std=0.45, missing_prob=0.0)),
        ("High Missing Readings (70%)", vo.ObservationConfig(noise_std=0.05, missing_prob=0.70)),
        ("Extreme Vehicle Bias (gain=0.3, bias=0.15)", vo.ObservationConfig(gain_std=0.30, bias_std=0.15)),
        ("Compound Stress (Noise+Bias+Missing)", vo.ObservationConfig(noise_std=0.35, gain_std=0.30, bias_std=0.12, missing_prob=0.60)),
    ]

    failure_records = []
    for label, cfg in scenarios:
        vehicles = vo.generate_vehicles(num_vehicles=50, config=cfg, seed=seed)
        observations = vo.simulate_observations(network, vehicles, trips_per_vehicle=3, seed=seed)

        _, est_mae, est_rmse, iters = evaluate_estimator_run(network, observations)
        failed = est_mae >= baseline_mae
        pct_improvement = ((baseline_mae - est_mae) / baseline_mae) * 100.0

        failure_records.append({
            "scenario": label,
            "estimator_mae": est_mae,
            "estimator_rmse": est_rmse,
            "baseline_mae": baseline_mae,
            "pct_improvement": pct_improvement,
            "status": "FAILED (Worse than prior)" if failed else "PASSED (Improved over prior)",
            "iterations": iters,
        })

    return failure_records


def plot_robustness_dashboard(
    noise_results: List[Dict[str, Any]],
    fleet_results: List[Dict[str, Any]],
    sparsity_results: Dict[str, Dict[str, float]],
    failure_results: List[Dict[str, Any]],
    save_path: str = "robustness_analysis.png",
):
    """
    Renders a unified 4-panel dashboard visualizing all robustness experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)

    # 1. Noise Sensitivity
    ax = axes[0, 0]
    noises = [r["parameter"] for r in noise_results]
    mae_noise = [r["estimator_mae"] for r in noise_results]
    base_mae = noise_results[0]["baseline_mae"]
    ax.plot(noises, mae_noise, marker='o', color='#1f77b4', linewidth=2, label="Iterative Estimator MAE")
    ax.axhline(base_mae, color='#d62728', linestyle='--', linewidth=1.5, label=f"Static Class Baseline ({base_mae:.3f})")
    ax.set_title("1. Estimation Error vs. Sensor Noise", fontweight='bold')
    ax.set_xlabel("Sensor Noise Std (noise_std)")
    ax.set_ylabel("MAE against Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    # 2. Fleet Size / Data Volume
    ax = axes[0, 1]
    fleets = [r["parameter"] for r in fleet_results]
    mae_fleet = [r["estimator_mae"] for r in fleet_results]
    ax.plot(fleets, mae_fleet, marker='s', color='#2ca02c', linewidth=2, label="Iterative Estimator MAE")
    ax.axhline(base_mae, color='#d62728', linestyle='--', linewidth=1.5, label="Static Class Baseline")
    ax.set_title("2. Estimation Error vs. Fleet Size", fontweight='bold')
    ax.set_xlabel("Number of Fleet Vehicles")
    ax.set_ylabel("MAE against Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    # 3. Observation Sparsity Buckets
    ax = axes[1, 0]
    buckets = list(sparsity_results.keys())
    b_maes = [sparsity_results[b]["mae"] for b in buckets]
    b_counts = [sparsity_results[b]["count"] for b in buckets]
    bars = ax.bar(buckets, b_maes, color=['#ff7f0e', '#aec7e8', '#7f7f7f', '#2ca02c'], alpha=0.85)
    for bar, count in zip(bars, b_counts):
        height = bar.get_height()
        ax.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_title("3. Error by Observation Frequency", fontweight='bold')
    ax.set_xlabel("Segment Visit Bucket (Samples per Segment)")
    ax.set_ylabel("MAE")
    ax.grid(axis='y', alpha=0.3)

    # 4. Stress Scenarios & Failure Conditions
    ax = axes[1, 1]
    scenarios = [r["scenario"] for r in failure_results]
    maes = [r["estimator_mae"] for r in failure_results]
    colors = ['#2ca02c' if "PASSED" in r["status"] else '#d62728' for r in failure_results]
    ax.barh(scenarios, maes, color=colors, alpha=0.85)
    ax.axvline(base_mae, color='#d62728', linestyle='--', label=f"Baseline Threshold ({base_mae:.3f})")
    ax.set_title("4. Stress Scenarios & Failure Thresholds", fontweight='bold')
    ax.set_xlabel("MAE against Ground Truth")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.legend(frameon=True, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Robustness dashboard saved to: {save_path}")


def save_robustness_csv(results: List[Dict[str, Any]], filename: str = "robustness_results.csv"):
    """Export benchmark records to CSV."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Robustness metrics saved to: {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("        ACCURACY AND ROBUSTNESS EVALUATION (ASHMITA'S DELIVERABLE)        ")
    print("=" * 70)

    # Setup standard simulation network
    net = generate_road_network(num_nodes=50, num_segments=80, seed=42)

    # 1. Noise Sweep
    print("\n[1/4] Running Sensor Noise Sensitivity Sweep...")
    noise_res = evaluate_noise_robustness(net)
    for r in noise_res:
        print(f"  Noise Std: {r['parameter']:.2f} -> Estimator MAE: {r['estimator_mae']:.4f} (Baseline: {r['baseline_mae']:.4f})")

    # 2. Fleet Size Sweep
    print("\n[2/4] Running Fleet Size & Observation Volume Sweep...")
    fleet_res = evaluate_fleet_size_robustness(net)
    for r in fleet_res:
        print(f"  Fleet Size: {r['parameter']:3d} -> Estimator MAE: {r['estimator_mae']:.4f} (Total Obs: {r['total_observations']})")

    # 3. Sparsity Analysis
    print("\n[3/4] Analyzing Performance across Sparse vs. Frequent Segments...")
    cfg = vo.ObservationConfig(noise_std=0.05)
    vehs = vo.generate_vehicles(30, cfg, seed=42)
    obs = vo.simulate_observations(net, vehs, trips_per_vehicle=3, seed=42)
    sparsity_res = evaluate_sparse_observations(net, obs)
    for bucket, vals in sparsity_res.items():
        print(f"  Bucket [{bucket}]: Count = {vals['count']:2d} segments | MAE = {vals['mae']:.4f}")

    # 4. Failure Mode Identification
    print("\n[4/4] Identifying Failure Boundaries under Extreme Stress Scenarios...")
    failure_res = identify_failure_conditions(net)
    for r in failure_res:
        print(f"  {r['scenario']:<36} | MAE: {r['estimator_mae']:.4f} | Status: {r['status']}")

    # Save outputs
    all_tabular = noise_res + fleet_res
    save_robustness_csv(all_tabular, "robustness_results.csv")
    plot_robustness_dashboard(noise_res, fleet_res, sparsity_res, failure_res, "robustness_analysis.png")

    print("\nEvaluation complete. All results recorded.")
