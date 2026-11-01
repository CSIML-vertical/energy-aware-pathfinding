"""
Visualization & Dashboard Module (Mihir)
Generates the core project presentation figures:
1. Convergence Curve (Iteration vs. Max Delta Error)
2. Road Roughness Comparison (Ground Truth vs. Initial Prior vs. Iteratively Refined)
3. Multi-Objective Route Comparison (Fastest Route vs. Energy-Aware Route)
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_curve(history_deltas, tolerance=1e-4, save_path="convergence_curve.png"):
    """Plot max segment delta decay across estimator iterations."""
    plt.figure(figsize=(8, 4.5), dpi=150)
    iters = list(range(1, len(history_deltas) + 1))
    plt.plot(iters, history_deltas, marker='o', color='#1f77b4', linewidth=2, label="Max Segment Delta (||ΔR||)")
    plt.axhline(tolerance, color='#d62728', linestyle='--', linewidth=1.5, label=f"Tolerance ({tolerance})")
    plt.yscale('log')
    plt.xlabel("Iteration Step", fontsize=11, fontweight='bold')
    plt.ylabel("Maximum Absolute Delta (Log Scale)", fontsize=11, fontweight='bold')
    plt.title("Iterative Road Quality Estimator Convergence", fontsize=12, fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roughness_comparison(net, initial_r, estimated_r, save_path="roughness_comparison.png"):
    """Bar chart comparing Initial Prior vs. Refined Estimate vs. Ground Truth."""
    plt.figure(figsize=(11, 5), dpi=150)
    seg_ids = list(net.segments.keys())[:20]  # First 20 segments for clear display
    truth_vals = [net.segments[s].true_roughness for s in seg_ids]
    init_vals = [initial_r[s] for s in seg_ids]
    est_vals = [estimated_r[s] for s in seg_ids]

    x = np.arange(len(seg_ids))
    width = 0.28

    plt.bar(x - width, init_vals, width, label="Initial Prior Estimate", color='#ff7f0e', alpha=0.85)
    plt.bar(x, est_vals, width, label="Refined Iterative Estimate", color='#2ca02c', alpha=0.85)
    plt.bar(x + width, truth_vals, width, label="Hidden Ground Truth", color='#1f77b4', alpha=0.85)

    plt.xlabel("Road Segment ID", fontsize=11, fontweight='bold')
    plt.ylabel("Normalized Roughness [0, 1]", fontsize=11, fontweight='bold')
    plt.title("Road Quality Estimation Accuracy Across Segments", fontsize=12, fontweight='bold')
    plt.xticks(x, seg_ids, rotation=45, ha='right', fontsize=9)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_route_network(net, fastest_path, energy_path, start_node, goal_node, save_path="route_comparison.png"):
    """Visualizes network graph overlaying time-optimal vs. energy-optimal routes."""
    import networkx as nx
    plt.figure(figsize=(10, 7.5), dpi=150)

    # Build networkx graph for a clean, deterministic 2D road-like layout
    G = nx.Graph()
    for seg in net.segments.values():
        G.add_edge(seg.start_node, seg.end_node, weight=seg.length_km)

    # Use deterministic spring layout with fixed seed
    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=50)

    # Draw all network road segments in light gray
    for seg in net.segments.values():
        p1, p2 = pos[seg.start_node], pos[seg.end_node]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#d0d4dc', linewidth=1.2, zorder=1)

    fast_segs = set(fastest_path["segments"])
    energy_segs = set(energy_path["segments"])
    shared_segs = fast_segs & energy_segs

    # 1. Highlight Fastest Route (Red)
    for sid in fastest_path["segments"]:
        seg = net.get_segment(sid)
        p1, p2 = pos[seg.start_node], pos[seg.end_node]
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color='#d62728', linewidth=4.0, alpha=0.85, zorder=3,
            label=f"Fastest Route (λ=1.0: {fastest_path['total_time_min']:.1f}m, {fastest_path['total_energy_kwh']:.1f}kWh)"
            if sid == fastest_path["segments"][0] else ""
        )

    # 2. Highlight Energy-Aware Route (Green dashed)
    for sid in energy_path["segments"]:
        seg = net.get_segment(sid)
        p1, p2 = pos[seg.start_node], pos[seg.end_node]
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color='#2ca02c', linewidth=2.8, linestyle='--', zorder=4,
            label=f"Energy-Aware Route (λ=0.0: {energy_path['total_time_min']:.1f}m, {energy_path['total_energy_kwh']:.1f}kWh)"
            if sid == energy_path["segments"][0] else ""
        )

    # Mark origin and destination
    p_start, p_goal = pos[start_node], pos[goal_node]
    plt.scatter([p_start[0]], [p_start[1]], color='#000000', s=140, zorder=6, label=f"Origin [{start_node}]")
    plt.scatter([p_goal[0]], [p_goal[1]], color='#9467bd', s=180, marker='*', zorder=6, label=f"Destination [{goal_node}]")

    saved_energy = fastest_path['total_energy_kwh'] - energy_path['total_energy_kwh']
    energy_saved_pct = (saved_energy / fastest_path['total_energy_kwh'] * 100) if fastest_path['total_energy_kwh'] > 0 else 0
    title_sub = f"Energy Saved: {saved_energy:.2f} kWh ({energy_saved_pct:.1f}%)" if saved_energy > 0 else "Optimal route coincides"

    plt.title(f"Multi-Objective Route Selection ({start_node} -> {goal_node})\n{title_sub}", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.legend(frameon=True, loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    from road_network import generate_road_network
    import vehicle_observation as vo
    from estimator import RoadQualityEstimator
    from observation_processing import aggregate_observations
    from energy_model import EnergyPredictor
    from energy_astar import EnergyAStarRouter

    # 1. Setup network & simulation
    net = generate_road_network(50, 80, seed=42)
    cfg = vo.ObservationConfig(noise_std=0.05)
    vehicles = vo.generate_vehicles(60, cfg, seed=42)
    obs = vo.simulate_observations(net, vehicles, 4, seed=42)

    # 2. Aggregation & estimation loop
    agg = aggregate_observations(obs)
    batch_means = {s: agg[s][0] for s in agg}
    estimator = RoadQualityEstimator(net, update_rate=0.5, tolerance=1e-4)

    deltas = []
    while len(deltas) < 50:
        d = estimator.update_batch(batch_means)
        deltas.append(d)
        if d < estimator.tolerance:
            break

    # 3. Generate Convergence & Accuracy Plots
    plot_convergence_curve(deltas)
    init_r = {s: net.segments[s].initial_roughness for s in net.segments}
    plot_roughness_comparison(net, init_r, estimator.estimates)

    # 4. Route plotting with distinct trade-off path selection
    predictor = EnergyPredictor(random_state=42)
    predictor.train_from_simulations(net, obs)
    router = EnergyAStarRouter(net, predictor)

    node_list = sorted(list(net.nodes))
    selected_start, selected_goal = node_list[0], node_list[-1]
    fast_route = router.find_path(selected_start, selected_goal, estimator.estimates, lambda_weight=1.0)
    energy_route = router.find_path(selected_start, selected_goal, estimator.estimates, lambda_weight=0.0)

    # Automatically pick a pair that clearly demonstrates the trade-off (diverging paths)
    for s in node_list[:15]:
        for g in node_list[20:45]:
            r_fast = router.find_path(s, g, estimator.estimates, lambda_weight=1.0)
            r_energy = router.find_path(s, g, estimator.estimates, lambda_weight=0.0)
            if (
                r_fast and r_energy
                and r_fast["segments"] != r_energy["segments"]
                and r_fast["total_energy_kwh"] > r_energy["total_energy_kwh"]
            ):
                selected_start, selected_goal = s, g
                fast_route, energy_route = r_fast, r_energy
                break
        if fast_route and energy_route and fast_route["segments"] != energy_route["segments"]:
            break

    plot_route_network(net, fast_route, energy_route, selected_start, selected_goal)
    print("All visualizations generated successfully.")