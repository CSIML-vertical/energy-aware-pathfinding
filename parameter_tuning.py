import csv
import math
from road_network import generate_road_network
from vehicle_observation import generate_vehicles, simulate_observations
from estimator import RoadQualityEstimator

def aggregate_observations(observations):
    # Calculates mean and count per segment
    agg = {}
    for o in observations:
        if math.isnan(o.calibrated_response):
            continue
            
        seg = o.segment_id
        if seg not in agg:
            agg[seg] = [0.0, 0]
            
        agg[seg][0] += o.calibrated_response
        agg[seg][1] += 1
        
    res = {}
    for k, v in agg.items():
        if v[1] > 0:
            res[k] = (v[0] / v[1], v[1])
    return res

def run_single_trial(network, agg_data, ur, max_iter=50, tol=1e-4):
    # Initialize real estimator
    est = RoadQualityEstimator(network, update_rate=ur, tolerance=tol)
    
    # Extract just the means for the update_batch method
    batch_means = {s_id: agg_data[s_id][0] for s_id in agg_data}
    
    iters = 0
    max_delta = 1.0
    conv = False
    
    while iters < max_iter and max_delta >= tol:
        max_delta = est.update_batch(batch_means)
        iters += 1
        
    conv = max_delta < tol

    # calculate error against ground truth
    errors = []
    for s_id, seg in network.segments.items():
        true_val = seg.true_roughness
        est_val = est.estimates[s_id]
        errors.append(abs(true_val - est_val))
        
    mae = sum(errors) / len(errors) if errors else 0.0
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0

    return {
        "update_rate": ur,
        "iterations": iters,
        "converged": conv,
        "mae": mae,
        "rmse": rmse,
        "final_estimates": est.estimates
    }

def sweep_parameters(network, agg_data):
    # Sweeping only update_rate (alpha) since prior_strength is not used
    ur_list = [0.05, 0.1, 0.2, 0.5, 0.8, 0.95]
    
    results = []
    print("Starting sweep...")
    
    for ur in ur_list:
        res = run_single_trial(network, agg_data, ur)
        results.append(res)
        print(f"UR:{ur:.2f} | Converged:{res['converged']} | Iters:{res['iterations']} | MAE:{res['mae']:.4f}")
            
    return results

def get_best_params(results):
    # keep only runs that converged
    valid = [r for r in results if r["converged"]]
    
    if not valid:
        print("None converged, falling back to lowest MAE.")
        return min(results, key=lambda x: x["mae"])
    
    # sort by MAE, then iterations
    valid.sort(key=lambda x: (x["mae"], x["iterations"]))
    return valid[0]

def check_road_classes(network, best_trial):
    est = best_trial["final_estimates"]
    classes = ["highway", "main_road", "small_road"]
    class_errs = {c: [] for c in classes}
    
    for s_id, seg in network.segments.items():
        err = abs(seg.true_roughness - est[s_id])
        class_errs[seg.road_class].append(err)
        
    print("\n--- Error by Class ---")
    for c in classes:
        if class_errs[c]:
            mae = sum(class_errs[c]) / len(class_errs[c])
            print(f"{c}: {mae:.4f}")

def save_csv(results, filename="parameter_tuning_results.csv"):
    keys = ["update_rate", "iterations", "converged", "mae", "rmse"]
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    # 1. generate data
    net = generate_road_network(num_nodes=50, num_segments=80, seed=42)
    vehicles = generate_vehicles(num_vehicles=100, seed=42)
    obs = simulate_observations(net, vehicles, trips_per_vehicle=5, seed=42)
    
    print(f"Total observations: {len(obs)}")
    agg_data = aggregate_observations(obs)
    
    # 2. run tuning
    res = sweep_parameters(net, agg_data)
    save_csv(res)
    
    # 3. find best
    best = get_best_params(res)
    print(f"\nBest params: UR={best['update_rate']}")
    print(f"Converged in {best['iterations']} iters")
    
    # 4. check classes
    check_road_classes(net, best)