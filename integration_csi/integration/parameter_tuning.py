import csv
import math
from road_network import generate_road_network
from vehicle_observation import generate_vehicles, simulate_observations

from observation_processing import aggregate_observations, ObservationAggregator

class MockEstimator:
    def __init__(self, update_rate, prior_strength, tol=1e-4):
        self.rate = update_rate
        self.prior_w = prior_strength
        self.tol = tol

    def update(self, curr, prior, obs_mean, obs_count):
        total_w = self.prior_w + obs_count
        weighted_avg = (prior * self.prior_w + obs_mean * obs_count) / total_w
        
        new_val = curr + self.rate * (weighted_avg - curr)
        return max(0.0, min(1.0, new_val))

    def has_converged(self, old_est, new_est):
        if not old_est or not new_est: 
            return False
        max_diff = max(abs(new_est[s] - old_est[s]) for s in old_est)
        return max_diff < self.tol
# --------------------------------------------------

def run_single_trial(network, agg_data, ur, ps, max_iter=50):
    est = MockEstimator(ur, ps)
    
    # start with initial roughness
    curr_est = {s_id: seg.initial_roughness for s_id, seg in network.segments.items()}
    
    iters = 0
    conv = False
    
    while iters < max_iter and not conv:
        old_est = curr_est.copy()
        
        for s_id, seg in network.segments.items():
            if s_id in agg_data:
                mean_obs, count = agg_data[s_id]
                prior = seg.initial_roughness
                curr_est[s_id] = est.update(old_est[s_id], prior, mean_obs, count)
            else:
                curr_est[s_id] = old_est[s_id]
                
        conv = est.has_converged(old_est, curr_est)
        iters += 1

    # calculate error
    errors = []
    for s_id, seg in network.segments.items():
        true_val = seg.true_roughness
        est_val = curr_est[s_id]
        errors.append(abs(true_val - est_val))
        
    mae = sum(errors) / len(errors) if errors else 0.0
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0

    return {
        "update_rate": ur,
        "prior_strength": ps,
        "iterations": iters,
        "converged": conv,
        "mae": mae,
        "rmse": rmse,
        "final_estimates": curr_est
    }

def sweep_parameters(network, agg_data):
    ur_list = [0.05, 0.1, 0.2, 0.5]
    ps_list = [0.5, 1.0, 2.0, 5.0]
    
    results = []
    print("Starting sweep...")
    
    for ur in ur_list:
        for ps in ps_list:
            res = run_single_trial(network, agg_data, ur, ps)
            results.append(res)
            print(f"UR:{ur} PS:{ps} | Converged:{res['converged']} | Iters:{res['iterations']} | MAE:{res['mae']:.4f}")
            
    return results

def get_best_params(results):
    # keep only runs that converged
    valid = [r for r in results if r["converged"]]
    
    if not valid:
        print("None converged, falling back to lowest MAE.")
        return min(results, key=lambda x: x["mae"])
    
    # sort by MAE, then iterations, then prior strength
    valid.sort(key=lambda x: (x["mae"], x["iterations"], x["prior_strength"]))
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
    keys = ["update_rate", "prior_strength", "iterations", "converged", "mae", "rmse"]
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
    print(f"\nBest params: UR={best['update_rate']}, PS={best['prior_strength']}")
    print(f"Converged in {best['iterations']} iters")
    
    # 4. check classes
    check_road_classes(net, best)