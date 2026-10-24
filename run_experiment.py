from road_network import generate_road_network
from estimator import RoadQualityEstimator

def run_convergence_experiment():
    print("Setting up the experiment...")
    
    # 1. Initialize the network and estimator
    network = generate_road_network(num_nodes=50, num_segments=80, seed=101)
    estimator = RoadQualityEstimator(network, update_rate=0.15, tolerance=1e-4)
    
    # 2. Create a static batch of observations to simulate Parth's output
    # We will pretend every vehicle reported a roughness of 0.5
    simulated_observations = {
        seg_id: 0.50 for seg_id in estimator.estimates.keys()
    }
    
    iterations = 0
    max_iterations = 200
    
    # 3. Open a file to record the results
    with open("estimator_experiment_results.txt", "w") as file:
        file.write("Iterative Road-Quality Estimator - Convergence Experiment\n")
        file.write("-" * 55 + "\n")
        file.write(f"Target Observation: 0.50 for all segments\n")
        file.write(f"Update Rate (alpha): {estimator.update_rate}\n")
        file.write(f"Tolerance: {estimator.tolerance}\n\n")
        
        # 4. Run the loop until convergence
        while iterations < max_iterations:
            max_delta = estimator.update_batch(simulated_observations)
            iterations += 1
            
            # Log every 10th iteration to keep the file readable
            if iterations % 10 == 0 or max_delta < estimator.tolerance:
                file.write(f"Iteration {iterations:03d} | Max Delta: {max_delta:.6f}\n")
                
            if max_delta < estimator.tolerance:
                file.write(f"\nSUCCESS: Network converged after {iterations} iterations.\n")
                print(f"Experiment complete! Converged in {iterations} iterations.")
                print("Results saved to 'estimator_experiment_results.txt'.")
                break

if __name__ == "__main__":
    run_convergence_experiment()