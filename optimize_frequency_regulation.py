import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from black_widow import BlackWidowOptimizer
from frequency_regulation_function import (
    power_system_itae_optimized,
    generate_system_responses,
    simulate_base_system,
    apply_controller_to_response
)


def ensure_system_responses_exist():
    """
    Ensure that the system responses file exists, generate it if not.
    
    This function checks if we already have pre-calculated system responses.
    If not, it generates them. This is like checking if we already have
    recorded data of how the car behaves on different hills, and if not,
    running those tests to get the data.
    """
    if not os.path.exists('system_responses.pkl'):
        print("Generating system responses...")
        generate_system_responses()
        print("System responses generated and saved.")
    else:
        print("Using existing system responses.")


def optimize_pid_parameters():
    """
    Optimize PID controller parameters using Black Widow Optimization.
    
    This function uses a nature-inspired optimization algorithm to find the best
    PID controller parameters. The Black Widow Optimization algorithm mimics the
    mating behavior of black widow spiders to search for optimal solutions.
    It's like having multiple test drivers try different cruise control settings
    and keeping the best ones while discarding the worst.
    
    Returns:
        Tuple of (best_parameters, best_itae)
    """
    # Ensure system responses exist
    ensure_system_responses_exist()
    
    # Define the dimensions and bounds for the optimization
    dimensions = 6  # Kp1, Ki1, Kd1, Kp2, Ki2, Kd2
    
    # Define bounds for each parameter
    # Typical ranges for PID parameters in power systems
    bounds = [
        (0.0, 2.0),   # Kp1: Proportional gain for area 1
        (0.0, 1.0),   # Ki1: Integral gain for area 1
        (0.0, 0.5),   # Kd1: Derivative gain for area 1
        (0.0, 2.0),   # Kp2: Proportional gain for area 2
        (0.0, 1.0),   # Ki2: Integral gain for area 2
        (0.0, 0.5)    # Kd2: Derivative gain for area 2
    ]
    
    # Create a wrapper for the objective function with progress reporting
    eval_count = [0]
    
    def objective_function_with_progress(x):
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"Evaluation {eval_count[0]}: Testing parameters {x}")
        return power_system_itae_optimized(x)
    
    # Create the optimizer with smaller population and fewer iterations for faster execution
    optimizer = BlackWidowOptimizer(
        objective_function=objective_function_with_progress,
        dimensions=dimensions,
        bounds=bounds,
        population_size=20,  # Reduced from 30
        max_iterations=30,   # Reduced from 50
        reproduction_rate=0.6,
        cannibalism_rate=0.4,
        mutation_rate=0.4,
        minimize=True  # We want to minimize ITAE
    )
    
    # Run the optimization
    print("Starting optimization...")
    best_parameters, best_itae = optimizer.optimize(verbose=True)
    
    # Print the results
    print("\nOptimization completed!")
    print(f"Best ITAE: {best_itae}")
    print("Optimal PID parameters:")
    print(f"Area 1: Kp={best_parameters[0]:.4f}, Ki={best_parameters[1]:.4f}, Kd={best_parameters[2]:.4f}")
    print(f"Area 2: Kp={best_parameters[3]:.4f}, Ki={best_parameters[4]:.4f}, Kd={best_parameters[5]:.4f}")
    
    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(optimizer.convergence_curve)), optimizer.convergence_curve)
    plt.xlabel('Iteration')
    plt.ylabel('ITAE (log scale)')
    plt.title('Convergence of Black Widow Optimization for PID Tuning')
    plt.grid(True)
    plt.savefig('pid_optimization_convergence.png')
    
    return best_parameters, best_itae


def compare_responses(best_parameters):
    """
    Compare system responses with and without the optimized controller.
    
    This function visualizes how the power system behaves under three conditions:
    1. Without any controller
    2. With a basic controller
    3. With the optimized controller
    
    It's like comparing how a car maintains speed on a hill with no cruise control,
    with basic cruise control, and with an optimized cruise control system.
    
    Args:
        best_parameters: The optimized PID parameters
    """
    # Load system responses
    with open('system_responses.pkl', 'rb') as f:
        system_responses = pickle.load(f)
    
    # Select a specific disturbance for visualization
    delta_PL = 0.1  # 10% load disturbance
    
    # Extract base response
    time = system_responses[delta_PL]['time']
    delta_f1_base = system_responses[delta_PL]['delta_f1']
    delta_f2_base = system_responses[delta_PL]['delta_f2']
    delta_Ptie_base = system_responses[delta_PL]['delta_Ptie']
    
    # Apply optimized controller
    Kp1, Ki1, Kd1, Kp2, Ki2, Kd2 = best_parameters
    delta_f1_opt, delta_f2_opt, delta_Ptie_opt = apply_controller_to_response(
        delta_f1_base, delta_f2_base, delta_Ptie_base,
        Kp1, Ki1, Kd1, Kp2, Ki2, Kd2
    )
    
    # Apply a basic controller for comparison
    # Simulate with basic controller
    _, delta_f1_basic, delta_f2_basic, delta_Ptie_basic = simulate_base_system(delta_PL, use_basic_controller=True)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot frequency deviation in area 1
    plt.subplot(3, 1, 1)
    plt.plot(time, delta_f1_base, 'r--', label='Without Controller')
    plt.plot(time, delta_f1_basic, 'g-', label='Basic Controller')
    plt.plot(time, delta_f1_opt, 'b-', label='Optimized Controller')
    plt.xlabel('Time (s)')
    plt.ylabel('Δf1 (Hz)')
    plt.title('Frequency Deviation in Area 1')
    plt.legend()
    plt.grid(True)
    
    # Plot frequency deviation in area 2
    plt.subplot(3, 1, 2)
    plt.plot(time, delta_f2_base, 'r--', label='Without Controller')
    plt.plot(time, delta_f2_basic, 'g-', label='Basic Controller')
    plt.plot(time, delta_f2_opt, 'b-', label='Optimized Controller')
    plt.xlabel('Time (s)')
    plt.ylabel('Δf2 (Hz)')
    plt.title('Frequency Deviation in Area 2')
    plt.legend()
    plt.grid(True)
    
    # Plot tie-line power deviation
    plt.subplot(3, 1, 3)
    plt.plot(time, delta_Ptie_base, 'r--', label='Without Controller')
    plt.plot(time, delta_Ptie_basic, 'g-', label='Basic Controller')
    plt.plot(time, delta_Ptie_opt, 'b-', label='Optimized Controller')
    plt.xlabel('Time (s)')
    plt.ylabel('ΔPtie (p.u.)')
    plt.title('Tie-Line Power Deviation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('system_responses_comparison.png')
    
    # Calculate performance metrics
    def calculate_metrics(delta_f1, delta_f2, delta_Ptie):
        """
        Calculate performance metrics for a given system response.
        
        This function computes various metrics to evaluate controller performance:
        - Maximum deviations: How far the system strays from the target
        - Settling times: How long it takes to stabilize
        - ITAE: A comprehensive measure of controller performance
        
        Args:
            delta_f1, delta_f2, delta_Ptie: System response variables
            
        Returns:
            Dictionary of performance metrics
        """
        # Maximum deviations
        max_f1 = np.max(np.abs(delta_f1))
        max_f2 = np.max(np.abs(delta_f2))
        max_Ptie = np.max(np.abs(delta_Ptie))
        
        # Settling times (time to reach and stay within 2% of final value)
        # For simplicity, we'll assume final value is 0 (steady state)
        threshold = 0.02 * max(max_f1, max_f2)
        
        # Find the last time the signal exceeds the threshold
        settling_time_f1 = time[np.where(np.abs(delta_f1) > threshold)[0][-1]] if np.any(np.abs(delta_f1) > threshold) else 0
        settling_time_f2 = time[np.where(np.abs(delta_f2) > threshold)[0][-1]] if np.any(np.abs(delta_f2) > threshold) else 0
        settling_time_Ptie = time[np.where(np.abs(delta_Ptie) > threshold)[0][-1]] if np.any(np.abs(delta_Ptie) > threshold) else 0
        
        # ITAE
        error = np.abs(delta_f1) + np.abs(delta_f2) + np.abs(delta_Ptie)
        itae = np.sum(time * error) * (time[1] - time[0])
        
        return {
            'max_f1': max_f1,
            'max_f2': max_f2,
            'max_Ptie': max_Ptie,
            'settling_time_f1': settling_time_f1,
            'settling_time_f2': settling_time_f2,
            'settling_time_Ptie': settling_time_Ptie,
            'itae': itae
        }
    
    # Calculate metrics for each case
    metrics_base = calculate_metrics(delta_f1_base, delta_f2_base, delta_Ptie_base)
    metrics_basic = calculate_metrics(delta_f1_basic, delta_f2_basic, delta_Ptie_basic)
    metrics_opt = calculate_metrics(delta_f1_opt, delta_f2_opt, delta_Ptie_opt)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("\nWithout Controller:")
    print(f"Maximum Frequency Deviation Area 1: {metrics_base['max_f1']:.6f} Hz")
    print(f"Maximum Frequency Deviation Area 2: {metrics_base['max_f2']:.6f} Hz")
    print(f"Maximum Tie-Line Power Deviation: {metrics_base['max_Ptie']:.6f} p.u.")
    print(f"Settling Time Area 1: {metrics_base['settling_time_f1']:.2f} s")
    print(f"Settling Time Area 2: {metrics_base['settling_time_f2']:.2f} s")
    print(f"Settling Time Tie-Line: {metrics_base['settling_time_Ptie']:.2f} s")
    print(f"ITAE: {metrics_base['itae']:.6f}")
    
    print("\nBasic Controller:")
    print(f"Maximum Frequency Deviation Area 1: {metrics_basic['max_f1']:.6f} Hz")
    print(f"Maximum Frequency Deviation Area 2: {metrics_basic['max_f2']:.6f} Hz")
    print(f"Maximum Tie-Line Power Deviation: {metrics_basic['max_Ptie']:.6f} p.u.")
    print(f"Settling Time Area 1: {metrics_basic['settling_time_f1']:.2f} s")
    print(f"Settling Time Area 2: {metrics_basic['settling_time_f2']:.2f} s")
    print(f"Settling Time Tie-Line: {metrics_basic['settling_time_Ptie']:.2f} s")
    print(f"ITAE: {metrics_basic['itae']:.6f}")
    
    print("\nOptimized Controller:")
    print(f"Maximum Frequency Deviation Area 1: {metrics_opt['max_f1']:.6f} Hz")
    print(f"Maximum Frequency Deviation Area 2: {metrics_opt['max_f2']:.6f} Hz")
    print(f"Maximum Tie-Line Power Deviation: {metrics_opt['max_Ptie']:.6f} p.u.")
    print(f"Settling Time Area 1: {metrics_opt['settling_time_f1']:.2f} s")
    print(f"Settling Time Area 2: {metrics_opt['settling_time_f2']:.2f} s")
    print(f"Settling Time Tie-Line: {metrics_opt['settling_time_Ptie']:.2f} s")
    print(f"ITAE: {metrics_opt['itae']:.6f}")
    
    # Calculate improvement percentages
    itae_improvement_over_base = (metrics_base['itae'] - metrics_opt['itae']) / metrics_base['itae'] * 100
    itae_improvement_over_basic = (metrics_basic['itae'] - metrics_opt['itae']) / metrics_basic['itae'] * 100
    
    print(f"\nITAE Improvement over No Controller: {itae_improvement_over_base:.2f}%")
    print(f"ITAE Improvement over Basic Controller: {itae_improvement_over_basic:.2f}%")


def main():
    """
    Main function to run the optimization and analysis.
    
    This function orchestrates the entire process:
    1. Optimize the PID controller parameters
    2. Compare the system responses with different controllers
    3. Display and save the results
    """
    # Run the optimization
    best_parameters, _ = optimize_pid_parameters()
    
    # Compare system responses
    compare_responses(best_parameters)
    
    print("\nOptimization and analysis completed. Results saved as images.")


if __name__ == "__main__":
    main()
