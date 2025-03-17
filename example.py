"""
Example usage of the Black Widow Optimization Algorithm.

This file demonstrates how to use the BlackWidowOptimizer class to solve
optimization problems with the benchmark functions from the original paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from black_widow import BlackWidowOptimizer
from benchmark_functions import (
    powell_sum, cigar, discus, rosenbrock, ackley
)


def run_optimization(func, func_name, dimensions, bounds, max_iterations=100, population_size=40):
    """
    Run the Black Widow Optimization algorithm on a given function.
    
    Args:
        func: The objective function to optimize
        func_name: Name of the function (for display)
        dimensions: Number of dimensions
        bounds: List of (min, max) bounds for each dimension
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (best solution, best fitness, convergence curve)
    """
    print(f"\nOptimizing {func_name} function ({dimensions} dimensions)...")
    
    # Create optimizer
    optimizer = BlackWidowOptimizer(
        objective_function=func,
        dimensions=dimensions,
        bounds=bounds,
        population_size=population_size,
        max_iterations=max_iterations,
        reproduction_rate=0.6,
        cannibalism_rate=0.4,
        mutation_rate=0.4,
        minimize=True
    )
    
    # Run optimization
    best_solution, best_fitness = optimizer.optimize(verbose=True)
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    return best_solution, best_fitness, optimizer.convergence_curve


def plot_convergence(convergence_curves, function_names, max_iterations, filename='convergence.png'):
    """
    Plot the convergence curves for multiple optimization runs.
    
    Args:
        convergence_curves: List of convergence curves
        function_names: List of function names
        max_iterations: Maximum number of iterations
    """
    plt.figure(figsize=(12, 8))
    
    for i, curve in enumerate(convergence_curves):
        plt.semilogy(range(len(curve)), curve, label=function_names[i], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title('Black Widow Optimization Algorithm Convergence', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # Don't try to show the plot in a terminal environment
    print(f"\nConvergence plot saved as '{filename}'")


def main():
    """Main function to run the examples with the benchmark functions from the original paper."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Common parameters
    dimensions = 10
    max_iterations = 100
    population_size = 40
    
    # Define the benchmark functions from the original paper with their bounds
    benchmark_functions = [
        (powell_sum, "F1: Powell Sum", [(-5.12, 5.12)] * dimensions),
        (cigar, "F2: Cigar", [(-5.12, 5.12)] * dimensions),
        (discus, "F3: Discus", [(-5.12, 5.12)] * dimensions),
        (rosenbrock, "F4: Rosenbrock", [(-30, 30)] * dimensions),
        (ackley, "F5: Ackley", [(-35, 35)] * dimensions)
    ]
    
    print("=" * 80)
    print("Running Black Widow Optimization on benchmark functions from the original paper")
    print("=" * 80)
    
    # Run optimization for each function
    convergence_curves = []
    function_names = []
    results = []
    
    for func, name, bounds in benchmark_functions:
        print(f"\nOptimizing {name} (Dimensions: {dimensions})")
        print(f"Search domain: {bounds[0]}")
        
        best_solution, best_fitness, curve = run_optimization(
            func, name, dimensions, bounds, max_iterations, population_size
        )
        
        convergence_curves.append(curve)
        function_names.append(name)
        results.append((name, best_fitness, best_solution))
    
    # Plot convergence curves
    plot_convergence(convergence_curves, function_names, max_iterations, 'paper_functions_convergence.png')
    
    # Print summary of results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    
    for name, fitness, solution in results:
        print(f"{name}:")
        print(f"  Best fitness: {fitness}")
        print(f"  Solution norm: {np.linalg.norm(solution)}")
        print("-" * 40)


if __name__ == "__main__":
    main()
