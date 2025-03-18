import numpy as np
from black_widow import BlackWidowOptimizer
from benchmark_functions import sphere, powell_sum


def test_function(func, func_name, dimensions, bounds, expected_optimum, threshold=1.0):
    """
    Test the Black Widow Optimization Algorithm on a given function.
    
    Args:
        func: The objective function to optimize
        func_name: Name of the function for display
        dimensions: Number of dimensions
        bounds: List of (min, max) bounds for each dimension
        expected_optimum: The expected optimal solution
        threshold: Threshold for considering the test passed
        
    Returns:
        bool: True if the test passed, False otherwise
    """
    print(f"\nTesting on {func_name} function ({dimensions} dimensions)...")
    
    # Create optimizer with small population and few iterations for quick testing
    optimizer = BlackWidowOptimizer(
        objective_function=func,
        dimensions=dimensions,
        bounds=bounds,
        population_size=20,
        max_iterations=30,
        reproduction_rate=0.6,
        cannibalism_rate=0.4,
        mutation_rate=0.4,
        minimize=True
    )
    
    # Run optimization
    best_solution, best_fitness = optimizer.optimize(verbose=True)
    
    print("\nTest Results:")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    # Check if the solution is close to the expected optimum
    solution_error = np.linalg.norm(best_solution - expected_optimum)
    print(f"Distance from expected optimum: {solution_error}")
    
    if best_fitness < threshold:
        print(f"Test PASSED: The algorithm found a good solution (fitness < {threshold}).")
        return True
    else:
        print(f"Test WARNING: The solution might not be optimal (fitness >= {threshold}).")
        return False


def main():
    """Run tests of the Black Widow Optimization Algorithm."""
    print("=" * 60)
    print("TESTING BLACK WIDOW OPTIMIZATION ALGORITHM")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test on sphere function
    dimensions = 5
    bounds_sphere = [(-10, 10)] * dimensions
    expected_optimum_sphere = np.zeros(dimensions)
    test1 = test_function(
        sphere, 
        "Sphere", 
        dimensions, 
        bounds_sphere, 
        expected_optimum_sphere,
        threshold=0.02  # Threshold to account for randomness in the algorithm
    )
    
    # Test on Powell Sum function
    bounds_powell = [(-5.12, 5.12)] * dimensions
    expected_optimum_powell = np.zeros(dimensions)
    test2 = test_function(
        powell_sum, 
        "Powell Sum", 
        dimensions, 
        bounds_powell, 
        expected_optimum_powell,
        threshold=0.1
    )
    
    # Print overall results
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)
    
    if test1 and test2:
        print("All tests PASSED! The Black Widow Optimization Algorithm is working correctly.")
    else:
        print("Some tests did not pass. The algorithm might need further tuning.")


if __name__ == "__main__":
    main()
