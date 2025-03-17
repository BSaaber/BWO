# Black Widow Optimization Algorithm

A Python implementation of the Black Widow Optimization Algorithm, a meta-heuristic approach for solving optimization problems, inspired by the mating behavior of black widow spiders.

## Overview

The Black Widow Optimization Algorithm is a population-based meta-heuristic algorithm inspired by the unique mating behavior of black widow spiders, where sexual cannibalism occurs (females sometimes eat males after mating). This algorithm was introduced in the paper "Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems" by Hayyolalam & Pourhaji Kazem (2020).

Key features of this implementation:
- Clean, object-oriented design
- Efficient NumPy-based operations
- Support for both minimization and maximization problems
- Customizable parameters
- Convergence tracking
- Comprehensive benchmark functions

## Project Structure

- `black_widow.py`: Main implementation of the algorithm
- `benchmark_functions.py`: Collection of benchmark functions including those from the original paper
- `example.py`: Example usage with benchmark functions from the original paper
- `test.py`: Test script to verify the implementation
- `README.md`: Documentation

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualization in examples)

## Benchmark Functions

This implementation includes the benchmark functions used in the original paper:

1. **F1: Powell Sum (Some of different powers)**
   - `f(x) = sum_{i=1}^n |x_i|^(i+1)`
   - Global minimum: f(0,0,...,0) = 0
   - Search domain: [-5.12, 5.12]^n

2. **F2: Cigar**
   - `f(x) = x_1^2 + 10^6 * sum_{i=2}^n x_i^2`
   - Global minimum: f(0,0,...,0) = 0
   - Search domain: [-5.12, 5.12]^n

3. **F3: Discus**
   - `f(x) = 10^6 * x_1^2 + sum_{i=2}^n x_i^2`
   - Global minimum: f(0,0,...,0) = 0
   - Search domain: [-5.12, 5.12]^n

4. **F4: Rosenbrock**
   - `f(x) = sum_{i=1}^{n-1} [100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2]`
   - Global minimum: f(1,1,...,1) = 0
   - Search domain: [-30, 30]^n

5. **F5: Ackley**
   - `f(x) = -20*exp(-0.2*sqrt(1/n * sum_{i=1}^n x_i^2)) - exp(1/n * sum_{i=1}^n cos(2*pi*x_i)) + 20 + e`
   - Global minimum: f(0,0,...,0) = 0
   - Search domain: [-35, 35]^n

Additional benchmark functions are also included for testing purposes.

## Usage

### Basic Usage

```python
from black_widow import BlackWidowOptimizer
from benchmark_functions import powell_sum

# Define problem dimensions and bounds
dimensions = 10
bounds = [(-5.12, 5.12)] * dimensions  # Same bounds for all dimensions

# Create optimizer
optimizer = BlackWidowOptimizer(
    objective_function=powell_sum,
    dimensions=dimensions,
    bounds=bounds,
    population_size=40,
    max_iterations=100,
    reproduction_rate=0.6,
    cannibalism_rate=0.4,
    mutation_rate=0.4,
    minimize=True  # Set to False for maximization problems
)

# Run optimization
best_solution, best_fitness = optimizer.optimize(verbose=True)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")

# Access convergence history
convergence_curve = optimizer.convergence_curve
```

### Running the Examples

The repository includes example scripts that demonstrate the algorithm:

```bash
# Run the main example with benchmark functions from the original paper
python example.py

# Run tests to verify the implementation
python test.py
```

The example script will run the optimization on the benchmark functions from the original paper and generate a convergence plot.

## Parameters

The `BlackWidowOptimizer` class accepts the following parameters:

- `objective_function`: The function to optimize
- `dimensions`: Number of dimensions of the problem
- `bounds`: List of tuples (min, max) for each dimension
- `population_size`: Size of the population (default: 40)
- `max_iterations`: Maximum number of iterations (default: 100)
- `reproduction_rate`: Percentage of population that will reproduce (default: 0.6)
- `cannibalism_rate`: Percentage of children that will be destroyed (default: 0.4)
- `mutation_rate`: Percentage of population that will mutate (default: 0.4)
- `minimize`: If True, minimize the objective function; otherwise maximize (default: True)

## Algorithm Steps

1. **Initialization**: Generate a random population of "widows" (solutions)
2. **Reproduction**: Select the best solutions to procreate and generate children
3. **Cannibalism**: Destroy some solutions with low fitness
4. **Mutation**: Randomly mutate some solutions
5. **Selection**: Create the next generation from the survivors
6. **Termination**: Return the best solution after reaching the maximum iterations

## Performance

The algorithm has been tested on the benchmark functions from the original paper and shows good convergence properties. The implementation is efficient and can handle high-dimensional optimization problems.

## References

- Hayyolalam, V., & Pourhaji Kazem, A. A. (2020). Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 87, 103249.
