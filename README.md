# Black Widow Optimization Algorithm

Python implementation of the Black Widow Optimization Algorithm for solving optimization problems, inspired by black widow spider mating behavior.

## Overview

The Black Widow Optimization Algorithm is a population-based meta-heuristic algorithm introduced by Hayyolalam & Pourhaji Kazem (2020). It's inspired by black widow spider mating behavior, including sexual cannibalism.

## Project Structure

- `black_widow.py`: Algorithm implementation
- `benchmark_functions.py`: Benchmark functions from the original paper
- `example.py`: Example usage with benchmark functions
- `test.py`: Test script
- `frequency_regulation_function.py`: Power system frequency regulation functions
- `optimize_frequency_regulation.py`: Application to power system frequency regulation

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Power System Frequency Regulation Application

This repository includes an application of the algorithm to optimize PID controller parameters for power system frequency regulation.

### Application Overview

The application optimizes PID controller parameters to improve frequency stability of a two-area power system after load disturbances.

Components:
- Two-area power system model (Area 1: PV system, Area 2: Thermal system with reheat)
- PID controller optimization using ITAE criterion
- System response visualization and performance metrics

### Running the Optimization

```bash
python optimize_frequency_regulation.py
```

This will:
1. Generate system responses if needed
2. Run the optimization to find optimal PID parameters
3. Compare system responses
4. Generate plots and display metrics

### Results

The optimization improves power system frequency stability:
- ITAE value reduced by ~76% compared to system without controller
- Maximum frequency deviations reduced by ~75% in both areas
- Maximum tie-line power deviation reduced by ~76%

Output plots:
- `pid_optimization_convergence.png`: Optimization convergence
- `system_responses_comparison.png`: System response comparison

## Benchmark Functions

Benchmark functions from the original paper:

1. **F1: Powell Sum**
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

## Usage

```python
from black_widow import BlackWidowOptimizer
from benchmark_functions import powell_sum

# Define problem
dimensions = 10
bounds = [(-5.12, 5.12)] * dimensions

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
    minimize=True
)

# Run optimization
best_solution, best_fitness = optimizer.optimize(verbose=True)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### Running Examples

```bash
# Run benchmark examples
python example.py

# Run tests
python test.py
```

## Parameters

- `objective_function`: Function to optimize
- `dimensions`: Problem dimensions
- `bounds`: List of tuples (min, max) for each dimension
- `population_size`: Size of population (default: 40)
- `max_iterations`: Maximum iterations (default: 100)
- `reproduction_rate`: Percentage of population that reproduces (default: 0.6)
- `cannibalism_rate`: Percentage of children destroyed (default: 0.4)
- `mutation_rate`: Percentage of population that mutates (default: 0.4)
- `minimize`: True for minimization, False for maximization (default: True)

## Algorithm Steps

1. **Initialization**: Generate random population
2. **Reproduction**: Select best solutions to procreate
3. **Cannibalism**: Destroy solutions with low fitness
4. **Mutation**: Randomly mutate solutions
5. **Selection**: Create next generation from survivors
6. **Termination**: Return best solution after reaching maximum iterations

## References

- Hayyolalam, V., & Pourhaji Kazem, A. A. (2020). Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 87, 103249.
