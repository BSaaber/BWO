import numpy as np
from typing import Callable, Tuple, List, Optional, Union


class BlackWidowOptimizer:
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 40,
        max_iterations: int = 100,
        reproduction_rate: float = 0.6,
        cannibalism_rate: float = 0.4,
        mutation_rate: float = 0.4,
        minimize: bool = True
    ):
        """
        Initialize BWO.
        
        Args:
            objective_function: Function to optimize (fitness function)
            dimensions: Number of dimensions of the problem
            bounds: List of tuples (min, max) for each dimension
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            reproduction_rate: Percentage of population that will reproduce
            cannibalism_rate: Percentage of children that will be destroyed
            mutation_rate: Percentage of population that will mutate
            minimize: If True, minimize the objective function; otherwise maximize
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.reproduction_rate = reproduction_rate
        self.cannibalism_rate = cannibalism_rate
        self.mutation_rate = mutation_rate
        self.minimize = minimize
        
        # Validate parameters
        self._validate_parameters()
        
        # Best solution found so far
        self.best_solution = None
        self.best_fitness = float('inf') if minimize else float('-inf')
        
        # Convergence history
        self.convergence_curve = []
        
        # Current population and fitness values
        self.population = None
        self.fitness_values = None
        self.current_iteration = 0
    
    def _validate_parameters(self):
        """Validate the algorithm parameters."""
        if not 0 <= self.reproduction_rate <= 1:
            raise ValueError("Reproduction rate must be between 0 and 1")
        
        if not 0 <= self.cannibalism_rate <= 1:
            raise ValueError("Cannibalism rate must be between 0 and 1")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if len(self.bounds) != self.dimensions:
            raise ValueError(f"Bounds must be provided for all {self.dimensions} dimensions")
    
    def _evaluate_fitness(self, widow: Union[np.ndarray, list]) -> float:
        """
        Evaluate the fitness of a widow (solution).
        
        Args:
            widow: A solution vector (numpy array or list)
            
        Returns:
            The fitness value
        """
        # Convert to numpy array if it's a list
        if isinstance(widow, list):
            widow = np.array(widow)
        
        # Ensure the solution is within bounds
        for i in range(self.dimensions):
            widow[i] = np.clip(widow[i], self.bounds[i][0], self.bounds[i][1])
        
        return self.objective_function(widow)
    
    def _generate_random_widows(self) -> List[np.ndarray]:
        """
        Generate a random initial population of widows.
        
        Returns:
            List of randomly generated widows (solutions)
        """
        widows = []
        for _ in range(self.population_size):
            widow = np.zeros(self.dimensions)
            for j in range(self.dimensions):
                lower_bound, upper_bound = self.bounds[j]
                widow[j] = np.random.uniform(lower_bound, upper_bound)
            widows.append(widow)
        return widows
    
    def _select_random_two(self, widows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select two random widows from the population.
        
        Args:
            widows: List of widows to select from
            
        Returns:
            Tuple of two randomly selected widows
        """
        indices = np.random.choice(len(widows), size=2, replace=False)
        return widows[indices[0]], widows[indices[1]]
    
    def _generate_children(self, parent1: np.ndarray, parent2: np.ndarray) -> List[np.ndarray]:
        """
        Generate children from two parents using crossover.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            List of children solutions
        """
        new_children = []
        # Generate D/2 pairs of children (D total children)
        for _ in range(self.dimensions // 2):
            # Generate random coefficients for crossover
            random_coefs = np.random.uniform(0, 1, self.dimensions)
            vector_ones = np.ones(self.dimensions)
            
            # Create two children using crossover
            child1 = random_coefs * parent1 + (vector_ones - random_coefs) * parent2
            child2 = random_coefs * parent2 + (vector_ones - random_coefs) * parent1
            
            new_children.extend([child1, child2])
        
        # If dimensions is odd, add one more child
        if self.dimensions % 2 != 0:
            random_coefs = np.random.uniform(0, 1, self.dimensions)
            vector_ones = np.ones(self.dimensions)
            child = random_coefs * parent1 + (vector_ones - random_coefs) * parent2
            new_children.append(child)
        
        return new_children
    
    def _mutate(self, widow: np.ndarray) -> np.ndarray:
        """
        Mutate a widow by swapping two random elements.
        
        Args:
            widow: The widow to mutate
            
        Returns:
            The mutated widow
        """
        mutated_widow = widow.copy()
        
        # Select two random indices
        index_a, index_b = np.random.choice(self.dimensions, size=2, replace=False)
        
        # Swap the values
        mutated_widow[index_a], mutated_widow[index_b] = mutated_widow[index_b], mutated_widow[index_a]
        
        return mutated_widow
    
    def _perform_reproduction(self, current_widows: List[np.ndarray], fitness_values: List[float]) -> List[np.ndarray]:
        """
        Perform reproduction and cannibalism phase of the algorithm.
        
        Args:
            current_widows: Current population of widows
            fitness_values: Fitness values of the current population
            
        Returns:
            List of widows after reproduction and cannibalism
        """
        # Calculate number of reproductions based on reproduction rate
        reproduction_count = int(len(current_widows) * self.reproduction_rate)
        reproduction_count = max(1, reproduction_count)  # Ensure at least one reproduction
        
        # Sort widows by fitness
        sorted_indices = np.argsort(fitness_values)
        if not self.minimize:
            sorted_indices = sorted_indices[::-1]  # Reverse for maximization
        
        # Select best widows for reproduction
        best_widows = [current_widows[i] for i in sorted_indices[:reproduction_count]]
        best_fitness = [fitness_values[i] for i in sorted_indices[:reproduction_count]]
        
        widows_after_reproduction = []
        
        # Create pairs of parents, ensuring each widow participates exactly once
        # Shuffle the widows to randomize pairing
        indices = list(range(len(best_widows)))
        np.random.shuffle(indices)
        
        # Process pairs
        for i in range(0, len(indices) - 1, 2):
            idx1, idx2 = indices[i], indices[i + 1]
            parent1, parent2 = best_widows[idx1], best_widows[idx2]
            fitness1, fitness2 = best_fitness[idx1], best_fitness[idx2]
            
            # Generate children
            new_gen = self._generate_children(parent1, parent2)
            
            # Evaluate fitness of children
            new_gen_fitness = [self._evaluate_fitness(child) for child in new_gen]
            
            # Determine which parent is better (father is worse, mother is better)
            if (self.minimize and fitness1 > fitness2) or \
               (not self.minimize and fitness1 < fitness2):
                mother, father = parent2, parent1
            else:
                mother, father = parent1, parent2
            
            # Add mother and children to new generation
            combined_gen = [mother] + new_gen
            combined_fitness = [self._evaluate_fitness(mother)] + new_gen_fitness
            
            # Sort by fitness
            sorted_indices = np.argsort(combined_fitness)
            if not self.minimize:
                sorted_indices = sorted_indices[::-1]  # Reverse for maximization
            
            # Apply cannibalism - keep only the best ones
            survivors_count = int(len(combined_gen) * (1 - self.cannibalism_rate))
            survivors_count = max(1, survivors_count)  # Keep at least one
            
            survivors = [combined_gen[i] for i in sorted_indices[:survivors_count]]
            widows_after_reproduction.extend(survivors)
        
        # Handle odd number of widows
        if len(indices) % 2 == 1:
            # Add the last widow directly
            widows_after_reproduction.append(best_widows[indices[-1]])
        
        return widows_after_reproduction
    
    def _perform_mutation(self, current_widows: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform mutation phase of the algorithm.
        
        Args:
            current_widows: Current population of widows
            
        Returns:
            List of mutated widows
        """
        # Calculate number of mutations based on mutation rate
        mutation_count = int(len(current_widows) * self.mutation_rate)
        mutation_count = max(1, mutation_count)  # Ensure at least one mutation
        
        mutated_widows = []
        
        for _ in range(mutation_count):
            # Select a random widow from current generation
            base_widow = current_widows[np.random.randint(0, len(current_widows))]
            
            # Mutate the widow
            mutated_widow = self._mutate(base_widow)
            
            # Add to mutated population
            mutated_widows.append(mutated_widow)
        
        return mutated_widows
    
    def initialize_population(self) -> None:
        """
        Initialize the population and evaluate initial fitness.
        This method should be called before iterate_once() when using step-by-step execution.
        """
        # Generate random population
        self.population = np.array(self._generate_random_widows())
        
        # Evaluate initial population
        self.fitness_values = [self._evaluate_fitness(widow) for widow in self.population]
        
        # Update best solution
        best_idx = np.argmin(self.fitness_values) if self.minimize else np.argmax(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        
        # Reset iteration counter and convergence curve
        self.current_iteration = 0
        self.convergence_curve = [self.best_fitness]
    
    def _initialize_population(self) -> Tuple[np.ndarray, List[float]]:
        """
        Initialize the population and evaluate initial fitness.
        
        Returns:
            Tuple of (population, fitness_values)
        """
        self.initialize_population()
        return self.population, self.fitness_values
    
    def _adjust_population_size(self, population: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Adjust the population size to match the desired population size.
        
        Args:
            population: Current population that may need size adjustment
            
        Returns:
            Tuple of (adjusted_population, fitness_values)
        """
        # If population size decreased, add random widows
        while len(population) < self.population_size:
            population.append(self._generate_random_widows()[0])
        
        # Evaluate fitness for all individuals
        fitness_values = [self._evaluate_fitness(widow) for widow in population]
        
        # If population size increased, keep only the best ones
        if len(population) > self.population_size:
            sorted_indices = np.argsort(fitness_values)
            if not self.minimize:
                sorted_indices = sorted_indices[::-1]  # Reverse for maximization
            
            population = [population[i] for i in sorted_indices[:self.population_size]]
            fitness_values = [fitness_values[i] for i in sorted_indices[:self.population_size]]
        
        return population, fitness_values
    
    def _update_best_solution(self, population: List[np.ndarray], fitness_values: List[float]) -> None:
        """
        Update the best solution found so far.
        
        Args:
            population: Current population
            fitness_values: Fitness values of the current population
        """
        best_idx = np.argmin(fitness_values) if self.minimize else np.argmax(fitness_values)
        current_best = population[best_idx]
        current_best_fitness = fitness_values[best_idx]
        
        if (self.minimize and current_best_fitness < self.best_fitness) or \
           (not self.minimize and current_best_fitness > self.best_fitness):
            self.best_solution = current_best.copy()
            self.best_fitness = current_best_fitness
        
        # Store best fitness in convergence curve
        self.convergence_curve.append(self.best_fitness)
    
    def iterate_once(self) -> None:
        """
        Perform a single iteration of the Black Widow Optimization algorithm.
        This method should be called after initialize_population() when using step-by-step execution.
        """
        if self.population is None or self.fitness_values is None:
            raise ValueError("Population not initialized. Call initialize_population() first.")
        
        # Reproduction and cannibalism
        widows_after_reproduction = self._perform_reproduction(self.population.tolist(), self.fitness_values)
        
        # Mutation
        mutated_widows = self._perform_mutation(self.population.tolist())
        
        # Create next generation
        next_generation = mutated_widows + widows_after_reproduction
        
        # Adjust population size and get updated fitness values
        self.population, self.fitness_values = self._adjust_population_size(next_generation)
        
        # Convert population to numpy array for easier indexing in GUI
        self.population = np.array(self.population)
        
        # Update best solution
        self._update_best_solution(self.population.tolist(), self.fitness_values)
        
        # Increment iteration counter
        self.current_iteration += 1
    
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run the Black Widow Optimization algorithm.
        
        Args:
            verbose: If True, print progress information
            
        Returns:
            Tuple of (best solution, best fitness)
        """
        # Initialize population
        self.initialize_population()
        
        # Main loop
        for _ in range(self.max_iterations):
            self.iterate_once()
            
            # Print progress if verbose
            if verbose and self.current_iteration % 10 == 0:
                print(f"Iteration {self.current_iteration}/{self.max_iterations}, Best fitness: {self.best_fitness}")
        
        if verbose:
            print(f"Optimization completed. Best fitness: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness
