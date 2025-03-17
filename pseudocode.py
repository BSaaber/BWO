#pseudocode
 
# ------------------------
 
def black_widow_algorithm(...):
    # INITIALIZATION
 
    # The initial population of black widow spiders
    # Each current_widows is a D-dimensional array of widows (chromosomes) for a D-dimensional problem
 
    current_widows = generate_random_widows()
 
    # ALGORITHM
 
    while (terminal_condition):
        # Based on procreating rate, calculate the number of reproduction “nr”;
        nr = len(current_widows) * reproduction_rate
 
        #  Select the best nr solutions that will procreate
        current_best_widows = sorted(current_widows, comparing_func=fitness)[:nr]
        widows_after_reproduction = list()
 
 
        # Procreating and cannibalism
        for i in range(nr):
            # Randomly select two solutions as parents from current_best_widows;
            parent1, parent2 = select_random_two(current_best_widows)
 
            new_gen = generate_children(parent1, parent2)
 
            father, mother = min(parent1, parent2, comparing_func=fitness), max(parent1, parent2, comparing_func=fitness)
 
            # destroying father (forgetting about him but saving info about mother widow)
            new_gen.append(mother)
            current_best_widows.delete(father)
 
            # Based on the cannibalism rate, destroy some of the children [and probably a mother] with low fitness
            new_gen = sorted(new_gen, comparing_func=fitness)
            new_gen = new_gen[: len(new_gen) * (1 - cannibalism_rate)]
 
            widows_after_reproduction.extend(new_gen)
 
        # Mutation
        # Based on the mutation rate, calculate the number of mutation children “nm”;
        nm = len(current_best_widows) * mutation_rate
        mutated_widows = list()
        for i in range(nm):
 
            # select random widow from current generation
            base_widow = pick_one(current_best_widows)
 
            # Mutate randomly one chromosome of the solution and generate a new solution;
            mutated_widow = mutate(base_widow)
 
            # save it to the new gen
            mutated_widows.append(mutated_widow)
 
        next_generation = mutated_widows + widows_after_reproduction
        current_widows = next_generation
 
    # Return the best solution from current_widows;
    return sorted(current_widows, key=fitness)[0]
 
# -----------------------------
 
# additional:
 
 
def generate_children(parent1, parent2):
    # generate D new children from parents 
    new_children = list()
    for i in range(len(parent1) / 2):
        random_coefs = generate_random_vector(size=len(parent1), min=0, max=1)
        vector_ones = [1] * len(parent1)
 
        child1 = random_coefs * parent1 + (vector_ones - random_coefs) * parent2
        child2 = random_coefs * parent2 + (vector_ones - random_coefs) * parent1
 
        new_children.extend([child1, child2])
 
    return new_children
 
def mutate(widow):
    # change places of two random numbers in a vector (widow)
 
    index_a = random_int(0, len(widow))
    index_b = random_int(0, len(widow), exclusions=[index_a])
 
    mutated_widow = copy(widow)
    mutated_widow[index_a], mutated_widow[index_b] = mutated_widow[index_b], mutated_widow[index_a]
 
    return mutated_widow