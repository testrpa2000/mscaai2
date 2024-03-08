import random

def generate_random_gene(genes):
    return random.choice(genes)

def generate_random_individual(target_string, genes):
    return ''.join(generate_random_gene(genes) for _ in range(len(target_string)))

def calculate_fitness(individual, target_string):
    return sum(1 for a, b in zip(individual, target_string) if a == b)

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual, mutation_rate, genes):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = generate_random_gene(genes)
    return ''.join(mutated_individual)

def genetic_algorithm(target_string, genes, population_size, mutation_rate):
    # Initialize population
    population = [generate_random_individual(target_string, genes) for _ in range(population_size)]

    generation = 1
    while True:
        # Evaluate fitness of each individual in the population
        fitness_scores = [calculate_fitness(individual, target_string) for individual in population]

        # Check for a perfect match
        if max(fitness_scores) == len(target_string):
            print("Target string reached!")
            break

        # Select the top individuals for reproduction
        selected_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:10]
        selected_parents = [population[i] for i in selected_indices]

        # Create a new population through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected_parents, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, genes)
            new_population.append(child)

        population = new_population

        # Print progress
        print(f"Generation {generation}: {max(fitness_scores)} / {len(target_string)}")
        generation += 1

if __name__ == "__main__":
    # Get user input
    target_string = input("Enter the target string: ")
    genes = input("Enter the possible genes (characters): ")
    population_size = int(input("Enter the population size: "))
    mutation_rate = float(input("Enter the mutation rate: "))
    genetic_algorithm(target_string, genes, population_size, mutation_rate)


# Enter the target string: HELLO/RAHUL
# Enter the possible genes (characters): ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz !"/AGSRAJDHUOL
# Enter the population size: 10
# Enter the mutation rate: 0.1
# Generation 1: 0 / 5
# Generation 2: 1 / 5
# Generation 3: 1 / 5
# Generation 4: 2 / 5
# Generation 5: 3 / 5
# Generation 6: 3 / 5
# Generation 7: 4 / 5
# Generation 8: 4 / 5
# Generation 9: 5 / 5
# Target string reached!




# 1. **Import Random Module**: The code begins by importing the `random` module, which is used for generating random numbers.

# 2. **Function to Generate a Random Gene**: The `generate_random_gene` function takes a list of genes and returns a randomly chosen gene from that list.

# 3. **Function to Generate a Random Individual**: The `generate_random_individual` function takes a target string and a list of genes. It generates a random individual by randomly selecting genes from the provided list to create a string of the same length as the target string.

# 4. **Function to Calculate Fitness**: The `calculate_fitness` function calculates the fitness score of an individual by comparing it to the target string. It returns the number of characters in the individual that match the corresponding characters in the target string.

# 5. **Crossover Function**: The `crossover` function takes two parent individuals and performs crossover to produce a child individual. It randomly selects a crossover point and combines parts of the parents before and after that point to create the child.

# 6. **Mutation Function**: The `mutate` function takes an individual, a mutation rate, and a list of genes. It randomly mutates each gene in the individual with the given mutation rate by replacing it with a randomly chosen gene from the list of genes.

# 7. **Genetic Algorithm Function**: The `genetic_algorithm` function implements the main genetic algorithm loop. It initializes a population of random individuals, evaluates their fitness scores, selects the top individuals for reproduction, creates a new population through crossover and mutation, and continues this process until a perfect match for the target string is found.

# 8. **Main Section**: In the main section of the code, user input is collected for the target string, possible genes, population size, and mutation rate. Then, the `genetic_algorithm` function is called with these parameters to execute the genetic algorithm.

# 9. **Execution**: During execution, the code continuously prints the progress of each generation, showing the highest fitness score achieved so far out of the length of the target string.

# This code demonstrates a basic implementation of a genetic algorithm for finding a target string using a population of randomly generated individuals, crossover, and mutation operations.