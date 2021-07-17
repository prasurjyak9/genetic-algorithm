import random
import math

n = 20
sols_per_pop = 10
num_gens = 5000
num_parents_mating = 6

coeff = random.sample(range(10, 30), n)
test_pts = random.sample(range(-100, 100), 100)

def f(x, coeff):
    sum = 0
    for i in range(n//2):
        sum += coeff[i] * math.cos(i*x)
        sum += coeff[n//2+i] * math.sin((n//2+i)*x)
    return sum

population = []
for i in range(sols_per_pop):
    sol =  random.sample(range(10, 30), n)
    population.append(sol)

def calc_sol_fitness(sol):
    error = 0
    for pt in test_pts:
        error += abs(f(pt, coeff) - f(pt, sol))
    return error

def calc_pop_fitness(population):
    fitness = []
    for sol in population:
        fitness.append(calc_sol_fitness(sol))
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    parents = []
    fitness_copy = fitness.copy()
    for i in range(num_parents):
        idx_of_parent = fitness_copy.index(min(fitness_copy))
        parents.append(idx_of_parent)
        fitness_copy[idx_of_parent] = 999999999
    return parents

def generate_offspring(parent1, parent2):
    offspring = []
    for i in range(len(parent1)):
        if i % 2 == 0:
            offspring.append(parent1[i])
        else:
            offspring.append(parent2[i])
    return offspring

def crossover(parents, num_offsprings):
    offprings = []
    for k in range(num_offsprings):
        parent_1_idx = parents[k % len(parents)]
        parent_2_idx = parents[(k+1) % len(parents)]
        child = generate_offspring(population[parent_1_idx], population[parent_2_idx])
        offprings.append(child)
    return offprings

def mutate_offspring(offspring):
    rand_idx = random.randrange(len(offspring))
    rand_val = random.uniform(-5, 5)
    offspring[rand_idx] += rand_val
    return offspring

def mutation(offspring_crossover):
    mutated_offsprings = []
    for offspring in offspring_crossover:
        mutated_offsprings.append(mutate_offspring(offspring))
    return mutated_offsprings

for gen in range(num_gens):
    fitness = calc_pop_fitness(population)
    print(min(fitness))
    parents = select_mating_pool(population, fitness, num_parents_mating)
    offspring_crossover = crossover(parents, num_offsprings=len(population)-len(parents))
    offspring_mutation = mutation(offspring_crossover)
    idx_to_chop = []
    for i in range(len(offspring_mutation)):
        idx_of_max = fitness.index(max(fitness))
        idx_to_chop.append(idx_of_max)
        fitness[idx_of_max] = -1000000000
    
    for i in range(len(offspring_mutation)):
        population[idx_to_chop[i]] = offspring_mutation[i]

print("coeff = ", coeff)
mn = 10000000000
ret = None
for sol in population:
    if calc_sol_fitness(sol) < mn:
        mn = calc_sol_fitness(sol)
        ret = sol

print("ret = ", ret)
