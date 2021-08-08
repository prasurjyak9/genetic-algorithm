import random
import math

from numpy.random import choice

N = 20
sols_per_pop = 88
num_gens = 1000
num_parents_mating = 22
MUTATION_LIMIT = 1

coeff = random.sample(range(-30, 30), N)
test_pts = random.sample(range(-1000, 1000), 150)


def f(x, coeff):
    sum = 0
    for i in range(N//2):
        sum += coeff[i] * math.cos(i*x)
        sum += coeff[N//2+i] * math.sin((N//2+i)*x)
    return sum

def generate_initial_population():
    population = []
    for i in range(sols_per_pop):
        sol =  random.sample(range(0, 100), N)
        population.append(sol)
    return population

def calc_sol_fitness(sol):
    error = 0
    for pt in test_pts:
        error += (f(pt, coeff)-f(pt, sol))*(f(pt, coeff)-f(pt, sol))
    return math.sqrt(error / len(test_pts))

def calc_sol_perc_error(sol):
    error = 0
    for pt in test_pts:
        error += abs((f(pt, coeff) - f(pt, sol) / f(pt, coeff)))
    return error

def calc_pop_fitness(population):
    fitness = []
    for sol in population:
        fitness.append(calc_sol_fitness(sol))
    return fitness

def select_mating_pool_by_fittest(pop, fitness, num_parents):
    parents = []
    fitness_copy = fitness.copy()
    for i in range(num_parents):
        idx_of_parent = fitness_copy.index(min(fitness_copy))
        parents.append(pop[idx_of_parent])
        fitness_copy[idx_of_parent] = 999999999
    return parents

def get_prob_without_ranking(fitness):
    fitness_sum = sum(fitness)
    inv_probs = [f/fitness_sum for f in fitness]
    probs = [(1.0-p) for p in inv_probs]
    probs = [p/sum(probs) for p in probs]
    return probs

def get_prob_with_ranking(fitness):
    tmp = sorted(fitness, reverse=True)
    probs_tmp = [tmp.index(x) for x in fitness]
    probs_sum = sum(probs_tmp)
    probs = [p/probs_sum for p in probs_tmp]
    return probs


def select_mating_pool_by_roulette(pop, fitness, num_parents):
    ret_idx = set()
    probs = get_prob_with_ranking(fitness)
    while len(ret_idx) != num_parents:
        idx = choice(range(len(probs)), p=probs)
        if idx not in ret_idx:
            ret_idx.add(idx)
    return [pop[idx] for idx in ret_idx]


def generate_offspring_alternate_pick(parent1, parent2):
    offspring = []
    for i in range(len(parent1)):
        if i % 2 == 0:
            offspring.append(parent1[i])
        else:
            offspring.append(parent2[i])
    return offspring


def generate_offspring_random_pick(parent1, parent2):
    offspring = []
    for i in range(len(parent1)):
        b = random.randint(0,1)
        if b == 0:
            offspring.append(parent1[i])
        else:
            offspring.append(parent2[i])
    return offspring

def generate_offspring_heuristic(parent1, parent2):
    alpha = 0.5
    fit, weak = parent1, parent2
    if calc_sol_fitness(fit) > calc_sol_fitness(weak):
        fit, weak = weak, fit
    offspring = []
    for i in range(len(parent1)):
        offspring.append(alpha * (fit[i]*fit[i] - weak[i]*weak[i]) + weak[i]*weak[i])
    return offspring

def generate_offspring_geometric_mean(parent1, parent2):
    offspring = []
    for i in range(len(parent1)):
        offspring.append(math.sqrt(parent1[i]*parent2[i]))
    return offspring

def generate_offspring_arithmetic_mean_of_single_idx(parent1, parent2):
    rand_idx = random.randrange(len(parent1))
    parent1[rand_idx] = (parent1[rand_idx] + parent2[rand_idx]) / 2
    return parent1

def generate_offspring_arithmetic_mean(parent1, parent2):
    offspring = []
    for i in range(len(parent1)):
        offspring.append((parent1[i]+parent2[i]) / 2)
    return offspring

def generate_offspring_onepoint(parent1, parent2):
    offspring = []
    p1f = calc_sol_fitness(parent1)
    p2f = calc_sol_fitness(parent2)
    if p1f < p2f:
        for i in range(len(parent1)):
            if i < n*(.75):
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
    else:
        for i in range(len(parent1)):
            if i > n*(0.75):
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
    return offspring

def generate_offspring_random_biased(parent1, parent2):
    offspring = []
    p1f = calc_sol_fitness(parent1)
    p2f = calc_sol_fitness(parent2)
    if p1f < p2f:
        for i in range(len(parent1)):
            if random.random() < 0.7:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])       
    else:
        for i in range(len(parent1)):
            if random.random() < 0.7:
                offspring.append(parent2[i])
            else:
                offspring.append(parent1[i])
        
    return offspring

def crossover(parents, num_offsprings):
    offprings = []
    for k in range(num_offsprings):
        parent_1 = parents[k % len(parents)]
        parent_2 = parents[(k+1) % len(parents)]
        child = generate_offspring_random_biased(parent_1, parent_2)
        offprings.append(child)
    return offprings

def mutate_offspring(offspring):
    rand_idx = random.randrange(len(offspring))
    rand_val = random.uniform(-MUTATION_LIMIT, MUTATION_LIMIT)
    offspring[rand_idx] += rand_val
    return offspring


def mutate_offspring_gd(offspring):
    rand_idx = random.randrange(len(offspring))
    rand_val = random.uniform(0, MUTATION_LIMIT)
    offspring[rand_idx] += rand_val
    err1 = calc_sol_fitness(offspring)
    offspring[rand_idx] -= 2*rand_val
    err2 = calc_sol_fitness(offspring)
    if(err1<err2):
        offspring[rand_idx] += 2*rand_val
    return offspring


def mutation(offspring_crossover):
    mutated_offsprings = []
    for offspring in offspring_crossover:
        mutated_offsprings.append(mutate_offspring(offspring))
    return mutated_offsprings

def best_solution_in_population(population):
    mn = 10000000000
    ret = None
    for sol in population:
        if calc_sol_fitness(sol) < mn:
            mn = calc_sol_fitness(sol)
            ret = sol
    return ret

def add_offsprings(population, offspring_mutation, fitness):
    idx_to_chop = []
    for i in range(len(offspring_mutation)):
        idx_of_max = fitness.index(max(fitness))
        idx_to_chop.append(idx_of_max)
        fitness[idx_of_max] = -1000000000

    for i in range(len(offspring_mutation)):
        population[idx_to_chop[i]] = offspring_mutation[i]
    return population 


def add_offsprings_with_prob(population, offspring_mutation, fitness):
    tmp = sorted(fitness)
    bb = [tmp.index(f) for f in fitness]
    probs = [x/sum(bb) for x in bb]
    indexes = []
    while len(indexes) != len(offspring_mutation):
        idx = choice(range(len(probs)), p=probs)
        if idx not in indexes:
            indexes.append(idx)
    for i in range(len(offspring_mutation)):
        idx = indexes[i]
        population[idx] = offspring_mutation[i]
    return population


def main():
    global MUTATION_LIMIT
    population = generate_initial_population()
    print(best_solution_in_population(population))

    for gen in range(num_gens):
        fitness = calc_pop_fitness(population)
        print("gen={g}, min_fitness={mf}".format(g=gen, mf=min(fitness)))
        parents = select_mating_pool_by_roulette(population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, num_offsprings=len(population)-len(parents))
        offspring_mutation = mutation(offspring_crossover)
        population = add_offsprings_with_prob(population, offspring_mutation, fitness)

    print(coeff)
    print(best_solution_in_population(population))

    print(calc_sol_perc_error(best_solution_in_population(population)))


if __name__ == '__main__':
    main()
