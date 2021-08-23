import random 
import numpy as np
import math
import timeit
from numpy.random import choice
import sys


N = 20
POPSIZE = 88
FITTEST    = 22
REPLACE   = 66
coeff = random.sample(range(-30, 30), N)
test_pts = random.sample(range(-1000, 1000), 150)
MUTATION_LIMIT = 1
GEN = 10
# -------------------------------------------------
# choose xover and mutation function

mutate_choice = 0
xover_choice  = 0
arg_len = len(sys.argv)
if(arg_len == 3):
	mutate_choice = int(sys.argv[1])
	xover_choice  = int(sys.argv[2])

#------------------------------------------------

def f(x, coeff):
    sum = 0
    for i in range(N//2):
        sum += coeff[i] * math.cos(i*x)
        sum += coeff[N//2+i] * math.sin((N//2+i)*x)
    return sum

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

#-----------------------------------------------------------

class solution:
    def __init__(self , genome=None):
        if genome is None:
            self.genome = 100*np.random.random_sample(N) # np.array(random.sample(range(0, 100), N), dtype='d') //
        else:
            self.genome = genome
        self.fitness = calc_sol_fitness(self.genome)
        self.error   = calc_sol_perc_error(self.genome)
       
    def mutate_normal(self):
        rand_idx = random.randrange(N)
        rand_val = random.uniform(-MUTATION_LIMIT, MUTATION_LIMIT)
        self.genome[rand_idx] += rand_val
        self.fitness = calc_sol_fitness(self.genome)
        self.error   = calc_sol_perc_error(self.genome)
    def mutate_backprop(self):
        rand_idx = random.randrange(N)
        rand_val = random.uniform(0, MUTATION_LIMIT)
        self.genome[rand_idx] += rand_val
        err1 = calc_sol_fitness(self.genome)
        self.genome[rand_idx] -= 2*rand_val
        err2 = calc_sol_fitness(self.genome)
        
        if(err1<err2):
            self.genome[rand_idx] += 2*rand_val
            
        self.fitness = calc_sol_fitness(self.genome)
        self.error   = calc_sol_perc_error(self.gnome)
            
    def mutate(self , choice= mutate_choice):
        if(choice==0):
            self.mutate_normal()
        else:
            self.mutate_backprop()
    def __str__ (self):
        return "".join(str(i)+" , " for i in self.genome)


#-----------------------------------------------------------

def xover_alternate ( parent1 , parent2):
    a = parent1.genome.copy()
    b = parent2.genome
    for i in range(N):
        if i % 2 == 0:
            a[i] = b[i]
    return solution(a)

def xover_random(parnt1 , parent2):
    a = parent1.genome.copy()
    b = parent2.genome
    for i in range(N):
        if random.randint(0,1):
            a[i] = b[i]
    return solution(a)

def xover_geometric_mean( parent1 , parent2):
    a = parent1.genome
    b = parent2.genome
    return solution(np.sqrt(np.absolute(a*b))*np.sign(a))     
         
    
def xover_arithmetic_mean( parent1 , parent2):
    a = parent1.genome
    b = parent2.genome
    return solution((a+b)/2.0)


def xover_one_point( parent1 , parent2):
    a = parent1.genome.copy()
    b = parent2.genome
    if parent1.fitness < parent2.fitness :
        for i in range(N//3 , N):
            a[i] = b[i]
    else:
        for i in range(0, N//3):
            a[i] = b[i]
    return solution(a)

def xover_random_biased(parent1, parent2):
    a = parent1.genome.copy()
    b = parent2.genome
    if parent1.fitness < parent2.fitness :
        for i in range(0,N):
            if random.random() < 0.7:
                a[i] = b[i]
    else:
        for i in range(0,N):
            if random.random() < 0.3:
                a[i] = b[i]
    return solution(a)

def tournament(items, n, tsize=8):
    ret = []
    for i in range(n):
        candidates = random.sample(items, tsize)
        ret.append(min(candidates, key=lambda x: x.fitness))
    return ret

        
def generate_children(parents ,n):
    ret = []
    for i in range(n):
        parent1  = random.choice(parents)
        parent2  = random.choice(parents)
        child = xover(parent1,parent2)
        child.mutate()
        #print(child)
        ret.append(child)
    return ret
 
xoverl = [xover_alternate , xover_random , xover_geometric_mean , xover_arithmetic_mean , xover_one_point , xover_random_biased ]
xover =  xoverl[xover_choice]

# --------------------------------------------------------------------------

class population:
    def __init__(self , pop = None):
        if pop is None:
            self.pop = [solution() for i in range(POPSIZE)]
        else:
            self.pop = pop
        self.gen = 0
        self.best = min(self.pop, key=lambda x: x.fitness)
        self.best_fitness = self.best.fitness
        
        
    def __str__(self):
        return "generation->"+str(self.gen)+" best fitness -> " + str(self.best_fitness)
    
    def next_gen(self):
        parents = tournament(self.pop , FITTEST )
        children = generate_children(parents , REPLACE)
        
        for child in children:
            #if child == None:
             #   print("f")
            replace_candidates = random.sample(range(POPSIZE) , 10)
           # if np.any(self.pop == None):
            #    print("yes")
            #else:
             #   print("No")
            replace_idx = max(replace_candidates , key = lambda x : self.pop[x].fitness)
            self.pop[replace_idx] = child
            
        self.best = min(self.pop, key=lambda x: x.fitness)
        self.best_fitness = self.best.fitness
        self.gen = self.gen+1
   
#----------------------------------------------------------

p = population()

while p.gen < GEN:
	print(p)
	p.next_gen()

print("solution")
print(p.best)
print("best fitness ")
print(p.best.fitness)