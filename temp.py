import matplotlib.pyplot as plt

filename1 = "random_biased+improved_random_mutation.txt"
filename2 = "random_biased+random_mutation.txt"
filename3 = "random_pick+improved_random_mutation.txt"
filename4 = "random_pick+random_mutation.txt"

with open(filename1) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

gens = []
rmses = []

for line in lines:
	l = line.split()
	
	if len(l) == 2 and "gen=" in l[0]:
		gen = int(l[0].replace("gen=", "").replace(",", ""))
		rmse = float(l[1].replace("min_fitness=", ""))
		gens.append(gen)
		rmses.append(rmse)

plt.plot(gens, rmses, 'k--', label='random biased crossover + improved random mutation')
#plt.show()

with open(filename2) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

gens = []
rmses = []

for line in lines:
	l = line.split()
	
	if len(l) == 2 and "gen=" in l[0]:
		gen = int(l[0].replace("gen=", "").replace(",", ""))
		rmse = float(l[1].replace("min_fitness=", ""))
		gens.append(gen)
		rmses.append(rmse)

plt.plot(gens, rmses, 'k-.', label='random biased crossover + random mutation')
#plt.show()

with open(filename3) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

gens = []
rmses = []

for line in lines:
	l = line.split()
	
	if len(l) == 2 and "gen=" in l[0]:
		gen = int(l[0].replace("gen=", "").replace(",", ""))
		rmse = float(l[1].replace("min_fitness=", ""))
		gens.append(gen)
		rmses.append(rmse)

plt.plot(gens, rmses, 'k-', label='random pick crossover + improved random mutation')
#plt.show()

with open(filename4) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

gens = []
rmses = []

for line in lines:
	l = line.split()
	
	if len(l) == 2 and "gen=" in l[0]:
		gen = int(l[0].replace("gen=", "").replace(",", ""))
		rmse = float(l[1].replace("min_fitness=", ""))
		gens.append(gen)
		rmses.append(rmse)

plt.plot(gens, rmses, 'k:', label='random pick crossover + random mutation')


plt.xlabel('Generation')
plt.ylabel('RMSE')
plt.legend(fontsize='small')
plt.show()
