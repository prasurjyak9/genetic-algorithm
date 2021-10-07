import math
import random
import numpy as np

N_GENS = 600
N_PARTICLES = 50
N_COEFF = 20
C1 = C2 = 0.7
W = 0.8

actual_coeff = np.random.rand(N_COEFF, 1)
test_pts = np.linspace(-1000, 1000, num=100)

X = np.random.rand(N_COEFF, N_PARTICLES) * 30
V = np.random.randn(N_COEFF, N_PARTICLES) * 0.1

def f(x, coeff):
    sum = 0
    for i in range(N_COEFF//2):
        sum += coeff[i] * math.cos(i*x)
        sum += coeff[N_COEFF//2+i] * math.sin((N_COEFF//2+i)*x)
    return sum

def fitness(estimated_coeff):
	error = 0
	for pt in test_pts:
		error += abs((f(pt, actual_coeff)-f(pt, estimated_coeff)/f(pt, actual_coeff)))
	return error

pbest = X
pbest_obj = fitness(X)

gbest = X[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

print(gbest_obj)

def update():
	global V, X, pbest, pbest_obj, gbest, gbest_obj

	r = np.random.rand(2)
	V = W*V + C1*r[0]*(pbest - X) + C2*r[1]*(gbest.reshape(-1,1)-X)
	X = X + V
	obj = fitness(X)
	pbest[:, (pbest_obj>=obj)] = X[:, (pbest_obj>=obj)]
	pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
	gbest = pbest[:, pbest_obj.argmin()]
	gbest_obj = pbest_obj.min()


for i in range(N_GENS):
	update()
	print(i, gbest_obj)
