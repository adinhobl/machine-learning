import mlrose_hiive as mlr
import numpy as np
import time

seed=121
np.random.seed(seed=seed)

# Define the first problem:
# Want the largest sum. Each term is found by multiplying the index of the term times
# the value of the term. This incentivizes later numbers being larger. The second condition is
# that the value of each term has to be either odd or even like the index, or its counts 
# against the total.

def toothymax(state, punishment=0):
    """
    Total equals the sum of the number times its index. Numbers that aren't the same parity as 
    their index are punished. Reused numbers are punished.
    """
    fitness = 0.0
    tracker = []
    for i in range(len(state)):
        if state[i] in tracker:
            fitness += punishment
        elif state[i] % 2 == 0 and i % 2 == 0:
            fitness += (state[i] * i)**0.25
            tracker.append(state[i])
        elif state[i] % 2 != 0 and i % 2 != 0:
            fitness += (state[i] * i)**0.25
            tracker.append(state[i])
        else:
            fitness += punishment
            tracker.append(state[i])
    
    return fitness

print(f"== Setup ==\n")

prob_len = 100
attemps = 1000
print(f"  Problem Length: {prob_len}\n")

init = np.random.choice(prob_len, size=prob_len, replace=False)

fitness_cust = mlr.CustomFitness(toothymax)
problem = mlr.DiscreteOpt(length=prob_len, fitness_fn=fitness_cust, 
                          maximize=True, max_val=prob_len)

total_time = 0.0
print(f"  Initialization:\n{init}")

## Randomized Hill Climbing
print(f"\n== Randomized Hill Climbing ==")
start = time.time()
best_rhc_state, best_rhc_fitness, rhc_curve = mlr.random_hill_climb(problem, 
                                                          max_attempts=attemps,
                                                          random_state=seed, curve=True,
                                                          init_state=init, restarts=10)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_rhc_fitness}")
print(f"  Best State:\n{best_rhc_state}")
print(f"  Fitness Curve:\n{rhc_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_rhc_state))

## Simulated Annealing
print(f"\n== Simulated Annealing - ExpDecay ==")
schedule = mlr.ExpDecay()
start = time.time()
best_saE_state, best_saE_fitness, saE_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, init_state=init)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_saE_fitness}")
print(f"  Best State:\n{best_saE_state}")
print(f"  Fitness Curve:\n{saE_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_saE_state))

## Simulated Annealing
print(f"\n== Simulated Annealing - GeomDecay ==")
schedule = mlr.GeomDecay()
start = time.time()
best_saG_state, best_saG_fitness, saG_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, init_state=init)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_saG_fitness}")
print(f"  Best State:\n{best_saG_state}")
print(f"  Fitness Curve:\n{saG_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_saG_state))

## Simulated Annealing
print(f"\n== Simulated Annealing - ArithDecay ==")
schedule = mlr.ArithDecay()
start = time.time()
best_saA_state, best_saA_fitness, saA_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, init_state=init)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_saA_fitness}")
print(f"  Best State:\n{best_saA_state}")
print(f"  Fitness Curve:\n{saA_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_saA_state))

## Genetic Algorithm
print(f"\n== Genetic Algorithm ==")
start = time.time()
best_ga_state, best_ga_fitness, ga_curve = mlr.genetic_alg(problem, pop_size=100, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, mutation_prob=0.1)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_ga_fitness}")
print(f"  Best State:\n{best_ga_state}")
print(f"  Fitness Curve:\n{ga_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_ga_state))




print(f"\n\nTotal Time Elapsed: {total_time}")

print(toothymax(np.arange(0,prob_len)))