import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt

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
            fitness += (state[i] * i)**0.5
            tracker.append(state[i])
        elif state[i] % 2 != 0 and i % 2 != 0:
            fitness += (state[i] * i)**0.5
            tracker.append(state[i])
        else:
            fitness += punishment
            tracker.append(state[i])
    
    return fitness

print(f"######### PART 1 #########\n")
print(f"=== Setup ===\n")

prob_len = 3
attemps = 10000
max_it = 2000
print(f"  Problem Length: {prob_len}\n")

print(f"  Max Fitness: {toothymax(np.arange(0,prob_len))}\n")

init = np.random.choice(prob_len, size=prob_len, replace=False)

fitness_cust = mlr.CustomFitness(toothymax)
problem = mlr.DiscreteOpt(length=prob_len, fitness_fn=fitness_cust, 
                          maximize=True, max_val=prob_len)

total_time = 0.0
print(f"  Initialization:\n{init}")

## Randomized Hill Climbing
print(f"\n=== Randomized Hill Climbing ===")
start = time.time()
best_rhc_state, best_rhc_fitness, rhc_curve = mlr.random_hill_climb(problem, 
                                                          max_attempts=attemps,
                                                          random_state=seed, curve=True,
                                                          init_state=init, restarts=1000,
                                                          max_iters=max_it)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_rhc_fitness}")
print(f"  Best State:\n{best_rhc_state}")
print(f"  Fitness Curve:\n{rhc_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_rhc_state))

## Simulated Annealing
# print(f"\n=== Simulated Annealing - ExpDecay ===")
# schedule = mlr.ExpDecay()
# start = time.time()
# best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
#                                                           max_attempts=attemps, random_state=seed, 
#                                                           curve=True, init_state=init,
#                                                           max_iters=max_it)
                                                          
# end = time.time()
# total_time += end - start
# print(f"  Fitness: {best_sa_fitness}")
# print(f"  Best State:\n{best_sa_state}")
# print(f"  Fitness Curve:\n{sa_curve}")
# print(f"  Elapsed: {end - start}")

## print(sorted(best_sa_state))

## Simulated Annealing
# print(f"\n=== Simulated Annealing - GeomDecay ===")
# schedule = mlr.GeomDecay()
# start = time.time()
# best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
#                                                           max_attempts=attemps, random_state=seed, 
#                                                           curve=True, init_state=init,
#                                                           max_iters=max_it)
                                                          
# end = time.time()
# total_time += end - start
# print(f"  Fitness: {best_sa_fitness}")
# print(f"  Best State:\n{best_sa_state}")
# print(f"  Fitness Curve:\n{sa_curve}")
# print(f"  Elapsed: {end - start}")

## print(sorted(best_sa_state))

## Simulated Annealing
print(f"\n=== Simulated Annealing - ArithDecay ===")
schedule = mlr.ArithDecay()
start = time.time()
best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, init_state=init,
                                                          max_iters=max_it)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_sa_fitness}")
print(f"  Best State:\n{best_sa_state}")
print(f"  Fitness Curve:\n{sa_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_sa_state))

## Genetic Algorithm
print(f"\n=== Genetic Algorithm ===")
start = time.time()
best_ga_state, best_ga_fitness, ga_curve = mlr.genetic_alg(problem, pop_size=200, 
                                                          max_attempts=attemps, random_state=seed, 
                                                          curve=True, mutation_prob=0.1,
                                                          max_iters=max_it)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_ga_fitness}")
print(f"  Best State:\n{best_ga_state}")
print(f"  Fitness Curve:\n{ga_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_ga_state))

## MIMIC
print(f"\n=== MIMIC ===")
start = time.time()
best_m_state, best_m_fitness, m_curve = mlr.mimic(problem, pop_size=200, 
                                                max_attempts=attemps, 
                                                random_state=seed, curve=True,
                                                max_iters=max_it)
                                                          
end = time.time()
total_time += end - start
print(f"  Fitness: {best_m_fitness}")
print(f"  Best State:\n{best_m_state}")
print(f"  Fitness Curve:\n{m_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_m_state))

print(f"\n\nTotal Time Elapsed: {total_time}\n")

## Plotting

print("Plotting Part 1\n")
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(rhc_curve, label="Randomized Hill Climbing")
ax1.plot(sa_curve, label="Simulated Annealing")
ax1.plot(ga_curve, label="Genetic Algorithm")
ax1.plot(m_curve, label="MIMIC")

plt.legend()
plt.show()
plt.savefig("Opt1Part1.png")


print(f"######### PART 2 #########\n")

# test_range = [3,5,8,10,15,20,30,50,100,200,500,1000]
test_range = [3,5,8,10,15,20]
total_time = 0.0

rhc_fitnesses = []
sa_fitnesses = []
ga_fitnesses = []
m_fitnesses = []
rhc_times = []
sa_times = []
ga_times = []
m_times = []

start = time.time()

for i in test_range:
    print(f"Running for subproblem size: {i}\n")

    init = np.random.choice(i, size=i, replace=False)
    problem = mlr.DiscreteOpt(length=i, fitness_fn=fitness_cust, 
                          maximize=True, max_val=i)

    ## Randomized Hill Climbing
    sub_start = time.time()
    best_rhc_state, best_rhc_fitness, rhc_curve = mlr.random_hill_climb(problem, 
                                                            max_attempts=attemps,
                                                            random_state=seed, curve=True,
                                                            init_state=init, restarts=100)
    sub_end = time.time()
    rhc_fitnesses.append(best_rhc_fitness)
    rhc_times.append(sub_end - sub_start)

    ## Simulated Annealing
    sub_start = time.time()
    best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                            max_attempts=attemps, random_state=seed, 
                                                            curve=True, init_state=init)
    sub_end = time.time()
    sa_fitnesses.append(best_sa_fitness)
    sa_times.append(sub_end - sub_start)

    ## Genetic Algorithm
    sub_start = time.time()
    best_ga_state, best_ga_fitness, ga_curve = mlr.genetic_alg(problem, pop_size=200, 
                                                            max_attempts=attemps, random_state=seed, 
                                                            curve=True, mutation_prob=0.1)                                           
    sub_end = time.time()
    ga_fitnesses.append(best_ga_fitness)
    ga_times.append(sub_end - sub_start)

    ## MIMIC
    sub_start = time.time()
    best_m_state, best_m_fitness, m_curve = mlr.mimic(problem, pop_size=200, 
                                                    max_attempts=attemps, 
                                                    random_state=seed, curve=True)                                                   
    sub_end = time.time()
    m_fitnesses.append(best_m_fitness)
    m_times.append(sub_end - sub_start)

end = time.time()
print(f"\n\nTotal Time Elapsed: {end-start}\n")

# Plotting
print("Plotting Part 2")

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(test_range,rhc_fitnesses)
ax2.plot(test_range,sa_fitnesses)
ax2.plot(test_range,ga_fitnesses)
ax2.plot(test_range,m_fitnesses)

plt.legend()
plt.show()
plt.savefig("Opt1Part2Fitnesses.png")

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

ax3.plot(test_range,rhc_times)
ax3.plot(test_range,sa_times)
ax3.plot(test_range,ga_times)
ax3.plot(test_range,m_times)

plt.legend()
plt.show()
plt.savefig("Opt1Part2Times.png")