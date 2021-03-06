import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

seed = 187
np.random.seed(seed=seed)

print(f"######### SETUP #########\n")


max_it = 200
attempts = max_it * 3/4

rhc_fitnesses = []
sa_fitnesses = []
ga_fitnesses = []
m_fitnesses = []
rhc_times = []
sa_times = []
ga_times = []
m_times = []

test_range = [4,6,8,10,15,20,25]
final_prob_len = 30

fitness_cust = mlr.SixPeaks(t_pct=0.1)
schedule = mlr.GeomDecay(decay=0.99)

print(f"Attempts:            {attempts}")
print(f"Max Iterations:      {max_it}")
print(f"Problem Sizes:       {test_range}")
print(f"Last Problem Size:   {final_prob_len}\n\n")

part2_time = 0.0

print(f"\n######### PART 2 #########\n")

for i in test_range:
    start = time.time()
    print(f"Running for subproblem size: {i}")

    init = np.random.choice([0,1], size=i, replace=True)
    problem = mlr.DiscreteOpt(length=i, fitness_fn=fitness_cust, 
                          maximize=True, max_val=2)

    ## Randomized Hill Climbing
    sub_start = time.time()
    best_rhc_state, best_rhc_fitness, rhc_curve = mlr.random_hill_climb(problem, 
                                                            max_attempts=attempts,
                                                            random_state=seed, curve=True,
                                                            init_state=init, restarts=400)
    sub_end = time.time()
    rhc_fitnesses.append(best_rhc_fitness)
    rhc_times.append(sub_end - sub_start)

    ## Simulated Annealing
    sub_start = time.time()
    best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                            max_attempts=attempts, random_state=seed, 
                                                            curve=True, init_state=init)
    sub_end = time.time()
    sa_fitnesses.append(best_sa_fitness)
    sa_times.append(sub_end - sub_start)

    ## Genetic Algorithm
    sub_start = time.time()
    best_ga_state, best_ga_fitness, ga_curve = mlr.genetic_alg(problem, pop_size=200, 
                                                            max_attempts=attempts, random_state=seed, 
                                                            curve=True, mutation_prob=0.1)                                           
    sub_end = time.time()
    ga_fitnesses.append(best_ga_fitness)
    ga_times.append(sub_end - sub_start)

    ## MIMIC
    sub_start = time.time()
    best_m_state, best_m_fitness, m_curve = mlr.mimic(problem, pop_size=400, 
                                                    max_attempts=attempts, 
                                                    random_state=seed, curve=True)                                                   
    sub_end = time.time()
    m_fitnesses.append(best_m_fitness)
    m_times.append(sub_end - sub_start)

    end = time.time()
    part2_time += end - start
    print(f"        Iteration Time Elapsed: {end-start}\n")

print(f"\n\nPart 2 Time Elapsed: {part2_time}\n")


print(f"######### PART 1 #########\n")

print(f"  Problem Length: {final_prob_len}\n")
# print(f"  Max Fitness: {fitness_cust.evaluate(np.arange(0,final_prob_len))}\n")

init = np.random.choice([0,1], size=final_prob_len, replace=True)
problem = mlr.DiscreteOpt(length=final_prob_len, fitness_fn=fitness_cust, 
                          maximize=True, max_val=2)

part1_time = 0.0
print(f"  Initialization:\n{init}")

## Randomized Hill Climbing
print(f"\n=== Randomized Hill Climbing ===")
start = time.time()
best_rhc_state, best_rhc_fitness, rhc_curve = mlr.random_hill_climb(problem, 
                                                          max_attempts=attempts,
                                                          random_state=seed, curve=True,
                                                          init_state=init, restarts=400,
                                                          max_iters=max_it)
                                                          
end = time.time()
part1_time += end - start

rhc_fitnesses.append(best_rhc_fitness)
rhc_times.append(end - start)

print(f"  Fitness: {best_rhc_fitness}")
print(f"  Best State:\n{best_rhc_state}")
print(f"  Fitness Curve:\n{rhc_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_rhc_state))

# # Simulated Annealing
# print(f"\n=== Simulated Annealing - ExpDecay ===")
# schedule = mlr.ExpDecay()
# start = time.time()
# best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
#                                                           max_attempts=attempts, random_state=seed, 
#                                                           curve=True, init_state=init,
#                                                           max_iters=max_it)
                                                          
# end = time.time()
# part1_time += end - start

# rhc_fitnesses.append(best_sa_fitness)
# rhc_times.append(end - start)

# print(f"  Fitness: {best_sa_fitness}")
# print(f"  Best State:\n{best_sa_state}")
# print(f"  Fitness Curve:\n{sa_curve}")
# print(f"  Elapsed: {end - start}")

# # # print(sorted(best_sa_state))

# Simulated Annealing
print(f"\n=== Simulated Annealing - GeomDecay ===")
schedule = mlr.GeomDecay()
start = time.time()
best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
                                                          max_attempts=attempts, random_state=seed, 
                                                          curve=True, init_state=init,
                                                          max_iters=max_it)
                                                          
end = time.time()
part1_time += end - start

sa_fitnesses.append(best_sa_fitness)
sa_times.append(end - start)

print(f"  Fitness: {best_sa_fitness}")
print(f"  Best State:\n{best_sa_state}")
print(f"  Fitness Curve:\n{sa_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_sa_state))

# ## Simulated Annealing
# print(f"\n=== Simulated Annealing - ArithDecay ===")
# schedule = mlr.ArithDecay()
# start = time.time()
# best_sa_state, best_sa_fitness, sa_curve = mlr.simulated_annealing(problem, schedule=schedule, 
#                                                           max_attempts=attempts, random_state=seed, 
#                                                           curve=True, init_state=init,
#                                                           max_iters=max_it)
                                                          
# end = time.time()
# part1_time += end - start

# sa_fitnesses.append(best_sa_fitness)
# sa_times.append(end - start)

# print(f"  Fitness: {best_sa_fitness}")
# print(f"  Best State:\n{best_sa_state}")
# print(f"  Fitness Curve:\n{sa_curve}")
# print(f"  Elapsed: {end - start}")

# # print(sorted(best_sa_state))

## Genetic Algorithm
print(f"\n=== Genetic Algorithm ===")
start = time.time()
best_ga_state, best_ga_fitness, ga_curve = mlr.genetic_alg(problem, pop_size=200, 
                                                          max_attempts=attempts, random_state=seed, 
                                                          curve=True, mutation_prob=0.1,
                                                          max_iters=max_it)
                                                          
end = time.time()
part1_time += end - start

ga_fitnesses.append(best_ga_fitness)
ga_times.append(end - start)

print(f"  Fitness: {best_ga_fitness}")
print(f"  Best State:\n{best_ga_state}")
print(f"  Fitness Curve:\n{ga_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_ga_state))

## MIMIC
print(f"\n=== MIMIC ===")
start = time.time()
best_m_state, best_m_fitness, m_curve = mlr.mimic(problem, pop_size=400, 
                                                max_attempts=attempts, 
                                                random_state=seed, curve=True,
                                                max_iters=max_it)
                                                          
end = time.time()
part1_time += end - start

m_fitnesses.append(best_m_fitness)
m_times.append(end - start)

print(f"  Fitness: {best_m_fitness}")
print(f"  Best State:\n{best_m_state}")
print(f"  Fitness Curve:\n{m_curve}")
print(f"  Elapsed: {end - start}")

# print(sorted(best_m_state))

print(f"\n\nPart 1 Time Elapsed: {part1_time}\n")

print(f"\n\nTotal Time Elapsed: {part1_time + part2_time}\n")

## Plotting
print("Plotting Part 1\n")
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(rhc_curve, label="Randomized Hill Climbing")
ax1.plot(sa_curve, label="Simulated Annealing")
ax1.plot(ga_curve, label="Genetic Algorithm")
ax1.plot(m_curve, label="MIMIC")

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Fitness")
ax1.set_title(f"Fitness vs. Iterations, n={final_prob_len}")

plt.legend()
plt.savefig("Opt3/Part1.png")

# Plotting
print("Plotting Part 2\n\n")

test_range.append(final_prob_len)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)

print(test_range, rhc_fitnesses)

ax2.plot(test_range,rhc_fitnesses, label="Randomized Hill Climbing")
ax2.plot(test_range,sa_fitnesses, label="Simulated Annealing")
ax2.plot(test_range,ga_fitnesses, label="Genetic Algorithm")
ax2.plot(test_range,m_fitnesses, label="MIMIC")

ax2.set_xlabel("Problem Size")
ax2.set_ylabel("Final Fitness")
ax2.set_title(f"Final Fitness vs. Problem Size")

plt.legend()
plt.savefig("Opt3/Part2Fitnesses.png")

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

ax3.plot(test_range,rhc_times, label="Randomized Hill Climbing")
ax3.plot(test_range,sa_times, label="Simulated Annealing")
ax3.plot(test_range,ga_times, label="Genetic Algorithm")
ax3.plot(test_range,m_times, label="MIMIC")

ax3.set_xlabel("Problem Size")
ax3.set_ylabel("Runtime (s)")
ax3.set_title(f"Runtime vs. Problem Size")

plt.legend()
plt.savefig("Opt3/Part2Times.png")


print(f"######### PART 3 #########\n")

print(f"  Problem Length: {final_prob_len}\n")

init = np.random.choice([0,1], size=final_prob_len, replace=True)
problem = mlr.DiscreteOpt(length=final_prob_len, fitness_fn=fitness_cust, 
                          maximize=True, max_val=2)

print(init)
keeps = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
_fitnesses = []
_times = []

start = time.time()
for i in keeps:
    it_start = time.time()
    best_m_state, best_m_fitness, m_curve = mlr.mimic(problem, pop_size=400, 
                                                max_attempts=attempts, 
                                                random_state=seed, curve=True,
                                                max_iters=175, keep_pct=i)
    it_end = time.time()
    print(i, best_m_fitness, best_m_state)
    _fitnesses.append(best_m_fitness)
    _times.append(it_end-it_start)

end = time.time()
part3_time = end-start

print(f"\n\nTotal Time Elapsed: {part1_time + part2_time + part3_time}\n")


# Plotting part 3
print("Plotting Part 3\n\n")

fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(keeps,_fitnesses)

ax4.set_xlabel("Proportion Kept")
ax4.set_ylabel("Final Fitness")
ax4.set_title(f"Final Fitness vs. Proportion Kept Each Iteration")

# plt.legend()
plt.savefig("Opt3/Part3keep.png")


## Saving data
part2df = pd.DataFrame()
part2df["rhc_fitnesses"] = rhc_fitnesses
part2df["sa_fitnesses"] = sa_fitnesses
part2df["ga_fitnesses"] = ga_fitnesses
part2df["m_fitnesses"] = m_fitnesses
part2df["rhc_times"] = rhc_times
part2df["sa_times"] = sa_times
part2df["ga_times"] = ga_times
part2df["m_times"] = m_times

part2df.to_csv("Opt3/Part2data.csv")

part1df = pd.DataFrame()
part1df["rhc_curve"] = rhc_curve
part1df["sa_curve"] = rhc_curve
part1df["ga_curve"] = rhc_curve
part1df["m_curve"] = m_curve

part1df.to_csv("Opt3/Part1data.csv")


