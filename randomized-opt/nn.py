import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

seed = 123

## Setup
print("######### Setup #########")
data = pd.read_csv("randomized-opt/NN/data.csv")
data["Class"] = data["Class"].map({"RB":1, "NRB":0})
data = data.to_numpy()
X_tr, X_test, y_tr, y_test = \
    train_test_split(data[:,0:-1],data[:,-1], train_size=0.7, 
                     random_state=seed, shuffle=True, stratify=data[:,-1])
scaler = StandardScaler().fit(X_tr)
print(f"Standardization mean: \n{scaler.mean_}")
print(f"Standardization std: \n{scaler.scale_}")

X_train = scaler.transform(X_tr)
train_num = round(0.7 * X_train.shape[0])
X_train_std = X_train[:train_num,:]
X_valid_std = X_train[train_num:,:]
y_train = y_tr[:train_num]
y_valid = y_tr[train_num:]
X_test_std = scaler.transform(X_test)

print(f"Data Sizes: \n  X_train: {X_train_std.shape}\n  X_valid: {X_valid_std.shape}\n  X_test: {X_test.shape}\n  y_train: {y_train.shape}\n  y_valid: {y_valid.shape}\n  y_test: {y_test.shape}\n")

hidden_nodes = [132]
iters = [10,50,100,300,500,800,1000,2000,5000]
final_its = 10000

for i in iters:
    print(f"Iters: {i}")

    #randomized hill climbing
    rhc_losses = []
    rhc_times = []
    rhc_train = []
    rhc_valid = []

    nn_rhc = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=False, 
                            algorithm='random_hill_climb', max_iters=i, random_state=seed,
                            learning_rate=1.7,
                            early_stopping=False,
                            restarts=100)

    start = time.time()
    nn_rhc.fit(X_train_std, y_train)
    end = time.time()

    train_pred = nn_rhc.predict(X_train_std)
    valid_pred = nn_rhc.predict(X_valid_std)
    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)

    rhc_losses.append(nn_rhc.loss)
    rhc_times.append(end-start)
    rhc_train.append(train_acc)
    rhc_valid.append(valid_acc)

    print(f"  RHC Valid Acc: {valid_acc}")

    # simulated annealing
    sa_losses = []
    sa_times = []
    sa_train = []
    sa_valid = []

    sch = mlr.GeomDecay() #mlr.ArithDecay mlr.ExpDecay
    nn_sa = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=False,
                            algorithm='random_hill_climb', max_iters=i, random_state=seed,
                            learning_rate=3.57,
                            early_stopping=False,
                            schedule=sch)

    start = time.time()
    nn_sa.fit(X_train_std, y_train)
    end = time.time()
    train_pred = nn_sa.predict(X_train_std)
    valid_pred = nn_sa.predict(X_valid_std)
    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)

    sa_losses.append(nn_rhc.loss)
    sa_times.append(end-start)
    sa_train.append(train_acc)
    sa_valid.append(valid_acc)

    print(f"  SA Valid Acc: {valid_acc}")

    # genetic algorithm
    ga_losses = []
    ga_times = []
    ga_train = []
    ga_valid = []

    nn_ga = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=False,
                            algorithm='random_hill_climb', max_iters=i, random_state=seed,
                            learning_rate=2.106,
                            early_stopping=False,
                            pop_size=400,
                            mutation_prob= 0.2)

    start = time.time()
    nn_ga.fit(X_train_std, y_train)
    end = time.time()
    train_pred = nn_ga.predict(X_train_std)
    valid_pred = nn_ga.predict(X_valid_std)
    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)

    ga_losses.append(nn_rhc.loss)
    ga_times.append(end-start)
    ga_train.append(train_acc)
    ga_valid.append(valid_acc)

    print(f"  GA Valid Acc: {valid_acc}")


#randomized hill climbing
print("\n######### RHC #########")

nn_rhc = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True, 
                        algorithm='random_hill_climb', max_iters=final_its, random_state=seed,
                        learning_rate=1.7,
                        early_stopping=False,
                        restarts=100)

start = time.time()
nn_rhc.fit(X_train_std, y_train)
end = time.time()

train_pred = nn_rhc.predict(X_train_std)
valid_pred = nn_rhc.predict(X_valid_std)
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

rhc_losses.append(nn_rhc.loss)
rhc_times.append(end-start)
rhc_train.append(train_acc)
rhc_valid.append(valid_acc)

# print(f"Fitness:\n{nn_rhc.fitness_curve}")
# print(f"Loss:\n{nn_rhc.loss}")
print(f"Training Accuracy:\n  {train_acc}")
print(f"Valid Accuracy:\n  {valid_acc}")


# simulated annealing
print("\n\n######### SA #########")

sch = mlr.GeomDecay() #mlr.ArithDecay mlr.ExpDecay
nn_sa = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True,
                        algorithm='random_hill_climb', max_iters=final_its, random_state=seed,
                        learning_rate=3.57,
                        early_stopping=False,
                        schedule=sch)

nn_sa.fit(X_train_std, y_train)
train_pred = nn_sa.predict(X_train_std)
valid_pred = nn_sa.predict(X_valid_std)
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

sa_losses.append(nn_rhc.loss)
sa_times.append(end-start)
sa_train.append(train_acc)
sa_valid.append(valid_acc)

# print(f"Fitness:\n{nn_sa.fitness_curve}")
# print(f"Loss:\n{nn_sa.loss}")
print(f"Training Accuracy:\n  {train_acc}")
print(f"Valid Accuracy:\n  {valid_acc}")


# #genetic algorithm
print("\n\n######### GA #########")

nn_ga = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True,
                        algorithm='random_hill_climb', max_iters=final_its, random_state=seed,
                        learning_rate=2.106,
                        early_stopping=False,
                        pop_size=400,
                        mutation_prob= 0.2)

nn_ga.fit(X_train_std, y_train)
train_pred = nn_ga.predict(X_train_std)
valid_pred = nn_ga.predict(X_valid_std)
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

ga_losses.append(nn_rhc.loss)
ga_times.append(end-start)
ga_train.append(train_acc)
ga_valid.append(valid_acc)

# print(f"Fitness:\n{nn_ga.fitness_curve}")
# print(f"Loss:\n{nn_ga.loss}")
print(f"Training Accuracy:\n  {train_acc}")
print(f"Valid Accuracy:\n  {valid_acc}")


## Plotting
print("Plotting\n")
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(nn_rhc.fitness_curve, label="Randomized Hill Climbing")
ax1.plot(nn_sa.fitness_curve, label="Simulated Annealing")
ax1.plot(nn_ga.fitness_curve, label="Genetic Algorithm")

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.set_title(f"Loss vs. Iterations")

plt.legend()
plt.savefig("NN/FinalIt.png")

iters.append(final_its)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(iters,rhc_losses, label="Randomized Hill Climbing")
ax2.plot(iters,sa_losses, label="Simulated Annealing")
ax2.plot(iters,ga_losses, label="Genetic Algorithm")

ax2.set_xlabel("Problem Size")
ax2.set_ylabel("Final Fitness")
ax2.set_title(f"Final Fitness vs. Problem Size")

plt.legend()
plt.savefig("NN/Part2losses.png")

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

ax3.plot(iters,rhc_times, label="Randomized Hill Climbing")
ax3.plot(iters,sa_times, label="Simulated Annealing")
ax3.plot(iters,ga_times, label="Genetic Algorithm")

ax3.set_xlabel("Problem Size")
ax3.set_ylabel("Runtime (s)")
ax3.set_title(f"Runtime vs. Problem Size")

plt.legend()
plt.savefig("NN/Part2Times.png")

# ## Saving data
# part2df = pd.DataFrame()
# part2df["rhc_losses"] = rhc_losses
# part2df["sa_losses"] = sa_losses
# part2df["ga_losses"] = ga_losses
# part2df["m_losses"] = m_losses
# part2df["rhc_times"] = rhc_times
# part2df["sa_times"] = sa_times
# part2df["ga_times"] = ga_times
# part2df["m_times"] = m_times

# part2df.to_csv("NN/Part2data.csv")

# part1df = pd.DataFrame()
# part1df["rhc_curve"] = rhc_curve
# part1df["sa_curve"] = sa_curve
# part1df["ga_curve"] = ga_curve
# part1df["m_curve"] = m_curve

# part1df.to_csv("NN/Part1data.csv")