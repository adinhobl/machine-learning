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
iters = [10,50,100,300,500,800,1000]


#randomized hill climbing
print("\n######### RHC #########")
rhc_fitnesses = []
rhc_times = []
for i in np.linspace(1.4,1.7,num=10):
    print(f"LR: {i}")
    nn_rhc = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True, 
                            algorithm='random_hill_climb', max_iters=iters[-1], random_state=seed,
                            learning_rate=i,
                            early_stopping=False,
                            restarts=100)

    start = time.time()
    nn_rhc.fit(X_train_std, y_train)
    end = time.time()

    train_pred = nn_rhc.predict(X_train_std)
    valid_pred = nn_rhc.predict(X_valid_std)
    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)

    # print(f"Fitness:\n{nn_rhc.fitness_curve}")
    # print(f"Loss:\n{nn_rhc.loss}")
    print(f"Training Accuracy:\n  {train_acc}")
    print(f"Valid Accuracy:\n  {valid_acc}")


# #simulated annealing
# print("\n\n######### SA #########")
# sa_fitnesses = []
# sa_times = []

# sch = mlr.GeomDecay() #mlr.ArithDecay mlr.ExpDecay
# nn_sa = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True,
#                            algorithm='random_hill_climb', max_iters=mi, random_state=seed,
#                            learning_rate=0.1,
#                            early_stopping=False,
#                            schedule=sch)

# nn_sa.fit(X_train_std, y_train)


# #genetic algorithm
# print("\n\n######### GA #########")
# ga_fitnesses = []
# ga_times = []

# nn_ga = mlr.NeuralNetwork(hidden_nodes=hidden_nodes, activation='sigmoid', curve=True,
#                            algorithm='random_hill_climb', max_iters=mi, random_state=seed,
#                            learning_rate=0.1,
#                            early_stopping=False,
#                            pop_size= 200,
#                            mutation_prob= 0.2)

# nn_ga.fit(X_train_std, y_train)

