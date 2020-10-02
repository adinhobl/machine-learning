import mlrose_hiive as mlr
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


## Prep Data
data = pd.read_csv("./NN/data.csv")
data["Class"] = data["Class"].map({"RB":1, "NRB":0})








data = data.to_numpy()
print(data)

#randomized hill climbing




#simulated annealing




#a genetic algorithm