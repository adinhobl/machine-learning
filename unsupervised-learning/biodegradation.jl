## Load Modules
using MLJ
using MultivariateStats
using ScikitLearn
using Plots
using CSV

## Import Data
data = CSV.read("biodegradation.csv")

## Clustering Algorithms - Run the clustering algorithms on the datasets and describe what you see.
# K-Means


# Expectation Maximization


## Dimensionality Reduction - Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
# PCA

# ICA

# Randomized Projections

# Classical Multidimensional Scaling


## Clustering Pt 2 - Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. 
#  Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method. 
#  You should look at all of them, but focus on the more interesting findings in your report.
# Kmeans - PCA

# EM - PCA

# Kmeans - ICA

# EM - ICA

# Kmeans - Randomized Projections

# EM - Randomized Projections

# Kmeans - Classical Multidimensional Scaling

# EM - Classical Multidimensional Scaling



#### BELOW ONLY FOR 1 DATASET

## Dimensionality Reduction + NN - Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 
#  (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) 
#  and rerun your neural network learner on the newly projected data.


## Clustering + NN - Apply the clustering algorithms to the same dataset to which you just applied the dimensionality 
#  reduction algorithms (you've probably already done this), treating the clusters as if they were new features. 
#  In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. 
#  Again, rerun your neural network learner on the newly projected data.
