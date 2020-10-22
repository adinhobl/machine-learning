## Load Modules
using MLJ
using MultivariateStats
using ScikitLearn
using Plots
using CSV
using DataFrames
using StatsBase
using Clustering: randindex, silhouettes, varinfo, vmeasure, mutualinfo

## Set RNG
RNG = 3552

## Import Data
data = CSV.read("biodegradation.csv")
data_stats = describe(data)
label_counts = countmap(data[:(Class)])
collect(label_counts[i] / size(data)[1] for i in keys(label_counts))


coerce!(data, :Class=>Multiclass)
schema(data)
y, X = unpack(data, ==(:Class), colname->true)
train, test = partition(eachindex(y), 0.7, shuffle=true, rng=123, stratify=values(data[:Class])) # gives 70:30 split

train_counts = countmap(data[train,:Class])
collect(train_counts[i] / size(train)[1] for i in keys(train_counts))

test_counts = countmap(data[test,:Class])
collect(test_counts[i] / size(test)[1] for i in keys(test_counts))

standardizer = Standardizer()
stand = machine(standardizer, X[train,:]) #only want to standardize on training distribution
MLJ.fit!(stand)
X_stand = MLJ.transform(stand, X);

task(model) = !model.is_supervised
models(task)

## Clustering Algorithms - Run the clustering algorithms on the datasets and describe what you see.
# K-Means
@load KMeans pkg=ParallelKMeans

k_range = 2:10

for i in k_range
    model = KMeans(k=i, rng=RNG)
end











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
