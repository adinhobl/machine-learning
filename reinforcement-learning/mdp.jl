using Plots
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using TabularTDLearning

## Small, non tabular - Tiger POMDP


# Value Iteration


# Policy Iteration


# Q-Learning
exppolicy = EpsGreedyPolicy(pomdp, 0.01)
solver = QLearningSolver(exploration_policy=exppolicy, n_episodes=5000, max_episode_length=50, learning_rate=0.1, eval_every=50, n_eval_traj=100, verbose=true)
policy = solve(solver, pomdp)





## Large, tabular - 
