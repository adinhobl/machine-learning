using Plots
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using TabularTDLearning
using DiscreteValueIteration


## Small, non tabular - Tiger POMDP

# Value Iteration


# Policy Iteration


# Q-Learning





## Large, tabular - GridWorld
include("gridworld.jl")
mdp = GridWorld()
mdp.tprob=1.0

# Value Iteration
solver = ValueIterationSolver(max_iterations=100, belres=1e-4; verbose=true)
policy = solve(solver, mdp)

# Policy Iteration



# Q-Learning
exppolicy = EpsGreedyPolicy(mdp, 0.05)
solver = QLearningSolver(exploration_policy=exppolicy, n_episodes=200, max_episode_length=50, learning_rate=0.001, eval_every=1, n_eval_traj=100; verbose=true)
policy = solve(solver, mdp)







for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps=20)
    @show s
    @show a
    @show r
    println()
end