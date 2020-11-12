using Plots
using GridWorlds
using ReinforcementLearning
using ReinforcementLearningBase
using ReinforcementLearningCore
using Flux: Optimise.Descent

include("iteration_methods.jl")

env = WindyGridWorld()

ns = size(env.world)[2] * size(env.world)[3]
na = length(get_legal_actions(env))

V = TabularApproximator(n_state=ns)

policy_iteration!(
    V = V,
    π = RandomPolicy(DiscreteSpace(na)),
    model = env
)








# agent = ReinforcementLearningCore.Agent(
#     policy=QBasedPolicy(
#         learner=TabularLearner(),
#         explorer=EpsilonGreedyExplorer(0.1)
#     ),
#     trajectory=Trajectory()
# );

# hook = StepsPerEpisode()
# run(agent, env, StopAfterStep(8000),hook)