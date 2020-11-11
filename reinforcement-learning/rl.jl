using Plots
using GridWorlds
using ReinforcementLearning
using ReinforcementLearningBase
using Flux: Optimise.Descent

include("iteration_methods.jl")

env = WindyGridWorld()

ns = size(env.world)[2] * size(env.world)[3]
na = length(get_legal_actions(env))

agent = Agent(
    policy=QBasedPolicy(
        learner=TDLearner(
            approximator=TabularApproximator(;n_state=ns, n_action=na),
            optimizer=Descent(0.5)
        ),
        explorer=EpsilonGreedyExplorer(0.1)
    ),
    trajectory=EpisodicCompactSARTSATrajectory()
);

hook = StepsPerEpisode()
run(agent, env, StopAfterStep(8000),hook)