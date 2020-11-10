using ReinforcementLearning
using Plots
# using GridWorlds
# using ReinforcementLearningBase
using Reexport

include("iteration_methods.jl")
include("WindyGridWorld.jl")

env = WindyGridWorldEnv()

ns = 7 * 10
na = length(get_actions(env))

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