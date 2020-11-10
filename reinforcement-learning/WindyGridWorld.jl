# Code From JuliaReinforcementLearningAnIntroduction
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl/blob/master/src/environments/WindyGridWorld.jl

# export WindyGridWorld
using GridWorlds

mutable struct WindyGridWorld <: AbstractGridWorld
    world::GridWorldBase{Tuple{Empty,Wall,Goal}}
    agent_pos::CartesianIndex{2}
    agent::Agent
    goal_reward::Float64
    reward::Float64
end

function WindyGridWorld(
    ;width=9, height=11, 
    agent_start_pos=CartesianIndex(4,1), 
    goal_pos=CartesianIndex(4,8)
    )

    objects = (EMPTY, WALL, GOAL)
    world = GridWorldBase(objects, height, width)
    world[EMPTY,:,:] .= true
    world[WALL,[1,end],:] .= true
    world[WALL,:,[1,end]] .= true
    world[GOAL,goal_pos] = true
    goal_reward = 1.0
    reward = 0.0

    return WindyGridWorld(world, agent_start_pos, Agent(dir=RIGHT), goal_reward, reward)
end





const Wind = [CartesianIndex(w, 0) for w in [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]]

const Actions = [
    CartesianIndex(0, -1),  # left
    CartesianIndex(0, 1),   # right
    CartesianIndex(-1, 0),  # up
    CartesianIndex(1, 0),   # down
]

const LinearInds = LinearIndices((NX, NY))



RLBase.get_state(env::WindyGridWorld) = DiscreteSpace(length(LinearInds))
RLBase.get_actions(env::WindyGridWorld) = DiscreteSpace(length(Actions))

function (env::WindyGridWorld)(a::Int)
    p = env.position + Wind[env.position[2]] + Actions[a]
    p = CartesianIndex(min(max(p[1], 1), NX), min(max(p[2], 1), NY))
    env.position = p
    if p == Goal
        env.terminal = true
    end
    nothing
end

observe(env::WindyGridWorld) =
    (
        state = LinearInds[env.position],
        terminal = env.position == Goal,
        reward = env.position == Goal ? 0.0 : -1.0,
    )

function RLBase.reset!(env::WindyGridWorld)
    env.position = StartPosition
    nothing
end

get_terminal(env::WindyGridWorld) = env.terminal

end
