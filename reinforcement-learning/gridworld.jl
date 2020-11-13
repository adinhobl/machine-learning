# adapted from https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/GridWorld.ipynb

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators

struct GridWorldState
    x::Int64
    y::Int64
    done::Bool
end

GridWorldState(x::Int64, y::Int64) = GridWorldState(x, y, false)

posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y

mutable struct GridWorld <: MDP{GridWorldState, Symbol}
    size_x::Int64
    size_y::Int64
    reward_states::Vector{GridWorldState}
    reward_values::Vector{Float64}
    tprob::Float64
    discount_factor::Float64
end

# we use key worded arguments so we can change any of the values we pass in 
function GridWorld(;sx::Int64 = 10,
                    sy::Int64 = 10,
                    rs::Vector{GridWorldState} = [GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)],
                    rv::Vector{Float64} = [-10., -5, 10, 3], #period is important on 10.
                    tp::Float64 = 0.7,
                    discount_factor::Float64 = 0.9
                )
        return GridWorld(sx, sy, rs, rv, tp, discount_factor)
end

function POMDPs.states(mdp::GridWorld)
    s = GridWorldState[]
    for d = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
        push!(s, GridWorldState(x,y,d))
    end
    return s
end

POMDPs.actions(mdp::GridWorld) = [:up, :down, :left, :right]

function inbounds(mdp::GridWorld, x::Int64, y::Int64)
    if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
        return true
    else
        return false
    end
end

inbounds(mpd::GridWorld, state::GridWorldState) = inbounds(mdp, state.x, state.y)

"""
Returns the neighbors of a state, but does not account for if they are terminal or not. That should
be caught before calling this function.
"""
function _neighbors(state::GridWorldState)
    x = state.x
    y = state.y

    return [
        GridWorldState(x+1, y, false), # right
        GridWorldState(x-1, y, false), # left
        GridWorldState(x, y-1, false), # down
        GridWorldState(x, y+1, false)  # up
    ]   
end

"""
Return the transition probability from one 'state' to all its neighbors after taking an 'action'.
Probability will depend on the tprob of the calling GridWorld.
"""
function POMDPs.transition(mdp::GridWorld, state::GridWorldState, action::Symbol)
    a = action
    x = state.x
    y = state.y

    # if in terminal state
    if state.done
        return SparseCat([GridWorldState(x,y,true)], [1.0])
    elseif state in mdp.reward_states
        return SparseCat([GridWorldState(x,y,true)], [1.0])
    end

    # otherwise
    targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4)
    target = targets[a]

    probability = fill(0.0, 4)

    neigh = _neighbors(state)

    if !inbounds(mdp, neigh[target])
        # if it would go out of bounds, stay in the same cell with probability 1.0
        return SparseCat([GridWorldState(x,y)], [1.0])
    else
        probability[target] = mdp.tprob

        oob_count = sum(!inbounds(mdp, n) for n in neigh)

        new_probability = (1.0 - mdp.tprob)/(3-oob_count)

        for i in 1:4
            if inbounds(mdp, neigh[i]) && i != target
                probability[i] = new_probability
            end
        end
    end
    
    return SparseCat(neigh, probability)
end


function POMDPs.reward(mdp::GridWorld, state::GridWorldState, action::Symbol, statep::GridWorldState)
    if state.done
        return 0.0
    end
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end

POMDPs.discount(mdp::GridWorld) = mdp.discount_factor

function POMDPs.stateindex(mdp::GridWorld, state::GridWorldState)
    sd = Int(state.done + 1)
    ci = CartesianIndices((mdp.size_x, mdp.size_y, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

function POMDPs.actionindex(mdp::GridWorld, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    end
    error("Invalid GridWorld action: $act")
end

POMDPs.isterminal(mdp::GridWorld, s::GridWorldState) = s.done

POMDPs.initialstate(pomdp::GridWorld) = Deterministic(GridWorldState(1,1))


### tests below ###

# mdp = GridWorld()
# mdp.tprob=1.0

# policy = RandomPolicy(mdp)
# left_policy = FunctionPolicy(s->:left)
# right_policy = FunctionPolicy(s->:right)

# for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps=11)
#     @show s
#     @show a
#     @show r
#     println()
# end
