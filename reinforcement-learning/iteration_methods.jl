# Code taken from JuliaReinforcementLearningAnIntroduction
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl/blob/master/src/extensions/iteration_methods.jl

"""
    policy_evaluation!(V::AbstractApproximator, π, model::AbstractEnvironmentModel; γ::Float64=0.9, θ::Float64=1e-4)
See more details at Section (4.1) on Page 75 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_evaluation!(
    ;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractGridWorld,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
)
    states, actions = 1:length(model.world[1,:,:]), 1:length(π.action_space)
    while true
        Δ = 0.0
        for s in states
            v = sum(
                a -> get_prob(π, s, a) *
                     sum(p * (r + γ * V(s′)) for (s′, r, p) in model(s, a)),
                actions,
            )
            error = v - V(s)
            update!(V, s => error)
            Δ = max(Δ, abs(error))
        end
        Δ < θ && break
    end
    V
end

"""
    policy_improvement!(;V::AbstractApproximator, π::AbstractPolicy, model::AbstractEnvironmentModel, γ::Float64 = 0.9)
See more details at Section (4.2) on Page 76 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_improvement!(
    ;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractGridWorld,
    γ::Float64 = 0.9,
)
    states, actions = 1:length(get_observation_space(model)), 1:length(get_action_space(model))
    is_policy_stable = true
    for s in states
        old_a = π(s)
        best_action_inds = find_all_max([sum(p * (r + γ * V(s′)) for (s′, r, p) in model(
            s,
            a,
        )) for a in actions])[2]
        new_a = actions[sample(best_action_inds)]
        if new_a != old_a
            update!(π, s => new_a)
            is_policy_stable = false
        end
    end
    is_policy_stable
end

"""
    policy_iteration!(V::AbstractApproximator, π, model::AbstractEnvironmentModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))
See more details at Section (4.3) on Page 80 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_iteration!(
    ;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractGridWorld,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
    max_iter = typemax(Int),
)
    for i = 1:max_iter
        @debug "iteration: $i"
        policy_evaluation!(; V = V, π = π, model = model, γ = γ, θ = θ)
        policy_improvement!(; V = V, π = π, model = model, γ = γ) && break
    end
end

"""
    value_iteration!(V::AbstractApproximator, model::AbstractEnvironmentModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))
See more details at Section (4.4) on Page 83 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function value_iteration!(
    ;
    V::AbstractApproximator,
    model::AbstractEnvironmentModel,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
    max_iter = typemax(Int),
)
    states, actions = 1:length(get_observation_space(model)), 1:length(get_action_space(model))
    for i = 1:max_iter
        Δ = 0.0
        for s in states
            v = maximum(sum(p * (r + γ * V(s′)) for (s′, r, p) in model(s, a)) for a in actions)
            error = v - V(s)
            update!(V, s => error)
            Δ = max(Δ, abs(error))
        end
        Δ < θ && break
    end
end