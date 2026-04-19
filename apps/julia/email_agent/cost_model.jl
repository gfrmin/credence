# Role: brain-side application
"""
    cost_model.jl — Uncertain, time-based cost model for actions

Every action has a cost belief: a NormalGammaMeasure over log(seconds).
Latencies are positive and right-skewed, so log(seconds) is approximately
Normal with unknown mean and variance — the NormalGamma is its conjugate
prior. After each action the host observes wall-clock time and conditions
the belief via standard conjugate updating.

Expected cost in utility units:
    E[cost(a)] = time_value × exp(μ + β/(2α))   (log-normal mean)
"""

using Credence: NormalGammaMeasure, condition, Kernel
using Credence: ProductSpace, Euclidean, PositiveReals, Space
using Credence: PushOnly

struct CostModel
    beliefs::Dict{Symbol, NormalGammaMeasure}
    time_value::Float64    # utility lost per second of waiting
end

"""
    default_cost_model(; time_value=0.02) → CostModel

Weak priors (κ=1, α=1, β=0.5 ≈ 1 pseudo-observation) centred on
estimated latencies per action type.
"""
function default_cost_model(; time_value::Float64=0.02)
    ng(μ) = NormalGammaMeasure(1.0, μ, 1.0, 0.5)
    CostModel(Dict{Symbol, NormalGammaMeasure}(
        :enumerate_more  => ng(log(0.5)),
        :perturb_grammar => ng(log(0.5)),
        :deepen          => ng(log(1.0)),
        :ask_llm         => ng(log(3.0)),
        :ask_user        => ng(log(10.0)),
        :jmap_fetch      => ng(log(1.0)),
        :jmap_mutate     => ng(log(0.3)),
    ), time_value)
end

"""
    expected_cost(cm, action) → Float64

Expected cost of `action` in utility units: time_value × E[seconds].
E[seconds] is the log-normal mean exp(μ + σ²/2) where σ² = β/α.
"""
function expected_cost(cm::CostModel, action::Symbol)::Float64
    belief = get(cm.beliefs, action, nothing)
    belief === nothing && return 0.0
    exp(belief.μ + belief.β / (2.0 * belief.α)) * cm.time_value
end

"""
    expected_seconds(cm, action) → Float64

Expected seconds for `action` (without time_value scaling).
"""
function expected_seconds(cm::CostModel, action::Symbol)::Float64
    belief = get(cm.beliefs, action, nothing)
    belief === nothing && return 0.0
    exp(belief.μ + belief.β / (2.0 * belief.α))
end

"""
    observe_cost!(cm, action, elapsed_seconds)

Condition the cost belief for `action` on an observed wall-clock latency.
Uses conjugate NormalGamma updating on log(elapsed_seconds).
"""
function observe_cost!(cm::CostModel, action::Symbol, elapsed_seconds::Float64)
    belief = get(cm.beliefs, action, nothing)
    belief === nothing && return
    elapsed_seconds > 0 || return
    k = Kernel(
        ProductSpace(Space[Euclidean(1), PositiveReals()]),
        Euclidean(1),
        _ -> error("generate not used for cost conditioning"),
        (μτ, obs) -> -0.5 * μτ[2] * (obs - μτ[1])^2 + 0.5 * log(μτ[2]);
        params = Dict{Symbol, Any}(:normal_gamma => true),
        likelihood_family = PushOnly()
    )
    cm.beliefs[action] = condition(belief, k, log(elapsed_seconds))
end
