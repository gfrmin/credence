"""
    host_helpers.jl — Host-level helpers for the credence agent

Extracted from examples/host_credence_agent.jl for use by external callers
(Python via juliacall, Julia host drivers).

These functions manage per-tool per-category reliability and coverage state
using the ontology types. They are host concerns (state management), not
axiom-constrained functions.
"""

using ..Ontology
import ..Ontology: truncate

"""Initial per-tool reliability state: single ProductMeasure of K Beta(1,1) priors."""
function initial_rel_state(n_categories::Int)
    factors = Measure[BetaMeasure() for _ in 1:n_categories]
    prod = ProductMeasure(factors)
    MixtureMeasure(prod.space, Measure[prod], [0.0])
end

"""Marginalize over components × categories to get effective per-tool measure as MixtureMeasure of Betas."""
function marginalize_betas(rel_state::MixtureMeasure, cat_w::Vector{Float64})
    components = Measure[]
    log_wts = Float64[]
    for (i, comp) in enumerate(rel_state.components)
        prod = comp::ProductMeasure
        for (c, w_c) in enumerate(cat_w)
            push!(components, prod.factors[c])
            push!(log_wts, rel_state.log_weights[i] + log(max(w_c, 1e-300)))
        end
    end
    prune(MixtureMeasure(Interval(0.0, 1.0), components, log_wts))
end

"""Initial per-tool coverage state: ProductMeasure of K Beta priors centered at declared coverage."""
function initial_cov_state(n_categories::Int, prior_coverage::Vector{Float64}; strength::Float64=10.0)
    factors = Measure[BetaMeasure(max(p * strength, 0.01), max((1-p) * strength, 0.01))
                      for p in prior_coverage]
    prod = ProductMeasure(factors)
    MixtureMeasure(prod.space, Measure[prod], [0.0])
end

"""Extract per-category mean reliability from a rel_state MixtureMeasure.

Returns a Vector{Float64} of E[r_c] for each category, where the expectation
is over the mixture components.
"""
function extract_reliability_means(rel_state::MixtureMeasure)
    w = weights(rel_state)
    n_factors = length(rel_state.components[1]::ProductMeasure |> x -> x.factors)
    means = Float64[]
    for c in 1:n_factors
        m = sum(w[i] * mean(rel_state.components[i]::ProductMeasure |> x -> x.factors[c]::BetaMeasure)
                for i in eachindex(w))
        push!(means, m)
    end
    means
end

"""
Update per-tool Beta state via structured ProductMeasure condition.

Prepends categorical belief to per-tool state, conditions the joint on
the Bernoulli observation using FactorSelector (only the active category's
Beta is updated per branch), then strips the categorical back out.
Works for any per-category-Beta state (reliability or coverage).
Pure function — no mutation.
"""
function update_beta_state(rel_state::MixtureMeasure,
                            cat_belief::CategoricalMeasure, obs)
    n_cat = length(cat_belief.space.values)

    # 1. Prepend categorical to each component → MixtureMeasure of bigger ProductMeasures
    joint_components = Measure[]
    for comp in rel_state.components
        prod = comp::ProductMeasure
        push!(joint_components, ProductMeasure(Measure[cat_belief, prod.factors...]))
    end
    joint_space = joint_components[1].space
    joint = MixtureMeasure(joint_space, joint_components, copy(rel_state.log_weights))

    # 2. Build kernel with FactorSelector and condition — ONE LINE of inference
    binary = Finite([0.0, 1.0])
    k = Kernel(joint_space, binary,
        _ -> error("generate not used"),
        (h, o) -> let r = h[Int(h[1]) + 2]; o == 1.0 ? log(r) : log(1.0 - r) end;
        factor_selector = FactorSelector(1, c -> [Int(c) + 2]),
        likelihood_family = BetaBernoulli())
    posterior = condition(joint, k, obs)

    # 3. Extract category posteriors (sum weights by categorical value)
    cat_log_post = fill(-Inf, n_cat)
    for (i, comp) in enumerate(posterior.components)
        prod = comp::ProductMeasure
        c_val = prod.factors[1]::CategoricalMeasure  # point mass
        ci = findfirst(==(c_val.space.values[1]), cat_belief.space.values)
        lw = posterior.log_weights[i]
        if cat_log_post[ci] == -Inf
            cat_log_post[ci] = lw
        else
            mx = max(cat_log_post[ci], lw)
            cat_log_post[ci] = mx + log(exp(cat_log_post[ci] - mx) + exp(lw - mx))
        end
    end
    new_cat_belief = CategoricalMeasure(cat_belief.space, cat_log_post)

    # 4. Strip categorical from each component → per-tool state
    stripped = Measure[ProductMeasure(Measure[comp.factors[2:end]...])
                       for comp in posterior.components]
    new_rel_state = MixtureMeasure(stripped[1].space, stripped, copy(posterior.log_weights))

    # 5. Prune and truncate
    new_rel_state = truncate(prune(new_rel_state); max_components=20)

    (new_rel_state, new_cat_belief)
end
