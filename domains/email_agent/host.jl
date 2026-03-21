#!/usr/bin/env julia
"""
    host.jl — Host driver for the email program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
flat MixtureMeasure of TaggedBetaMeasures → action EU → conditioning →
grammar perturbation → repeat.

Tier 3: email-domain-specific. Uses Tier 1 (Credence DSL) and Tier 2
(ProgramSpace) for domain-independent inference machinery.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: expect, condition, weights, mean
using Credence: CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, MixtureMeasure
using Credence: Finite, Interval, Kernel, Measure
using Credence: prune, truncate
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: SensorConfig, SensorChannel
using Credence: enumerate_programs, compile_kernel
using Credence: analyse_posterior_subtrees, perturb_grammar
using Credence: aggregate_grammar_weights
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, GTExpr

include("features.jl")
include("terminals.jl")
include("preferences.jl")
include("metrics.jl")

using Random

# ═══════════════════════════════════════
# Per-grammar sensor projection for email
# ═══════════════════════════════════════

"""
    project_email_per_grammar(email, grammars) → Dict{Int, Vector{Float64}}

Project an email's features through each grammar's sensor config.
"""
function project_email_per_grammar(email::Email, grammars::Vector{Grammar})
    features = extract_features(email)
    result = Dict{Int, Vector{Float64}}()
    for g in grammars
        readings = Float64[]
        for ch in g.sensor_config.channels
            raw = features[ch.source_index + 1]
            noisy = raw + randn() * ch.noise_σ
            push!(readings, clamp(noisy, 0.0, 1.0))
        end
        result[g.id] = readings
    end
    result
end

# ═══════════════════════════════════════
# Action EU computation
# ═══════════════════════════════════════

"""
    evaluate_predicates!(cache, compiled_kernels, grammar_sensor_vectors, temporal_state)

Evaluate all predicates and cache results. Returns the cache.
"""
function evaluate_predicates!(
    cache::Dict{Int, Bool},
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any}
)
    for (tag, ck) in enumerate(compiled_kernels)
        haskey(cache, tag) && continue
        sv = get(grammar_sensor_vectors, ck.grammar_id, Float64[])
        cache[tag] = !isempty(sv) && ck.evaluate(sv, temporal_state)
    end
    cache
end

"""
    compute_eu(state, action, pred_cache, component_weights; ask_cost) → Float64

Expected approval for any action, including :ask_user.

For domain actions — the weighted-average approval rate among programs
that fire and recommend this action:
    EU(a) = Σ_{j: fires∧action_j==a} w_j × mean(beta_j)
            / Σ_{j: fires∧action_j==a} w_j

Returns 0.5 (base rate) if no matching programs fire.

For :ask_user:
    EU = 1.0 - ask_cost  (always correct, costs user attention)

Dynamics: uniform priors give EU ≈ 0.5 < 1-ask_cost → agent asks.
As posteriors concentrate, correct action's EU rises → agent acts.
"""
function compute_eu(
    state::AgentState,
    action::Symbol,
    pred_cache::Dict{Int, Bool},
    component_weights::Vector{Float64};
    ask_cost::Float64=0.1
)::Float64
    action == :ask_user && return 1.0 - ask_cost

    weighted_approval = 0.0
    matching_weight = 0.0
    for (j, comp) in enumerate(state.belief.components)
        pred_cache[j] || continue
        state.compiled_kernels[j].action == action || continue
        tbm = comp::TaggedBetaMeasure
        w = component_weights[j]
        weighted_approval += w * mean(tbm.beta)
        matching_weight += w
    end
    matching_weight < 1e-300 && return 0.5
    weighted_approval / matching_weight
end

# ═══════════════════════════════════════
# Observation kernel
# ═══════════════════════════════════════

"""
    build_email_observation_kernel(compiled_kernels, grammar_sensor_vectors,
                                   temporal_state, user_action) → Kernel

Build a kernel for conditioning. The user chose `user_action`:
- Firing program with matching action → log(p) (positive)
- Firing program with non-matching action → log(1-p) (negative)
- Non-firing program → log(0.5) (base rate)

Always condition with obs=1.0.
"""
function build_email_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any},
    user_action::Symbol
)
    pred_cache = Dict{Int, Bool}()
    obs_space = Finite([0.0, 1.0])

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used in condition"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                tag = m_or_θ.tag
                fired = get!(pred_cache, tag) do
                    ck = compiled_kernels[tag]
                    sv = get(grammar_sensor_vectors, ck.grammar_id, Float64[])
                    isempty(sv) && return false
                    ck.evaluate(sv, temporal_state)
                end
                if !fired
                    log(0.5)
                else
                    ck = compiled_kernels[tag]
                    p = mean(m_or_θ.beta)
                    if ck.action == user_action
                        obs == 1.0 ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
                    else
                        obs == 1.0 ? log(max(1.0 - p, 1e-300)) : log(max(p, 1e-300))
                    end
                end
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end)
end

# ═══════════════════════════════════════
# Main agent loop
# ═══════════════════════════════════════

"""
    run_agent(; corpus, user_pref, ...) → (metrics, state, grammar_pool)

Main email agent loop. Processes emails sequentially, learning the user's
preference profile through approve/override feedback.
"""
function run_agent(;
    corpus::Vector{Email},
    user_pref::UserPreference,
    program_max_depth::Int = 2,
    min_log_prior::Float64 = -20.0,
    perturbation_interval::Int = 25,
    ask_cost::Float64 = 0.1,
    population_grammar::Union{Nothing, Vector{Grammar}} = nothing,
    rng_seed::Int = 42,
    verbose::Bool = true
)
    Random.seed!(rng_seed)

    # 1. INITIALISE
    grammar_pool = if population_grammar !== nothing
        copy(population_grammar)
    else
        generate_email_seed_grammars()
    end

    if verbose
        println("Generated $(length(grammar_pool)) seed grammars")
    end

    # Enumerate all (grammar, predicate, action) triples
    components = Measure[]
    log_prior_weights = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled_kernels = CompiledKernel[]
    all_programs = Program[]

    idx = 0
    for g in grammar_pool
        programs = enumerate_programs(g, program_max_depth;
                                       actions=DOMAIN_ACTIONS,
                                       min_log_prior=min_log_prior)
        for (pi, p) in enumerate(programs)
            idx += 1
            push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
            lw = -g.complexity * log(2) - p.complexity * log(2)
            push!(log_prior_weights, lw)
            push!(metadata, (g.id, pi))
            push!(compiled_kernels, compile_kernel(p, g, pi))
            push!(all_programs, p)
        end
    end

    if verbose
        println("Total components: $(length(components))")
        println("Grammars: $(length(grammar_pool))")
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
    state = AgentState(belief, metadata, compiled_kernels, all_programs)

    temporal_state = Dict{Symbol, Any}(:recent => Vector{Float64}[])
    metrics = EmailMetricsTracker()

    # 2. MAIN LOOP
    for (step, email) in enumerate(corpus)
        # Per-grammar sensor projection
        grammar_sensor_vectors = project_email_per_grammar(email, grammar_pool)

        # Evaluate all predicates (cached)
        pred_cache = Dict{Int, Bool}()
        evaluate_predicates!(pred_cache, state.compiled_kernels,
                             grammar_sensor_vectors, temporal_state)

        # Compute EU for all actions uniformly (including :ask_user)
        w = weights(state.belief)
        action_eus = Dict{Symbol, Float64}()
        for a in EMAIL_ACTIONS
            action_eus[a] = compute_eu(state, a, pred_cache, w; ask_cost)
        end

        # Select action (indifference → ask, per CLAUDE.md)
        chosen_action = argmax(action_eus)
        asked = chosen_action == :ask_user

        # Get the user's true preferred action
        correct_action = user_pref.decide(email)

        # For accuracy tracking: asked counts as neither correct nor incorrect
        action_correct_for_metrics = !asked && (chosen_action == correct_action)

        # Surprise: how unexpected was the correct action?
        eu_correct = compute_eu(state, correct_action, pred_cache, w; ask_cost)
        surprise = -log(max(eu_correct, 1e-300))

        # Condition belief on user's true preferred action
        k = build_email_observation_kernel(
            state.compiled_kernels, grammar_sensor_vectors,
            temporal_state, correct_action)
        state.belief = condition(state.belief, k, 1.0)

        # Prune and truncate
        sync_prune!(state; threshold=-30.0)
        sync_truncate!(state; max_components=2000)

        if verbose
            println("Step $step: $(asked ? "ASK→$correct_action" : string(chosen_action)) " *
                    "(correct=$correct_action, " *
                    "$(action_correct_for_metrics ? "✓" : asked ? "?" : "✗"), " *
                    "surprise=$(round(surprise, digits=2)), " *
                    "components=$(length(state.belief.components)))")
        end

        # Record metrics
        w_post = weights(state.belief)
        gw = aggregate_grammar_weights(w_post, state.metadata)
        record_email!(metrics;
                      step=step,
                      action_taken=chosen_action,
                      correct_action=correct_action,
                      is_correct=action_correct_for_metrics,
                      asked=asked,
                      grammar_weights=gw,
                      n_components=length(state.belief.components),
                      surprise=surprise)

        # Periodic grammar perturbation
        if step % perturbation_interval == 0 && step < length(corpus)
            w_pert = weights(state.belief)
            gw_pert = aggregate_grammar_weights(w_pert, state.metadata)

            freq_table = analyse_posterior_subtrees(
                state.all_programs, w_pert;
                min_frequency=0.01, min_complexity=2)

            top_gids = sort(collect(keys(gw_pert)), by=gi -> -get(gw_pert, gi, 0.0))[1:min(3, length(gw_pert))]

            new_components = Measure[]
            new_lw = Float64[]
            new_meta = Tuple{Int, Int}[]
            new_ck = CompiledKernel[]
            new_progs = Program[]

            base_idx = length(state.compiled_kernels)
            for gid in top_gids
                g_idx = findfirst(g -> g.id == gid, grammar_pool)
                g_idx === nothing && continue

                new_g = perturb_grammar(grammar_pool[g_idx], freq_table)
                push!(grammar_pool, new_g)

                new_programs = enumerate_programs(new_g, program_max_depth;
                                                   actions=DOMAIN_ACTIONS,
                                                   min_log_prior=min_log_prior)
                for (pi, p) in enumerate(new_programs)
                    base_idx += 1
                    push!(new_components, TaggedBetaMeasure(Interval(0.0, 1.0), base_idx, BetaMeasure(1.0, 1.0)))
                    lw = -new_g.complexity * log(2) - p.complexity * log(2)
                    push!(new_lw, lw)
                    push!(new_meta, (new_g.id, pi))
                    push!(new_ck, compile_kernel(p, new_g, pi))
                    push!(new_progs, p)
                end
            end

            if !isempty(new_components)
                all_comps = Measure[state.belief.components..., new_components...]
                all_lw = Float64[state.belief.log_weights..., new_lw...]
                state.belief = MixtureMeasure(Interval(0.0, 1.0), all_comps, all_lw)
                append!(state.metadata, new_meta)
                append!(state.compiled_kernels, new_ck)
                append!(state.all_programs, new_progs)
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)

                if verbose
                    println("  [Perturbation at step $step: +$(length(new_components)) components, " *
                            "total=$(length(state.belief.components))]")
                end
            end
        end
    end

    if verbose
        print_email_summary(metrics; last_n=20)
    end

    (metrics=metrics, state=state, evolved_grammars=grammar_pool)
end

# ═══════════════════════════════════════
# Entry point
# ═══════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("Email Agent — Program-Space Preference Learning")
    println("=" ^ 60)

    corpus = generate_email_corpus(100)

    println("\n--- urgency_responsive profile ---")
    result = run_agent(
        corpus=corpus,
        user_pref=PREFERENCE_PROFILES[:urgency_responsive],
        verbose=true)
end
