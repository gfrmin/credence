#!/usr/bin/env julia
"""
    host.jl — Host driver for the email program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
flat MixtureMeasure of TaggedBetaMeasures → action EU (domain + meta) →
conditioning → repeat.

Meta-actions (enumerate_more, perturb_grammar, deepen) are evaluated by
the same EU machinery as domain actions. The agent decides whether to act
on the current email or invest in improving its hypothesis space.

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
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, GTExpr

include("features.jl")
include("terminals.jl")
include("preferences.jl")
include("metrics.jl")
include("llm_prosthetic.jl")
include("action_composition.jl")

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
    evaluate_programs!(cache, compiled_kernels, grammar_sensor_vectors, temporal_state)

Evaluate all programs and cache their recommended actions. Returns the cache.
"""
function evaluate_programs!(
    cache::Dict{Int, Symbol},
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any}
)
    for (tag, ck) in enumerate(compiled_kernels)
        haskey(cache, tag) && continue
        haskey(grammar_sensor_vectors, ck.grammar_id) || continue
        sv = grammar_sensor_vectors[ck.grammar_id]
        cache[tag] = ck.evaluate(sv, temporal_state)
    end
    cache
end

"""
    compute_action_entropy(state, rec_cache, component_weights) → Float64

Shannon entropy of the action distribution. High entropy means the agent
is uncertain which action to take — meta-actions have positive VOI.
"""
function compute_action_entropy(
    state::AgentState,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64}
)::Float64
    action_weights = Dict{Symbol, Float64}()
    for (j, _) in enumerate(state.metadata)
        haskey(rec_cache, j) || continue
        a = rec_cache[j]
        action_weights[a] = get(action_weights, a, 0.0) + component_weights[j]
    end
    total = sum(values(action_weights))
    total < 1e-300 && return 0.0
    H = 0.0
    for (_, w) in action_weights
        p = w / total
        p > 1e-300 && (H -= p * log(p))
    end
    H
end

"""
    mean_observation_count(state) → Float64

Average number of observations across components: mean(α + β - 2).
"""
function mean_observation_count(state::AgentState)::Float64
    isempty(state.belief.components) && return 0.0
    total = 0.0
    for comp in state.belief.components
        tbm = comp::TaggedBetaMeasure
        total += tbm.beta.alpha + tbm.beta.beta - 2.0
    end
    total / length(state.belief.components)
end

"""
    compute_meta_eu(state, action, rec_cache, component_weights; meta_cost_this_turn) → Float64

EU for meta-actions. The entropy of the action distribution proxies for
the value of information from improving the hypothesis space.
"""
function compute_meta_eu(
    state::AgentState,
    action::Symbol,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64};
    meta_cost_this_turn::Float64=0.0
)::Float64
    action == :do_nothing && return -Inf

    H = compute_action_entropy(state, rec_cache, component_weights)
    n_obs = mean_observation_count(state)
    entropy_benefit = H / (1.0 + 0.1 * n_obs)

    if action == :enumerate_more
        return entropy_benefit * 0.5 - ENUMERATE_COST - meta_cost_this_turn
    elseif action == :perturb_grammar
        base = n_obs > 5.0 ? entropy_benefit * 0.6 : 0.0
        return base - PERTURB_COST - meta_cost_this_turn
    elseif action == :deepen
        return entropy_benefit * 0.4 - DEEPEN_COST - meta_cost_this_turn
    end
    -Inf
end

"""
    compute_sensor_eu(state, rec_cache, component_weights; llm_cost, already_enriched) → Float64

EU for sensor enrichment via LLM. High when action entropy is high
(enriched features might resolve ambiguity), returns -Inf if already enriched.
"""
function compute_sensor_eu(
    state::AgentState,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64};
    llm_cost::Float64=LLM_COST,
    already_enriched::Bool=false
)::Float64
    already_enriched && return -Inf
    H = compute_action_entropy(state, rec_cache, component_weights)
    n_obs = mean_observation_count(state)
    entropy_benefit = H / (1.0 + 0.1 * n_obs)
    entropy_benefit * 0.7 - llm_cost
end

"""
    compute_eu(state, action, rec_cache, component_weights; ...) → Float64

Expected utility for any action: domain, :ask_user, meta-action, or sensor action.
Single argmax over the combined action space.
"""
function compute_eu(
    state::AgentState,
    action::Symbol,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64};
    ask_cost::Float64=0.1,
    meta_cost_this_turn::Float64=0.0,
    already_enriched::Bool=false
)::Float64
    # Meta-actions
    action in META_ACTIONS && return compute_meta_eu(
        state, action, rec_cache, component_weights;
        meta_cost_this_turn=meta_cost_this_turn)

    # Sensor actions
    action == :ask_llm && return compute_sensor_eu(
        state, rec_cache, component_weights;
        already_enriched=already_enriched)

    # :ask_user — always correct, costs user attention
    action == :ask_user && return 1.0 - ask_cost

    # Domain actions — weighted-average approval rate
    weighted_approval = 0.0
    matching_weight = 0.0
    for (j, comp) in enumerate(state.belief.components)
        haskey(rec_cache, j) || continue
        rec_cache[j] == action || continue
        tbm = comp::TaggedBetaMeasure
        w = component_weights[j]
        weighted_approval += w * mean(tbm.beta)
        matching_weight += w
    end
    matching_weight < 1e-300 && return 0.5
    weighted_approval / matching_weight
end

# ═══════════════════════════════════════
# Meta-action execution
# ═══════════════════════════════════════

"""
    execute_meta_action!(state, action; ...) → Int

Execute a meta-action that modifies the hypothesis space.
Returns the number of programs added.
"""
function execute_meta_action!(
    state::AgentState,
    action::Symbol;
    action_space::Vector{Symbol}=DOMAIN_ACTIONS,
    min_log_prior::Float64=-20.0,
    verbose::Bool=false
)::Int
    if action == :enumerate_more
        top_gids = top_k_grammar_ids(state, 3)
        n_added = 0
        for gid in top_gids
            haskey(state.grammars, gid) || continue
            n_added += add_programs_to_state!(state, state.grammars[gid],
                state.current_max_depth;
                action_space=action_space, min_log_prior=min_log_prior)
        end
        verbose && println("  [Meta: enumerate_more → +$n_added components]")
        return n_added

    elseif action == :perturb_grammar
        w = weights(state.belief)
        freq_table = analyse_posterior_subtrees(state.all_programs, w;
                                                min_frequency=0.01, min_complexity=2)
        top_gids = top_k_grammar_ids(state, 3)
        n_added = 0
        for gid in top_gids
            haskey(state.grammars, gid) || continue
            new_g = perturb_grammar(state.grammars[gid], freq_table)
            state.grammars[new_g.id] = new_g
            n_added += add_programs_to_state!(state, new_g, state.current_max_depth;
                action_space=action_space, min_log_prior=min_log_prior)
        end
        verbose && println("  [Meta: perturb_grammar → +$n_added components]")
        return n_added

    elseif action == :deepen
        state.current_max_depth += 1
        top_gids = top_k_grammar_ids(state, 3)
        n_added = 0
        for gid in top_gids
            haskey(state.grammars, gid) || continue
            n_added += add_programs_to_state!(state, state.grammars[gid],
                state.current_max_depth;
                action_space=action_space, min_log_prior=min_log_prior)
        end
        verbose && println("  [Meta: deepen → depth=$(state.current_max_depth), +$n_added components]")
        return n_added
    end
    0
end

# ═══════════════════════════════════════
# Observation kernel
# ═══════════════════════════════════════

"""
    build_email_observation_kernel(compiled_kernels, grammar_sensor_vectors,
                                   temporal_state, user_action) → Kernel

Build a kernel for conditioning. Each program evaluates features →
recommends an action. Compared to user_action:
- Matching recommendation → log(p) (correct prediction)
- Non-matching recommendation → log(1-p) (incorrect prediction)

Populates correct_cache in kernel params for per-component Beta update.
Always condition with obs=1.0.
"""
function build_email_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any},
    user_action::Symbol
)
    recommendation_cache = Dict{Int, Symbol}()
    correct_cache = Dict{Int, Bool}()
    obs_space = Finite([0.0, 1.0])

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used in condition"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                tag = m_or_θ.tag
                recommended = get!(recommendation_cache, tag) do
                    ck = compiled_kernels[tag]
                    sv = grammar_sensor_vectors[ck.grammar_id]
                    ck.evaluate(sv, temporal_state)
                end
                correct = recommended == user_action
                correct_cache[tag] = correct
                p = mean(m_or_θ.beta)
                correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end,
        nothing,
        Dict{Symbol, Any}(:correct_cache => correct_cache))
end

# ═══════════════════════════════════════
# Main agent loop
# ═══════════════════════════════════════

"""
    run_agent(; corpus, user_pref, ...) → (metrics, state, evolved_grammars)

Main email agent loop. Processes emails sequentially, learning the user's
preference profile through approve/override feedback.

Meta-actions (enumerate_more, perturb_grammar, deepen) are evaluated by
the same EU machinery as domain actions. The agent decides whether to act
or invest in improving its hypothesis space at each step.
"""
function run_agent(;
    corpus::Vector{Email},
    user_pref::UserPreference,
    program_max_depth::Int = 3,
    min_log_prior::Float64 = -20.0,
    max_meta_per_step::Int = 3,
    ask_cost::Float64 = 0.1,
    population_grammar::Union{Nothing, Vector{Grammar}} = nothing,
    llm_config::LLMConfig = default_llm_config(),
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
                                       action_space=DOMAIN_ACTIONS,
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
    grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
    state = AgentState(belief, metadata, compiled_kernels, all_programs,
                       grammar_dict, program_max_depth)

    temporal_state = Dict{Symbol, Any}(:recent => Vector{Float64}[])
    metrics = EmailMetricsTracker()

    # 2. MAIN LOOP
    for (step, email) in enumerate(corpus)
        # Per-grammar sensor projection
        grammar_sensor_vectors = project_email_per_grammar(
            email, collect(values(state.grammars)))

        # Inner loop: agent may take meta/sensor actions before committing
        meta_cost_this_turn = 0.0
        meta_actions_taken = 0
        already_enriched = false
        used_llm = false
        chosen_action = :do_nothing

        while true
            # Evaluate all programs (fresh cache each iteration — new programs may exist)
            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               grammar_sensor_vectors, temporal_state)

            w = weights(state.belief)

            # Compute EU for all actions
            action_eus = Dict{Symbol, Float64}()
            for a in ALL_ACTIONS
                action_eus[a] = compute_eu(state, a, rec_cache, w;
                    ask_cost=ask_cost, meta_cost_this_turn=meta_cost_this_turn,
                    already_enriched=already_enriched)
            end

            chosen_action = argmax(action_eus)

            # Handle meta-actions
            if chosen_action in META_ACTIONS && chosen_action != :do_nothing &&
               meta_actions_taken < max_meta_per_step
                execute_meta_action!(state, chosen_action;
                    action_space=DOMAIN_ACTIONS, min_log_prior=min_log_prior,
                    verbose=verbose)
                meta_actions_taken += 1
                meta_cost_this_turn += (chosen_action == :deepen ? DEEPEN_COST :
                                        chosen_action == :perturb_grammar ? PERTURB_COST :
                                        ENUMERATE_COST)
                grammar_sensor_vectors = project_email_per_grammar(
                    email, collect(values(state.grammars)))
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
                continue
            end

            # Handle sensor actions (LLM enrichment)
            if chosen_action == :ask_llm && !already_enriched
                enriched = llm_enrich_features(llm_config, email, extract_features(email))
                grammar_sensor_vectors = project_enriched_per_grammar(
                    enriched, collect(values(state.grammars)))
                already_enriched = true
                used_llm = true
                verbose && println("  [Sensor: ask_llm at step $step]")
                continue
            end

            break
        end

        # If chosen_action is still non-domain, fall back to domain EU
        if chosen_action in META_ACTIONS || chosen_action in SENSOR_ACTIONS
            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               grammar_sensor_vectors, temporal_state)
            w = weights(state.belief)
            action_eus = Dict{Symbol, Float64}()
            for a in EMAIL_ACTIONS
                action_eus[a] = compute_eu(state, a, rec_cache, w; ask_cost=ask_cost)
            end
            chosen_action = argmax(action_eus)
        end

        asked = chosen_action == :ask_user

        # Get the user's true preferred action
        correct_action = user_pref.decide(email)

        # For accuracy tracking: asked counts as neither correct nor incorrect
        action_correct_for_metrics = !asked && (chosen_action == correct_action)

        # Surprise: how unexpected was the correct action?
        rec_cache = Dict{Int, Symbol}()
        evaluate_programs!(rec_cache, state.compiled_kernels,
                           grammar_sensor_vectors, temporal_state)
        w = weights(state.belief)
        eu_correct = compute_eu(state, correct_action, rec_cache, w; ask_cost=ask_cost)
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
            meta_str = meta_actions_taken > 0 ? ", meta=$meta_actions_taken" : ""
            llm_str = used_llm ? ", llm" : ""
            println("Step $step: $(asked ? "ASK→$correct_action" : string(chosen_action)) " *
                    "(correct=$correct_action, " *
                    "$(action_correct_for_metrics ? "✓" : asked ? "?" : "✗"), " *
                    "surprise=$(round(surprise, digits=2)), " *
                    "components=$(length(state.belief.components))$meta_str$llm_str)")
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
                      surprise=surprise,
                      n_meta_actions=meta_actions_taken,
                      used_llm=used_llm)
    end

    if verbose
        print_email_summary(metrics; last_n=20)
    end

    (metrics=metrics, state=state, evolved_grammars=collect(values(state.grammars)))
end

# ═══════════════════════════════════════
# Multi-user meta-learning
# ═══════════════════════════════════════

"""
    run_multi_user(; user_prefs, ...) → Vector{NamedTuple}

Run the agent sequentially across multiple users, passing evolved grammars
forward. Later users benefit from grammar pool transfer — structural
regularities discovered for earlier users accelerate learning.
"""
function run_multi_user(;
    user_prefs::Vector{UserPreference},
    corpus_per_user::Int = 60,
    rng_seed::Int = 42,
    verbose::Bool = true,
    program_max_depth::Int = 2,
    min_log_prior::Float64 = -15.0,
    max_meta_per_step::Int = 3,
    ask_cost::Float64 = 0.1
)
    grammar_pool = nothing
    results = NamedTuple{(:user, :metrics, :n_grammars), Tuple{Symbol, EmailMetricsTracker, Int}}[]

    for (i, pref) in enumerate(user_prefs)
        corpus = generate_email_corpus(corpus_per_user; rng_seed=rng_seed + i)
        result = run_agent(;
            corpus=corpus,
            user_pref=pref,
            program_max_depth=program_max_depth,
            min_log_prior=min_log_prior,
            max_meta_per_step=max_meta_per_step,
            ask_cost=ask_cost,
            population_grammar=grammar_pool,
            rng_seed=rng_seed + i,
            verbose=verbose
        )
        grammar_pool = result.evolved_grammars

        if verbose
            println("  User $i ($(pref.name)): $(length(grammar_pool)) grammars in pool")
        end

        push!(results, (user=pref.name, metrics=result.metrics, n_grammars=length(grammar_pool)))
    end

    results
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
