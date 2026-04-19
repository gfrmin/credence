#!/usr/bin/env julia
# Role: brain-side application
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

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: expect, condition, push_measure, density, weights, mean
using Credence: load_dsl
using Credence: CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, MixtureMeasure
using Credence: Finite, Interval, Kernel, Measure, ProductSpace, Euclidean, PositiveReals, Space
using Credence: prune, truncate
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
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
include("cost_model.jl")

using Random

# Load stdlib — derived functions (optimise, value, voi) live in the DSL
const _STDLIB_ENV = load_dsl("")
const optimise = _STDLIB_ENV[:optimise]
const value_fn = _STDLIB_ENV[:value]  # avoid shadowing Base.value
const voi_fn = _STDLIB_ENV[:voi]

# ═══════════════════════════════════════
# Primitive action execution (multi-step episodes)
# ═══════════════════════════════════════

"""
    execute_primitive!(ps::ProcessingState, action::Symbol)

Execute a primitive action by updating the processing state. No-op for :done.
"""
function execute_primitive!(ps::ProcessingState, action::Symbol)
    action == :add_label_urgent    && (ps.has_label_urgent = true; return)
    action == :add_label_delegated && (ps.has_label_delegated = true; return)
    action == :move_to_archive     && (ps.is_in_archive = true; return)
    action == :move_to_priority    && (ps.is_in_priority = true; return)
    action == :move_to_later       && (ps.is_in_later = true; return)
    action == :mark_read           && (ps.is_read = true; return)
    action == :notify_user         && (ps.user_notified = true; return)
    action == :draft_reply         && (ps.reply_drafted = true; return)
    action == :assign_to           && (ps.is_assigned = true; return)
    action == :done                && return
    nothing
end

"""
    remaining_target_actions(ps::ProcessingState, target::Set{Symbol}) → Set{Symbol}

Return the set of target primitives not yet completed.
"""
function remaining_target_actions(ps::ProcessingState, target::Set{Symbol})::Set{Symbol}
    done = Set{Symbol}()
    ps.has_label_urgent    && push!(done, :add_label_urgent)
    ps.has_label_delegated && push!(done, :add_label_delegated)
    ps.is_in_archive       && push!(done, :move_to_archive)
    ps.is_in_priority      && push!(done, :move_to_priority)
    ps.is_in_later         && push!(done, :move_to_later)
    ps.is_read             && push!(done, :mark_read)
    ps.user_notified       && push!(done, :notify_user)
    ps.reply_drafted       && push!(done, :draft_reply)
    ps.is_assigned         && push!(done, :assign_to)
    setdiff(target, done)
end

# ═══════════════════════════════════════
# Action EU computation
# ═══════════════════════════════════════

"""
    evaluate_programs!(cache, compiled_kernels, features, temporal_state)

Evaluate all programs and cache their recommended actions. Returns the cache.
"""
function evaluate_programs!(
    cache::Dict{Int, Symbol},
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any}
)
    for (tag, ck) in enumerate(compiled_kernels)
        haskey(cache, tag) && continue
        cache[tag] = ck.evaluate(features, temporal_state)
    end
    cache
end

"""
    build_predictive(state, rec_cache, action_space) → CategoricalMeasure

Push the belief through program recommendations to get the predictive
distribution over the user's true action. For each action a:
  P(user wants a) = Σ_j w_j * [r_j == a ? E[θ_j] : E[1-θ_j]/(|A|-1)]

This IS push_measure(belief, action_kernel), computed using the mixture
structure: each component's Beta gives E[θ] via expect, the mixture
weights marginalize over programs.
"""
function build_predictive(
    state::AgentState,
    rec_cache::Dict{Int, Symbol},
    action_space::Vector{Symbol}
)::CategoricalMeasure
    w = weights(state.belief)
    n_actions = length(action_space)
    action_probs = Dict{Symbol, Float64}(a => 0.0 for a in action_space)

    for (j, comp) in enumerate(state.belief.components)
        haskey(rec_cache, j) || continue
        tbm = comp::TaggedBetaMeasure
        # E[θ_j] via credence's expect on the Beta component
        θ_mean = expect(tbm, identity)
        r_j = rec_cache[j]
        for a in action_space
            p = a == r_j ? θ_mean : (1.0 - θ_mean) / max(n_actions - 1, 1)
            action_probs[a] += w[j] * p
        end
    end

    logw = [log(max(action_probs[a], 1e-300)) for a in action_space]
    CategoricalMeasure(Finite(action_space), logw)
end

const DEFAULT_UTILITY = (true_action, chosen_action) -> true_action == chosen_action ? 1.0 : 0.0

"""
    select_action_eu(state, features, temporal_state; ...) → NamedTuple

Action selection via push + optimise:
1. Evaluate programs → rec_cache
2. build_predictive (= push) → CategoricalMeasure over user's true action
3. optimise(predictive, actions, utility) → best action
4. value_fn(predictive, actions, utility) vs skip_utility → skip if uncertain
"""
function select_action_eu(
    state::AgentState,
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any};
    action_space::Vector{Symbol}=DOMAIN_ACTIONS,
    utility::Function=DEFAULT_UTILITY,
    skip_utility::Float64=-Inf,
)
    rec_cache = Dict{Int, Symbol}()
    evaluate_programs!(rec_cache, state.compiled_kernels, features, temporal_state)

    predictive = build_predictive(state, rec_cache, action_space)
    actions_finite = Finite(action_space)

    best = optimise(predictive, actions_finite, utility)
    best_eu = value_fn(predictive, actions_finite, utility)

    chosen = best_eu > skip_utility ? best : :ask_user
    (action=chosen, eu=best_eu, predictive=predictive, rec_cache=rec_cache)
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
    compute_meta_eu(state, action, rec_cache, component_weights; cost_model, meta_cost_this_turn) → Float64

EU for meta-actions. The entropy of the action distribution proxies for
the value of information from improving the hypothesis space. Cost is
the expected time from the CostModel.
"""
function compute_meta_eu(
    state::AgentState,
    action::Symbol,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64};
    cost_model::CostModel=default_cost_model(),
    meta_cost_this_turn::Float64=0.0
)::Float64
    action == :do_nothing && return -Inf

    H = compute_action_entropy(state, rec_cache, component_weights)
    n_obs = mean_observation_count(state)
    entropy_benefit = H / (1.0 + 0.1 * n_obs)

    if action == :enumerate_more
        return entropy_benefit * 0.5 - expected_cost(cost_model, :enumerate_more) - meta_cost_this_turn
    elseif action == :perturb_grammar
        base = n_obs > 5.0 ? entropy_benefit * 0.6 : 0.0
        return base - expected_cost(cost_model, :perturb_grammar) - meta_cost_this_turn
    elseif action == :deepen
        return entropy_benefit * 0.4 - expected_cost(cost_model, :deepen) - meta_cost_this_turn
    end
    -Inf
end

"""
    compute_sensor_voi(state, features, email, temporal_state; ...) → Float64

VOI for LLM sensor enrichment. Computes the improvement in decision
value from enriched features minus the expected time cost.
Returns -Inf if already enriched.
"""
function compute_sensor_voi(
    state::AgentState,
    features::Dict{Symbol, Float64},
    email::Email,
    temporal_state::Dict{Symbol, Any};
    action_space::Vector{Symbol}=DOMAIN_ACTIONS,
    utility::Function=DEFAULT_UTILITY,
    cost_model::CostModel=default_cost_model(),
    already_enriched::Bool=false
)::Float64
    already_enriched && return -Inf

    actions_finite = Finite(action_space)

    # Current decision value
    rec_cache = Dict{Int, Symbol}()
    evaluate_programs!(rec_cache, state.compiled_kernels, features, temporal_state)
    predictive_now = build_predictive(state, rec_cache, action_space)
    val_now = value_fn(predictive_now, actions_finite, utility)

    # Value after enrichment
    enriched = simulate_llm_enrichment(email, features)
    rec_cache_e = Dict{Int, Symbol}()
    evaluate_programs!(rec_cache_e, state.compiled_kernels, enriched, temporal_state)
    predictive_e = build_predictive(state, rec_cache_e, action_space)
    val_after = value_fn(predictive_e, actions_finite, utility)

    # net-VOI = improvement - cost
    (val_after - val_now) - expected_cost(cost_model, :ask_llm)
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
    cost_model::CostModel=default_cost_model(),
    ask_cost::Float64=0.1,
    utility::Dict{Symbol, Tuple{Float64, Float64}}=Dict{Symbol, Tuple{Float64, Float64}}(),
    meta_cost_this_turn::Float64=0.0,
    already_enriched::Bool=false
)::Float64
    # Meta-actions
    action in META_ACTIONS && return compute_meta_eu(
        state, action, rec_cache, component_weights;
        cost_model=cost_model, meta_cost_this_turn=meta_cost_this_turn)

    # Sensor actions — handled by compute_sensor_voi at call site
    action == :ask_llm && return -Inf

    # :ask_user — fixed utility independent of correctness
    if action == :ask_user
        u_skip = get(utility, :ask_user, (1.0 - ask_cost, 1.0 - ask_cost))
        return u_skip[1]
    end

    # Domain actions — E[U] = Σ w_j * [θ_j * U(correct) + (1-θ_j) * U(wrong)]
    weighted_eu = 0.0
    matching_weight = 0.0
    u_c, u_w = get(utility, action, (1.0, 0.0))
    for (j, comp) in enumerate(state.belief.components)
        haskey(rec_cache, j) || continue
        rec_cache[j] == action || continue
        tbm = comp::TaggedBetaMeasure
        w = component_weights[j]
        θ = mean(tbm.beta)
        weighted_eu += w * (θ * u_c + (1.0 - θ) * u_w)
        matching_weight += w
    end
    matching_weight < 1e-300 && return (u_c + u_w) / 2.0
    weighted_eu / matching_weight
end

# ═══════════════════════════════════════
# EU computation for multi-step episodes
# ═══════════════════════════════════════

"""
    compute_eu_primitive(state, action, rec_cache, component_weights, remaining_target) → Float64

EU for a primitive action in a multi-step episode. An action is valuable
if it's in the remaining target set and programs recommending it have high confidence.
"""
function compute_eu_primitive(
    state::AgentState,
    action::Symbol,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64},
    remaining_target::Set{Symbol}
)::Float64
    if action == :done
        return isempty(remaining_target) ? 0.9 : 0.0
    end
    action in remaining_target || return 0.0

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
    matching_weight < 1e-300 ? 0.5 : weighted_approval / matching_weight
end

"""
    compute_eu_step(state, action, rec_cache, component_weights; ...) → Float64

EU router for multi-step episodes. Routes to compute_eu_primitive for
primitives/:done, existing meta/sensor EU for meta/sensor actions.
"""
function compute_eu_step(
    state::AgentState,
    action::Symbol,
    rec_cache::Dict{Int, Symbol},
    component_weights::Vector{Float64};
    cost_model::CostModel=default_cost_model(),
    remaining_target::Set{Symbol}=Set{Symbol}(),
    ask_cost::Float64=0.1,
    meta_cost_this_turn::Float64=0.0,
    already_enriched::Bool=false
)::Float64
    action in META_ACTIONS && return compute_meta_eu(
        state, action, rec_cache, component_weights;
        cost_model=cost_model, meta_cost_this_turn=meta_cost_this_turn)

    action == :ask_llm && return -Inf  # handled by compute_sensor_voi at call site

    action == :ask_user && return 1.0 - ask_cost

    (action in PRIMITIVE_ACTIONS || action == :done) && return compute_eu_primitive(
        state, action, rec_cache, component_weights, remaining_target)

    0.0
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
            new_g = perturb_grammar(state.grammars[gid], freq_table, ALL_EMAIL_FEATURES_EXTENDED)
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
    build_email_observation_kernel(compiled_kernels, features, temporal_state, user_action) → Kernel

Build a kernel for conditioning. Each program evaluates features →
recommends an action. Compared to user_action:
- Matching recommendation → log(p) (correct prediction)
- Non-matching recommendation → log(1-p) (incorrect prediction)

Populates correct_cache in kernel params for per-component Beta update.
Always condition with obs=1.0.
"""
function build_email_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
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
                    ck.evaluate(features, temporal_state)
                end
                correct = recommended == user_action
                correct_cache[tag] = correct
                p = mean(m_or_θ.beta)
                correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end;
        params = Dict{Symbol, Any}(:correct_cache => correct_cache),
        likelihood_family = BetaBernoulli())
end

# ═══════════════════════════════════════
# Per-step conditioning (multi-step episodes)
# ═══════════════════════════════════════

"""
    build_step_kernel(compiled_kernels, features, temporal_state,
                       correct_actions, rec_cache) → Kernel

Build a kernel for per-step conditioning. A program's recommendation is correct
if it's in the set of actions that were still needed at this step.
"""
function build_step_kernel(
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any},
    correct_actions::Set{Symbol},
    rec_cache::Dict{Int, Symbol}
)
    correct_cache = Dict{Int, Bool}()
    obs_space = Finite([0.0, 1.0])

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used in condition"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                tag = m_or_θ.tag
                recommended = get(rec_cache, tag, :done)
                correct = recommended in correct_actions
                correct_cache[tag] = correct
                p = mean(m_or_θ.beta)
                correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end;
        params = Dict{Symbol, Any}(:correct_cache => correct_cache),
        likelihood_family = BetaBernoulli())
end

"""
    condition_step!(state, features, temporal_state, correct_actions, rec_cache)

Condition belief at a single step within a multi-step episode. Programs
recommending correct actions gain weight; others lose weight.
"""
function condition_step!(
    state::AgentState,
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any},
    correct_actions::Set{Symbol},
    rec_cache::Dict{Int, Symbol}
)
    k = build_step_kernel(state.compiled_kernels, features,
                           temporal_state, correct_actions, rec_cache)
    state.belief = condition(state.belief, k, 1.0)
    sync_prune!(state; threshold=-30.0)
    sync_truncate!(state; max_components=2000)
end

# ═══════════════════════════════════════
# Multi-step episode loop (spec §7.4)
# ═══════════════════════════════════════

"""
    run_episode!(state, email, target, temporal_state; ...) → Vector{Symbol}

Run a multi-step episode for one email using polling execution.
"""
function run_episode!(
    state::AgentState,
    email::Email,
    target::Set{Symbol},
    temporal_state::Dict{Symbol, Any};
    cost_model::CostModel=default_cost_model(),
    max_steps::Int=6,
    ask_cost::Float64=0.1,
    llm_config::LLMConfig=default_llm_config(),
    max_meta_per_step::Int=3,
    min_log_prior::Float64=-20.0,
    verbose::Bool=false
)
    ps = ProcessingState()
    episode_actions = Symbol[]

    for step in 1:max_steps
        # 1. Extract features with current processing state
        features = extract_features(email, ps)

        # 2. Inner loop: meta/sensor actions before domain primitive
        meta_cost = 0.0
        meta_taken = 0
        already_enriched = false
        chosen = :do_nothing

        while true
            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               features, temporal_state)
            w = weights(state.belief)
            remaining = remaining_target_actions(ps, target)

            action_eus = Dict{Symbol, Float64}()
            for a in PRIMITIVE_ALL_ACTIONS
                action_eus[a] = compute_eu_step(state, a, rec_cache, w;
                    cost_model=cost_model, remaining_target=remaining, ask_cost=ask_cost,
                    meta_cost_this_turn=meta_cost, already_enriched=already_enriched)
            end
            chosen = argmax(action_eus)

            # Handle meta-actions
            if chosen in META_ACTIONS && chosen != :do_nothing && meta_taken < max_meta_per_step
                elapsed = @elapsed execute_meta_action!(state, chosen;
                    action_space=vcat(PRIMITIVE_ACTIONS, [:done]),
                    min_log_prior=min_log_prior, verbose=verbose)
                observe_cost!(cost_model, chosen, elapsed)
                meta_taken += 1
                meta_cost += expected_cost(cost_model, chosen)
                features = extract_features(email, ps)
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
                continue
            end

            # Handle sensor action
            if chosen == :ask_llm && !already_enriched
                elapsed = @elapsed (features = llm_enrich_features(llm_config, email, features))
                observe_cost!(cost_model, :ask_llm, elapsed)
                already_enriched = true
                verbose && println("    [Sensor: ask_llm at episode step $step, $(round(elapsed, digits=2))s]")
                continue
            end

            break
        end

        # If chosen is still meta/sensor, fall back to primitive EU
        if chosen in META_ACTIONS || chosen in SENSOR_ACTIONS
            rec_cache_fb = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache_fb, state.compiled_kernels,
                               features, temporal_state)
            w_fb = weights(state.belief)
            remaining_fb = remaining_target_actions(ps, target)
            prim_eus = Dict{Symbol, Float64}()
            for a in vcat(PRIMITIVE_ACTIONS, [:ask_user])
                prim_eus[a] = compute_eu_step(state, a, rec_cache_fb, w_fb;
                    cost_model=cost_model, remaining_target=remaining_fb, ask_cost=ask_cost)
            end
            chosen = argmax(prim_eus)
        end

        # 3. Handle chosen action
        remaining = remaining_target_actions(ps, target)

        # Need rec_cache for conditioning
        rec_cache = Dict{Int, Symbol}()
        evaluate_programs!(rec_cache, state.compiled_kernels,
                           features, temporal_state)

        if chosen == :done
            correct = isempty(remaining) ? Set([:done]) : Set{Symbol}()
            condition_step!(state, features, temporal_state, correct, rec_cache)
            push!(episode_actions, :done)
            break
        elseif chosen == :ask_user
            remaining_now = remaining_target_actions(ps, target)
            for prim in remaining_now
                execute_primitive!(ps, prim)
            end
            push!(episode_actions, :ask_user)
            condition_step!(state, features, temporal_state, remaining_now, rec_cache)
            break
        elseif chosen in PRIMITIVE_ACTIONS
            remaining_before = remaining_target_actions(ps, target)
            execute_primitive!(ps, chosen)
            push!(episode_actions, chosen)
            condition_step!(state, features, temporal_state, remaining_before, rec_cache)
        else
            error("No action selected — EU routing failure. chosen=$chosen, remaining=$(remaining_target_actions(ps, target))")
        end

        # 4. Check completion
        if isempty(remaining_target_actions(ps, target))
            break
        end
    end

    episode_actions
end

# ═══════════════════════════════════════
# Main agent loop
# ═══════════════════════════════════════

"""
    run_agent(; corpus, user_pref, ...) → (metrics, state, evolved_grammars)

Main email agent loop. Processes emails sequentially, learning the user's
preference profile through approve/override feedback.
"""
function run_agent(;
    corpus::Vector{Email},
    user_pref::UserPreference,
    program_max_depth::Int = 3,
    min_log_prior::Float64 = -20.0,
    max_meta_per_step::Int = 3,
    ask_cost::Float64 = 0.1,
    cost_model::CostModel = default_cost_model(),
    population_grammar::Union{Nothing, Vector{Grammar}} = nothing,
    llm_config::LLMConfig = default_llm_config(),
    use_primitives::Bool = false,
    rng_seed::Int = 42,
    verbose::Bool = true
)
    Random.seed!(rng_seed)

    # 1. INITIALISE
    action_space_for_enum = use_primitives ? vcat(PRIMITIVE_ACTIONS, [:done]) : DOMAIN_ACTIONS

    grammar_pool = if population_grammar !== nothing
        copy(population_grammar)
    else
        use_primitives ? generate_email_seed_grammars_extended() : generate_email_seed_grammars()
    end

    if verbose
        println("Generated $(length(grammar_pool)) seed grammars")
        use_primitives && println("Mode: multi-step episodes (primitives)")
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
                                       action_space=action_space_for_enum,
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

    temporal_state = Dict{Symbol, Any}(:recent => Dict{Symbol, Float64}[])
    metrics = EmailMetricsTracker()

    # 2. MAIN LOOP
    for (step, email) in enumerate(corpus)
        correct_action = user_pref.decide(email)

        if use_primitives
            # Multi-step episode (spec §7.4)
            target = ACTION_TARGET_STATE[correct_action]
            ep_actions = run_episode!(state, email, target, temporal_state;
                cost_model=cost_model, max_steps=6, ask_cost=ask_cost,
                llm_config=llm_config, max_meta_per_step=max_meta_per_step,
                min_log_prior=min_log_prior, verbose=verbose)

            asked = :ask_user in ep_actions
            # Correctness: all target primitives were completed
            ps_check = ProcessingState()
            for a in ep_actions
                a != :ask_user && a != :done && execute_primitive!(ps_check, a)
            end
            all_done = isempty(remaining_target_actions(ps_check, target))
            action_correct_for_metrics = !asked && all_done

            # Surprise: use component-weighted approval for the first target action
            first_target = isempty(target) ? :done : first(target)
            w_now = weights(state.belief)
            rec_cache = Dict{Int, Symbol}()
            features_now = extract_features(email)
            evaluate_programs!(rec_cache, state.compiled_kernels, features_now, temporal_state)
            eu_first = compute_eu_primitive(state, first_target, rec_cache, w_now, target)
            surprise = -log(max(eu_first, 1e-300))

            if verbose
                println("Step $step: episode=$(ep_actions) " *
                        "(target=$correct_action, $(action_correct_for_metrics ? "✓" : asked ? "?" : "✗"), " *
                        "len=$(length(ep_actions)), components=$(length(state.belief.components)))")
            end

            w_post = weights(state.belief)
            gw = aggregate_grammar_weights(w_post, state.metadata)
            record_email!(metrics;
                          step=step, action_taken=correct_action,
                          correct_action=correct_action,
                          is_correct=action_correct_for_metrics, asked=asked,
                          grammar_weights=gw,
                          n_components=length(state.belief.components),
                          surprise=surprise, n_meta_actions=0, used_llm=false,
                          episode_length=length(ep_actions),
                          episode_action_list=ep_actions)
        else
            # Single-decision-per-email
            features = extract_features(email)

            meta_cost_this_turn = 0.0
            meta_actions_taken = 0
            already_enriched = false
            used_llm = false
            chosen_action = :do_nothing

            while true
                rec_cache = Dict{Int, Symbol}()
                evaluate_programs!(rec_cache, state.compiled_kernels,
                                   features, temporal_state)
                w = weights(state.belief)

                action_eus = Dict{Symbol, Float64}()
                for a in ALL_ACTIONS
                    action_eus[a] = compute_eu(state, a, rec_cache, w;
                        cost_model=cost_model, ask_cost=ask_cost,
                        meta_cost_this_turn=meta_cost_this_turn,
                        already_enriched=already_enriched)
                end
                # Sensor VOI: replaces ad-hoc entropy heuristic
                action_eus[:ask_llm] = compute_sensor_voi(
                    state, features, email, temporal_state;
                    action_space=DOMAIN_ACTIONS, cost_model=cost_model,
                    already_enriched=already_enriched)
                chosen_action = argmax(action_eus)

                if chosen_action in META_ACTIONS && chosen_action != :do_nothing &&
                   meta_actions_taken < max_meta_per_step
                    elapsed = @elapsed execute_meta_action!(state, chosen_action;
                        action_space=DOMAIN_ACTIONS, min_log_prior=min_log_prior,
                        verbose=verbose)
                    observe_cost!(cost_model, chosen_action, elapsed)
                    meta_actions_taken += 1
                    meta_cost_this_turn += expected_cost(cost_model, chosen_action)
                    sync_prune!(state; threshold=-30.0)
                    sync_truncate!(state; max_components=2000)
                    continue
                end

                if chosen_action == :ask_llm && !already_enriched
                    elapsed = @elapsed (features = llm_enrich_features(llm_config, email, features))
                    observe_cost!(cost_model, :ask_llm, elapsed)
                    already_enriched = true
                    used_llm = true
                    verbose && println("  [Sensor: ask_llm at step $step, $(round(elapsed, digits=2))s]")
                    continue
                end

                break
            end

            if chosen_action in META_ACTIONS || chosen_action in SENSOR_ACTIONS
                rec_cache = Dict{Int, Symbol}()
                evaluate_programs!(rec_cache, state.compiled_kernels,
                                   features, temporal_state)
                w = weights(state.belief)
                action_eus = Dict{Symbol, Float64}()
                for a in EMAIL_ACTIONS
                    action_eus[a] = compute_eu(state, a, rec_cache, w;
                        cost_model=cost_model, ask_cost=ask_cost)
                end
                chosen_action = argmax(action_eus)
            end

            asked = chosen_action == :ask_user
            action_correct_for_metrics = !asked && (chosen_action == correct_action)

            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               features, temporal_state)
            w = weights(state.belief)
            eu_correct = compute_eu(state, correct_action, rec_cache, w; ask_cost=ask_cost)
            surprise = -log(max(eu_correct, 1e-300))

            k = build_email_observation_kernel(
                state.compiled_kernels, features,
                temporal_state, correct_action)
            state.belief = condition(state.belief, k, 1.0)

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

            w_post = weights(state.belief)
            gw = aggregate_grammar_weights(w_post, state.metadata)
            record_email!(metrics;
                          step=step, action_taken=chosen_action,
                          correct_action=correct_action,
                          is_correct=action_correct_for_metrics, asked=asked,
                          grammar_weights=gw,
                          n_components=length(state.belief.components),
                          surprise=surprise, n_meta_actions=meta_actions_taken,
                          used_llm=used_llm)
        end
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
forward. Later users benefit from grammar pool transfer.
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
            cost_model=default_cost_model(),
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
