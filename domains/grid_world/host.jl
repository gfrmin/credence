#!/usr/bin/env julia
"""
    host.jl — Host driver for the grid-world program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
flat MixtureMeasure of TaggedBetaMeasures → DSL inference → action selection →
world step → repeat.

Meta-actions (enumerate_more, perturb_grammar, deepen) are evaluated before
each domain decision. The agent decides whether to invest in improving its
hypothesis space or proceed with the interact/move decision.

Tier 3: grid-world-specific. Uses Tier 1 (Credence DSL) and Tier 2
(ProgramSpace) for domain-independent inference machinery.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: expect, condition, draw, optimise, value, weights, mean
using Credence: CategoricalMeasure, BetaMeasure, TaggedBetaMeasure, MixtureMeasure
using Credence: Finite, Interval, Kernel, Measure
using Credence: density, log_density_at, prune, truncate
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: SensorConfig, SensorChannel
using Credence: enumerate_programs, compile_kernel
using Credence: analyse_posterior_subtrees, perturb_grammar
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef, ActionExpr, IfExpr
using Credence: SubprogramFrequencyTable

include("simulation.jl")
include("terminals.jl")
include("metrics.jl")

using Random

# ═══════════════════════════════════════
# Meta-action constants
# ═══════════════════════════════════════

const GW_META_ACTIONS = [:gw_enumerate_more, :gw_perturb_grammar, :gw_deepen, :gw_do_nothing]
const GW_ENUMERATE_COST = 0.05
const GW_PERTURB_COST = 0.05
const GW_DEEPEN_COST = 0.10

# ═══════════════════════════════════════
# Per-grammar sensor projection
# ═══════════════════════════════════════

"""
    project_per_grammar(true_state, grammars; ...) → Dict{Int, Vector{Float64}}

Project an entity's true state through each grammar's sensor config.
Returns a mapping from grammar_id to the projected sensor vector.
"""
function project_per_grammar(
    true_state::Vector{Float64},
    grammars::Vector{Grammar};
    entity_id::Int=0,
    temporal_state::Dict{Int, Vector{Vector{Float64}}}=Dict{Int, Vector{Vector{Float64}}}()
)
    result = Dict{Int, Vector{Float64}}()
    for g in grammars
        result[g.id] = project(true_state, g.sensor_config;
                               entity_id=entity_id, temporal_state=temporal_state)
    end
    result
end

# ═══════════════════════════════════════
# Build the observation kernel
# ═══════════════════════════════════════

"""
    build_observation_kernel(compiled_kernels, grammar_sensor_vectors, temporal_state, true_type)

Build a single Kernel whose log_density dispatches per-component via
TaggedBetaMeasure tags. Each program evaluates features → recommends an
action symbol (:food or :enemy). Recommendation is compared to true_type.

Populates a correct_cache in kernel params for per-component Beta update
direction in the condition dispatch.
"""
function build_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any},
    true_type::Symbol
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
                correct = recommended == true_type
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
# Action selection
# ═══════════════════════════════════════

"""
    compute_eu_interact(belief, compiled_kernels, grammar_sensor_vectors, temporal_state)

Estimate P(enemy) from program recommendations weighted by posterior confidence,
then compute EU of interacting: P(enemy)*(-5) + P(food)*(+5).
"""
function compute_eu_interact(
    belief::MixtureMeasure,
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any}
)
    w = weights(belief)
    p_enemy = 0.0
    for (j, comp) in enumerate(belief.components)
        ck = compiled_kernels[j]
        haskey(grammar_sensor_vectors, ck.grammar_id) || continue
        sv = grammar_sensor_vectors[ck.grammar_id]
        rec = ck.evaluate(sv, temporal_state)
        mean_j = mean(comp.beta)
        # If program recommends :enemy and is correct (mean_j), entity is enemy
        # If program recommends :food and is correct (mean_j), entity is food → P(enemy) = 1-mean_j
        p_enemy += w[j] * (rec == :enemy ? mean_j : 1.0 - mean_j)
    end
    energy_enemy = -5.0
    energy_food = 5.0
    p_enemy * energy_enemy + (1.0 - p_enemy) * energy_food
end

function select_action(eu_interact::Float64, nearest_dist::Float64)
    if nearest_dist <= 1 && eu_interact >= -1e-10  # indifference → explore (robust to float error)
        return INTERACT
    elseif nearest_dist <= 1 && eu_interact < -1e-10
        return rand([MOVE_N, MOVE_S, MOVE_E, MOVE_W])
    else
        return rand([MOVE_N, MOVE_S, MOVE_E, MOVE_W])
    end
end

# ═══════════════════════════════════════
# Meta-action EU and execution
# ═══════════════════════════════════════

"""
    mean_observation_count_gw(state) → Float64

Average number of observations across components: mean(α + β - 2).
"""
function mean_observation_count_gw(state::AgentState)::Float64
    isempty(state.belief.components) && return 0.0
    total = 0.0
    for comp in state.belief.components
        tbm = comp::TaggedBetaMeasure
        total += tbm.beta.alpha + tbm.beta.beta - 2.0
    end
    total / length(state.belief.components)
end

"""
    compute_gw_meta_eu(state, action, eu_interact, n_obs; meta_cost_this_turn) → Float64

EU for grid-world meta-actions. Uses |eu_interact| as confidence proxy:
low |eu| means the agent is near indifference → meta-actions have high VOI.
"""
function compute_gw_meta_eu(
    state::AgentState,
    action::Symbol,
    eu_interact::Float64,
    n_obs::Float64;
    meta_cost_this_turn::Float64=0.0
)::Float64
    action == :gw_do_nothing && return -Inf

    confidence = abs(eu_interact) / 5.0  # normalize by max reward magnitude
    uncertainty_benefit = (1.0 - confidence) / (1.0 + 0.1 * n_obs)

    if action == :gw_enumerate_more
        return uncertainty_benefit * 0.5 - GW_ENUMERATE_COST - meta_cost_this_turn
    elseif action == :gw_perturb_grammar
        base = n_obs > 5.0 ? uncertainty_benefit * 0.6 : 0.0
        return base - GW_PERTURB_COST - meta_cost_this_turn
    elseif action == :gw_deepen
        return uncertainty_benefit * 0.4 - GW_DEEPEN_COST - meta_cost_this_turn
    end
    -Inf
end

"""
    execute_gw_meta_action!(state, action; ...) → Int

Execute a grid-world meta-action. Returns the number of programs added.
"""
function execute_gw_meta_action!(
    state::AgentState,
    action::Symbol;
    include_temporal::Bool=false,
    verbose::Bool=false
)::Int
    gw_action_space = Symbol[:food, :enemy]

    if action == :gw_enumerate_more
        top_gids = top_k_grammar_ids(state, 3)
        n_added = 0
        for gid in top_gids
            haskey(state.grammars, gid) || continue
            n_added += add_programs_to_state!(state, state.grammars[gid],
                state.current_max_depth;
                action_space=gw_action_space, include_temporal=include_temporal)
        end
        verbose && println("  [Meta: enumerate_more → +$n_added components]")
        return n_added

    elseif action == :gw_perturb_grammar
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
                action_space=gw_action_space, include_temporal=include_temporal)
        end
        verbose && println("  [Meta: perturb_grammar → +$n_added components]")
        return n_added

    elseif action == :gw_deepen
        state.current_max_depth += 1
        top_gids = top_k_grammar_ids(state, 3)
        n_added = 0
        for gid in top_gids
            haskey(state.grammars, gid) || continue
            n_added += add_programs_to_state!(state, state.grammars[gid],
                state.current_max_depth;
                action_space=gw_action_space, include_temporal=include_temporal)
        end
        verbose && println("  [Meta: deepen → depth=$(state.current_max_depth), +$n_added components]")
        return n_added
    end
    0
end

# ═══════════════════════════════════════
# Main agent loop
# ═══════════════════════════════════════

function run_agent(;
    world_rules::Vector{Symbol}=[:colour_typed],
    max_steps::Int=200,
    regime_change_steps::Vector{Int}=Int[],
    program_max_depth::Int=3,
    max_meta_per_step::Int=3,
    include_temporal::Bool=false,
    verbose::Bool=true,
    rng_seed::Int=42
)
    Random.seed!(rng_seed)

    # 1. INITIALISE
    world = create_world(world_rules[1])

    grammar_pool = generate_seed_grammars()
    if verbose
        println("Generated $(length(grammar_pool)) seed grammars")
    end

    # Enumerate all (grammar, program) pairs
    components = Measure[]
    log_prior_weights = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled_kernels = CompiledKernel[]
    all_programs = Program[]

    idx = 0
    for g in grammar_pool
        programs = enumerate_programs(g, program_max_depth; include_temporal, action_space=[:food, :enemy])
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

    # Temporal state
    temporal_window = TemporalWindow(max_history=10)
    temporal_state = Dict{Symbol, Any}(:recent => Vector{Float64}[])

    metrics = MetricsTracker()

    # 2. MAIN LOOP
    regime_idx = 1

    for step in 1:max_steps
        # Regime change
        if step in regime_change_steps
            regime_idx = min(regime_idx + 1, length(world_rules))
            set_rule!(world, world_rules[regime_idx])
            if verbose
                println("\n*** REGIME CHANGE at step $step → $(world_rules[regime_idx]) ***\n")
            end
        end

        # Observe entities
        entity_states = get_entity_states(world)
        update!(temporal_window, entity_states)

        # Update temporal state for compiled kernels
        for (eid, _) in entity_states
            true_st = last(temporal_window.history[eid])
            sv = project(true_st, full_sensor_config();
                         entity_id=eid, temporal_state=temporal_window.history)
            push!(get!(temporal_state, :recent, Vector{Float64}[]), sv)
            while length(temporal_state[:recent]) > 10
                popfirst!(temporal_state[:recent])
            end
        end

        # Find nearest entity
        nearest = nearest_entity(world)
        meta_actions_taken = 0

        if nearest !== nothing
            eid, entity = nearest
            dist = abs(entity.pos.x - world.agent_pos.x) + abs(entity.pos.y - world.agent_pos.y)

            # Per-grammar sensor projection for this entity
            true_st = entity_true_state(entity, world.agent_pos, world.config.grid_size)
            grammar_sensor_vectors = project_per_grammar(
                true_st, collect(values(state.grammars));
                entity_id=eid, temporal_state=temporal_window.history)

            # Meta-action inner loop: improve hypothesis space before domain decision
            meta_cost_this_turn = 0.0
            while meta_actions_taken < max_meta_per_step
                eu = compute_eu_interact(state.belief, state.compiled_kernels,
                                          grammar_sensor_vectors, temporal_state)
                n_obs = mean_observation_count_gw(state)

                best_meta_eu = -Inf
                best_meta = :gw_do_nothing
                for ma in GW_META_ACTIONS
                    meu = compute_gw_meta_eu(state, ma, eu, n_obs;
                                              meta_cost_this_turn=meta_cost_this_turn)
                    if meu > best_meta_eu
                        best_meta_eu = meu
                        best_meta = ma
                    end
                end

                best_meta_eu <= 0.0 && break

                execute_gw_meta_action!(state, best_meta;
                    include_temporal=include_temporal, verbose=verbose)
                meta_actions_taken += 1
                meta_cost_this_turn += (best_meta == :gw_deepen ? GW_DEEPEN_COST :
                                        best_meta == :gw_perturb_grammar ? GW_PERTURB_COST :
                                        GW_ENUMERATE_COST)

                # Re-project for new grammars
                grammar_sensor_vectors = project_per_grammar(
                    true_st, collect(values(state.grammars));
                    entity_id=eid, temporal_state=temporal_window.history)
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
            end

            # Domain decision
            eu = compute_eu_interact(state.belief, state.compiled_kernels,
                                      grammar_sensor_vectors, temporal_state)
            action = select_action(eu, Float64(dist))
        else
            action = rand([MOVE_N, MOVE_S, MOVE_E, MOVE_W])
        end

        # Execute action
        feedback = world_step!(world, action)

        # If we got feedback, condition belief
        prediction_correct = false
        surprise = 0.0
        energy_delta = feedback !== nothing ? feedback : 0.0

        if feedback !== nothing
            is_enemy = feedback < 0
            true_type = is_enemy ? :enemy : :food

            # Compute P(enemy) and surprise before conditioning
            if nearest !== nothing
                w = weights(state.belief)
                p_enemy_val = 0.0
                for (j, comp) in enumerate(state.belief.components)
                    ck = state.compiled_kernels[j]
                    haskey(grammar_sensor_vectors, ck.grammar_id) || continue
                    sv = grammar_sensor_vectors[ck.grammar_id]
                    rec = ck.evaluate(sv, temporal_state)
                    mean_j = mean(comp.beta)
                    p_enemy_val += w[j] * (rec == :enemy ? mean_j : 1.0 - mean_j)
                end
                p_obs = is_enemy ? p_enemy_val : (1.0 - p_enemy_val)
                surprise = -log(max(p_obs, 1e-300))
            else
                surprise = 0.0
                p_enemy_val = 0.5
            end

            # Build conditioning kernel
            if nearest !== nothing
                eid, entity = nearest
                true_st = entity_true_state(entity, world.agent_pos, world.config.grid_size)
                grammar_sensor_vectors = project_per_grammar(
                    true_st, collect(values(state.grammars));
                    entity_id=eid, temporal_state=temporal_window.history)
                k = build_observation_kernel(
                    state.compiled_kernels, grammar_sensor_vectors, temporal_state, true_type)
            else
                # Fallback: uniform kernel
                k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
                    _ -> error("not used"),
                    (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))
            end

            # Single condition call
            state.belief = condition(state.belief, k, 1.0)

            # Prune and truncate
            sync_prune!(state; threshold=-30.0)
            sync_truncate!(state; max_components=2000)

            # Was our prediction correct?
            prediction_correct = (p_enemy_val > 0.5) == is_enemy

            if verbose
                meta_str = meta_actions_taken > 0 ? ", meta=$meta_actions_taken" : ""
                println("Step $step: $(action) → $(is_enemy ? "ENEMY" : "FOOD") " *
                        "(predicted $(p_enemy_val > 0.5 ? "enemy" : "food"), " *
                        "P(enemy)=$(round(p_enemy_val, digits=3)), " *
                        "surprise=$(round(surprise, digits=2)), " *
                        "energy=$(round(world.agent_energy, digits=1)), " *
                        "components=$(length(state.belief.components))$meta_str)")
            end
        end

        # Record metrics
        w = weights(state.belief)
        gw = aggregate_grammar_weights(w, state.metadata)
        tp = top_k_programs(w, state.metadata; k=5)
        record!(metrics;
                step=step,
                grammar_weights=gw,
                top_programs=tp,
                correct=prediction_correct,
                energy=energy_delta,
                surprise=surprise,
                n_components=length(state.belief.components),
                n_grammars=length(unique(gi for (gi, _) in state.metadata)),
                n_meta_actions=meta_actions_taken)

        # Respawn entities if all dead
        alive = count(e -> e.alive, world.entities)
        if alive == 0
            world.entities = spawn_entities(world.config.rule_name, world.config.grid_size)
        end
    end

    if verbose
        print_summary(metrics; last_n=20)
    end

    (metrics, state, collect(values(state.grammars)))
end

# ═══════════════════════════════════════
# Entry point
# ═══════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("Program-Space Bayesian Agent")
    println("=" ^ 60)

    println("\n--- Single regime: colour-typed ---")
    metrics1, _, _ = run_agent(
        world_rules=[:colour_typed],
        max_steps=100,
        verbose=true)

    println("\n\n--- Regime change: colour → motion ---")
    metrics2, _, _ = run_agent(
        world_rules=[:colour_typed, :motion_typed],
        max_steps=150,
        regime_change_steps=[75],
        verbose=true)
end
