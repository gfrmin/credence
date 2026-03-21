#!/usr/bin/env julia
"""
    host.jl — Host driver for the grid-world program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
flat MixtureMeasure of TaggedBetaMeasures → DSL inference → action selection →
world step → repeat.

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
using Credence: aggregate_grammar_weights
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef
using Credence: SubprogramFrequencyTable

include("simulation.jl")
include("terminals.jl")
include("metrics.jl")

using Random

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
    build_observation_kernel(compiled_kernels, grammar_sensor_vectors, temporal_state)

Build a single Kernel whose log_density dispatches per-component via
TaggedBetaMeasure tags. Each tag indexes into compiled_kernels; the
compiled kernel's grammar_id selects the right sensor vector.

Predicate results are cached in a side-channel Dict populated during
_predictive_ll and read during condition, avoiding double evaluation.
"""
function build_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    grammar_sensor_vectors::Dict{Int, Vector{Float64}},
    temporal_state::Dict{Symbol, Any}
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
                if fired
                    p = mean(m_or_θ.beta)
                    obs == 1.0 ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
                else
                    log(0.5)  # non-firing → base-rate prediction (50/50)
                end
            else
                # Scalar θ fallback
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end)
end

# ═══════════════════════════════════════
# Action selection
# ═══════════════════════════════════════

function compute_eu_interact(belief::MixtureMeasure)
    p_enemy = expect(belief, θ -> θ)
    energy_enemy = -5.0
    energy_food = 5.0
    p_enemy * energy_enemy + (1.0 - p_enemy) * energy_food
end

function select_action(eu_interact::Float64, nearest_dist::Float64)
    if nearest_dist <= 1 && eu_interact >= 0
        return INTERACT
    elseif nearest_dist <= 1 && eu_interact < 0
        return rand([MOVE_N, MOVE_S, MOVE_E, MOVE_W])
    else
        return rand([MOVE_N, MOVE_S, MOVE_E, MOVE_W])
    end
end

# ═══════════════════════════════════════
# Main agent loop
# ═══════════════════════════════════════

function run_agent(;
    world_rules::Vector{Symbol}=[:colour_typed],
    max_steps::Int=200,
    regime_change_steps::Vector{Int}=Int[],
    program_max_depth::Int=2,
    grammar_perturbation_interval::Int=50,
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
        programs = enumerate_programs(g, program_max_depth; include_temporal)
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

        if nearest !== nothing
            eid, entity = nearest
            dist = abs(entity.pos.x - world.agent_pos.x) + abs(entity.pos.y - world.agent_pos.y)

            # Per-grammar sensor projection for this entity
            true_st = entity_true_state(entity, world.agent_pos, world.config.grid_size)
            grammar_sensor_vectors = project_per_grammar(
                true_st, grammar_pool;
                entity_id=eid, temporal_state=temporal_window.history)

            # Build kernel that dispatches per-component via TaggedBetaMeasure tags
            k = build_observation_kernel(
                state.compiled_kernels, grammar_sensor_vectors, temporal_state)

            # EU of interacting
            eu = compute_eu_interact(state.belief)
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
            obs = is_enemy ? 1.0 : 0.0

            # Compute surprise before conditioning
            p_enemy = expect(state.belief, θ -> θ)
            p_obs = is_enemy ? p_enemy : (1.0 - p_enemy)
            surprise = -log(max(p_obs, 1e-300))

            # Build conditioning kernel (need per-grammar projection for entity)
            if nearest !== nothing
                eid, entity = nearest
                true_st = entity_true_state(entity, world.agent_pos, world.config.grid_size)
                grammar_sensor_vectors = project_per_grammar(
                    true_st, grammar_pool;
                    entity_id=eid, temporal_state=temporal_window.history)
                k = build_observation_kernel(
                    state.compiled_kernels, grammar_sensor_vectors, temporal_state)
            else
                # Fallback: uniform kernel
                k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
                    _ -> error("not used"),
                    (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))
            end

            # Single condition call — TaggedBetaMeasure dispatches handle per-component logic
            state.belief = condition(state.belief, k, obs)

            # Prune and truncate (synced with parallel arrays)
            sync_prune!(state; threshold=-30.0)
            sync_truncate!(state; max_components=2000)

            # Was our prediction correct?
            prediction_correct = (p_enemy > 0.5) == is_enemy

            if verbose
                println("Step $step: $(action) → $(is_enemy ? "ENEMY" : "FOOD") " *
                        "(predicted $(p_enemy > 0.5 ? "enemy" : "food"), " *
                        "P(enemy)=$(round(p_enemy, digits=3)), " *
                        "surprise=$(round(surprise, digits=2)), " *
                        "energy=$(round(world.agent_energy, digits=1)), " *
                        "components=$(length(state.belief.components)))")
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
                n_grammars=length(unique(gi for (gi, _) in state.metadata)))

        # Periodic grammar perturbation
        if step % grammar_perturbation_interval == 0 && step < max_steps
            w = weights(state.belief)

            freq_table = analyse_posterior_subtrees(
                state.all_programs, w;
                min_frequency=0.01, min_complexity=2)

            top_gids = sort(collect(keys(gw)), by=gi -> -get(gw, gi, 0.0))[1:min(3, length(gw))]

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

                new_programs = enumerate_programs(new_g, program_max_depth; include_temporal)
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

        # Respawn entities if all dead
        alive = count(e -> e.alive, world.entities)
        if alive == 0
            world.entities = spawn_entities(world.config.rule_name, world.config.grid_size)
        end
    end

    if verbose
        print_summary(metrics; last_n=20)
    end

    (metrics, state, grammar_pool)
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
