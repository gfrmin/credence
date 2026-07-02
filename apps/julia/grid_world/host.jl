#!/usr/bin/env julia
# Role: brain-side application
"""
    host.jl — Host driver for the grid-world program-space agent

Orchestrates: grammar pool → program enumeration → kernel compilation →
flat MixturePrevision of TaggedBetaPrevisions → DSL inference → action selection →
world step → repeat.

Meta-actions (enumerate_more, perturb_grammar, deepen) are evaluated before
each domain decision. The agent decides whether to invest in improving its
hypothesis space or proceed with the interact/move decision.

Tier 3: grid-world-specific. Uses Tier 1 (Credence DSL) and Tier 2
(ProgramSpace) for domain-independent inference machinery.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: expect, condition, draw, optimise, value, weights, mean
using Credence: CategoricalMeasure, BetaPrevision, TaggedBetaPrevision, MixturePrevision
using Credence: Finite, Interval, Kernel, Measure
using Credence: density, log_density_at, prune, truncate
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: enumerate_programs, compile_kernel
using Credence: analyse_posterior_subtrees, perturb_grammar
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef, ActionExpr, IfExpr
using Credence: SubprogramFrequencyTable
# Move 3 — the belief-aware exploration budget (threshold refinement) + the Move-2 saturation signal.
using Credence: explore_grammar, explore_features, ExploreObservation
using Credence: update_learning_regime, plateau_probability, reset_learning_regime!
# Dominance move — the real single-currency argmax at the selection seam.
using Credence: exploration_voi, feature_discovery_voi, perturbation_voc, entropy

include("simulation.jl")
include("terminals.jl")
include("metrics.jl")

using Random

# ═══════════════════════════════════════
# Meta-action constants
# ═══════════════════════════════════════

# Tie order is load-bearing: the selection argmax resolves ties by first-listed. Enumerate
# (breadth) precedes deepen (depth) so equal escape-mass scores take the cheaper op first
# (dominance-design §5.1's named refinement — entropy does not distinguish them — deferred).
const GW_META_ACTIONS = [:gw_enumerate_more, :gw_perturb_grammar, :gw_deepen, :gw_explore,
                         :gw_add_feature, :gw_do_nothing]
# One currency — Δ log-evidence, nats (Move 5) — and one declared price. The escape-mass ops
# (:gw_enumerate_more/:gw_deepen) have no exact VOI: their value is the un-entertained tail's,
# myopic-unreachable by construction, so they are scored by posterior entropy (a NAMED HEURISTIC,
# dominance-design §0/§5.1) against this compute price: one bit. It is utility DATA — the
# caller's declared price of search compute, overridable via run_agent(escape_cost=…) — not a
# decision mechanism. Every proxy constant of the pre-dominance scheme (BASE/COST/VOI_FLOOR and
# the inline confidence arithmetic) is retired; the exact and surrogate tiers price their own
# compute through the engine's compute_cost kwargs (default 0.0).
const GW_ESCAPE_COST_DEFAULT = log(2.0)

# ═══════════════════════════════════════
# Build the observation kernel
# ═══════════════════════════════════════

"""
    build_observation_kernel(compiled_kernels, features, temporal_state, true_type)

Build a single Kernel whose log_density dispatches per-component via
TaggedBetaPrevision tags. Each program evaluates features → recommends an
action symbol (:food or :enemy). Recommendation is compared to true_type.

Populates a correct_cache in kernel params for per-component Beta update
direction in the condition dispatch.
"""
function build_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any},
    true_type::Symbol
)
    recommendation_cache = Dict{Int, Symbol}()
    correct_cache = Dict{Int, Bool}()
    obs_space = Finite([0.0, 1.0])

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used in condition"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaPrevision
                tag = m_or_θ.tag
                recommended = get!(recommendation_cache, tag) do
                    ck = compiled_kernels[tag]
                    ck.evaluate(features, temporal_state)
                end
                correct = recommended == true_type
                correct_cache[tag] = correct
                p = mean(m_or_θ.beta)
                correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))  # credence-lint: allow — precedent:declarative-construction — Kernel log-density closure: Bernoulli likelihood from Beta mean
            else
                obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
            end
        end;
        params = Dict{Symbol, Any}(:correct_cache => correct_cache),
        likelihood_family = BetaBernoulli())
end

# ═══════════════════════════════════════
# Action selection
# ═══════════════════════════════════════

"""
    compute_eu_interact(belief, compiled_kernels, features, temporal_state)

Estimate P(enemy) from program recommendations weighted by posterior confidence,
then compute EU of interacting: P(enemy)*(-5) + P(food)*(+5).
"""
function compute_eu_interact(
    belief::MixturePrevision,
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any}
)
    energy_enemy = -5.0
    energy_food = 5.0
    # Per-component EU is affine in the Beta mean θ_j: a program recommending
    # :enemy contributes energy_food + (energy_enemy-energy_food)·θ_j (it is right
    # with prob θ_j → entity is enemy), one recommending :food contributes the
    # complement, energy_enemy + (energy_food-energy_enemy)·θ_j. FiringChoice
    # selects the branch per component and `expect` does the weighted mixture sum.
    fired = [compiled_kernels[j].evaluate(features, temporal_state) == :enemy
             for j in eachindex(compiled_kernels)]
    d = energy_enemy - energy_food
    expect(belief, FiringChoice(fired,
        LinearCombination(Tuple{Float64, TestFunction}[( d, Identity())], energy_food),
        LinearCombination(Tuple{Float64, TestFunction}[(-d, Identity())], energy_enemy)))
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
    score_gw_meta_actions(state, explore_buffer; escape_cost) → Dict{Symbol, Float64}

Score every grid-world meta-action in the ONE currency — Δ log-evidence, nats
(Move 5; dominance-design §4). Each op is priced at its own fidelity:

    :gw_explore          plateau · exploration_voi          exact re-conditioned lookahead
    :gw_add_feature      plateau · feature_discovery_voi    exact lookahead; hard-gated on
                                                            thresholds-exhausted (attribution)
    :gw_perturb_grammar  perturbation_voc                   prior-only surrogate, never
                                                            re-conditioned (§5.3 cascade)
    :gw_enumerate_more,  entropy(belief) − escape_cost      escape-mass HEURISTIC (named, §0),
    :gw_deepen                                              eligible only when the exact tier
                                                            is non-positive (saturation-ordered
                                                            strictly below any positive exact VOI)
    :gw_do_nothing       0.0                                the act-now reference

`plateau` is the regime-marginalised soft gate (Move 2): with probability `plateau` the
residual is a persistent plateau and the measured Δℓ is real; with probability 1−plateau
the agent is still learning and the residual decays on its own (VOI ≈ 0) — so
E[VOI] = plateau·Δℓ. Never a hard threshold. threshold_exhausted stays HARD on
:gw_add_feature with zero constants: a feature Δℓ measured against a coarse grid is
confounded — inflated by residual that refinement would ALSO capture — so the
un-confounded baseline exists iff no positive-VOI refinement remains, i.e. the already-
computed exploration score is ≤ 0. Recomputed each call, never carried (the ladder is
cyclic). The compression rung stays soft (#174 PR 2): perturbation_voc competes in the
argmax at its own prior-only score, no veto in either direction.

Asserted by test_grid_world_meta.jl.
"""
function score_gw_meta_actions(
    state::AgentState,
    explore_buffer::Vector{ExploreObservation};
    escape_cost::Float64 = GW_ESCAPE_COST_DEFAULT
)::Dict{Symbol, Float64}
    gw_action_space = Symbol[:food, :enemy]
    scored = Dict{Symbol, Float64}(:gw_do_nothing => 0.0)

    top = top_k_grammar_ids(state, 1)
    if isempty(top)
        for ma in GW_META_ACTIONS
            ma == :gw_do_nothing || (scored[ma] = -Inf)
        end
        return scored
    end
    g_top = state.grammars[top[1]]

    # Prior-only surrogate (the cheap fidelity): best compression net-VOC over the posterior
    # subtree table. Never re-conditioned at selection (§5.3 — real VOI replaces proxy-for-exact
    # on the exploration ops ONLY; flattening the cascade would delete its EU-optimal
    # evaluation-cost ordering, Move 5 §4/§6).
    ft = analyse_posterior_subtrees(state.all_programs, weights(state.belief);
                                    min_frequency = 0.01, min_complexity = 2)
    scored[:gw_perturb_grammar] = perturbation_voc(g_top, ft)

    # Exact re-conditioned lookahead tier, regime-marginalised by the plateau soft gate.
    plateau = plateau_probability(state.learning_regime)
    voi_explore = exploration_voi(g_top, explore_buffer, state.current_max_depth;
                                  action_space = gw_action_space)
    scored[:gw_explore] = plateau * voi_explore
    scored[:gw_add_feature] = voi_explore > 0.0 ? -Inf :
        plateau * feature_discovery_voi(g_top, explore_buffer, ALL_GW_FEATURES,
                                        state.current_max_depth;
                                        action_space = gw_action_space)

    # Escape-mass heuristic tier, saturation-ordered strictly below any positive exact VOI (§0):
    # eligible only once the exact tier is non-positive, then scored by posterior entropy against
    # the declared compute price. :gw_enumerate_more with every top grammar already enumerated at
    # the current depth provably adds no hypothesis (add_programs_to_state! dedup) — but knowing
    # that costs the enumeration itself, so both ops carry the same H − cost and ties resolve by
    # GW_META_ACTIONS order (breadth before depth).
    exact_max = max(scored[:gw_explore], scored[:gw_add_feature])
    escape = exact_max > 0.0 ? -Inf : entropy(state.belief) - escape_cost
    scored[:gw_enumerate_more] = escape
    scored[:gw_deepen] = escape

    scored
end

"""
    default_eu_max_policy(scored) → Symbol

The agent's selection policy: deterministic argmax over `scored` in GW_META_ACTIONS
order (strict `>`, first-listed wins ties), with the act-now floor — returns
:gw_do_nothing unless some op's score strictly exceeds 0.0 (any op with net value ≤ 0
must lose to acting now, dominance-design §0). Benchmark baselines substitute this
function via run_agent(meta_policy=…); the seam adds no behaviour of its own.
"""
function default_eu_max_policy(scored::Dict{Symbol, Float64})::Symbol
    best = :gw_do_nothing
    best_score = 0.0
    for ma in GW_META_ACTIONS
        ma == :gw_do_nothing && continue
        s = get(scored, ma, -Inf)
        if s > best_score
            best_score = s
            best = ma
        end
    end
    best
end

# The seam passes the step so schedule-based benchmark baselines (fixed-schedule, clairvoyant)
# can key on it; the EU-max agent ignores it — its information is the scores.
default_eu_max_policy(scored::Dict{Symbol, Float64}, ::Int)::Symbol = default_eu_max_policy(scored)

"""
    execute_gw_meta_action!(state, action; ...) → Int

Execute a grid-world meta-action. Returns the number of programs added.
"""
function execute_gw_meta_action!(
    state::AgentState,
    action::Symbol,
    explore_buffer::Vector{ExploreObservation};
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
            new_g = perturb_grammar(state.grammars[gid], freq_table, ALL_GW_FEATURES)
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

    elseif action == :gw_explore
        # Belief-aware threshold refinement (Move 3): refine the top grammar's grid by the candidate whose
        # lookahead VOI (against the residual buffer) clears compute_cost; no-op if none does. The ONLY
        # meta-action that resets the residual regime — it expands the threshold alphabet, so pre-change
        # residuals are stale (Q1b, read precisely as "alphabet expansion" — perturb/deepen/enumerate are
        # within-alphabet and their effects show up in later residuals, so they do NOT reset).
        top = top_k_grammar_ids(state, 1)
        isempty(top) && return 0
        gid = top[1]
        new_g = explore_grammar(state.grammars[gid], explore_buffer, state.current_max_depth;
                                action_space=gw_action_space)
        new_g.id == gid && return 0   # no positive-VOI refinement → no-op
        state.grammars[new_g.id] = new_g
        n_added = add_programs_to_state!(state, new_g, state.current_max_depth;
            action_space=gw_action_space, include_temporal=include_temporal)
        reset_learning_regime!(state)
        empty!(explore_buffer)
        verbose && println("  [Meta: explore → grammar $gid→$(new_g.id) (threshold refined), +$n_added components]")
        return n_added

    elseif action == :gw_add_feature
        # Feature discovery (Move 4): add the host-furnished feature whose lookahead VOI (two-axis: fit Δℓ
        # MINUS the log2 prior-Occam explore_features charges internally) is greatest; no-op if none clears.
        # Like explore, an ALPHABET EXPANSION ⇒ resets the residual regime + clears the buffer (Q1b). The
        # candidate source is ALL_GW_FEATURES — the full superset the host already extracts every step, so a
        # selected feature's value is already in each observation's features Dict (base-feature SELECTION,
        # not construction). The reset re-opens the next explore pass on the NEW feature's grid (the cycle).
        top = top_k_grammar_ids(state, 1)
        isempty(top) && return 0
        gid = top[1]
        new_g = explore_features(state.grammars[gid], explore_buffer, ALL_GW_FEATURES,
                                 state.current_max_depth;
                                 action_space=gw_action_space)
        new_g.id == gid && return 0   # no positive-VOI feature → no-op
        state.grammars[new_g.id] = new_g
        n_added = add_programs_to_state!(state, new_g, state.current_max_depth;
            action_space=gw_action_space, include_temporal=include_temporal)
        reset_learning_regime!(state)
        empty!(explore_buffer)
        verbose && println("  [Meta: add_feature → grammar $gid→$(new_g.id) (feature acquired), +$n_added components]")
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
    rng_seed::Int=42,
    meta_policy::Function=default_eu_max_policy,
    escape_cost::Float64=GW_ESCAPE_COST_DEFAULT,
    respawn::Bool=false
)
    Random.seed!(rng_seed)

    # 1. INITIALISE
    world = create_world(world_rules[1]; respawn=respawn)

    grammar_pool = generate_seed_grammars()
    if verbose
        println("Generated $(length(grammar_pool)) seed grammars")
    end

    # Enumerate all (grammar, program) pairs
    components = TaggedBetaPrevision[]
    log_prior_weights = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled_kernels = CompiledKernel[]
    all_programs = Program[]

    idx = 0
    for g in grammar_pool
        programs = enumerate_programs(g, program_max_depth; include_temporal, action_space=[:food, :enemy])
        for (pi, p) in enumerate(programs)
            idx += 1
            push!(components, TaggedBetaPrevision(idx, BetaPrevision(1.0, 1.0)))
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

    belief = MixturePrevision(components, log_prior_weights)
    grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
    state = AgentState(belief, metadata, compiled_kernels, all_programs,
                       grammar_dict, program_max_depth)

    # The explore buffer (Move 3): host-side record of observations under the current threshold alphabet
    # (data, not belief — brain/body split, Q2b). Fed each conditioning step; the lookahead replays it;
    # cleared on threshold refinement (alphabet expansion).
    explore_buffer = ExploreObservation[]

    # Temporal state
    temporal_window = TemporalWindow(max_history=10)
    temporal_state = Dict{Symbol, Any}(:recent => Dict{Symbol, Float64}[])

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
        for (eid, feats) in entity_states
            push!(get!(temporal_state, :recent, Dict{Symbol, Float64}[]), feats)
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

            # Feature dict for this entity
            features = entity_features(entity, world.agent_pos, world.config.grid_size)

            # Meta-action inner loop: improve hypothesis space before domain decision.
            # One scored dict per iteration (each execution changes the state, so scores are
            # recomputed fresh — no within-turn cost accumulator; the loop is bounded by
            # max_meta_per_step and by the policy returning :gw_do_nothing). The policy owns
            # the stop rule: default_eu_max_policy implements the act-now floor; benchmark
            # baselines (random, fixed-schedule) may deliberately act on non-positive scores —
            # that waste is exactly what the dominance benchmark measures.
            while meta_actions_taken < max_meta_per_step
                scored = score_gw_meta_actions(state, explore_buffer; escape_cost=escape_cost)
                chosen = meta_policy(scored, step)::Symbol
                chosen == :gw_do_nothing && break

                execute_gw_meta_action!(state, chosen, explore_buffer;
                    include_temporal=include_temporal, verbose=verbose)
                meta_actions_taken += 1

                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
            end

            # Domain decision
            eu = compute_eu_interact(state.belief, state.compiled_kernels,
                                      features, temporal_state)
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
                eid, entity = nearest
                features = entity_features(entity, world.agent_pos, world.config.grid_size)
                # P(enemy) = Σ_j w_j·(rec_j == :enemy ? θ_j : 1-θ_j) — a per-component
                # firing split over the mixture (the same shape compute_eu_interact uses).
                fired = [state.compiled_kernels[j].evaluate(features, temporal_state) == :enemy
                         for j in eachindex(state.compiled_kernels)]
                p_enemy_val = expect(state.belief, FiringChoice(fired, Identity(),
                    LinearCombination(Tuple{Float64, TestFunction}[(-1.0, Identity())], 1.0)))
                p_obs = is_enemy ? p_enemy_val : (1.0 - p_enemy_val)
                surprise = -log(max(p_obs, 1e-300))

                # Feed the residual-plateau regime (the Move-2 saturation signal, wired here in Move 3 —
                # `surprise` IS ℓ = −log predictive) and accumulate the explore buffer. Belief-conditioning
                # below is untouched: this only updates the Move-2/3 side state.
                state.learning_regime = update_learning_regime(state.learning_regime,
                                                               state.last_residual, surprise)
                state.last_residual = surprise
                push!(explore_buffer, ExploreObservation(features,
                    Dict{Symbol, Any}(:recent => copy(get(temporal_state, :recent, Dict{Symbol, Float64}[]))),
                    Set([true_type]), surprise))
            else
                surprise = 0.0
                p_enemy_val = 0.5
            end

            # Build conditioning kernel
            if nearest !== nothing
                k = build_observation_kernel(
                    state.compiled_kernels, features, temporal_state, true_type)
            else
                # Fallback: uniform kernel
                k = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
                    _ -> error("not used"),
                    (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300));
                    likelihood_family = BetaBernoulli())
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
