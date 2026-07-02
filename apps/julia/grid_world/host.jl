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
using Credence: show_expr, FeatureRef, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef, ActionExpr, IfExpr
using Credence: SubprogramFrequencyTable
# Move 3 — the belief-aware exploration budget (threshold refinement) + the Move-2 saturation signal.
using Credence: explore_grammar, explore_features, ExploreObservation
using Credence: update_learning_regime, plateau_probability, reset_learning_regime!
# Dominance move — the real single-currency argmax at the selection seam.
using Credence: exploration_voi, feature_discovery_voi, perturbation_voc
# Belief-derived valuation — horizon-completed growth + the learned returns-to-growth model.
using Credence: exploration_fit, feature_discovery_fit, growth_value, complexity_logprior
using Credence: GrowthReturns, observe_yield!, escape_score, injection_yield_nats

include("simulation.jl")
include("terminals.jl")
include("metrics.jl")

using Random

# ═══════════════════════════════════════
# Meta-action constants
# ═══════════════════════════════════════

# Tie order is load-bearing: the selection argmax resolves ties by first-listed. Enumerate
# (breadth) precedes deepen (depth) so ties at the shared returns prior take the cheaper op
# first (ratified Q4 — equal priors, the tie order does the breadth-before-depth work; evidence
# differentiates the cells from the first firing on).
const GW_META_ACTIONS = [:gw_enumerate_more, :gw_perturb_grammar, :gw_deepen, :gw_explore,
                         :gw_add_feature, :gw_do_nothing]
# One currency — Δ log-evidence, nats (Move 5) — and one declared PRICE. The escape ops'
# VALUE side is no longer hand-written (the entropy heuristic is retired,
# belief-derived-valuation §2b): it is the learned returns-to-growth posterior's expected next
# yield. What remains declared is only this compute price — utility DATA, the caller's price of
# search compute, overridable via run_agent(op_compute_cost=…) — never a value claim (ratified
# Q6). The exact and surrogate tiers price their own compute through the engine's compute_cost
# kwargs (default 0.0).
const GW_OP_COMPUTE_COST_DEFAULT = log(2.0)
# The one-time prior-Occam charge of admitting one feature symbol (Δcomplexity = +1), charged at
# the score seam exactly as the engine's _best_feature_addition charges it internally — the two
# sites agree by both calling complexity_logprior(1; λ = log 2).
const GW_FEATURE_PRIOR_TERM = complexity_logprior(1; λ = log(2.0))

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
    score_gw_meta_actions(state, explore_buffer, returns, changed; op_compute_cost, horizon)
        → Dict{Symbol, Float64}

Score every grid-world meta-action in the ONE currency — Δ log-evidence, nats
(Move 5; belief-derived-valuation §2). Every score is a posterior expectation or a declared
datum — no hand-written value claims:

    :gw_explore          growth_value(fit, n_buf, plateau, H)             horizon-completed exact
                                                                          lookahead (§2a)
    :gw_add_feature      growth_value(fit, …; prior_term = −log 2)        as above + the one-time
                                                                          Occam charge; hard-gated
                                                                          on thresholds-exhausted
                                                                          (attribution, unchanged)
    :gw_perturb_grammar  -Inf (PROVISIONAL)                               removal-consumption
                                                                          redesign pending — see
                                                                          the inline comment
    :gw_enumerate_more,  escape_score(returns, op, changed)               LEARNED returns-to-growth
    :gw_deepen                                                            posterior (§2b) minus the
                                                                          declared compute price;
                                                                          competes FREELY (the
                                                                          saturation-ordering gate
                                                                          retired, ratified Q5)
    :gw_do_nothing       0.0                                              the act-now reference

`plateau` keeps its Move-2 semantics (*whether* the measured gain is real); `horizon` is the
expected remaining conditioning events (*how long* it pays) — declared episode data × the
observed event rate, host bookkeeping. `nothing` ⇒ the window-total valuation (H = n_buf,
the pre-move behaviour). The `fit_explore > 0` hard gate on :gw_add_feature is the attribution
argument (a feature Δℓ measured against a coarse grid is confounded by residual refinement
would also capture) — a measurement concern, valuation-independent, so it gates on FIT.

Score/edit consistency (Invariant 3, host level): with `compute_cost = 0` the engine's
edit-application floor (`fit > 0` at defaults) and this score's positivity
(`plateau·fit·(H/n) > 0` for `plateau, H > 0`) are the SAME predicate, so a chosen growth op
never no-ops and a no-op is never chosen (except the H = 0 last step, where the score is 0 and
the floor already keeps do_nothing).

Asserted by test_grid_world_meta.jl.
"""
function score_gw_meta_actions(
    state::AgentState,
    explore_buffer::Vector{ExploreObservation},
    returns::GrowthReturns,
    changed::Dict{Symbol, Bool};
    op_compute_cost::Float64 = GW_OP_COMPUTE_COST_DEFAULT,
    horizon::Union{Nothing, Float64} = nothing,
    voi_cache::Union{Nothing, Dict{NTuple{4, Int}, Float64}} = nothing,
    cache_epoch::Int = 0
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

    # PROVISIONAL (flagged for ratification, like :gw_deepen below): :gw_perturb_grammar is
    # scored -Inf, not by perturbation_voc. Empirically (2026-07-02 smoke), voc is a surrogate
    # execution falsifies: an applied REMOVAL adds a cleaned SIBLING grammar whose duplicate
    # programs can never displace the evidence-rich dirty incumbent, so the same removal
    # re-proposes at +log2 forever — a 3-ops/step treadmill of duplicate injection (the exact
    # shape perturbation.jl's own no-op docstring calls "a silent posterior reset — A3", missed
    # for the applied case because the old entropy tier always outbid voc). The correct fix is
    # REPLACEMENT semantics for removals (re-key components to the cleaned grammar, realising
    # the prior reclaim in their weights) — but that is a prior-revision write outside
    # `condition` and needs its own design doc (plus the Move-1 OQ-4 sound reference count).
    # Until that lands, perturb is out of the agent's argmax; score-blind baselines still
    # execute it as before.
    scored[:gw_perturb_grammar] = -Inf

    # Exact lookahead tier, horizon-completed (belief-derived-valuation §2a). The FIT halves are
    # PURE functions of (grammar, buffer, depth) — the memoisable component; the per-step
    # valuation (plateau, H) is applied through the engine's growth_value functional. The epoch
    # (owned by the run loop, bumped on growth-op execution and window trims) prevents key
    # collisions across content changes at equal length.
    n_buf = length(explore_buffer)
    plateau = plateau_probability(state.learning_regime)
    h = horizon === nothing ? Float64(n_buf) : horizon
    fit_explore = voi_cache === nothing ?
        exploration_fit(g_top, explore_buffer, state.current_max_depth;
                        action_space = gw_action_space) :
        get!(voi_cache, (1000 + cache_epoch, g_top.id, n_buf, state.current_max_depth)) do
            exploration_fit(g_top, explore_buffer, state.current_max_depth;
                            action_space = gw_action_space)
        end
    scored[:gw_explore] = growth_value(fit_explore, n_buf, plateau, h)
    if fit_explore > 0.0
        scored[:gw_add_feature] = -Inf
    else
        fit_feature = voi_cache === nothing ?
            feature_discovery_fit(g_top, explore_buffer, ALL_GW_FEATURES,
                                  state.current_max_depth; action_space = gw_action_space) :
            get!(voi_cache, (2000 + cache_epoch, g_top.id, n_buf, state.current_max_depth)) do
                feature_discovery_fit(g_top, explore_buffer, ALL_GW_FEATURES,
                                      state.current_max_depth; action_space = gw_action_space)
            end
        scored[:gw_add_feature] = growth_value(fit_feature, n_buf, plateau, h;
                                               prior_term = GW_FEATURE_PRIOR_TERM)
    end

    # Learned returns tier (belief-derived-valuation §2b): the posterior-predictive expected
    # next yield of each escape op in its (op, changed-since-last-fire) context, net of the
    # declared compute price. No eligibility gate (ratified Q5) — bounded prior optimism decays
    # under zero-yield evidence, which the old entropy score never did. Ties at the shared prior
    # resolve by GW_META_ACTIONS order (breadth before depth, ratified Q4).
    scored[:gw_enumerate_more] = escape_score(returns, :gw_enumerate_more,
                                              get(changed, :gw_enumerate_more, true);
                                              compute_cost = op_compute_cost)
    # PROVISIONAL (flagged for ratification — amends ratified Q5 for this one op): :gw_deepen
    # is scored -Inf, not by its learned returns. Empirically (2026-07-02 smoke), deepen is
    # structurally unpriceable in a free per-step argmax: one prior-optimism fire ratchets the
    # GLOBAL enumeration depth, exploding every subsequent lookahead ~100× (depth-4 candidate
    # enumeration ≈ 225k programs) — no flat declared price is honest for an op whose compute
    # cost is super-exponential in state. Bounded depth escalation is the drafted
    # escalate-depth design's brief (docs/escalate-depth branch); deepen re-enters the argmax
    # when that lands. Its returns cells stay tracked (harmless) for that day.
    scored[:gw_deepen] = -Inf

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
    ScoreBlind(f) — a meta-policy wrapper declaring that `f` never reads the scores.

Score-blind baselines (random, fixed-schedule) select ops without consulting `scored`;
computing the exact VOI lookaheads on their behalf is pure waste, so the seam skips
`score_gw_meta_actions` for policies wrapped in this. Behaviour-neutral by construction —
the wrapped policy receives an act-now-only dict it was never going to read.
"""
struct ScoreBlind <: Function
    f::Function
end
(p::ScoreBlind)(scored::Dict{Symbol, Float64}, step::Int) = p.f(scored, step)::Symbol
score_blind(::Function) = false
score_blind(::ScoreBlind) = true

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
                observations=explore_buffer,
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
                observations=explore_buffer,
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
                observations=explore_buffer,
                action_space=gw_action_space, include_temporal=include_temporal)
        end
        verbose && println("  [Meta: deepen → depth=$(state.current_max_depth), +$n_added components]")
        return n_added

    elseif action == :gw_explore
        # Belief-aware threshold refinement (Move 3): refine the top grammar's grid by the candidate whose
        # lookahead VOI (against the residual buffer) clears compute_cost; no-op if none does. Resets the
        # residual REGIME — it expands the threshold alphabet, so pre-change regime residuals are stale
        # (Q1b, read precisely as "alphabet expansion" — perturb/deepen/enumerate are within-alphabet and
        # do NOT reset). The BUFFER is retained: raw (features, correct_actions) records are world data,
        # alphabet-independent — they inform the coherent injection and age out via explore_window only
        # (coherent-injection-design.md §1, the Q2b amendment).
        top = top_k_grammar_ids(state, 1)
        isempty(top) && return 0
        gid = top[1]
        new_g = explore_grammar(state.grammars[gid], explore_buffer, state.current_max_depth;
                                action_space=gw_action_space)
        new_g.id == gid && return 0   # no positive-VOI refinement → no-op
        state.grammars[new_g.id] = new_g
        n_added = add_programs_to_state!(state, new_g, state.current_max_depth;
            observations=explore_buffer,
            action_space=gw_action_space, include_temporal=include_temporal)
        reset_learning_regime!(state)
        verbose && println("  [Meta: explore → grammar $gid→$(new_g.id) (threshold refined), +$n_added components]")
        return n_added

    elseif action == :gw_add_feature
        # Feature discovery (Move 4): add the host-furnished feature whose lookahead VOI (two-axis: fit Δℓ
        # MINUS the log2 prior-Occam explore_features charges internally) is greatest; no-op if none clears.
        # Like explore, an ALPHABET EXPANSION ⇒ resets the residual regime (buffer retained — Q2b
        # amendment, coherent-injection-design.md §1). The
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
            observations=explore_buffer,
            action_space=gw_action_space, include_temporal=include_temporal)
        reset_learning_regime!(state)
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
    op_compute_cost::Float64=GW_OP_COMPUTE_COST_DEFAULT,
    respawn::Bool=false,
    observe_adjacent::Bool=false,
    seed_grammars::Union{Nothing, Vector{Grammar}}=nothing,
    explore_window::Int=typemax(Int)
)
    Random.seed!(rng_seed)

    # 1. INITIALISE
    world = create_world(world_rules[1]; respawn=respawn)

    # The starting hypothesis-space vocabulary is task DATA the caller may declare (the
    # dominance benchmark starts from an impoverished basis so discovery is load-bearing);
    # default is the full stock pool.
    grammar_pool = seed_grammars === nothing ? generate_seed_grammars() : seed_grammars
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

    # The explore buffer (Move 3; Q2b as amended by coherent-injection-design.md §1): host-side
    # record of observations (data, not belief — brain/body split). Fed each conditioning step;
    # the lookahead replays it; the coherent injection conditions newcomers on it. Never cleared
    # by growth ops — explore_window aging is the sole trim. Each record's residual is the live
    # surprise (−log predictive), the incumbents' normalisation ledger the injection re-applies.
    explore_buffer = ExploreObservation[]
    # Exact memoisation of the pure lookahead FITS (see score_gw_meta_actions): epoch bumps on
    # growth-op execution and window trims so equal-length buffers with different content never
    # collide with stale keys. The per-step valuation (plateau, horizon) is applied outside the
    # cache through growth_value.
    voi_cache = Dict{NTuple{4, Int}, Float64}()
    cache_epoch = 0

    # The learned returns-to-growth belief (belief-derived-valuation §2b) + its bookkeeping DATA:
    # space_epoch counts hypothesis-space changes (any injection, any depth change);
    # last_fire_epoch records the epoch each escape op last fired under — the (op,
    # changed-since-last-fire) context bit. n_cond_events counts conditioning events for the
    # declared-horizon estimate H = event rate × remaining steps (ratified Q2: episodic hosts
    # declare; max_steps is this host's episode length).
    growth_returns = GrowthReturns(Symbol[:gw_enumerate_more, :gw_deepen])
    space_epoch = 0
    last_fire_epoch = Dict{Symbol, Int}()
    n_cond_events = 0

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
                # The declared-horizon estimate (ratified Q2): observed conditioning-event rate
                # × declared remaining episode steps. Bookkeeping counts — data, not beliefs.
                h_events = (n_cond_events / step) * (max_steps - step)
                changed = Dict{Symbol, Bool}(
                    op => get(last_fire_epoch, op, -1) != space_epoch
                    for op in (:gw_enumerate_more, :gw_deepen))
                scored = score_blind(meta_policy) ?
                    Dict{Symbol, Float64}(:gw_do_nothing => 0.0) :
                    score_gw_meta_actions(state, explore_buffer, growth_returns, changed;
                                          op_compute_cost=op_compute_cost, horizon=h_events,
                                          voi_cache=voi_cache, cache_epoch=cache_epoch)
                chosen = meta_policy(scored, step)::Symbol
                chosen == :gw_do_nothing && break

                n_added_meta = execute_gw_meta_action!(state, chosen, explore_buffer;
                    include_temporal=include_temporal, verbose=verbose)
                chosen in (:gw_explore, :gw_add_feature) && (cache_epoch += 1)
                meta_actions_taken += 1

                # The realised yield is an OBSERVATION (belief-derived-valuation §2b): measured
                # BEFORE prune/truncate (they may drop the very components), conditioned into the
                # returns belief at the context the op fired under. The op's own effect then
                # bumps the space epoch — other ops see a changed space; the op itself does not
                # (its post-fire epoch is recorded post-bump).
                if chosen in (:gw_enumerate_more, :gw_deepen)
                    y = injection_yield_nats(state, n_added_meta)
                    observe_yield!(growth_returns, chosen, changed[chosen], y)
                end
                (n_added_meta > 0 || chosen == :gw_deepen) && (space_epoch += 1)
                chosen in (:gw_enumerate_more, :gw_deepen) &&
                    (last_fire_epoch[chosen] = space_epoch)

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

        # Evidence for conditioning. An interaction's outcome labels the entity by its energy
        # sign (the historical channel). With the opt-in adjacent-inspection sensor
        # (observe_adjacent), the nearest entity's type is observed whenever the agent ends the
        # step adjacent to it, interaction or not — a host-provided observation (the task's
        # sensor model; the host's constitutional job is providing observations). It decouples
        # evidence flow from the energy decision: without it, two early negative interactions
        # freeze the myopic interact rule and the belief never receives data again.
        prediction_correct = false
        surprise = 0.0
        energy_delta = feedback !== nothing ? feedback : 0.0

        observed_type = nothing
        if feedback !== nothing
            observed_type = feedback < 0 ? :enemy : :food
        elseif observe_adjacent && nearest !== nothing && nearest[2].alive &&
               abs(nearest[2].pos.x - world.agent_pos.x) +
               abs(nearest[2].pos.y - world.agent_pos.y) <= 1
            observed_type = nearest[2].kind == ENEMY ? :enemy : :food
        end

        if observed_type !== nothing
            is_enemy = observed_type == :enemy
            true_type = observed_type

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
                n_cond_events += 1   # the horizon estimate's event count (declared-Q2 bookkeeping)
                push!(explore_buffer, ExploreObservation(features,
                    Dict{Symbol, Any}(:recent => copy(get(temporal_state, :recent, Dict{Symbol, Float64}[]))),
                    Set([true_type]), surprise))
                # The residual record's span is host task data (like TemporalWindow's
                # max_history): under non-stationarity an unbounded record mixes regimes and
                # the prequential mll correctly scores a stationary grammar as unable to
                # explain the whole sequence — suppressing discovery of the CURRENT regime's
                # predictor. Trimming shifts content at constant length, so the memo epoch
                # advances (stale (length, depth) keys must not hit).
                if length(explore_buffer) > explore_window
                    while length(explore_buffer) > explore_window
                        popfirst!(explore_buffer)
                    end
                    cache_epoch += 1
                end
                # Single condition call. Every condition has its buffer record above — the
                # ledger contract of coherent injection (agent_state.jl docstring): the buffer
                # must witness every normalisation the live weights absorb. `observed_type`
                # requires an adjacent entity on both branches, so nearest !== nothing whenever
                # evidence exists — the old nearest-less fallback kernel was unreachable and,
                # having no buffer record, would have broken the contract; removed.
                k = build_observation_kernel(
                    state.compiled_kernels, features, temporal_state, true_type)
                state.belief = condition(state.belief, k, 1.0)

                # Prune and truncate
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
            else
                surprise = 0.0
                p_enemy_val = 0.5
            end

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

    # 4th element (additive; callers destructuring three names are unaffected): the explore
    # buffer, for benchmark/diagnostic observability of the residual record.
    (metrics, state, collect(values(state.grammars)), explore_buffer)
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
