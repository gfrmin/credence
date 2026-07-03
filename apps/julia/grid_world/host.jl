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
# Removal consumption — replacement semantics for applied removals (the #189 deviation-3 discharge).
using Credence: replacement_value, best_replacement, replace_grammar_in_state!
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr, FeatureRef, GTExpr, LTExpr, AndExpr, OrExpr, NotExpr, NonterminalRef, ActionExpr, IfExpr
using Credence: SubprogramFrequencyTable
# Moves 3–4 as re-founded by the winners-curse design: the growth lookahead is a virtual
# injection; score and transition share ONE GrowthProposal (T-3.55 as identity).
using Credence: GrowthProposal, threshold_growth_proposal, feature_growth_proposal, adopt!
using Credence: ExploreObservation
using Credence: update_learning_regime, reset_learning_regime!
# Belief-derived valuation — horizon-completed growth + the learned returns-to-growth model.
using Credence: net_value
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
# (GW_FEATURE_PRIOR_TERM is RETIRED — winners-curse design §5 Q4: the one-time prior-Occam
# charge of a feature symbol is carried by each newcomer's own complexity prior INSIDE the
# union mixture; charging it again at the score seam would double-count. Pinned by
# test_virtual_injection.jl §1.)

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
    gw_growth_proposal(op, state, explore_buffer, growth_cache, cache_epoch; include_temporal)
        → Union{Nothing, GrowthProposal}

The ONE candidate function score and transition share (T-3.55): the growth class's virtual
injection for the current top grammar, memoised in `growth_cache` under
(op, cache_epoch, gid, n_buf, depth). `score_gw_meta_actions` prices what this returns;
`execute_gw_meta_action!` adopts the same object's scratch. With `growth_cache === nothing`
(engine tests, score-blind execute paths) the proposal is computed directly.
"""
function gw_growth_proposal(
    op::Symbol,
    state::AgentState,
    explore_buffer::Vector{ExploreObservation},
    growth_cache::Union{Nothing, Dict{Tuple{Symbol, Int, Int, Int, Int}, Union{Nothing, GrowthProposal}}},
    cache_epoch::Int;
    include_temporal::Bool = false
)::Union{Nothing, GrowthProposal}
    gw_action_space = Symbol[:food, :enemy]
    top = top_k_grammar_ids(state, 1)
    isempty(top) && return nothing
    g_top = state.grammars[top[1]]
    compute() = op === :explore ?
        threshold_growth_proposal(state, g_top, explore_buffer;
                                  action_space = gw_action_space,
                                  include_temporal = include_temporal) :
        feature_growth_proposal(state, g_top, explore_buffer, ALL_GW_FEATURES;
                                action_space = gw_action_space,
                                include_temporal = include_temporal)
    growth_cache === nothing && return compute()
    get!(compute, growth_cache,
         (op, cache_epoch, g_top.id, length(explore_buffer), state.current_max_depth))
end

"""
    score_gw_meta_actions(state, explore_buffer, returns, changed; op_compute_cost, horizon)
        → Dict{Symbol, Float64}

Score every grid-world meta-action in the ONE currency — Δ log-evidence, nats
(Move 5; belief-derived-valuation §2). Every score is a posterior expectation or a declared
datum — no hand-written value claims:

    :gw_explore          net_value(prop.yield_nats, op_compute_cost)      the VIRTUAL INJECTION
                                                                          (winners-curse §8.2):
    :gw_add_feature      as :gw_explore — NO prior term (Q4: the Occam    the union of ALL
                         charge rides inside the mixture); gated -Inf     candidates coherently
                         iff the threshold proposal's own score clears    injected on a scratch;
                         the floor (refinement fires first — §8.4)        the yield IS the union-
                                                                          over-incumbent window
                                                                          Bayes factor; fire when
                                                                          realised evidence clears
                                                                          the declared price (no
                                                                          flow, no ×H, no plateau
                                                                          — §8.2's wait-option
                                                                          argument)
    :gw_perturb_grammar  max over top-k gids of                           the exact realised
                         replacement_value(state, gid; compute_cost)      Δ log-evidence of
                                                                          consuming the best dead
                                                                          item (removal-
                                                                          consumption design §1)
    :gw_enumerate_more,  escape_score(returns, op, changed)               LEARNED returns-to-growth
    :gw_deepen                                                            posterior (§2b) minus the
                                                                          declared compute price;
                                                                          competes FREELY (the
                                                                          saturation-ordering gate
                                                                          retired, ratified Q5)
    :gw_do_nothing       0.0                                              the act-now reference

No horizon, no plateau at this seam (winners-curse §8.2): the wait-option argument cancels
the horizon-extrapolated term (the far-term value is common to adopt-now and
wait-and-re-decide), and the yield is realised posterior evidence — multiplying by P(plateau)
would charge the reality-of-gain doubt twice. The regime belief stays maintained (Move 2's
signal; the reset discipline is unchanged) but no longer multiplies scores.

Score/edit consistency (T-3.55, now an IDENTITY): the growth score prices a GrowthProposal —
the union scratch itself — memoised in `growth_cache` keyed by (op, cache_epoch, gid, n_buf,
depth); `execute_gw_meta_action!` reads the SAME cache entry and ADOPTS the scratch (a field
swap, zero recompute — ratified §5 Q1). The cache epoch bumps on EVERY hypothesis-space change
(not just growth fires — the proposal depends on the live belief, unlike the retired pure-fit
cache) and on window trims, so a stale scratch can never be priced or adopted. The execute-time
apply floor is `prop.yield_nats > 0`; the score's positivity (`yield > cost ≥ 0`) is strictly
stronger, so a chosen growth op never no-ops. Score-blind baselines execute without a prior
score — the same proposal is computed at execute time (the §5 Q1 fallback path).

Asserted by test_grid_world_meta.jl.
"""
function score_gw_meta_actions(
    state::AgentState,
    explore_buffer::Vector{ExploreObservation},
    returns::GrowthReturns,
    changed::Dict{Symbol, Bool};
    op_compute_cost::Float64 = GW_OP_COMPUTE_COST_DEFAULT,
    growth_cache::Union{Nothing, Dict{Tuple{Symbol, Int, Int, Int, Int}, Union{Nothing, GrowthProposal}}} = nothing,
    cache_epoch::Int = 0,
    include_temporal::Bool = false
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

    # Removal consumption (the #189 deviation-3 discharge; removal-consumption design §2): perturb
    # competes at the exact realised Δ log-evidence of the best replacement over the top-k
    # grammars — log1p((e^Δ − 1)·W_G) net of the declared price, where the transition is
    # REPLACEMENT (the cleaned grammar consumes its ancestor: components re-keyed, the reclaim
    # realised in their weights, the ancestor unregistered — so the same edit can never
    # re-propose; the sibling-injection treadmill is unrepresentable). The score and the executed
    # edit share one candidate function (`best_replacement`), T-3.55. A grammar with no dead item
    # scores 0.0 and loses to the act-now floor.
    # (No init: every metadata gid is registered — top_k_grammar_ids cannot yield an unregistered
    # gid, so the generator is non-empty whenever `top` was; an empty maximum failing loud is the
    # invariant check. A negative max is honest: worth less than acting now.)
    scored[:gw_perturb_grammar] = maximum(
        replacement_value(state, gid; compute_cost = op_compute_cost)
        for gid in top_k_grammar_ids(state, 3) if haskey(state.grammars, gid))

    # The virtual-injection tier (winners-curse §8.2): each growth class's GrowthProposal is
    # the union scratch, memoised so the fire adopts the very object the score priced (T-3.55
    # as identity). The score is the realised evidence net of the declared price — the yield
    # IS the union-over-incumbent window Bayes factor; nothing multiplies by a horizon.
    prop_explore = gw_growth_proposal(:explore, state, explore_buffer,
                                      growth_cache, cache_epoch;
                                      include_temporal = include_temporal)
    scored[:gw_explore] = net_value(
        prop_explore === nothing ? 0.0 : prop_explore.yield_nats, op_compute_cost)
    if scored[:gw_explore] > 0.0
        # Refinement fires first (§8.4): the attribution gate re-expressed — features wait
        # only while the threshold class itself clears its price, not on any positive fit.
        scored[:gw_add_feature] = -Inf
    else
        prop_feature = gw_growth_proposal(:add_feature, state, explore_buffer,
                                          growth_cache, cache_epoch;
                                          include_temporal = include_temporal)
        scored[:gw_add_feature] = net_value(
            prop_feature === nothing ? 0.0 : prop_feature.yield_nats, op_compute_cost)
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
    execute_gw_meta_action!(state, action; ...) → (n_added::Int, applied_replacement::Bool)

Execute a grid-world meta-action. `n_added` is the number of programs added (the injection-yield
observable); `applied_replacement` is true iff a `:gw_perturb_grammar` replacement actually fired —
a hypothesis-space change that adds ZERO components, so the run loop's `n_added > 0` epoch trigger
alone would miss it (removal-consumption design §2).

The growth branches (:gw_explore, :gw_add_feature) ADOPT the memoised GrowthProposal's scratch
(`growth_cache`/`cache_epoch` — the same cache the score seam filled, ratified §5 Q1): the
belief the score priced becomes, by identity, the belief the agent holds. Score-blind callers
(no prior score, cache miss) compute the same proposal at execute time — the Q1 fallback. The
apply floor is `prop.yield_nats > 0` (strictly weaker than any positive net score, so a
chosen op always applies; a blind fire on an evidence-free union is a structural no-op).
"""
function execute_gw_meta_action!(
    state::AgentState,
    action::Symbol,
    explore_buffer::Vector{ExploreObservation};
    include_temporal::Bool=false,
    verbose::Bool=false,
    growth_cache::Union{Nothing, Dict{Tuple{Symbol, Int, Int, Int, Int}, Union{Nothing, GrowthProposal}}}=nothing,
    cache_epoch::Int=0
)::NamedTuple{(:n_added, :applied_replacement), Tuple{Int, Bool}}
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
        return (n_added = n_added, applied_replacement = false)

    elseif action == :gw_perturb_grammar
        # Replacement semantics (removal-consumption design §2): consume the top-k grammar whose
        # best replacement carries the greatest realised Δ log-evidence. The gid argmax uses the
        # cost-free values (the declared price is a constant across gids, so the argmax is
        # unchanged); the strict > 0 floor makes a blind fire (score-blind baselines) a no-op
        # when nothing is dead. The applied candidate is `best_replacement`'s — the SAME candidate
        # the score priced (T-3.55). Replacement adds no components (re-description, not
        # exploration — T-3.52); pure prior re-description still jolts the predictive stream the
        # residual regime models, so the regime resets (the Q1b caused-change-point rationale).
        best_gid = 0
        best_v = 0.0
        for gid in top_k_grammar_ids(state, 3)
            haskey(state.grammars, gid) || continue
            v = replacement_value(state, gid)
            v > best_v && (best_v = v; best_gid = gid)
        end
        best_gid == 0 && return (n_added = 0, applied_replacement = false)  # nothing dead → no-op
        cand = best_replacement(state, best_gid)
        new_g = replace_grammar_in_state!(state, cand)
        reset_learning_regime!(state)
        verbose && println("  [Meta: perturb_grammar → grammar $(best_gid)→$(new_g.id) consumed " *
                           "($(cand.kind) $(cand.kind === :remove_rule ? cand.payload.name : cand.payload), " *
                           "Δ = $(round(log(2.0) * cand.payoff_symbols, digits=4)) nats)]")
        return (n_added = 0, applied_replacement = true)

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
        return (n_added = n_added, applied_replacement = false)

    elseif action == :gw_explore || action == :gw_add_feature
        # The growth transition (winners-curse §1, FIRE): adopt the virtual injection's scratch —
        # the union of ALL candidates' deduped programs, coherently conditioned on the window.
        # No winner is installed (install-the-argmax was an argmax_m P(m|D) collapse,
        # average-not-collapse applied to the transition): junk candidates ride at the mass the
        # window granted them and the existing hygiene (sync_prune!/sync_truncate!, #193's
        # replacement consumption) self-heals. Resets the residual REGIME — an alphabet
        # expansion, so pre-change regime residuals are stale (Q1b; perturb/deepen/enumerate are
        # within-alphabet and do NOT reset). The BUFFER is retained: raw records are world data,
        # alphabet-independent (coherent-injection-design.md §1, the Q2b amendment).
        op = action == :gw_explore ? :explore : :add_feature
        prop = gw_growth_proposal(op, state, explore_buffer, growth_cache, cache_epoch;
                                  include_temporal = include_temporal)
        (prop === nothing || prop.yield_nats <= 0.0) &&
            return (n_added = 0, applied_replacement = false)   # no evidence-bearing union → no-op
        adopt!(state, prop.scratch)
        reset_learning_regime!(state)
        verbose && println("  [Meta: $(op) → union adopted, +$(prop.n_added) programs " *
                           "(yield $(round(prop.yield_nats, digits=3)) nats, " *
                           "P_newcomers $(round(prop.p_newcomers, digits=4)))]")
        return (n_added = prop.n_added, applied_replacement = false)
    end
    (n_added = 0, applied_replacement = false)
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
    # Memoisation of the growth PROPOSALS (see gw_growth_proposal): the score prices the
    # cached scratch and the fire adopts it (T-3.55 as identity). Unlike the retired pure-fit
    # cache, a proposal depends on the LIVE BELIEF, so the epoch bumps on EVERY hypothesis-
    # space change (any injection, deepen, replacement — wherever space_epoch bumps) and on
    # window trims — a stale scratch must never be priced or adopted (winners-curse §6 risk 3;
    # conditioning between steps is covered by n_buf in the key + the trim bump). The op lives
    # in the key as a Symbol (a typed field, not an arithmetic offset). The score is applied
    # outside the cache through net_value against the declared price.
    growth_cache = Dict{Tuple{Symbol, Int, Int, Int, Int}, Union{Nothing, GrowthProposal}}()
    cache_epoch = 0

    # The learned returns-to-growth belief (belief-derived-valuation §2b) + its bookkeeping DATA:
    # space_epoch counts hypothesis-space changes (any injection, any depth change);
    # last_fire_epoch records the epoch each escape op last fired under — the (op,
    # changed-since-last-fire) context bit. (The declared-horizon event counter retired with
    # the growth seam's ×H — winners-curse §8.4; nothing consumed it.)
    growth_returns = GrowthReturns(Symbol[:gw_enumerate_more, :gw_deepen])
    space_epoch = 0
    last_fire_epoch = Dict{Symbol, Int}()

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
                changed = Dict{Symbol, Bool}(
                    op => get(last_fire_epoch, op, -1) != space_epoch
                    for op in (:gw_enumerate_more, :gw_deepen))
                scored = score_blind(meta_policy) ?
                    Dict{Symbol, Float64}(:gw_do_nothing => 0.0) :
                    score_gw_meta_actions(state, explore_buffer, growth_returns, changed;
                                          op_compute_cost=op_compute_cost,
                                          growth_cache=growth_cache, cache_epoch=cache_epoch,
                                          include_temporal=include_temporal)
                chosen = meta_policy(scored, step)::Symbol
                chosen == :gw_do_nothing && break

                meta_result = execute_gw_meta_action!(state, chosen, explore_buffer;
                    include_temporal=include_temporal, verbose=verbose,
                    growth_cache=growth_cache, cache_epoch=cache_epoch)
                n_added_meta = meta_result.n_added
                meta_actions_taken += 1

                # The realised yield is an OBSERVATION (belief-derived-valuation §2b): measured
                # BEFORE prune/truncate (they may drop the very components), conditioned into the
                # returns belief at the context the op fired under. The op's own effect then
                # bumps the space epoch — other ops see a changed space; the op itself does not
                # (its post-fire epoch is recorded post-bump). An applied replacement changes the
                # space while adding ZERO components (re-description), so it carries its own
                # trigger (removal-consumption design §2).
                if chosen in (:gw_enumerate_more, :gw_deepen)
                    y = injection_yield_nats(state, n_added_meta)
                    observe_yield!(growth_returns, chosen, changed[chosen], y)
                end
                # Any hypothesis-space change stales both the (op, changed) context bits AND
                # every cached growth proposal (its scratch was copied from the pre-change
                # belief) — the two epochs advance together.
                (n_added_meta > 0 || chosen == :gw_deepen ||
                 meta_result.applied_replacement) && (space_epoch += 1; cache_epoch += 1)
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
