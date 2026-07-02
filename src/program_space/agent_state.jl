"""
    agent_state.jl — AgentState and parallel-array management

Bundles the MixturePrevision belief with its parallel arrays (metadata,
compiled_kernels, all_programs). sync_prune!/sync_truncate! keep them
in lock-step and reindex TaggedBetaPrevision tags.
"""

using .Ontology

# ═══════════════════════════════════════
# AgentState — bundles belief with parallel arrays
# ═══════════════════════════════════════

mutable struct AgentState
    belief::Ontology.MixturePrevision
    metadata::Vector{Tuple{Int, Int}}       # (grammar_id, program_id)
    compiled_kernels::Vector{CompiledKernel}
    all_programs::Vector{Program}
    grammars::Dict{Int, Grammar}            # grammar_id → Grammar
    current_max_depth::Int                  # current enumeration depth
    # Belief-aware saturation signal (exploration-budget Move 2). The residual-plateau regime belief is
    # a 2-regime BMA — a Measure (state-is-measure), the residual history summarised; `last_residual` is
    # the previous step's predictive log-loss, a sufficient statistic for the decrement (one scalar
    # suffices BECAUSE the history lives in the regime posterior). Both reset on grammar change
    # (reset_learning_regime!). Move 2 is signal-only — nothing reads these for a decision until Move 3.
    learning_regime::Ontology.MixturePrevision
    last_residual::Union{Nothing, Float64}
end

# Backward-compatible 6-arg constructor: defaults the Move-2 saturation fields (uninformative regime, no
# residual yet) and forwards to the 8-arg auto-constructor. Every existing AgentState(...) call site
# (hosts, skin, persistence, tests) uses this and is unaffected.
AgentState(belief, metadata, compiled_kernels, all_programs, grammars, current_max_depth) =
    AgentState(belief, metadata, compiled_kernels, all_programs, grammars, current_max_depth,
               initial_learning_regime(), nothing)

"""
    reset_learning_regime!(state) → state

Reset the residual-plateau regime belief to its uninformative prior AND clear `last_residual`
(exploration-budget Move 2, Q1b). Call on every grammar change — pre-change residuals were generated
under a superseded alphabet and are stale; carrying them would drag the fresh inference. Starting the
residual Measure afresh (not merely re-weighting toward :improving) is the principled response to a
*caused* change-point (no BOCPD inference needed — the agent knows it changed the alphabet).
"""
function reset_learning_regime!(state::AgentState)
    state.learning_regime = initial_learning_regime()
    state.last_residual = nothing
    state
end

"""
    sync_prune!(state; threshold) → state

Prune negligible components AND the parallel arrays together.
Reindex TaggedBetaPrevision tags so that tag == array position.
"""
function sync_prune!(state::AgentState; threshold::Float64=-30.0)
    max_lw = maximum(state.belief.log_weights)
    keep = [i for i in eachindex(state.belief.log_weights)
            if state.belief.log_weights[i] - max_lw > threshold]
    length(keep) == length(state.belief.components) && return state
    new_comps = [Ontology.TaggedBetaPrevision(j, state.belief.components[k].beta)
                 for (j, k) in enumerate(keep)]
    state.belief = Ontology.MixturePrevision(new_comps, state.belief.log_weights[keep])
    state.metadata = state.metadata[keep]
    state.compiled_kernels = state.compiled_kernels[keep]
    state.all_programs = state.all_programs[keep]
    state
end

"""
    sync_truncate!(state; max_components) → state

Keep only the top-weighted components. Reindex tags.
"""
function sync_truncate!(state::AgentState; max_components::Int=2000)
    length(state.belief.components) <= max_components && return state
    perm = sortperm(state.belief.log_weights, rev=true)
    keep = perm[1:min(max_components, length(perm))]
    new_comps = [Ontology.TaggedBetaPrevision(j, state.belief.components[k].beta)
                 for (j, k) in enumerate(keep)]
    state.belief = Ontology.MixturePrevision(new_comps, state.belief.log_weights[keep])
    state.metadata = state.metadata[keep]
    state.compiled_kernels = state.compiled_kernels[keep]
    state.all_programs = state.all_programs[keep]
    state
end

# ═══════════════════════════════════════
# Grammar weight aggregation
# ═══════════════════════════════════════

"""Aggregate per-component weights into grammar-level weights."""
function aggregate_grammar_weights(component_weights::Vector{Float64},
                                    metadata::Vector{Tuple{Int, Int}})::Dict{Int, Float64}
    gw = Dict{Int, Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        gw[gi] = get(gw, gi, 0.0) + component_weights[i]
    end
    gw
end

"""Return top-k grammar IDs by aggregated posterior weight."""
function top_k_grammar_ids(state::AgentState, k::Int)::Vector{Int}
    w = Ontology.weights(state.belief)
    gw = aggregate_grammar_weights(w, state.metadata)
    sorted = sort(collect(keys(gw)), by=gi -> -get(gw, gi, 0.0))
    sorted[1:min(k, length(sorted))]
end

# ═══════════════════════════════════════
# Add programs to state (with deduplication)
# ═══════════════════════════════════════

"""
    add_programs_to_state!(state, grammar, max_depth; observations, ...) → Int

Enumerate programs from `grammar` at `max_depth`, compile kernels, and
append to all parallel arrays. Deduplicates: skips programs whose
(grammar_id, expr) already exists in state.all_programs — injecting
fresh Beta(1,1) for already-observed hypotheses disrupts the posterior.

`observations` (REQUIRED — no default) is the retained evidence window: every
observation the live belief conditioned on since window start, in order, with each
record's `residual` = −log_predictive at its conditioning time. Newcomers are injected
COHERENTLY: assembled as a newcomers-only mixture, prequentially conditioned on the
window through Tier-1 `condition` (the one learning mechanism), then aligned to the
incumbents' scale so each arrives at

    log-weight = complexity prior + Σₜ pred_llₜ + Σₜ residualₜ      (shared offset with incumbents)

`MixturePrevision`'s constructor normalises on every construction, so two ledgers restore
the cross-group constant that normalisation discards: `Z_new + Σ log_predictive` during
the replay de-normalises the replay, and `Σ residual` (the incumbents' recorded surprises
— the same normalisers their live trajectory absorbed) re-applies the incumbents' shift.
The result is bit-identical to having injected the newcomers at window start and
conditioned jointly: hypothesis addition commutes with conditioning, asserted `==` by
test_coherent_injection.jl §1 (host-side it is exact up to `sync_prune!`/`sync_truncate!`
mass drops, ≤ e⁻³⁰ relative per prune). The kwarg has no default so a call site can never
silently inject ignorant components again — an empty window is an explicit declaration
(honest at t=0), not an omission. See docs/exploration-budget/coherent-injection-design.md.

Returns count of programs added.
"""
function add_programs_to_state!(
    state::AgentState,
    grammar::Grammar,
    max_depth::Int;
    observations::Vector{ExploreObservation},
    action_space::Vector{Symbol}=Symbol[:classify],
    min_log_prior::Float64=-20.0,
    include_temporal::Bool=false
)::Int
    programs = enumerate_programs(grammar, max_depth;
                                   action_space=action_space,
                                   min_log_prior=min_log_prior,
                                   include_temporal=include_temporal)

    # Build set of existing expressions for this grammar (for dedup)
    existing_exprs = [state.all_programs[i].expr
                      for i in eachindex(state.all_programs)
                      if state.metadata[i][1] == grammar.id]

    n_added = 0
    new_components = TaggedBetaPrevision[]
    new_lw = Float64[]
    new_meta = Tuple{Int, Int}[]
    new_ck = CompiledKernel[]
    new_progs = Program[]

    for (pi, p) in enumerate(programs)
        # Skip if this expression already exists for this grammar
        any(e -> expr_equal(e, p.expr), existing_exprs) && continue

        # Tags are LOCAL (1..n) during the replay below — program_space_observation_kernel
        # dispatches by tag into new_ck — then re-tagged to global positions on append
        # (the sync_prune! re-tag discipline).
        n_added += 1
        push!(new_components, Ontology.TaggedBetaPrevision(
            n_added, Ontology.BetaPrevision(1.0, 1.0)))
        # Program node-count prior (two-part MDL): the SPEC §1.3 complexity log-prior
        # (`complexity.jl`), λ = log(2). Bit-identical to the old literal (test_complexity.jl).
        lw = complexity_logprior(grammar.complexity; λ = log(2)) +
             complexity_logprior(p.complexity; λ = log(2))
        push!(new_lw, lw)
        push!(new_meta, (grammar.id, pi))
        push!(new_ck, compile_kernel(p, grammar, pi))
        push!(new_progs, p)
    end

    if !isempty(new_components)
        # Coherent injection: replay the evidence window through the newcomers-only mixture
        # via Tier-1 condition, then restore the cross-group scale (docstring above; the
        # constructor normalises, so the replay's and the incumbents' normalisation
        # constants are re-applied via the two ledgers).
        if !isempty(observations)
            # De-normalisation ledger: the newcomers' prior normaliser + every replay-step
            # predictive (accumulated BEFORE each condition, prequentially).
            denorm = Ontology.logsumexp(new_lw)
            nm = Ontology.MixturePrevision(Prevision[new_components...], new_lw)
            for obs in observations
                k = program_space_observation_kernel(new_ck, obs.features,
                                                     obs.temporal_state, obs.correct_actions)
                denorm += Ontology.log_predictive(nm, k, 1.0)
                nm = Ontology.condition(nm, k, 1.0)
            end
            # Incumbent ledger: the surprises the live trajectory recorded are exactly the
            # normalisers its weights absorbed over the same window.
            ledger = sum(obs.residual for obs in observations)
            new_components = TaggedBetaPrevision[c for c in nm.components]
            new_lw = nm.log_weights .+ (denorm + ledger)
        end
        offset = length(state.compiled_kernels)
        retagged = TaggedBetaPrevision[Ontology.TaggedBetaPrevision(offset + i, c.beta)
                                       for (i, c) in enumerate(new_components)]
        all_comps = Prevision[state.belief.components..., retagged...]
        all_lw = Float64[state.belief.log_weights..., new_lw...]
        state.belief = Ontology.MixturePrevision(all_comps, all_lw)
        append!(state.metadata, new_meta)
        append!(state.compiled_kernels, new_ck)
        append!(state.all_programs, new_progs)
    end
    n_added
end

# ═══════════════════════════════════════
# Grammar ID counter
# ═══════════════════════════════════════

let grammar_counter = Ref(0)
    global function next_grammar_id()
        grammar_counter[] += 1
        grammar_counter[]
    end
    global function reset_grammar_counter!()
        grammar_counter[] = 0
    end
end
