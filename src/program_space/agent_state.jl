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
# Replacement semantics — removal consumption
# (docs/exploration-budget/removal-consumption-design.md; the #189 deviation-3 discharge)
#
# The state-aware half of the replacement path. The declared candidate type (`ReplacementCandidate`)
# lives in perturbation.jl beside its sibling `PerturbationCandidate`; the functions live HERE because
# they take `AgentState`, which is defined after perturbation.jl loads (a residence constraint, not a
# design choice — the design doc's §2 places them in perturbation.jl).
# ═══════════════════════════════════════

"""
    replacement_candidates(state, gid) → Vector{ReplacementCandidate}

The SOUND, group-local, live-state dead-item enumeration (removal-consumption design §2, the
Move-1 OQ-4 discharge). Reference sets are collected over **every** component of group
`G = {i : metadata[i] = (gid, ·)}` — NO weight cut of any kind — unioned with the grammar's own
rule bodies (the #174 transitive device), via the existing full-depth `collect_nonterminal_refs!` /
`collect_feature_refs!` walks. A rule/feature is a candidate iff NOTHING in the group references it.

Why not the `SubprogramFrequencyTable`'s reference sets: that walk cuts support at `w > 1e-15` — a
*value*-appropriate screen but not a soundness guarantee (post-prune live components can sit below
it: prune keeps relative-to-max mass > e⁻³⁰ while normalisation divides by the full component
count). Re-keying a component whose program references the removed item would put a program outside
its grammar's language — unsound, not merely suboptimal — so the soundness predicate is exact and
does not share a representation with the frequency estimate (Invariant 3). Group-local is also
*sharper*: a feature referenced only by another grammar's programs is dead FOR THIS grammar.
Asserted by test_replacement.jl §3 (the sub-1e-15 case that defeats the table-based check).

Feature candidates are sorted for a deterministic vector (the `_feature_removal_payoff` discipline).
"""
function replacement_candidates(state::AgentState, gid::Int)::Vector{ReplacementCandidate}
    haskey(state.grammars, gid) || return ReplacementCandidate[]
    g = state.grammars[gid]
    nt_refs = Set{Symbol}()
    feat_refs = Set{Symbol}()
    for i in eachindex(state.metadata)
        state.metadata[i][1] == gid || continue
        collect_nonterminal_refs!(nt_refs, state.all_programs[i].expr)
        collect_feature_refs!(feat_refs, state.all_programs[i].expr)
    end
    for r in g.rules
        collect_nonterminal_refs!(nt_refs, r.body)
        collect_feature_refs!(feat_refs, r.body)
    end
    cands = ReplacementCandidate[]
    for r in g.rules
        r.name in nt_refs && continue
        push!(cands, ReplacementCandidate(:remove_rule, r, 1 + expr_complexity(r.body), gid))
    end
    for f in sort(collect(g.feature_set))
        f in feat_refs && continue
        push!(cands, ReplacementCandidate(:remove_feature, f, 1, gid))
    end
    cands
end

"""
    _cleaned_grammar_parts(g, cand) → (feature_set, rules, thresholds)

The candidate's edit as grammar-constructor data — the SINGLE home of the cleaning arithmetic,
shared by `_replacement_delta` (the score's Δ) and `replace_grammar_in_state!` (the transition's
`g′`), so the priced edit and the applied edit cannot drift (T-3.55). Thresholds are threaded
through (the Move-3 grid-survival discipline): `:remove_rule` keeps every grid; `:remove_feature`
drops exactly the dead feature's grid. A fresh Dict per call (grids shared by reference, the
`_add_feature` idiom) so the successor never aliases the consumed grammar's Dict.
"""
function _cleaned_grammar_parts(g::Grammar, cand::ReplacementCandidate)
    if cand.kind === :remove_rule
        name = (cand.payload::ProductionRule).name
        (g.feature_set, [r for r in g.rules if r.name != name],
         Dict{Symbol, Vector{Float64}}(f => grid for (f, grid) in g.thresholds))
    elseif cand.kind === :remove_feature
        feat = cand.payload::Symbol
        (Set(f for f in g.feature_set if f != feat), g.rules,
         Dict{Symbol, Vector{Float64}}(f => grid for (f, grid) in g.thresholds if f != feat))
    else
        error("unknown ReplacementCandidate kind: $(cand.kind)")
    end
end

# Δ = λ·(complexity(g) − complexity(g′)) in nats, λ pinned to log 2 (SPEC §1.3) — computed from the
# two ACTUAL complexities (the same `compute_grammar_complexity` the Grammar constructor uses), never
# from the candidate's assumed payoff. `payoff_symbols == Δ/λ` by the MDL arithmetic; the score and
# the transition both route through this, so they agree bit-for-bit with the constructed `g′`.
function _replacement_delta(g::Grammar, cand::ReplacementCandidate)::Float64
    (nf, nr, _) = _cleaned_grammar_parts(g, cand)
    log(2.0) * (g.complexity - compute_grammar_complexity(nf, nr))
end

# Total order for the deterministic argmax: (1) larger reclaim (score-monotone: all of a gid's
# candidates share W_G, and the score is increasing in Δ); (2) on a tie, lexicographically smaller
# name (version-stable, the `_candidate_better` discipline). test_replacement.jl §5.
_replacement_name(c::ReplacementCandidate) =
    c.kind === :remove_rule ? string((c.payload::ProductionRule).name) : string(c.payload::Symbol)
function _replacement_better(a::ReplacementCandidate, b::ReplacementCandidate)
    a.payoff_symbols != b.payoff_symbols && return a.payoff_symbols > b.payoff_symbols
    return _replacement_name(a) < _replacement_name(b)
end

"""
    best_replacement(state, gid) → Union{ReplacementCandidate, Nothing}

The deterministic argmax over `replacement_candidates(state, gid)` (see `_replacement_better`), or
`nothing` when the grammar has no dead item — the consumed-grammar/no-op signal. Both the score
(`replacement_value`) and the host's transition route through this, so the scored candidate IS the
applied candidate (T-3.55; test_replacement.jl §5 pins `===` across calls).
"""
function best_replacement(state::AgentState, gid::Int)::Union{ReplacementCandidate, Nothing}
    best = nothing
    for c in replacement_candidates(state, gid)
        (best === nothing || _replacement_better(c, best)) && (best = c)
    end
    best
end

"""
    replacement_value(state, gid; compute_cost = 0.0) → Float64

The realised Δ log-evidence of firing the best replacement, against the current posterior
(removal-consumption design §1):

    net_value(log1p((e^Δ − 1)·W_G), compute_cost),    W_G = posterior mass of group G

with `W_G` read canalised as `probability(state.belief, TagSet(group tags))` (the Prevision-level
mass read). Exactly the log-evidence the transition realises (test_replacement.jl §4 measures the
equality); the prior-only `net_voc` surrogate is this score's `W_G → 1` limit and overstates by the
mass the group does not hold — the exact form self-extinguishes on zombie lineages (`W_G → 0`).
`0.0` when the grammar has no dead item (the no-op is worth nothing; the host's act-now floor is
strict, so a 0.0 never fires).
"""
function replacement_value(state::AgentState, gid::Int; compute_cost::Float64 = 0.0)::Float64
    cand = best_replacement(state, gid)
    cand === nothing && return 0.0
    delta = _replacement_delta(state.grammars[gid], cand)
    tags = Set(i for i in eachindex(state.metadata) if state.metadata[i][1] == gid)
    mass = Ontology.probability(state.belief, Ontology.TagSet(Ontology.Interval(0.0, 1.0), tags))
    net_value(log1p((exp(delta) - 1.0) * mass), compute_cost)
end

"""
    replace_grammar_in_state!(state, cand::ReplacementCandidate) → Grammar

Apply a replacement: build `g′` from the candidate (thresholds threaded), re-key group `G`'s
metadata to `g′.id`, shift the group's log-weights by `Δ = λ·(complexity(g) − complexity(g′))`,
delete the consumed grammar from the registry, register `g′`. Tags, Beta posteriors, kernels and
programs are untouched — strictly less invasive than `sync_prune!`. Returns `g′`.

**This is the constitutional write** (`coherent-space-edit` precedent, docs/precedents.md): a
mixture log-weight write outside `condition`, legal because it is the UNIQUE map under which
replacement commutes with conditioning. Derivation (removal-consumption design §1): every live
component of `G` has a program expressible unchanged in `g′` (that is what *dead* means, under
`replacement_candidates`' sound walk) with an identical compiled kernel, hence identical likelihood
on the entire observation history; the counterfactual agent that started with `g′` in place of `g`
therefore holds, after the same history, exactly `lw′ᵢ = lwᵢ + Δ·1[i ∈ G]` — every likelihood term
cancels in the ratio and only the grammar-complexity prior term differs. Exact for the FULL history
(no window replay needed — stronger than injection's two-ledger construction, because here the
likelihoods cancel identically). The commutation equality is asserted by **test_replacement.jl §1**
(the precedent's named test). The `MixturePrevision` constructor re-normalises; the shift is
cross-group, which is the point — the group's reclaim is realised against everyone else's mass.
"""
function replace_grammar_in_state!(state::AgentState, cand::ReplacementCandidate)::Grammar
    haskey(state.grammars, cand.gid) ||
        error("replace_grammar_in_state!: grammar $(cand.gid) is not registered (stale candidate?)")
    g = state.grammars[cand.gid]
    (nf, nr, nt) = _cleaned_grammar_parts(g, cand)
    g2 = Grammar(nf, nr, nt, next_grammar_id())
    delta = log(2.0) * (g.complexity - g2.complexity)   # == _replacement_delta (same arithmetic)
    lw = copy(state.belief.log_weights)
    for i in eachindex(state.metadata)
        state.metadata[i][1] == cand.gid || continue
        state.metadata[i] = (g2.id, state.metadata[i][2])
        lw[i] += delta
    end
    state.belief = Ontology.MixturePrevision(state.belief.components, lw)
    delete!(state.grammars, cand.gid)
    state.grammars[g2.id] = g2
    g2
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

    isempty(new_components) ||
        _inject_coherently!(state, new_components, new_lw, new_meta, new_ck, new_progs,
                            observations)
    n_added
end

# The shared injection arithmetic (extracted verbatim from the single-grammar method above;
# both entry points differ only in how they COLLECT newcomers — dedup scope — never in how
# they inject). Replays the window through the newcomers-only mixture via Tier-1 condition,
# then restores the cross-group scale via the two ledgers (see the docstring above); appends
# to the parallel arrays under the re-tag discipline.
function _inject_coherently!(
    state::AgentState,
    new_components::Vector{TaggedBetaPrevision},
    new_lw::Vector{Float64},
    new_meta::Vector{Tuple{Int, Int}},
    new_ck::Vector{CompiledKernel},
    new_progs::Vector{Program},
    observations::Vector{ExploreObservation}
)
    # Coherent injection: replay the evidence window through the newcomers-only mixture
    # via Tier-1 condition, then restore the cross-group scale (the constructor normalises,
    # so the replay's and the incumbents' normalisation constants are re-applied via the
    # two ledgers).
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
    state
end

"""
    add_programs_to_state!(state, grammars::Vector{Grammar}, max_depth; observations, ...) → Int

The UNION injection (winners-curse design §1): enumerate every candidate grammar, collect the
deduped union of their genuinely-new programs, and inject them as ONE coherent injection — one
newcomers-only replay, one two-ledger scale restoration, one construction. This is the virtual
transition the growth lookahead scores and, on a fire, adopts.

Dedup scope differs from the single-grammar method, deliberately: the union dedups by
expression GLOBALLY — against every incumbent program (any grammar) and across the candidates
(first occurrence in the given order wins). Sibling candidates are all derived from one
incumbent and share most of their language; an expr-equal program is the same hypothesis (the
same compiled kernel), and injecting it once per sibling would multiply one hypothesis's prior
mass by K — exactly the double-count the mixture's multiplicity pricing forbids. The
single-grammar method keeps its per-grammar-id dedup (re-enumeration of independently DECLARED
grammars is a different situation: those duplicates are distinct BMA members by declaration).

Candidate grammars must already be registered in `state.grammars` (the yield read and the
parallel-array discipline need them). One joint injection — not K sequential calls — is
load-bearing for exactness: `MixturePrevision` normalises on construction, so sequential
injections against the same window would misalign each later group's ledger by the earlier
groups' normalisation shifts; the joint replay has no such drift. Commutation is inherited
from the single-injection theorem (test_virtual_injection.jl §5 asserts it for this method).

Returns the count of programs added (the union size after dedup).
"""
function add_programs_to_state!(
    state::AgentState,
    grammars::Vector{Grammar},
    max_depth::Int;
    observations::Vector{ExploreObservation},
    action_space::Vector{Symbol}=Symbol[:classify],
    min_log_prior::Float64=-20.0,
    include_temporal::Bool=false
)::Int
    for g in grammars
        haskey(state.grammars, g.id) ||
            error("union injection requires candidate grammar $(g.id) registered in state.grammars")
    end

    # Global dedup base: every incumbent expression, any grammar.
    existing_exprs = [p.expr for p in state.all_programs]

    n_added = 0
    new_components = TaggedBetaPrevision[]
    new_lw = Float64[]
    new_meta = Tuple{Int, Int}[]
    new_ck = CompiledKernel[]
    new_progs = Program[]

    for g in grammars
        programs = enumerate_programs(g, max_depth;
                                      action_space=action_space,
                                      min_log_prior=min_log_prior,
                                      include_temporal=include_temporal)
        for (pi, p) in enumerate(programs)
            any(e -> expr_equal(e, p.expr), existing_exprs) && continue
            any(q -> expr_equal(q.expr, p.expr), new_progs) && continue
            n_added += 1
            push!(new_components, Ontology.TaggedBetaPrevision(
                n_added, Ontology.BetaPrevision(1.0, 1.0)))
            push!(new_lw, complexity_logprior(g.complexity; λ = log(2)) +
                          complexity_logprior(p.complexity; λ = log(2)))
            push!(new_meta, (g.id, pi))
            push!(new_ck, compile_kernel(p, g, pi))
            push!(new_progs, p)
        end
    end

    isempty(new_components) ||
        _inject_coherently!(state, new_components, new_lw, new_meta, new_ck, new_progs,
                            observations)
    n_added
end

"""
    copy_agent_state(state) → AgentState

The scratch copy the virtual injection mutates: fresh parallel arrays and grammar Dict (the
mutable containers), shared immutable leaves (the belief, Programs, CompiledKernels — every
in-place mutation in this file goes through `append!`/rebinding on the containers, never
through the leaves). The live state and the scratch evolve independently after the copy.
"""
copy_agent_state(state::AgentState) =
    AgentState(state.belief, copy(state.metadata), copy(state.compiled_kernels),
               copy(state.all_programs), copy(state.grammars), state.current_max_depth,
               state.learning_regime, state.last_residual)

"""
    adopt!(state, scratch) → state

The growth-op transition (winners-curse design §1, FIRE): the scored scratch BECOMES the live
state — a field swap, zero recompute. The belief the score priced is, by identity (`===`), the
belief the agent now holds: score ≡ transition holds by construction, not by a shared candidate
function (T-3.55 as an identity; test_virtual_injection.jl §1).
"""
function adopt!(state::AgentState, scratch::AgentState)
    state.belief = scratch.belief
    state.metadata = scratch.metadata
    state.compiled_kernels = scratch.compiled_kernels
    state.all_programs = scratch.all_programs
    state.grammars = scratch.grammars
    state.current_max_depth = scratch.current_max_depth
    state.learning_regime = scratch.learning_regime
    state.last_residual = scratch.last_residual
    state
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
