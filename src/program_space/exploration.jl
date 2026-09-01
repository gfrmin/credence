"""
    exploration.jl — the belief-aware exploration budget: the growth lookahead as a virtual
    injection (exploration-budget Moves 3–4, re-founded by the winners-curse design rev 3).

The belief-aware sibling of prior-only `perturbation.jl`. Where `perturb_grammar` prices the
COMPRESSION class depth-one in prior nats (`net_voc`), the growth classes here — threshold
refinement and feature discovery — are priced by **virtually performing the transition**: copy
the state, coherently inject ALL candidates' deduped programs at their two-part complexity
priors, condition on the evidence window (the coherent-injection code path, verbatim), and
read the score off the scratch through an existing canalised op: `injection_yield_nats` —
the union-over-incumbent window Bayes factor under the complexity prior (design §8.2), netted
against the declared compute price. Firing ADOPTS the scratch (score ≡ transition, T-3.55,
as an identity).
Selection, extrapolation and collapse are priced by the mixture itself — candidates are
CARRIED at the mass the window likelihood grants them, never compared by an argmax
(average-not-collapse, applied to the transition; see the design doc §1 for the three
pathologies of the retired per-candidate marginal-log-loss argmax).

Architecture: self-contained in `src/` so the replay and yield arithmetic (probability
arithmetic feeding the growth decision) stays out of `apps/` (Invariant 1, spatial).
"""

using .Ontology

# ═══════════════════════════════════════
# The observation buffer — the host's record of evidence
# ═══════════════════════════════════════
# `ExploreObservation` itself lives in types.jl (declared data; agent_state.jl's coherent
# injection needs it before this file loads — coherent-injection-design.md §1).

# ═══════════════════════════════════════
# Candidate generation — the complete, finite candidate set (Q2a)
# ═══════════════════════════════════════

"""
    _threshold_candidates(g, observations) → Vector{Tuple{Symbol, Float64}}

The candidate refinements: for each feature, the midpoints between adjacent OBSERVED values of that
feature that are not already grid points. A threshold cannot matter except where it crosses an
observation, so the observed values are the COMPLETE, finite candidate set — generation is exhaustive,
which is exactly why thresholds are EU-max-complete and need no heuristic proposer (master plan §3.1).
Midpoints (not the observed values themselves) so the split sits strictly between two observations.
Deterministic: features in sorted order, values sorted.
"""
function _threshold_candidates(g::Grammar,
                               observations::Vector{ExploreObservation})::Vector{Tuple{Symbol, Float64}}
    candidates = Tuple{Symbol, Float64}[]
    for feat in sort(collect(g.feature_set))
        vals = sort(unique(obs.features[feat] for obs in observations if haskey(obs.features, feat)))
        length(vals) < 2 && continue
        existing = g.thresholds[feat]
        for i in 1:(length(vals) - 1)
            mid = (vals[i] + vals[i + 1]) / 2.0
            # Exclude midpoints that coincide (within fp tolerance) with an existing grid point — the
            # existing threshold already provides that split, so the candidate is a wasteful duplicate
            # (and the VOI gate would reject it as Δℓ ≈ 0 anyway; this just keeps the candidate set tight).
            any(isapprox(mid, e; atol = 1e-9) for e in existing) && continue
            push!(candidates, (feat, mid))
        end
    end
    candidates
end

"""
    _refine_grammar(g, feature, threshold) → Grammar

The candidate grammar: `g` with `threshold` inserted into `feature`'s grid (sorted, deduplicated), a
fresh id. Complexity is recomputed identically — threshold-count-invariant (Q1(b)). All other features'
grids are shared by reference (unchanged); only `feature`'s grid is a fresh sorted vector.
"""
function _refine_grammar(g::Grammar, feature::Symbol, threshold::Float64)::Grammar
    new_grid = sort(unique(vcat(g.thresholds[feature], threshold)))
    refined = Dict{Symbol, Vector{Float64}}(f => (f == feature ? new_grid : g.thresholds[f])
                                            for f in keys(g.thresholds))
    Grammar(g.feature_set, g.rules, refined, next_grammar_id())
end

"""
    _feature_candidates(g, available_features) → Vector{Symbol}

The feature-discovery candidate set: the host-furnished features `g` does NOT yet use,
`available_features \\ g.feature_set`, in sorted (deterministic) order. Move 4's headline realisation —
base-feature discovery is pure EU-max **selection**, NOT construction: the host already extracts the full
feature superset every step (its value is in every observation's `features` Dict), so a candidate feature's
predicates work immediately on adding it. The host *provides* the candidates (`available_features`); the
lookahead *ranks* them. The fully-closed selection half of the master plan's §3.1 selection/generation seam
— no proposer, no AST extension (those are the deferred construction frontier: arithmetic feature products).
"""
function _feature_candidates(g::Grammar, available_features::Set{Symbol})::Vector{Symbol}
    sort(collect(setdiff(available_features, g.feature_set)))
end

"""
    _add_feature(g, feature) → Grammar

The candidate grammar: `g` with `feature` added to `feature_set` and seeded with the default threshold
grid, a fresh id. **Complexity RISES by 1** — unlike a threshold refinement (`_refine_grammar`, which is
complexity-invariant per Q1(b)), a feature is a genuine description-length unit: `compute_grammar_complexity`
counts `length(feature_set)`, so the new symbol costs one bit of prior. This is Q2, and the exact converse
of Move 3's Q1(b): a finer grid adds no symbol (fineness-Occam rides the marginal likelihood), a new feature
adds a symbol (the Occam rides the prior). All existing features' grids are preserved by reference (a refined
grid SURVIVES the add — the Move-3 review lesson); only `feature` gets a fresh default grid.
"""
function _add_feature(g::Grammar, feature::Symbol)::Grammar
    new_features = union(g.feature_set, Set([feature]))
    grids = Dict{Symbol, Vector{Float64}}(f => g.thresholds[f] for f in keys(g.thresholds))
    grids[feature] = copy(THRESHOLDS)
    Grammar(new_features, g.rules, grids, next_grammar_id())
end

# ═══════════════════════════════════════
# The generic program-space observation kernel (lifted from the hosts; Invariant 1 — replay arithmetic
# must live in src/, so the kernel the replay conditions through lives here too)
# ═══════════════════════════════════════

"""
    program_space_observation_kernel(compiled_kernels, features, temporal_state, correct_actions) → Kernel

The standard program-space BetaBernoulli per-component conditioning kernel: each component (a
`TaggedBetaPrevision` tagged by program index) evaluates its compiled kernel on `features` → a recommended
action; the recommendation is "correct" iff it is in `correct_actions`. The per-program learning is
carried by the **mixture reweight** — `_predictive_ll` returns `log(p)` for a correct component and
`log(1−p)` for an incorrect one (the closure's return), so `condition(::MixturePrevision)` shifts weight
toward programs that predict the outcome. `correct_actions::Set` generalises the single-outcome hosts
(`Set([true_type])`) and email_agent's multi-action step (the set of still-needed actions). `correct_cache`
in `params` is populated as a side effect for parity with the host kernels (and a future dedup), but is
NOT read by the conditioning dispatch: `update(ConjugatePrevision{BetaPrevision, BetaBernoulli}, 1.0)`
increments α unconditionally — the discrimination is the reweight, not the per-component Beta direction.

This is the engine-level home of logic the hosts currently duplicate (grid_world/email_agent
`build_observation_kernel` + email_agent `build_step_kernel` are the same closure); the lookahead replay
(`_grammar_marginal_log_loss`) conditions through it. NOTE: the host copies can later delegate here (a DRY
follow-up) — left untouched in Move 3 to keep the live conditioning trajectories bit-stable.
"""
function program_space_observation_kernel(
    compiled_kernels::Vector{CompiledKernel},
    features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any},
    correct_actions::Set{Symbol}
)
    recommendation_cache = Dict{Int, Symbol}()
    correct_cache = Dict{Int, Bool}()
    Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        _ -> error("generate not used in condition"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaPrevision
                tag = m_or_θ.tag
                recommended = get!(recommendation_cache, tag) do
                    compiled_kernels[tag].evaluate(features, temporal_state)
                end
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

# ═══════════════════════════════════════
# The lookahead as a virtual injection (winners-curse design §1)
#
# A growth op is scored by VIRTUALLY PERFORMING its transition: copy the state, coherently
# inject ALL candidates' deduped programs at their two-part complexity priors, condition on
# the window (the coherent-injection code path, verbatim, via the union method of
# add_programs_to_state!), and read the score off the scratch through existing canalised ops.
# Firing ADOPTS the scratch (adopt!), so score ≡ transition holds as an identity (T-3.55).
#
# What died here, and by whose hand (design §1): the per-candidate marginal-log-loss argmax
# (_grammar_marginal_log_loss / _best_threshold_refinement / _best_feature_addition) — the
# max-over-K was an order statistic over chance-fitting candidates (selection unpriced), its
# window rate was treated as a known future rate (extrapolation unlicensed), and installing
# the winner was an argmax_m P(m|D) collapse (average-not-collapse, applied to the transition).
# All three pathologies are priced by the mixture itself: candidates are CARRIED, not compared.
# ═══════════════════════════════════════

"""
    GrowthProposal — one virtual injection, projected by score and transition (Invariant 3)

The declared result of a growth lookahead. One computation produces it; the score seam reads
`yield_nats` (through `net_value` against the declared compute price — design §8.2), the op
log reads `(yield_nats, p_newcomers)` (the per-fire mechanism pair), and the transition adopts
`scratch`. No projection can drift from another because there is only one object.

    scratch       the union state: incumbents + every candidate's deduped programs, coherently
                  conditioned on the window (the counterfactual union-from-start agent)
    n_added       programs injected (the union size after global dedup)
    yield_nats    injection_yield_nats(scratch, n_added) — the ratified evidence-relative
                  observable = the union-over-incumbent window Bayes factor under the
                  complexity prior (design §8.2), computed exactly where the escape ops learn
                  it (T-3.53: one observable, one currency, ONE SCORE FORM, two fidelities)
    p_newcomers   the newcomers' posterior mass in the scratch (the incumbent-domination read;
                  gate mechanism claim (i))
"""
struct GrowthProposal
    scratch::AgentState
    n_added::Int
    yield_nats::Float64
    p_newcomers::Float64
end

"""
    _virtual_injection(state, candidate_gs, observations; action_space, include_temporal)
        → (scratch, n_added)

Copy the state, register the candidate grammars, and union-inject their deduped programs
coherently against the window — the growth op's transition, performed on a scratch. The same
code path the adoption keeps (add_programs_to_state!'s union method); commutation with
conditioning is inherited (test_virtual_injection.jl §5).
"""
function _virtual_injection(state::AgentState, candidate_gs::Vector{Grammar},
                            observations::Vector{ExploreObservation};
                            action_space::Vector{Symbol} = Symbol[:classify],
                            include_temporal::Bool = false)
    scratch = copy_agent_state(state)
    for g in candidate_gs
        scratch.grammars[g.id] = g
    end
    n_added = add_programs_to_state!(scratch, candidate_gs, scratch.current_max_depth;
                                     observations = observations, action_space = action_space,
                                     include_temporal = include_temporal)
    (scratch, n_added)
end

# The shared proposal core: virtually inject the candidate set, read the mechanism pair off
# the scratch. Returns `nothing` when there is nothing to propose (no candidates, or the whole
# union dedups away) — the score seam's act-now floor handles it. `yield_nats` is measured
# HERE, immediately post-injection, before any prune/truncate can drop the very components
# (growth_returns.jl discipline).
function _growth_proposal(state::AgentState, candidate_gs::Vector{Grammar},
                          observations::Vector{ExploreObservation};
                          action_space::Vector{Symbol} = Symbol[:classify],
                          include_temporal::Bool = false)::Union{Nothing, GrowthProposal}
    isempty(candidate_gs) && return nothing
    scratch, n_added = _virtual_injection(state, candidate_gs, observations;
                                          action_space = action_space,
                                          include_temporal = include_temporal)
    n_added == 0 && return nothing
    n = length(state.belief.components) + n_added
    p_new = probability(scratch.belief,
                        TagSet(Interval(0.0, 1.0), Set((n - n_added + 1):n)))
    GrowthProposal(scratch, n_added, injection_yield_nats(scratch, n_added), p_new)
end

"""
    threshold_growth_proposal(state, g, observations; action_space, include_temporal)
        → Union{Nothing, GrowthProposal}

The threshold-refinement class under the virtual injection: candidates are ALL midpoint
refinements of `g` against the observed values (`_threshold_candidates` — the complete finite
set, ratified §5 Q3: inject all; a mass-based top-m pre-screen is a T-3.53 priced-fidelity
decision to be taken only if measured wall-clock demands it, and logged). Returns `nothing`
when the buffer is empty or no candidate exists.
"""
function threshold_growth_proposal(state::AgentState, g::Grammar,
                                   observations::Vector{ExploreObservation};
                                   action_space::Vector{Symbol} = Symbol[:classify],
                                   include_temporal::Bool = false)::Union{Nothing, GrowthProposal}
    isempty(observations) && return nothing
    cands = _threshold_candidates(g, observations)
    _growth_proposal(state, Grammar[_refine_grammar(g, feat, t) for (feat, t) in cands],
                     observations;
                     action_space = action_space, include_temporal = include_temporal)
end

"""
    feature_growth_proposal(state, g, observations, available_features; action_space,
                            include_temporal) → Union{Nothing, GrowthProposal}

The feature-discovery class under the virtual injection: candidates are ALL host-furnished
features `g` does not yet use (`_feature_candidates`), each as a feature-added grammar. The
one-time Occam charge (−log 2 per feature symbol) is carried by each newcomer's own complexity
prior INSIDE the mixture — there is no explicit prior term at the score seam (ratified §5 Q4:
charging it there again would double-count; test_virtual_injection.jl §1 pins the identity).
"""
function feature_growth_proposal(state::AgentState, g::Grammar,
                                 observations::Vector{ExploreObservation},
                                 available_features::Set{Symbol};
                                 action_space::Vector{Symbol} = Symbol[:classify],
                                 include_temporal::Bool = false)::Union{Nothing, GrowthProposal}
    isempty(observations) && return nothing
    _growth_proposal(state,
                     Grammar[_add_feature(g, f) for f in _feature_candidates(g, available_features)],
                     observations;
                     action_space = action_space, include_temporal = include_temporal)
end
