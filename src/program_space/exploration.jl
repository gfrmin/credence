"""
    exploration.jl — the belief-aware exploration budget: compute-budgeted lookahead VOI for
    threshold refinement (exploration-budget Move 3).

The belief-aware sibling of prior-only `perturbation.jl`. Where `perturb_grammar` prices the COMPRESSION
class depth-one in prior nats (`net_voc`), `explore_grammar` prices the GENERATIVE-CHANGE class — here,
threshold refinement — by **lookahead against the belief's predictive residual** (master plan §2): a
threshold is complexity-invariant (zero prior signal, all fit), so the only honest valuation is to
*actually* refine the grid, re-enumerate, re-condition the buffer, and measure the realised marginal
log-loss reduction `Δℓ` (Q3a — predictive nats). The fineness-Occam is carried by that marginal
likelihood, NOT the prior (SPEC §1.3 margin; Q1(b)≡Q3(a)) — a noise-fitting split does not improve the
predictive, so refinement self-limits.

Architecture (code-time refinement of the ratified `explore_grammar(belief, g, observations)` signature):
self-contained in `src/` so the mll accumulation (probability arithmetic feeding the explore decision)
stays out of `apps/` (Invariant 1, spatial). Belief-awareness flows through the buffer's per-observation
`residual` (where the belief mispredicts — the screen ORDER, Q2a) and the counterfactual replay (mll under
each candidate grammar), not a live belief argument — the lookahead reconstructs beliefs per grammar.
"""

using .Ontology

# ═══════════════════════════════════════
# The observation buffer — the host's record of evidence under the current alphabet
# ═══════════════════════════════════════

"""
    ExploreObservation(features, temporal_state, correct_actions, residual)

One conditioning event in the explore buffer (host-side DATA, not belief — brain/body split, Q2b). Carries
exactly what the lookahead needs:
- `features` — the feature dict, for (a) proposing threshold candidates at observed values and (b) replay.
- `temporal_state` — temporal context, threaded to the compiled kernels during replay.
- `correct_actions` — the outcome(s) that count as "correct" for this obs (a `Set` generalises the
  single-outcome hosts — `Set([true_type])` — and email_agent's multi-action step). Defines the
  Beta-update direction per component during the replay's conditioning.
- `residual` — `−log predictive` of this obs under the belief at conditioning time (the host's already-
  computed `surprise`). Used to ORDER candidate evaluation (descending residual mass), not as a gate — the
  residual ranks where to look first, no cutoff (Q2a).

The buffer is `all-history-since-reset` (Q2b): the full evidence under the current alphabet, cleared by
`reset_learning_regime!` on every grammar change (Move 2 Q1b).
"""
struct ExploreObservation
    features::Dict{Symbol, Float64}
    temporal_state::Dict{Symbol, Any}
    correct_actions::Set{Symbol}
    residual::Float64
end

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

# ═══════════════════════════════════════
# The generic program-space observation kernel (lifted from the hosts; Invariant 1 — replay arithmetic
# must live in src/, so the kernel the replay conditions through lives here too)
# ═══════════════════════════════════════

"""
    program_space_observation_kernel(compiled_kernels, features, temporal_state, correct_actions) → Kernel

The standard program-space BetaBernoulli per-component conditioning kernel: each component (a
`TaggedBetaPrevision` tagged by program index) evaluates its compiled kernel on `features` → a recommended
action; the recommendation is "correct" iff it is in `correct_actions`; the per-component Beta updates
toward correct/incorrect. `correct_actions::Set` generalises the single-outcome hosts (`Set([true_type])`)
and email_agent's multi-action step (the set of still-needed actions). The `correct_cache` in `params` is
read by the `BetaBernoulli` condition dispatch for the per-component update direction.

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
# The lookahead — counterfactual marginal log-loss of the buffer under a grammar
# ═══════════════════════════════════════

"""
    _grammar_marginal_log_loss(g, observations, max_depth, action_space) → Float64

The prequential marginal log-loss of the buffer under grammar `g`: enumerate `g`'s programs at
`max_depth`, build the fresh complexity-prior belief the host would (Beta(1,1) per program, two-part-MDL
log-weights), and replay the buffer — accumulating `−log_predictive` and conditioning each step through
the Tier-1 mixture `condition`. This is `Σ_t −log P(obs_t | obs_<t, g)`, the evidence for `g` on the
buffer; the marginal likelihood's parameter integration IS the Bayesian Occam that makes leaving the
prior fineness-blind sound (Q1(b)≡Q3(a)). All arithmetic is canalised through `log_predictive`/`condition`
(Invariant 1). A grammar that enumerates nothing returns `Inf` (it predicts nothing).
"""
function _grammar_marginal_log_loss(g::Grammar, observations::Vector{ExploreObservation},
                                    max_depth::Int, action_space::Vector{Symbol})::Float64
    progs = enumerate_programs(g, max_depth; action_space = action_space)
    isempty(progs) && return Inf
    cks = CompiledKernel[compile_kernel(p, g, i) for (i, p) in enumerate(progs)]
    components = TaggedBetaPrevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0))
                                     for i in eachindex(progs)]
    logw = Float64[complexity_logprior(g.complexity; λ = log(2)) +
                   complexity_logprior(p.complexity; λ = log(2)) for p in progs]
    belief = MixturePrevision(components, logw)

    mll = 0.0
    for obs in observations
        k = program_space_observation_kernel(cks, obs.features, obs.temporal_state, obs.correct_actions)
        mll += -log_predictive(belief, k, 1.0)
        belief = condition(belief, k, 1.0)
    end
    mll
end

"""
    _candidate_residual_mass(feat, t, observations) → Float64

The residual screen ORDER (Q2a): the summed predictive residual of the observations bracketing the split
`t` on feature `feat` (those at the nearest observed value below and above `t`). Higher mass ⇒ the split
sits where the current belief mispredicts most ⇒ evaluate first. This ORDERS evaluation; it never gates
(no cutoff). Because `explore_grammar` then evaluates the full finite candidate set and takes the argmax,
the order is the screen and the argmax is the decision — no positive-VOI candidate is skipped (Q3b,
one-sided). The order becomes load-bearing only if a future move adds budget-limited early termination.
"""
function _candidate_residual_mass(feat::Symbol, t::Float64,
                                  observations::Vector{ExploreObservation})::Float64
    below = -Inf
    above = Inf
    for obs in observations
        haskey(obs.features, feat) || continue
        v = obs.features[feat]
        v < t && v > below && (below = v)
        v > t && v < above && (above = v)
    end
    mass = 0.0
    for obs in observations
        haskey(obs.features, feat) || continue
        v = obs.features[feat]
        (v == below || v == above) && (mass += obs.residual)
    end
    mass
end

# ═══════════════════════════════════════
# explore_grammar — the belief-aware threshold-refinement meta-action (Q2-master: forced separate entry)
# ═══════════════════════════════════════

"""
    explore_grammar(g, observations, max_depth; action_space, compute_cost = 0.0) → Grammar

Refine `g`'s threshold grid by the single candidate whose compute-budgeted lookahead VOI is greatest,
applied iff that VOI clears `compute_cost`; otherwise a structural no-op (the input `g` unchanged). The
belief-aware sibling of prior-only `perturb_grammar`: the value of a threshold is invisible to depth-one
prior `net_voc` (complexity-invariant — all fit, no prior signal), so this *actually* refines, re-enumerates,
re-conditions the buffer, and measures the realised marginal-log-loss reduction.

Mechanism (Q2a/Q3a/Q3b):
- Candidates = midpoints between adjacent observed values (`_threshold_candidates`) — the complete finite
  set, no proposer (master plan §3.1).
- Evaluated in descending residual-mass order (the screen ORDERS; it never gates).
- VOI of a candidate = `net_value(Δℓ, compute_cost)` where `Δℓ = mll(buffer|g) − mll(buffer|g')` is the
  predictive-log-loss reduction (the SAME nats the saturation residual is measured in). `compute_cost`
  prices the lookahead spend; raising it suppresses refinement smoothly (no cliff, no hard cap — Q3b).
- The full finite candidate set is evaluated and the argmax taken (one-sided: no positive-VOI candidate is
  skipped — Q3b's provable form, not a heuristic early-stop). Returns `g` unchanged when none clears.

One refinement per call (like `perturb_grammar` adds one rule): the host applies it, resets the residual
regime (the alphabet changed — Move 2 Q1b), accrues more data, and may explore again.
"""
function explore_grammar(g::Grammar, observations::Vector{ExploreObservation}, max_depth::Int;
                         action_space::Vector{Symbol} = Symbol[:classify],
                         compute_cost::Float64 = 0.0)::Grammar
    isempty(observations) && return g
    candidates = _threshold_candidates(g, observations)
    isempty(candidates) && return g

    masses = Float64[_candidate_residual_mass(feat, t, observations) for (feat, t) in candidates]
    order = sortperm(masses, rev = true)   # stable: residual order, deterministic tie-break

    baseline = _grammar_marginal_log_loss(g, observations, max_depth, action_space)

    best_voi = 0.0   # a candidate must clear net VOI > 0 to win; else no-op
    best = g
    for idx in order
        feat, t = candidates[idx]
        g_cand = _refine_grammar(g, feat, t)
        mll = _grammar_marginal_log_loss(g_cand, observations, max_depth, action_space)
        voi = net_value(baseline - mll, compute_cost)   # Δℓ − compute_cost, through the stdlib functional
        if voi > best_voi
            best_voi = voi
            best = g_cand
        end
    end
    best
end
