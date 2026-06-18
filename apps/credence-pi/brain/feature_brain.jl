# Role: brain
"""
    feature_brain.jl — the credence-pi feature-conditioned brain (Pass 2, Move 3).

Structure-BMA over Bayesian-network edges (feature → approval). Replaces the
Pass-1 global `Beta(2,2)` with `P(approve | X)`, X = the declared features, so the
governor can block a *repeated* wasteful call while still approving a *novel* one —
impossible for one global Beta.

ROUTE B (docs/credence-pi-pass-2/move-3-design.md). This is typed Julia
brain-side code that DECLARES the structure family + readout Functionals and CALLS
the Tier-1 axiom ops — the constitution's "applications declare data and call
primitives" pattern. It reimplements NO axiom-constrained op: every belief change
goes through `condition`; every decision through the typed stdlib `optimise`/`voi`.
The `.bdsl` files keep the DECLARED DATA (feature spaces in features.bdsl, the
capability manifest, the utility constants in utility.bdsl).

Two representations, kept separate (Invariant 3):
  * `top`         — the persistent belief: a `MixturePrevision` over the 2ⁿ edge
                    structures, each a `ProductPrevision` of `TaggedBeta` cells.
                    Learning conditions THIS (`observe`).
  * `belief_at_X` — a transient per-decision view: a mixture of each structure's
                    cell-for-X carrying the same structure weights. Decisions read
                    THIS (`decide`). Equivalence lemma (design doc §"decision side"):
                    conditioning the view reweights structures identically to
                    conditioning `top`, so `voi`/`value` on the view are exact.

No raw probability arithmetic lives here — the `credence-lint` brain/ rule
enforces it. EU is closed-form via typed `LinearCombination` over `Identity`; the
action argmax is the single canonical `optimise`.
"""
module FeatureBrain

using Main.Credence: BetaPrevision, TaggedBetaPrevision, SparseStructurePrevision,
    ProductPrevision, MixturePrevision, GammaPrevision,
    Prevision, Kernel, Interval, Finite, BetaBernoulli, SoftBernoulli, Flat, FiringByTag, Identity,
    Projection, LinearCombination, TestFunction, GeometricTail, mean, FeatureDecl, condition, expect,
    cell_at, wrap_in_measure
import Main.Credence.Ontology: optimise, value, voi, net_voi, with_components,
    ProductMeasure, Measure
using JSON3

export StructureBMA, build_model, build_model_from_env, build_model_from_decls,
       build_prior, build_prior_dense, wire_brain!, context_from_features, firing_tags,
       belief_at_context, observe, observe_soft, decide, decide_multi,
       reconstruct_posterior, reconstruct_harm_posterior

# ── The model: feature spec + structure enumeration + tag bookkeeping ──
#
# A `structure` is a subset of feature indices — the parents of the approval leaf
# A (edges INTO A; feature-feature edges describe P(X) and cancel, design doc
# §"The model"). Each structure has one cell per element of the cross-product of
# its parents' value-sets; every cell carries a GLOBALLY-UNIQUE integer tag so a
# single per-context `FiringByTag` kernel routes correctly through the whole nested
# object (within each structure exactly one cell's tag fires).

struct StructureBMA
    feature_names::Vector{String}
    feature_values::Vector{Vector{String}}     # per feature, its value-set as strings
    structures::Vector{Vector{Int}}            # the 2ⁿ parent-index subsets
    cell_tag::Vector{Dict{Tuple, Int}}         # per structure: cell-key (parent values) -> tag
    tag_lo::Vector{Int}                        # per structure: lowest cell tag (contiguous)
    tag_hi::Vector{Int}                        # per structure: highest cell tag
    n_features::Int
    alpha0::Float64
    beta0::Float64
    p_edge::Float64
end

# The cell key for context X under a structure = the tuple of X's values at the
# structure's parents, in parent order. Empty parents ⇒ `()` (the global cell).
cell_key(parents::Vector{Int}, X::AbstractVector) = Tuple(X[f] for f in parents)

# Enumerate the cell-context keys (tuples of parent values) for a structure, in a
# deterministic order (matches `cell_key`'s parent ordering).
function _cell_contexts(parents::Vector{Int}, feature_values::Vector{Vector{String}})
    isempty(parents) && return Tuple[()]
    keys = Tuple[()]
    for f in parents
        keys = Tuple[(k..., v) for k in keys for v in feature_values[f]]
    end
    keys
end

"""
    build_model(feature_names, feature_values; alpha0, beta0, p_edge) -> StructureBMA

Enumerate all 2ⁿ edge structures, allocate a globally-unique tag per cell. Pure
construction of the declared structure family; no beliefs yet.
"""
function build_model(feature_names::Vector{String}, feature_values::Vector{Vector{String}};
                     alpha0::Float64 = 2.0, beta0::Float64 = 2.0, p_edge::Float64 = 0.5)
    n = length(feature_names)
    length(feature_values) == n || error("build_model: names/values length mismatch")
    structures = Vector{Int}[]
    for mask in 0:(2^n - 1)
        push!(structures, [i for i in 1:n if (mask >> (i - 1)) & 1 == 1])
    end
    cell_tag = Dict{Tuple, Int}[]
    tag_lo = Int[]
    tag_hi = Int[]
    tag = 0
    for parents in structures
        d = Dict{Tuple, Int}()
        lo = tag + 1                  # tags are contiguous per structure
        for cell in _cell_contexts(parents, feature_values)
            tag += 1
            d[cell] = tag
        end
        push!(cell_tag, d)
        push!(tag_lo, lo)
        push!(tag_hi, tag)
    end
    StructureBMA(feature_names, feature_values, structures, cell_tag, tag_lo, tag_hi,
                 n, alpha0, beta0, p_edge)
end

"""
    build_model_from_env(env; ...) -> StructureBMA

Read the declared feature spaces from `env[:features]` (a `Vector{FeatureDecl}`
populated by features.bdsl). The features are declared DATA; this turns them into
the typed structure family.
"""
function build_model_from_env(env; alpha0::Float64 = 2.0, beta0::Float64 = 2.0,
                              p_edge::Float64 = 0.5)
    decls = get(env, Symbol("features"), nothing)
    decls isa AbstractVector || error("feature brain: env[:features] must be a list of FeatureDecl")
    build_model_from_decls(decls; alpha0 = alpha0, beta0 = beta0, p_edge = p_edge)
end

"""
    build_model_from_decls(decls; ...) -> StructureBMA

Build a model from a `Vector{FeatureDecl}` (declared DATA). Shared by the waste model
(`env[:features]`) and the harm model (`env[:safety-features]`).
"""
function build_model_from_decls(decls; alpha0::Float64 = 2.0, beta0::Float64 = 2.0,
                                p_edge::Float64 = 0.5)
    decls isa AbstractVector || error("feature brain: expected a list of FeatureDecl")
    names = String[]
    vals = Vector{String}[]
    for d in decls
        d isa FeatureDecl || error("feature brain: expected FeatureDecl, got $(typeof(d))")
        d.space isa Finite ||
            error("feature brain: feature '$(d.name)' must have a Finite space, got $(typeof(d.space))")
        push!(names, string(d.name))
        push!(vals, String[string(v) for v in d.space.values])
    end
    build_model(names, vals; alpha0 = alpha0, beta0 = beta0, p_edge = p_edge)
end

# Are all of a model's declared features present in a body-sent feature dict? (Used to
# decide whether the harm posterior can be consulted for this event; if the body has not
# yet been upgraded to emit safety features, harm governance stays off — backward-compat.)
function _has_features(model::StructureBMA, features)
    fd = Dict{String, String}(string(k) => string(v) for (k, v) in features)
    all(haskey(fd, n) for n in model.feature_names)
end

# Structure-inclusion log-weights (independent edge-inclusion at p_edge; uniform
# when p_edge = 0.5). Shared by the sparse and dense priors.
_structure_logweights(model::StructureBMA) =
    Float64[length(parents) * log(model.p_edge) +
            (model.n_features - length(parents)) * log(1.0 - model.p_edge)
            for parents in model.structures]

"""
    build_prior(model) -> MixturePrevision

The structure-BMA prior: a mixture over structures, each a SPARSE product of
`Beta(alpha0, beta0)` cells (`SparseStructurePrevision` — only observed cells are
materialised). Bit-identical to `build_prior_dense` but O(structures) to build and
to condition, so the brain scales to many features. This is the daemon `make-prior`.
"""
function build_prior(model::StructureBMA)
    comps = Prevision[SparseStructurePrevision(model.alpha0, model.beta0,
                                               model.tag_lo[s], model.tag_hi[s],
                                               Dict{Int, BetaPrevision}())
                      for s in eachindex(model.structures)]
    MixturePrevision(comps, _structure_logweights(model))
end

"""
    build_prior_dense(model) -> MixturePrevision

The DENSE reference prior: each structure a full `ProductPrevision` of
`TaggedBetaPrevision` cells. Materialises the entire cross-product (exponential),
so it is used only as the exactness reference for `build_prior`
(see test/test_sparse_structure_equivalence.jl) and by tests that inspect cell
internals via `.factors`.
"""
function build_prior_dense(model::StructureBMA)
    comps = Prevision[]
    for (s, parents) in enumerate(model.structures)
        factors = TaggedBetaPrevision[]
        for cell in _cell_contexts(parents, model.feature_values)
            push!(factors, TaggedBetaPrevision(model.cell_tag[s][cell],
                                               BetaPrevision(model.alpha0, model.beta0)))
        end
        push!(comps, ProductPrevision(factors))
    end
    MixturePrevision(comps, _structure_logweights(model))
end

# ── Context plumbing ──

"""
    context_from_features(model, features) -> Vector{String}

Map a body-sent feature dict (string→bucket-string) to X, indexed by the model's
feature order. Validates each value is a declared bucket (a body/brain contract
violation fails loud rather than `KeyError`-ing deep in routing).
"""
function context_from_features(model::StructureBMA, features)
    fd = Dict{String, String}(string(k) => string(v) for (k, v) in features)
    X = String[]
    for (i, name) in enumerate(model.feature_names)
        haskey(fd, name) || error("feature brain: event missing declared feature '$name'")
        v = fd[name]
        v in model.feature_values[i] ||
            error("feature brain: feature '$name' value '$v' is not a declared bucket " *
                  "($(model.feature_values[i])) — body/brain vocabulary drift")
        push!(X, v)
    end
    X
end

# F(X): one matching cell-tag per structure (globally unique ⇒ a single Set routes
# the whole nested object).
firing_tags(model::StructureBMA, X::AbstractVector) =
    Set{Int}(model.cell_tag[s][cell_key(parents, X)] for (s, parents) in enumerate(model.structures))

# Measure-aware Bernoulli log-density (mirrors kernel.bdsl + the oracle test): takes
# the cell MEASURE and returns the cell's marginal log-predictive.
_approve_logdensity(m, o) = (a = mean(m.beta); o == 1 ? log(a) : log(1.0 - a))

# Per-context LEARNING kernel: BetaBernoulli on the cells in F(X), Flat (no-op)
# elsewhere. `condition(top, this, obs)` updates each structure's cell-for-X and
# reweights structures by the firing cell's predictive = the chain-rule marginal
# likelihood (Code-1; verified by test_product_bma_routing.jl).
_learn_kernel(fires::Set{Int}) =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _approve_logdensity;
           likelihood_family = FiringByTag(fires, BetaBernoulli(), Flat()))

# DECISION kernel: a plain BetaBernoulli (every component of `belief_at_X` IS the
# relevant cell, so all fire). Used only inside `voi`.
_decision_kernel() =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _approve_logdensity;
           likelihood_family = BetaBernoulli())

"""
    observe(model, top, X, obs) -> MixturePrevision

Bayesian update of the persistent belief on one `(context, response)`: a single
`condition` through the canalised path. `obs` is 1 (approve) or 0 (deny).
"""
observe(model::StructureBMA, top::MixturePrevision, X::AbstractVector, obs) =
    condition(top, _learn_kernel(firing_tags(model, X)), obs)

# Soft-evidence log-density: the firing cell's predictive marginal under virtual
# evidence with likelihoods (r, w) = (P(S|label=1), P(S|label=0)) — the chain-rule
# term the BMA reweights structures by. Generalises `_approve_logdensity`: the hard
# label o=1 is (r,w)=(1,0) ⇒ log(mean), o=0 is (0,1) ⇒ log(1-mean).
_soft_logdensity(m, obs) = (a = mean(m.beta); r = obs[1]; w = obs[2];
                            log(max(r * a + w * (1.0 - a), 1e-300)))

# Per-context SOFT-EVIDENCE learning kernel: SoftBernoulli on the cells in F(X),
# Flat elsewhere. The cell update is the mean-exact ADF collapse (α += π, β += 1-π,
# π = r·θ̄/(r·θ̄+w·(1-θ̄))); structures reweight by `_soft_logdensity`.
_soft_learn_kernel(fires::Set{Int}) =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _soft_logdensity;
           likelihood_family = FiringByTag(fires, SoftBernoulli(), Flat()))

"""
    observe_soft(model, top, X, r, w) -> MixturePrevision

Bayesian update on INDIRECT evidence about the (latent) label at context X: `r`
and `w` are the likelihoods of the observed signal under label = 1 / label = 0
(P(S|1), P(S|0)). A single `condition` through the canalised path with a
`SoftBernoulli` kernel — the firing cell's mean-exact soft-count, the BMA
structure-reweight by the signal's predictive marginal. Reduces EXACTLY to
`observe(model, top, X, 1)` at (r,w)=(1,0) and `observe(…, 0)` at (0,1) — the
hard label is the degenerate certain-signal corner.
"""
observe_soft(model::StructureBMA, top::MixturePrevision, X::AbstractVector,
             r::Real, w::Real) =
    condition(top, _soft_learn_kernel(firing_tags(model, X)), (Float64(r), Float64(w)))

# Select a structure component's cell-for-X by tag. Two representations:
# sparse (O(1) dict-backed cell_at) and dense (linear scan of factors). Both
# return the same TaggedBetaPrevision, so the decision sees identical beliefs.
_cell_for(comp::SparseStructurePrevision, tag::Int) = cell_at(comp, tag)
function _cell_for(comp::ProductPrevision, tag::Int)
    idx = findfirst(f -> f isa TaggedBetaPrevision && f.tag == tag, comp.factors)
    idx === nothing && error("feature brain: cell tag $tag not found")
    comp.factors[idx]
end

"""
    belief_at_context(model, top, X) -> MixturePrevision

The transient per-decision view: each structure's cell-for-X, carrying the
structure posterior weights verbatim. Pure construction — selects sub-beliefs and
copies weights; no arithmetic on probabilities.
"""
function belief_at_context(model::StructureBMA, top::MixturePrevision, X::AbstractVector)
    cells = Prevision[]
    for (s, parents) in enumerate(model.structures)
        tag = model.cell_tag[s][cell_key(parents, X)]
        push!(cells, _cell_for(top.components[s], tag))
    end
    # Carry the structure posterior verbatim onto the per-context cells (the
    # weight-carry lives in the `with_components` stdlib op, not here — so the
    # brain reads no private `log_weights`).
    with_components(top, cells)
end

# ── Decision: per-context EU-max with the (tail-aware) linear-cost utility ──
#
# With θ = P(approve|X), per-call cost c (dollars), false-block aversion λ
# (unitless: how many wrongly-blocked wanted-calls equal one allowed wasteful
# call), interruption cost q (dollars), and m = `expected_repeats` (the posterior-
# expected number of FURTHER identical calls a block prevents — the multi-turn
# look-ahead; m=0 ⇒ the myopic per-call form), against a proceed baseline:
#     EU(proceed) = 0
#     EU(block)   = c·(1+m)·(1−θ) − λ·c·θ = c·(1+m) − c·[(1+m)+λ]·θ   (LinearCombination/Identity)
#     EU(ask)     = voi − q                              (constant Functional)
# Blocking a WASTE call saves the whole expected tail — (1+m) calls: this one plus m
# more — whereas a wrongly-blocked WANTED call is not a loop, so its penalty stays λ·c
# (one call's friction). Block wins iff θ < (1+m)/((1+m)+λ): m=0 recovers the λ-only
# threshold 1/(1+λ) (c only scales the stakes); a large detected tail drives the cutoff
# toward 1. The `ask` VOI is computed against the SAME tail-aware block payoff, so
# resolving a likely-long loop is worth (1+m)× — an attention-precious (high-q) profile
# asks about a loop it would myopically wave through. m enters ONLY as a Functional
# coefficient; all three actions compare through the ONE canonical `optimise` (Invariant 1).

const _ID = Identity()
_lin(coeff::Float64, off::Float64) = LinearCombination(Tuple{Float64, TestFunction}[(coeff, _ID)], off)
_const(off::Float64) = LinearCombination(Tuple{Float64, TestFunction}[], off)

"""
    decide(model, top, X, cost; aversion, interrupt_cost, expected_repeats=0.0) -> Symbol

Return `:proceed`, `:block`, or `:ask`. `cost` is the per-call USD estimate;
`aversion` is λ (false-block aversion, unitless); `interrupt_cost` is q (dollars);
`expected_repeats` is m — the posterior-expected number of further identical calls a
block prevents (0 ⇒ the myopic per-call decision; supplied by the tail brain as
`expect(continuation_belief, GeometricTail())`).
"""
function decide(model::StructureBMA, top::MixturePrevision, X::AbstractVector, cost::Float64;
                aversion::Float64, interrupt_cost::Float64, expected_repeats::Float64 = 0.0,
                w_time::Float64 = 0.0, exp_time::Float64 = 0.0)
    bx = belief_at_context(model, top, X)
    tf = 1.0 + expected_repeats                       # expected calls a block prevents: this one + the tail
    # TIME coordinate: a block also SAVES the call's wall-clock (E[time]·tf seconds), valued at
    # w_time $/sec. Folds into the block offset next to the money saving c·tf. 0 ⇒ bit-identical.
    tcost = w_time * exp_time * tf
    block_fn = _lin(-cost * (tf + aversion), cost * tf + tcost)   # c·(1+m) + w_time·E[time]·(1+m) − c·[(1+m)+λ]·θ
    fpa_pb = Dict{Symbol, LinearCombination}(:proceed => _const(0.0), :block => block_fn)
    # EU(ask) = voi − q, computed through the canonical `net_voi` stdlib op (the
    # cost subtraction lives in stdlib, not as host arithmetic here) and entered
    # into `optimise` as a constant Functional so all three actions compare
    # through the ONE argmax.
    eu_ask = net_voi(bx, _decision_kernel(), [:proceed, :block], fpa_pb, [0, 1], interrupt_cost)
    fpa = Dict{Symbol, LinearCombination}(:proceed => _const(0.0),
                                          :block => block_fn,
                                          :ask => _const(eu_ask))
    optimise(bx, [:proceed, :block, :ask], fpa)
end

# ── Multi-outcome decision: one EU integrating waste AND harm in one currency ──
#
# Two posteriors at decision time: θ_a = P(approve|Xw) (waste brain, live-learned) and
# θ_u = P(unsafe|Xh) (harm brain, warm-frozen). With m = `expected_repeats` (the tail-aware
# look-ahead — waste only; a harmful action is one-shot, not a repeated loop), proceed = 0
# baseline (harm_cost H=0 AND m=0 recovers the single-outcome `decide` exactly):
#     EU(proceed) = 0
#     EU(block)   = c·(1+m)·[1 − θ_a] − cλθ_a + H·θ_u   (block avoids the waste TAIL and the harm)
#     EU(ask)     = voi − q                              (VOI of the user resolving approve)
# expressed as a LinearCombination over Projections of the JOINT belief (the constitution-
# clean "one expect over the joint"; ProductMeasure + Projection are existing Tier-1 ops),
# maximised by the ONE canonical `optimise`. Block beats proceed iff
#     θ_a < (1+m)/((1+m)+λ) + H·θ_u/(c·[(1+m)+λ]) — the waste cutoff SLIDES with BOTH the harm
# belief and the detected tail; no OR-of-thresholds can express it (eval/multi_outcome.jl +
# REGEX_IMPOSSIBLE.md).
"""
    decide_multi(waste_model, top_waste, harm_model, harm_top, Xw, Xh, cost;
                 aversion, interrupt_cost, harm_cost, expected_repeats=0.0) -> Symbol

Multi-outcome EU decision over the joint of the approve-belief and the unsafe-belief, via
the single canonical `optimise`. `harm_cost = 0` AND `expected_repeats = 0` reduce it to
the single-outcome myopic `decide`. `expected_repeats` (m) scales only the waste tail.
"""
function decide_multi(waste_model::StructureBMA, top_waste::MixturePrevision,
                      harm_model::StructureBMA, harm_top::MixturePrevision,
                      Xw::AbstractVector, Xh::AbstractVector, cost::Float64;
                      aversion::Float64, interrupt_cost::Float64, harm_cost::Float64,
                      expected_repeats::Float64 = 0.0,
                      w_time::Float64 = 0.0, exp_time::Float64 = 0.0)
    bxa = belief_at_context(waste_model, top_waste, Xw)   # P(approve|Xw)
    bxu = belief_at_context(harm_model, harm_top, Xh)     # P(unsafe|Xh)
    joint = ProductMeasure(Measure[wrap_in_measure(bxa), wrap_in_measure(bxu)])
    tf = 1.0 + expected_repeats        # expected calls a block prevents (waste tail; harm is one-shot)
    tcost = w_time * exp_time * tf      # time a block saves (w_time $/sec × E[time]·tf); 0 ⇒ bit-identical
    # EU(block) = c·(1+m)·[1−θ_a] + w_time·E[time]·(1+m) − cλθ_a + H·θ_u over the joint
    block_fn = LinearCombination(Tuple{Float64, TestFunction}[
        (-cost * (tf + aversion), Projection(1)), (harm_cost, Projection(2))], cost * tf + tcost)
    # EU(ask): VOI of the user resolving approve, against the SAME tail+time-aware block payoff so
    # asking about a likely-long loop inherits the (1+m)× value; a constant so all three
    # actions compare through one optimise.
    eu_ask = net_voi(bxa, _decision_kernel(), [:proceed, :block],
                     Dict(:proceed => _const(0.0), :block => _lin(-cost * (tf + aversion), cost * tf + tcost)),
                     [0, 1], interrupt_cost)
    fpa = Dict(:proceed => _const(0.0), :block => block_fn, :ask => _const(eu_ask))
    optimise(joint, [:proceed, :block, :ask], fpa)
end

# ── harm posterior persistence: version-stable per-context COUNTS ──
#
# Bayesian updating is order-independent, so the final posterior depends only on the per-
# context counts (n1 = harm-labelled, n0 = safe-labelled). We ship those as JSON and
# reconstruct by replaying `observe` — robust across Julia versions (unlike Serialization).
# The JSON shape: { "contexts": [ { "ctx": ["external-send","tainted-external-target","yes","no"],
#                                   "n1": 12, "n0": 1 }, ... ], ...metadata... }.
"""
    reconstruct_posterior(model, counts_path) -> MixturePrevision

Rebuild a frozen posterior from shipped per-context counts by replaying `observe`.
Bayesian updating is order-independent, so the posterior depends only on each
context's (n1, n0) counts — making this artifact version-stable (JSON), unlike a
`Serialization` blob (fragile across Julia versions; CI/image pin 1.11). Generic
over the StructureBMA: the harm posterior (P(unsafe|safety-features)) and the warm
WASTE posterior (P(approve|waste-features)) both reconstruct through this one path.
"""
function reconstruct_posterior(model::StructureBMA, counts_path::AbstractString)
    data = JSON3.read(read(counts_path, String))
    entries = data.contexts
    top = build_prior(model)
    for e in entries
        ctx = String[String(v) for v in e.ctx]
        for _ in 1:Int(e.n1); top = observe(model, top, ctx, 1); end
        for _ in 1:Int(e.n0); top = observe(model, top, ctx, 0); end
    end
    top
end

# Back-compat alias: the harm posterior was the first consumer of this path.
const reconstruct_harm_posterior = reconstruct_posterior

# Pass-1 followup logic, kept (yes→proceed, no→block); deterministic in Move 3.
function followup_after_response(event)
    resp = string(get(event, "response", ""))
    resp == "yes" ? :proceed : resp == "no" ? :block : :nothing
end

# ── Wiring: inject the brain closures into the BDSL env ──
#
# The daemon's call-sites are unchanged in SHAPE — it still does
# `env[:make-prior]()`, `env[:decide-action](...)`, `env[:observe-response](...)`,
# `env[:followup-after-response](event)` — but those symbols now resolve to Julia
# brain functions. Utility constants are declared DATA read from the env
# (utility.bdsl); defaults keep the brain runnable without that file.

function wire_brain!(env)
    p_edge   = Float64(get(env, Symbol("edge-inclusion-prior"), 0.5))
    λ        = Float64(get(env, Symbol("false-block-aversion"), 1.0))
    q        = Float64(get(env, Symbol("interrupt-cost"), 0.02))
    fallback = Float64(get(env, Symbol("fallback-call-cost"), 0.5))
    a0       = Float64(get(env, Symbol("cell-prior-alpha"), 2.0))
    b0       = Float64(get(env, Symbol("cell-prior-beta"), 2.0))

    H        = Float64(get(env, Symbol("harm-cost"), 0.0))
    hresp    = Symbol(get(env, Symbol("harm-response"), "ask"))  # :ask (confirm) | :block (enforce)
    wtime    = Float64(get(env, Symbol("w-time"), 0.0))          # TIME coordinate: $/sec of wall-clock

    model = build_model_from_env(env; alpha0 = a0, beta0 = b0, p_edge = p_edge)

    # Optional harm posterior (multi-outcome governance). ACTIVE only when the operator
    # declared `safety-features`, shipped a trained harm posterior (`harm-brain-path`),
    # AND set `harm-cost > 0`. Any missing piece (or a load failure) leaves decide-action
    # as exactly the single-outcome waste path — backward-compatible, fail-loud-then-off.
    harm_model = nothing
    harm_top = nothing
    sfeat = get(env, Symbol("safety-features"), nothing)
    # Default to the harm posterior shipped next to this brain file; an operator may
    # override with a `harm-brain-path` define. INERT until harm-cost > 0. The artifact is
    # version-stable per-context COUNTS (JSON), reconstructed via `observe` — NOT a Julia
    # Serialization blob, which is fragile across Julia versions (CI/image pin 1.11).
    hpath = get(env, Symbol("harm-brain-path"), joinpath(@__DIR__, "harm_brain.counts.json"))
    if H > 0.0 && sfeat isa AbstractVector && hpath !== nothing && !isempty(string(hpath))
        if isfile(string(hpath))
            try
                harm_model = build_model_from_decls(sfeat; alpha0 = a0, beta0 = b0, p_edge = p_edge)
                harm_top = reconstruct_posterior(harm_model, string(hpath))
                @info "credence-pi: harm posterior reconstructed; multi-outcome governance ON" path=string(hpath)
            catch e
                @warn "credence-pi: harm posterior failed to load; harm governance OFF" error=e
                harm_model = nothing
                harm_top = nothing
            end
        else
            @warn "credence-pi: harm-brain-path set but file missing; harm governance OFF" path=string(hpath)
        end
    end

    # Optional tail belief (multi-turn look-ahead) — OPT-IN via (define tail-aware "on").
    # The continuation posterior P(another identical call follows | features) uses the SAME
    # feature model as the waste brain, trained on loop continue/stop events
    # (eval/train_tail_brain.jl). When on, the per-call expected remaining repeats
    # m = expect(belief, GeometricTail()) scales the waste tail in decide/decide_multi (block
    # a long loop earlier; ask about an uncertain one); off ⇒ m=0, the myopic per-call EU.
    # OFF by default: it preserves the calibration-friendly cold-start (at θ≈0.5 we ASK to
    # learn, not pre-emptively block on a continuation prior), and on the WARM brain it
    # changes 0 ClawsBench decisions (high-m contexts already have low θ) — it bites in the
    # uncertain-persistent regime (test_feature_brain.jl §8; eval/tail_lookahead.jl). Version-
    # stable COUNTS JSON, reconstructed via `observe` like the warm/harm posteriors.
    tail_top = nothing
    if string(get(env, Symbol("tail-aware"), "off")) == "on"
        tpath = get(env, Symbol("tail-brain-path"), joinpath(@__DIR__, "tail_brain.counts.json"))
        if tpath !== nothing && !isempty(string(tpath)) && isfile(string(tpath))
            try
                tail_top = reconstruct_posterior(model, string(tpath))
                @info "credence-pi: tail (continuation) posterior ON; multi-turn look-ahead governance" path=string(tpath)
            catch e
                @warn "credence-pi: tail posterior failed to load; look-ahead OFF (myopic m=0)" error=e
                tail_top = nothing
            end
        else
            @warn "credence-pi: tail-aware on but tail brain missing; look-ahead OFF (myopic m=0)" path=string(tpath)
        end
    end

    # Expected remaining identical repeats a block prevents, per context: the closed-form
    # geometric-tail mean of the continuation posterior (m=0 with no tail brain ⇒ myopic EU).
    # This is the multi-turn look-ahead, computed by `expect` of a declared Functional.
    m_at(X) = tail_top === nothing ? 0.0 :
              expect(belief_at_context(model, tail_top, X), GeometricTail())

    # Optional governance LATENCY belief (TIME coordinate) — OPT-IN by file presence, like the
    # tail belief. Per-context Poisson-Gamma E[turns|X] (read by `expect`=α/β) × measured s̄/turn
    # ⇒ E[time|X] seconds a block would save. Absent ⇒ exp_time=0 ⇒ time-blind governance
    # (bit-identical). Version-stable counts-JSON (sufficient statistic), reconstructed exactly.
    gov_latency = nothing
    lpath = get(env, Symbol("governance-latency-path"), joinpath(@__DIR__, "governance_latency.counts.json"))
    if lpath !== nothing && !isempty(string(lpath)) && isfile(string(lpath))
        try
            data = JSON3.read(read(string(lpath), String))
            α0, β0 = Float64(data["turns_prior"][1]), Float64(data["turns_prior"][2])
            rate = Float64(data["rate_s"])
            tm = Dict{String, Float64}()
            for ctx in data["contexts"]
                g = GammaPrevision(α0 + Float64(ctx["sum_turns"]), β0 + Float64(ctx["n_obs"]))
                tm[join(string.(ctx["ctx"]), "|")] = Float64(expect(g, Identity())) * rate
            end
            gov_latency = tm
            @info "credence-pi: governance latency belief ON (time coordinate)" path=string(lpath)
        catch e
            @warn "credence-pi: governance latency failed to load; time-blind" error=e
            gov_latency = nothing
        end
    end
    time_at(X) = gov_latency === nothing ? 0.0 : get(gov_latency, join(string.(X), "|"), 0.0)

    env[Symbol("make-prior")] = () -> build_prior(model)

    env[Symbol("decide-action")] = (top, features, cost) -> begin
        c = cost === nothing ? fallback : Float64(cost)
        c = c <= 0.0 ? fallback : c
        # Multi-outcome only when harm is active AND the body sent the safety features for
        # this event; otherwise the single-outcome waste decision (unchanged).
        if harm_model !== nothing && harm_top !== nothing && _has_features(harm_model, features)
            Xw = context_from_features(model, features)
            Xh = context_from_features(harm_model, features)
            m = m_at(Xw)
            d = decide_multi(model, top, harm_model, harm_top, Xw, Xh, c;
                             aversion = λ, interrupt_cost = q, harm_cost = H, expected_repeats = m,
                             w_time = wtime, exp_time = time_at(Xw))
            # Research-stage effector policy (harm-response = :ask): a harm-DRIVEN stop is a
            # CONFIRMATION, not a refusal — the harm belief is benchmark-seeded, not yet
            # user-calibrated, so asking has value and the response is the calibration
            # signal we learn from. Waste-driven blocks are unchanged (waste is proven).
            # Like shadowMode, this is an effector policy, not a change to the EU reasoning:
            # the harm term decided "do not proceed"; :ask realises that as "confirm".
            if d === :block && hresp === :ask
                d_waste = decide(model, top, Xw, c; aversion = λ, interrupt_cost = q,
                                 expected_repeats = m, w_time = wtime, exp_time = time_at(Xw))
                d = d_waste === :block ? :block : :ask   # harm was the driver ⇒ confirm
            end
            d
        else
            X = context_from_features(model, features)
            decide(model, top, X, c; aversion = λ, interrupt_cost = q, expected_repeats = m_at(X),
                   w_time = wtime, exp_time = time_at(X))
        end
    end

    env[Symbol("observe-response")] = (top, features, obs) -> begin
        X = context_from_features(model, features)
        observe(model, top, X, obs)
    end

    env[Symbol("followup-after-response")] = followup_after_response

    model
end

end # module FeatureBrain
