# structure_bma.jl — the structure-BMA builder + observe + readout, lifted into
# engine stdlib (decouple Move 3). Compositions over existing Tier-1 objects
# (`SparseStructurePrevision`/`condition`/`with_components`); NO new frozen type and
# NO new axiom-constrained function. Previously lived in
# apps/credence-pi/brain/feature_brain.jl (a co-released app brain); promoted here so
# a separable, non-embedding consumer can drive structure inference over the skin wire.
#
# Structure-BMA over Bayesian-network edges (feature → a Bernoulli leaf): a mixture over
# the 2ⁿ parent subsets, each a sparse product of Beta cells. Conditioning reweights
# the structures by the firing cell's chain-rule marginal likelihood (the Occam work),
# learns the firing cell, and leaves the rest untouched.
#
# Two representations kept separate (Invariant 3):
#   * the persistent belief — a `MixturePrevision` over the 2ⁿ structures, each a
#     `SparseStructurePrevision` of `TaggedBeta` cells; learning conditions THIS.
#   * the `StructureBMA` descriptor — the structural-analysis representation (feature
#     vocabulary, the 2ⁿ enumeration, the per-cell tag bookkeeping) used to build the
#     prior, route observations, and select the per-context decision view. It is NOT a
#     Prevision; it carries no beliefs.

# ── The descriptor: feature spec + structure enumeration + tag bookkeeping ──
#
# A `structure` is a subset of feature indices — the parents of the leaf (edges INTO
# the leaf). Each structure has one cell per element of the cross-product of its
# parents' value-sets; every cell carries a GLOBALLY-UNIQUE integer tag so a single
# per-context `FiringByTag` kernel routes correctly through the whole nested object
# (within each structure exactly one cell's tag fires).

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
    build_structure_model(feature_names, feature_values; alpha0, beta0, p_edge) -> StructureBMA

Enumerate all 2ⁿ edge structures, allocate a globally-unique tag per cell. Pure
construction of the declared structure family; no beliefs yet.
"""
function build_structure_model(feature_names::Vector{String}, feature_values::Vector{Vector{String}};
                               alpha0::Float64 = 2.0, beta0::Float64 = 2.0, p_edge::Float64 = 0.5)
    n = length(feature_names)
    length(feature_values) == n || error("build_structure_model: names/values length mismatch")
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

# (The `FeatureDecl`-reading constructor `build_model_from_decls` stays app-side in the
# credence-pi shim: `FeatureDecl` is an eval-layer type included after this module, and
# reading a BDSL env is consumer wiring, not engine machinery.)

# Structure-inclusion log-weights (independent edge-inclusion at p_edge; uniform when
# p_edge = 0.5). Shared by the sparse and dense priors.
_structure_logweights(model::StructureBMA) =
    Float64[length(parents) * log(model.p_edge) +
            (model.n_features - length(parents)) * log(1.0 - model.p_edge)
            for parents in model.structures]

"""
    build_structure_prior(model) -> MixturePrevision

The structure-BMA prior: a mixture over structures, each a SPARSE product of
`Beta(alpha0, beta0)` cells (`SparseStructurePrevision` — only observed cells are
materialised). Bit-identical to `build_structure_prior_dense` but O(structures) to
build and to condition, so the brain scales to many features.
"""
function build_structure_prior(model::StructureBMA)
    comps = Prevision[SparseStructurePrevision(model.alpha0, model.beta0,
                                               model.tag_lo[s], model.tag_hi[s],
                                               Dict{Int, BetaPrevision}())
                      for s in eachindex(model.structures)]
    MixturePrevision(comps, _structure_logweights(model))
end

"""
    reconstruct_structure_prior_from_data(model, data) -> MixturePrevision

Warm-seed a structure-BMA posterior from already-parsed per-context counts (the skin/shim
parses the JSON; the engine never reads the host FS) by replaying `structure_observe`.
Bayesian updating is order-independent ⇒ the posterior depends only on each context's
(n1, n0) counts, so reconstruction is exact + version-stable. `data === nothing` ⇒ the cold
prior. JSON shape: `{ "contexts": [ {"ctx":[…], "n1":N, "n0":M}, … ] }`. Symmetric with
routing's `reconstruct_routing_tops_from_data` — lets a wire consumer warm-seed governance in
ONE server-side call instead of N `structure_observe` round-trips.
"""
function reconstruct_structure_prior_from_data(model::StructureBMA, data)
    top = build_structure_prior(model)
    data === nothing && return top
    for e in data["contexts"]
        ctx = String[String(v) for v in e["ctx"]]
        for _ in 1:Int(e["n1"]); top = structure_observe(model, top, ctx, 1); end
        for _ in 1:Int(e["n0"]); top = structure_observe(model, top, ctx, 0); end
    end
    top
end

"""
    build_structure_prior_dense(model) -> MixturePrevision

The DENSE reference prior: each structure a full `ProductPrevision` of
`TaggedBetaPrevision` cells. Materialises the entire cross-product (exponential), so it
is used only as the exactness reference for `build_structure_prior` and by tests that
inspect cell internals via `.factors`.
"""
function build_structure_prior_dense(model::StructureBMA)
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

Map a feature dict (string→bucket-string) to X, indexed by the model's feature order.
Validates each value is a declared bucket (a contract violation fails loud rather than
`KeyError`-ing deep in routing).
"""
function context_from_features(model::StructureBMA, features)
    fd = Dict{String, String}(string(k) => string(v) for (k, v) in features)
    X = String[]
    for (i, name) in enumerate(model.feature_names)
        haskey(fd, name) || error("structure-bma: event missing declared feature '$name'")
        v = fd[name]
        v in model.feature_values[i] ||
            error("structure-bma: feature '$name' value '$v' is not a declared bucket " *
                  "($(model.feature_values[i])) — vocabulary drift")
        push!(X, v)
    end
    X
end

# F(X): one matching cell-tag per structure (globally unique ⇒ a single Set routes the
# whole nested object).
structure_firing_tags(model::StructureBMA, X::AbstractVector) =
    Set{Int}(model.cell_tag[s][cell_key(parents, X)] for (s, parents) in enumerate(model.structures))

# Measure-aware Bernoulli log-density: takes the cell MEASURE and returns the cell's
# marginal log-predictive.
_approve_logdensity(m, o) = (a = mean(m.beta); o == 1 ? log(a) : log(1.0 - a))

# Per-context LEARNING kernel: BetaBernoulli on the cells in F(X), Flat (no-op)
# elsewhere. `condition(top, this, obs)` updates each structure's cell-for-X and
# reweights structures by the firing cell's predictive = the chain-rule marginal
# likelihood.
_learn_kernel(fires::Set{Int}) =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _approve_logdensity;
           likelihood_family = FiringByTag(fires, BetaBernoulli(), Flat()))

# DECISION kernel: a plain BetaBernoulli (every component of the per-context view IS the
# relevant cell, so all fire). Used only inside `voi`/`net_voi` on a `belief_at_context`
# view. Exported so a consumer's decision path can reach it without re-deriving it.
structure_decision_kernel() =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _approve_logdensity;
           likelihood_family = BetaBernoulli())

"""
    structure_observe(model, top, X, obs) -> MixturePrevision

Bayesian update of the persistent belief on one `(context, response)`: a single
`condition` through the canalised path. `obs` is 1 or 0.
"""
structure_observe(model::StructureBMA, top::MixturePrevision, X::AbstractVector, obs) =
    condition(top, _learn_kernel(structure_firing_tags(model, X)), obs)

# Soft-evidence log-density: the firing cell's predictive marginal under virtual
# evidence with likelihoods (r, w) = (P(S|label=1), P(S|label=0)).
_soft_logdensity(m, obs) = (a = mean(m.beta); r = obs[1]; w = obs[2];
                            log(max(r * a + w * (1.0 - a), 1e-300)))

# Per-context SOFT-EVIDENCE learning kernel: SoftBernoulli on the cells in F(X), Flat
# elsewhere (the mean-exact ADF collapse).
_soft_learn_kernel(fires::Set{Int}) =
    Kernel(Interval(0.0, 1.0), Finite([0, 1]), theta -> theta, _soft_logdensity;
           likelihood_family = FiringByTag(fires, SoftBernoulli(), Flat()))

"""
    structure_observe_soft(model, top, X, r, w) -> MixturePrevision

Bayesian update on INDIRECT evidence about the (latent) label at context X: `r`, `w`
are the likelihoods of the observed signal under label = 1 / label = 0. Reduces EXACTLY
to `structure_observe(model, top, X, 1)` at (r,w)=(1,0) and `…, 0` at (0,1).
"""
structure_observe_soft(model::StructureBMA, top::MixturePrevision, X::AbstractVector,
                       r::Real, w::Real) =
    condition(top, _soft_learn_kernel(structure_firing_tags(model, X)), (Float64(r), Float64(w)))

# Select a structure component's cell-for-X by tag. Sparse (O(1) dict-backed cell_at)
# and dense (linear scan of factors) both return the same TaggedBetaPrevision.
_cell_for(comp::SparseStructurePrevision, tag::Int) = cell_at(comp, tag)
function _cell_for(comp::ProductPrevision, tag::Int)
    idx = findfirst(f -> f isa TaggedBetaPrevision && f.tag == tag, comp.factors)
    idx === nothing && error("structure-bma: cell tag $tag not found")
    comp.factors[idx]
end

"""
    belief_at_context(model, top, X) -> MixturePrevision

The transient per-decision view: each structure's cell-for-X, carrying the structure
posterior weights verbatim. Pure construction — selects sub-beliefs and copies weights
(via the `with_components` stdlib op); no arithmetic on probabilities.
"""
function belief_at_context(model::StructureBMA, top::MixturePrevision, X::AbstractVector)
    cells = Prevision[]
    for (s, parents) in enumerate(model.structures)
        tag = model.cell_tag[s][cell_key(parents, X)]
        push!(cells, _cell_for(top.components[s], tag))
    end
    with_components(top, cells)
end
