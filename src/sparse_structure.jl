# sparse_structure.jl — methods for SparseStructurePrevision (the struct lives
# in prevision.jl, mirroring the TaggedBetaPrevision struct/methods split).
#
# An exact, sparse execution-layer backend for a structure-BMA component. A dense
# component is a ProductPrevision of TaggedBeta cells, one per context (the
# cross-product of the structure's parent feature values) — exponential in the
# feature count and almost entirely unobserved. This stores only the OBSERVED
# cells (a Dict) over a shared Beta(alpha0,beta0) prior, and delegates every
# operation to the dense TaggedBetaPrevision methods on the SINGLE firing cell,
# so results are bit-identical to the dense product (the non-firing cells are
# Flat no-ops, skipped in _predictive_ll). Proven by
# test/test_sparse_structure_equivalence.jl.
#
# Routing: tags are allocated contiguously per structure (FeatureBrain.build_model
# loops structures then cells, tag += 1), so this component owns [tag_lo, tag_hi]
# and its firing cell for a context is the unique tag in `fires` within that range.

# The firing tag for this structure under a FiringByTag kernel: the unique
# element of `fires` in [tag_lo, tag_hi], or nothing if none fires here.
function _firing_in_range(sc::SparseStructurePrevision, fires)
    for t in fires
        sc.tag_lo <= t <= sc.tag_hi && return t
    end
    nothing
end

"""
    cell_at(sc::SparseStructurePrevision, tag) -> TaggedBetaPrevision

The cell for `tag`: its observed Beta, or the shared prior if never observed.
Returned as a `TaggedBetaPrevision` so consumers (e.g. `belief_at_context`) see
exactly what the dense `ProductPrevision` path produced.
"""
cell_at(sc::SparseStructurePrevision, tag::Int) =
    TaggedBetaPrevision(tag, get(sc.observed, tag, BetaPrevision(sc.alpha0, sc.beta0)))

# condition: update only the firing cell, via the existing TaggedBeta condition
# (which resolves FiringByTag -> BetaBernoulli for the firing tag and
# conjugate-updates the Beta). Non-firing cells remain implicit at the prior.
function condition(sc::SparseStructurePrevision, k::Kernel, obs)
    k.likelihood_family isa FiringByTag ||
        error("SparseStructurePrevision: condition expects a FiringByTag kernel, " *
              "got $(typeof(k.likelihood_family))")
    t = _firing_in_range(sc, k.likelihood_family.fires)
    t === nothing && return sc            # nothing fires in this structure -> no-op
    updated = condition(cell_at(sc, t), k, obs)::TaggedBetaPrevision
    new_observed = copy(sc.observed)
    new_observed[t] = updated.beta
    SparseStructurePrevision(sc.alpha0, sc.beta0, sc.tag_lo, sc.tag_hi, new_observed)
end

# predictive marginal-likelihood: only the firing cell contributes (the dense
# sum skips Flat factors), so delegate to that cell's TaggedBeta predictive.
function _predictive_ll(sc::SparseStructurePrevision, k::Kernel, obs)
    k.likelihood_family isa FiringByTag ||
        error("SparseStructurePrevision: _predictive_ll expects a FiringByTag kernel")
    t = _firing_in_range(sc, k.likelihood_family.fires)
    t === nothing && return 0.0
    _predictive_ll(cell_at(sc, t), k, obs)
end
