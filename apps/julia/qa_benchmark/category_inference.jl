# Role: brain-side application
"""
    category_inference.jl — Gaussian Naive Bayes category classifier.

Paper 1, Phase B2b (`docs/paper1/move-2-design.md` §3.1;
`docs/paper1/master-plan.md` §4a). Infers a soft category posterior
`P(category | embedding)` for a question, so the same inferred category
information is available to every agent and fairness can be enforced
(master-plan OQ1(c), OQ5).

Model — option (b-NB): per-(class, dimension) Normal-Gamma conjugate
posteriors over the embedding coordinates, plus a Dirichlet class prior.
The class posterior for a new embedding is computed by `condition`-ing a
`CategoricalMeasure` (built from the Dirichlet-Categorical predictive
over classes) through a naive-Bayes kernel whose per-dimension
log-density is the Normal-Gamma marginal predictive (Student-t). Every
belief update is a `condition` call — no host-side weight surgery
(Invariant 1).

State is held at the Prevision level (master-plan D4): a
`DirichletPrevision` plus a `Matrix{NormalGammaPrevision}`. The public
`infer` / `loo_category_inference` return `Dict{Symbol, Float64}`.

Model-agnostic in the embeddings (D5): takes a `Matrix{Float64}` of shape
`(N, D)`; producing real sentence embeddings for the question bank is a
later move.

Tests: `test/test_qa_benchmark_category_inference.jl`.
"""

using Credence
using Credence: DirichletPrevision, NormalGammaPrevision, log_predictive
using SpecialFunctions: lgamma

# Vague priors. Class prior: symmetric Dirichlet(1, …, 1) — one
# pseudo-count per class. Per-(class, dim) Normal-Gamma: κ=1 (one
# pseudo-observation), μ=0, α=2, β=2.
const _NG_PRIOR = (κ = 1.0, μ = 0.0, α = 2.0, β = 2.0)

# Scalar-observation kernel: drives the Normal-Gamma conjugate update.
# Source: the (μ, σ²) hypothesis space the Normal-Gamma declares.
# Target: a single observed scalar (one embedding coordinate). `condition`
# uses the conjugate path (maybe_conjugate → update); the log-density
# below is the honest observation model and is not consulted on that path.
const SCALAR_OBS_KERNEL = Kernel(
    ProductSpace(Space[Euclidean(1), PositiveReals()]),
    Euclidean(1),
    h -> error("SCALAR_OBS_KERNEL.generate not exercised (conjugate path)"),
    (h, o) -> begin
        μ_val, σ² = h[1], h[2]
        -0.5 * log(2π * σ²) - (Float64(o) - μ_val)^2 / (2.0 * σ²)
    end;
    likelihood_family = NormalGammaLikelihood(),
)

# Class-observation kernel: drives the Dirichlet conjugate update. Built
# per category vocabulary (the simplex dimension depends on K).
_class_obs_kernel(cats::Vector{Symbol}) = Kernel(
    Simplex(length(cats)),
    Finite(cats),
    h -> error("class kernel generate not exercised (conjugate path)"),
    (h, o) -> begin
        idx = findfirst(==(o), cats)
        idx === nothing && error("unknown category $o")
        log(h[idx])
    end;
    likelihood_family = Categorical(Finite(cats)),
)

# Normal-Gamma marginal predictive (Student-t). NormalGamma(κ, μ, α, β)
# predictive for x_new is t_{2α}(x; μ, β(κ+1)/(ακ)). This is the
# Normal-Gamma conjugate family's own closed form — the conjugate-
# machinery vocabulary category of the `expect-through-accessor`
# precedent — not a bypass of `expect` for a probabilistic property the
# stdlib already exposes. Uses SpecialFunctions.lgamma.
function student_t_logpdf(x::Real, ν::Real, m::Real, s²::Real)
    z = (x - m)^2 / s²
    lgamma((ν + 1) / 2) - lgamma(ν / 2) -
        0.5 * (log(ν) + log(π) + log(s²)) -
        ((ν + 1) / 2) * log(1 + z / ν)
end

function ng_predictive_logpdf(ng::NormalGammaPrevision, x::Real)
    ν = 2 * ng.α
    s² = ng.β * (ng.κ + 1) / (ng.α * ng.κ)
    student_t_logpdf(x, ν, ng.μ, s²)
end

# Inference kernel: Finite(cats) → Euclidean(D). Its log-density is the
# naive-Bayes per-dimension Student-t sum. `Flat()` because
# CategoricalMeasure's `condition` uses the per-hypothesis density loop
# and never consults maybe_conjugate.
_inference_kernel(params::Matrix{NormalGammaPrevision}, cats::Vector{Symbol}) = Kernel(
    Finite(cats),
    Euclidean(size(params, 2)),
    c -> error("inference kernel generate not exercised (density path)"),
    (c, e) -> begin
        cidx = findfirst(==(c), cats)
        cidx === nothing && error("unknown category $c")
        sum(ng_predictive_logpdf(params[cidx, j], e[j]) for j in 1:size(params, 2))
    end;
    likelihood_family = Flat(),
)

"""
    CategoryClassifier

A fitted Gaussian-NB category classifier. `class_prior` is the Dirichlet
posterior over the K categories; `params` is the K×D matrix of
per-(class, dimension) Normal-Gamma posteriors; `categories` is the
sorted category vocabulary (length K).
"""
struct CategoryClassifier
    class_prior::DirichletPrevision
    params::Matrix{NormalGammaPrevision}   # K × D
    categories::Vector{Symbol}             # sorted, length K
end

"""
    fit(embeddings::Matrix{Float64}, categories::Vector{Symbol}) -> CategoryClassifier

Fit per-(class, dim) Normal-Gamma posteriors and a Dirichlet class prior
by conditioning on each (embedding, category) pair. `embeddings` is
shape `(N, D)`; `categories` is length `N`. Every belief update is a
`condition` call.
"""
function fit(embeddings::Matrix{Float64}, categories::Vector{Symbol})
    N = size(embeddings, 1)
    N == length(categories) ||
        error("CategoryClassifier.fit: embeddings rows ($N) must match categories length ($(length(categories)))")
    N >= 1 ||
        error("CategoryClassifier.fit: empty training set; need at least one (embedding, category) pair")

    cats = sort(unique(categories))
    K = length(cats)
    D = size(embeddings, 2)

    class_prior = DirichletPrevision(ones(K))
    params = [NormalGammaPrevision(_NG_PRIOR.κ, _NG_PRIOR.μ, _NG_PRIOR.α, _NG_PRIOR.β)
              for _ in 1:K, _ in 1:D]
    class_kernel = _class_obs_kernel(cats)

    for i in 1:N
        c = categories[i]
        cidx = findfirst(==(c), cats)
        class_prior = condition(class_prior, class_kernel, c)
        for j in 1:D
            params[cidx, j] = condition(params[cidx, j], SCALAR_OBS_KERNEL, embeddings[i, j])
        end
    end

    CategoryClassifier(class_prior, params, cats)
end

"""
    infer(clf::CategoryClassifier, embedding::Vector{Float64}) -> Dict{Symbol, Float64}

Soft category posterior `P(category | embedding)`; sums to 1.0. Computed
by `condition`-ing the Dirichlet-Categorical class predictive through the
naive-Bayes kernel.
"""
function infer(clf::CategoryClassifier, embedding::Vector{Float64})
    length(embedding) == size(clf.params, 2) ||
        error("CategoryClassifier.infer: embedding length $(length(embedding)) ≠ classifier dim $(size(clf.params, 2))")

    cats = clf.categories
    class_kernel = _class_obs_kernel(cats)
    # Dirichlet-Categorical predictive over classes via the sanctioned
    # log_predictive accessor (not a raw α read).
    log_class = [log_predictive(clf.class_prior, class_kernel, c) for c in cats]
    cat_prior = CategoricalMeasure(Finite(cats), log_class)

    nb_k = _inference_kernel(clf.params, cats)
    post = condition(cat_prior, nb_k, embedding)

    w = weights(post)
    Dict(cats[i] => w[i] for i in eachindex(cats))
end

"""
    loo_category_inference(embeddings::Matrix{Float64}, categories::Vector{Symbol})
        -> Vector{Dict{Symbol, Float64}}

Leave-one-out cross-validated category inference (master-plan OQ3(c)).
For each row `i`, fit a classifier on every other row and infer on row
`i`. Returns one soft posterior per input row, in input order. Cost is
`N` fits of `O(N·D)` each, i.e. `O(N²·D)`.
"""
function loo_category_inference(embeddings::Matrix{Float64}, categories::Vector{Symbol})
    N = size(embeddings, 1)
    N == length(categories) ||
        error("loo_category_inference: embeddings rows ($N) must match categories length ($(length(categories)))")
    N >= 2 ||
        error("loo_category_inference: need at least two (embedding, category) pairs for leave-one-out")

    out = Vector{Dict{Symbol,Float64}}(undef, N)
    keep = trues(N)
    for i in 1:N
        keep[i] = false
        clf = fit(embeddings[keep, :], categories[keep])
        out[i] = infer(clf, embeddings[i, :])
        keep[i] = true
    end
    out
end
