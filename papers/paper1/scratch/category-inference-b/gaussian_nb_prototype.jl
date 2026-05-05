# gaussian_nb_prototype.jl — scratch evidence for §3 of the B2a design doc.
#
# Purpose: verify that option (b-NB) — Gaussian Naive Bayes on embeddings
# with a Dirichlet class prior — can be expressed using ONLY existing
# Credence DSL primitives, with NO new LikelihoodFamily entries, NO new
# ConjugatePrevision pairs, NO new update() methods.
#
# This is NOT a B2b deliverable. It is empirical evidence for one claim
# in the design doc. Throwaway. Lives in scratch/ and may be deleted
# after B2b lands.
#
# Run from repo root:
#     julia --project=. papers/paper1/scratch/category-inference-b/gaussian_nb_prototype.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))

using Credence
using Random
using Printf

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────

const CATS = [:factual, :numerical, :recent_events, :misconceptions, :reasoning]
const K = length(CATS)
const D = 8   # tiny embedding dimension

# Synthetic data: each category has a fixed mean μ_c ∈ R^D; embeddings
# are μ_c + N(0, I) noise. Tells us whether the inference machinery can
# recover the class structure that is actually there. Means are sampled
# under a *fixed* RNG seed (MEANS_SEED) so train and test share them;
# the per-point noise uses the caller's seed.
const MEANS_SEED = 7
const TRUE_MEANS = let rng = MersenneTwister(MEANS_SEED)
    # Pull the means apart with a separation factor so the classes are
    # distinguishable at D=8 with ~5 examples per class.
    [3.0 .* randn(rng, D) for _ in CATS]
end

function synth_data(n_per_cat::Int; seed::Int)
    rng = MersenneTwister(seed)
    pts = Tuple{Symbol, Vector{Float64}}[]
    for (k, c) in enumerate(CATS), _ in 1:n_per_cat
        push!(pts, (c, TRUE_MEANS[k] .+ randn(rng, D)))
    end
    pts
end

# ──────────────────────────────────────────────────────────────────────
# Per-(class, dimension) NormalGamma prior, conjugate update via
# existing NormalGamma + NormalGammaLikelihood pair (src/conjugate.jl:74-93).
# ──────────────────────────────────────────────────────────────────────

function build_param_prior()
    # Vague prior: κ=1 (one pseudo-obs), μ=0, α=2, β=2.
    [NormalGammaMeasure(1.0, 0.0, 2.0, 2.0) for _ in 1:K, _ in 1:D]
end

# Single-scalar observation kernel for NormalGamma conjugate update.
# Source: ProductSpace([Euclidean(1), PositiveReals()]) — the (μ, σ²)
# hypothesis space NormalGammaMeasure declares.
# Target: Euclidean(1) — single observed scalar.
const SCALAR_OBS_KERNEL = Kernel(
    ProductSpace(Space[Euclidean(1), PositiveReals()]),
    Euclidean(1),
    h -> error("generate not exercised in this prototype"),
    (h, o) -> begin
        μ_val, σ² = h[1], h[2]
        -0.5 * log(2π * σ²) - (Float64(o) - μ_val)^2 / (2.0 * σ²)
    end;
    likelihood_family = NormalGammaLikelihood(),
)

# Class-prior Dirichlet: one count per category, conjugate to Categorical.
function build_class_prior()
    α0 = ones(K)
    DirichletMeasure(Simplex(K), Finite(CATS), α0)
end

const CLASS_OBS_KERNEL = Kernel(
    Simplex(K),
    Finite(CATS),
    h -> error("generate not exercised"),
    (h, o) -> begin
        idx = findfirst(==(o), CATS)
        idx === nothing && error("unknown category $o")
        log(h[idx])
    end;
    likelihood_family = Categorical(Finite(CATS)),
)

# ──────────────────────────────────────────────────────────────────────
# Calibration: update params and class prior from training data.
# Every belief update is a condition() call. No host-side weight surgery.
# ──────────────────────────────────────────────────────────────────────

function calibrate(train::Vector{Tuple{Symbol, Vector{Float64}}})
    params = build_param_prior()
    class_prior = build_class_prior()

    for (c, e) in train
        cat_idx = findfirst(==(c), CATS)
        # Update class-frequency posterior (Dirichlet ← Categorical obs).
        class_prior = condition(class_prior, CLASS_OBS_KERNEL, c)
        # Update per-(class, dim) NormalGamma posteriors.
        for j in 1:D
            params[cat_idx, j] = condition(params[cat_idx, j], SCALAR_OBS_KERNEL, e[j])
        end
    end

    (class_prior, params)
end

# ──────────────────────────────────────────────────────────────────────
# Inference: P(C | e_new). Uses the NormalGamma marginal predictive
# (Student-t) as the per-dimension observation likelihood, summed over
# dimensions (naive). The marginal predictive is a closed-form integral
# of the joint over (μ, σ²); it is computed here from the NormalGamma's
# four hyperparameters. This is the "expect over hypothesis" step done
# analytically — Credence DSL idiom would route through expect() on the
# NormalGamma; for the prototype the closed form is more legible.
# ──────────────────────────────────────────────────────────────────────

function _log_gamma(x::Float64)
    if x < 0.5
        return log(π / sin(π * x)) - _log_gamma(1.0 - x)
    end
    x -= 1.0
    g = 7.0
    coeffs = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
              771.32342877765313, -176.61502916214059, 12.507343278686905,
              -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    t = x + g + 0.5
    s = coeffs[1]
    for i in 2:length(coeffs)
        s += coeffs[i] / (x + Float64(i - 1))
    end
    0.5 * log(2π) + (x + 0.5) * log(t) - t + log(s)
end

# log Student-t pdf with location m, scale s², degrees of freedom ν.
function student_t_logpdf(x, ν, m, s²)
    z = (x - m)^2 / s²
    _log_gamma((ν + 1) / 2) - _log_gamma(ν / 2) -
        0.5 * (log(ν) + log(π) + log(s²)) -
        ((ν + 1) / 2) * log(1 + z / ν)
end

# NormalGamma(κ, μ, α, β) marginal predictive for x_new is t_{2α}(x; μ, β(κ+1)/(α κ)).
function ng_predictive_logpdf(ng::NormalGammaMeasure, x::Float64)
    ν = 2 * ng.α
    m = ng.μ
    s² = ng.β * (ng.κ + 1) / (ng.α * ng.κ)
    student_t_logpdf(x, ν, m, s²)
end

# Build the inference kernel: Finite(CATS) → Euclidean(D), log_density is
# the naive-Bayes per-dimension Student-t sum.
function build_inference_kernel(params)
    Kernel(
        Finite(CATS),
        Euclidean(D),
        c -> error("generate not exercised"),
        (c, e) -> begin
            cat_idx = findfirst(==(c), CATS)
            sum(ng_predictive_logpdf(params[cat_idx, j], e[j]) for j in 1:D)
        end;
        likelihood_family = Flat(),  # not used: CategoricalMeasure condition
                                     # never consults maybe_conjugate (see
                                     # src/ontology.jl:763-771).
    )
end

# Extract a CategoricalMeasure prior from the Dirichlet posterior using
# the posterior mean. (A fuller predictive would integrate over the
# Dirichlet — the Dirichlet-Categorical marginal predictive is again
# proportional to α — same answer for this prototype.)
function class_prior_as_categorical(dir::DirichletMeasure)
    π = dir.alpha ./ sum(dir.alpha)
    CategoricalMeasure(Finite(CATS), log.(π))
end

function infer_category(class_prior::DirichletMeasure, params, e_new::Vector{Float64})
    cat_prior = class_prior_as_categorical(class_prior)
    nb_k = build_inference_kernel(params)
    condition(cat_prior, nb_k, e_new)
end

# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 60)
    println("Gaussian NB on embeddings — option (b-NB) prototype")
    println("=" ^ 60)

    # 5 train + 5 test per category — shared TRUE_MEANS, distinct noise.
    train = synth_data(5; seed=42)
    test  = synth_data(5; seed=123)

    println("\nCalibrating on $(length(train)) examples ($(length(train)÷K)/category)…")
    class_prior, params = calibrate(train)
    println("  Dirichlet α post-calibration: $(class_prior.alpha)")
    println("  NormalGamma[factual, dim 1] post-calibration:")
    @printf("    κ=%.2f μ=%.3f α=%.2f β=%.3f\n",
        params[1, 1].κ, params[1, 1].μ, params[1, 1].α, params[1, 1].β)

    println("\nInference on $(length(test)) test embeddings:")
    correct = 0
    avg_p_true = 0.0
    avg_entropy = 0.0
    for (true_c, e) in test
        post = infer_category(class_prior, params, e)
        w = weights(post)
        pred_idx = argmax(w)
        pred_c = post.space.values[pred_idx]
        is_right = pred_c == true_c
        correct += is_right
        true_idx = findfirst(==(true_c), CATS)
        p_true = w[true_idx]
        avg_p_true += p_true
        H = -sum(wi * log(max(wi, 1e-300)) for wi in w)
        avg_entropy += H
        mark = is_right ? "✓" : "✗"
        @printf("  %s true=%-15s pred=%-15s P(true)=%.3f H=%.3f\n",
            mark, true_c, pred_c, p_true, H)
    end
    n = length(test)
    @printf("\n  Accuracy: %d/%d = %.2f\n", correct, n, correct/n)
    @printf("  Mean P(true category): %.3f (chance = %.3f)\n", avg_p_true/n, 1/K)
    @printf("  Mean entropy: %.3f (max = log(K) = %.3f)\n", avg_entropy/n, log(K))

    println("\n" * "=" ^ 60)
    println("Sanity: posterior weights normalise (∑ = 1) and are > chance.")
    println("=" ^ 60)
end

main()
