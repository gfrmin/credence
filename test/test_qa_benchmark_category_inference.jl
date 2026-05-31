# test_qa_benchmark_category_inference.jl — Paper 1, Phase B2b.
#
# Tests the body-side Gaussian Naive Bayes category classifier
# (`apps/julia/qa_benchmark/category_inference.jl`). Provenance:
# `docs/paper1/move-2-design.md` §3.1 (the (b-NB) reduction) and
# `docs/paper1/master-plan.md` §4a (OQ1(c)/OQ2(b-NB)/OQ3(c)/OQ5 + the
# (γ) assumption).
#
# Tripwire discipline (mirrors test/test_prevision_conjugate.jl): the
# conjugate dispatch path is asserted BEFORE the value assertions. A
# silent registry miss — NormalGamma or Dirichlet calibration falling
# through to the particle fallback instead of the conjugate update —
# would otherwise pass the value assertion for the wrong reason.
#
# Run from the repo root:
#     julia test/test_qa_benchmark_category_inference.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: DirichletPrevision, NormalGammaPrevision, _dispatch_path
using Random

include(joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "category_inference.jl"))

const _PASS = Ref(0)
function check(name, cond, detail="")
    if cond
        _PASS[] += 1
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("category_inference assertion failed: $name")
    end
end

# Synthetic well-separated categories: fixed class means under means_seed
# (shared across train/test draws); per-point Gaussian noise under
# noise_seed. Mirrors the (b-NB) prototype's data model.
function synth_dataset(cats::Vector{Symbol}, n_per_cat::Int;
                       D::Int=8, sep::Float64=3.0, noise::Float64=1.0,
                       means_seed::Int=7, noise_seed::Int=42)
    mrng = MersenneTwister(means_seed)
    means = [sep .* randn(mrng, D) for _ in cats]
    nrng = MersenneTwister(noise_seed)
    N = length(cats) * n_per_cat
    X = Matrix{Float64}(undef, N, D)
    y = Vector{Symbol}(undef, N)
    i = 1
    for (k, c) in enumerate(cats), _ in 1:n_per_cat
        X[i, :] = means[k] .+ noise .* randn(nrng, D)
        y[i] = c
        i += 1
    end
    X, y
end

argmax_cat(post::Dict{Symbol,Float64}) = findmax(post)[2]

println("="^60)
println("Paper 1 B2b — Gaussian-NB category inference")
println("="^60)

# ── Test 1: fit-and-infer roundtrip on well-separated data ──
let
    cats = [:alpha, :beta, :gamma]
    Xtr, ytr = synth_dataset(cats, 8; noise_seed=42)
    Xte, yte = synth_dataset(cats, 4; noise_seed=123)   # same means, fresh noise
    clf = fit(Xtr, ytr)

    # Tripwire FIRST: the calibration kernels must hit the conjugate path.
    check("NormalGamma calibration → :conjugate",
          _dispatch_path(NormalGammaPrevision(1.0, 0.0, 2.0, 2.0), SCALAR_OBS_KERNEL) === :conjugate,
          "got $(_dispatch_path(NormalGammaPrevision(1.0, 0.0, 2.0, 2.0), SCALAR_OBS_KERNEL))")
    check("Dirichlet calibration → :conjugate",
          _dispatch_path(DirichletPrevision(ones(length(cats))), _class_obs_kernel(sort(unique(ytr)))) === :conjugate,
          "got $(_dispatch_path(DirichletPrevision(ones(length(cats))), _class_obs_kernel(sort(unique(ytr)))))")

    correct = 0
    for i in 1:size(Xte, 1)
        post = infer(clf, Xte[i, :])
        argmax_cat(post) == yte[i] && (correct += 1)
    end
    check("fit/infer recovers ≥80% of held-out labels on separated data",
          correct >= 0.8 * size(Xte, 1), "got $correct/$(size(Xte, 1))")
end

# ── Test 2: posterior normalises ──
let
    cats = [:alpha, :beta, :gamma]
    Xtr, ytr = synth_dataset(cats, 6; noise_seed=11)
    clf = fit(Xtr, ytr)
    post = infer(clf, Xtr[1, :])
    check("posterior sums to 1.0 (atol 1e-10)",
          abs(sum(values(post)) - 1.0) < 1e-10, "sum=$(sum(values(post)))")
    check("posterior has one entry per training category",
          length(post) == length(cats), "len=$(length(post))")
end

# ── Test 3: determinism ──
let
    cats = [:alpha, :beta, :gamma]
    Xtr, ytr = synth_dataset(cats, 6; noise_seed=11)
    e = Xtr[1, :]
    p1 = infer(fit(Xtr, ytr), e)
    p2 = infer(fit(Xtr, ytr), e)
    check("identical inputs → identical posteriors (Dict ==)", p1 == p2, "p1=$p1 p2=$p2")
end

# ── Test 4: empty training set raises a clear, specific error ──
let
    threw = false
    msg = ""
    try
        fit(Matrix{Float64}(undef, 0, 8), Symbol[])
    catch err
        threw = true
        msg = sprint(showerror, err)
    end
    check("fit on empty training set throws", threw)
    check("empty-set error names the empty training set",
          occursin("empty training set", msg), "msg=$msg")
end

# ── Test 5: single-class training data → degenerate-but-valid posterior ──
let
    X, y = synth_dataset([:only], 10; noise_seed=5)
    clf = fit(X, y)
    post = infer(clf, X[1, :])
    check("single-class posterior P(only) = 1.0 (atol 1e-12)",
          isapprox(post[:only], 1.0; atol=1e-12), "got $(post[:only])")
    check("single-class posterior has exactly one category",
          length(post) == 1, "len=$(length(post))")
end

println("="^60)
println("ALL $(_PASS[]) CHECKS PASSED (B2b.2)")
println("="^60)
