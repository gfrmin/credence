# test_qa_benchmark_post_credit.jl — issue #111.
#
# Tests the posterior-weighted reliability update (the deployed default)
# in `apps/julia/qa_benchmark/category_update.jl`: `post_update` and its
# `posterior_credit_weights` (the exact one-step category posterior ρ).
#
# `post` credits each category by ρ_c ∝ π_c·ℓ_{t,c}(outcome) instead of the
# classifier prior π_c (`soft`/B2c). Both ρ and the per-category Beta
# updates go through `condition` — no host-side Bayes.
#
# Key guards: (1) ρ via `condition` equals the closed-form π·ℓ/normaliser;
# (2) one-hot π reduces `post` EXACTLY (`==`) to the soft / unit-count
# update; (3) `post` lands STRICTLY closer to the exact 2-component mixture
# than `soft` (the better-projection claim — the whole point of #111);
# (4) the fractional (α,β) are exact; (5) determinism; (6) a zero-prior
# category (−Inf log-weight) is handled and stays zero.
#
# Run from the repo root:
#     julia test/test_qa_benchmark_post_credit.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision

include(joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "category_update.jl"))

const _PASS = Ref(0)
function check(name, cond, detail="")
    if cond
        _PASS[] += 1
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("post_credit assertion failed: $name")
    end
end

# Clean Beta mean via the BetaMeasure view (no raw field read).
betamean(p::BetaPrevision) = mean(wrap_in_measure(p))

println("="^60)
println("Issue #111 — posterior-weighted credit (post_update)")
println("="^60)

# Shared oracle case (matches test_qa_benchmark_category_update.jl Test 3):
# row = [Beta(2,3), Beta(1,1)], π = [0.7, 0.3], observe CORRECT (o=1).
# Predictive likelihoods ℓ_c(1) = mean(row[c]) = [0.4, 0.5].
# ρ ∝ π·ℓ = [0.7·0.4, 0.3·0.5] = [0.28, 0.15]  →  ρ = [0.28,0.15]/0.43.
const ROW = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0)]
const Π   = [0.7, 0.3]
const ΡEXACT = [0.28, 0.15] ./ 0.43

# ── Test 1: ρ via `condition` equals the closed-form category posterior ──
let
    ρ = posterior_credit_weights(ROW, Π, 1)
    check("ρ (correct) == π·ℓ/normaliser, ℓ=mean (rtol 1e-12)",
          isapprox(ρ, ΡEXACT; rtol=1e-12), "got $ρ vs $ΡEXACT")
    check("ρ sums to 1 (rtol 1e-12)", isapprox(sum(ρ), 1.0; rtol=1e-12), "got $(sum(ρ))")

    # Wrong outcome: ℓ_c(0) = 1 - mean = [0.6, 0.5]; ρ ∝ [0.42, 0.15].
    ρw = posterior_credit_weights(ROW, Π, 0)
    check("ρ (wrong) == π·(1-mean)/normaliser (rtol 1e-12)",
          isapprox(ρw, [0.42, 0.15] ./ 0.57; rtol=1e-12), "got $ρw")
end

# ── Test 2: one-hot π reduces post EXACTLY to soft / unit-count update ──
let
    for (oh, o) in (([1.0, 0.0], 1), ([1.0, 0.0], 0), ([0.0, 1.0], 1))
        p = post_update(ROW, oh, o)
        s = update_reliability(ROW, oh, o)
        check("one-hot $oh o=$o: post == soft (==)",
              all(i -> p[i].alpha == s[i].alpha && p[i].beta == s[i].beta, eachindex(p)),
              "post=$(p) soft=$(s)")
    end
    # And that one-hot is the literal unit-count update on the certain category.
    p = post_update(ROW, [1.0, 0.0], 1)
    check("one-hot correct: cat1 Beta(2,3)→Beta(3,3), cat2 untouched (==)",
          p[1].alpha == 3.0 && p[1].beta == 3.0 &&
          p[2].alpha == 1.0 && p[2].beta == 1.0, "got $p")
end

# ── Test 3: post is STRICTLY closer to the exact mixture than soft ──
# Exact posterior marginal mean of θ₁ after the correct observation (the
# 2-component mixture ρ collapses): component c=1 (weight ρ₁) → Beta(3,3),
# mean 0.5; component c=2 (weight ρ₂) → Beta(2,3), mean 0.4.
let
    exact = ΡEXACT[1] * 0.5 + ΡEXACT[2] * 0.4            # 0.46512 closed-form
    post = post_update(ROW, Π, 1)
    soft = update_reliability(ROW, Π, 1)
    dpost = abs(betamean(post[1]) - exact)
    dsoft = abs(betamean(soft[1]) - exact)
    check("post mean θ₁ closer to exact mixture than soft (strict)",
          dpost < dsoft, "dpost=$dpost dsoft=$dsoft exact=$exact")
end

# ── Test 4: exact fractional (α,β) — post credits with ρ, not π ──
let
    post = post_update(ROW, Π, 1)                        # correct → α gets ρ
    check("post correct: cat1 α 2→2+ρ₁, cat2 α 1→1+ρ₂ (rtol 1e-12)",
          isapprox(post[1].alpha, 2.0 + ΡEXACT[1]; rtol=1e-12) && post[1].beta == 3.0 &&
          isapprox(post[2].alpha, 1.0 + ΡEXACT[2]; rtol=1e-12) && post[2].beta == 1.0,
          "got ($(post[1].alpha),$(post[1].beta)),($(post[2].alpha),$(post[2].beta))")

    ρw = [0.42, 0.15] ./ 0.57
    postw = post_update(ROW, Π, 0)                       # wrong → β gets ρ
    check("post wrong: cat1 β 3→3+ρ₁, cat2 β 1→1+ρ₂ (rtol 1e-12)",
          postw[1].alpha == 2.0 && isapprox(postw[1].beta, 3.0 + ρw[1]; rtol=1e-12) &&
          postw[2].alpha == 1.0 && isapprox(postw[2].beta, 1.0 + ρw[2]; rtol=1e-12),
          "got ($(postw[1].alpha),$(postw[1].beta)),($(postw[2].alpha),$(postw[2].beta))")

    # Definitional consistency: post == update_reliability(row, ρ, o).
    ρ = posterior_credit_weights(ROW, Π, 1)
    viaρ = update_reliability(ROW, ρ, 1)
    check("post_update ≡ update_reliability(row, ρ, o) (==)",
          all(i -> post[i].alpha == viaρ[i].alpha && post[i].beta == viaρ[i].beta, eachindex(post)))
end

# ── Test 5: determinism ──
let
    a = post_update(ROW, [0.6, 0.4], 1)
    b = post_update(ROW, [0.6, 0.4], 1)
    check("determinism: identical (α,β) across calls (==)",
          all(i -> a[i].alpha == b[i].alpha && a[i].beta == b[i].beta, eachindex(a)))
end

# ── Test 6: a zero-prior category (−Inf log-weight) stays zero ──
let
    row3 = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0), BetaPrevision(4.0, 1.0)]
    ρ = posterior_credit_weights(row3, [0.7, 0.0, 0.3], 1)
    check("zero-prior category gets ρ=0 (==)", ρ[2] == 0.0, "got ρ=$ρ")
    check("ρ still normalised over the surviving categories (rtol 1e-12)",
          isapprox(sum(ρ), 1.0; rtol=1e-12), "got $(sum(ρ))")
    p = post_update(row3, [0.7, 0.0, 0.3], 1)
    check("zero-prior category Beta untouched (==)",
          p[2].alpha == 1.0 && p[2].beta == 1.0, "got $(p[2])")
end

println("="^60)
println("ALL $(_PASS[]) CHECKS PASSED (post-credit / issue #111)")
println("="^60)
