# test_qa_benchmark_category_update.jl — Paper 1, B2c.
#
# Tests the reliability-update mechanism under inferred category
# uncertainty (`apps/julia/qa_benchmark/category_update.jl`):
# `update_reliability` (full-posterior-weighted condition) and
# `marginal_reliability` (category-marginalised MixturePrevision).
# Provenance: `docs/paper1/move-2c-design.md`.
#
# Key guards: (1) one-hot π reduces the update EXACTLY (`==`) to today's
# unit-count learning; (2) the weighted update tracks the closed-form
# exact mixture (the negligible-loss / resource-rational claim); (3) the
# one-hot marginal expect collapses to the single-category belief.
#
# Run from the repo root:
#     julia test/test_qa_benchmark_category_update.jl

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
        error("category_update assertion failed: $name")
    end
end

# Clean Beta mean via the BetaMeasure view (no raw field read).
betamean(p::BetaPrevision) = mean(wrap_in_measure(p))

println("="^60)
println("Paper 1 B2c — reliability update under category uncertainty")
println("="^60)

# ── Test 1: one-hot π reduces the update to today's unit-count learning ──
let
    row = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0), BetaPrevision(4.0, 1.0)]
    out = update_reliability(row, [1.0, 0.0, 0.0], 1)   # correct; category 1 certain
    check("one-hot correct: cat1 Beta(2,3) → Beta(3,3) (==)",
          out[1].alpha == 3.0 && out[1].beta == 3.0, "got ($(out[1].alpha),$(out[1].beta))")
    check("one-hot: zero-weight categories untouched (==)",
          out[2].alpha == 1.0 && out[2].beta == 1.0 &&
          out[3].alpha == 4.0 && out[3].beta == 1.0)

    outw = update_reliability(row, [1.0, 0.0, 0.0], 0)  # wrong; category 1 certain
    check("one-hot wrong: cat1 Beta(2,3) → Beta(2,4) (==)",
          outw[1].alpha == 2.0 && outw[1].beta == 4.0, "got ($(outw[1].alpha),$(outw[1].beta))")
end

# ── Test 2: fractional updates use the whole posterior ──
let
    row = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0)]
    c = update_reliability(row, [0.7, 0.3], 1)          # correct
    check("fractional correct: cat1 α 2→2.7, cat2 α 1→1.3 (rtol 1e-12)",
          isapprox(c[1].alpha, 2.7; rtol=1e-12) && c[1].beta == 3.0 &&
          isapprox(c[2].alpha, 1.3; rtol=1e-12) && c[2].beta == 1.0,
          "got ($(c[1].alpha),$(c[1].beta)),($(c[2].alpha),$(c[2].beta))")
    w = update_reliability(row, [0.7, 0.3], 0)          # wrong
    check("fractional wrong: cat1 β 3→3.7, cat2 β 1→1.3 (rtol 1e-12)",
          w[1].alpha == 2.0 && isapprox(w[1].beta, 3.7; rtol=1e-12) &&
          w[2].alpha == 1.0 && isapprox(w[2].beta, 1.3; rtol=1e-12),
          "got ($(w[1].alpha),$(w[1].beta)),($(w[2].alpha),$(w[2].beta))")
end

# ── Test 3: the weighted update tracks the exact mixture (negligible loss) ──
# Design-doc §6 oracle: row=[Beta(2,3),Beta(1,1)], π=[0.7,0.3], observe
# correct. Exact posterior marginal mean of θ₁:
#   component c=1 weight ∝ 0.7·(2/5)=0.28 → θ₁~Beta(3,3), mean 0.5
#   component c=2 weight ∝ 0.3·(1/2)=0.15 → θ₁~Beta(2,3), mean 0.4
let
    row = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0)]
    out = update_reliability(row, [0.7, 0.3], 1)
    weighted_mean = betamean(out[1])                    # mean(Beta(2.7,3)) = 0.4737
    w1 = 0.28 / (0.28 + 0.15)
    exact_mean = w1 * 0.5 + (1 - w1) * 0.4              # = 0.4651 (closed-form oracle)
    check("weighted update tracks exact-mixture mean (|Δ| < 0.02)",
          abs(weighted_mean - exact_mean) < 0.02,
          "weighted=$weighted_mean exact=$exact_mean Δ=$(abs(weighted_mean - exact_mean))")
end

# ── Test 4: determinism ──
let
    row = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0)]
    a = update_reliability(row, [0.6, 0.4], 1)
    b = update_reliability(row, [0.6, 0.4], 1)
    check("determinism: identical (α,β) across calls (==)",
          all(i -> a[i].alpha == b[i].alpha && a[i].beta == b[i].beta, eachindex(a)))
end

# ── Test 5: marginal_reliability — one-hot collapses, general = Σ π_c E[θ_c] ──
# NOTE: expect over a MixturePrevision uses a coarser quadrature than
# expect over a bare Beta, so the one-hot marginal is NOT bit-identical to
# the single-category expect (≈5e-5 apart) — only the *update* side is
# bit-exact on one-hot. We assert against the analytic per-category means
# (Beta(2,3)=0.4, Beta(1,1)=0.5) at a decision-irrelevant quadrature
# tolerance. (Recorded as the design-doc §7 residual finding.)
let
    row = [BetaPrevision(2.0, 3.0), BetaPrevision(1.0, 1.0)]
    f = r -> r                                          # identity → mean

    m1 = marginal_reliability(row, [1.0, 0.0])          # collapses to category 1
    check("one-hot marginal expect ≈ category-1 mean 0.4 (atol 1e-3)",
          isapprox(expect(m1, f), 0.4; atol=1e-3), "got $(expect(m1, f))")

    m = marginal_reliability(row, [0.7, 0.3])           # Σ π_c·mean_c = 0.7·0.4+0.3·0.5
    check("marginal expect ≈ Σ π_c·mean_c = 0.43 (atol 1e-3)",
          isapprox(expect(m, f), 0.43; atol=1e-3), "got $(expect(m, f))")
end

println("="^60)
println("ALL $(_PASS[]) CHECKS PASSED (B2c update mechanism)")
println("="^60)
