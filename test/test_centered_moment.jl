# test_centered_moment.jl — the exact Beta moment (CenteredPower closed form), the engine primitive
# the life-agent body decouple needs (decouple Move 1). Self-contained (no apps/ dependency).
# Asserts:
#   (1) raw moment E[θ^n] = CenteredPower{n}(0) on Beta(α,β) equals the exact rising-factorial
#       form (n=2 → α(α+1)/((α+β)(α+β+1))); the Prevision path equals the Measure path;
#   (2) central second moment CenteredPower{2}(mean) equals Var(Beta) exactly;
#   (3) the integrated claim-inclusion EU LinearCombination([(u_c−u_w, CP2(0)), (u_w, Id)], −κ)
#       = E_θ[θ·u_assert(θ)]−κ exactly, and that it differs from the OLD body's point-estimate
#       p̄² by exactly Var·(u_c−u_w) — the variance term the exact integral keeps (NOT a port);
#       and the decision via expect flips withhold↔include across the reliability bar.
#
# (The grid-coupled `marginalise(state, shape, axis)` verb this file once pinned was retired in
#  Phase B — coupled folds moved engine-side to `marginal(::MvQuadraturePrevision, i)`; see
#  test_mv_quadrature.jl.)
#
# Run from repo root:  julia test/test_centered_moment.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaMeasure, BetaPrevision, CategoricalMeasure, Finite, CenteredPower,
                CenteredSquare, Identity, LinearCombination, Functional, expect

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end
approx(a, b; atol = 1e-12) = abs(a - b) <= atol

println("="^64)
println("exact Beta moment (CenteredPower) — Move 1 finish")
println("="^64)

# ── (1) raw moments: E[θ^n] = ∏_{i=0}^{n-1}(α+i)/(α+β+i) ──
for (α, β) in [(2.0, 3.0), (20.0, 2.0), (1.0, 1.0), (5.0, 7.0)]
    m = BetaMeasure(α, β)
    e2_closed = α * (α + 1) / ((α + β) * (α + β + 1))
    check("E[θ²] Beta($α,$β) exact", approx(expect(m, CenteredPower{2}(0.0)), e2_closed),
          "$(expect(m, CenteredPower{2}(0.0))) vs $e2_closed")
    check("E[θ²] prevision path == measure path Beta($α,$β)",
          approx(expect(BetaPrevision(α, β), CenteredPower{2}(0.0)), e2_closed))
    check("E[θ¹] == mean Beta($α,$β)", approx(expect(m, CenteredPower{1}(0.0)), α / (α + β)))
    check("E[θ⁰] == 1 Beta($α,$β)", approx(expect(m, CenteredPower{0}(0.0)), 1.0))
end

# ── (2) central second moment == variance ──
for (α, β) in [(2.0, 3.0), (20.0, 2.0), (5.0, 7.0)]
    μ = α / (α + β)
    var_closed = α * β / ((α + β)^2 * (α + β + 1))
    check("Var Beta($α,$β) via CenteredSquare(mean)", approx(expect(BetaMeasure(α, β), CenteredSquare(μ)), var_closed))
end

# ── (3) integrated claim-inclusion EU = E_θ[θ·u_assert(θ)] − κ ──
u_c, u_w, κ = 1.0, -1.0, 0.05
include_fn = LinearCombination([(u_c - u_w, CenteredPower{2}(0.0)), (u_w, Identity())], -κ)
withhold_fn = LinearCombination(Tuple{Float64, Functional}[], 0.0)   # the gauge zero
eu_include_exact(α, β) = begin
    e2 = α * (α + 1) / ((α + β) * (α + β + 1))
    e1 = α / (α + β)
    (u_c - u_w) * e2 + u_w * e1 - κ
end
for (α, β) in [(2.0, 3.0), (20.0, 2.0), (8.0, 4.0)]
    got = expect(BetaMeasure(α, β), include_fn)
    check("include EU exact Beta($α,$β)", approx(got, eu_include_exact(α, β)), "$got vs $(eu_include_exact(α,β))")
end

# The variance term is real, not ported: exact − point-estimate(p̄²) = Var·(u_c−u_w).
# Beta(1.4,0.6): mean 0.7, Var = (1.4·0.6)/(2²·3) = 0.07 → Δ = 0.07·2 = 0.14.
let α = 1.4, β = 0.6
    p = α / (α + β)
    point = (u_c - u_w) * p^2 + u_w * p - κ           # the OLD body's plug-at-mean estimate
    exact = expect(BetaMeasure(α, β), include_fn)
    check("exact integral keeps Var·(u_c−u_w) over the point estimate", approx(exact - point, 0.14),
          "Δ = $(exact - point) (expected 0.14)")
end

# The decision through expect (no host arithmetic): a low-reliability cell withholds, a
# high-reliability cell includes — the optimise{include,withhold} the wire runs.
decide(α, β) = expect(BetaMeasure(α, β), include_fn) > expect(BetaMeasure(α, β), withhold_fn) ? :include : :withhold
check("low-reliability cell withholds", decide(2.0, 3.0) == :withhold, string(eu_include_exact(2.0, 3.0)))
check("high-reliability cell includes", decide(20.0, 2.0) == :include, string(eu_include_exact(20.0, 2.0)))

println("="^64)
println("ALL PASSED")
println("="^64)
