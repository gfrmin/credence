# test_measure_view_expect.jl — Phase 1 of the measure-as-view arc.
#
# Inverts generic-closure `expect` delegation for the scalar families to Prevision-primary, fixing the
# BETA correctness asymmetry: the constitutionally-primary `BetaPrevision` path used an inferior uniform
# grid (~1e-4) while the `BetaMeasure` *view* used Gauss-Jacobi (~1e-13) — the view was more accurate
# than the primary. Post-Phase-1: Gauss-Jacobi is Prevision-primary; the Measure path delegates
# (bit-preserved); Gaussian/Gamma invert with ZERO behaviour change (both sides already shared the
# uniform grid). The Gaussian/Gamma accuracy upgrade (Hermite/Laguerre) is deferred (a separate move).
#
# Capture-before-refactor: the Measure Gauss-Jacobi values are pinned as literals (the bit-preserve
# target); the OLD Prevision uniform-grid values are pinned to prove the Prevision path CHANGED toward
# exact. Literals captured pre-refactor via scratchpad/capture.jl.
#
# Run from repo root:
#     julia test/test_measure_view_expect.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: BetaPrevision, GaussianPrevision, GammaPrevision, TaggedBetaPrevision,
                expect, wrap_in_measure

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Exact Beta moment E[x^k] = ∏_{i=0}^{k-1} (α+i)/(α+β+i).
beta_moment(α, β, k) = prod((α + i) / (α + β + i) for i in 0:k-1)   # credence-lint: allow — precedent:test-oracle — independent closed form for the GJ rule

println("="^64)
println("measure-as-view Phase 1 — expect inversion (Beta correctness fix)")
println("="^64)

# ── (1) BETA — the fix: Prevision now equals Measure (delegation), and matches the EXACT moment ──
# Beta(0.5,0.5) is the α+β=1 REGRESSION GUARD: the Gauss-Jacobi node recurrence hit a removable
# 0/0 there (NaN → LAPACK's MRRR eigensolver infinite-loops — a hang, not a wrong number). The
# cancelled-k=1 form in _gauss_jacobi_expect fixes it; reaching the finite, exact E[x³]=0.3125
# check below at all is the guard (a hang would never return).
for (α, β) in ((2.0, 5.0), (0.5, 0.5), (3.0, 3.0))
    p = BetaPrevision(α, β); m = wrap_in_measure(p)
    for (nm, f) in (("x^3", x -> x^3), ("sqrt", sqrt), ("sin3x", x -> sin(3x)))
        check("Beta($α,$β) $nm: Prevision == Measure (both Gauss-Jacobi via delegation)",
              expect(p, f) == expect(m, f), "p=$(expect(p,f)) m=$(expect(m,f))")
    end
    # The property the old uniform grid FAILED (~1e-4): GJ is exact for the degree-3 polynomial.
    check("Beta($α,$β): expect(Prevision, x^3) == exact moment",
          isapprox(expect(p, x -> x^3), beta_moment(α, β, 3); rtol = 1e-12),
          "got $(expect(p, x->x^3)) exact $(beta_moment(α,β,3))")
end

# ── (2) capture-before-refactor: the Measure path is BIT-PRESERVED (still Gauss-Jacobi n=32) ──
check("Beta(2,5) sqrt  Measure bit-preserved",
      expect(wrap_in_measure(BetaPrevision(2.0, 5.0)), sqrt) == 0.51148859847560213)
check("Beta(2,5) sin3x Measure bit-preserved",
      expect(wrap_in_measure(BetaPrevision(2.0, 5.0)), x -> sin(3x)) == 0.66723706676609618)
check("Beta(3,3) sqrt  Measure bit-preserved",
      expect(wrap_in_measure(BetaPrevision(3.0, 3.0)), sqrt) == 0.69264069302310205)

# ── (3) the Prevision path CHANGED — it no longer equals the OLD uniform-grid value ──
check("Beta(2,5) sqrt Prevision moved off the old uniform value (the fix landed)",
      expect(BetaPrevision(2.0, 5.0), sqrt) != 0.51134797934176113)

# ── (4) GAUSSIAN/GAMMA — zero-behaviour-change inversion: Prevision === Measure, bit-identical ──
for (nm, p) in (("Gaussian(1,2)", GaussianPrevision(1.0, 2.0)), ("Gamma(3,2)", GammaPrevision(3.0, 2.0)))
    for (fn, f) in (("x^3", x -> x^3), ("smooth", x -> exp(-x^2 / 9)))
        check("$nm $fn: Prevision === Measure (bit-identical inversion)",
              expect(p, f) === expect(wrap_in_measure(p), f))
    end
end
check("Gaussian(1,2) x^3 unchanged (captured)", expect(GaussianPrevision(1.0, 2.0), x -> x^3) == 12.989986878098938)
check("Gamma(3,2) x^3 unchanged (captured)", expect(GammaPrevision(3.0, 2.0), x -> x^3) == 7.4431782434717517)

# ── (5) TaggedBeta inherits the Beta fix (it routes through the inner BetaPrevision) ──
check("TaggedBetaPrevision inherits the Gauss-Jacobi fix",
      isapprox(expect(TaggedBetaPrevision(1, BetaPrevision(2.0, 5.0)), x -> x^3), beta_moment(2.0, 5.0, 3); rtol = 1e-12))

println("="^64)
println("ALL CHECKS PASSED — measure-as-view Phase 1")
println("="^64)
