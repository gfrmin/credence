# Phase 1 design doc — Invert `expect` delegation for the scalar families (fix the Beta correctness bug)

> Seven-section template (`docs/measure-as-view/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/measure-as-view/master-plan.md`. Precedents: `docs/precedents.md`.

## 1. Purpose

Phase 1 as scoped in the master plan: make the **Prevision the primary** for generic-closure `expect`
on the scalar conjugate families and reduce the Measure to a thin delegating view, restoring the
constitutional direction (`prevision-not-measure`). The move is carrier-space-safe *because* `expect`
integrates `f` over the distribution and never reads the carrier — unlike `condition` (deferred to
Phase 2). Two outcomes, by family:

- **Beta — the correctness fix.** Today `expect(m::BetaMeasure, f::Function)` uses Gauss-Jacobi
  quadrature (`ontology.jl:553`, ~1e-13) while `expect(p::BetaPrevision, f::Function)` uses an inferior
  uniform grid (`ontology.jl:821`, ~1e-4) — so the **primary path is silently less accurate than its
  view** (verified: Beta(2,5) `E[x³]` exact `0.0476190476` on the Measure path, `0.0476045236` on the
  Prevision path). Relocating Gauss-Jacobi to be Prevision-primary fixes it: the Prevision path becomes
  accurate (the intended, one-directional change); the Measure path is bit-preserved.
- **Gaussian/Gamma — structural inversion only, zero behaviour change.** Both sides already share the
  *same* uniform grid (`:627`==`:832`, `:638`==`:843`; verified Δ=0.0), so the Measure facade simply
  delegates to the identical Prevision grid — Prevision-primary, bit-identical. The Gauss-Hermite /
  Gauss-Laguerre accuracy upgrade for their shared grid is a **separate deferred enhancement**, NOT
  bundled here (it adds new quadrature; it is not the primary-vs-view fix).

## 2. Files touched

- **`src/ontology.jl`** — *modify*:
  - **Extract** `_gauss_jacobi_expect(alpha::Float64, beta::Float64, f, n::Int)::Float64` from the body
    of `expect(m::BetaMeasure, f::Function)` (`:553`–`:625`, the hand-rolled Golub–Welsch rule). No
    arithmetic change — a pure extraction.
  - `expect(p::BetaPrevision, f::Function; n::Int = 32)` — **replace** the uniform-grid body (`:821`)
    with `_gauss_jacobi_expect(p.alpha, p.beta, f, n)`. (Default `n` 64→32; see §5.)
  - `expect(m::BetaMeasure, f::Function; n::Int = 32)` — **replace** the Gauss-Jacobi body with the
    delegation `expect(m.prevision, f; n = n)`. `m.prevision::BetaPrevision` (accessor at `:174`).
  - `expect(m::GaussianMeasure, f::Function; n::Int = 64) = expect(m.prevision, f; n = n)` — delegate
    (was the identical grid at `:627`). `expect(p::GaussianPrevision, …)` (`:832`) is unchanged (now the
    primary).
  - `expect(m::GammaMeasure, f::Function; n::Int = 64) = expect(m.prevision, f; n = n)` — delegate (was
    the identical grid at `:638`). `expect(p::GammaPrevision, …)` (`:843`) unchanged (now primary).
  - The `TaggedBetaMeasure`/`TaggedBetaPrevision` generic-`f` methods (`:625`/`:830`) already delegate to
    the inner beta — **no change**; they inherit the fix.
- **`test/test_measure_view_expect.jl`** — *new* (see §7).

No new exports, no signature changes visible to consumers (the `n` kwargs keep defaults). `condition`,
`_predictive_ll`, the structured-Functional `expect` methods — **untouched** (Phase 2+).

## 3. Behaviour preserved

- **Beta Measure path — bit-preserved.** `expect(m::BetaMeasure, f; n=32)` delegates to
  `expect(m.prevision, f; n=32)`, which is `_gauss_jacobi_expect(α, β, f, 32)` — the *same* rule, same
  `n`, the body extracted verbatim. Strata-1 equivalence: `==` against pre-refactor captured values.
- **Gaussian/Gamma — bit-identical both sides.** The Measure body was character-identical to the
  Prevision body (modulo `m.`/`p.` field access); delegation routes the Measure through the Prevision
  grid. Strata-1: `expect(p, f) == expect(wrap_in_measure(p), f)` holds at `Δ=0.0` (already true; stays).
- **Structured Functionals unchanged.** `Identity`/`CenteredPower`/`GeometricTail`/`LinearCombination`
  paths are not touched; their closed forms already agree on both sides.
- **The intended change (NOT preserved, by design):** `expect(p::BetaPrevision, f::Function)` moves
  from uniform-grid (~1e-4) to Gauss-Jacobi (~1e-13). Captured as an explicit before/after, not silent.

## 4. Worked end-to-end example

`expect(p, f)` with `p = BetaPrevision(2.0, 5.0)`, `f = x -> x^3`:
- **Before:** `expect(p, f)` → uniform-grid body (`:821`): 64-point midpoint grid, Riemann weights →
  `0.0476045236` (Δ=1.45e-5 from the exact `24/504 = 0.047619047…`).
- **After:** `expect(p, f)` → `_gauss_jacobi_expect(2.0, 5.0, f, 32)` *(owner: the extracted helper)* →
  the 32-node Gauss-Jacobi rule integrates the degree-3 polynomial **exactly** → `0.0476190476`.
- **Measure view, after:** `expect(wrap_in_measure(p), f)` → `expect(m::BetaMeasure, f; n=32)` →
  delegates → `expect(m.prevision, f; n=32)` → same `_gauss_jacobi_expect(2.0, 5.0, f, 32)` →
  `0.0476190476`. Primary and view now **agree, and both are exact**.
- Dual residency: the Gauss-Jacobi rule had a single home (Measure); Phase 1 moves that home to the
  Prevision and makes the Measure a one-line delegation — net dual-residency *decreases*.

## 5. Open design questions

1. **Beta Prevision default `n`: 32 (recommended) vs 64?** The Measure path uses `n=32`; to
   **bit-preserve** it under delegation, the unified default must be `32`. Gauss-Jacobi `n=32` is exact
   for polynomials up to degree 63 — strictly better than the old uniform `n=64` for every `f` — so
   dropping the Prevision default from 64 to 32 loses nothing and keeps the Measure reference intact.
   Recommend **32**. (Alternative: unify at 64 and re-capture the Measure reference — rejected, it
   perturbs a correct path to no benefit.)
2. **Scope: does Phase 1 also upgrade Gaussian/Gamma to Gauss-Hermite/Laguerre?** Recommend **no** —
   keep Phase 1 to the *primary-vs-view* fix (Beta) plus the *zero-behaviour-change* structural
   inversion (Gaussian/Gamma). Their shared grid inaccuracy is real but symmetric (no constitutional
   violation), and adding Hermite/Laguerre is new numerics that deserves its own move with its own
   exactness tests. Fold it in only if you'd rather not leave a known ~1e-4 on those generic-closure
   paths.
3. **Is `expect` truly carrier-space-free for these Measures?** The premise of the safe inversion. The
   bodies read only `α,β` / `μ,σ` — never `m.space`. Confirm by grep (§6); if any scalar `expect` reads
   the carrier, that family stays Measure-primary and is flagged.

## 6. Risk + mitigation

- **A caller passes `n` to `expect(BetaPrevision, …)` and relies on `n=64`.** *Pre-emptive grep:*
  `grep -rn 'expect(.*; *n *=' src/ apps/ test/` and `grep -rn 'expect(.*BetaPrevision' …` — list each
  hit's disposition. The kwarg default change only affects callers that *omit* `n` (they get the more
  accurate rule); explicit `n=…` callers are unaffected.
- **`m.prevision` does not round-trip the Measure's parameters exactly.** *Failure mode:*
  `expect(m::GaussianMeasure, f)` delegating to `m.prevision` would change the value if
  `m.prevision` reconstructs `μ,σ` lossily. *Mitigation:* the accessors are field reads
  (`:237`/`:322`); a test asserts `expect(wrap_in_measure(p), f) == expect(p, f)` for Gaussian/Gamma
  (Δ=0.0) post-refactor, and the Beta Measure path `==` its captured pre-refactor value.
- **Capture-before-refactor.** Pin, PRE-refactor: (a) `expect(BetaMeasure(α,β), f)` for a grid of
  `(α,β)` × generic `f` (the Gauss-Jacobi reference, to assert the Measure path is bit-preserved); (b)
  the *old* `expect(BetaPrevision, f)` uniform-grid values (to assert the Prevision path *changed* and
  moved *toward* the exact closed form). Both captured as literals in the test.
- **Pre-emptive grep for carrier-space reads:** `grep -n 'm.space' ontology.jl` within the scalar
  `expect` bodies — expected empty; if not, that family is excluded from the inversion.

## 7. Verification cadence

End of Phase-1 code (from repo root):
```
julia test/test_measure_view_expect.jl
julia test/test_core.jl
julia test/test_prevision_unit.jl
julia test/test_centered_moment.jl
```
Then the full `test/test_*.jl` suite + lint corpus self-test + `check apps/`, and **stop and report**.
Skin smoke is **optional** for Phase 1 (`expect` is not the wire-crossing path; `condition` is — that
is Phase 2).

`test_measure_view_expect.jl` (repo `check(name, cond, detail)` idiom):
- **Beta correctness (the fix):** for `(α,β) ∈ {(2,5),(0.5,0.5),(3,3)}` and `f ∈ {x³, √x, sin 3x}`,
  `expect(BetaPrevision(α,β), f) ≈ expect(wrap_in_measure(p), f)` to `rtol=1e-12` (both Gauss-Jacobi
  now), AND for the polynomial `f=x^k` the value equals the **exact** closed form
  `∏_{i=0}^{k-1}(α+i)/(α+β+i)` to `rtol=1e-12` (the strongest property — the old uniform grid failed
  this at ~1e-4).
- **Beta Measure bit-preservation:** `expect(BetaMeasure(α,β), f; n=32)` `==` the captured pre-refactor
  literal (Gauss-Jacobi unchanged).
- **Gaussian/Gamma structural inversion:** `expect(p, f) == expect(wrap_in_measure(p), f)` (Δ=0.0) for
  Gaussian/Gamma — bit-identical, primary and view, pre- and post-refactor.
- **TaggedBeta inherits the fix:** `expect(TaggedBetaPrevision(tag, Beta(α,β)), f)` matches the exact
  closed form (it routes through the fixed BetaPrevision path).

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red.
