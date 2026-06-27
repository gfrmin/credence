# Credence — Core-library engine arc: *measure-as-view* (master plan)

> Durable, in-repo master plan for the `measure-as-view` branch family. A thematic engine arc
> (unnumbered, like `collapse-towers` and `decouple`), opened 2026-06-27 *after* collapse-towers
> closed. Each phase lands design-doc-before-code (template at
> `docs/measure-as-view/DESIGN-DOC-TEMPLATE.md`), each commit green + bisectable, **stop-and-report
> at every phase boundary.** Origin: the duplication audit triggered during collapse-towers Phase 2,
> recorded in `docs/collapse-towers/master-plan.md` ("Follow-on arc"), now promoted to its own arc.

## Context — the engine drifts from "Measure is a declared view over Prevision"

The constitution makes Prevision the frozen primitive and **Measure a declared view over it**
(precedent `prevision-not-measure`; "Measure binds carrier space, Prevision doesn't"). All 11 Measure
facades hold a `.prevision`, so the ideal is: Measure methods are thin facades delegating to
Prevision-primary logic. Three classes of drift violate that ideal today:

1. **Backwards delegation (Prevision → Measure, inverting the constitution).** The non-conjugate
   fallbacks of `condition`/`_predictive_ll`/`log_predictive` for the scalar conjugate families and
   `Product` delegate *up* to the Measure facade: `condition(wrap_in_measure(p), k, obs)` at
   `ontology.jl:1159` (Beta), `:1297`/`:1318` (Gaussian), `:1341` (Gamma); `_predictive_ll(wrap_in_measure(p),…)`
   at `:1585`/`:1601`/`:1604`/`:1607` and `log_predictive` at `:1618`. The constitutionally-primary
   object reaches its own behaviour by round-tripping through its view.

2. **A latent CORRECTNESS bug (highest priority) — the primary path is *less accurate* than its view.
   BETA-ONLY (grounding corrected the memory note's overstatement).** `expect(m::BetaMeasure,
   f::Function)` uses **Gauss-Jacobi quadrature** (`ontology.jl:553`, exact for polynomials up to
   degree 2n−1, ~1e-13). `expect(p::BetaPrevision, f::Function)` uses the **old uniform-grid Riemann
   sum** (`ontology.jl:821`, O(1/n²), ~1e-4) — the very sum the Measure path's comment says it
   "replaces." **Verified live (2026-06-27):** under Beta(2,5), `E[x³]` is exact `0.0476190476` on the
   Measure path but `0.0476045236` (Δ=1.45e-5) on the Prevision path; `sqrt`/`sin(3x)` diverge
   ~1.4–2.0e-4. **Gaussian and Gamma show NO asymmetry** — `expect(::GaussianMeasure,…)` (`:627`) and
   `expect(::GammaMeasure,…)` (`:638`) use the *same* uniform grid as their Previsions (`:832`/`:843`),
   so primary and view **agree exactly** (verified Δ=0.0); they are equally inferior, but that is a
   *shared-inaccuracy* enhancement (add Gauss-Hermite/Gauss-Laguerre), NOT a primary-vs-view
   constitutional bug. The asymmetry (only structured-Functional paths — `Identity`, `CenteredPower`,
   `GeometricTail` — are closed-form and agree on both sides; only the **generic-closure fallback**, the
   path a DSL `lambda` takes, diverges) is **Beta-specific**.

3. **Full structural duplication of the mixture methods.** Mixture `condition`/`prune`/`truncate`/`draw`
   are verbatim `MixtureMeasure ↔ MixturePrevision` twins. (collapse-towers Phase 2 *attempted* the
   `condition(MixtureMeasure)`→facade collapse and **reverted** it: a naive facade drops per-component
   carrier-space context and changes the consumer-visible component type, breaking `wrap_in_measure`/
   consumer code. The dedup is real but must thread the carrier space.)

## The load-bearing constraint (surfaced by Phase 2)

**Measure binds carrier space; Prevision does not.** Any Measure→Prevision delegation must thread the
space where the operation needs it. This splits the arc cleanly:
- **`expect` does NOT use carrier space** (it integrates `f` over the distribution) → inverting its
  delegation is *safe* and is the natural home of the correctness fix. **This is Phase 1.**
- **`condition` DOES bind carrier space** (the posterior lives on a declared carrier) → inverting its
  delegation must thread the space. **Later phases.**

## Phases (design-doc-before-code each; stop-and-report at every boundary)

### Phase 1 — Invert `expect` delegation for the scalar families (fixes the Beta correctness bug)
Make the **Prevision the primary** for generic-closure `expect` and the Measure facade delegate
(`expect(m::BetaMeasure, f) → expect(m.prevision, f)`, ditto Gaussian/Gamma). Two outcomes by family:
- **Beta — fixes item 2.** Relocate the Gauss-Jacobi quadrature (a shared `_gauss_jacobi_expect`
  helper) to be `expect(p::BetaPrevision, f)`-primary; the Measure path delegates to it. The Prevision
  generic-closure path gets *more accurate* (the intended, one-directional behaviour change); the
  Measure path is **bit-preserved** (it routes through the same Gauss-Jacobi it already used).
- **Gaussian/Gamma — structural inversion only, no behaviour change.** Both sides already use the same
  uniform grid, so the Measure facade simply delegates to the (identical) Prevision grid — Prevision-
  primary, **bit-identical** both sides. The accuracy upgrade (Gauss-Hermite / Gauss-Laguerre for the
  generic-closure path) is a **separate, deferred enhancement** — it adds new quadrature, it is not the
  primary-vs-view fix, and Phase 1 does not bundle it.

See `phase-1-design.md`. (Open question there: unify at the Measure's Gauss-Jacobi `n=32` so the Beta
Measure path is bit-preserved, vs the Prevision's old `n=64`.)

### Phase 2 — Invert `condition`/`_predictive_ll` delegation (carrier-space-threading)
Move the non-conjugate `condition`/`_predictive_ll`/`log_predictive` fallbacks for Beta/Gaussian/Gamma/
Product to Prevision-primary, threading the carrier space where the posterior needs it; the Measure
facades become thin delegating views. Capture-before-refactor: posteriors pinned `==` pre-refactor.

### Phase 3 — Collapse the duplicated mixture twins (`condition`/`prune`/`truncate`/`draw`)
The Phase-2-of-collapse-towers deferral. Thread the per-component carrier space so the
`MixtureMeasure` facade can delegate to `MixturePrevision` without dropping space context or changing
the consumer-visible component type. Capture-before-refactor against `test_flat_mixture`/`test_host`/
`test_core` TEST 53 (the tests the naive facade broke).

## Hard constraints (inherited)
Spec-first; design-doc-before-code; stop-and-report at every phase boundary; **no new constitutional
text**; no silent fallbacks; tolerance inside the boolean; no `using Test`; **capture-before-refactor**
on every behaviour-preserving move (canonical values pinned PRE-refactor, asserted `==`). The
correctness fix (Phase 1) is the one *intended* behaviour change — captured as an explicit
before/after, not a silent drift.

## Verification (per phase, from repo root; Julia tests not CI-gated)
- Phase 1: `julia test/test_measure_view_expect.jl && julia test/test_core.jl && julia test/test_prevision_unit.jl`
- Each boundary: full `test/test_*.jl` suite green; lint corpus + `check apps/`; skin smoke (Phases 2/3
  touch `condition`, the wire-relevant path); stop and report.

## Key risks
1. **Phase 1 `n`-default drift.** The Measure path uses Gauss-Jacobi `n=32`; the Prevision path used a
   uniform grid `n=64`. Unify at the Measure's `n=32` so the Measure path is **bit-preserved** (the
   correct reference); the Prevision path changes (the fix). Decide in the design doc.
2. **Phase 2/3 carrier-space loss.** The naive facade dropped space context (Phase-2 collapse-towers
   evidence). Every delegation threads the space; capture-before-refactor guards it.
3. **Generic-closure vs structured-Functional.** Only the generic-closure path is wrong; structured
   Functionals agree exactly. Tests must cover the generic-closure path specifically.
