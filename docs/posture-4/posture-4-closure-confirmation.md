# Posture 4 closure confirmation

## Summary

The original Move 8 completion audit (PR #69) reported Posture 4's substrate cleanup complete pending three small gaps. Investigation surfaced a deeper substrate-tightening gap — Move 5's committed `Vector{Prevision}` field types for `MixturePrevision` and `ProductPrevision` never landed — plus two architectural cleanups: `ParticlePrevision` parametricity and `EnumerationPrevision` retirement. Move 8b (design doc PR #71, code PRs #72–#75) closed all three. Posture 4 is now properly complete.

## Merged PRs

| Phase | PR | Title |
|-------|-----|-------|
| Design doc | #71 | Move 8b design doc: substrate field-type tightening |
| Phase A | #72 | Move 8b Phase A: Mixture/Product field-type tightening |
| Phase A′ | #73 | Move 8b Phase A′: ParticlePrevision{T} + EnumerationPrevision retirement |
| Phase B | #74 | Move 8b Phase B: consumer construction-site typing |
| Phase C | #75 | Move 8b Phase C: untyped-mixture-construction lint slug |

## End-state properties

After Move 8b, every Prevision type in the substrate satisfies five properties (from Move 8b design doc §0):

1. **Every Prevision holds only algebraic content.** Parameters and weights that enable `expect` to compute; no carrier-space objects, no domain elements, no observational data. Verified by Phase A (PR #72) tightening `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}`; Phase A′ (PR #73) making `ParticlePrevision{T}` parametric and retiring `EnumerationPrevision`.

2. **Every Measure is `(Prevision, carrier-space)`.** Observational carrier binding happens at the Measure layer. Verified by Phase A′ (PR #73) introducing `EnumerationMeasure{T}` which stores `CategoricalPrevision` (simplex) + `carrier::Vector{T}` (domain objects) + `space::Finite{T}` (carrier space).

3. **Parametric typing for fields whose element type varies across instances but is uniform within each instance.** Verified by Phase A′ (PR #73): `ParticlePrevision{T}` uses `samples::Vector{T}`, not `Vector{Any}`.

4. **No type named for what it isn't.** `EnumerationPrevision` held carrier objects inside a Prevision; it is retired (PR #73) and replaced by `EnumerationMeasure{T}`.

5. **Lint covers container element-type discipline.** The `untyped-mixture-construction` slug (PR #75) prevents the blind spot from recurring. Corpus self-test passes; `credence-lint check apps/` reports 0 violations.

## Carrier-dependent dispatch invariant (§5.1a)

Phase A's implementation surfaced an architectural invariant that is permanent, not a Posture 4 transient:

> Prevision-level `condition`, `expect`, and `optimise` are available for Prevision trees whose leaves do not require carrier mappings to evaluate kernel likelihoods. Measure-level dispatch is required for Prevision trees containing Categorical (or any future carrier-dependent type) leaves.

This follows from the `CategoricalMeasure` principled exception (Move 6 §5.1): where a Prevision cannot self-represent without a carrier, operations that recurse through it cannot dispatch at the Prevision level. Threading carrier-space through the Prevision-level path would violate the design principle that Previsions don't carry spaces. The dispatch-level split is the principled consequence, not a partial implementation.

Future move authors and Posture 6's interface design inherit this invariant. See Move 8b design doc §5.1a for the full rationale.

## Open item

`apps/julia/pomdp_agent/` status remains undetermined and is the next bookkeeping task (Closure PR 3). The package was excluded from Posture 4's lint scope per issue #5 (own `src/`, own invariants); its 46 Measure-vocabulary sites are unresolved.
