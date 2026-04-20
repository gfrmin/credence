# Posture 3 — Prevision-first reconstruction

Branch: `de-finetti/migration`. Master plan: `/home/g/.claude/plans/in-this-branch-we-imperative-grove.md`.

## What this branch does

Reconstructs Credence's foundation prevision-first, in the de Finetti / Whittle tradition: `expect` becomes the primitive coherence-justified operator, and Measure becomes the restriction of a prevision to indicator functions. The reconstruction is operational, not just philosophical — it dissolves three current pains:

1. **Conjugate dispatch is case-analytic.** Replaced by a type-structural `ConjugatePrevision{Prior, Likelihood}` registry.
2. **Mixture conditioning duct-tapes `FiringByTag` to a `_predictive_ll` helper.** Replaced by `MixturePrevision` carrying per-component routing as a property of the combination.
3. **Exchangeability is inexpressible in the type system.** Replaced by `ExchangeablePrevision` with de Finetti's representation theorem as a method.

The deliverable is the operational substrate for a paper — *"A de Finettian foundation for Bayesian agent architectures"* — drafted in parallel and gating the work.

See `decision-log.md` for the four settled decisions that bound this branch's scope. See `paper-draft.md` for the work in progress that this branch is making true. See `DESIGN-DOC-TEMPLATE.md` for the structure every move design doc must follow.

## Posture-2 dependency

This branch waits for `de-finetti/posture-2-events` to merge to master before any Move 7 work lands. **Crucially: only Move 7 is gated on that merge.** Moves 1 through 6 touch no code paths Posture 2 is in flight on, and proceed in true parallel with Posture 2's remaining PR sequence.

If you are reading this and tempted to serialise the whole Posture 3 sequence behind Posture 2, don't — that's a four-week loss for no gain. The dependency is real but narrow.

After Posture 2 fully merges, this branch rebases onto master. Move 7 inherits the `Event` type from master rather than redefining it; it then refactors `condition(p, e::Event)` from sibling-form (Posture 2's gate-2) to primary-form (Posture 3's philosophical pivot).

## Eight moves, one paper

| Move | Scope | Risk | Blocks |
|------|-------|------|--------|
| 1. Prevision primitive type + TestFunctionSpace | New file `src/prevision.jl`, ~200 lines | Low | — |
| 2. `expect` as definitional, per-Prevision methods | Per-subtype `expect` migration, Strata-1 tests open | Low | 1 |
| 3. Measure as derived view over Prevision | Refactor `src/ontology.jl`; persistence v2 + fixture test | Medium | 1, 2 |
| 4. Conjugate dispatch as type-structural registry | Replace ~200 lines of case-analytic `condition`; Strata-2 tests open | Medium | 3 |
| 5. MixturePrevision + ExchangeablePrevision | New types; representation theorem as method | Medium | 1, 4 |
| 6. Execution layer refactor (particle especially) | ParticlePrevision, QuadraturePrevision; deterministic seed preserved | **High** | 4, 5 |
| 7. `condition` as conditional prevision (event-primary) | Inverts the dispatch hierarchy; SPEC.md §1 + CLAUDE.md update | Low | 6, **Posture 2 merged** |
| 8. Grammar + program-space adaptation; paper draft complete | Mostly cosmetic ontology renames; final paper-section pass | Low | 7 |

Each move opens with a `move-N-design.md` docs-only PR following the template, then a code PR. All design docs include an "Open design questions" section as a grounds-for-rejection requirement.

## Risks

### Particle filtering preservation (Move 6)

The current particle path constructs `CategoricalMeasure(Finite(samples), log_weights)` from importance-weighted samples (`src/ontology.jl:660-684, 848-860`). Refactoring to `ParticlePrevision` must preserve every silent invariant: weight normalisation point, seeding discipline, effective sample size, resampling triggers. Test suites that currently pass deterministically run-to-run under seeded RNG must continue to pass under `isapprox(rtol=1e-12)`. Looser tolerance (e.g. 1e-6) would silently mask sample-order changes that *are* posterior-changing.

### Persistence migration (Move 3)

`Measure` becoming a thin view over `Prevision` changes the serialised shape. Schema v2 includes a `__SCHEMA_VERSION` marker; load_state detects v1 (raw Measure with `logw`, `alpha`, `beta`) and reconstructs the v2 wrapped shape. The migration test loads commit-pinned v1 fixtures from `test/fixtures/` (provenance protocol in `test/fixtures/README.md`) — synthetic round-trip tests do not catch migration bugs.

### Cadence guardrail removal — deliberate

Earlier iterations of the master plan included a "one move per week minimum, one per fortnight maximum" cadence guardrail. It has been **deliberately removed**. Artificial pace pressure incentivises shipping over halting, which is exactly the opposite of what the "halt and escalate" rule wants to protect. The natural throttles are reviewer attention and the parallelism with Posture 2's merge sequence; both are adequate given everything else this plan gates (design-doc-before-code, end-of-PR test green-light, halt-the-line on operational equivalence breaks).

The implication is that the branch's pace is reviewer-driven, not author-driven. If reviewer attention slows, the branch slows; that is the intended behaviour. If `de-finetti/posture-2-events` merges quickly and reviewer attention is plentiful, the branch may move faster than the rough sequencing in the table above implies. The risk to manage: branch rot vs master if the sequence stretches very long. Mitigation: rebase onto master after every upstream merge, not just the Posture 2 merge.

## Verification

End-to-end verification at the end of every code PR; never batched. The full command set lives in the master plan's "Verification" section. Three things to flag here:

- **Skin smoke test mandatory at end of Moves 3, 4, 6, 7.** `python -m skin.test_skin` from repo root. The test driver is `apps/skin/test_skin.py` (verified to exist; eight test functions covering condition / mean / weights / optimise / factor / replace_factor / snapshot / restore / error paths). See `move-0-skin-surface-audit.md` for which moves' refactor surface is covered today and which need the smoke test extended.
- **Strata-1/2/3 tolerances are floor-of-the-strongest-claim.** 1e-14 / 1e-12 / 1e-10. Particle paths under deterministic seeding belong in the 1e-12 bucket, not 1e-6. A test that legitimately needs looser tolerance must justify it in the move's design doc; "particle, sampling is noisy" is not a valid reason when seeding is deterministic.
- **Lint pass.** `grep -r 'credence-lint:' .` should show the existing 7 precedent slugs plus the three new ones added by Move 7 (`prevision-not-measure`, `conjugate-registry`, `exchangeable-declaration`). No regression in slug counts; no orphaned slugs.

## Conventions

- File path:line citations everywhere. The reconstruction touches dozens of sites; reviewers need to be able to navigate.
- "Should" and "must" mean what they sound like. The master plan is law for this branch; design docs may extend it but cannot contradict it without amending `decision-log.md` in the same PR.
- The paper draft is the gating artifact. If a choice arises between code-feature scope and paper completeness, default to paper completeness. Move 8's completion gate is "paper draft complete" — every section has prose, even rough — *not* "paper outline updated".
