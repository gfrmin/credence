# Posture 4 Completion Audit

**Date:** 2026-04-26
**Scope:** Determine whether Posture 4 ends at Move 8 or has remaining substrate-shaped work.
**Strategic context:** The MVP target is credence-proxy v0.1, not the personal-agent direction from the Move 9 design doc. Move 9 is shelved.

## Check 1: Apps Prevision-vocabulary inventory

**Question:** Do all apps speak Prevision vocabulary, or do Measure-vocabulary construction sites remain that would constitute substrate work?

### Findings

**Fully migrated (7/9 Julia apps + all Python apps):**

| App | Status | Notes |
|-----|--------|-------|
| `apps/julia/email_agent/host.jl` | Clean | No BetaMeasure/MixtureMeasure/ProductMeasure references |
| `apps/julia/grid_world/host.jl` | Clean | Migrated in Move 6 |
| `apps/julia/rss/host.jl` | Clean | Migrated in Move 6 |
| `apps/julia/qa_benchmark/host.jl` | Clean | Migrated in Move 6 |
| `examples/host_credence_agent.jl` | Clean | Migrated in Move 6 |
| `apps/python/credence_agents/` | Clean | julia_bridge.py migrated in Move 8 |
| `apps/python/credence_router/` | Clean | routing_domain.py, router.py migrated in Move 8 |
| `apps/python/bayesian_if/` | Clean | No direct Measure construction |
| `apps/python/credence_bindings/` | Clean | Thin wrapper layer |

**Remaining Measure-vocabulary sites (non-blocking):**

`apps/julia/email_agent/live.jl` — 2 sites:
- Line 183: `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))`
- Line 191: `MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)`

`apps/julia/email_agent/eval_retrospective.jl` — 4 sites:
- Line 17: `using Credence: TaggedBetaMeasure, MixtureMeasure, BetaMeasure`
- Line 150: `TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0))`
- Line 159: `MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)`
- Line 295: `state.belief.components[i]::TaggedBetaMeasure`

These are secondary files (interactive CLI driver, retrospective eval script), not the primary host. They construct the same pattern as the migrated host.jl and can be updated as application-level cleanup without substrate changes.

**Constitutionally excluded:**

`apps/julia/pomdp_agent/` — 46 Measure-vocabulary sites. This is a separate Julia package with its own `Project.toml`, `src/`, `test/`, and `CLAUDE.md`. It is explicitly excluded from credence-lint per CLAUDE.md: "apps/julia/pomdp_agent/ is excluded (own src/, own invariants)." Its migration is not Posture 4 work.

**Lint status:** 144 files scanned, 0 violations.

### Verdict

No substrate work remaining. The 6 email_agent secondary-file sites are application-level cleanup (same pattern as the already-migrated host.jl). pomdp_agent is constitutionally excluded.

## Check 2: Proxy-substrate compatibility

**Question:** Does the credence-proxy product surface have substrate type mismatches that would block v0.1?

### Findings

**Source migration complete:**
- `credence_router/routing_domain.py` — state construction uses `ProductPrevision`, `BetaPrevision`, `MixturePrevision` vocabulary throughout; docstrings updated
- `credence_router/router.py` — `MixtureMeasure` → `MixturePrevision` in comments; CategoricalMeasure references preserved (principled exception)
- `credence_router/julia_bridge.py` (via credence_agents) — all seval strings migrated to Prevision constructors

**Test coverage:**
- 10 test files in `apps/python/credence_router/tests/`
- Move 8 test run: 150 passed, 2 failed
- The 2 failures are known skin startup flakes (issue #9, documented in project memory `project_credence_skin_lifecycle_flakes.md`), not type mismatches

**No substrate blockers identified.** The router's state is a nested `ProductPrevision` of `BetaPrevision` factors, all of which have complete Prevision-primary dispatch (expect, condition, draw, variance, weights).

### Verdict

credence-proxy v0.1 has no substrate blockers. The 2 test failures are infrastructure flakes unrelated to the Measure→Prevision migration.

## Check 3: BDSL stdlib gap audit

**Question:** Does every BDSL operation have Prevision-primary dispatch, or are there operations that only work on Measure types?

### Findings

**Axiom-constrained functions:**

| Function | Prevision dispatch | Notes |
|----------|--------------------|-------|
| `expect` | `src/previsions.jl` | Primary dispatch target since Move 4 |
| `condition` | `src/previsions.jl` | Event-form + parametric-form, both Prevision-primary |
| `push` | `src/ontology.jl` | Works via Measure facade (principled — push produces a new measure on a target space) |
| `density` | `src/ontology.jl` | Kernel-side, space-agnostic |

**Stdlib (`src/stdlib.jl`):**

| Function | Prevision dispatch | Notes |
|----------|--------------------|-------|
| `mean` | `mean(p::Prevision)` | Via `expect(p, Identity())` |
| `variance` | `variance(p::Prevision)` + closed-form for BetaPrevision, GaussianPrevision | Added Move 8 |
| `probability` | `probability(p::Prevision, e::Event)` | Via `expect(p, Indicator(e))` |
| `weights` | All 5 discrete Prevision subtypes | CategoricalPrevision, MixturePrevision, ParticlePrevision, QuadraturePrevision, EnumerationPrevision |
| `marginal` | `marginal(p::MixturePrevision, ...)` | Subset extraction |

**Infrastructure (`src/ontology.jl`, added Move 8):**

| Function | Prevision dispatch |
|----------|--------------------|
| `draw` | BetaPrevision, TaggedBetaPrevision, GaussianPrevision, GammaPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision |
| `prune` | MixturePrevision |
| `truncate` | MixturePrevision |

**Host helpers (`src/host_helpers.jl`, added Move 8):**

| Function | Prevision dispatch |
|----------|--------------------|
| `extract_reliability_means` | MixturePrevision |
| `marginalize_betas` | MixturePrevision |
| `update_beta_state` | MixturePrevision (via wrap_in_measure delegation) |

**CategoricalMeasure exception:** Remains as Measure, not Prevision. This is the principled exception documented across all moves — CategoricalMeasure binds a carrier space (`Finite`), which is an observational property that belongs on Measure, not Prevision.

### Verdict

BDSL stdlib has complete Prevision-primary coverage. No gaps. Every operation that a consumer (credence-proxy, credence_agents, email_agent host, qa_benchmark) calls has Prevision-level dispatch.

## Overall verdict

**Posture 4 is complete at Move 8.**

The Prevision-primary migration is done across the substrate (`src/`), the BDSL stdlib, and all consumer apps. The remaining Measure-vocabulary in email_agent secondary files (6 sites in `live.jl` and `eval_retrospective.jl`) is application-level cleanup that does not require substrate changes and does not block credence-proxy v0.1. The pomdp_agent package is constitutionally excluded from Posture 4 scope.

No further moves are needed before shifting focus to credence-proxy v0.1 product work.
