# Posture 4 — Complete the de Finettian migration

Branch: `de-finetti/complete`. Master plan: `docs/posture-4/master-plan.md`.

## What this branch does

Finishes the reconstruction Posture 3 started. Posture 3 landed prevision-first in `src/` and left a compatibility shim across the rest of the repository — Measure as a user-facing surface, JSON-RPC wire preservation, Previsions holding Measures, `getproperty` shields dispatching on Prevision subtype, v1→v2 persistence migration protecting fixtures that only exist to audit the migration itself. Every one of those costs was paid to a compatibility burden that does not exist. Nothing in this repository ships externally. The precaution has outlived its usefulness.

Posture 4 retires the compatibility layer entirely. Measure is deleted as a user-facing type. Every caller — skin, Python bindings, apps, tests, BDSL stdlib, examples — speaks the Prevision API directly. Presentation helpers (`mean`, `variance`, `probability`, `weights`) become one-line wrappers over `expect` on particular test functions, because in the de Finettian view that is what they are. The body work (Gmail connection, feature extraction, Telegram loop, server loop, production persistence) lands in the same branch, against the clean foundation, as the evidence that the foundation is usable rather than merely correct.

The deliverable is a repository in which the paper's foundational claim — that `expect` on a declared test function space is the single probabilistic primitive — holds end-to-end, not only below the axiom line.

## The design principle for every ambiguous call

**Take the most de Finettian option.** When a function could be a named method or a one-liner over `expect(p, f)`, it is the one-liner. When a field could be typed as `Prevision` or as `Any` for load-order convenience, it is `Prevision`. When a value could be carried as a declared structural type (an `Event`) or as an opaque closure, it is the declared type. The module structure bends to accommodate the type system, not the other way around. There is no "pragmatic impurity" docstring anywhere in the tip of this branch; if one would be needed, the pragmatic choice is wrong and the disciplined one is needed instead.

Concretely, the principle produces:

- `mean(p) = expect(p, Identity())` — mean is not a structural accessor; it is the action of the prevision on the identity test function, by definition.
- `probability(p, e) = expect(p, Indicator(e))` — same.
- `variance`, higher moments, `weights` over a finite space, `marginal` over a product space — all `expect(p, f)` for some declared `f`.
- `MixturePrevision.components::Vector{Prevision}` — no `Any`, no untyped `Vector`, no Measure-view shortcuts.
- `ConditionalPrevision{E <: Event}` — the event is carried as its declared type, with the event hierarchy and the prevision hierarchy co-located in whichever module lets both constraints hold simultaneously.

There are exactly two primitive operations at the axiom layer: `expect` and `condition`. Everything else is derived. Structural fields on concrete Prevision subtypes exist as performance representations — `CategoricalPrevision.log_weights` is the representation that makes `expect(p, Indicator(a))` computable in O(1) — not as alternative probabilistic surfaces. If a numerical query on a prevision does not route through `expect`, the design is wrong.

## Scope — everything

Posture 3's scope boundary (decision-log #3) carved `apps/skin/server.jl`, `apps/python/*`, and body work out of the reconstruction. Posture 4 explicitly lifts that boundary. Every layer of the repository migrates on this branch:

| Layer | Migration |
|-------|-----------|
| `src/ontology.jl` | Delete nine Measure subtypes; retain Space, Event, Kernel; presentation helpers as functions over Prevision |
| `src/prevision.jl` | Previsions hold Previsions; `ConditionalPrevision{E <: Event}`; `log_weights` convention throughout |
| `src/persistence.jl` | v3 only; v1 and v2 paths deleted; fixtures recaptured |
| `src/stdlib.bdsl`, `examples/*.bdsl` | Prevision vocabulary; presentation helpers as DSL stdlib functions |
| `test/test_*.jl` | Every Measure construction site rewritten as Prevision construction |
| `apps/julia/{email_agent,qa_benchmark}/` | Host files speak Prevision |
| `apps/skin/server.jl` | JSON-RPC surface redesigned; Prevision vocabulary on the wire |
| `apps/python/{credence_bindings,credence_agents,credence_router,bayesian_if}/` | Rewritten against Prevision API |
| `apps/julia/body/` (new) | Gmail connection, feature extraction, Telegram loop, server loop, persistence |
| `SPEC.md`, `CLAUDE.md`, `README.md`, `docs/` | Prevision-first throughout |
| `docs/posture-3/paper-draft.md` | Reconciled: implementation section reflects Posture 4 tip; Measure-as-view concessions removed |

Nothing is out of scope. That is the point.

## Ten moves, one completion

| Move | Scope | Risk | Blocks |
|------|-------|------|--------|
| 0 | Pre-branch capture — commit SHA pinned, invariance test output captured at Strata 1/2/3 tolerances against current master | Low | — |
| 1 | Field-name unification (`log_weights` throughout); `ConditionalPrevision{E}`; shield retirement inside `src/` | Low | 0 |
| 2 | Previsions hold Previsions — `Vector{Prevision}`, no `Any`, no view shortcuts | Medium | 1 |
| 3 | Persistence v3-only; v1/v2 migration code deleted; fixtures recaptured | Low | 2 |
| 4 | Test suite migrated — every Measure construction rewritten against Prevision constructors | Medium | 3 |
| 5 | Measure deleted from `src/` — the point of no return | **High** | 4 |
| 6 | Apps (`email_agent`, `qa_benchmark`) and BDSL stdlib migrated | Medium | 5 |
| 7 | Skin (`apps/skin/server.jl`) rewritten — internal belief is Prevision; JSON-RPC surface redesigned | Medium | 6 |
| 8 | Python bindings (`apps/python/*`) rewritten against Prevision API | Medium | 7 |
| 9 | Body work — Gmail, feature extraction, Telegram loop, server loop, production persistence | High | 8 |
| 10 | Documentation (SPEC, CLAUDE, README, docs/) and paper draft reconciled | Low | 9 |

Each move opens with a `docs/posture-4/move-N-design.md` docs-only PR, then a code PR. Roughly 20 PRs total.

Move 0 is non-trivial — see `move-0-design.md`. The pre-branch capture is the invariance target for every subsequent behavioural assertion on the branch. Without it, a ten-move migration has no ground truth to check itself against.

## What "rip the plaster" means operationally

Posture 3's cadence was Socratic — design doc first, code second, review gate between them, reviewer-driven pace. Posture 4 inherits the cadence but with a sharper end-condition: at the tip, `grep -r 'pragmatic impurity' src/ apps/ test/ docs/` returns nothing; `grep -r 'Measure' src/` returns only references in docstrings that explicitly frame Measure as the deleted pre-Posture-4 type; and the paper's §4 Implementation section describes the Posture 4 tip without footnoted concessions.

The reconstruction is not done when the tests pass. It is done when no caller, anywhere, is still speaking the Kolmogorov vocabulary as its primary idiom, and when the body work demonstrates that the Prevision vocabulary is a usable surface for real applications.

## Conventions

- File path:line citations everywhere, inherited from Posture 3.
- "Should" and "must" mean what they sound like.
- The paper draft is the gating artifact for Move 10 and by implication for the branch; if a choice arises between code-feature scope and paper completeness, default to paper completeness.
- Design docs at `docs/posture-4/move-N-design.md` follow `DESIGN-DOC-TEMPLATE.md`. Non-boilerplate "Open design questions" sections are mandatory; empty or shallow ones are grounds for revision.

## Reading order for new reviewers

1. `decision-log.md` — the five decisions settled before this branch begins.
2. `master-plan.md` — the full ten-move specification.
3. `move-0-design.md` — the pre-branch capture protocol.
4. `docs/posture-3/paper-draft.md` — the paper this branch is making true, unchanged until Move 10.
5. `CLAUDE.md` — the repo constitution, updated incrementally across Moves 1–9 and consolidated at Move 10.
