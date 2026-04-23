# Posture 4 — Decision log

The five decisions settled before any code or design-doc work begins. Subsequent design-doc PRs reference this file by name; if a design doc proposes anything that conflicts with a decision recorded here, the design doc must propose an amendment to this file (with rationale) in the same PR. The bar for amendment is high — each of these decisions was load-bearing for the branch's scope or discipline and the costs of revisiting mid-branch are substantial.

## Decision 1 — Measure is deleted, not retained as a view

The Posture 3 concession was to keep `Measure` as a user-facing wrapper over `Prevision` so that existing consumer code (`m.alpha`, `weights(m)`, `mean(m)`) continued to work unchanged. Posture 4 retires the concession: every Measure subtype is deleted, every `getproperty` shield goes with them, every consumer site migrates to the Prevision API.

Rejected alternatives:

- **Keep Measure as a view, remove only the pragmatic impurities.** Considered as "Posture 4 cleanup branch" in an earlier draft of this plan. Rejected because keeping the view perpetuates two probabilistic vocabularies across the codebase, and every layer above `src/` then carries the cognitive burden of deciding which vocabulary applies. The paper's central claim is that the prevision is the single primitive; the implementation should not be negotiating with that claim.

- **Delete Measure from `src/` but keep a translation layer at the skin boundary.** Considered as a halfway point. Rejected because the only callers of the translation layer would be other Credence modules that have not yet migrated, which puts the branch in an unstable middle state for however long the translation layer persists. No external consumer depends on Measure; there is nothing for a translation layer to serve.

### Implication for presentation helpers

`weights`, `mean`, `variance`, `probability` are retained as named functions but reimplemented as one-line wrappers over `expect` on declared test functions. This is not a compromise; it is the de Finettian reading. `mean(p) = expect(p, Identity())` is not a convenience — it is the definition. The helpers exist for ergonomics and for named call sites in the stdlib; they are not an alternative surface to `expect`.

## Decision 2 — `expect` is the single numerical query; structural fields are performance, not surface

Concrete `Prevision` subtypes have structural fields (`CategoricalPrevision.log_weights`, `BetaPrevision.alpha`, `MixturePrevision.components`). These fields are the performance representation that makes `expect(p, f)` computable — `expect(::CategoricalPrevision, ::Indicator{Atom})` consults `log_weights` because that is the representation that makes singleton-event probability evaluation O(1). The fields are part of the implementation, not part of the public probabilistic surface.

Every numerical query on a prevision — anything that returns a `Float64` describing a probabilistic property — must route through `expect`. Directly reading `p.alpha` to compute `α / (α + β)` as "the mean of a Beta" is permitted inside the method body of `expect(::BetaPrevision, ::Identity)`; it is not permitted at call sites that want the mean. Call sites call `mean(p)` or `expect(p, Identity())` directly.

Rejected alternatives:

- **Accessor methods as the public surface.** `mean(p)`, `weights(p)`, `probability(p, e)` as the primary API with `expect` as an implementation detail. Rejected because it reproduces the measure-theoretic vocabulary at the type-system level — each new Prevision subtype would need a bespoke set of accessor methods — and the pattern that made conjugate dispatch case-analytic in the first place would reassert itself.

- **Category-theoretic wrapper types (e.g. `Mean{P}`, `Variance{P}`).** Considered for type-clarity; rejected as over-engineered. `mean(p)` returning a `Float64` is clearer than `mean(p)` returning a `Mean{P}` that forwards to `Float64` on unwrapping.

### Lint enforcement

Claude Code implements a lint slug `expect-through-accessor` that flags any call site in `apps/`, `test/`, or `src/stdlib.bdsl` that reads a structural field of a concrete `Prevision` subtype to compute a numerical property — e.g. `p.alpha / (p.alpha + p.beta)` outside of `expect(::BetaPrevision, ::Identity)`'s method body. The slug retires at the tip when no violations remain.

## Decision 3 — No compatibility, no migration path, no wire preservation

The JSON-RPC surface in `apps/skin/server.jl` is redesigned for the Prevision vocabulary on the wire. No shipping users depend on the current wire format. Persistence uses schema v3 only; v1 and v2 fixtures and their loader paths are deleted, not migrated. Python bindings are rewritten rather than transitioned.

Rejected alternatives:

- **Preserve JSON-RPC wire shape.** The skin surface currently exposes `condition`, `expect`, `weights`, `mean`, `optimise`, `factor`, `snapshot_state` over a Measure-structured wire format. Preserving the shape was considered because it would narrow Move 7's blast radius. Rejected because the wire format inherits the Kolmogorov vocabulary (Measure handles, field-access patterns) and preserving it would force Python bindings to translate between vocabularies indefinitely. The `de-finetti/complete` branch does not leave translation layers behind.

- **Persistence migration v2 → v3.** Considered as a mechanical extension of Posture 3 Move 3's v1 → v2 path. Rejected because no deployed state exists at v2 that needs protecting; the fixtures are there to audit the migration protocol itself and not to preserve data. Recapturing fixtures at v3 is simpler than porting the migration path forward.

Nothing in the repository is shipped externally; compatibility with a user base that does not exist is a cost paid for no benefit.

## Decision 4 — Body work lands on this branch, not a follow-up

Posture 3's decision-log #3 scoped body work out. Posture 4 brings it in. The Gmail connection (Maildir via mbsync), feature extraction (Ollama enrichment → `Dict{Symbol, Float64}`), Telegram training loop, server loop, and production persistence all land as Move 9 of this branch.

The rationale is architectural, not schedule-driven. Body work built against a half-migrated foundation accumulates its own compatibility debt — every Prevision-consuming body component would need a companion Measure-consuming shim for the duration of the migration, which negates the ten-move discipline. Body work built against the Posture 4 tip, after Moves 1–8 have retired Measure entirely, has no such debt. The incremental cost of delivering body work on this branch is lower than the incremental cost of delivering it on a follow-up branch that has to coexist with a half-migrated base.

Move 9 is also the evidence that the foundation is usable. A reconstruction whose only consumers are the reconstruction's own tests is a reconstruction that has not been applied yet. The email agent with Maildir, feature enrichment, and the Telegram training loop is the first consumer of the foundation; the qa_benchmark and grid_world examples are test harnesses, not applications.

Rejected alternatives:

- **Body work as a follow-up branch.** The natural sequencing. Rejected for the reason above: the compatibility cost of delivering body work on a half-migrated base is substantial and avoidable.

- **Body work as a parallel branch, merged after `de-finetti/complete`.** Considered for review-load reasons; rejected because it requires either (a) the parallel branch to build against a non-existent Posture 4 tip (impossible) or (b) the parallel branch to build against Posture 3 and then migrate at merge time (which is exactly the compatibility cost the decision is avoiding).

## Decision 5 — PR cadence: move-per-PR with design docs, inherited from Posture 3

Each of the 10 moves opens with a `docs/posture-4/move-N-design.md` docs-only PR, then a code PR. Roughly 20 PRs total (Move 0 capture + 10 design + 10 code; Move 10's design-and-code may fold into a single PR per the Posture 3 precedent if the documentation-only scope does not require review separation).

Every move design doc follows `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Reviewers reject design docs that omit "Open design questions" or that fill it with questions whose answer is obvious from the plan.

The reviewer-driven pace rule from Posture 3 carries over. No artificial cadence targets.

Rejected alternatives:

- **One PR per move, no design doc gates.** Saves ten PR openings; loses the Socratic discipline that catches reconstruction-level mistakes before they ship as code. The Posture 3 precedent validated the discipline; Posture 4 inherits it.

- **Two-move PR batches.** Considered for velocity; rejected because the moves are tightly sequenced and a batched PR review that finds a problem in the first move of the batch invalidates the second move's diff. Each move as its own PR keeps the rollback granularity tight.
