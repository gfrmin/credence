# Posture 3 — Decision log

The four decisions settled before any code or design-doc work begins. Each is recorded verbatim from the planning session that produced this branch's master plan. Subsequent design-doc PRs reference this file by name; if a design doc proposes anything that conflicts with a decision recorded here, the design doc must propose an amendment to this file (with rationale) in the same PR.

## Decision 1 — Posture 2 sequencing

**Resolved: Posture 2 is on master.** This branch rebased onto the post-merge master on 2026-04-20; Move 0 sits directly atop gate-7.

Posture 2's 7 gates, as they landed on master:
- `1d54b94` gate-1: Event types and `indicator_kernel`
- `2d42ddb` gate-2: `condition(::Measure, ::Event)` sibling form
- `326cbdb` gate-3: `test/test_events.jl` and MixtureMeasure zero-mass guard
- `5c7f63f` gate-4: equivalence test — `condition(m, e) ≡ condition(m, indicator_kernel(e), true)`
- `0c182bb` gate-5: rewrite `compute_eu_primitive` via `condition(m, TagSet(...))`
- `c85e879` gate-6: CLAUDE.md event-conditioning precedent + corpus good_ examples
- `7b08576` gate-7: SPEC.md §1.0 Foundations + §6.3 Connections.events

Move 7 (`condition` as conditional prevision, primary form) inherits Event from master rather than redefining it. The historical sequencing concern — Moves 1-6 proceeding in parallel with Posture 2's in-flight PR sequence — is moot now that the merge is done; all eight moves of Posture 3 proceed linearly on this branch.

(Previously an open decision question: whether to wait for Posture 2, cherry-pick in, or redesign Event from scratch. Settled by waiting; resolved by the merge.)

## Decision 2 — Application interaction style: (b) Both, Prevision preferred for new code

Existing applications under `apps/` keep the Measure surface and continue calling `weights(m)`, `expect(m, f)`, `condition(m, k, obs)`. New domain code (e.g. the planned Gmail connection, future apps) writes against Previsions directly.

Rejected alternatives:
- **(a) Both, Measure-as-view canonical** — too conservative; the email-agent paper case study reads naturally in prevision language ("programs are the ergodic components of an `ExchangeablePrevision`"), awkwardly in measure language. The framework's contribution is the prevision-first reconstruction; making it invisible understates the work.
- **(c) Prevision only, deprecate Measure** — too much churn for a single branch; risks silent breakage in the long tail of consumer call sites enumerated in the master plan. Reserve as a possible follow-up after the body work lands.

## Decision 3 — Scope boundary: Julia core + tests + SPEC/CLAUDE/paper

In scope on this branch:
- `src/` — full reconstruction (Prevision primitive, conjugate registry, MixturePrevision/ExchangeablePrevision, execution layer refactor, condition as conditional prevision, program-space adaptation).
- `test/` — Strata 1, 2, 3 test suites; v1 fixtures for persistence migration.
- `SPEC.md` §1 — full rewrite under the prevision-first foundation.
- `CLAUDE.md` — frozen-types update (four types: Space, Prevision, Event, Kernel — Measure as view), Invariants 1-2 amendments, three new precedent slugs.
- `docs/posture-3/` — design docs, paper draft, decision log, audits.

Out of scope on this branch:
- `apps/skin/server.jl` — JSON-RPC API surface (Measure handles, condition/expect/push calls) is preserved bit-for-bit. Verified by `python -m skin.test_skin` smoke runs at end of Moves 3, 4, 6, 7.
- `apps/python/*` — Python bindings, credence_router, credence_agents, bayesian_if are untouched.
- Body work — Gmail connection, Telegram loop, persistence beyond the Move-3 schema migration.

Python-side prevision idioms (e.g. `Prevision.beta(α, β)` in `credence_bindings`) are a follow-up branch.

## Decision 4 — PR cadence: move-per-PR with design docs

Each of the 8 moves opens with a `docs/posture-3/move-N-design.md` docs-only PR, then a code PR. Roughly 16 PRs total (Move 0 + 8 design + 8 code, with Move 8's design folded into its code PR per the master plan's PR cadence table).

Every move design doc must follow the template at `docs/posture-3/DESIGN-DOC-TEMPLATE.md`. Reviewers reject design docs that omit the "Open design questions" section or fill it with boilerplate.

Rejected alternatives:
- **One PR per move, no design doc gates** — saves process overhead but loses the Socratic discipline that catches reconstruction-level mistakes before they ship as code.
- **Single mega-PR at end** — reviewable as a unit but unreviewable as code. Ruled out.
