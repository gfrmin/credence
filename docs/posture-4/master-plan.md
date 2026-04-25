# Master plan — Complete the de Finettian migration

Branch: `de-finetti/complete`.

This is the durable in-repo copy of the branch's master plan. Surgical updates during the branch are permitted; contradictions with `decision-log.md` require amendment of the log in the same PR.

## Context

Posture 3 landed prevision-first in `src/` and stopped at the scope boundary set by its decision-log #3: `apps/skin/server.jl`, `apps/python/*`, and body work were carved out to narrow the blast radius and preserve JSON-RPC wire compatibility. The result is a codebase in which `src/` makes the paper's foundational claim and every layer above it contradicts the claim in order to preserve a compatibility surface that nothing depends on.

Posture 4 finishes the reconstruction. The compatibility surface is retired. Every layer speaks Prevision directly. The body work lands against the clean foundation as Move 9 and serves as evidence that the foundation is usable, not merely correct.

The guiding principle is set in `decision-log.md`: at every ambiguous design choice, take the most de Finettian option. There is exactly one numerical primitive — `expect(p, f)` — and one conditioning primitive — `condition(p, e)` or `condition(p, k, obs)`, peer primary per Posture 3 Move 7. Every other named function on a prevision is sugar over these. The type system bends to accommodate the foundation, not the other way around.

## Prevision-primary principle (amended 2026-04-24 post-Move-2-Phase-4-pivot)

> Prevision is the primary type. Where a structural question asks whether information belongs on Prevision or on Measure, the default is Prevision unless the information is observational (spaces, sampling). Per-component space variation in posteriors is not observational; it is the mathematical content of concentration and belongs on the Prevision.

Prevision-as-primary isn't stylistic preference — it's the load-bearing claim of the Credence programme. Cox gives probabilities, Savage gives EU maximisation, and Prevision-primary makes those axioms **operative** rather than decorative. Expectations are the thing the axioms license; space-reduction is a representational workaround that exists only because the library started in Measure-first land.

Every move's design doc that asks "should this field / operation live on Prevision or on Measure?" resolves by this principle. Measure's reason to exist is shrinking with each move; Prevision's surface is what Posture 4 is building. See `docs/posture-4/move-2-design.md` §5.1 (post-Phase-4 amendment) for the case that surfaced this principle — Move 2's attempt to tighten `MixturePrevision.components::Vector{Prevision}` required per-component space info that Prevision doesn't carry and shouldn't carry. Move 5's Prevision-primary `condition` (concentrated Previsions rather than reduced-space Measures) is the native resolution.

**Meta-observation on the design-doc cadence.** Two Posture 4 moves now have had design-doc premises challenged by mid-flight implementation — Move 0's `Test.@test` assumption, and Move 2's shield-reconstruction-preserves-components assumption. The design-doc-first cadence is working as intended: design docs are proposals to reality, and the implementation layer has an earned veto. A future move author should treat design-doc survival through implementation as the exception, not the rule, and halting on a surfaced premise-failure is the correct mid-flight response.

## Final-state architecture

The Posture 4 tip looks like this. Design docs reference this section as the convergence target; moves transform the current tip toward it.

### Types retained

- **Space hierarchy** — `Space`, `Finite`, `Interval`, `ProductSpace`, `Simplex`, `Euclidean`, `PositiveReals`. Structural, not probabilistic.
- **Event hierarchy** — `Event`, `TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`. Structural predicates over a space.
- **Kernel** — `Kernel`, `FactorSelector`, `LikelihoodFamily` and subtypes.
- **Prevision hierarchy** — `Prevision`, concrete subtypes: `BetaPrevision`, `CategoricalPrevision`, `GaussianPrevision`, `GammaPrevision`, `DirichletPrevision`, `NormalGammaPrevision`, `TaggedBetaPrevision`, `ProductPrevision`, `MixturePrevision`, `ExchangeablePrevision`, `ConjugatePrevision{Prior, Likelihood}`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`, `ConditionalPrevision{E <: Event}`.
- **TestFunction hierarchy** — `TestFunction`, `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `Indicator{E}`, `OpaqueClosure`.
- **Grammar / Program / AgentState** — unchanged from Posture 3 except for the final-state rename of `Functional` → `TestFunction` throughout (Posture 3 left a type alias; Posture 4 deletes the alias).

### Types deleted

- `Measure` and every concrete subtype: `CategoricalMeasure`, `BetaMeasure`, `TaggedBetaMeasure`, `GaussianMeasure`, `DirichletMeasure`, `GammaMeasure`, `NormalGammaMeasure`, `ProductMeasure`, `MixtureMeasure`. Gone entirely from the type system, exports, SPEC, CLAUDE, tests, apps, BDSL stdlib, and paper.
- The `Functional = TestFunction` alias (Posture 3 Move 2 scaffolding). All references updated to `TestFunction`.
- Persistence v1 and v2 loader paths; all associated fixtures.
- The `getproperty` shields on every Measure subtype; they retire with the Measure subtypes themselves.
- The `_predictive_ll` helper and any other dual-residency artefact Posture 3 could not remove without breaking the Measure surface.

### Internal representation invariants

- **Previsions hold Previsions.** `TaggedBetaPrevision.beta::BetaPrevision`. `ProductPrevision.factors::Vector{Prevision}`. `MixturePrevision.components::Vector{Prevision}`. No `Any`, no untyped `Vector`. The `decompose(::ExchangeablePrevision) → MixturePrevision` path returns a genuine `MixturePrevision{Vector{Prevision}}`, making the paper's §1.4 claim structurally true.

- **Unified field name.** `log_weights::Vector{Float64}` on `CategoricalPrevision`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`, `MixturePrevision`. `CategoricalPrevision.logw` is renamed. The Posture 3 `getproperty` branch that dispatched on Prevision subtype to map `.logw` to the right underlying name retires with the shield.

- **Events carried as declared types.** `ConditionalPrevision{E <: Event}` with `event::E`. The opaque-closure `event_predicate::Function` representation is gone. Where a truly structure-free conditional is needed, `OpaqueConditional` is the fallback, clearly marked.

### Operational surface

Exactly two primitive operations:

```julia
expect(p::Prevision, f::TestFunction) → Float64
condition(p::Prevision, e::Event) → Prevision
condition(p::Prevision, k::Kernel, obs) → Prevision
```

Everything else is sugar. The stdlib provides:

```julia
mean(p::Prevision)        = expect(p, Identity())
variance(p::Prevision)    = expect(p, CenteredSquare(mean(p)))    # new TestFunction subtype
probability(p, e::Event)  = expect(p, Indicator(e))
weights(p::CategoricalPrevision) = [probability(p, SingletonEvent(a)) for a in atoms(p.space)]
marginal(p::ProductPrevision, i::Int) = # pushforward via Projection(i)
```

Some of these definitions are implemented directly (`mean` = literally `expect(p, Identity())`, no special case) and some are implemented with performance representations that happen to sit directly on the concrete subtype's fields (`weights(p::CategoricalPrevision)` uses `exp.(p.log_weights)` directly inside its method body). The outer contract is the same: every numerical query on a prevision routes through `expect` at the call site, even when the inner implementation reads a structural field.

The `expect-through-accessor` lint slug enforces this at call sites outside the concrete Prevision subtype's own method bodies.

### Host boundary

`draw(p::Prevision)` returns a sample. Sampling is a host operation, not a prevision operation — it requires RNG state and is not a coherent linear functional on test functions. The distinction is the same as Posture 3's; Posture 4 tightens it by retiring the `sample` alias if any remains.

### New module structure

```
src/
  Credence.jl          — module, exports
  spaces.jl            — Space hierarchy (new file; split from ontology.jl)
  events.jl            — Event hierarchy + indicator_kernel (new file; split from ontology.jl)
  kernels.jl           — Kernel, LikelihoodFamily (new file; split from ontology.jl)
  test_functions.jl    — TestFunction hierarchy (renamed from prevision.jl §TestFunction)
  previsions.jl        — Prevision hierarchy, concrete subtypes, expect, condition (renamed from prevision.jl)
  conjugate.jl         — ConjugatePrevision registry (new file; split from ontology.jl)
  execution.jl         — Particle, Quadrature, Enumeration carriers + their expect methods
  stdlib.jl            — mean, variance, probability, weights, marginal (new file; replaces the accessor soup currently in ontology.jl)
  persistence.jl       — v3 only
  host_helpers.jl      — draw, sampling, RNG-boundary operations
  parse.jl             — unchanged
  eval.jl              — unchanged structurally; updated for Prevision surface
  program_space/       — unchanged structurally; updated to reference TestFunction directly
```

The `ontology.jl` file retires as a single 1860-line monolith. The split reflects the axiom / structure / representation tiers explicitly rather than mixing them in one file.

## Moves

### Move 0 — Pre-branch invariance capture

Docs-only. Pin a commit SHA at the tip of current master (the Posture 3 completion point). Capture test output at Strata 1/2/3 tolerances for every behavioural assertion in `test/`. The capture becomes the invariance target for every subsequent move — every posterior-bearing assertion at the Posture 4 tip must match the capture at the declared tolerance.

Ship `docs/posture-4/move-0-design.md` with:
- The pinned SHA.
- The capture protocol (Julia version, RNG seeding discipline, test invocation command).
- The captured output, stored at `test/fixtures/posture-3-capture/` and checked in.
- The list of assertions whose tolerance is Stratum 1 (`==` or `1e-14`), Stratum 2 (`1e-12`), and Stratum 3 (`1e-10`).

Without Move 0, every subsequent move is flying blind on behavioural preservation. The capture is cheap and the savings downstream are substantial.

### Move 1 — `src/` internal cleanup: field-name unification, parametric `ConditionalPrevision`, shield retirement inside `src/`

Scope: `src/prevision.jl`, `src/ontology.jl`. Low blast radius — no caller outside `src/` is touched yet.

Changes:
- Rename `CategoricalPrevision.logw` → `log_weights`. Update the four carriers (`CategoricalPrevision`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`) and `MixturePrevision` to share the name. Update every internal reader.
- Rewrite `ConditionalPrevision` to `ConditionalPrevision{E <: Event}` with `event::E`. Move the type to a module where both constraints hold simultaneously (see Open design questions).
- Remove the `getproperty` branch on `CategoricalMeasure` that dispatched on stored Prevision subtype. The shield becomes an unconditional forward, or retires entirely in favour of a direct accessor.

Tests: unchanged externally; internal method bodies updated to read the new field name. Stratum-1 and Stratum-2 behavioural output bit-exact against the Move 0 capture.

### Move 2 — Previsions hold Previsions

Scope: `src/prevision.jl`, `src/ontology.jl`.

Changes:
- `TaggedBetaPrevision.beta::BetaPrevision`. Constructor takes `BetaPrevision`, not `BetaMeasure`. Existing consumers within `src/` that pass a BetaMeasure get updated to pass the underlying Prevision (or use the outer constructor that accepts `BetaMeasure` and extracts `.prevision`).
- Surface-ready APIs for Move 5/7: `push_component!(::MixturePrevision, ::Prevision, log_weight)` and `replace_component!(::MixturePrevision, ::Int, ::Prevision)`; a `FrozenVectorView{T}` read-only wrapper type; a `wrap_in_measure(p::Prevision) → Measure` helper. These land as unused surface for Move 5/7's shield-retirement and skin-rewrite migrations; not wired into the shields in Move 2.

Measure subtypes still exist at this point — they are retired in Move 5.

**Deferred to Move 5: `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}`.** Their tightening is architecturally coupled to the Prevision-primary `condition` operation — specifically, `condition` producing concentrated Previsions rather than reduced-space Measures — which is Move 5's scope. `TaggedBetaPrevision.beta::BetaPrevision` tightens in Move 2 independently; it carries no per-component space.

The coupling was surfaced by mid-Phase-4 implementation: shield reconstruction requires per-component space info (e.g., posterior components with reduced `Finite(1)` spaces after conditioning). Prevision doesn't carry that info and shouldn't — spaces are observational content Measure adds; per-component space variation is mathematical content of concentration that the Prevision-primary `condition` (Move 5) resolves natively by producing concentrated Previsions. Storing per-component spaces on the Prevision would be architectural layer confusion; passing `m.space` through shield reconstruction loses per-component reduction. Move 2 delivers the scope that's cleanly decoupled; Move 5 absorbs the rest. See `docs/posture-4/move-2-design.md` §5.1 (post-Phase-4 amendment) for the worked record.

Behavioural capture: bit-exact against Move 0.

### Move 3 — Persistence v3-only

Scope: `src/persistence.jl`, `test/fixtures/`, `test/test_persistence.jl`.

Changes:
- Delete v1 and v2 loader paths.
- Delete `test/fixtures/agent_state_v1.jls`, `test/fixtures/email_agent_state_v1.jls`, and any v2 analogues.
- Recapture v3 fixtures from the Move 2 tip.
- Update `test/test_persistence.jl` to assert v3 round-trip only.

Justification: no deployed state depends on v1/v2; the fixtures exist to audit the migration and the migration is done. Keeping the paths alive through Move 5 would require either updating them as Measure retires (pointless churn) or keeping Measure alive inside persistence (which perpetuates the type system's disagreement with the rest of `src/`).

### Move 4 — Test suite migrated

Scope: every `test/test_*.jl` file.

Changes: every Measure construction rewritten as Prevision construction. `CategoricalMeasure(Finite([:a, :b]), [0.0, 0.0])` → `CategoricalPrevision([0.0, 0.0])` with the space information carried separately where needed. `BetaMeasure(1.0, 1.0)` → `BetaPrevision(1.0, 1.0)`. And so on for each subtype.

Test files are renamed where the rename improves clarity: `test_flat_mixture.jl` describes a Prevision-level concern under the new foundation and may become `test_mixture_prevision.jl`; `test_prevision_mixture.jl` absorbs or co-locates with it.

Behavioural capture: every assertion's value matches the Move 0 capture at the declared tolerance. If any assertion *fails* this check, the cause is investigated before Move 5 proceeds — Move 5 is the point of no return and a surviving behavioural regression at Move 4 is a diagnostic signal about Moves 1–3.

### Move 5 — Measure deleted from `src/`

Scope: `src/ontology.jl` (and its split successors per the final-state module structure), `src/Credence.jl`.

Changes:
- Delete the nine Measure subtypes.
- Delete their `getproperty` shields, constructors, exports.
- Split `src/ontology.jl` into `spaces.jl`, `events.jl`, `kernels.jl`, `test_functions.jl`, `conjugate.jl`, `stdlib.jl` per the final-state module structure. This is the natural moment to perform the split because the file is being gutted regardless.
- Introduce `src/stdlib.jl` with `mean(p) = expect(p, Identity())`, `probability(p, e) = expect(p, Indicator(e))`, `variance`, `weights`, `marginal`.
- Extend the existing `credence_lint.py` pass-two taint analysis (landed in PR #40) with the `expect-through-accessor` slug. This is an extension of the machinery already in `tools/credence-lint/`, not a from-scratch implementation.
- Retire every posterior-iteration pragma site tracked by issue #39 (twelve sites pragma'd as `# credence-lint: allow — precedent:posterior-iteration — tracked in issue #39`; the thirteenth retired incidentally during Move 4) via one of: the new stdlib one-liner (`mean`, `variance`, `probability`), a new `TestFunction` subtype (`CenteredPower{n}`), or a declared-likelihood extension.
- **Tighten `MixturePrevision.components::Vector{Prevision}` and `ProductPrevision.factors::Vector{Prevision}`** (deferred from Move 2 per the architectural coupling — see §Move 2). Activate the `FrozenVectorView{T}` / `wrap_in_measure` / `push_component!` / `replace_component!` surfaces that Move 2 landed as unused.
- **Rewrite `condition` to produce concentrated Previsions, not reduced-space Measures.** This is the Move-5 edit that resolves the per-component-space problem which blocked Move 2's full scope: rather than a posterior component being `ProductMeasure` over a 1-element `Finite`, it is a concentrated `ProductPrevision` whose `CategoricalPrevision` factor has log_weights `[0.0, -Inf, -Inf]` over the unchanged ambient space. Concentration as mathematical content (Prevision weights) replaces concentration as representational content (reduced Measure space). Consumers reading posterior components post-Move-5 see the ambient space, not reduced ones.

**Halting condition.** If any of the twelve issue-#39 sites remain pragma'd at the Move 5 tip, the stdlib is incomplete and Move 5 does not merge. The design doc for Move 5 must track every site's retirement mechanism (new stdlib one-liner, new `TestFunction` subtype, declared-likelihood extension) with evidence that the replacement compiles and produces the captured behavioural value within tolerance. Sites without a disciplined retirement path are evidence that the foundation is not yet ready for Measure deletion.

This is the largest diff of the branch. It is also the simplest: if Moves 1–4 landed correctly and every issue-#39 site retires cleanly, Move 5 is mostly deletion and file splits.

Behavioural capture: bit-exact against Move 0. Any regression is a bug in a prior move that Move 4 did not catch, and Move 5 halts until the regression is fixed at its source.

### Move 6 — Apps and BDSL stdlib migrated

Scope: `apps/julia/email_agent/`, `apps/julia/qa_benchmark/`, `src/stdlib.bdsl`, `examples/*.bdsl`, `examples/host_credence_agent.jl`.

Changes: every Measure construction in the apps and examples migrated to Prevision construction. The BDSL stdlib functions that construct probability objects produce Previsions. The host files realise Previsions directly.

Tests: the apps have their own test harnesses (28 tests in `test_email_agent.jl`, 10 in `test_grid_world.jl`) which were migrated in Move 4; they pass unchanged here.

### Move 7 — Skin rewritten

Scope: `apps/skin/server.jl`, `apps/skin/test_skin.py`, any `apps/skin/*.jl` helpers.

Changes:
- Internal belief state is `Prevision`, not `Measure`. Where the current code maintains `state.belief::MixtureMeasure`, the new code maintains `state.belief::MixturePrevision`.
- JSON-RPC method signatures redesigned for Prevision vocabulary on the wire. `condition`, `expect`, `weights`, `mean`, `optimise`, `factor`, `snapshot_state` are Prevision-over-the-wire.
- Mutation operations become explicit Prevision-level APIs: `push_component!(::MixturePrevision, ::Prevision, log_weight)`, `replace_component!(::MixturePrevision, i, ::Prevision)`. No more `push!(state.belief.components, ...)` through field access.
- `apps/skin/test_skin.py` rewritten for the new wire format.

Design decision: Python callers now speak Prevision vocabulary over JSON-RPC, which motivates Move 8.

### Move 8 — Python bindings rewritten

Scope: `apps/python/credence_bindings/`, `apps/python/credence_agents/`, `apps/python/credence_router/`, `apps/python/bayesian_if/`.

Changes: every Python module that speaks to Credence speaks Prevision. Constructor surface:

```python
Prevision.beta(alpha, beta)
Prevision.gaussian(mu, sigma)
Prevision.categorical(log_weights)
Prevision.mixture(components, log_weights)
```

Query surface:

```python
p.expect(f)      # f is a TestFunction handle
p.mean()
p.probability(event)
p.condition(event)
p.condition(kernel, obs)
```

The `weights`, `mean`, `variance` Python-side methods are one-liners over `expect`, mirroring the Julia stdlib.

Python-side tests pass; the credence-proxy gateway Docker image rebuilds clean with the new surface.

### Move 9 — Body work

Scope: `apps/julia/body/` (new directory), plus minor updates to `src/persistence.jl` for production-state persistence.

Components:

- **Gmail connection.** Maildir via `mbsync`. A `Connection` with `extract(event) → Dict{Symbol, Float64}` and `execute!(action) → Outcome`. Reading emails is reading files; actions are filesystem operations.
- **Feature extraction.** Ollama enrichment pipeline producing the named-feature Dict. The LLM is a prosthetic; it proposes structured descriptions of emails that the Bayesian layer treats as features.
- **Telegram training loop.** Existing bot wired to send the agent's proposed action for user confirmation. 👍 / 👎 reactions encode cost tolerance; user reactions update the CIRL prior over utility parameters.
- **Server loop.** Polling execution of active programs against the evolving feature dictionary. Per-step conditioning. Meta-actions where the agent modifies its own hypothesis space.
- **Persistence.** Production state save/load. Under decision-log #4 of posture-3, the state was to be `MixtureMeasure` of `ProductMeasure` of `BetaMeasure`; under Posture 4 it is `MixturePrevision` of `ProductPrevision` of `BetaPrevision`. Persistence round-trip tested under canonical operating conditions.

The body work is scoped to close the "email is the book" Amazon-analogy loop: first domain demonstrates the Connection abstraction; calendar, files, tasks follow on subsequent branches. Move 9 does not attempt to deliver calendar/files/tasks connections.

### Move 10 — Documentation and paper reconciled

Scope: `SPEC.md`, `CLAUDE.md`, `README.md`, `docs/` throughout, `docs/posture-3/paper-draft.md`.

Changes:
- `SPEC.md` §1 Foundations already prevision-first after Posture 3's Move 7; the rest of SPEC is updated to match.
- `CLAUDE.md` frozen-types section: Space, Event, Kernel, Prevision, TestFunction (five frozen types; Measure concession removed). Forbidden-patterns entries updated: directly reading a Prevision's structural fields to compute a probabilistic property is a forbidden pattern, enforced by `expect-through-accessor`.
- `README.md` updated for the Prevision-first surface.
- `docs/posture-3/paper-draft.md` reconciled:
  - Abstract: remove "We retain Measure as a thin wrapper" clause.
  - §1.2: rewrite the "Two consequences of taking the prevision as primitive" paragraph — Measure is not retained, so the Kolmogorov-familiar surface is a set of named functions over `expect`, not a separate type.
  - §2.1: the pre/post code snippet no longer needs the `BetaMeasure` wrapping on the post-refactor side.
  - §4 Implementation: description updated to Posture 4 tip; "pragmatic impurity" concessions removed. Reference implementation pin updated from the Posture 3 SHA to the Posture 4 tip SHA.

Move 10 lands last because it describes the completed state; landing it earlier would describe a state that does not yet exist.

## Verification cadence

End-to-end verification at the end of every code PR; never batched. The command set for each move lives in its design doc. Shared invariants:

- Every behavioural assertion captured at Move 0 holds at the declared tolerance after each move.
- `grep -r 'pragmatic impurity' src/ apps/ test/ docs/` returns nothing after Move 5.
- `grep -rE 'struct (Categorical|Beta|Tagged|Gaussian|Dirichlet|Gamma|NormalGamma|Product|Mixture)Measure' src/` returns nothing after Move 5.
- `credence-lint` passes with zero `expect-through-accessor` violations after Move 6.
- Skin smoke test passes against the new wire format after Move 7.
- Python bindings integration tests pass after Move 8.
- Body-work end-to-end test passes after Move 9: email arrives at Maildir → feature extraction → Prevision update → action selection → user reaction → further Prevision update.

## Risks

### Move 5 as the point of no return

Move 5 deletes the Measure types. Once merged, reverting requires a branch-level rollback, not a single-PR revert. The design doc for Move 5 must include an explicit pre-merge checklist that confirms Moves 1–4 are green and the Move 0 capture matches at all tolerances. Any open Stratum-2 deviation or unexplained Stratum-3 drift blocks the Move 5 merge.

### Body work timing

Move 9 is the largest individual move by line-count and by external-surface complexity (Maildir sync, Ollama prompting discipline, Telegram bot webhook handling, server loop). Budget accordingly. If Move 9 stretches beyond a reasonable window, consider splitting into 9a (Gmail + feature extraction + persistence) and 9b (Telegram + server loop + meta-actions) at a design-doc PR; the split should not reopen the foundational decisions.

### Wire-format redesign debt (Move 7)

Redesigning the JSON-RPC wire format is net positive — the current shape inherits Measure vocabulary — but it is a schema-design task in its own right and design-document time on Move 7 should reflect that. The Python bindings (Move 8) depend on the final wire shape; if Move 7's wire design is unsettled, Move 8's diff is rework waiting to happen.

### Paper timing

Move 10 is scheduled last. The AISTATS 2027 deadline and the NeurIPS 2026 workshop deadline both precede the realistic completion date for a ten-move branch if the cadence is reviewer-driven per Posture 3 precedent. If a submission deadline forces paper reconciliation before Move 9 closes, the paper lands in a state that describes the branch tip as of its submission rather than the final tip; this is acceptable provided the submission is clearly labelled against its branch SHA.

## Prerequisites before any code lands

- **Move 0 capture complete and checked in.** Nothing else starts until the invariance target exists on disk.
- **Branch rebased onto current master.** Posture 3 Move 8 merged; the Posture 4 branch starts from that point.
- **`docs/posture-4/README.md`, `decision-log.md`, `master-plan.md`, `DESIGN-DOC-TEMPLATE.md`, `move-0-design.md` land together as the opening docs-only PR.** This is the Posture 3 precedent applied; the substantive Move 0 plan is itself the opening shot.

## Grounds-for-reopen

Reviewers reopen PRs that:
- Skip the design-doc step and land code without a matching design PR.
- Omit the "Open design questions" section or fill it with answered questions.
- Introduce a new Measure subtype or a new Prevision-held-Measure field under any justification.
- Introduce a new `getproperty` shield on a Prevision subtype for any reason.
- Add a presentation helper that reads a concrete Prevision's structural field outside the Prevision's own `expect` method body.
- Leave a "pragmatic impurity" or "for backwards compatibility" docstring anywhere in `src/`, `apps/`, or `test/`.
