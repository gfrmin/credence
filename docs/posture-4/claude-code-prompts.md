# Prompts for Claude Code — Posture 4

This file contains one prompt per move, in the order they should be fed to Claude Code. Each prompt assumes the previous move has landed (design doc + code PR) and is on master. The Posture 3 review-then-implement discipline inherits: Claude Code produces the design doc for review first, then the code PR after the design doc is approved.

The Move 0 artifacts (`README.md`, `decision-log.md`, `master-plan.md`, `DESIGN-DOC-TEMPLATE.md`, `move-0-design.md`) are already drafted and live in the repo at `docs/posture-4/`. The first Claude Code prompt is for Move 0's capture script and fixture generation — the design doc exists; the code does not.

---

## Prompt 0 — Move 0 code: capture script and fixtures

You are implementing Move 0 of Posture 4, the pre-branch invariance capture. The design doc is at `docs/posture-4/move-0-design.md`; read it first, along with `README.md`, `decision-log.md`, and `master-plan.md` in the same directory. These are authoritative; your job is to realise them in code.

Tasks:

1. Implement `scripts/capture-invariance.jl` per the design doc. It walks every `test/test_*.jl` file, instruments every `@test` assertion, runs the test suite, and serialises the results to `test/fixtures/posture-3-capture/{strata-1,strata-2,strata-3}.jls` with the manifest at `test/fixtures/posture-3-capture/manifest.toml`.

2. The instrumentation must not modify test behaviour. The captured LHS value is the literal LHS of the assertion expression at the time of evaluation; the captured RHS is the RHS; the tolerance is whatever `atol` / `rtol` the assertion declared, defaulting to `==` (Stratum 1) if none is declared.

3. Resolve open design questions 1–4 in the Move 0 design doc by picking the disciplined option in each case and documenting your choice in the fixture README. My priors:

   - Q1: Structural vs numerical equality — introduce the third axis. A `@test length(programs) == 22` assertion is captured as "structural:length==22" and compared via `==` regardless of stratum. Numerical assertions carry the tolerance.
   - Q2: Particle seed capture — the conservative answer. Capture the sample sequences downstream of every `Random.seed!` call site in `test/`.
   - Q3: Platform pinning — pin Linux x86_64 on the Docker image already used for the repo's CI. Record the exact Julia version and CPU string in the manifest.
   - Q4: Broken/flaky assertions — capture them in their current broken state with a `status = "broken"` manifest entry. Move 10 has explicit permission to upgrade broken → passing if the cleaner foundation fixes them.

4. Run the double-verification from §4 of the design doc. Confirm the fixtures are stable across two clean-checkout runs. Commit the fixtures and the manifest.

5. Open the PR against `de-finetti/complete` with the design doc and code together. The design doc is already written; your PR is the code that realises it. Do not modify `src/`, `apps/`, or any `test/test_*.jl` files.

Before writing code, state in a preamble: which modules of the Julia ecosystem you intend to use for the instrumentation (`Test.Testset` customisation, `ReTest`, `Aqua`, a bespoke macro-walker), and why. The preamble should be under 200 words and should name one alternative you considered and rejected.

---

## Prompt 1 — Move 1 design doc

You are producing the Move 1 design doc at `docs/posture-4/move-1-design.md`. Follow `docs/posture-4/DESIGN-DOC-TEMPLATE.md` exactly, including §0 (final-state alignment), §8 (de Finettian self-audit), and the reviewer checklist.

The move's scope is set in `master-plan.md`:

- Rename `CategoricalPrevision.logw` to `log_weights` throughout.
- Unify the field name across `CategoricalPrevision`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`, `MixturePrevision`.
- Rewrite `ConditionalPrevision` to `ConditionalPrevision{E <: Event}` with a typed `event::E` field, replacing the opaque `event_predicate::Function`.
- Remove the `getproperty` branch on `CategoricalMeasure` that dispatches on the stored Prevision subtype (`src/ontology.jl:135-148`). The shield becomes unconditional forward, or retires if no external consumer still depends on `.logw` as a name.

Scope discipline: no caller outside `src/` is touched in Move 1. Apps, tests, skin, Python, and BDSL stdlib are left untouched; they migrate in Moves 4, 6, 7, 8.

Open design questions you must populate (examples — you may add more if genuinely open):

1. Where does `ConditionalPrevision{E <: Event}` live — `previsions.jl` with `E` unconstrained at the struct level (matching `Indicator{E}`'s current discipline), or a new module where both `ConditionalPrevision <: Prevision` and `E <: Event` constraints hold at the struct level? The latter is more disciplined; it requires module restructure. Argue.

2. If `log_weights` is the unified name, does `CategoricalMeasure.logw` retire entirely (the shield is deleted, not forwarded), or does the shield remain as a one-liner `getproperty(m, :logw) = m.prevision.log_weights` for a deprecation window? My prior: retire entirely; the shield exists for a reason that does not survive Posture 4.

3. The `_dispatch_path` observability hook in `prevision.jl:679` is underscore-prefixed per Posture 3 conventions. Does it stay as-is, or does it migrate to a cleaner observability pattern now that the type system is disciplined?

Address each open question by stating your preferred answer and the argument for it; the review cycle is where I push back. Non-trivial questions only — "should we use `log_weights` or `logs_weight`" is not a design question.

The design doc ends with the reviewer checklist uncrossed. I cross it during review. Do not land code until the design doc is approved.

---

## Prompt 2 — Move 1 code

The Move 1 design doc is approved and on master (merge SHA: fill in). Implement the code per the design doc. The fixtures at `test/fixtures/posture-3-capture/` are the invariance target; every Stratum-1 assertion must match bit-exactly, every Stratum-2 within `1e-12`, every Stratum-3 within `1e-10`.

If any assertion diverges outside its stratum tolerance, halt and diagnose. Do not paper over a divergence with a tolerance relaxation.

The code PR modifies only `src/`. Tests pass unchanged (they still construct Measure subtypes; Measure is still alive and the shields on it — now unconditional — forward correctly).

Ship the PR with the full test suite output showing zero regressions against the Move 0 capture. The verification command lives in the Move 1 design doc §7.

---

## Prompt 3 — Move 2 design doc

You are producing the Move 2 design doc at `docs/posture-4/move-2-design.md`. Same template, same discipline.

The move's scope:

- `TaggedBetaPrevision.beta::BetaPrevision` (was `::Any`).
- `ProductPrevision.factors::Vector{Prevision}` (was `::Vector`).
- `MixturePrevision.components::Vector{Prevision}` (was `::Vector`).

Internal to `src/`. Measure subtypes still alive; their `getproperty` shields may need to reconstruct Measure-shaped views on the fly for consumers that still read `.components` and expect a `Vector{Measure}`. This transient reconstruction retires with Measure in Move 5.

Open design questions:

1. **On-the-fly Measure reconstruction via shields.** `MixtureMeasure.components` returning `[wrap_in_measure(p) for p in prevision.components]` is a per-access allocation. Is this acceptable for the Moves 2–4 window, given that Move 5 deletes it? The alternative — updating every caller in `src/` that reads `.components` as `Vector{Measure}` at Move 2, instead of at Move 5 — front-loads work. My prior: accept the allocation for the transient window; Move 5 removes it.

2. **`ExchangeablePrevision.decompose` return type.** Currently returns `MixturePrevision` whose components are whatever the implementation constructs. With `Vector{Prevision}`, the return type is structurally the same but the elements are guaranteed Previsions. Should the return type be parametrically tightened to `MixturePrevision{T}` where `T <: Prevision` names the component type, or does Julia's existing `MixturePrevision` shape suffice?

3. **Consumer migration inside `src/`.** Every `src/` site reading `m.components[i].alpha` now reads `m.components[i].alpha` where `m.components[i]` is a `BetaPrevision`, not a `BetaMeasure`. The read succeeds because `BetaPrevision` has an `alpha` field. But it bypasses the `expect-through-accessor` discipline we're preparing for Move 5. Do we update the internal callers in Move 2 to use `mean(components[i])` instead, or leave that for Move 5's stdlib introduction?

Resolve each. Land the design doc. Await review.

---

## Prompt 4 — Move 2 code

Implement Move 2 per the approved design doc. Invariance target unchanged. Transient `getproperty` reconstruction must not leak allocated Measure instances beyond the call site that requested them; profile if unsure.

---

## Prompt 5 — Move 3 design doc

Scope: persistence v3-only. Delete v1 and v2 loader paths in `src/persistence.jl`. Delete v1 fixtures at `test/fixtures/agent_state_v1.jls` and `test/fixtures/email_agent_state_v1.jls`. Recapture v3 fixtures from the current tip (post-Move-2). Update `test/test_persistence.jl` to assert v3 round-trip only.

Open design questions:

1. **Fixture regeneration discipline.** The Posture 3 precedent was that fixtures are captured once and never regenerated to fix load bugs (fix the load code instead). Posture 4's v3 fixtures are regenerated because they're new; future Move 9 body-work state is another first capture. Does the "never regenerate" rule apply to v3 as soon as Move 3 lands, or only after Move 9's production-state fixtures are also captured? The difference matters if a Move 4–8 refactor changes the serialised shape of the belief state. My prior: "never regenerate" applies post-Move-9. Moves 3–8 may invalidate and recapture. State this explicitly.

2. **Schema versioning in a single-schema world.** With v1 and v2 gone, the `__SCHEMA_VERSION` field is marking a single version. Does the field stay (for forward compatibility when future schema changes arrive) or retire (YAGNI)? If it stays, the value is `3` and Move 9's production persistence is free to bump it if body-work persistence requires fields not in v3.

---

## Prompt 6 — Move 3 code

Implement Move 3. The v1 and v2 fixtures are deleted with `git rm`; the v3 fixtures are `git add`ed from the regeneration. Persistence round-trip test passes; all other tests pass against the Move 0 capture.

---

## Prompt 7 — Move 4 design doc

Scope: migrate every `test/test_*.jl` file to construct Previsions directly rather than Measures. This is a mechanical but volumetric diff — every `CategoricalMeasure(...)`, `BetaMeasure(...)`, `MixtureMeasure(...)` construction becomes the Prevision equivalent.

Open design questions:

1. **Test file renames.** Currently there is `test_flat_mixture.jl`, `test_prevision_mixture.jl`, `test_prevision_conjugate.jl`, `test_prevision_particle.jl`, `test_prevision_unit.jl`. Posture 3 left some Measure-facing test files and added Prevision-facing ones in parallel. Move 4 unifies: do we consolidate `test_flat_mixture.jl` into `test_prevision_mixture.jl`, or rename? My prior: rename where the rename improves clarity; consolidate where two files have overlapping scope. The design doc lists the final test file inventory.

2. **Constructor surface change.** Previously `CategoricalMeasure(Finite([:a, :b]), [0.0, 0.0])` constructed a measure over a finite space. The Prevision equivalent is `CategoricalPrevision([0.0, 0.0])` — but the space information is gone. Does `CategoricalPrevision` carry a space, or is the space a separate argument passed to `expect(p, Indicator(e))` where `e` references the space? The paper's §1.2 notes that `CategoricalPrevision{T}` is "not parametric on the atom type — the type connection lives at the Measure level." Move 4 is the moment to revisit: with Measure going, where does the atom type live? On the `Indicator{E}` side? On a new `Space` field of `CategoricalPrevision`? State and defend.

3. **Test output diff vs Move 0 capture.** Because tests are being rewritten against a new constructor surface, the Julia value constructed is different even when the semantic content is the same. The invariance check is behavioural — `expect(p, f)` returns the same number — not syntactic. The design doc must state explicitly how the Move 0 capture's invariance is preserved when the LHS of the captured assertion is a Measure field access (`m.alpha`) and the post-refactor test reads a Prevision field (`p.alpha`). Options: capture the value, not the expression; or update the capture to reference the Prevision-surface access. My prior: value, not expression. The capture protocol should have been value-based from Move 0; verify this.

---

## Prompt 8 — Move 4 code

Implement Move 4. The diff is large. Split into sub-PRs if review load warrants; otherwise ship as one code PR with the consolidated test suite migration.

---

## Prompt 9 — Move 5 design doc

This is the point of no return. Read `master-plan.md` §"Move 5" carefully and `decision-log.md` Decision 1 in particular. The Measure type and every concrete subtype is deleted; `src/ontology.jl` is gutted and split.

Scope:

- Delete: `CategoricalMeasure`, `BetaMeasure`, `TaggedBetaMeasure`, `GaussianMeasure`, `DirichletMeasure`, `GammaMeasure`, `NormalGammaMeasure`, `ProductMeasure`, `MixtureMeasure`.
- Delete: every `getproperty` shield, every Measure constructor, every Measure export in `src/Credence.jl`, the `Functional = TestFunction` alias.
- Split `src/ontology.jl` into the six-file structure specified in `master-plan.md` §"New module structure": `spaces.jl`, `events.jl`, `kernels.jl`, `test_functions.jl`, `conjugate.jl`, `stdlib.jl`.
- Introduce `src/stdlib.jl` with `mean`, `variance`, `probability`, `weights`, `marginal` as one-liners over `expect` per `decision-log.md` Decision 2.
- Introduce the `expect-through-accessor` lint slug in `tools/credence-lint/`.

Open design questions:

1. **`variance` implementation.** `mean(p) = expect(p, Identity())` is a one-liner. `variance(p) = expect(p, ???) - mean(p)^2` requires a TestFunction that evaluates `x^2`. Do we introduce `Square` as a TestFunction subtype, `Power{n}` parametrically, or is variance implemented via `expect(p, CenteredSquare(mean(p)))` where `CenteredSquare` is a closure-over-mean? The last is the least de Finettian; the first is the most specific. My prior: introduce a small family — `Square`, `Cube`, `Power{n}` — and implement variance, third central moment, etc. in terms of them. State your answer.

2. **`weights` for non-categorical previsions.** `weights(p::CategoricalPrevision)` returns the normalised probability vector over atoms. What is `weights(p::BetaPrevision)` — undefined, or the continuous analogue (the density function evaluated on a grid)? My prior: undefined for non-finite-space previsions; `weights` carries a domain restriction to `CategoricalPrevision`, `MixturePrevision`, `ParticlePrevision`, `QuadraturePrevision`, `EnumerationPrevision`. For continuous cases, consumers compute specific probabilities via `probability(p, Interval(lo, hi))`.

3. **Module split granularity.** Six files per the master plan. Is this the right granularity — too fine (small files, one-concept-per-file, good navigation) or too coarse (keep `ontology.jl` but heavily trimmed)? I lean toward the split. Argue.

4. **The lint slug.** `expect-through-accessor` flags call sites that read a Prevision's structural field to compute a probabilistic property. Implementation: a syntactic walker over `src/`, `apps/`, `test/` that flags `*.alpha`, `*.beta`, `*.log_weights` reads outside the module that defines the type. False positives: internal reads inside `expect` method bodies are legitimate; the lint must distinguish. State the false-positive mitigation.

This design doc receives extra scrutiny. Expect back-and-forth; do not merge until every open question has a landed answer and the pre-merge checklist in the master plan is green.

---

## Prompt 10 — Move 5 code

Only after Move 5 design doc is approved and `scripts/capture-invariance.jl` reports zero regressions against the Move 0 capture at the Move 4 tip.

Implement Move 5. The diff is large (thousand-plus lines deleted, six new files created) but structurally simple — most of the complexity is in Moves 1–4 preparing the ground.

Post-merge: the repo `grep -rE 'struct (Categorical|Beta|Tagged|Gaussian|Dirichlet|Gamma|NormalGamma|Product|Mixture)Measure' src/` returns empty.

---

## Prompt 11 — Move 6 design doc

Scope: apps migration. `apps/julia/email_agent/host.jl`, `apps/julia/qa_benchmark/host.jl`, `examples/host_credence_agent.jl`, `src/stdlib.bdsl`, `examples/*.bdsl`. Every Measure construction rewritten against Prevision constructors; every accessor call migrated to `mean`, `probability`, `expect(p, f)` per the stdlib.

Open design questions:

1. **BDSL stdlib surface.** The BDSL currently exposes probability constructors and accessors. Does the BDSL surface mirror Julia (`beta 1 1`, `mean`, `probability`), or does it adopt its own idiom? The DSL/host boundary is strict (CLAUDE.md precedent); the DSL constructs mathematical objects, the host realises them. My prior: BDSL mirrors Julia naming; ergonomics are the host's concern, not the DSL's.

2. **Host file structure.** The existing host files are long and mix setup, main loop, and observation handling. Move 6 is an opportunity to split — but splitting host files is out of scope per the master plan's "migration not elaboration" discipline. Is an in-place structural cleanup (reorganising without adding functionality) in scope, or is it deferred to a follow-up? My prior: defer. State.

3. **`apps/julia/qa_benchmark/` ablation variants.** The benchmark host currently has no ablation variants implemented (per `papers/RESULTS.md` "Not Yet Run"). Move 6 migrates what exists; the ablation re-implementation is out of scope per the master plan. State this explicitly so Move 6 does not accidentally take on Paper 1's experimental work.

---

## Prompt 12 — Move 6 code

Implement Move 6. All app tests and integration tests pass against the Move 0 capture. BDSL examples parse and execute under the new stdlib.

---

## Prompt 13 — Move 7 design doc

Scope: `apps/skin/server.jl` and `apps/skin/test_skin.py`. Internal belief state is `Prevision`. JSON-RPC wire shape redesigned for Prevision vocabulary. Mutation operations are explicit Prevision-level APIs.

Open design questions:

1. **JSON-RPC method signatures.** The wire currently exposes `condition`, `expect`, `weights`, `mean`, `optimise`, `factor`, `snapshot_state`. Under Prevision vocabulary, do we keep these names (they generalise cleanly) or rename to flag the break (`prevision_expect`, etc.)? My prior: keep the names; the semantics are the same, the underlying types are cleaner. Breaking the names signals "new API" without adding meaning.

2. **Prevision handle encoding over the wire.** JSON is a textual format; a Prevision has no canonical textual form. Does the wire carry handles (server-side opaque IDs that reference in-memory Previsions) or serialised forms (full JSON encodings of the Prevision's structure)? My prior: handles for working state, serialised forms for `snapshot_state` output. State.

3. **Mutation API on the wire.** `push_component!` and `replace_component!` are Julia-internal APIs. The wire equivalent — `mixture.push_component(handle, component, log_weight)` — is the natural shape, but mutation over JSON-RPC is unusual; most RPC APIs are functional-style (return a new handle). Functional is cleaner; mutation matches Julia's in-place patterns. Pick one.

---

## Prompt 14 — Move 7 code

Implement Move 7. Skin smoke test (`python -m skin.test_skin`) passes against the new wire format. Manifest comparison against Move 0 capture: only the wire shape changed, not the underlying belief values, so `expect(prevision, Identity())` over the wire returns the same numbers.

---

## Prompt 15 — Move 8 design doc

Scope: Python bindings. `apps/python/credence_bindings/`, `apps/python/credence_agents/`, `apps/python/credence_router/`, `apps/python/bayesian_if/`. Constructor surface, query surface, and stdlib helpers per `master-plan.md` §"Move 8".

Open design questions:

1. **Python API idiom.** Does `mean` on a Python-side Prevision handle get exposed as `p.mean()` (method) or `mean(p)` (free function) or `p.mean` (property)? My prior: method, to match Julia and to leave `mean` free at the module level. Properties imply cheap synchronous reads; an RPC round trip is neither cheap nor synchronous in the general case.

2. **TestFunction handles on the Python side.** Constructing an `Identity()` test function in Python requires either (a) a Python-side `TestFunction` class mirror, (b) a string name convention (`p.expect("identity")`), or (c) a callable convention (`p.expect(lambda s: s)`). The callable is flexibly, but it ships a closure over the wire — anti-pattern per Posture 4. (a) is the disciplined option. State.

3. **Migration of `bayesian_if`.** This is the CLI / external-facing surface. Does it migrate to the new Python API at Move 8, or is it a thin follow-up? My prior: migrate at Move 8. The "rip the plaster" directive does not leave a Kolmogorov-vocabulary CLI in place for the body work to speak to.

---

## Prompt 16 — Move 8 code

Implement Move 8. Python integration tests pass; credence-proxy Docker image rebuilds and runs clean.

---

## Prompt 17 — Move 9 design doc

This is the body-work move. The largest by line count, the most external-surface-complex, the evidence that the foundation is usable.

Scope per `master-plan.md` §"Move 9": Gmail connection (Maildir via mbsync), feature extraction (Ollama enrichment → `Dict{Symbol, Float64}`), Telegram training loop, server loop (polling execution, per-step conditioning, meta-actions), production persistence.

Open design questions — expect many:

1. **Connection abstraction interface.** Per the userMemories pattern: a `Connection` has `extract(event) → partial Dict` and `execute!(action) → Outcome`. The body assembles the full feature dictionary by merging all connections. State the formal interface — is `Connection` an abstract type with required methods, a protocol defined by trait, or a `@kwdef struct` with function fields? My prior: abstract type with required methods; traits in Julia are fragile.

2. **Event-form vs parametric-form `condition` in body work.** Per `master-plan.md` §"Move 9" and the userMemories guidance. Email observations are feature dictionaries — structural predicates over the feature space — which map naturally to Event. Ollama enrichment produces noisy predictions, which are Kernel-obs pairs. The body work uses both, not one; state the convention for when each applies.

3. **Meta-action implementation.** The hypothesis-space modification primitive. Meta-actions modify `MixturePrevision.components` by adding or removing components. Does this use the `push_component!` / `replace_component!` API introduced in Move 7, or does it construct a new MixturePrevision from scratch? Mutation is faster; reconstruction is cleaner. My prior: reconstruction unless a profile shows the allocation cost dominates.

4. **Ollama discipline.** The prosthetic. State the prompting discipline: how the LLM's output is constrained to feature-dictionary shape, how parsing failures are handled (retry, fallback to empty dict, error), what the cost model for LLM calls is (per the no-separate-cost-model principle in userMemories).

5. **Telegram training-loop preference encoding.** 👍 / 👎 reactions update the CIRL prior over utility parameters. State the exact mechanism: which TestFunction the reaction corresponds to, how the reaction is encoded as an Event or Kernel-obs pair, how the resulting `condition` call modifies the utility-parameter prevision.

6. **Server loop scheduling.** Polling cadence, per-step conditioning overhead, when programs re-evaluate against the evolving feature dictionary. The existing design in userMemories is "closed-loop (options/Sutton 1999) always; programs re-evaluate at every step against evolving processing state." State this in the design doc with citations and concrete polling interval.

7. **Production state schema.** The belief state is `MixturePrevision` of `ProductPrevision` of `BetaPrevision`. Persistence round-trips this. Does the schema version bump from v3 → v4 for production state (which may have fields — connection registries, program caches — that v3 did not)? My prior: yes. Move 3's v3 covers internal state; Move 9's v4 covers production state and gets its own fixture.

8. **Smoke test for the end-to-end flow.** Email arrives at Maildir → feature extraction → Prevision update → action selection → user reaction → further Prevision update. State the test fixture: a set of emails in a test Maildir, a stubbed Ollama endpoint with deterministic responses, a stubbed Telegram bot that replies 👍 / 👎 on a schedule. The test asserts that the Prevision state evolves correctly across a defined sequence.

This design doc is the one that warrants the most back-and-forth. Do not attempt to land it in one pass.

---

## Prompt 18 — Move 9 code

Implement Move 9 per the approved design doc. This may split into 9a (Gmail + feature extraction + persistence) and 9b (Telegram + server loop + meta-actions) at your discretion, provided the split is proposed in the design doc and approved.

---

## Prompt 19 — Move 10 design doc and code (may be combined)

Scope: documentation and paper reconciliation. `SPEC.md`, `CLAUDE.md`, `README.md`, `docs/` throughout, `docs/posture-3/paper-draft.md`.

Per the Posture 3 precedent, Move 10's docs-only scope may fold design and code into a single PR.

Open design questions:

1. **CLAUDE.md frozen-types list.** Posture 3: Space, Prevision, Event, Kernel (Measure as view). Posture 4: Space, Prevision, Event, Kernel, TestFunction (Measure deleted). Is TestFunction a frozen type, or is it a derived concern that does not warrant frozen-type status? My prior: frozen. The TestFunction hierarchy is load-bearing for the `expect`-is-primitive story.

2. **Paper reconciliation scope.** The paper was drafted against the Posture 3 tip. Move 10 reconciles it against the Posture 4 tip. Reconciliation is primarily editing — the claims stand, the implementation footnotes retire. Is there a deeper rewrite warranted? For example, §2.1's pre/post code snippets no longer need the Measure wrapping step, which simplifies the "post" side and clarifies the paper's claim. State which sections get only footnote-level updates and which warrant paragraph-level rewrites.

3. **Reference implementation SHA pin.** The paper pins a SHA (`6ebec0767ab6231afdfc6367bbffe5856762c9a9` per the current draft) for reproducibility. Move 10 pins the Posture 4 tip SHA instead. State whether the Posture 3 SHA is retained in a "prior implementation" footnote for historical reference.

---

## Branch-level close-out

After Move 10 merges, the branch is complete. The final verification sequence:

```bash
# The disciplines assertions
grep -rE 'struct (Categorical|Beta|Tagged|Gaussian|Dirichlet|Gamma|NormalGamma|Product|Mixture)Measure' src/
# Empty.

grep -r 'pragmatic impurity' src/ apps/ test/ docs/
# Empty.

grep -r 'Measure' src/
# Only references in docstrings that explicitly frame Measure as the deleted pre-Posture-4 type.

# The behavioural assertion
julia --project=. -e 'using Pkg; Pkg.test()'
# All green.

# The full invariance check against Move 0 capture
julia --project=. scripts/verify-invariance.jl \
  --against test/fixtures/posture-3-capture/
# All assertions within their stratum tolerance.

# The body-work end-to-end
julia --project=. test/test_body_integration.jl
# Passes.

# The paper check
# Open docs/posture-3/paper-draft.md. §4 references the Posture 4 tip SHA.
# Abstract does not mention "Measure as a thin wrapper".
# §2.1 post-refactor snippet has no BetaMeasure wrapping step.
```

If any assertion in the close-out fails, the branch is not done. Find the failing move, reopen the corresponding design doc with the failure as evidence, and ship a correction.

The branch merges to master when the close-out is clean.
