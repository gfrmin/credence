# Master plan — Migrate Credence to a de Finetti / Prevision-first foundation

Branch: `de-finetti/migration`.

This is the durable in-repo copy of the branch's master plan, lifted from the author's session notes so that future Claude Code sessions (including ones on different machines without access to `~/.claude/plans/`) can follow it. Surgical updates for post-Posture-2-merge state applied on 2026-04-20; otherwise preserved verbatim from the initial authoring.

## Context

Credence's current axioms name `condition` and `expect` as peers, but measure-theoretically they are not — `expect` is defined as an integral against a measure, i.e. a derived operation on Measure. The de Finettian view (Whittle 1970; de Finetti 1974) inverts this: `expect` is the primitive, coherence-justified operator, and Measure is what you get when you restrict `expect` to indicator functions.

Three current operational pains motivate the rewrite, not the philosophy alone:

1. **Conjugate dispatch is case-analytic.** `condition(::TaggedBetaMeasure, k, obs)` (`src/ontology.jl:581-617`) and similar methods inspect `k.likelihood_family` and branch. New conjugate pairs add new branches; there is no registry.
2. **Mixture conditioning duct-tapes `FiringByTag` (`src/ontology.jl:300-304`) to `_predictive_ll` (`src/ontology.jl:733-760`).** The `MixtureMeasure` `condition` path (`src/ontology.jl:780-798`) handles flattening but the per-tag routing logic lives inside the LikelihoodFamily hierarchy, not as a property of the mixture itself.
3. **Exchangeability is inexpressible in the type system.** The email-agent runs 22 programs in parallel as independently-weighted hypotheses (`apps/julia/email_agent/host.jl:771`); they are exchangeable-within-tag-class, but nothing in the type signature says so. The CEG-style partial-mixture machinery that would be natural here has no surface to attach to.

Posture 3 reconstructs the foundation prevision-first. Conjugate dispatch becomes type-structural via a `ConjugatePrevision{Prior,Likelihood}` registry. Mixtures become coherent combinations of previsions with `ExchangeablePrevision` as a declared subtype carrying de Finetti's representation theorem as a method. Measure is preserved as a thin view over Prevision so existing consumers keep working.

The deliverable is the operational substrate for a paper — *"A de Finettian foundation for Bayesian agent architectures"*. No existing PPL takes prevision as primitive; the paper is novel because the framework is.

## Settled decisions (from clarification)

1. **Posture 2 sequencing.** *Resolved on 2026-04-20:* all 7 Posture 2 gates landed on master (SHAs `1d54b94` through `7b08576`). This branch rebased onto master; Move 7 (`condition` as conditional prevision) inherits Event from master rather than redefining it.
2. **Application interaction style: (b) Both, Prevision preferred for new code.** Existing apps/ keep the Measure surface; new domain code (e.g. Gmail connection) writes against Previsions. The email-agent paper case study reads naturally in prevision language.
3. **Scope boundary.** Julia core (`src/`) + tests + SPEC/CLAUDE/paper. `apps/skin/server.jl` and `apps/python/*` are out of scope for this branch — their JSON-RPC API surface (Measure handles, condition/expect/push calls) is preserved. Python-side prevision idioms are a follow-up branch.
4. **PR cadence: move-per-PR with design docs.** Each of the 8 moves opens with a `docs/posture-3/move-N-design.md` docs-only PR, then a code PR. Roughly 16 PRs total.

## Design doc template (mandatory structure for every move design doc)

Every `docs/posture-3/move-N-design.md` PR must include the following sections. Reviewers should reject design docs that omit "Open design questions" or that fill it with questions whose answer is obvious from the plan.

1. **Purpose.** What this move accomplishes; one paragraph.
2. **Files touched.** Exhaustive list with line ranges.
3. **Behaviour preserved.** What the strata-1/2/3 tests will assert about pre-/post-refactor equivalence.
4. **Worked end-to-end example.** Trace one representative call through the new dispatch, naming which module owns each step. **Mandatory** for any move that introduces or extends dual residency (a type, value, or hook present in both an old home and a new home — e.g. Move 5's FiringByTag relocation).
5. **Open design questions.** 1-3 specific points where Claude Code is expected to argue back. Example shapes: "should X be a sibling primitive or a derived form?", "do we pick option (a) or (b)?", "is this category-different from the type it would inherit from?". Empty or boilerplate is grounds for revision.
6. **Risk + mitigation.** What could go wrong; what test catches it.
7. **Verification cadence.** Which test files run at end-of-PR (per Verification section below).

## Hard prerequisites before any code lands

- **Posture 2 is on master (resolved).** All 7 gates committed as `1d54b94` (gate-1: Event types + `indicator_kernel`), `2d42ddb` (gate-2: `condition(::Measure, ::Event)` sibling form), `326cbdb` (gate-3: `test/test_events.jl` + MixtureMeasure zero-mass guard), `5c7f63f` (gate-4: equivalence test), `0c182bb` (gate-5: `compute_eu_primitive` rewrite via `condition(m, TagSet(…))`), `c85e879` (gate-6: CLAUDE.md precedent + corpus), `7b08576` (gate-7: SPEC.md §1.0 Foundations + §6.3). Historical note: earlier iterations of this plan treated Posture 2 as gating only Move 7, with Moves 1-6 proceeding in parallel; that sequencing concern is moot now.

- **Move-0 PR: substantial, not a lightweight prelude.** Move 0 ships:
  - `docs/posture-3/README.md` — overview, decision log, Posture-2 dependency clarification.
  - `docs/posture-3/decision-log.md` — the four settled decisions verbatim.
  - `docs/posture-3/DESIGN-DOC-TEMPLATE.md` — the mandatory design-doc template (see below) embedded as a file for reviewers to reference.
  - `docs/posture-3/paper-draft.md` — **populated**, not stub. Required content: abstract drafted, introduction drafted, foundations section drafted (§1.1 Coherence, §1.2 Prevision, §1.3 Conditional prevision, §1.4 Exchangeability and the representation theorem, §1.5 Complexity prior, §1.6 Alignment); remaining sections present as structured placeholders with the specific claims each will defend articulated explicitly (operational consequences §2: which three worked examples; comparison §3: which four prior-art entries; implementation §4: which artefacts the paper points to).
  - `test/fixtures/README.md` — fixture provenance file (see Move 3 below). At Move 0 it lists planned fixtures and their capture protocol; the fixtures themselves capture later (after Posture 2 merges).

  **Grounds-for-reopen:** if Move 0 lands as just README + decision log with a stub paper-draft, the PR is reopened. The paper being the gating artifact means it has to exist in concrete form from day one, not materialise gradually.

## Move 1 — `Prevision` primitive type + `TestFunction` hierarchy

**Files to create:**
- `src/prevision.jl` (~150 lines) — `abstract type Prevision end`, `TestFunction` hierarchy, `apply(f::TestFunction, s)` evaluator. (Move 1 ships no `TestFunctionSpace` container — see `docs/posture-3/move-1-design.md` §5.1 for the three-option analysis that settled on option (c), defer to Move 5 if ever needed.)

**Files to modify:**
- `src/Credence.jl:18-19` — add `include("prevision.jl")` *before* `include("ontology.jl")`. Order matters: Measure becomes a view over Prevision, so Prevision must load first.
- `src/Credence.jl` exports — add `Prevision`, `TestFunction`, `Indicator`, and stub forwards for the migrated `Identity`/`Projection`/etc. (which become `TestFunction` subtypes in Move 2).

**TestFunction hierarchy (Move 1 declares; Move 2 migrates the existing Functional types into it):**
```
abstract type TestFunction end
# Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
# all become TestFunction subtypes.
# Indicator(e::Event) is a new subtype that depends on Posture 2's Event type.
```

**Coherence axioms** (de Finetti 1974; Walley 1991) documented in the docstring as the rationality axiom and σ-continuity as optional strengthening.

**Out of scope for Move 1:** the Functional → TestFunction migration itself (that is Move 2). Move 1 only declares the new abstract types; existing Functional code keeps working.

**Tests:** none yet — abstract types only. Stratum-1 tests start with Move 2.

**Risk:** low. New file, no consumer churn.

## Move 2 — `expect` as definitional; per-Prevision `expect` methods

**Files to modify:**
- `src/prevision.jl` — add `function expect end` declaration with the de Finettian docstring. Each Prevision subtype implements `expect(p::P, f::TestFunction)` directly.
- `src/ontology.jl:429-473` — the existing `Functional` abstract type and its concrete subtypes (`Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`) become aliases for `TestFunction`. Concrete: `const Functional = TestFunction` plus alias the concrete types. This preserves all existing `expect(m::Measure, ::Functional)` dispatch.

**The migration is mechanical but volumetric:**
- Existing `expect(m::CategoricalMeasure, ::Identity)` (`src/ontology.jl:481`) keeps its body, becomes a `TestFunction` dispatch.
- Same for the 9-measure × 6-functional matrix at `src/ontology.jl:476-543`.

**Stratum 1 tests open here:** for each Measure subtype M and each TestFunction f, `expect(M(args...), f)` must equal the pre-refactor result to within 1e-14. Test file: `test/test_prevision_unit.jl` (new). Several hundred cases — script-generated from the existing constructor signatures.

**Risk:** low. Type-alias preserves dispatch.

## Move 3 — `Measure` as derived view over `Prevision`

This is the move the (b)-decision shapes. Under (b), Measure remains as a user-facing thin wrapper, but new Prevision types are added alongside — `BetaPrevision`, `CategoricalPrevision`, `GaussianPrevision`, etc. Measure subtypes get a `prevision::P` field and forward all accessors.

**Files to modify (all in `src/ontology.jl`):**
- Lines 66-79 (`CategoricalMeasure`) → keep struct, add `CategoricalPrevision{T}` in `prevision.jl`, refactor `CategoricalMeasure` to wrap `CategoricalPrevision`.
- Lines 93-103 (`BetaMeasure`) → wrap `BetaPrevision(α, β)`.
- Lines 110-114 (`TaggedBetaMeasure`) → wrap `TaggedBetaPrevision`.
- Lines 123-127 (`GaussianMeasure`), 152-161 (`GammaMeasure`), 134-145 (`DirichletMeasure`), 172-188 (`NormalGammaMeasure`), 194-205 (`ProductMeasure`), 217-232 (`MixtureMeasure`) → analogous wrap.

**Accessor forwarding:**
- `weights(m)` (`src/ontology.jl:85-89`) → `weights(m.prevision)`.
- `mean(m)`, `variance(m)` → forward.
- Field reads in consumer code: `m.alpha`, `m.beta`, `m.logw`, `m.factors`, `m.components`, `m.log_weights` — keep as forwarding properties via `Base.getproperty` overrides on the Measure types so existing consumers (`apps/julia/email_agent/host.jl:213-216`, `apps/julia/qa_benchmark/host.jl:66`, `src/host_helpers.jl:54,67,77,95`) keep working unchanged.

**New `probability` accessor:**
```julia
probability(μ::Measure, e::Event) = expect(μ.prevision, Indicator(e))
```
This is the de Finettian primitive surface, available alongside the Kolmogorov-familiar `weights`/`mean`/etc.

**Persistence:** `src/persistence.jl:19-25` saves/loads Measure subtypes via `Serialization`. Wrapping Measure around Prevision changes the serialised shape. Add a `__SCHEMA_VERSION = 2` field; load_state detects v1 (raw Measure with `logw`, `alpha`, `beta`, etc.) and reconstructs the v2 wrapped shape. One-shot migration on next save.

**Persistence migration test (mandatory in this PR, not a follow-up).** A round-trip test that constructs a v2 state, serialises, and deserialises in the same process is *not* a migration test — it is a serialisation-symmetry test, and it cannot detect the migration bug. The only adequate test loads a real v1 fixture written by pre-Move-3 code:
- `test/fixtures/agent_state_v1.jls` — captured from master at the SHA immediately preceding Move 3's code PR opening (i.e. **after Posture 2 fully merges** — Posture 2's gate-7 touches Measure-adjacent code, so fixtures must capture the post-Posture-2 v1 shape). Checked in; never regenerated to fix loading bugs (fix the load code instead).
- `test/fixtures/email_agent_state_v1.jls` — analogous, captures the email-agent shape (MixtureMeasure of ProductMeasure of BetaMeasure).
- `test/fixtures/README.md` — provenance file: lists each fixture's source SHA, capture date, what the fixture represents, and what changes invalidate it. Initial draft lands in Move 0; SHAs filled in immediately before Move 3's code PR opens.
- New test in `test/test_persistence.jl` (create if missing): load each fixture in v2 code, assert resulting object's weights/parameters/structure match expected values.
- The Move 3 PR includes the fixtures, the SHA-pinned README entries, and the test; deferring any of them is a recipe for a silent break post-merge.

**Stratum 1 tests broaden:** every Measure constructor must produce a wrapped object that behaves identically.

**Risk:** medium. Many consumer sites (`apps/julia/grid_world/host.jl:419,440-445`, `apps/julia/email_agent/host.jl` ~14 sites, `apps/julia/rss/host.jl:228,243,274`, `apps/julia/pomdp_agent/src/probability/cpd.jl:36-152`) read Measure fields directly. The `getproperty` forwarding shield avoids touching them, but persistence migration is a real moving piece.

## Move 4 — Conjugate dispatch as a type-structural registry

Replaces the case-analytic `condition` methods at `src/ontology.jl:561-654` with a single dispatch path through a registry.

**Files to modify:**
- `src/prevision.jl` — add `ConjugatePrevision{Prior,Likelihood}` parametric struct, `update(p::ConjugatePrevision{...}, obs)` methods (one per conjugate pair).
- `src/ontology.jl:545-860` — refactor `condition` into a thin facade:
  ```julia
  function condition(p::Prevision, k::Kernel, obs)
      cp = maybe_conjugate(p, k)
      cp === nothing && return _condition_particle(p, k, obs)
      update(cp, obs).prior
  end
  ```
- `src/prevision.jl` — `maybe_conjugate(p, k)` lookup table keyed on `(typeof(p), k.likelihood_family)`. Returns a `ConjugatePrevision` if matched, `nothing` otherwise.

**Conjugate pairs to register (all currently inline-dispatched):**
- `(BetaPrevision, BetaBernoulli)` — replaces `src/ontology.jl:606-611`.
- `(BetaPrevision, Flat)` — no-op, replaces `src/ontology.jl:612-613`.
- `(GaussianPrevision, NormalNormal)` — replaces `src/ontology.jl:619-629`.
- `(GammaPrevision, Exponential)` — currently no fast-path, add it.
- `(DirichletPrevision, Categorical)` — replaces `src/ontology.jl:634-644`.
- `(NormalGammaPrevision, NormalGammaLikelihood)` — replaces `src/ontology.jl:646-654`.

**`FiringByTag` and `DispatchByComponent` (`src/ontology.jl:300-313`) move to Move 5** — they belong with `MixturePrevision` rather than the conjugate registry.

**Move 4 design-doc decision required: TaggedBetaMeasure routing relocation.** The custom routing loop at `src/ontology.jl:584-617` iterates components and dispatches per-tag — it is mixture-aware in a way `maybe_conjugate` is not. Two relocation options:
- **(a) Loop stays as a `MixturePrevision`-level operation that calls `update` on each component's `ConjugatePrevision`.** Cleaner separation of concerns: the registry handles atomic conjugate pairs; the mixture handles per-component routing.
- **(b) Routing logic moves into the registry as a compound entry `(TaggedBetaPrevision, FiringByTag)`.** More uniform dispatch: every conjugate path goes through `maybe_conjugate`.

Recommendation in the design doc: (a), because compound entries in the registry would replicate mixture-level logic across rows. But this is an "Open design questions" point — Claude Code should argue (b) if there's a reason `MixturePrevision` doesn't want this responsibility. Decision must land with the Move 4 design-doc PR, not during the code PR.

**Stratum 2 tests open here:** for `(prior, kernel, observation)` triples covering every conjugate pair, `condition(Measure(prior), k, obs)` must match the pre-refactor result. Tolerance: 1e-12 for conjugate pairs (same arithmetic). Test file: `test/test_prevision_conjugate.jl`.

**Risk:** medium. The registry must produce bit-identical results for conjugate pairs; the dispatch order matters because `TaggedBetaMeasure` currently has a custom routing loop (`src/ontology.jl:584-617`).

## Move 5 — `MixturePrevision` and `ExchangeablePrevision`

Mixtures become coherent convex combinations of previsions; exchangeable types become first-class.

**Files to modify:**
- `src/prevision.jl` — add `MixturePrevision`, `ExchangeablePrevision`, `decompose(p::ExchangeablePrevision)::MixturePrevision`.
- `src/ontology.jl:780-798` — `condition(::MixtureMeasure, k, obs)` becomes a thin facade calling `condition(m.prevision, k, obs)` where the MixturePrevision implements component-wise update + flattening internally.
- `src/ontology.jl:300-313` — `FiringByTag` and `DispatchByComponent` move to `prevision.jl` as accessors on `MixturePrevision` (component_prevision(p, tag) → Prevision). LikelihoodFamily values that route remain in the LikelihoodFamily hierarchy for kernel construction; the routing semantics now live on the prevision side.

**Dual residency hazard.** Leaving `FiringByTag` alive on the kernel side (for construction) while moving the routing semantics to the prevision side is exactly the kind of dual residency where bugs hide. The Move 5 design doc **must** include a worked end-to-end example tracing `condition(mixture, kernel_with_firingbytag, obs)`: which module owns construction, which owns dispatch, which owns the per-tag routing decision, what the call chain looks like step by step, and what the result is. Without that worked example the design doc fails review. If the worked example reveals that one home is genuinely vestigial, the design doc must name the deletion timeline and gate.

**`ExchangeablePrevision`** is the move that makes the email-agent's 22-programs-as-exchangeable-hypotheses story native:
```julia
struct ExchangeablePrevision <: Prevision
    component_space::Space
    prior_on_components::Prevision
end
function decompose(p::ExchangeablePrevision)::MixturePrevision
    # Representation theorem: exchangeable ⟹ mixture of ergodic components.
end
```
Tag-indexed exchangeability (the email case) decomposes into per-tag-class ergodic components; the existing `FiringByTag` machinery becomes the `component_prevision` accessor.

**Email-agent migration is a Move-5 follow-up, not part of Move 5 itself.** The plan files `apps/julia/email_agent/host.jl:771` (constructs MixtureMeasure with 22 components) become an `ExchangeablePrevision` construction in a separate PR after Move 5 lands.

**Stratum 3 test impact:** `test/test_flat_mixture.jl` (538 lines, ~25 assertions) and `test/test_email_agent.jl` must pass under `isapprox(rtol=1e-10)`.

**Risk:** medium. Component flattening invariants and zero-mass guards (gate-3 from Posture 2: `MixtureMeasure zero-mass guard`) must transfer cleanly.

## Move 6 — Execution layer refactor (the high-risk move)

Four execution strategies need Prevision-aware refactor: conjugate (done in Move 4), quadrature, **particle**, and enumeration.

**Particle is where the risk lives.** The `_condition_particle` importance-sampling fallback (`src/ontology.jl:1497` as of Move 6 Phase 1, introduced by the R4 rename) and the `_condition_by_grid` paths (`src/ontology.jl:1242,1254`) construct CategoricalMeasure of sampled points with importance weights. Move 6 replaces that representation with typed Prevision subtypes (`ParticlePrevision`, `QuadraturePrevision`) wrapped by a CategoricalMeasure facade for consumer-surface compatibility.

**Files to modify:**
- `src/prevision.jl` — add `ParticlePrevision`, `QuadraturePrevision`. `ParticlePrevision` carries samples + log-weights + the seeded RNG strategy.
- `src/ontology.jl:660-684, 848-860` — refactor particle/grid conditioning to construct `ParticlePrevision` rather than `CategoricalMeasure(Finite(samples), log_weights)`.
- `src/program_space/enumeration.jl` — program enumeration constructs a `CategoricalPrevision` over programs (cosmetic; the construction site is the only change).

**Critical preservation:** every silent invariant in current particle filtering must be preserved bit-for-bit:
- Weight normalisation point (currently inside `CategoricalMeasure` constructor, `src/ontology.jl:70-79`).
- Seeding discipline — particle tests will break for the wrong reason if the seed is set at a different point in the call chain.
- Effective sample size and resampling triggers (currently implicit via `prune`/`truncate` at `src/ontology.jl:1022-1034`).

**Stratum 2 tests** for particle paths use tolerance **1e-12, not 1e-6**. The current particle test suite passes deterministically run-to-run under seeded RNG; the only legitimate source of drift from the Posture 3 refactor is floating-point reassociation from constructor changes, which is bounded by ~1e-12. Looser tolerance is "we don't know what we broke but it roughly still works" territory — and would silently mask sample-order changes that *are* posterior-changing. If the design-doc reveals a defensible reason a particular test must reorder samples (e.g. parallelism that didn't exist before), the design doc must name it explicitly. Stratum 3 sweeps `test/test_grid_world.jl` (uses particle filtering directly in 2 of 3 tests) and `test/test_email_agent.jl` (depends on particle filtering indirectly through MixturePrevision).

**Risk:** **high**. This is the only move where the document warns to budget extra care for the particle refactor and its test suite.

## Move 7 — `condition` as conditional prevision (event-primary, kernel-derived)

Inverts the dispatch hierarchy. Under Posture 2 (now on master after merge), `condition(m, e::Event)` is a sibling form. Under Posture 3, it becomes the **primary** form, and `condition(p, k, obs)` becomes the derived form.

**Hard dependency (resolved):** Posture 2 is on master; `condition(m::Measure, e::Event)` from gate-2 (`2d42ddb`) is in the source tree. The rebase of this branch onto master happened on 2026-04-20.

**Files to modify:**
- `src/prevision.jl` — declare `condition(p::Prevision, e::Event)` as the primitive:
  ```julia
  # p(f | e) = p(f · 1_e) / p(1_e), when p(1_e) > 0.
  function condition(p::Prevision, e::Event)
      mass = expect(p, Indicator(e))
      mass > 0 || error("conditioning on measure-zero event; see disintegrate")
      # Construct the conditional prevision.
  end
  ```
- `src/ontology.jl` — refactor `condition(p::Prevision, k::Kernel, obs)` to be derived:
  ```julia
  condition(p::Prevision, k::Kernel, obs) =
      condition(p, ObservationEvent(k, obs))
  ```
  where `ObservationEvent` is a new Event subtype witnessing the likelihood structure. The conjugate registry from Move 4 still fires; `maybe_conjugate` is consulted inside the conditional-prevision evaluation path.

**Move 7 design-doc Socratic: does `ObservationEvent` belong in the `Event` hierarchy?** Posture 2's `Event` types (`TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`) are *structural predicates over a Space* — they declare which subset of the space they pick out. `ObservationEvent(k, obs)` is categorically different: it carries a kernel and an observation value, encoding "this kernel emitted this observation", which is a *likelihood-structured* object. This is a one-paragraph Socratic question for the Move 7 design doc:

> Is `ObservationEvent` an Event, or is it a category-different witness that should keep `condition(p, k, obs)` as a sibling primitive rather than as sugar for `condition(p, ObservationEvent(k, obs))`?

If the honest answer is "parametric update is a sibling primitive to event-conditioning, not a derived form," then Move 7's philosophical pivot unwinds — and that is a legitimate outcome to surface in the design doc rather than ship sugar that papers over a category difference. The design doc must take a position; it must not assert without argument that ObservationEvent is an Event.
- `src/stdlib.bdsl` — add `(condition-on m e)` form for the event surface (does not yet exist; verified). Keeps `(condition m k o)` working.
- DSL `default_env` (`src/eval.jl:23-45`) — register `:condition-on`.

**Stratum 1 + 2 + 3 tests** should pass unchanged; the refactor preserves operational equivalence.

**Risk:** low. The hard work was Posture 2 (event introduction) and Move 4 (conjugate registry). Move 7 is the philosophical pivot but operationally a re-routing.

## Move 8 — Grammar and program-space adaptation

Mostly cosmetic. The complexity prior is currently described as "a measure over program ASTs" (`src/program_space/types.jl`); under Posture 3 it's "a prevision over program ASTs" with the test function space being declared subprogram-frequency features and complexity scoring functions.

**Files to modify:**
- `src/program_space/types.jl` — docstrings and type signatures for `Grammar`, `Program`. Operationally identical; ontologically cleaner.
- `src/program_space/agent_state.jl:122-148` — `add_programs_to_state!` constructs `MixtureMeasure` of `TaggedBetaMeasure` components. Under (b), this is unchanged at the measure-construction level; the underlying prevision is `MixturePrevision` of `TaggedBetaPrevision`.
- `src/program_space/enumeration.jl, perturbation.jl` — confirm `analyse_posterior_subtrees`, `perturb_grammar` work via Measure-as-view (they should, by Move 3's wrapping).

**Tests:** `test/test_program_space.jl` (851 lines, ~44 assertions) must pass unchanged.

**Risk:** low.

## Test strategy — three strata in order

Per the document's Stratum 1 / 2 / 3 plan, run in order; do not skip ahead.

**Stratum 1 (unit equivalence):** `test/test_prevision_unit.jl` (new). For each Measure subtype × TestFunction pair, `expect(M(args...), f) == expect(M(args...).prevision, f)` to within 1e-14. Auto-generated cases from constructor signatures. Opens at Move 2; expanded at Moves 3, 5, 6.

**Stratum 2 (composition equivalence):** `test/test_prevision_conjugate.jl` (new, opens at Move 4). For `(prior, kernel, obs)` triples covering every conjugate pair plus several particle-fallback cases. Tolerance 1e-12 for conjugate pairs; **1e-12 for particle paths under deterministic seeding** (the only legitimate drift is floating-point reassociation from constructor changes; looser tolerance would silently mask sample-order changes that *are* posterior-changing).

**Stratum 3 (end-to-end):** every existing test file under `test/`:
- `test/test_core.jl` (1,436 lines, ~140 assertions)
- `test/test_program_space.jl` (851 lines, ~44 assertions)
- `test/test_email_agent.jl` (1,248 lines, ~111 println markers, ~90 assertions)
- `test/test_flat_mixture.jl` (538 lines, ~25 assertions)
- `test/test_grid_world.jl` (464 lines, ~20 assertions)
- `test/test_host.jl` (351 lines, ~25 assertions)
- `test/test_rss.jl` (329 lines, ~15 assertions)
- `apps/julia/pomdp_agent/test/runtests.jl` (separate package; runs via `cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'`)

All pass under `isapprox(rtol=1e-10)`. Drift > 1e-10 = halt-the-line investigation.

## Documentation deliverables

**SPEC.md.** Rewrite §1 Foundations (Posture 2's gate-7 `7b08576` already touches §1.0 — Posture 3 extends it):
- §1.1 Coherence (Dutch-book; de Finetti 1937).
- §1.2 Prevision (operator-valued; test function spaces; σ-continuity optional).
- §1.3 Conditional prevision (primitive; parametric Bayes update derived).
- §1.4 Exchangeability and the representation theorem.
- §1.5 Complexity prior (prevision over programs).
- §1.6 Alignment (CIRL) — recast as prevision over utility functions.

Lands with Move 7's code PR.

**CLAUDE.md.** Structural edits:
- Frozen types: **four** (Space, Prevision, Event, Kernel). Measure declared as a view, not a primitive.
- Invariant 2 expands to enumerate `TestFunction` (generalising Functional) and `Event` (from Posture 2).
- Invariant 1 strengthens: exactly one axiom-constrained `condition` (on events); `condition(p, k, obs)` is stdlib.
- New precedent slugs: `prevision-not-measure`, `conjugate-registry`, `exchangeable-declaration`. Each under "## Precedents".

Lands with Move 7.

**Paper draft** is the gating artifact. The code is the paper's proof, not the other way round.

- `docs/posture-3/paper-draft.md` lands populated (outline + foundations section drafted) in **Move 0**, not as an outline-only file in Move 1.
- Each subsequent code PR (Moves 1-8) updates the paper draft with whatever the move now justifies — operational consequences of conjugate registry (Move 4), worked exchangeability example for the email agent (Move 5), particle-prevision treatment (Move 6), event-primary `condition` and the measure-zero discussion (Move 7), comparison entries against Staton/Jacobs/Hakaru/MonadBayes (throughout).
- **Move 8's completion gate is "paper draft complete"** — every section has prose, even rough — *not* "paper outline updated".
- The paper's methodology section should write itself from the design docs if the design docs are doing their job; this is the test of design-doc quality.
- If at any point a choice arises between code-feature scope and paper completeness on this branch, default to paper completeness.

**`docs/posture-3/`** is the design-doc home:
- `README.md` — overview + decision log + Posture-2 dependency.
- `decision-log.md` — the four settled decisions verbatim.
- `move-1-design.md` through `move-8-design.md` — one per move, lands as a docs-only PR before the corresponding code PR. Each follows the mandatory design-doc template (Purpose, Files, Behaviour preserved, Worked example, **Open design questions**, Risk, Verification).
- `paper-draft.md` — paper draft (not outline). Lands populated in Move 0; grows through every subsequent move; "draft complete" is the Move 8 completion gate.

## PR cadence — 16 PRs

| PR # | Type | Content | Dependency |
|------|------|---------|------------|
| 0 | docs | README + decision log + DESIGN-DOC-TEMPLATE.md + **paper-draft.md (populated: abstract + intro + foundations §1.1-§1.6 drafted; remaining sections as structured placeholders)** + `test/fixtures/README.md` (planned fixtures + capture protocol) + `move-0-skin-surface-audit.md` (verifies `apps/skin/test_skin.py` covers Moves 3/4/6/7 surface). Grounds-for-reopen if paper-draft is a stub. | None |
| 1a | docs | `docs/posture-3/move-1-design.md` | PR 0 |
| 1b | code | Move 1: Prevision primitive + TestFunctionSpace | PR 1a |
| 2a | docs | `docs/posture-3/move-2-design.md` | PR 1b |
| 2b | code | Move 2: expect as definitional + TestFunction migration + Stratum-1 tests | PR 2a |
| 3a | docs | `docs/posture-3/move-3-design.md` | PR 2b, decision (b) |
| 3b | code | Move 3: Measure as view + persistence v2 | PR 3a |
| 4a | docs | `docs/posture-3/move-4-design.md` | PR 3b |
| 4b | code | Move 4: Conjugate registry + Stratum-2 tests | PR 4a |
| 5a | docs | `docs/posture-3/move-5-design.md` | PR 4b |
| 5b | code | Move 5: MixturePrevision + ExchangeablePrevision | PR 5a |
| 6a | docs | `docs/posture-3/move-6-design.md` | PR 5b |
| 6b | code | Move 6: Execution layer (particle especially) — high-risk | PR 6a |
| 7a | docs | `docs/posture-3/move-7-design.md` | PR 6b |
| 7b | code | Move 7: condition as conditional prevision (primary) + SPEC §1 + CLAUDE.md | PR 7a |
| 8 | code | Move 8: Grammar/program-space cosmetic adaptation + **paper draft complete (every section has prose)** | PR 7b |

## Verification

End-to-end verification at the end of each code PR (no batching):

```bash
# Full Julia test suite
julia test/test_core.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_flat_mixture.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_prevision_unit.jl       # new in Move 2
julia test/test_prevision_conjugate.jl  # new in Move 4
julia test/test_persistence.jl          # new in Move 3 (loads v1 fixtures)

# Strata-1 to Strata-3 sweeps:
#   Strata-1 (unit): isapprox(atol=1e-14)
#   Strata-2 conjugate: isapprox(rtol=1e-12)
#   Strata-2 particle (seeded RNG): isapprox(rtol=1e-12) — NOT 1e-6
#   Strata-3 end-to-end: isapprox(rtol=1e-10) — halt-the-line at greater drift

# POMDP agent (separate package — Move 3 onward)
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Smoke-run a domain to confirm end-to-end semantics
julia apps/julia/grid_world/host.jl

# Lint pass — no precedent slug regressions
grep -r 'credence-lint:' . | grep -o 'precedent:[^[:space:]]*' | sort -u
# Expected: existing 7 slugs + new 3 (prevision-not-measure, conjugate-registry, exchangeable-declaration) by Move 7
```

**Skin smoke test — mandatory at end of Moves 3, 4, 6, 7.** The plan declares the JSON-RPC API surface preserved; this test proves it. Posture 2's branch surfaced a teardown flake in the skin server (issue #9, since fixed); Prevision refactoring changes what JSON-RPC serialises when a Measure handle is returned over the wire.

The test surface already exists: `apps/skin/test_skin.py` (verified) drives `create_state` / `mean` / `weights` / `condition` / `optimise` / `destroy_state` cycles through the JSON-RPC boundary including clean `skin.shutdown()` in a `finally` block. The CLAUDE.md-documented invocation is `python -m skin.test_skin` from repo root. At the end of each of Moves 3, 4, 6, 7 code PRs, run:

```bash
# CLAUDE.md-documented invocation — apps/skin/test_skin.py is the real client.
# Spawns the Julia skin server as a subprocess; tests handle teardown internally.
python -m skin.test_skin
```

If the smoke test fails (functional, teardown, or serialisation), it is halt-the-line for the current move's PR. Skin runs on Moves 3, 4, 6, 7 because those are the moves that change what crosses the JSON-RPC boundary (Measure shape in Move 3, conjugate dispatch in Move 4, particle representation in Move 6, condition primary form in Move 7). Moves 1, 2, 5, 8 are skin-invariant and the smoke test is optional there.

**If `apps/skin/test_skin.py` does not exercise a code path that a future move depends on** (e.g. `push_measure` over the wire, ParticlePrevision serialisation, ExchangeablePrevision construction), the corresponding move's design doc must extend the smoke test as a sub-task — not discover the gap during the code PR. Move 0 should audit `apps/skin/test_skin.py` against the full Moves 3/4/6/7 surface and flag gaps in `docs/posture-3/move-0-skin-surface-audit.md`.

```bash
# Python workspace stays untouched on this branch — sanity check the API still works
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/credence_router/tests/ -x
```

End-to-end paper-claim verification (after Move 8): the email-agent runs unchanged with `ExchangeablePrevision` carrying the 22 programs as ergodic components; the `FiringByTag` accessor is now an `ExchangeablePrevision` method; `condition(state, TagSet(...))` decomposes natively through the mixture (no application-level reconstruction).

## Out of scope on this branch

- **Disintegration as an axiom.** Measure-zero conditioning on continuous features stays unsupported. `disintegrate` is a future axiom extension, orthogonal to Posture 3.
- **DSL surface syntax changes.** `(expect m f)`, `(condition m k o)`, `(optimise m actions pref)` continue to parse and evaluate identically. Only `(condition-on m e)` is added.
- **Python bindings, skin layer, body work.** `apps/python/*` and `apps/skin/server.jl` see no changes. JSON-RPC API surface preserved. Python-side prevision idioms are a follow-up branch.
- **Functional / TestFunction unification.** `Functional` becomes an alias for `TestFunction` during the transition; a later cleanup pass collapses them once all sites have migrated.
- **Bit-for-bit reproduction of pre-refactor numbers.** Floating-point reassociation is expected at 1e-12. Tests use `isapprox` per the strata plan.

## Critical files (modified during the migration)

- `src/Credence.jl` (include order, exports) — Move 1.
- `src/prevision.jl` (new file) — Moves 1, 2, 4, 5, 6, 7.
- `src/ontology.jl` — Moves 2, 3, 4, 5, 6, 7. The 1,055-line file shrinks as logic moves to `prevision.jl`.
- `src/persistence.jl` — Move 3 (schema v2 migration).
- `src/eval.jl` — Move 7 (DSL `condition-on` form).
- `src/stdlib.bdsl` — Move 7 (`(condition-on m e)`).
- `src/program_space/agent_state.jl, enumeration.jl, perturbation.jl, types.jl` — Move 8 (mostly docstrings).
- `test/test_prevision_unit.jl` (new) — Move 2.
- `test/test_prevision_conjugate.jl` (new) — Move 4.
- All existing `test/*.jl` files — verification at the end of every move.
- `SPEC.md` §1 — Move 7.
- `CLAUDE.md` (frozen types, Invariants 1-2, precedents) — Move 7.
- `docs/posture-3/*.md` — every PR.

## Commit cadence guardrails

- Every session ends with a green test suite, even mid-refactor. No "will fix in next commit" states.
- When a move reveals a domain-code assumption that can't be preserved (e.g. particle filtering depending on a specific weight-normalisation point that the Prevision refactor breaks), halt and escalate. Operational equivalence is the contract.
