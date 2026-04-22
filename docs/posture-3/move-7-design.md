# Move 7 design — `condition` as conditional prevision (event-form primary, parametric-form sibling)

Status: design doc (docs-only PR 7a). Corresponding code PR is 7b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 7 — `condition` as conditional prevision (event-primary, kernel-derived)".

Working reference: `docs/posture-3/precedents.md`.

## 1. Purpose

Move 7 elevates `condition` from a Measure-level operation to a Prevision-level primitive, landing on master the philosophical pivot Posture 2 and Posture 3 together have been building toward: conditional prevision IS the primitive; parametric Bayes update is what it does when the conditioning object is a kernel+observation pair. Before Move 7, `condition(m::Measure, e::Event)` existed as a Posture 2 sibling form that expanded to `condition(m, indicator_kernel(e), true)` at the Measure level — the event form was derived. Move 7 inverts: `condition(p::Prevision, e::Event)` becomes primary; the Measure-level method becomes a thin facade.

The master plan's Move 7 section proposes the stronger reduction `condition(p, k, obs) = condition(p, ObservationEvent(k, obs))` — unifying both forms under a single event-primary primitive. §5.1 engages whether `ObservationEvent` actually fits the `Event` hierarchy's structural-predicate-over-a-space shape; the Socratic resolves to **Option B: parametric-form stays as a sibling primitive at the Prevision level**, on two technical grounds developed in §5.1.

This recommendation partially walks back the master plan's stronger framing. The paper's §1.3 claim ("conditional prevision is the primitive; parametric Bayes is derived") generalises but does not fully reduce to event-conditioning under the honest measure-theoretic treatment available without disintegration (out of scope per master plan). The §5.1 reasoning makes the case.

Move 7 is rated low-risk per the master plan. The hard work was Posture 2 (event introduction) and Move 4 (conjugate registry). Move 7 is operationally a re-routing: adding a new `condition(p::Prevision, e::Event)` method at the Prevision level, updating the Measure-level facade, and landing the SPEC.md + CLAUDE.md structural edits.

## 2. Files touched

**Modified:**

- `src/prevision.jl` — declares `condition` as a generic function at the Prevision level (currently declared implicitly at the first Measure-level method in `ontology.jl`, via Move 2's `expect` precedent). Adds `condition(p::Prevision, e::Event)` as the primary event-form method:
  ```julia
  # Conditional prevision: p(f | e) = p(f · 1_e) / p(1_e), when p(1_e) > 0.
  function condition(p::Prevision, e::Event)
      mass = expect(p, Indicator(e))
      mass > 0 || error("conditioning on measure-zero event; see disintegrate")
      # Returns a ConditionalPrevision wrapper (§5.3 committed shape).
  end
  ```
- `src/ontology.jl:1582` — `condition(m::Measure, e::Event)` becomes a thin facade over the Prevision-level method:
  ```julia
  condition(m::Measure, e::Event) = condition(m.prevision, e)  # delegates; wraps back if needed
  ```
  The 1-line body that expanded to `condition(m, indicator_kernel(e), true)` is replaced; the Prevision-level method owns the expansion.
- `src/ontology.jl` — the ~12 Measure-subtype `condition(m::<Type>, k::Kernel, obs)` methods (lines 1094–1290) stay. Parametric-form remains a sibling primitive at both the Measure and Prevision levels (§5.1 Option B).
- `src/eval.jl:23-45` — `default_env` registers `:condition-on` as a DSL form that dispatches to the event-form `condition(m, e::Event)`.
- `src/stdlib.bdsl` — adds `(condition-on m e)` for the DSL event surface. `(condition m k o)` continues to dispatch to the parametric form.
- `SPEC.md` §1 — rewrite Foundations per master plan §SPEC.md:
  - §1.1 Coherence (Dutch-book; de Finetti 1937).
  - §1.2 Prevision (operator-valued; test function spaces; σ-continuity optional).
  - §1.3 Conditional prevision (**primary form: event; parametric-form as sibling primitive**, §5.1 Option B). Names disintegration as the out-of-scope path to full unification.
  - §1.4 Exchangeability and the representation theorem.
  - §1.5 Complexity prior (prevision over programs).
  - §1.6 Alignment (CIRL) — prevision over utility functions.
- `CLAUDE.md` — structural edits per master plan:
  - Frozen types: **four** (Space, Prevision, Event, Kernel). Measure declared as a view over Prevision (continues Move 3's framing; Move 7 makes it constitutional).
  - Invariant 1 strengthens: exactly one axiom-constrained `condition` at the Prevision level, with two primary forms (event, parametric) as peer primitives.
  - New precedent slugs: `prevision-not-measure`, `event-primary-condition`, `parametric-form-sibling`. Each under `## Precedents`.

**New:**

- `src/prevision.jl` — `ConditionalPrevision(base::Prevision, event::Event)` struct that represents the conditional prevision `p | e`. Evaluation deferred to `expect(::ConditionalPrevision, f)` which computes `expect(base, Indicator(e) * f) / expect(base, Indicator(e))` — the de Finetti-Whittle form. §5.3 commits the shape.

**Not touched in Move 7:**

- Measure-subtype `condition(m::<Type>, k::Kernel, obs)` methods stay at their current arithmetic — the parametric form is preserved byte-for-byte. Move 4's registry continues to fire; Move 5's MixturePrevision coordinator continues to iterate; Move 6's typed carriers continue to wrap.
- The ~50 `condition(m, k, obs)` call sites across `src/`, `test/`, `apps/` stay unchanged — the parametric form is still public API (§5.1 Option B).

## 3. Behaviour preserved

Stratum-1, Stratum-2, and Stratum-3 tests pass unchanged. Tolerances per `precedents.md` §4:
- **Event-form conditioning** (`condition(m, e::Event)`): bit-exact `==` with pre-Move-7 behaviour. The Posture 2 gate-4 test already asserts `condition(m, e)` ≡ `condition(m, indicator_kernel(e), true)`; Move 7 preserves that equivalence by construction (event-form owns the expansion, parametric fallback unchanged).
- **Parametric-form conditioning**: bit-exact `==` with pre-Move-7. No arithmetic change; the ~12 Measure-subtype methods are untouched.
- **New `ConditionalPrevision`**: `expect(ConditionalPrevision(p, e), f)` must equal `expect(p, Indicator(e) * f) / expect(p, Indicator(e))` per the Whittle formula. Stratum-1 tolerance `atol=1e-14` on derived scalars; `==` where arithmetic is closed-form (e.g. event with mass 1 → `ConditionalPrevision(p, e) ≡ p`).

No pre-Move-7 behaviour changes; the pivot is additive at the Prevision level. The SPEC.md and CLAUDE.md edits reframe the documentation without changing operational semantics.

## 4. Worked end-to-end example

**Inputs:** a 3-component MixtureMeasure with tagged components; a `TagSet` event; assert the event-form condition produces the same posterior as the parametric form through `indicator_kernel`.

```julia
# From Move 5's worked example.
comp1 = TaggedBetaMeasure(Interval(0, 1), 1, BetaMeasure(2.0, 3.0))
comp2 = TaggedBetaMeasure(Interval(0, 1), 2, BetaMeasure(5.0, 5.0))
comp3 = TaggedBetaMeasure(Interval(0, 1), 3, BetaMeasure(1.0, 4.0))
mix = MixtureMeasure(Interval(0, 1), [comp1, comp2, comp3], [0.0, 0.0, 0.0])

e = TagSet(Set([1, 3]))   # condition on "tag is 1 or 3"
```

**Step-by-step dispatch (Move 7):**

```julia
condition(mix, e)
  ↓
# 1. Measure-level facade (ontology.jl) delegates to Prevision-level.
condition(mix.prevision, e)
  ↓
# 2. Prevision-level primary (prevision.jl).
#    Computes mass = expect(p, Indicator(e)); guards against measure-zero;
#    constructs ConditionalPrevision(p, e).
mass = expect(mix.prevision, Indicator(TagSet(Set([1, 3]))))
#     = mix.prevision.log_weights dot (is-tag-1-or-3 ? 1 : 0) per component
#     = 2/3 (components 1 and 3 fire; uniform weights; mass = 2/3).
# mass > 0 → proceed.

# 3. Return ConditionalPrevision(mix.prevision, e). Evaluation deferred.
# The MixtureMeasure facade wraps back: MixtureMeasure(mix.space, cp_components, cp_log_weights)
# after evaluation. For this TagSet case the evaluation reduces to restricting
# the mixture to its tag-1 and tag-3 components and re-normalising — equivalent
# to condition(mix, indicator_kernel(TagSet(Set([1,3]))), true).
```

**Equivalence trace with parametric form:**

```julia
condition(mix, e) ≡ condition(mix, indicator_kernel(e), true)
```

Posture 2 gate-4 (commit `5c7f63f`) asserts this equivalence at test time; Move 7 preserves it by construction. Di Lavore–Román–Sobociński Proposition 4.9 grounds the equivalence for deterministic events.

**Dual-residency trace.** Move 7 introduces one new dispatch path (`condition(p::Prevision, e::Event)`) and retains the old (`condition(p, k, obs)`). Both are primary at the Prevision level; neither derives from the other. The Measure-level facade delegates to the Prevision level for both forms. No vestigial homes: each method serves its structural role.

## 5. Open design questions

### 5.1 (substantive — THE live Socratic) Is `ObservationEvent` an Event, or is parametric-form a sibling primitive?

The master plan proposes `condition(p, k, obs) = condition(p, ObservationEvent(k, obs))` — unifying both forms under event-primary dispatch. The Socratic raised in the master plan itself: does `ObservationEvent` structurally fit the `Event` hierarchy's shape?

Posture 2's `Event` types (`TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`) are **structural predicates over a Space** — each declares a subset of the hypothesis space. `indicator_kernel(e::Event)` maps each `Event` to a deterministic `Kernel` into `BOOLEAN_SPACE`. The event's coherent semantics depend on the subset being well-defined as a measurable set, and `p(1_e)` being well-defined as an expectation against that indicator.

`ObservationEvent(k, obs)` is categorically different. It carries a kernel and an observation value, encoding "this kernel would produce this observation at this hypothesis." At each hypothesis `h`, the kernel's log-density `k.log_density(h, obs)` is a real-valued likelihood weight, not a Boolean indicator. The "event" of `{k emits exactly obs}` is a real-valued density, which is not a subset of hypothesis space.

- **Option A (committed derivation, master-plan default):** `ObservationEvent` is an `Event`; `condition(p, k, obs) = condition(p, ObservationEvent(k, obs))`. Requires `indicator_kernel(::ObservationEvent)` witness.
- **Option B (parametric-form is a sibling primitive):** `condition(p::Prevision, e::Event)` and `condition(p::Prevision, k::Kernel, obs)` are BOTH primary forms at the Prevision level; neither derives from the other. They are provably equivalent for deterministic events (DLRS Prop. 4.9); for continuous/stochastic observations they diverge without disintegration.
- **Option C (hybrid):** `ObservationEvent` exists as an `Event` subtype, but only for deterministic kernels where the reduction is well-defined; non-deterministic kernels keep the parametric sibling primitive.

**Recommendation: B.**

Two technical reasons:

1. **Continuous-kernel measure-zero problem.** For continuous observation spaces with continuous kernels (GaussianMeasure + NormalNormal kernel on real-valued `obs`; BetaMeasure + continuous `quality` kernel), the event `{k emits exactly obs}` has Lebesgue measure zero. Event-conditioning requires `p(1_e) > 0` to be well-defined; continuous observations violate this. Reducing `condition(p, k, obs)` to `condition(p, ObservationEvent(k, obs))` in these cases requires disintegration — the master plan explicitly scopes disintegration out ("disintegration as an axiom... stays unsupported... is a future axiom extension, orthogonal to Posture 3"). Option A ships sugar that papers over a mathematically undefined reduction; Option B ships the honest structure.

2. **Invariant 2 declared-structure coherence.** The `Event` hierarchy's members carry declared subset-of-space information: `TagSet.fires::Set{Int}`, `FeatureEquals.feature + .value`, `Conjunction.left + .right`. Each member is a typed predicate the type system can reason about. `ObservationEvent(k, obs)` would be a `Kernel` + observation value — a likelihood-structured object with stochastic semantics, categorically different from the typed predicates. Forcing it into the hierarchy strains Invariant 2's declared-structure discipline; the type system's semantic coherence pays for the philosophical unification. Option B keeps the hierarchy coherent.

**Invitation to argue.** Option A becomes correct if the master plan's exclusion of disintegration is revisited — a future Move 9 or Posture 4 could land disintegration as an axiom extension, at which point `condition(p, ObservationEvent(k, obs))` becomes well-defined for continuous cases and Option A's unification claim is recoverable. Option C addresses the continuous case by scope-restricting `ObservationEvent` to deterministic kernels, but introduces a split dispatch surface for a half-unification that doesn't deliver the master plan's full pivot. Commit B; revisit if disintegration lands.

**Paper implication.** Paper §1.3's "conditional prevision is the primitive; parametric Bayes is derived" framing needs amendment under Option B. The honest phrasing: "conditional prevision is the primitive; parametric-form conditioning is a peer primary form, equivalent to event-form conditioning on deterministic events (DLRS Prop. 4.9), with disintegration as the out-of-scope path to full reduction." The paper's philosophical pivot is narrower than the master plan's original framing but mathematically honest. The paper draft's §1.3 edit lands with Move 7's code PR.

### 5.2 (substantive) Frozen-layer elevation of `Event`; demotion of `Measure` to view

The master plan's CLAUDE.md edit names the frozen types as **four** post-Move-7: Space, Prevision, Event, Kernel. `Measure` becomes a declared view over `Prevision` (continuation of Move 3's framing; Move 7 makes it constitutional). This is a CLAUDE.md surgery, not just a documentation update — the frozen-types section defines what's non-negotiable.

**The question.** Is this elevation justified structurally, or is it framing-inflation?

Technical case for elevation:
- `Event` has been a first-class declared type since Posture 2 (gate-1, commit `1d54b94`). Move 7 adds `condition(p::Prevision, e::Event)` as an axiom-constrained primary form — Event is now on the dispatch surface of a frozen function.
- `Prevision` is the de Finettian primitive that Posture 3 is built on; Measure-as-view is the honest reframing (Move 3 landed the wrapping; Move 7 makes it doctrinal).

Technical case against elevation:
- Adding to the frozen layer raises the bar for future extensions. If `Event`'s hierarchy needs to be extended (new predicate types; observation-witness types under a future disintegration framework), the extension is constitutional rather than stdlib.
- `Measure` remaining "frozen" despite being a view is also defensible: consumer code treats Measure as primary; demoting to "view" in CLAUDE.md might produce confusion for readers coming from Kolmogorov-framed material.

**Recommendation: elevate.** One technical reason, load-bearing:

1. **Dispatch surface alignment.** A function whose primary dispatch is on a type should have that type in the frozen layer. Post-Move-7, `condition(p::Prevision, e::Event)` is a primary axiom-constrained form; `Event` dispatch is constitutional by usage. Keeping `Event` stdlib while making it constitutional by dispatch is a framing-to-semantics mismatch — the frozen layer defines what axiom-constrained functions dispatch on, and Event now qualifies.

**Invitation to argue.** If a reviewer thinks `Event` should stay stdlib, the case would be that Event's hierarchy is narrow enough that constitutional protection over-specifies — `TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement` are not "all possible events an agent might condition on," they're the six Posture 2 shipped. Future events (e.g. `WithinEpsilon(feature, value, ε)`) should be stdlib extensions. That argument is about extensibility, not about Event's current structural role. Respond by affirming that constitutional elevation is about the dispatch surface of the frozen functions, not about exhaustive enumeration of Event's subtypes — subtypes extend naturally under the constitutional frame.

### 5.3 (calibrating) Shape of the returned `ConditionalPrevision`

`condition(p::Prevision, e::Event)` returns... what? Three candidate shapes:

- **Option A (lazy wrapper):** `ConditionalPrevision(base, event)` struct that defers evaluation. `expect(cp, f)` computes `expect(base, Indicator(e) * f) / expect(base, Indicator(e))` lazily.
- **Option B (eager restriction):** For cases where the event's indicator has a closed-form effect on the prevision (MixturePrevision restricted to firing tags; CategoricalMeasure restricted to event-satisfying values), return the restricted prevision directly. For other cases, return a lazy wrapper.
- **Option C (always eager):** Materialise the conditional prevision fully; no wrapper type.

**Recommendation: B (eager restriction where closed-form exists; lazy wrapper otherwise).**

Two technical reasons:

1. **MixturePrevision and CategoricalPrevision have closed-form event-restriction.** For a MixturePrevision conditioned on `TagSet(S)`, the posterior is a MixturePrevision with components whose tags are in `S`, weights re-normalised. Materialising this eagerly returns the same type the user started with; wrapping in `ConditionalPrevision` would lose type information and force consumers through the wrapper's dispatch.

2. **Lazy wrapper preserves generality for non-closed-form cases.** Continuous-space events (e.g. `FeatureInterval` on a continuous feature) don't always admit a closed-form restriction; lazy wrapping defers the computation to `expect` calls where the integrand determines the quadrature strategy.

**Type-stability trade-off named explicitly.** Option B trades consumer-side type-stability for type-preservation in closed-form cases. `condition(p::Prevision, e::Event)` returns `Prevision` but the concrete subtype varies by `(typeof(p), typeof(e))` — a `MixturePrevision` conditioned on `TagSet` returns `MixturePrevision`; the same prior conditioned on a continuous `FeatureInterval` returns `ConditionalPrevision`. Callers requiring a specific concrete subtype (e.g. `.components` access on `MixturePrevision`, or the isa-drilldown pattern from Move 6 §5.3) must dispatch accordingly; generic-Prevision callers using `expect` are unaffected because `expect` is uniform across subtypes. The trade is acceptable because most consumers already dispatch on the concrete type, but it is a real discipline cost — not hidden, not free.

**Invitation to argue.** Option A (always lazy) is the simplest implementation — one code path, uniform return type — but sacrifices the type-preserving property for closed-form cases. If downstream consumer surveys in Moves 7/8 reveal that the isa-drilldown pattern is rarely exercised in practice, the type-stability argument for Option A strengthens. Option C (always eager) would require implementing event-restriction for every Prevision subtype; lands more code for uncertain benefit. Commit B; the split is along a natural structural line (closed-form restriction ↔ eager return; non-closed-form ↔ lazy wrapper), and the type-stability cost is explicit rather than surprising.

## 6. Risk + mitigation

**Risk R1 (low): dispatch ambiguity between Measure-facade and Prevision-primary methods.** Move 7 adds `condition(p::Prevision, e::Event)` at the Prevision level; the existing `condition(m::Measure, e::Event)` becomes a facade delegating to the Prevision form. Julia's method-table must route correctly: a `condition(mix, e)` call where `mix isa MixtureMeasure` should hit the Measure facade, which delegates to the Prevision-level method on `mix.prevision`. *Caught by:* existing `test/test_events.jl` (35 assertions); Posture 2 gate-4 equivalence test at `test/test_core.jl`.

**Risk R2 (low): pre-emptive grep for `condition(…, e::Event)` call sites + ObservationEvent pre-Option-B expectation check.** Pattern search 2026-04-22 across `src/`, `test/`, `apps/`, `docs/`:

| Target | Hits | Category (a) | (b) | (c) |
|--------|-----|--------------|-----|-----|
| `condition(m, e::Event)` call sites (Measure-level sibling form) | 2 src + 3 test | All covered by the Measure facade delegating to the Prevision primary | 0 | 0 |
| `condition(m::<Type>, k::Kernel, obs)` call sites (parametric form; unchanged under Option B) | 50+ across src/test/apps | All unchanged — parametric form is sibling primitive | 0 | 0 |
| `indicator_kernel` call sites (Posture 2) | 1 src + 3 test | All unchanged — the Prevision-level primary uses `Indicator(e)` at the expect level, not `indicator_kernel(e)` as a kernel construction | 0 | 0 |
| `ObservationEvent` references (expected zero under Option B) | 1 src + 4 doc | See disposition below | 0 | 0 |

**`ObservationEvent` disposition, per-hit.** Under Option B the type `ObservationEvent` does not exist; the grep confirms the expected state — zero executable call sites, no test references, no apps code.

- `src/prevision.jl:547` — docstring comment in `function expect end` declaration reading "Move 7 inverts the derivation (`expect(p, f)` becomes the primitive, `condition(p, k, obs)` derived via `Indicator(ObservationEvent(k, obs))`)." **Needs one-line update in code PR 7b** to reflect Option B: either remove the forward-looking reference or reword as "Move 7 elevates `condition` to the Prevision level; see move-7-design.md §5.1 for the parametric-form-as-sibling-primitive resolution."
- `docs/posture-3/master-plan.md:230–238` — master plan's Option A proposal. Historical/expected; the design doc's role is to resolve the Socratic the master plan posed. No update needed.
- `docs/posture-3/paper-draft.md:92,94` — paper §2 currently echoes the master plan's Option A framing. **§1.3 amendment lands in code PR 7b** per §5.1 paper implication; these two lines rewrite at that point.
- `docs/posture-3/move-0-skin-surface-audit.md:108,128` — Move 0's audit prose references the Socratic. Historical; no update needed.
- `docs/posture-3/move-7-design.md` (this file) — expected; the design doc discusses Options A/B/C with `ObservationEvent` as the rejected type.

Go/no-go gate: **GO.** 100% (a); the one executable reference (`src/prevision.jl:547` docstring) carries a known-update disposition for PR 7b rather than indicating any consumer site writing code against the rejected Option A framing.

**Risk R3 (medium): paper §1.3 amendment required under Option B.** The paper draft currently echoes the master plan's Option A framing. Under the committed Option B, §1.3 must be reworded to frame parametric-form as a peer primary (not derived). *Mitigation:* paper-draft edit lands in the Move 7 code PR (7b), not as a follow-up — the paper-is-gating-artifact posture requires this. Reviewer of PR 7b checks that §1.3 names disintegration as the out-of-scope path to full reduction, and acknowledges parametric-form as sibling rather than derived.

**Risk R4 (low): CLAUDE.md frozen-types edit is constitutional, not cosmetic.** Elevating `Event` to the frozen layer and demoting `Measure` to a view reframes the repo's constitution. If a reviewer believes the elevation is premature or the demotion confusing, the CLAUDE.md edit can be scoped back (leave `Measure` in the frozen layer; add `Event` as a new entry). *Mitigation:* the design doc commits to elevation in §5.2 with two technical reasons; PR review has the Socratic framed explicitly. If the reviewer prefers a narrower edit, §5.2 accommodates.

**Risk R5 (low): skin smoke extensions at Move 7.** Per master plan §Verification, Moves 3, 4, 6, 7 are mandatory-skin-smoke. Move 7 changes the wire path for `condition(m, e::Event)` (now delegates through Prevision level). *Caught by:* new `test_condition_on_event` in `apps/skin/test_skin.py` — creates a MixtureMeasure, conditions on a TagSet event via the new `condition_on_event` RPC, asserts posterior weights match the equivalent `condition(m, indicator_kernel(e), true)` result bit-exactly. Plus `test_event_kernel_equivalence`: two side-by-side conditions of the same prior (one via event form, one via kernel form), assert resulting states have `==` weights. Both tests per Move 0 skin audit §Move 7.

## 7. Verification cadence

At end of Move 7's code PR (7b):

```bash
# Existing suite — must pass unchanged.
julia test/test_core.jl
julia test/test_events.jl         # 35 assertions covering event-form condition.
julia test/test_flat_mixture.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_persistence.jl
julia test/test_prevision_unit.jl
julia test/test_prevision_conjugate.jl
julia test/test_prevision_mixture.jl
julia test/test_prevision_particle.jl

# POMDP agent.
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke — MANDATORY at Move 7. Adds test_condition_on_event +
# test_event_kernel_equivalence per Move 0 audit §Move 7.
python -m skin.test_skin
```

**Halt-the-line conditions:**

- Any `test/test_events.jl` regression — the Posture-2 event-form equivalence must survive the dispatch inversion.
- Any skin smoke failure on the new `condition_on_event` / `event_kernel_equivalence` tests.
- Any existing test regression below `rtol=1e-10` (Stratum 3).
- Any POMDP agent test failure.

Per `precedents.md` §8 (checkpoint-per-phase): each phase of the code PR lands as its own commit, each commit leaves the branch green. Estimated 4–5 phases — (i) Prevision-level `condition(p, e::Event)` + `ConditionalPrevision` struct; (ii) Measure facade delegation; (iii) DSL `(condition-on m e)` + `:condition-on` registration; (iv) SPEC.md §1 rewrite + CLAUDE.md frozen-types edit + paper §1.3 amendment; (v) skin smoke extensions.
