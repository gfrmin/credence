# Move 1 design — `Prevision` primitive type + `TestFunction` hierarchy

Status: design doc (docs-only PR 1a). Corresponding code PR is 1b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 1 — Prevision primitive type + TestFunctionSpace".

## 1. Purpose

Declare the abstract `Prevision` primitive type and the `TestFunction` hierarchy. Establish `src/prevision.jl` as the file that subsequent moves extend with dispatch (Move 2), conjugate pairs (Move 4), mixture and exchangeability (Move 5), execution-layer strategies (Move 6), and event-primary conditioning (Move 7). Move 1 *only* declares the abstract hierarchy; no dispatch, no consumer churn, no behaviour change.

The point of doing this as a standalone PR is that every subsequent move depends on these declarations loading before `src/ontology.jl`, and getting the include order wrong is the kind of fast-to-fix-now, slow-to-fix-later mistake that warrants its own PR.

## 2. Files touched

**New:**
- `src/prevision.jl` (~150 lines — shrinks with no `TestFunctionSpace`). Declares:
  - `abstract type Prevision end` with the coherence-axiom docstring (de Finetti 1974; Walley 1991).
  - `abstract type TestFunction end` with the operator-action docstring.
  - Concrete `TestFunction` subtype shells (Move 2 migrates the Functional subtypes into these): `Identity`, `Projection(index::Int)`, `NestedProjection(indices::Vector{Int})`, `Tabular(values::Vector{Float64})`, `LinearCombination(terms::Vector{Tuple{Float64, TestFunction}}, offset::Float64)`, `OpaqueClosure(f::Function)`. These are declared in Move 1 so that Move 2's migration is purely about wiring `expect` dispatch, not about creating types.
  - `Indicator(e::Event)` — new subtype bridging to Posture 2's `Event` hierarchy (loaded from `ontology.jl` for now; moves to `event.jl` if we ever split).
  - `function apply end` — abstract evaluator `apply(f::TestFunction, s) → ℝ`. Methods land in Move 2.

**Modified:**
- `src/Credence.jl:18-19`. Add `include("prevision.jl")` **before** `include("ontology.jl")`. Order matters: Measure becomes a view over Prevision in Move 3, so `Prevision` must be a loadable name when `ontology.jl` starts parsing.
- `src/Credence.jl` export list. Add `Prevision`, `TestFunction`, `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`, `Indicator`, `apply`. Note: `Identity`, `Projection`, etc. currently exist in `Ontology` — the Move 1 shells in `prevision.jl` are the *new* authoritative declarations; Move 2 turns the `Ontology` versions into aliases.

**Not touched in Move 1:**
- No existing `expect` methods move. No `condition` dispatch changes. No consumer code changes. The `Ontology` module's `Functional` type and its concrete subtypes (`Identity`, `Projection`, etc.) remain in place and functional; Move 1 simply introduces *parallel* declarations in `Prevision` that Move 2 then unifies.

## 3. Behaviour preserved

None yet — Move 1 declares abstract types only. All existing tests under `test/*.jl` must continue to pass unchanged. Strata-1 tests open at Move 2 (`test/test_prevision_unit.jl`, new).

The non-behaviour-change claim is verified by: (a) `julia -e 'push!(LOAD_PATH, "src"); using Credence'` succeeds; (b) running the existing test suite produces identical output pre- and post-Move-1. No new assertions are added.

## 4. Worked end-to-end example

At Move 1, `BetaPrevision` does not yet exist (it lands in Move 3 when `BetaMeasure` becomes a view). So the worked example traces the parallel construction path for an existing concrete Functional through the new `TestFunction` type:

```julia
# Existing code (unchanged after Move 1):
m = BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0)
expect(m, Identity())  # returns 2/5 = 0.4

# New path declared by Move 1 (does not yet dispatch):
f = Identity()                          # this is now a TestFunction subtype
                                        # (plus still a Functional subtype via
                                        # the alias Move 2 will add)
# apply(f, some_h)                      # abstract method; Move 2 adds dispatch
```

The payoff becomes visible at Move 2 when `expect(m::BetaMeasure, f::TestFunction)` dispatches through the new hierarchy instead of the old `Functional` one, and at Move 3 when `BetaMeasure` becomes a view over `BetaPrevision(2.0, 3.0)` with `expect(p::BetaPrevision, ::Identity)` defined directly on the prevision.

Move 1's worked example is therefore: "trace why declaring `TestFunction` *now* — as a new abstract type with shell concretes — is prerequisite for Move 2's migration being mechanical instead of structural." If reviewers aren't convinced by this, the fallback is to fold Move 1 into Move 2 (one PR with declarations + dispatch). See §5 Open design questions.

## 5. Open design questions

### 5.1 (GENUINE) The `TestFunctionSpace` container: parametric, field, or defer to Move 5

**The question.** Where does the metadata about "which test functions a prevision acts on, and under what closure" live?

**(a) Parametric `TestFunctionSpace{S <: Space}`.** Type-structural dispatch on source space:

```julia
struct TestFunctionSpace{S <: Space}
    source::S
    basis::Vector{TestFunction}
    closure::Symbol    # :linear, :bounded_measurable, :polynomial_k
end
```

*Cost — propagation.* A `Prevision` subtype carrying one of these must itself be parametric: `BetaPrevision{Interval}`, `ExchangeablePrevision{S}(component_space::S, ...)`. That threads `S` through the Move 4 conjugate registry's key type `(typeof(p), k.likelihood_family)`, through the Move 5 `decompose` return type, and through every `expect` method signature in Move 2. Consequential across five subsequent moves; not a local decision.

**(b) Field-based `TestFunctionSpace` with `source::Space` as an abstract field.** Swap-at-runtime friendly; dispatch falls back to abstract `Space` checks at call time.

*Cost — defeats the declared-structure argument.* Posture 3's whole argument is that declared structure in the type system catches mismatches runtime inference misses. A field-based container has to check `source::Space` compatibility at dispatch, which is exactly what the reconstruction is reconstructing away from. If we want (b), the honest question is why we introduced `TestFunctionSpace` at all — it becomes a documentation-only object the type system ignores.

**(c) No `TestFunctionSpace` at Move 1.** Ship just the `TestFunction` hierarchy. The space is inferred at dispatch from the `Measure`/`Prevision` argument in `expect(m, f)`, exactly as the current `Functional` semantics work. `Identity()`, `Projection(1)`, etc. carry no source reference; they never have. Defer `TestFunctionSpace` to the first move with a real consumer — if ever.

*Cost — one line of §1 and one bullet of §2 in this design doc drop out.* Nothing else changes.

**Stance on `ExchangeablePrevision.component_space` (Move 5): parametric.** The email agent's 22-program belief has a fixed `Program` component space; grammar perturbation grows the pool within the same space, never morphs. No post-Posture-3 application on the drawing board has a swap-at-runtime use case. `ExchangeablePrevision{S <: Space}(component_space::S, prior_on_components)` wins — type-structural dispatch downstream, no type-parameter cost actually paid by a real consumer. This commits Move 5 and lets Move 1's decision flow from it.

**Move 1 decision: (c).** No `TestFunctionSpace` at Move 1. Reasoning:

- At Move 1, *no consumer exists.* `expect(m, f::TestFunction)` at Move 2 takes a TestFunction directly; no container consulted. Auto-generated Stratum-1 tests iterate over Measure × TestFunction pairs; no container. `BetaPrevision(α, β)` at Move 3 doesn't carry one.
- At Move 5, *maybe a consumer.* If `ExchangeablePrevision`'s `decompose` method wants to carry basis+closure metadata explicitly, `TestFunctionSpace` lands then, parametric on `S <: Space` to match `ExchangeablePrevision`'s committed stance. If `decompose` turns out not to need it, `TestFunctionSpace` never lands at all — which is cleaner than carrying speculative infrastructure for five moves.
- (a) is speculative parametric propagation through five moves on the basis of "might be useful at Move 5 maybe."
- (b) undermines Posture 3's declared-structure argument and makes the container dispatch-inert.

**Invitation to argue.** The strongest counter-argument I can see is "`TestFunctionSpace` as documentation of intent, even if no code consumes it." If a reviewer thinks basis+closure metadata is load-bearing at Move 1 for some reason not yet named, surface it. Otherwise (c) is a genuine scope reduction.

### 5.2 Fold Move 1 into Move 2?

Move 1 is genuinely slim — abstract declarations only. An alternative structure combines Moves 1 and 2 into a single PR: declare the types *and* wire the dispatch at once. The master plan splits them because Move 2's Strata-1 test suite is substantial (several hundred auto-generated cases from constructor signatures) and that test-writing plus the dispatch migration is already a full-sized PR without the type declarations riding along.

**Recommendation:** keep them split. The split costs one extra PR of overhead but gives reviewers a clean "types declared, no behaviour change" landing point that makes the subsequent type-alias migration easier to reason about. The cost is low (Move 1 is the smallest of the eight) and the review clarity is real.

**Invitation to argue.** If Move 2's design doc ends up looking small because the dispatch is mechanical, folding makes more sense. Reviewers should revisit this at Move 2's design-doc PR and, if the Move 2 doc looks thin, propose the fold.

### 5.3 `Indicator(e::Event)` placement

`Indicator` depends on `Event`, which lives in `ontology.jl` (Posture 2 introduced it there). Declaring `Indicator` in `prevision.jl` means `prevision.jl` imports `Event` from `Ontology`. That is fine — `Ontology` is already loaded by `Credence.jl` — but it introduces a module dependency cycle risk if Move 3 later moves some `Ontology` types into `prevision.jl`.

**Recommendation:** declare `Indicator` in `prevision.jl`, import `Event` from `Ontology` explicitly. If Move 3 later triggers a circular dependency, we split `Event` out to its own `event.jl` at that point (not speculatively now).

**Invitation to argue.** Should `Event` move to its own file in Move 1 pre-emptively? Arguable; the cost is low but speculative. Default: no.

## 6. Risk + mitigation

**Risk R1 (low).** Include order wrong: `include("prevision.jl")` after `include("ontology.jl")` in `Credence.jl` → `Ontology`'s future `BetaMeasure(prevision=...)` field (Move 3) can't resolve `BetaPrevision`. *Caught by:* Move 3's PR will fail to compile if Move 1's include order is wrong. *Mitigation now:* Move 1's PR description calls out the include order explicitly; reviewer diffs against `src/Credence.jl` confirm `include("prevision.jl")` precedes `include("ontology.jl")`.

**Risk R2 (low).** Export name clash with `Ontology.Identity`, `Ontology.Projection`, etc. — the Move 1 `TestFunction` subtypes share names with existing `Functional` subtypes. If both are exported under the same name without qualification, `using Credence` becomes ambiguous. *Caught by:* `julia -e 'push!(LOAD_PATH, "src"); using Credence; Identity'` must print a concrete type unambiguously. *Mitigation now:* Move 1 exports the *new* `TestFunction` subtypes under `Prevision.Identity` etc.; Move 2's alias step resolves the clash by making `Ontology.Identity` an alias for `Prevision.Identity`. Until then, `Credence.jl` re-exports only the `Prevision` versions; `Ontology` versions stay module-local.

**Risk R3 (low).** The Move 5 decision to make `ExchangeablePrevision` parametric on `component_space` turns out wrong — e.g. a later application requires swap-at-runtime. *Blast radius if wrong:* the parametric-on-S type parameter has propagated into whatever subset of `ExchangeablePrevision`'s methods and subtypes Moves 5-8 produce; reversing it is a type-signature rewrite across those methods, not a local edit. *Caught by:* the first application whose use case requires swappable `component_space` — not found yet; speculative. *Mitigation now:* the stance is committed in §5.1 explicitly, so if Move 5 rediscovers the question, it's a re-litigation against a documented decision, not a fresh one. Move 1 itself carries no `TestFunctionSpace`, so there is no parametric-vs-field scaffolding at Move 1 to unwind if the Move 5 decision reverses; the risk is scoped to Moves 5-8.

## 7. Verification cadence

At end of Move 1's code PR (PR 1b):

```bash
# Compile-only — the whole contract at Move 1
julia -e 'push!(LOAD_PATH, "src"); using Credence'

# Existing test suite must pass unchanged
julia test/test_core.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_flat_mixture.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_events.jl

# POMDP agent (separate package)
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'
```

**Skin smoke test:** optional at Move 1 (Move 1 is skin-invariant per the master plan — no JSON-RPC surface change). Recommended as a sanity check; not a halt-the-line requirement.

**Strata-1 tests:** do not open at Move 1; open at Move 2.

**Lint pass:** no precedent-slug changes expected at Move 1.

Halt-the-line: any test failure at end of PR 1b is a halt. The compile-only contract makes this unlikely — Move 1 adds declarations without touching dispatch paths — but Julia's type system does sometimes surface unexpected method-dispatch ambiguities on new abstract types, and those count as halt conditions, not "fix forward" conditions.
