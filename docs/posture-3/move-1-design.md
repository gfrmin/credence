# Move 1 design — `Prevision` primitive type + `TestFunction` hierarchy

Status: design doc (docs-only PR 1a). Corresponding code PR is 1b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 1 — Prevision primitive type + TestFunctionSpace".

## 1. Purpose

Declare the abstract `Prevision` primitive type, the `TestFunction` hierarchy, and the `TestFunctionSpace` container. Establish `src/prevision.jl` as the file that subsequent moves extend — Move 2 adds the `expect` dispatch, Move 4 adds `ConjugatePrevision{Prior, Likelihood}`, Move 5 adds `MixturePrevision` and `ExchangeablePrevision`, Move 6 adds `ParticlePrevision` and `QuadraturePrevision`, Move 7 adds the primary form of `condition`. Move 1 *only* declares the abstract hierarchy and the `TestFunctionSpace` container; no dispatch, no consumer churn, no behaviour change.

The point of doing this as a standalone PR is that every subsequent move depends on these declarations loading before `src/ontology.jl`, and getting the include order wrong is the kind of fast-to-fix-now, slow-to-fix-later mistake that warrants its own PR.

## 2. Files touched

**New:**
- `src/prevision.jl` (~200 lines). Declares:
  - `abstract type Prevision end` with the coherence-axiom docstring (de Finetti 1974; Walley 1991).
  - `abstract type TestFunction end` with the operator-action docstring.
  - Concrete `TestFunction` subtype shells (Move 2 migrates the Functional subtypes into these): `Identity`, `Projection(index::Int)`, `NestedProjection(indices::Vector{Int})`, `Tabular(values::Vector{Float64})`, `LinearCombination(terms::Vector{Tuple{Float64, TestFunction}}, offset::Float64)`, `OpaqueClosure(f::Function)`. These are declared in Move 1 so that Move 2's migration is purely about wiring `expect` dispatch, not about creating types.
  - `Indicator(e::Event)` — new subtype bridging to Posture 2's `Event` hierarchy (loaded from `ontology.jl` for now; moves to `event.jl` if we ever split).
  - `struct TestFunctionSpace` — the declared domain a Prevision acts on. Fields TBD pending §5 Open design questions.
  - `function apply end` — abstract evaluator `apply(f::TestFunction, s) → ℝ`. Methods land in Move 2.

**Modified:**
- `src/Credence.jl:18-19`. Add `include("prevision.jl")` **before** `include("ontology.jl")`. Order matters: Measure becomes a view over Prevision in Move 3, so `Prevision` must be a loadable name when `ontology.jl` starts parsing.
- `src/Credence.jl` export list. Add `Prevision`, `TestFunction`, `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`, `Indicator`, `TestFunctionSpace`, `apply`. Note: `Identity`, `Projection`, etc. currently exist in `Ontology` — the Move 1 shells in `prevision.jl` are the *new* authoritative declarations; Move 2 turns the `Ontology` versions into aliases.

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
ts = TestFunctionSpace(...)             # Move 1 declares the container
# apply(f, some_h)                      # abstract method; Move 2 adds dispatch
```

The payoff becomes visible at Move 2 when `expect(m::BetaMeasure, f::TestFunction)` dispatches through the new hierarchy instead of the old `Functional` one, and at Move 3 when `BetaMeasure` becomes a view over `BetaPrevision(2.0, 3.0)` with `expect(p::BetaPrevision, ::Identity)` defined directly on the prevision.

Move 1's worked example is therefore: "trace why declaring `TestFunction` *now* — as a new abstract type with shell concretes — is prerequisite for Move 2's migration being mechanical instead of structural." If reviewers aren't convinced by this, the fallback is to fold Move 1 into Move 2 (one PR with declarations + dispatch). See §5 Open design questions.

## 5. Open design questions

### 5.1 (GENUINE) `TestFunctionSpace` parametric vs field

**The question.** Should `TestFunctionSpace` be parametric on the Space type:
```julia
struct TestFunctionSpace{S <: Space}
    source::S
    basis::Vector{TestFunction}
    closure::Symbol    # :linear, :bounded_measurable, :polynomial_k
end
```
or carry the Space as an abstract field:
```julia
struct TestFunctionSpace
    source::Space
    basis::Vector{TestFunction}
    closure::Symbol
end
```

**Trade-off:**
- **Parametric form** preserves type-structural dispatch: `TestFunctionSpace{Interval}` is a different type from `TestFunctionSpace{Finite{T}}`, so a future `apply(f, s, ::TestFunctionSpace{Interval})` method dispatches by type, not by runtime check. Catches space/test-function mismatches at compile time. Cost: Julia type parameter explosion when mixing (rare; we don't currently mix).
- **Field form** trades type-structural dispatch for the ability to have test function spaces spanning multiple base spaces (e.g. a joint distribution's test functions that navigate a `ProductSpace`). Simpler to write; slower dispatch.

**Recommendation:** parametric. The Posture 3 reconstruction's whole argument is that declared structure in the type system catches dispatch mismatches that field-based structure can't. Making `TestFunctionSpace` parametric is coherent with that argument. The concrete mixing case that field form serves — `ProductSpace` test functions — is already handled by `NestedProjection`, which navigates nested product structure at the TestFunction level, not the Space level.

**Invitation to argue.** Is there a case in Moves 3-7 where a field-based `TestFunctionSpace` is cleaner? Specifically: does `ExchangeablePrevision(component_space, prior_on_components)` from Move 5 want a field-based space so the `component_space` can be swapped without reconstructing the prevision's type? If yes, the trade-off re-opens.

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

**Risk R3 (low).** The `TestFunctionSpace` parametric-vs-field decision is wrong. *Caught by:* Move 3+ reveals the field form was needed (e.g. for swappable `component_space` in `ExchangeablePrevision`). *Mitigation:* changing a struct from parametric to field (or vice versa) in a later move is a small surgical diff (one struct decl, maybe a few constructor call sites). Not worth blocking Move 1 to pre-decide.

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
