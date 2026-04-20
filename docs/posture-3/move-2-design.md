# Move 2 design — `expect` as definitional; `Functional` → `TestFunction` alias migration; Stratum-1 opens

Status: design doc (docs-only PR 2a). Corresponding code PR is 2b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 2 — `expect` as definitional; per-Prevision `expect` methods".

## 1. Purpose

Move 2 makes `expect` definitional for `Prevision` by declaring `function expect end` in the `Previsions` module and having `Ontology` import it; unifies the `Functional` hierarchy in `ontology.jl` with the `TestFunction` hierarchy in `prevision.jl` via `const Functional = TestFunction` plus per-subtype aliases; and opens the Stratum-1 test suite (`test/test_prevision_unit.jl`) that pins the axiom-constrained expected values for every `(Measure, TestFunction)` pair the existing dispatch matrix covers.

The Move 2 / Move 1 split earned its keep: Move 1 came in at the predicted scoped-tight size (182 lines, type declarations only; no behaviour change). Move 1's §5.2 "fold Move 1 into Move 2?" question does not reopen — the split let reviewers land Move 1 as a clean "types declared, no behaviour change" PR and lets Move 2 focus on the dispatch migration and the substantial Stratum-1 corpus.

## 2. Files touched

**New:**
- `test/test_prevision_unit.jl` — Stratum-1 corpus. Iterates over declarative `(Measure constructor, TestFunction value)` table pairs; asserts the expected closed-form / quadrature / Monte-Carlo / OpaqueClosure values per §3. Matches the codebase's println-assertion house style.

**Modified:**
- `src/prevision.jl` — add `function expect end` declaration with the de Finettian docstring; extend the `export` list to include `expect` and the shell concretes (`Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`) that Move 1 kept unexported to avoid clash.
- `src/ontology.jl:657-701` — delete the abstract `Functional` declaration and its six concrete subtypes. Replace with imports from `Previsions` plus `const Functional = TestFunction`:
  ```julia
  import ..Previsions: expect, TestFunction, Identity, Projection, NestedProjection,
                       Tabular, LinearCombination, OpaqueClosure
  const Functional = TestFunction
  ```
- `src/ontology.jl:704-770` — the existing 16 dispatch methods (`expect(::CategoricalMeasure, ::Identity)` through `expect(::MixtureMeasure, ::OpaqueClosure; kwargs...)`) keep their bodies unchanged; only the resolution of the type annotations changes (now `Previsions.Identity` via the import).
- `src/ontology.jl` export list — keep `Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure` exported as before; they now resolve via the alias chain.
- `src/Credence.jl` — remove the duplicate `Identity, Projection, ...` re-export on line 42 (`Functional`-family exports coming from `Ontology` collide with the now-exported-from-`Previsions` concretes; pick one source, and since the aliases mean the types are identical, `Previsions` is the canonical source). Net change: move those names off the `Functional`-family export line onto the `Previsions` line, or consolidate.

**Not touched in Move 2:**
- Any `Measure` subtype definition. Move 3 wraps `Measure` around `Prevision`; Move 2 is dispatch-only.
- Any application code in `apps/`. The `Functional` alias keeps consumer-visible names stable.

## 3. Behaviour preserved

The point of Stratum-1 at Move 2 is not to prove Move 2 is operationally equivalent — the method bodies are unchanged, so equivalence is by construction. The point is to *pin the expected values now* so Moves 3–7 have a tripwire for silent drift when they refactor around the dispatch.

**Four tolerance cases, deliberately distinct because the underlying arithmetic differs:**

- **Closed-form methods** (`expect(::BetaMeasure, ::Identity) = α/(α+β)`, etc.): **`==`**. These are the 4 one-liner methods at `src/ontology.jl:704-707` plus the 6 Projection/NestedProjection/Tabular closed forms at 715-761. Same arithmetic pre- and post-refactor; bit-exact.

- **Quadrature paths** (`expect(::BetaMeasure, f; n=64)` and similar): **`isapprox(atol=1e-14)`**. Grid midpoints, weights, and reduction order are unchanged, but Julia's `sum()` is allowed internal pairwise-reduction reorderings; 1e-14 is the floating-point reassociation budget, stricter misfires on legitimate reorders.

- **Monte Carlo paths under deterministic seeding** (`expect(::DirichletMeasure, f; n_samples=1000)`, `expect(::NormalGammaMeasure, f; n_samples=1000)`, `expect(::ProductMeasure, f; n_samples=1000)`): **`==`**. Seeded Monte Carlo is deterministic by construction; the same seed + same samples + same arithmetic must produce the same result. If a seeded MC path drifts under the alias migration — where no arithmetic has changed — that is a bug (likely a subtle RNG-consumption-order change), not floating-point reassociation. Tests set the RNG seed explicitly at the start of each MC case; assertion is `==`.

- **`OpaqueClosure` fallback methods** (method delegation `expect(m, o.f; kwargs...)`): **`==`** to the direct-`Function`-argument call. See §5.3 for why this is load-bearing rather than tautological.

The Monte Carlo / quadrature separation is the precedent Move 6 will inherit. Move 6 refactors particle filtering; the particle path is Monte Carlo under deterministic seeding; Strata-2 tolerance is `rtol=1e-12` (per the master plan's Move 6 correction, for arithmetic reassociation from the Prevision wrapper). Strata-1 at Move 2 sets the `==`-for-seeded-MC foundation that argument builds on.

## 4. Worked end-to-end example

**Pre-refactor:**
```julia
# src/ontology.jl:659 declares `struct Identity <: Functional end`
# src/ontology.jl:704 declares `expect(m::BetaMeasure, ::Identity) = m.alpha / (m.alpha + m.beta)`
expect(BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0), Identity())
# → 2.0 / 5.0 = 0.4 (exact, bit-identical to 2/5)
```

**Post-Move-2:**
```julia
# src/prevision.jl declares `abstract type TestFunction end` and `struct Identity <: TestFunction end`
# src/ontology.jl imports: `import ..Previsions: Identity, TestFunction, expect`
# src/ontology.jl declares: `const Functional = TestFunction`
# src/ontology.jl:704 method `expect(m::BetaMeasure, ::Identity) = m.alpha / (m.alpha + m.beta)`
#   — same body, same dispatch signature (Previsions.Identity), same arithmetic
expect(BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0), Identity())
# → 2.0 / 5.0 = 0.4 (exact, bit-identical to pre-refactor; Stratum-1 asserts ==)
```

**Method-table trace, step by step.** Julia's dispatch at `expect(BetaMeasure(...), Identity())`:

1. Caller's `Identity` resolves in `Credence` scope. Both `Previsions.Identity` and (post-alias) `Ontology.Identity` are in scope; the alias `const Identity = Previsions.Identity` means they point at the same type object.
2. `Identity()` constructs an instance of `Previsions.Identity` — the singular concrete type.
3. Julia's method-table lookup for `expect` starts in `Previsions` (since `expect` is declared there post-Move-2) and finds the method `expect(::BetaMeasure, ::Previsions.Identity)` attached to that generic. The alias chain is transparent — `Ontology.Identity === Previsions.Identity` is resolved via structural equality, not via a separate fallback path.
4. Dispatch lands in the closed-form body `m.alpha / (m.alpha + m.beta)` at `src/ontology.jl:704`. Same arithmetic as pre-refactor; Stratum-1 asserts `==`.

The alias is invisible to the dispatch mechanism and to the caller. It is visible only to documentation tools and `typeof` introspection — both unchanged because `typeof(Identity()) === Previsions.Identity` whether the `Identity` name came from `Ontology` or `Previsions`.

## 5. Open design questions

### 5.1 (substantive) Alias strategy — keep `const Functional = TestFunction` or retire `Functional` entirely?

- *Option A (conservative):* `Ontology` retains `const Functional = TestFunction` plus const aliases for each concrete subtype. `Ontology`'s export list stays the same. Existing consumers (tests, apps, POMDP agent package) that import from `Credence` or `Ontology` see no name change. The `Functional` name becomes an alias with indefinite lifetime; a future cleanup pass collapses it when all sites have migrated to `TestFunction`.
- *Option B (clean):* Retire `Functional` outright at Move 2. `Ontology` removes `Functional` from exports; consumers must migrate to `TestFunction`. A one-time find-and-replace across `apps/julia/*`, `test/*.jl`, and the POMDP agent package.

**Recommendation: A.** The aliasing overhead is one `const` declaration; the churn cost of Option B is a multi-file find-and-replace that provides no operational benefit at Move 2 (the types are literally identical). Option B's cleanup is the long-term-debt resolution from the master plan's "Out of scope — Functional/TestFunction unification" clause, and it should land in the *final* Move 8 cleanup pass or as a separate follow-up PR after all eight moves merge, not mid-sequence.

**Invitation to argue.** Two legitimate cases for Option B:
- *Confusion hazard:* a specific consumer where `Functional` as an alias creates a documentation or dispatch conflict (docstrings that contradict themselves, error messages that name the wrong type). Not currently identified.
- *Deferred-work debt:* the master plan's "final cleanup pass" presumes a future find-and-replace when the aliasing period ends. Indefinite-lifetime aliases accumulate carrying cost (confused readers, outdated tutorials, stale error messages). A short aliasing window is easier to clean up later than an open-ended one. Option B at Move 2 converts "indefinite-lifetime alias" into "one mid-stream PR of find-and-replace churn" — arguably cleaner than deferring if downstream moves don't need the `Functional` name for any reason.

My view remains Option A: the churn is real cost now, nothing in Moves 3–7 needs `TestFunction` to be the only name, and the cleanup pass is a known quantity. But the counter-case deserves a fair statement; a reviewer who weighs deferred-work-debt higher than current-PR churn has a legitimate argument for B.

### 5.2 (substantive) Stratum-1 corpus generation strategy

- *Option A:* Hand-written `test/test_prevision_unit.jl` matching the codebase's println-assertion style (see `test/test_core.jl`). One explicit assertion per `(Measure, TestFunction)` pair.
- *Option B:* Macro-generated — a `@test_all_pairs CONSTRUCTORS FUNCTIONALS` macro expands at compile time into assertion blocks.
- *Option C:* Runtime iteration over declarative tables — `const CONSTRUCTORS = [...]` and `const FUNCTIONALS = [...]` in the test file; a single loop body evaluates `expect(c(), f)` and compares against a recorded expected value.

**Recommendation: C.** Matches the codebase's existing style (println-assertion is the house pattern; `@testset` is not used). Keeps the corpus declarative — adding a new `TestFunction` subtype at Move 5/6 requires one new row, not a new assertion block. Readable diffs when an expected value changes.

**Invitation to argue.** If a reviewer has a specific per-pair assertion that doesn't fit the table form (e.g. because the expected value depends on runtime-computed parameters), accommodate with a mixed approach. Default C.

### 5.3 (calibrating) Does `OpaqueClosure` get Stratum-1 coverage?

- *For-inclusion (the load-bearing case):* The `OpaqueClosure` method-table entry can silently break under aliasing in a specific way — a missing or misrouted alias entry means `expect(m, OpaqueClosure(f))` falls through to the generic `expect(m::Measure, f::Function; kwargs...)` overload instead of the `expect(m::Measure, o::OpaqueClosure; kwargs...)` wrapper. The two have *different kwargs defaults* (the wrapper forwards caller-supplied kwargs verbatim; the Function overload uses its own hard-coded defaults like `n=64` for Beta quadrature, `n_samples=1000` for MC). The result: a numerically different answer with no `MethodError` — the kind of silent routing bug that's invisible without a test pinning the expected value against the OpaqueClosure-wrapped call. A targeted `expect(m, OpaqueClosure(f)) == <pinned value>` is exactly the test that catches this.
- *For-exclusion (the weaker case):* The method body is a one-liner `expect(m, o.f; kwargs...)` that delegates to the `Function`-taking overload. If dispatch is correct, the assertion is tautological.

**Recommendation: include one `OpaqueClosure` case per Measure subtype.** The test isn't tautological — it's load-bearing for the kwargs-default silent-routing failure mode described above. The "principle of completeness" framing understates why the test matters. Cheap to add; catches a real failure mode; pins a precedent for Moves 3–7 to follow when they add new Measure subtypes with their own kwargs defaults.

## 6. Risk + mitigation

**Risk R1 (main risk): subtle floating-point drift at the tolerance boundary in Stratum-1.** Not "dispatch breaks" — the alias migration is mechanical, and a broken dispatch would surface as a `MethodError` at test load time (immediate, loud). The real risk is a silent drift caused by the method-table routing changing an innocent-looking `+` pairing or `*` ordering inside a method body. Stratum-1's `==` tolerance for closed-form methods and `==` for seeded Monte Carlo are tight precisely to catch this. *Investigation posture if breached:* halt. Read the failing method's body. Check for any implicit reordering (Julia's `sum()` internal pairwise reduction, `broadcast` vectorisation, the order of RNG consumption). Likely fix: re-inline the arithmetic explicitly (`sum(w[i] * v[i] for i in eachindex(w))` → manual loop) to pin the order, or pin the RNG seed at a different point. Do not relax the tolerance.

**Risk R2 (low): `Functional` alias breaks existing `typeof(x) <: Functional` runtime checks.** `const Functional = TestFunction` means `typeof(x) <: Functional` becomes `typeof(x) <: TestFunction`, which is semantically what consumers want — but if any consumer has the check plus a `x isa SomeSpecificFunctional` in the same expression, Julia's type inference might now behave differently. *Caught by:* existing test suite at Move 2's verification step.

*Pre-emptive grep (run at design-doc time, 2026-04-20):* `grep -rn '<: Functional\|isa Functional\|::Functional'` across `src/`, `test/`, `apps/`, `docs/`. Hits:

- `src/ontology.jl:659, 661, 672, 682, 693, 699` — the six `struct X <: Functional` declarations that Move 2 replaces with imports + aliases. *Mechanical replacement — expected.*
- `src/ontology.jl:761` — `expect(m::MixtureMeasure, φ::Functional)` method signature. Post-alias this dispatches on `::TestFunction` (semantically identical). *No edit needed.*
- `src/ontology.jl:768` — comment about dispatch ambiguity resolution. Post-alias it still describes the same ambiguity (`::TestFunction` vs `::OpaqueClosure`). *Harmless — comment left as-is.*
- `test/test_core.jl:1360` — comment only; same disambiguation note. *Harmless.*
- `apps/skin/server.jl:304` — `function build_function(spec)::Functional` return-type annotation. Post-alias the return type is `TestFunction`; same object, different canonical name. *No edit needed.*

No `isa Functional` and no `<: Functional` in consumer-reachable runtime checks that could behave differently under aliasing. R2 is confirmed low — the pre-emptive grep is clean.

**Risk R3 (low): export ambiguity on `using Credence`.** `Credence.jl`'s export list currently re-exports `Functional, Identity, Projection, ...` from `Ontology` (line 42 of `Credence.jl`). After Move 2, these are aliases pointing at `Previsions` types. If `Credence.jl` also re-exports `Identity` from `Previsions` (because `Previsions` now exports them), Julia complains about duplicate exports. *Caught by:* compile-time error on `using Credence` during Move 2's verification. *Mitigation:* `Previsions` adds `Identity, Projection, ...` to its export list; `Credence.jl` consolidates the concrete-subtype exports onto a single line to avoid the duplicate. The specific edit lands with the code PR (2b) and is verifiable by `julia -e 'push!(LOAD_PATH, "src"); using Credence; Identity'` returning an unambiguous concrete type.

## 7. Verification cadence

At end of Move 2's code PR (2b):

```bash
# Stratum 1 opens
julia test/test_prevision_unit.jl

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

**Skin smoke test:** optional at Move 2. The JSON-RPC API surface doesn't change (`Functional` → `TestFunction` rename is internal; external consumers use `Identity()`, `Projection(i)`, etc. which still resolve). Recommended as a sanity check, not halt-the-line.

**Halt-the-line:** any Stratum-1 assertion failure (closed-form `!=`, seeded MC `!=`, quadrature outside `atol=1e-14`). Investigate per R1 posture; do not relax the tolerance.
