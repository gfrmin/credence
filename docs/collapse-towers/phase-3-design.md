# Phase 3 design doc — Extract the net-value functional

> Seven-section template (`docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/collapse-towers/master-plan.md`. Precedents: `docs/precedents.md`.

## 1. Purpose

Phase 3 as scoped in the master plan: introduce the single net-expected-value shape
`net_value(Δvalue, cost) = Δvalue − cost` (`src/net_value.jl`) — the operational form of SPEC's
meta-action passage, "the expected improvement an action buys, minus its cost." `net_voi` becomes the
instance with `action = observe` (`net_value(voi(...), cost)`), bit-identically. This is a pure
extraction: no behaviour changes. Its payoff is downstream — **Phase 5's `net_voc`** is the *other*
scalar instance (`net_value(E[value after compute] − value(now), compute_cost)`), so extracting the
shared shape now is what lets Phase 5 land the metalevel decision as "the same `net_value` as the
object level." The routing EU and the `decide_with_voi` ask-gate already carry the **same semantics**
in the *general* Functional-offset representation (`E[value] − cost`, cost in a `LinearCombination`
offset, evaluated by `expect`); the scalar `net_value` is that semantics' **reduction** for
already-scalar value. Two representations of one semantics — nothing to merge (see §5).

## 2. Files touched

- **`src/net_value.jl`** — *new* (~8 lines). `net_value(delta_value::Real, cost::Real) = delta_value - cost`,
  with a docstring naming the instances (net_voi = observe; net_voc = a meta-action) and a comment that
  the routing EU (`routing.jl:54`) and `decide_with_voi` block payoff are the same shape realised in the
  Functional-offset representation (referenced, not merged).
- **`src/ontology.jl`** — *modify*: `include("net_value.jl")` immediately **before** `include("stdlib.jl")`
  (`:2079`) — `net_value` is the helper, `net_voi` (in stdlib.jl) the consumer. 1 line.
- **`src/stdlib.jl:202-203`** — *modify*: re-express `net_voi` as `net_value(voi(belief, k, actions, fpa,
  possible_obs), cost)`. Signature unchanged.
- **`test/test_net_value.jl`** — *new* (see §7).

No export needed: `net_voi` is reached via explicit `using Credence.Ontology: net_voi` (not in any
`export` list); `net_value` follows the same convention. (Confirmed callers: `decide_with_voi` internal,
`apps/answer-brain/brain/answer_brain.jl:210`, `test_typed_decision.jl`.)

## 3. Behaviour preserved

- **Bit-exact (`==`).** `net_value(a, c) = a - c`, so `net_voi = net_value(voi(...), cost)` is *the
  identical subtraction* as the old `net_voi = voi(...) - cost` — same two operands, same operator. No
  FP reassociation. `net_voi`'s value, and therefore `decide_with_voi`'s `eu_ask` and every downstream
  argmax, is unchanged for all inputs.
- **Consumers stay green unchanged:** `test_typed_decision.jl:58` (`net_voi = voi − cost`, `isapprox
  1e-12`), `test_decide_with_voi.jl` (the ask-gate flows through `net_voi`), and the external
  `apps/answer-brain` brain + its test (`answer_brain.jl:210` calls `net_voi` directly).

## 4. Worked end-to-end example

`decide_with_voi(approve_belief, k; cost=1.0, aversion=1.0, interrupt_cost=0.05, …)` (`stdlib.jl:231`).
The ask branch computes `eu_ask = net_voi(approve_belief, k, [:proceed,:block], fpa, [0,1], 0.05)`.
- **Before:** `net_voi(...) = voi(approve_belief, k, …, [0,1]) - 0.05`. Owner: `stdlib.jl` `net_voi`.
- **After:** `net_voi(...) = net_value(voi(approve_belief, k, …, [0,1]), 0.05)`; `net_value(v, 0.05) =
  v - 0.05`. Owner: `net_value.jl` `net_value`; `voi` unchanged (`stdlib.jl`).
- The intermediate `v = voi(...)` is identical; `v - 0.05` is the identical FP op ⇒ `eu_ask` bit-identical
  ⇒ `optimise([:proceed,:block,:ask], …)` returns the identical action.

Not dual residency: `net_value` is a new single home; `net_voi` delegates to it (extraction, not a type
living in two places). The routing EU is the same *shape* (`reward·θ_a − cost` via `eu(joint,
LinearCombination([(reward, Projection(a))], -(cost+time_cost)))`) but a different *representation*
(cost in a Functional offset, evaluated by `expect`) — not touched here.

## 5. Open design questions

1. **`net_value` is scalar-valued — RESOLVED: two representations of one semantics, not a deferred
   unification.** The net-value *semantics* — `E[value] − cost` — is **already unified across all four**.
   `_eu_functional` (`routing.jl:54`) is `LinearCombination([(reward, Projection(a))], -(cost+time_cost))`
   evaluated by `eu`, i.e. `E[reward·θ_a] − cost`; by linearity that is the *same semantics* as the
   scalar `a − c`. So there is nothing to "defer" or "merge" — it lives in **two representations**:
   - the **Functional-offset form** (the *general* case: value integrated over the joint by `expect`,
     cost in the offset) — routing EU, the `decide_with_voi` block payoff;
   - the **scalar `net_value`** (its *reduction*, for when value is already a scalar) — `net_voi`, `net_voc`.
   Scalar-only is therefore the **correct factoring, not a compromise.** Forcing routing/ask-gate through
   the scalar `net_value` would be a *regression* — it would evaluate the expectation eagerly and throw
   away the `LinearCombination` algebra that `expect` exists to consume. This is what keeps the arc's
   headline honest when Phase 5 lands: "does EU now subsume every lever? — yes, in one of two
   representations," not "mostly, but routing was never folded in." (No other open questions — mechanical
   extraction; reviewers invited to challenge that claim.)

   **Guardrail (two representations of one semantics can drift if edited independently):** paired
   back-reference comments at `net_value.jl` and `_eu_functional` (`routing.jl:54`) state the shared
   invariant — *pure linear `value − cost`: no clamp, no nonlinearity*. If anyone adds a nonlinearity to
   either site, the unification claim breaks and must be revisited; the paired comments make that visible
   at the point of edit.

## 6. Risk + mitigation

- **An FP difference sneaks into `net_voi`.** *Mitigation:* there is none to sneak — `net_value(a,c)`
  and `a - c` are the same operation; `test_net_value.jl` asserts `net_voi` through `net_value` `==` the
  old `voi(...) - cost` for representative inputs (capture-before-refactor, bit-exact).
- **A `net_voi` consumer breaks.** *Pre-emptive grep (done):* `grep -rn 'net_voi' src/ apps/ test/` →
  `decide_with_voi` (internal), `apps/answer-brain/brain/answer_brain.jl:210` (external brain, explicit
  import), `test_typed_decision.jl`, `test_decide_with_voi.jl`. All use the **unchanged** 6-arg
  signature; *disposition: no edit, bit-identical result.* The Python `net_voi` hits are local variable
  names (display formatting), not the Julia function — no impact.
- **Include ordering.** `net_value.jl` before `stdlib.jl` (helper before consumer); a failed precompile
  is loud.

## 7. Verification cadence

End of Phase-3 code (from repo root):
```
julia test/test_net_value.jl
julia test/test_typed_decision.jl
julia test/test_decide_with_voi.jl
julia apps/answer-brain/tests/julia/test_answer_brain.jl   # external net_voi consumer
```
Then the full `test/test_*.jl` suite green + the lint corpus self-test + `check apps/`, and **stop and
report**. Skin smoke is **optional** for Phase 3 (no JSON-RPC boundary change; `net_voi`'s wire-visible
behaviour is unchanged).

`test_net_value.jl` (repo `check(name, cond, detail)` idiom):
- Unit: `net_value(0.7, 0.05) == 0.65`; `net_value(v, c) == v - c` for representative `(v, c)` incl.
  negatives (cost > value ⇒ negative net value).
- Equivalence: `net_voi(belief, k, actions, fpa, obs, cost)` (now routed through `net_value`) `==`
  `voi(belief, k, actions, fpa, obs) - cost` on a `test_typed_decision`-style mixture fixture
  (bit-exact, the independent oracle carries the `test-oracle` pragma).
- `net_voc` shape forward-check: `net_value(Δv, compute_cost)` for a hand `Δv` reduces to `Δv -
  compute_cost` (documents the Phase-5 instance; no Phase-5 code yet).
