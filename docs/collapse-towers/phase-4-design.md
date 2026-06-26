# Phase 4 design doc — Compute-cost coordinate

> Seven-section template (`docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/collapse-towers/master-plan.md`. Precedents: `docs/precedents.md`.

## 1. Purpose

Phase 4 as scoped in the master plan: add a **`compute_cost`** coordinate to `decide_with_voi`'s
utility — the agent's **inference spend** — in the *one currency* with `cost`, `interrupt_cost`, and
`time_cost`. `compute_cost` prices the **forward inference an action commits to**, not inference already
spent: choosing `:ask` continues the episode (interrupt → await → `condition` → re-decide), whereas
`:proceed`/`:block` *terminate* it, so the cost is differential and forward-looking. That is exactly the
shape of Phase 5's `net_voc`, which prices a *meta*-action's compute **before** running it — so Phase 4
is the object-level half and Phase 5 the meta-level half of *one* `argmax EU` over thinking-vs-acting
(CLAUDE.md Invariant 1's heuristics clause, "when computational cost enters the utility … it is the
optimal strategy"; SPEC's meta-action passage). Degenerate-reducing: `compute_cost = 0` is bit-for-bit
today's behaviour.

## 2. Files touched

- **`src/stdlib.jl`** `decide_with_voi` — *modify*: add `compute_cost::Float64 = 0.0` to the keyword
  signature; change `eu_ask = net_voi(…, interrupt_cost)` to `eu_ask = net_voi(…, interrupt_cost) -
  compute_cost`. Two lines. `eu_ask` is computed once and used in both the single- and multi-outcome
  branches (`:ask => _const(eu_ask)`), so the one subtraction covers both.
- **`test/test_compute_cost.jl`** — *new* (see §7).

## 3. Behaviour preserved

- **`compute_cost = 0` ⇒ bit-for-bit identical, for any belief.** `eu_ask - 0.0 == eu_ask` for every
  finite `eu_ask` (subtracting `+0.0` is the IEEE-754 identity), so the `:ask` EU — and therefore the
  `optimise` over `[:proceed, :block, :ask]` in *both* the single- and multi-outcome (`harm_belief`)
  branches — is unchanged. `test_decide_with_voi.jl` stays green unchanged.
- **Skin wire unchanged (no protocol bump).** `decide_with_voi` is a wire verb (`apps/skin/server.jl:1320`);
  the new keyword **defaults to `0.0`**, so the existing server call (which passes no `compute_cost`) is
  bit-identical. A future protocol version may *expose* `compute_cost`, but Phase 4 does not require it.

## 4. Worked end-to-end example

`decide_with_voi(unc, k; cost=1.0, aversion=1.0, interrupt_cost=0.0, compute_cost=…)` with an uncertain
`unc` belief (voi `v > 0`):
- **`compute_cost = 0`:** `eu_ask = net_voi(unc, k, …, interrupt_cost=0.0) - 0.0 = v - 0 - 0 = v > 0`.
  `EU(proceed)=0`, `EU(block)≤0` ⇒ `optimise` returns `:ask`. (Matches `test_decide_with_voi.jl:56`.)
- **`compute_cost = c` with `c > v`:** `eu_ask = v - 0 - c < 0` ⇒ `:ask`'s committed forward inference
  now outweighs its info value, so it loses to a *terminate-now* action (`:proceed`/`:block` commit to no
  further inference). The flip is exactly at `c = v` — i.e. ask iff `v - interrupt_cost - compute_cost >
  0`, which shows the three costs **sum** in one currency.
- Owner: `stdlib.jl` `decide_with_voi` assembles `eu_ask`; `net_voi` (→ `net_value`) computes
  `v - interrupt_cost`; the `- compute_cost` is the new term. Not dual residency (one new keyword on
  one function).

## 5. Open design questions

1. **`compute_cost` rides on `:ask` as a separate constant subtraction — RESOLVED (forward, not sunk).**
   The reason `:ask` bears it is **forward and differential**: `:ask` is the only action that commits the
   agent to *further* inference — interrupt the user, await the answer, `condition` on it, re-decide —
   whereas `:proceed`/`:block` *terminate* the episode. So `compute_cost` prices that **committed-to
   forward inference**: a cost only `:ask` incurs (a cost common to all three would cancel in the argmax,
   which is *why* the directional test is sound), and a known scalar at decision time, so it rides as a
   constant subtraction from the `:ask` EU (the offset-equivalent for a constant action), like
   `time_cost`/`tcost` — **not** a belief-weighted coefficient. **Explicitly NOT the EVPI look-ahead**
   (`net_voi`/`eu_ask` is computed *unconditionally before* `optimise`, so it is **sunk and common to all
   three actions**); pricing *that* — "is it even worth computing the VOI?" — is the meta-decision *above*
   `decide_with_voi`, i.e. Phase 5's `net_voc` territory, not an object-level property of `:ask`. This
   forward reading is precisely what makes `compute_cost` "the object-level half of `net_voc`": both price
   *forward, not-yet-incurred* inference. (Contrast `harm_cost`, a coefficient on `Projection(2)` because
   harm is *uncertain*; the master plan's "additively, like harm_cost" means "additive in the one
   currency," not "as a belief-weighted term.")
2. **`compute_cost` (inference) and `interrupt_cost` (attention) are distinct currencies that SUM, not
   two names for one cost — confirm.** They enter `eu_ask` as **two separate subtractions**:
   `interrupt_cost` inside `net_voi` (the EVPI/attention price of the user resolving θ), `compute_cost`
   as the new outer term (the agent's inference price). `eu_ask = voi - interrupt_cost - compute_cost`.
   Folding `compute_cost` into `net_voi`'s cost argument (`net_voi(…, interrupt_cost + compute_cost)`)
   would give the same number but conflate the two at the call site — rejected for that reason; the
   separate subtraction keeps them legibly distinct. (No other open questions — mechanical.)

## 6. Risk + mitigation

- **`compute_cost = 0` is not actually bit-for-bit.** *Mitigation:* `eu_ask - 0.0 == eu_ask` for all
  finite `eu_ask` (incl. `0.0` and `-0.0`, since `-0.0 == 0.0`); `test_compute_cost.jl` asserts
  `decide_with_voi(…, compute_cost=0.0) === decide_with_voi(…)` (identical action) across `lo`/`hi`/`unc`
  beliefs, both with and without `harm_belief` (the `test_decide_with_voi.jl:69` degenerate style).
- **A `decide_with_voi` caller breaks.** *Pre-emptive grep (done):* `grep -rn decide_with_voi src/ apps/
  test/` → `apps/skin/server.jl:1320` (wire verb) and `test_decide_with_voi.jl`. The new keyword has a
  `0.0` default ⇒ both are unaffected; *disposition: no edit.*

## 7. Verification cadence

End of Phase-4 code (from repo root):
```
julia test/test_compute_cost.jl
julia test/test_decide_with_voi.jl
```
Then the full `test/test_*.jl` suite + the lint corpus self-test + `check apps/`, and **stop and
report**. Skin smoke (`JULIA_PROJECT=. uv run python apps/skin/test_skin.py`) is **optional** — the
wire is unchanged (kwarg defaults to 0.0) — but worth running once to confirm the `decide_with_voi`
verb still round-trips.

`test_compute_cost.jl` (repo `check(name, cond, detail)` idiom):
- **Degenerate (`compute_cost = 0`):** `decide_with_voi(b, k; …, compute_cost=0.0) === decide_with_voi(b,
  k; …)` for `b ∈ {lo, hi, unc}`, with and without `harm_belief` (bit-for-bit, `===`).
- **Directional:** for an `unc` belief that returns `:ask` at `compute_cost=0` (interrupt_cost=0), a large
  `compute_cost` shifts the argmax off `:ask` to the cheaper action (no magic numbers — the flip is
  asserted relative to `v`, not a hardcoded threshold).
- **Distinct-sum:** `:ask` wins iff `voi − interrupt_cost − compute_cost > 0` — assert the flip point
  moves by `δ` when `compute_cost` increases by `δ` at fixed `interrupt_cost` (the two costs are additive
  and independent, not double-counted). The independent oracle carries the `test-oracle` pragma.
