# Phase 3 design doc — Give the data-valued carrier a Prevision-native home (mixture twins + the 4 binding sites)

> Phase 3 as scoped in `docs/measure-as-view/master-plan.md`, with the refinement the gate forced (below).
> Tracking issue: **#163**. This is the crux of the arc: the discrete/per-component carrier is *data*, and
> the master plan's thesis — "giving it a carrier-free home is *the same act* as collapsing the duplicated
> mixture twins" — is what this doc makes concrete.

## 1. Purpose

Phases 1–2 inverted the **carrier-free-in-disguise** half of the engine: `expect` (carrier-free by
construction) and `condition`/`_predictive_ll` for the *continuous* families, whose support is
type-recoverable (`Interval(0,1)`/`Euclidean(1)`/`PositiveReals`). Four backwards-delegation sites
remained, each round-tripping a Prevision through its own `wrap_in_measure` view, marked `# NOTE: #163`.
Phase 3 closes them and, in the same act, collapses the duplicated `MixtureMeasure ↔ MixturePrevision`
twins (`condition`/`prune`/`truncate`/`draw`).

**The keystone is one observation about `CategoricalPrevision`.** It holds *only* `log_weights`
(`prevision.jl:422` — "Not parametric on the atom type … `log_weights` values stand alone as a probability
vector"). It is therefore a distribution over an **index set `{1..n}`**; the index→value map (the carrier)
lives in `CategoricalMeasure{T}.space::Finite{T}`. This is the seam stated exactly: **the index is
*structure* (carrier-free); the value is *data* (Measure-owned).** So the "missing" `draw(::CategoricalPrevision)`
is not missing — it was conflated with the value lookup. Split them:

```julia
draw(p::CategoricalPrevision) = _sample_index(weights(p))   # carrier-free: returns the INDEX
draw(m::CategoricalMeasure)   = m.space.values[draw(m.prevision)]   # the Measure owns index→value
```

That is the Prevision-native home the master plan asked for. With it, every mixture twin collapses,
because `MixtureMeasure` holds a **single shared `space`** (`ontology.jl:428`) and reconstructs *every*
component through that one space (`wrap_in_measure(c, sp)`, `:445`) — so a conditioned/pruned/truncated
mixture re-attaches its carrier with one `MixtureMeasure(m.space, p2)`.

**The refinement the gate forces — what "invert" means for a genuinely carrier-bound op.** Three of the
four sites are carrier-free-in-disguise and become Prevision-native. But two operations underneath them
are *genuinely* carrier-bound — they read atom values, not indices: `condition`'s `factor_selector`
expansion (reads `cat.space.values`, `:1793`) and `_predictive_ll`/`log_predictive` over a *bare*
`CategoricalPrevision` (needs `expect` against a closure evaluated at the atoms). For these, "invert the
backwards delegation" cannot mean "force carrier-free" — there is no carrier-free form. It means **remove
the round-trip wart**: a Prevision method that exists only to call `wrap_in_measure(p)` and bounce back is
the wart; the honest end-state is that the carrier-bound operation is *cleanly Measure-resident* (no
Prevision-level entry that pretends otherwise), exactly as `condition(m::CategoricalMeasure)` (`:1053`,
reads `m.space.values[i]`) already is and which no one calls a constitutional violation. **`prevision-not-measure`
says Measure = Prevision + carrier; it does not say every operation is Prevision-primary. Operations that
need the carrier live at the Measure level, and that is the view relationship working, not drifting from
it.** This reading is the load-bearing thing to ratify (§5 Q1).

What unblocks: the arc reaches its "no backwards delegation" end-state; the mixture machinery has a single
source per operation (Invariant 3); and `draw(::CategoricalPrevision)` lets a Prevision-level mixture of
categoricals be sampled without a Measure, which the rho-latent / family-BMA Prevision-primary consumers
want.

## 2. Files touched

All modifications to `src/ontology.jl` unless noted. Line numbers are pre-Phase-3 (post-`4ce6bd3`).

**The keystone + mixture twins:**
- `src/ontology.jl` — **new** `_sample_index(w::Vector{Float64})::Int` (the cumulative-sum index sampler,
  one `rand()`), the single home of the sampling loop now duplicated across four `draw` methods.
- `src/ontology.jl` — **new** `weights(p::CategoricalPrevision)` (mirrors `weights(m::CategoricalMeasure)`
  `:153` exactly, reads `p.log_weights`).
- `src/ontology.jl` — **new** `draw(p::CategoricalPrevision) = _sample_index(weights(p))` (carrier-free index).
- `src/ontology.jl:1934` — `draw(m::CategoricalMeasure)` → `m.space.values[draw(m.prevision)]` (re-bind).
- `src/ontology.jl:1994` / `:2036` — `draw(m::MixtureMeasure)` / `draw(p::MixturePrevision)` → both route the
  index through `_sample_index`; the Measure draws the reconstructed Measure component (carrier-threaded),
  the Prevision draws the Prevision component.
- `src/ontology.jl:1698` — `condition(m::MixtureMeasure, k, obs)` → facade:
  `MixtureMeasure(m.space, condition(m.prevision, k, obs))`. The 18-line Measure-level loop is deleted; its
  comment (`:1690-1697`, "stays a Measure-level loop on purpose") is replaced by a one-line delegation note.
- `src/ontology.jl:2049` / `:2063` — `prune`/`truncate(m::MixtureMeasure)` → facades over the Prevision
  form + `m.space` re-bind (identity-preserving early return when nothing changes).

**The four binding sites:**
- `src/ontology.jl:1335` — `condition(p::ProductPrevision, k, obs; kwargs...)` → Prevision-native for the
  carrier-free paths (`FiringByTag`/`DispatchByComponent` → `ProductPrevision` of conditioned factors;
  non-routed → `_condition_particle(p, …)` → `ParticlePrevision`); the dead ternary
  (`conditioned isa MixtureMeasure ? .prevision : .prevision`, both branches identical) is removed. A
  `factor_selector` kernel reaching this method is a **loud error** (carrier-bound — condition the
  `ProductMeasure`). The `# NOTE: #163` marker is deleted.
- `src/ontology.jl:1768` — `condition(m::ProductMeasure, k, obs)` → the non-`factor_selector` branches
  become a facade over `condition(p::ProductPrevision)` + `ProductMeasure(m.space, …)` re-bind; the
  `factor_selector` expansion (`:1786-1822`, reads `cat.space.values`) **stays Measure-resident, unchanged**
  (the canonical carrier-bound op).
- `src/ontology.jl:1611` — `_predictive_ll(p::GammaPrevision)` → carrier-free
  `log(max(expect(p, h -> exp(density(k, h, obs))), 1e-300))` (Gamma's `expect` is carrier-free, Phase 1).
  Intended behaviour change: sampling → exact (§5 Q2). Marker deleted.
- `src/ontology.jl:1616` / `:1629` — generic `_predictive_ll(p::Prevision)` / `log_predictive(p::Prevision)`
  → carrier-free `expect` form (no `wrap_in_measure`). `CategoricalPrevision` has no carrier-free
  `expect(·, ::Function)`, so it `MethodError`s here loudly — categorical predictive is a Measure op
  (`expect(m::CategoricalMeasure, f)` `:548`). Markers deleted.

**New test file:**
- `test/test_measure_view_mixture.jl` — Phase 3 capture-before-refactor + the new `draw(::CategoricalPrevision)`
  behaviour (TDD). Asserts twin-collapse bit-exactness and the four-site inversions.

**Docs:**
- `docs/measure-as-view/master-plan.md` — Phase 3 entry refined with the "invert = remove the round-trip;
  carrier-bound ops are Measure-resident" reading.
- This file.

## 3. Behaviour preserved

The arc's discipline: **capture-before-refactor** — pin canonical values PRE-refactor, assert `==`
throughout. Phase 3 has one *intended* change (Gamma predictive, §5 Q2), captured as an explicit
before/after, not a silent drift.

Capture targets (canonical posteriors/draws pinned on this branch's parent `4ce6bd3`, asserted `==`):

| Test | What it pins | Class |
|------|--------------|-------|
| `test_flat_mixture` | flat `MixtureMeasure` of Betas, per-component `BetaBernoulli` condition | Strata-1, `==` |
| `test_rho_latent` | `condition(MixturePrevision)` of `LabelledCategorical` under `DispatchByComponent` | Strata-2, `==` |
| `test_family_bma` | family-BMA posterior (`MixturePrevision`, `DispatchByComponent`) | Strata-2, `==` |
| `test_structure_bma` | structure-BMA (`FiringByTag` mixture) posterior | Strata-2, `==` |
| `test_core` TEST 53 | mixture core (the test the naive collapse broke in collapse-towers Phase 2) | Strata-1, `==` |
| `test_host` | `prune` + `ProductMeasure` `factor_selector` expansion | Strata-1, `==` |
| `test_prevision_mixture`, `test_prevision_unit` | mixture `expect`/`weights`/shared-ref contract | Strata-1, `==` |

**The bit-exactness argument for the `condition(MixtureMeasure)` collapse (the central claim).** The Measure
twin (`:1698`) and the Prevision twin (`:1662`) differ in exactly two ways: (i) the Prevision twin resolves
per-component families when `k.likelihood_family isa DispatchByComponent` (`:1671`); the Measure twin does
not. (ii) component type (Measure vs Prevision). For (ii), Phase 2 already made
`condition(m::ScalarMeasure).prevision == condition(p::ScalarPrevision)`, so the stored
`MixturePrevision` is identical. For (i): **no consumer conditions a `MixtureMeasure` with a
`DispatchByComponent` kernel** — `test_rho_latent`/`test_family_bma` route those through `MixturePrevision`
(verified: `test_rho_latent.jl:53,62` `condition(MixturePrevision(...), gnc, …)`), and the only Measure-path
consumers (flat mixture `BetaBernoulli`, structure-BMA `FiringByTag`) are non-`DispatchByComponent`, so
`routed = false` ⇒ `k_i == k` ⇒ the loops are identical. The collapse is therefore **bit-exact on the
entire suite**; the routing it adds to the Measure path is a latent correctness gain with no current
exerciser (a future `DispatchByComponent`-over-`MixtureMeasure` now routes instead of silently not). The
fixture `==` is the proof; a failure is a halt-the-line signal, never papered over.

`draw` bit-exactness: `_sample_index(weights(x))` performs the identical `weights` read + single `rand()` +
cumulative loop the four inlined loops perform today; component draw is unchanged. Seeded `==` under
`Random.seed!(42)` (precedents.md §4 — `==` class, not relaxable to `rtol`).

Tolerances: Strata-1 `isapprox(atol=1e-14)` where arithmetic reassociates; **`==` for the seeded-draw and
stored-posterior captures** (no reassociation — the same operations in the same order).

## 4. Worked end-to-end example

Two traces, because the centrepiece is the twin collapse and its routing behaviour differs by entry point.

**(a) Flat `MixtureMeasure`, `BetaBernoulli` (the bit-exact path).** `m = MixtureMeasure(Interval(0,1),
[Beta(1,1), Beta(1,1)], [log 0.5, log 0.5])`, conjugate kernel, `obs = 1`.
1. `condition(m::MixtureMeasure, k, 1)` (ontology) → `MixtureMeasure(m.space, condition(m.prevision, k, 1))`.
2. `condition(p::MixturePrevision, k, 1)`: `k.likelihood_family` is `BetaBernoulli`, not `DispatchByComponent`
   ⇒ `routed = false` ⇒ `k_i = k`. Per component: `condition(Beta(1,1), k, 1) = Beta(2,1)` (conjugate),
   `_predictive_ll(Beta(1,1), k, 1)` reweights. Returns `MixturePrevision([Beta(2,1), Beta(2,1)], …)`.
3. Facade re-binds: `MixtureMeasure(Interval(0,1), thatPrevision)`. Reading `.components` reconstructs
   `wrap_in_measure(Beta(2,1), Interval(0,1)) = BetaMeasure`. **Authoritative home:** the posterior is owned
   by `condition(MixturePrevision)` (Prevision); the carrier `Interval(0,1)` is owned by the facade.
   Bit-identical to today (today's Measure loop produced the same stored prevision).

**(b) `draw` of a `MixtureMeasure` with a categorical component (the keystone).** `m = MixtureMeasure(sp,
[catMeasure, betaMeasure], [log 0.5, log 0.5])`.
1. `draw(m::MixtureMeasure) = draw(m.components[_sample_index(weights(m))])`. `_sample_index` consumes one
   `rand()` → say index 1 (the categorical).
2. `m.components[1]` reconstructs `wrap_in_measure(catPrevision, sp) = CategoricalMeasure(sp, …)`.
3. `draw(catMeasure) = sp.values[draw(catPrevision)] = sp.values[_sample_index(weights(catPrevision))]`.
   The Prevision picks the **index** (carrier-free); the Measure maps it to the **value** via `sp.values`.
   **Authoritative home:** index distribution = Prevision; index→value = Measure. Today this works only
   because `draw(catMeasure)` inlined both; Phase 3 splits them, bit-exact under seed.

## 5. Open design questions

> **Ratified 2026-06-28 (author).** All three ratified, with three refinements.
> **Q1 — (a), and the justification is promoted from "precedented" to *forced*.** The strongest argument is
> not the precedent but a *reductio* on (b): "make every op carrier-free" for a genuinely carrier-bound op
> has exactly one implementation — give the Prevision the atom values, i.e. make `CategoricalPrevision`
> carry its atoms. But that *is* binding the carrier into the Prevision, so there is no longer a carrier-free
> Prevision for the Measure to be a view *over*. (b) achieves "every op Prevision-primary" only by destroying
> the carrier-free core that is the entire point of the arc — self-defeating; it dissolves the distinction it
> claims to honour. So (a) is not a pragmatic settlement but the *only* disposition under which
> "Measure is a view over Prevision" survives. The `MethodError` at the Prevision level is the boundary
> asserting itself correctly — the measure-level analogue of `condition` refusing to run without a likelihood;
> you do not manufacture data-free versions of data-dependent operations. The keystone makes this per-op and
> sharp: the index/value seam runs *through* the operations — index-touching ops are structure (Prevision-primary),
> value-touching ops are data (Measure-resident). Because this is an *entailment* of `prevision-not-measure`,
> not a new commitment, a **one-line corollary** is landed (docs/precedents.md + the CLAUDE.md slug line:
> carrier-bound ops are Measure-resident; do not thread the carrier into the Prevision) — a corollary, not an
> axiom. **Q2 — exact** (the phase's one intended change, before/after; sampling was a Monte-Carlo
> approximation of a quantity with an exact value). **Q3 — one phase, contingency *sequenced* not merely
> pre-authorised: run Product's capture FIRST as the phase's canary**, before the coupled twin work commits —
> front-load the riskiest piece (result-type change, recursion, `factor_selector` carve-out) so the
> split decision is made early and cheaply, not mid-flight. The prose below is retained as the rationale of
> record.

1. **The seam's disposition — does "invert the binding site" mean "make carrier-free" or "remove the
   round-trip, leaving carrier-bound ops Measure-resident"?** This is the constitutional reading the whole
   phase rests on (§1). Two of the four sites have operations underneath with *no* carrier-free form:
   `condition`'s `factor_selector` expansion (reads `cat.space.values`) and predictive over a bare
   `CategoricalPrevision` (needs `expect` at the atoms). **Recommendation: "remove the round-trip."** Two
   technical reasons: (a) there is no carrier-free `expect(::CategoricalPrevision, ::Function)` to route to —
   forcing one would require `CategoricalPrevision` to carry its atoms, which *is* binding the carrier and
   directly contradicts `prevision.jl:422` and the frozen view relationship; (b) the end-state is already
   precedented and uncontested — `condition(m::CategoricalMeasure)` reads `m.space.values` today and is the
   canonical *correct* carrier-bound op, not a violation. So the deliverable is: the carrier-free-in-disguise
   sites (Gamma predictive, Product's routed/particle paths, Particle/Quadrature predictive) go
   Prevision-native; the genuinely carrier-bound ops lose their Prevision-level round-trip entry and are
   reached only through the Measure (a `MethodError` at the Prevision level is the *correct* loud signal,
   not a regression). **Counter to weigh:** this means the arc's end-state still has carrier-bound
   operations at the Measure level — if the author reads `prevision-not-measure` as "every op must be
   Prevision-primary," that is a different target and Phase 3's scope changes (we would need a carrier
   threaded into the Prevision signatures, which I argue is the wrong direction). Ratify the reading before
   code.

2. **Gamma predictive: exact-via-`expect` (intended approx→exact change) or sampling-Prevision-primary
   (bit-preserve the approximation)?** `_predictive_ll(p::GammaPrevision)` currently samples (200 draws via
   the generic Measure sampler, no fixture). Gamma's `expect` is carrier-free and deterministic (Phase 1),
   so the predictive can be the **exact** `log ∫ p(obs|h) dμ(h)`. **Recommendation: exact.** Two reasons:
   (a) a predictive likelihood is a *deterministic* function of the belief — sampling it was an unfixtured
   approximation, and the standing direction is exact-unless-an-approximation-is-Bayesian-validated; (b) it
   mirrors Phase 2's `_predictive_ll(::BetaPrevision)` (also `expect`-based) — uniform treatment of the
   continuous families. Captured as explicit before (seeded sample) / after (exact), per the master plan's
   "one intended change." **Counter:** if any consumer depends on the *sampled* value bit-for-bit, exact
   breaks it — grep (R-grep below) says no; the only caller is the mixture predictive sum.

3. **Scope: one phase, or split 3a (twins + keystone + Gamma) / 3b (Product + catch-alls)?** The twin
   collapse *couples* to the site inversions: collapsing `condition(MixtureMeasure)` routes component
   predictive/condition through the Prevision versions, which for `ProductPrevision` components (structure-BMA)
   and the generic catch-alls are exactly sites 1/3/4. So they are not cleanly separable — splitting would
   leave 3a depending on the still-warty Prevision catch-alls (which still work, via round-trip, so 3a is
   green, but the coupling means 3b is not optional follow-up — it is the other half of the same change).
   **Recommendation: one phase**, matching the master plan's "the same act." **Counter:** Product is the
   single hardest piece (result-type change, recursion, the `factor_selector` carve-out); if its capture
   surfaces a non-bit-exact change, splitting it to its own PR for isolated review is the fallback — flagged
   here so the option is pre-authorised rather than litigated mid-flight.

## 6. Risk + mitigation

- **R1 — the `condition(MixtureMeasure)` collapse changes a posterior.** *Failure mode:* a consumer does
  condition a `MixtureMeasure` with `DispatchByComponent` (contradicting §3's claim), so the added routing
  changes output. *Blast radius:* `test_rho_latent`, `test_family_bma`, `test_structure_bma`,
  `test_flat_mixture`, `test_core` TEST 53. *Mitigation:* capture-before-refactor `==` across all five; the
  grep below enumerates every `condition(<MixtureMeasure>` call site and its kernel family. A `==` failure
  is investigated (is the new routing *correct*? then it is the intended fix, documented), never silenced.
- **R2 — `draw` seed-consumption reorder.** *Failure mode:* `_sample_index` consumes `rand()` in a
  different order than the inlined loops → seeded draws drift. *Blast radius:* any seeded `draw` test
  (`test_core`, `test_host`, `test_prevision_*`). *Mitigation:* `_sample_index` is the inlined loop verbatim
  (one `rand()`, same cumulative compare); seeded `==` under `seed!(42)` catches any reorder; `==` not
  relaxable to `rtol`.
- **R3 — the facade's shield reconstruction errors on a non-conjugate mixture component.** *Failure mode:*
  a `MixtureMeasure` whose component conditions to a `QuadraturePrevision`/`ParticlePrevision` (grid/particle)
  has `.components` read → `wrap_in_measure(QuadraturePrevision, sp)` → `wrap_in_measure(QuadraturePrevision)`
  → error (the `:894` no-carrier stance). *Blast radius:* same as today — this is **pre-existing** (the
  current Measure loop stores the same component prevision under `m.space`); the collapse does not regress
  it. *Mitigation:* capture confirms the suite's mixture components are conjugate (stay-in-family) or
  Product/Labelled; document the grid/particle-component case as a known pre-existing edge, out of Phase 3
  scope (no new failure surface).
- **R4 — Product `factor_selector` reached at the Prevision level.** *Failure mode:* `condition(p::ProductPrevision)`
  errors on a `factor_selector` kernel a consumer actually sends. *Blast radius:* `test_host` (the
  `factor_selector` tests). *Mitigation:* those tests condition a **`ProductMeasure`** (`test_host.jl:30,76,…`),
  which keeps the `factor_selector` path Measure-resident, unchanged; the grep confirms no
  `factor_selector` kernel is sent to a bare `ProductPrevision`.
- **Pre-emptive grep (run before the PR opens; list each hit's disposition):**
  - `grep -rn 'condition(' src/ apps/ test/ | grep -i mixture` — enumerate every `MixtureMeasure`/`MixturePrevision`
    condition site + its kernel family; confirm Measure-path sites are all non-`DispatchByComponent` (R1).
  - `grep -rn 'factor_selector' src/ apps/ test/` — confirm every `factor_selector` kernel targets a
    `ProductMeasure`, never a bare `ProductPrevision` (R4). (Known hits: `host_helpers.jl:114`, `test_host.jl`.)
  - `grep -rn '_predictive_ll\|log_predictive' src/ apps/ test/` — confirm the only callers of the generic
    catch-alls are the mixture predictive sums; no consumer relies on the *sampled* Gamma value (R-Q2).
  - `grep -rn 'draw(' src/ apps/ test/` — confirm no consumer relies on `draw(::MixturePrevision)` returning
    a value for a *categorical* component (it returns an index post-Phase-3); known: none in suite.
  - `grep -rn 'wrap_in_measure' src/ apps/ test/` — confirm the four removed round-trip sites are the only
    deletions; the `wrap_in_measure(p, space)` constructors and the scalar facades stay.

## 7. Verification cadence

End of Phase-3 code (from repo root; Julia tests not CI-gated):
```
julia test/test_measure_view_mixture.jl     # new — twin-collapse == capture + draw(::CategoricalPrevision)
julia test/test_flat_mixture.jl             # capture guard (== unchanged)
julia test/test_rho_latent.jl               # DispatchByComponent routing == unchanged
julia test/test_family_bma.jl
julia test/test_structure_bma.jl
julia test/test_core.jl                      # incl. TEST 53
julia test/test_host.jl                      # prune + factor_selector
julia test/test_prevision_mixture.jl test/test_prevision_unit.jl
julia test/test_measure_view_condition.jl test/test_measure_view_expect.jl   # Phases 1–2 stay green
```
Then the **full** `test/test_*.jl` suite + lint corpus self-test (`python tools/credence-lint/credence_lint.py
test`) + `check apps/`, and **stop and report**.

**Skin smoke — required for Phase 3.** `condition`/`draw` are wire-crossing verbs (`apps/skin/server.jl`),
and the mixture twins are the structure-BMA / rho-latent consumption surface, so run
`JULIA_PROJECT=. uv run python apps/skin/test_skin.py`. Phase 3 changes internals, not the wire schema
(Measures stay server-side as opaque IDs); the smoke confirms the consumption surface is intact.

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red. The seeded-draw and
stored-posterior `==` classes are **not** relaxable to `rtol` (precedents.md §4 — relaxing masks the
routing/reorder regressions these captures exist to catch).
