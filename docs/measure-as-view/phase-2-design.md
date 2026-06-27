# Phase 2 design doc — Invert `condition`/`_predictive_ll` delegation for the scalar families (carrier-space re-bind)

> Seven-section template (`docs/measure-as-view/DESIGN-DOC-TEMPLATE.md`). Master plan:
> `docs/measure-as-view/master-plan.md`. Precedents: `docs/precedents.md`.

## 1. Purpose

Phase 2 as scoped in the master plan ("invert `condition`/`_predictive_ll` delegation, threading the
carrier space"), **with one refinement forced by the carrier-space gate** (§5 Q1): the inversion is
scoped to the **five named continuous-scalar methods**, and `Product` + the three generic catch-alls
are deferred to Phase 3.

The master plan's Phase 1/Phase 2 boundary was drawn on the premise "`expect` does *not* bind the
carrier (Phase 1, safe); `condition` *does* bind it (Phase 2, must thread the space)." Running the
gate (`grep -n '\.space'` across the delegated-to methods, then classifying each read) **refines that
premise**: for the continuous-scalar families the carrier is *not* live state — every `m.space` read is
either (a) **result-wrapper reconstruction** discarded the moment `.prevision` is extracted, or (b) a
**type-constant** (`Interval(0,1)` / `Euclidean(1)` / `PositiveReals`) the Prevision recovers from its
own type, or (c) a **DSL-declared space** the facade simply re-attaches. So the scalar inversion threads
the carrier through a thin **re-bind** at the Measure facade (`Measure binds carrier; Prevision does
not`), not through any per-component plumbing.

Where the premise *does* hold in full force — `condition(::ProductPrevision)` (threads per-factor
`cat.space.values`/`factor.space` and **returns a different type**, `MixtureMeasure`) and the generic
catch-alls (`_predictive_ll(::Prevision)`, `log_predictive(::Prevision)`, `condition(::Measure)`, whose
`draw`/`expect` need the carrier for the *discrete* families `Categorical`/`Dirichlet`) — the work is
identical to Phase 3's per-component-carrier-space remit and moves there.

What unblocks: after Phase 2 the five scalar primitives reach their own non-conjugate behaviour
**Prevision-first** (no `wrap_in_measure` round-trip through their own view), and the correctness
direction (`prevision-not-measure`) is restored for `condition`/`_predictive_ll` on the families that
carry no live carrier — leaving Phase 3 a *single* coherent problem (per-component/discrete carrier
threading) instead of that problem entangled with five trivial inversions.

**The seam the gate found (why the split lands here, not at the master plan's line).** The 5/4 split is
not arbitrary — it is the **structural-vs-data carrier seam**. The continuous families invert cleanly
because their support is *type-recoverable*: `Interval(0,1)`, `Euclidean(1)`, `PositiveReals` are read
off the Prevision's *type*, so `draw`/`expect` produce values with no carrier in hand. The discrete
families bind because their support is *data*: the category values are arbitrary (`{:red,:green,:blue}`,
a vector of reals, anything), unrecoverable from type, and they live in `m.space.values`, not in the
Prevision — confirmed by `CategoricalPrevision` holding only `log_weights` (the probabilities) with **no
`draw(::CategoricalPrevision)` method at all** (only `draw(::CategoricalMeasure)`, `:1856`): a categorical
Prevision can give you an index distribution but not an outcome. So `condition` is carrier-free exactly
when the carrier is determined by structure, and must be handed the carrier exactly when the carrier *is*
the data. This is the constitution's "Bayes charges you for the model" reappearing one level down:
`condition` charges you for the carrier precisely when the carrier is data. Giving the discrete families
a carrier-free home is therefore *the same act* as collapsing the mixture twins — which is why the four
binding sites belong to Phase 3, and why the rescope is a discovery (the right dividing line) rather than
a retreat.

## 2. Files touched

- **`src/ontology.jl`** — *modify*:
  - **Relocate the grid helper to Prevision-native.** New `_condition_by_grid(p::BetaPrevision, k, obs;
    n=64)` and `_condition_by_grid(p::GaussianPrevision, k, obs; n=64)` returning a
    `QuadraturePrevision` (the body of the current `_condition_by_grid(m::BetaMeasure)` `:1346`–`:1357`
    and `(m::GaussianMeasure)` `:1359`–`:1372`, verbatim, with `m.space.lo/.hi` → the constants `0.0/1.0`
    for Beta and `m.mu±4m.sigma` → `p.mu±4p.sigma` for Gaussian). The current
    `_condition_by_grid(m::Measure)` methods are reduced to **thin re-bind facades**:
    `_condition_by_grid(m::BetaMeasure, k, obs; n=64) = (q = _condition_by_grid(m.prevision, k, obs; n);
    CategoricalMeasure(Finite(q.grid), q))`, ditto Gaussian. (Preserves both direct callers in
    `test_prevision_particle.jl` and the `CategoricalMeasure(Finite(grid), qp)` result shape.)
  - **Relocate the particle helper to Prevision-native (scalar).** New `_condition_particle(p::Prevision,
    k, obs; n_particles=1000, seed=0)` returning a `ParticlePrevision` (body of `_condition_particle`
    `:1814`–`:1819`, with `draw(m)` → `draw(p)`; `draw(p::GammaPrevision)` exists at `:1998`). The
    existing `_condition_particle(m::Measure, …)` (`:1814`) is **left in place unchanged** — it is the
    Phase-3 catch-all (arbitrary `m`, `draw(m)` over discrete carriers). Only the scalar
    `condition(p::…)` methods call the new Prevision overload.
  - `condition(p::BetaPrevision, k, obs; n=64)` (`:1155`) — **replace** the `wrap_in_measure` fallback
    (`:1160`) with the Bernoulli fast-path (relocated from `condition(m::BetaMeasure)` `:1069`–`:1075`,
    `BetaMeasure(m.space, …)` → `BetaPrevision(…)`) then `_condition_by_grid(p, k, obs; n)`. Conjugate arm
    (`:1156`–`:1158`) unchanged (already Prevision-primary).
  - `condition(p::GaussianPrevision, k, obs; n=64)` (`:1293`) — **replace** the `wrap_in_measure`
    fallback (`:1298`) with `_condition_by_grid(p, k, obs; n)`. Conjugate arm unchanged.
  - `condition(p::GammaPrevision, k, obs; n_particles=1000)` (`:1314`) — **replace** the
    `wrap_in_measure` fallback (`:1319`) with `_condition_particle(p, k, obs; n_particles)`. Conjugate
    arm unchanged.
  - `condition(m::BetaMeasure / m::GaussianMeasure / m::GammaMeasure)` (`:1063`/`:1100`/`:1144`) —
    **reduce to a thin facade** `_rebind(m, condition(m.prevision, k, obs; kwargs…))` where `_rebind`
    re-attaches the carrier (the conjugate/Bernoulli result `→ FamilyMeasure(m.space, …)`; the
    grid/particle result `QuadraturePrevision`/`ParticlePrevision → CategoricalMeasure(Finite(...), p)`,
    i.e. `wrap_in_measure` where it exists). Threads `m.space` (see §5 Q3).
  - `_predictive_ll(p::BetaPrevision, k, obs)` (`:1585`) — **replace** the `wrap_in_measure` delegation
    with the body of `_predictive_ll(m::BetaMeasure)` (`:1538`): `val = expect(p, h ->
    exp(k.log_density(h, obs))); log(max(val, 1e-300))`. (`expect(p::BetaPrevision)` is Prevision-primary
    as of Phase 1 — the two expressions are now the *same* call.) `_predictive_ll(m::BetaMeasure)` reduces
    to `_predictive_ll(m.prevision, k, obs)`.
  - `_predictive_ll(p::GaussianPrevision, k, obs)` (`:1601`) — **replace** the `wrap_in_measure`
    delegation with the NormalNormal closed form (body of `_predictive_ll(m::GaussianMeasure)` `:1521`–
    `:1526`, `m.prevision` → `p`); the non-NormalNormal branch falls to the generic
    `_predictive_ll(p::Prevision)` (still backwards-delegating — deferred, §5 Q1).
    `_predictive_ll(m::GaussianMeasure)` reduces to `_predictive_ll(m.prevision, k, obs)`.
  - **Deferred to Phase 3 — logic untouched, but each gains a one-line tracking marker (§5 Q1 ratified):**
    the four live backwards-delegation sites — `condition(p::ProductPrevision)` (`:1341`),
    `_predictive_ll(p::GammaPrevision)` (`:1604`, sampling, no fixture), `_predictive_ll(p::Prevision)`
    (`:1607`), `log_predictive(p::Prevision)` (`:1618`) — each get a `# NOTE: measure-as-view Phase 3
    (#163) — backwards delegation; <one-line why it binds: Product returns a MixtureMeasure / discrete
    families lack a carrier-free draw>` immediately above the `wrap_in_measure` call. (`condition(m::Measure)`
    /`_condition_particle(m::Measure)` at `:1821`/`:1814` are the Phase-3 catch-all *targets*, not
    backwards-delegation themselves — no marker; left byte-for-byte.) Mixture twins unchanged.
- **`test/test_measure_view_condition.jl`** — *new* (see §7).
- **`test/fixtures/particle_canonical_v1.jls`** — *reused, not modified* (the capture-before-refactor
  reference; §3/§6).

No new exports. No new constitutional text. The `; n` / `; n_particles` kwargs are preserved exactly
where the inverted methods replace paths that had them.

## 3. Behaviour preserved

Tolerance classes (per template §3), mapped to this move:

- **Strata-1 / Strata-2 grid + particle — fixture `==` (the load-bearing guard).**
  `particle_canonical_v1.jls` (source SHA `173411b`, `==` tolerance, README §`particle_canonical_v1`)
  pins the *exact* outputs of all three scalar non-conjugate fallbacks. `test_prevision_particle.jl`
  already asserts the **Measure** entry points `==` these (green under Julia 1.12.6 as of this writing).
  Phase 2 keeps those assertions untouched and **adds Prevision-entry assertions against the same
  canonical values** (no new capture): `condition(GammaPrevision(2,3), k_pushonly, 2.5; n_particles=50)`
  under `seed!(42)` yields a `ParticlePrevision` whose `.samples == :gamma_generic_samples` and
  `.log_weights == :gamma_generic_logw`; `condition(BetaPrevision(2,3), k_pushonly, 0.5)` /
  `condition(GaussianPrevision(0,1), k_pushonly, 1.5)` yield `QuadraturePrevision`s whose `.grid` /
  `.log_weights` `==` `:beta_grid_*` / `:gaussian_grid_*`. The RNG-op sequence is preserved because
  `draw(p::GammaPrevision)` (`:1998`) and `draw(m::GammaMeasure)` (`:1932`) issue **identical**
  operations (`_draw_gamma(α)/β` on the same α,β), so the relocation is bit-exact under fixed seed.
- **Conjugate paths — `===`, untouched.** The conjugate arms of the scalar `condition`s are *already*
  Prevision-primary (`maybe_conjugate(p, k)` + `update(cp).prior`); Phase 2 does not touch them.
- **Bernoulli fast-path — `==`.** Relocating `BetaMeasure(m.space, m.α+1, m.β)` → `BetaPrevision(m.α+1,
  m.β)` is the same integer-offset arithmetic; the Measure facade re-binds `m.space`, reproducing the
  old `BetaMeasure(m.space, …)` exactly.
- **`_predictive_ll` Beta/Gaussian — `==`.** Beta becomes literally `expect(p, …)` (post-Phase-1 the
  Measure path *already* routed through `expect(p)`); Gaussian becomes the identical NormalNormal closed
  form with `m.prevision.{mu,sigma}` → `p.{mu,sigma}` (field reads, lossless).
- **Carrier preserved.** The shared-reference contract tests in `test_prevision_particle.jl`
  (`cm.logw === pp.log_weights`, `cm.space.values === qp.grid`) continue to hold — the facade builds the
  same `CategoricalMeasure(Finite(prevision-vector), prevision)`, so the reference identity the shield
  guarantees is unchanged.
- **Strata-3 end-to-end — `rtol=1e-10`.** A multi-observation `condition` fold through the skin
  (`condition` is the wire-crossing verb) on a Gaussian prior reproduces the pre-refactor posterior
  mean/variance.

No *intended* behaviour change in Phase 2 (unlike Phase 1's Beta accuracy fix). This is a pure
structural inversion: every assertion is `==`/`===`/`rtol` equivalence, captured pre-refactor.

## 4. Worked end-to-end example

`condition(m, k, 1.5)` with `m = wrap_in_measure(GaussianPrevision(0.0, 1.0))` (a `GaussianMeasure` on
`Euclidean(1)`) and `k` the non-conjugate `k_pushonly` kernel (`likelihood_family = PushOnly()`, the
fixture's Case 3) — the centrepiece, because it exercises the grid fallback *and* the carrier re-bind.

- **Before (backwards delegation).** `condition(m::GaussianMeasure, k, 1.5)` *(owner: Measure)* →
  `maybe_conjugate(m.prevision, k)` returns `nothing` (PushOnly) → `_condition_by_grid(m, k, 1.5)`
  *(owner: Measure)* → 64-pt grid over `m.mu ± 4m.sigma`, `logw[i] = log_density_at(m, x) + density(k, x,
  1.5)` → `QuadraturePrevision(grid, logw)` → wrapped `CategoricalMeasure(Finite(grid), qp)`. (The
  Prevision path `condition(p::GaussianPrevision)` would instead `wrap_in_measure(p)` and re-enter this
  very method — the round-trip the arc removes.)
- **After (Prevision-primary + facade re-bind).**
  1. `condition(m::GaussianMeasure, k, 1.5)` *(owner: Measure facade)* → `condition(m.prevision, k, 1.5)`.
  2. `condition(p::GaussianPrevision, k, 1.5)` *(owner: **Prevision**, authoritative)* →
     `maybe_conjugate` `nothing` → `_condition_by_grid(p, k, 1.5)` *(owner: **Prevision**)* → same 64-pt
     grid over `p.mu ± 4p.sigma` (`p.{mu,sigma} === m.prevision.{mu,sigma}`, lossless) → returns
     `QuadraturePrevision(grid, logw)` — **no carrier space involved**.
  3. `_rebind(m, qp)` *(owner: Measure facade, authoritative for the carrier)* →
     `CategoricalMeasure(Finite(qp.grid), qp)`.
- **Result.** Bit-identical `CategoricalMeasure` to "before" — `result.space.values ==
  :gaussian_grid_values`, `result.logw == :gaussian_grid_logw` (asserted in both
  `test_prevision_particle.jl`, unchanged, and the new `test_measure_view_condition.jl` via the Prevision
  entry point).
- **Dual residency.** The grid arithmetic had one home (Measure); Phase 2 moves it to the Prevision and
  leaves a one-line facade — net dual-residency **decreases**. The split of authority is the
  constitutional one: the **Prevision** owns the space-free posterior (`QuadraturePrevision`); the
  **Measure facade** owns the carrier binding (`Finite(grid)`). That is "Measure binds carrier space;
  Prevision does not" made literal.

## 5. Open design questions

> **Ratified 2026-06-28 (author).** All three verified against the code (the `CategoricalPrevision`
> carrier-free-`draw` gap and the `QuadraturePrevision`/`ParticlePrevision` `wrap_in_measure` asymmetry
> both confirmed). **Q1 — defer, contingency made binding:** the four binding sites move to Phase 3.
> Because Phase 3 adjacency is *not* guaranteed (the paper / Genesis / credence-pi compete for the next
> slot), the contingency triggers: the four live `wrap_in_measure` sites **must carry an explicit
> tracking marker** naming them as a known-incomplete inversion (a one-line `# NOTE:` at each, citing the
> Phase 3 tracking issue), so the half-state — the constitution asserting "Measure is a view over
> Prevision" while four sites round-trip a Prevision through its own view — is *owned and visible*, not a
> trap for the next reader. **Product is NOT swallowed** into Phase 2 (that would merge the hard mixture
> work into the safe scalar move). **Q2 — local `_rebind` (a).** **Q3 — thread `m.space` (preserve).**
> The prose below is retained as the rationale of record.

1. **Scope: defer `Product` + the three generic catch-alls to Phase 3, or force them into Phase 2?**
   The gate classification (§1) splits the 9 backwards-delegation sites **5 clean / 4 binding**. The
   clean five (`condition` Beta/Gaussian/Gamma, `_predictive_ll` Beta/Gaussian) carry no live carrier and
   are fixture- or exactness-covered. The binding four —
   `condition(::ProductPrevision)` (per-factor `cat.space.values`/`factor.space`, **returns a
   `MixtureMeasure`**, recurses into factor `condition`), `_predictive_ll(::GammaPrevision)` (sampling,
   routed through the generic Measure sampler, **no fixture**), `_predictive_ll(::Prevision)` and
   `log_predictive(::Prevision)` and `condition(::Measure)` (catch-alls whose `draw`/`expect` need the
   carrier for the *discrete* families) — are the **same problem Phase 3 already owns** (per-component /
   discrete carrier-space threading + Monte-Carlo capture). **Recommendation: defer the four to Phase
   3.** Two technical reasons: (a) `condition(::ProductPrevision)`'s result-type change to `MixtureMeasure`
   and its recursion into the mixture machinery make it indivisible from the mixture-twin collapse; (b)
   the catch-alls' only *new* failure surface is the discrete families, whose `draw(::CategoricalPrevision)`
   does not exist carrier-free — i.e. the exact gap Phase 3 closes. Counter-argument to weigh: this
   leaves four `wrap_in_measure` call-sites live after Phase 2, so the arc's "no backwards delegation"
   end-state lands only at Phase 3 — acceptable if Phase 3 is the immediate successor, a smell if the arc
   might pause after Phase 2.

2. **Re-bind home for the grid/particle results.** The grid fallback yields a `QuadraturePrevision`,
   which **deliberately has no `wrap_in_measure` method** (the `:894` stance: "no carrier Space"). Two
   ways for the facade to re-bind: **(a)** a local `_rebind(m, q::QuadraturePrevision) =
   CategoricalMeasure(Finite(q.grid), q)` per family (recommended — respects `:894`, keeps the
   carrier-binding visibly in the facade where the constitution puts it), or **(b)** add
   `wrap_in_measure(::QuadraturePrevision)` for a uniform `wrap_in_measure(condition(m.prevision, …))`
   facade — rejected, it reverses a deliberate design stance and would silently re-route the generic
   `_predictive_ll(::Prevision)`/`condition(::Measure)` fallbacks through a new wrap. Note the particle
   case is **already** uniform: `wrap_in_measure(::ParticlePrevision)` exists (`:528`) and equals
   `CategoricalMeasure(Finite(samples), pp)`, so `_rebind(m, pp) = wrap_in_measure(pp)` there.
   Recommend **(a)** with the particle case using the existing `wrap_in_measure(::ParticlePrevision)`.

3. **Thread `m.space` or canonicalise it?** The conjugate/Bernoulli facade re-bind is
   `FamilyMeasure(m.space, …)`. `src/eval.jl:489/505/518` construct `BetaMeasure(space, …)` /
   `GammaMeasure(space, …)` / `GaussianMeasure(space, …)` with a **DSL-declared `space`**, so `m.space`
   is *not* guaranteed to be the type-canonical constant. **Recommendation: thread `m.space`** (preserve
   it), not `wrap_in_measure(p2)` (which would force `Interval(0,1)`/`Euclidean(1)`/`PositiveReals` and
   silently rewrite a DSL-declared carrier). This is both the bit-preserving choice (capture-before-
   refactor asserts `==` including the space field) and the honest one ("Measure binds *the declared*
   carrier"). The alternative (canonicalise) is rejected: it changes observable state for any
   DSL-declared non-canonical space and is not what the pre-refactor conjugate arm did.

## 6. Risk + mitigation

- **R1 — seed-consumption reorder in the particle relocation.** *Failure mode:* moving
  `_condition_particle`'s `draw` loop from `draw(m)` to `draw(p)` reorders or adds RNG draws → particle
  samples drift. *Blast radius:* `test_prevision_particle.jl` Case 1 (Measure entry, unchanged) **and**
  the new Prevision-entry assertion both `==` `:gamma_generic_samples`. *Mitigation:* `draw(p::GammaPrevision)`
  and `draw(m::GammaMeasure)` are the same `_draw_gamma(α)/β` with identical α,β — verified by reading
  both (`:1998`, `:1932`); the loop count and order are unchanged; fixture `==` under `seed!(42)` catches
  any reorder. **The fixture is never regenerated to fix a drift** (README provenance protocol) — a `==`
  failure is a halt-the-line signal, investigated, not papered over.
- **R2 — facade re-bind drops or rewrites the carrier.** *Failure mode:* `_rebind` forces a canonical
  space, or returns the wrong wrapper type, changing `result.space`. *Blast radius:* `test_core`,
  `test_prevision_unit`, any consumer reading `.space` off a conditioned scalar Measure; the
  shared-reference contract tests. *Mitigation:* §5 Q3 (thread `m.space`); capture-before-refactor `==`
  on the full conditioned Measure (space + weights) for a DSL-declared non-canonical-space Beta.
- **R3 — kwarg threading lost.** *Failure mode:* the inverted `condition(p::…)` drops `; n` / `;
  n_particles`, so `condition(…; n_particles=50)` silently uses the default 1000. *Blast radius:* the
  fixture's `n_particles=50` (50 vs 1000 samples — loud length mismatch). *Mitigation:* the inverted
  signatures declare the same kwargs; the new Prevision-entry test passes `n_particles=50` and asserts
  the 50-sample `==`.
- **R4 — a deferred site silently regresses.** *Failure mode:* touching the shared `_condition_particle`
  /`_condition_by_grid` helper names perturbs the Phase-3 catch-all path. *Mitigation:* the
  `_condition_particle(m::Measure, …)` catch-all (`:1814`) is **left byte-for-byte unchanged**; only a
  new `(p::Prevision, …)` overload is added. Pre-emptive grep below confirms the call graph.
- **Pre-emptive grep (per template suggested practice).** Run before the PR opens; list each hit's
  disposition (mechanical / no-edit / attention):
  - `grep -rn 'wrap_in_measure' src/ apps/ test/` — confirm the only removed call-sites are the five
    scalar fallbacks; the four deferred sites + the `wrap_in_measure(p, space)` constructors stay.
  - `grep -rn '_condition_by_grid\|_condition_particle' src/ test/` — confirm `test_prevision_particle.jl`
    is the only external caller and it calls the **Measure** form (preserved as a thin facade) — no edit.
  - `grep -rn 'condition(.*Prevision' apps/ test/` — confirm no consumer relies on the scalar
    `condition(p::…)` *returning a Measure* (it returns a Prevision; the facade returns the Measure).
  - `grep -rn '\.space' src/ontology.jl` within the touched method bodies — confirm post-refactor the
    Prevision methods read **no** `m.space` (the gate, re-run on the new code).

## 7. Verification cadence

End of Phase-2 code (from repo root; Julia tests not CI-gated):
```
julia test/test_measure_view_condition.jl      # new — Prevision-entry == fixture + facade equivalence
julia test/test_prevision_particle.jl          # the capture-before-refactor fixture guard (== unchanged)
julia test/test_measure_view_expect.jl         # Phase 1 stays green
julia test/test_core.jl
julia test/test_prevision_unit.jl
```
Then the **full** `test/test_*.jl` suite + lint corpus self-test (`python tools/credence-lint/credence_lint.py
test`) + `check apps/`, and **stop and report**.

**Skin smoke — required for Phase 2.** `condition` is the JSON-RPC wire-crossing verb (`apps/skin/server.jl`
constructs `GaussianMeasure`/`GammaMeasure`/`NormalGammaMeasure` and conditions them server-side), so run
`JULIA_PROJECT=. uv run python apps/skin/test_skin.py`. Phase 2 changes `condition` internals but **not**
the wire schema (Measures stay server-side as opaque IDs); the smoke confirms the consumption surface is
intact.

`test_measure_view_condition.jl` (repo `check(name, cond, detail)` idiom):
- **Prevision-entry == fixture (reuse, no new capture):** under `Random.seed!(42)`,
  `condition(GammaPrevision(2,3), k_pushonly, 2.5; n_particles=50).samples == CANONICAL[:gamma_generic_samples]`
  (and `.log_weights ==` `:gamma_generic_logw`); `condition(BetaPrevision(2,3), k_pushonly, 0.5)` and
  `condition(GaussianPrevision(0,1), k_pushonly, 1.5)` return `QuadraturePrevision`s whose `.grid`/
  `.log_weights` `==` the `:beta_grid_*`/`:gaussian_grid_*` canonical vectors.
- **Facade equivalence:** for each scalar family, `condition(wrap_in_measure(p), k, obs) ==
  _rebind-of condition(p, k, obs)` — the Measure facade and the Prevision-primary path agree, on both
  the conjugate arm (`===` family Measure with `m.space` preserved) and the non-conjugate arm
  (`CategoricalMeasure` over the grid/particle).
- **DSL-declared space preserved (R2/Q3):** a `BetaMeasure(Interval(0.0,1.0), 2.0, 3.0)` conditioned on
  a conjugate kernel yields a `BetaMeasure` whose `.space === m.space` (threaded, not canonicalised).
- **`_predictive_ll` inversion:** `_predictive_ll(BetaPrevision(2,3), k, obs) ==
  _predictive_ll(wrap_in_measure(p), k, obs)` (both now `expect(p, …)`); same for Gaussian NormalNormal.

Halt-the-line: any failure at end-of-PR is a halt; the branch never sleeps red. The fixture `==` class
is **not** relaxable to `rtol` (precedents.md §4 — relaxing masks the seed-reorder regression the test
exists to catch).
