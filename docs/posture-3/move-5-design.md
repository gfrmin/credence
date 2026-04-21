# Move 5 design — `MixturePrevision` and `ExchangeablePrevision`

Status: design doc (docs-only PR 5a). Corresponding code PR is 5b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md`.

Master plan reference: `docs/posture-3/master-plan.md` § "Move 5 — `MixturePrevision` and `ExchangeablePrevision`".

## 1. Purpose

Move 5 lands two prevision-side types whose absence has been a load-bearing gap in the Posture 3 story so far. First, `MixturePrevision` takes over the per-component routing currently implemented at the Measure level (`condition(::TaggedBetaMeasure, k, obs)` at `src/ontology.jl:911-947`) — the loop that unwraps `FiringByTag` / `DispatchByComponent` and dispatches each component's update. The relocation dissolves the transitional scaffolding Move 4 deliberately left in place. Second, `ExchangeablePrevision` declares exchangeability as a first-class type with a `decompose` method implementing the de Finetti representation theorem; this is the move the master plan scopes as the one that makes "the email-agent's 22-programs-as-exchangeable-hypotheses story native."

Move 5 is medium-risk — the same rating the plan assigns to Moves 3 and 4. Move 4's registry + `_dispatch_path` observability are infrastructure Move 5 inherits: each component's `update` call goes through `maybe_conjugate`, which has a working test suite with `_dispatch_path == :conjugate` assertions. Move 5's risk narrows to the mixture-coordination logic itself (component flattening, zero-mass guards, weight reweighting by predictive likelihood).

Email-agent migration is a follow-up, not Move 5's scope. `apps/julia/email_agent/host.jl:762` constructs a `MixtureMeasure` of 22 `TaggedBetaMeasure` components; rewriting that site to use `ExchangeablePrevision` lands as a separate PR after Move 5 merges.

## 2. Files touched

**Modified:**

- `src/prevision.jl` — adds `MixturePrevision` (carries `components::Vector{Prevision}` and `log_weights::Vector{Float64}`), `ExchangeablePrevision(component_space::Space, prior_on_components::Prevision)`, `decompose(p::ExchangeablePrevision) :: MixturePrevision` (representation theorem — returns the mixture-of-ergodic-components decomposition). Adds `component_prevision(p::MixturePrevision, tag::Int) -> Prevision` accessor: the public way to reach a tagged sub-prevision, which `FiringByTag` routing consults internally. Exports extended.
- `src/ontology.jl:1110-1128` — `condition(::MixtureMeasure, k::Kernel, obs)` becomes a thin facade: `MixtureMeasure(m.space, condition(m.prevision, k, obs)…)`. The current body (component iteration, per-component `_predictive_ll`, flattening sub-mixtures, `log_weights` reassembly) moves to `condition(p::MixturePrevision, k::Kernel, obs)` inside `src/prevision.jl`.
- `src/ontology.jl:911-947` — `condition(::TaggedBetaMeasure, k, obs)` deletes the 37-line FiringByTag/DispatchByComponent unwrap loop; becomes a short delegation to `MixturePrevision` (construct a one-component mixture with the TaggedBetaPrevision's tag, delegate, return). See §5.3 for the alternative of deleting the method entirely.
- `src/ontology.jl:444-457` — `FiringByTag` and `DispatchByComponent` struct definitions stay in `src/ontology.jl` (they are LikelihoodFamily subtypes; kernels declare them at construction per Invariant 2, which has not moved). Their *routing semantics* — the unwrap loop that inspects `fam.fires`, `fam.when_fires`, `fam.when_not`, and `fam.classify(m)` — move to `src/prevision.jl` as methods on `MixturePrevision`.
- `src/Credence.jl` — add `MixturePrevision`, `ExchangeablePrevision`, `decompose`, `component_prevision` to the exports list.
- `CLAUDE.md:206` — narrative update: the Invariant 2 paragraph about FiringByTag/DispatchByComponent notes that declaration stays on the kernel side; routing semantics live on the prevision side as of Move 5. No change to the precedent's force; the relocation is a Posture 3 mechanical detail.

**New:**

- `test/test_prevision_mixture.jl` — Stratum-1 / Stratum-2 corpus for `MixturePrevision` and `ExchangeablePrevision`. Component flattening tests (3-component mixture conditioned → check weights and posterior structure); FiringByTag routing tests (two components fire, one doesn't; assert `_dispatch_path == :conjugate` on the firing two, `_dispatch_path == :particle` on the non-firing); `ExchangeablePrevision.decompose` tests (representation-theorem round-trip on a declared-exchangeable prior).

**Not touched in Move 5:**

- `apps/julia/email_agent/host.jl:762` — the 22-component MixtureMeasure construction. Email-agent migration is a Move-5 follow-up (master plan). Move 5 ships the type; the follow-up switches the consumer.
- `apps/julia/rss/host.jl:147,182` — FiringByTag kernel construction sites. Unchanged: kernels still declare FiringByTag at construction; the routing semantics just resolve through the new prevision-side path.
- All 13+ TaggedBetaMeasure construction sites in apps and tests. Unchanged: constructor signature preserved.

## 3. Behaviour preserved

### Stratum-2 tolerances for the mixture condition path

- **Conjugate-per-component:** `==`. Move 4's precedent — each component's `update(cp, obs).prior` is integer-accumulated α/β or closed-form μ/σ; bit-exact.
- **Mixture flattening (log-weight reassembly + logsumexp renormalisation):** `atol=1e-14`. `sum(weights[i] * exp(pred_ll[i]))` is pairwise-reduction-legal; the 1e-14 budget covers arithmetic reassociation from Julia's internal `sum` without masking a posterior-changing sample-order change. Same reasoning as Move 2 §3 quadrature tolerance.
- **Particle fallback** (e.g. a non-firing component with a non-conjugate kernel, falling through `maybe_conjugate`): `==` under deterministic seeding. Move 2 precedent; Move 4 reaffirmed; Move 5 inherits.

### Verification invariants

For a mixture of $n$ components where $k$ of them fire under `FiringByTag` and $n - k$ do not, the Stratum-2 test asserts:

1. Post-condition, each of the $k$ firing components has `_dispatch_path == :conjugate` (the routing resolved to `BetaBernoulli` which is registered).
2. Each of the $n - k$ non-firing components also has `_dispatch_path == :conjugate` — the unwrap routes them to `Flat`, which Move 4 Phase 2a registered as the no-op conjugate pair `(BetaPrevision, Flat)`. A `_dispatch_path == :particle` result on a FiringByTag-resolved component would signal a registry miss and is halt-the-line. The `:particle` path only fires when a component's unwrap resolves to a family the registry hasn't keyed on (e.g. a paper-case-study kernel declaring a novel LikelihoodFamily subtype without a Move-4-style pair registration).
3. Component flattening: if any component's posterior is itself a `MixturePrevision` (possible under nested exchangeable decomposition), sub-components are spliced into the outer mixture with log-weights multiplied correctly. Pre-Posture-3 behaviour preserved bit-for-bit.
4. Zero-mass guard (Posture 2 gate-3): if all components have `_predictive_ll == -Inf`, the outer `condition` raises with the same message the master plan's gate-3 contract names.

### Behaviour NOT preserved

None. The FiringByTag / DispatchByComponent routing is semantically identical pre- and post-relocation: the same unwrap loop runs, just owned by `MixturePrevision` instead of by `condition(::TaggedBetaMeasure, …)`. No algorithmic change.

## 4. Worked end-to-end example

**Inputs:** a 3-component mixture with distinct tags, a FiringByTag kernel that fires on tags 1 and 3 but not tag 2, and a Bernoulli observation `obs = 1`.

```julia
p1 = TaggedBetaPrevision(1, BetaPrevision(2.0, 3.0))   # tag 1
p2 = TaggedBetaPrevision(2, BetaPrevision(5.0, 5.0))   # tag 2
p3 = TaggedBetaPrevision(3, BetaPrevision(1.0, 4.0))   # tag 3
mp = MixturePrevision([p1, p2, p3], [0.0, 0.0, 0.0])   # uniform

k = Kernel(Interval(0, 1), Finite([0.0, 1.0]),
           θ -> error("generate not used"),
           (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300));
           likelihood_family = FiringByTag(Set([1, 3]), BetaBernoulli(), Flat()))

obs = 1
```

**Step-by-step dispatch:**

```julia
condition(mp, k, obs)
  ↓
# Module: src/prevision.jl — MixturePrevision owns the per-component loop now.
# Replaces the pre-Move-5 routing loop at src/ontology.jl:911-947.
for (i, comp) in enumerate(mp.components)    # comp is a TaggedBetaPrevision
    pred_ll = _predictive_ll(comp, k, obs)
    # ↑ uses comp.tag to resolve the FiringByTag. For i=1: tag 1 ∈ {1,3} →
    #   BetaBernoulli; for i=2: tag 2 ∉ {1,3} → Flat; for i=3: tag 3 ∈ {1,3}
    #   → BetaBernoulli. Same unwrap as pre-Move-5, different owner.

    conditioned = condition(comp, k, obs)
    # ↑ THIS is where Move 4's registry fires. For TaggedBetaPrevision
    #   with a resolved BetaBernoulli family, maybe_conjugate routes to
    #   ConjugatePrevision{BetaPrevision, BetaBernoulli}; update increments
    #   α by 1 (obs == 1). For the Flat branch, maybe_conjugate routes to
    #   ConjugatePrevision{BetaPrevision, Flat}; update is identity.
    #   _dispatch_path(comp, k) for the resolved (not bare) family returns
    #   :conjugate in all three cases.

    base_lw = mp.log_weights[i] + pred_ll
    # Flatten-if-nested: if conditioned isa MixturePrevision (e.g. from
    # ExchangeablePrevision.decompose), splice sub-components with
    # multiplied log-weights. For this example no sub-mixture arises.

    push!(new_components, conditioned)
    push!(new_log_wts, base_lw)
end

# Final step: logsumexp normalisation of new_log_wts.
MixturePrevision(new_components, new_log_wts)
```

**Result for concrete inputs:**

- Component 1 posterior: `TaggedBetaPrevision(1, BetaPrevision(3.0, 3.0))` (α + 1, exact).
- Component 2 posterior: `TaggedBetaPrevision(2, BetaPrevision(5.0, 5.0))` (unchanged; Flat is obs-agnostic).
- Component 3 posterior: `TaggedBetaPrevision(3, BetaPrevision(2.0, 4.0))` (α + 1, exact).
- New log-weights: component $i$ reweighted by log-predictive-likelihood of its component under the resolved family at `obs = 1`. Logsumexp-normalised.

**Dual-residency trace.** Three modules participate:

- `src/ontology.jl` **owns construction**: `FiringByTag(Set([1,3]), BetaBernoulli(), Flat())` is a LikelihoodFamily literal declared at kernel construction. The struct definition at `src/ontology.jl:444` is unchanged; consumer code (`test/test_flat_mixture.jl:314`, `apps/julia/rss/host.jl:147`, …) constructs these exactly as before. Invariant 2 compliance preserved.
- `src/prevision.jl` **owns routing semantics**: the unwrap loop (`fam isa FiringByTag ? (m.tag in fam.fires ? fam.when_fires : fam.when_not) : …`) lives on `MixturePrevision` as a private helper. `condition(p::MixturePrevision, k, obs)` calls it per component.
- `src/ontology.jl` **owns the Measure-side facade**: `condition(::MixtureMeasure, k, obs) = MixtureMeasure(m.space, condition(m.prevision, k, obs)…)`. Thin shim; arithmetic lives on the prevision side.

The `FiringByTag` value is constructed in the kernel and read by the mixture-prevision-side loop. Construction and routing are in different homes, by design. If a future refactor wants to move construction to the prevision side too, that is a separate question (Move 8 cosmetic adaptation, or a post-Posture-3 follow-up); Move 5 does not touch it.

**Not vestigial.** Both homes are necessary: construction must be kernel-side because kernels are the Invariant-2 container for declared structure, and routing must be mixture-side because per-component dispatch is a mixture-prevision operation. The plan's "dual residency hazard" warning is real but the homes are each load-bearing; neither can be deleted.

## 5. Open design questions

### 5.1 (substantive — THE live Socratic) `ExchangeablePrevision` at Move 5 or defer to email-agent migration follow-up?

The master plan scopes `ExchangeablePrevision` with `decompose` to Move 5. The email-agent migration — the first and only consumer — is a Move-5 follow-up PR, not part of Move 5 itself. So the question is: does Move 5 ship the type without its consumer, or defer until the consumer PR opens?

**Option A (ship at Move 5, master-plan scope):** `ExchangeablePrevision` struct + `decompose` land in `src/prevision.jl` as part of PR 5b. Tests exercise the representation theorem on a synthetic exchangeable prior: construct `ExchangeablePrevision(Finite([1,2,3]), DirichletPrevision([1.,1.,1.]))`, call `decompose`, assert the returned `MixturePrevision` has the mixture-of-ergodic-components structure the representation theorem promises. No consumer site rewrites.

**Option B (defer to email-agent follow-up):** Move 5 lands `MixturePrevision` alone. `ExchangeablePrevision` + `decompose` land in the email-agent migration PR, where the 22-component construction switches over to them. The type materialises alongside its consumer.

**Recommendation: A.** Three load-bearing reasons:

1. **Master plan scopes it to Move 5 explicitly** — the master plan isn't random; it reflects the sequencing logic the earlier design iterations arrived at. Reopening the scope mid-sequence carries a cost: it signals the plan is negotiable, which invites other scope-shifts that compound. A committed scope is easier to defend than an opportunistically-shrunk one.
2. **The type declaration is genuinely cheap** — `ExchangeablePrevision` is a 3-field struct plus a `decompose` method implementing the representation theorem. The type is stateless; `decompose` is a deterministic function of its inputs. Synthetic tests (construct declared-exchangeable, decompose, check the mixture structure matches the closed-form) exercise the code just as well as consumer-site tests.
3. **The paper draft needs `ExchangeablePrevision` as a citable artifact** — paper-draft.md §1.4 and the operational-consequences section reference exchangeability as a *declared first-class type*, not a post-hoc construction. If Move 5 ships without it, Move 5's paper-draft update would either (a) lie about the type's availability or (b) leave the paper's claims un-backed. Neither is acceptable per the "paper is the gating artifact" posture.

**Invitation to argue.** Two legitimate cases for B:

- *Scope-compounding.* If `MixturePrevision`'s relocation work turns out heavier than §2 anticipates — e.g. a subtle interaction with Move 3's shared-reference contract surfaces, or a zero-mass-guard edge case requires substantial rework — then adding `ExchangeablePrevision` on top is scope-compounding and the PR ships later as a result. If during code PR work (PR 5b) `MixturePrevision` alone is obviously the full day's work, the design doc's Option A commitment is revisable: deferring `ExchangeablePrevision` is a permitted fallback, not an abandonment, and the follow-up PR's design doc (or an explicit Move-5.5) takes up the slack.
- *Deferred-work-debt scoping*. Ship-at-Move-5 means `ExchangeablePrevision.decompose`'s correctness rests entirely on synthetic tests until the email-agent migration lands. If the representation theorem's implementation has a subtle bug that only manifests on real-agent shapes, the bug lives undetected through Move 6, Move 7, and Move 8 until the follow-up PR. Defer-to-consumer avoids that window.

**Committed position: A.** If during code PR work the scope compounds, drop back to B via an in-PR revision note; don't defer silently.

### 5.1 downstream consequences for Move 6

Move 6 is the particle / quadrature / enumeration refactor — adds `ParticlePrevision` and `QuadraturePrevision` as Prevision subtypes. The Socratic resolution in 5.1 has direct downstream consequences:

- **If Move 5 ships `ExchangeablePrevision` (Option A):** Move 6 inherits `ParticlePrevision` as a fresh type with no back-compatibility debt. Move 6's design-doc grep can confirm no production consumer constructs `CategoricalMeasure(Finite(samples), log_weights)` outside the particle path (currently `src/ontology.jl:660-684,848-860`). Clean additive refactor.
- **If Move 5 defers (Option B):** Move 6's scope doesn't strictly grow, but the follow-up PR adding `ExchangeablePrevision` now falls *after* Move 6. That's a sequencing hazard: Move 6 introduces `ParticlePrevision`, Move 7 inverts `condition` to event-primary, Move 8 is cosmetic adaptation, and the email-agent migration (with `ExchangeablePrevision`) has to land somewhere. The risk is an accumulated tail of Move-5.5 / follow-up / cleanup PRs all landing in a compressed window; reviewer bandwidth runs thin and silent bugs survive.

The recommended posture (A) keeps the sequence clean. Defer (B) doesn't break anything, but it extends the branch's tail.

### 5.2 (substantive) `component_prevision(p, tag)` accessor shape

The master plan names `component_prevision(p, tag) -> Prevision` as the accessor that `FiringByTag` routing consults internally. The design question: public-surface accessor vs module-private helper?

- **Option A (public accessor):** `component_prevision` is exported. Future consumers (paper-case-study kernels, a follow-up post-Move-5 domain file) can ask a MixturePrevision "what's the prevision at tag 5?" Natural for introspection.
- **Option B (module-private helper):** the function exists but is unexported. `FiringByTag` routing uses it internally; consumer code goes through `condition(p::MixturePrevision, k, obs)` as the public interface.

**Recommendation: B.** Three reasons: (1) Invariant 1's topological face — `component_prevision` outside the mixture's own dispatch is a way for consumer code to reach into a mixture's internals and inspect per-component beliefs, which invites the same "loop over components, compute something" pattern that `posterior-iteration` precedent rejects. (2) The master plan's phrasing is descriptive-not-prescriptive on export surface. (3) Future consumers that need per-tag introspection have the `event-conditioning` precedent: `condition(p, TagSet([tag]))` gets the marginal at a tag via the declared path. That's the Invariant-2 compliant way.

**Invitation to argue.** If the email-agent migration follow-up turns out to need per-tag introspection for a legitimate structural-inspection reason (e.g. reporting per-program posterior summaries in the paper's operational-consequences section), promoting to exported is a one-line change. Start private; promote if a real consumer needs it.

### 5.3 (calibrating) `condition(::TaggedBetaMeasure, k, obs)` post-relocation — delegate or delete?

Move 4 left this method as transitional scaffolding (37-line unwrap loop). Move 5 relocates the routing semantics. The method body shrinks to one of:

- **Option A (delegate):** `condition(m::TaggedBetaMeasure, k, obs) = MixtureMeasure(Interval(0,1), condition(MixturePrevision([m.prevision], [0.0]), k, obs)…)`. The TaggedBetaMeasure-as-standalone-measure path still works; the delegation hides the detail that internally it round-trips through a one-component MixturePrevision.
- **Option B (delete):** remove the method. Callers that use `condition` on a standalone TaggedBetaMeasure (outside a mixture) must construct the MixtureMeasure themselves, or use TaggedBetaPrevision + MixturePrevision directly.

**Recommendation: A.** Delegation keeps the Measure-surface API stable; no consumer code changes (the grep confirms no site constructs a TaggedBetaMeasure *as a standalone measure* and conditions on it — every site builds TaggedBetaMeasure as a mixture component — but the API keeps the pattern available). The cost is a one-line stub; the benefit is not breaking a pattern that isn't used today but might be used tomorrow.

**Invitation to argue.** If delegation via a one-component MixtureMeasure wrapper introduces a measurable constructor-level overhead (Stratum-3 regression on `test_rss.jl` which exercises standalone TaggedBetaMeasure conditioning in some code paths), delete becomes the right call. Not anticipated; name the measurement if it arises.

## 6. Risk + mitigation

**Risk R1 (main risk): component flattening / zero-mass guard silent regression.** Move 5 moves the `for (i, comp) in enumerate(m.components); push!(new_components, ...); push!(new_log_wts, ...)` loop body from `src/ontology.jl:1110` to a `condition(p::MixturePrevision, ...)` method in `src/prevision.jl`. The loop must preserve bit-for-bit: (i) the sub-mixture-splicing branch (Posture 2 gate-4 invariant), (ii) the zero-mass guard (Posture 2 gate-3 invariant), (iii) the logsumexp renormalisation order. *Caught by:* `test/test_flat_mixture.jl` (538 lines, 25 assertions) under Stratum-3 `rtol=1e-10`, plus the new `test/test_prevision_mixture.jl` under Stratum-2 `atol=1e-14`. *Investigation posture if breached:* halt. Read the new loop body. Check for any `sum` / `reduce` call whose internal ordering might differ from the pre-Move-5 `for` loop. Likely fix: pin the arithmetic with a manual loop (as in Move 2 R1's investigation posture).

**Risk R2 (low): pre-emptive grep for the three relocation targets.** Pattern search 2026-04-21 across `src/`, `test/`, `apps/`, `docs/`:

| Target | Total hits | Category (a) | Category (b) | Category (c) |
|--------|-----------|--------------|--------------|--------------|
| `condition(m::TaggedBetaMeasure, …)` — routing loop owner | 1 site (src/ontology.jl:911) | 1 (the relocation target itself; not a consumer) | 0 | 0 |
| `FiringByTag` construction + consumption | 13 code sites, 2 doc | 13 construction (test_flat_mixture.jl ×4, test_core.jl ×4, apps/julia/rss/host.jl ×2, routing-loop consumption in ontology.jl ×2 which relocate, plus 1 CLAUDE.md narrative) | 0 | 0 |
| `DispatchByComponent` construction + consumption | 6 code sites | 6 construction (test_core.jl ×6, routing-loop consumption in ontology.jl which relocates) | 0 | 0 |

**Category (a) — covered unchanged.** All construction sites (`FiringByTag(...)`, `DispatchByComponent(...)` as kernel-construction arguments) stay valid post-Move-5; the types are unchanged, only their runtime-routing-semantics move. All `m_or_θ isa TaggedBetaMeasure` isa-checks inside kernel log_density closures (`apps/skin/server.jl:239`, `apps/julia/{rss,grid_world,email_agent}/host.jl`) are dispatching on the hypothesis type inside a density call — unaffected by mixture-level routing relocation.

**Category (b) — minor adaptation: 0.** No consumer code pattern-matches on the routing-loop internals.

**Category (c) — mutations or plan-amending hits: 0.** Nothing to scope-amend.

Go/no-go gate result: **GO.** Ratio is 100% (a); master-plan thresholds (≥90% a / <15% b / 0 c) cleared with margin. Move 5 refactors the routing loop owner without touching any consumer site. The dual residency is real but narrow: construction sites (kernel-side) and routing semantics (prevision-side) have non-overlapping blast radii.

**Risk R3 (low): `ExchangeablePrevision.decompose` subtle correctness bug.** Per §5.1's Option-A posture, `decompose` lands at Move 5 without its production consumer. A bug in the representation theorem implementation (e.g. an incorrect ergodic-components extraction for a degenerate prior) would survive through Move 6-7-8 undetected until the email-agent migration follow-up. *Caught by:* synthetic tests in `test/test_prevision_mixture.jl` — construct `ExchangeablePrevision(Finite([1,2,3]), DirichletPrevision([α₁, α₂, α₃]))`, call `decompose`, assert the returned `MixturePrevision` has three components with log-weights matching the Dirichlet's marginal at each category (closed-form, `rtol=1e-12`). The test asserts the representation theorem's specific structural promise; if `decompose` implements it incorrectly, the assertion fails. Accept that synthetic tests can't catch every bug; note that the email-agent follow-up PR will re-test on real-agent shapes and surface residual bugs then.

**Risk R4 (low): Move 4 transitional scaffolding cleanup introduces a dispatch gap.** Move 4's `condition(::TaggedBetaMeasure, k, obs)` (the 37-line routing loop) is replaced per §5.3 by either Option A (delegate to one-component MixturePrevision) or Option B (delete). If Option B is chosen but a caller somewhere constructs a standalone TaggedBetaMeasure and conditions it (without a MixtureMeasure wrapper), the caller breaks with `MethodError`. *Mitigation:* §5.3 commits to Option A. A pre-commit sanity check: grep for `condition(…::TaggedBetaMeasure)` constructor patterns outside src/ and confirm no standalone usage. If the grep is clean, Option A and Option B are functionally equivalent; if it's not clean, Option A is the only safe call.

**Risk R5 (low): Move 4's `_dispatch_path` observability must report correctly under the new dispatch path.** Move 5's component routing goes through the same `maybe_conjugate` + `update` path Move 4 established. `_dispatch_path(p::MixturePrevision, k)` must report a meaningful Symbol — either `:conjugate` if all components route to conjugate pairs, or `:mixed` / `:particle` otherwise. *Design question deferred to code-PR review:* whether to return a composite Symbol per component or a single roll-up. The Move 4 design-doc §5.2 committed to the state-free query-only shape; Move 5 extends naturally. Exact return shape is an implementation detail settled in PR 5b.

## 7. Verification cadence

At end of Move 5's code PR (5b):

```bash
# Stratum 1 / Stratum 2 — new mixture + exchangeable tests
julia test/test_prevision_mixture.jl

# Stratum 2 inherited from Move 4 — registry still dispatches correctly
# when called per-component by MixturePrevision.
julia test/test_prevision_conjugate.jl

# Stratum 1 inherited from Move 2 and 3
julia test/test_prevision_unit.jl
julia test/test_persistence.jl

# Existing test suite — all must pass unchanged
julia test/test_core.jl
julia test/test_program_space.jl
julia test/test_email_agent.jl
julia test/test_flat_mixture.jl
julia test/test_grid_world.jl
julia test/test_host.jl
julia test/test_rss.jl
julia test/test_events.jl

# POMDP agent (separate package; exercises mixture paths indirectly
# through program-space agents)
cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

# Skin smoke — optional per template (Move 5 is skin-invariant;
# JSON-RPC shape of MixtureMeasure is preserved via the Move 3
# getproperty shield). Recommended as sanity check, not halt-the-line.
python -m skin.test_skin
```

**Halt-the-line conditions:**

- Any `test/test_flat_mixture.jl` Stratum-3 regression — component flattening / zero-mass guard invariant breach per R1.
- Any `test/test_prevision_mixture.jl` assertion failure — the new `MixturePrevision` test corpus is load-bearing for Move 5's scope.
- Any `_dispatch_path` assertion failure on mixture routing paths — silent registry-or-routing miss per R5.
- Any POMDP agent test failure — the POMDP package's factored models use program-space mixtures indirectly; breakage there is a real downstream signal.
- Skin smoke failures are not halt-the-line for Move 5 but should be investigated — the skin-invariance claim in §7 depends on the Move 3 shield still holding through the new mixture path.

Per the Move 3 and Move 4 precedent: the routing relocation, the `MixturePrevision` type, and the new test corpus must land in the same PR. Partial landings — e.g. `MixturePrevision` type without the routing relocation — leave the branch in a state where both the old routing loop and the new prevision-side loop exist, which is a dual-residency hazard more dangerous than the master plan's "construction vs routing" case. One PR, one relocation.
