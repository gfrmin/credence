# credence-pi Pass 2 — Move 3 design: the feature-conditioned brain (Occam prior over BN edges)

> Per `docs/posture-3/DESIGN-DOC-TEMPLATE.md`. This is the capability core — it lifts the Pass-1
> global-Beta ceiling (which cannot tell a wasteful loop from a legitimate call) to a context-aware
> brain that learns `P(approve | features)`. Built now (not data-gated): the design is verifiable on
> synthetic data with exact oracle tests, and the real-data unlock (deploy) remains the separate,
> user-gated step. Feasibility de-risked by the G1 verification spike (`wf_97e1b871-883`).

## Purpose

Replace the single global `Beta(2,2)` over `P(approve)` with a **structure-averaged** belief
`P(approve | X)`, X = the five already-declared discrete features
(`apps/credence-pi/bdsl/features.bdsl`). The product win is *surgical* governance — block the
*repeated* call while still approving the *novel* one — instead of the global brain's all-or-nothing
blocking once it has been denied a few times.

## The model — Bayesian model averaging over BN-edge structures

- **Target.** `P(approve | X)`, discriminative. X is *always observed* at decision time and approval
  `A` is a *leaf*, so A's Markov blanket is exactly `parents(A)`. Edges *among features* describe
  `P(X)`, which a decision never consumes and which cancels from the structure posterior. So the
  decision-relevant structure space is precisely **which features are parents of A** — edges *into* A.
- **Prior over edges.** Each of the 5 feature→A edges is present independently with probability
  `p = 0.5` ⇒ a **uniform prior over the 2⁵ = 32 structures**. We deliberately keep `p = 0.5` for the
  first cut so that Occam falls out of the *marginal likelihood* rather than being smuggled into the
  prior as a sparsity bias we would then have to defend. `p < 0.5` is reintroduced later as a
  documented belief, once we have observed whether the 32 structures fragment as expected (Open Q1).
- **Per-structure belief.** Structure `S` (parent set) has one cell per element of the cross-product
  of its parents' finite spaces; each cell carries an independent `Beta(2,2)` over that context's
  approval rate. `∅` = a single cell = today's global brain; all-5-edges = the full ~4608-cell
  cross-product. The full cross-product is therefore **the all-edges vertex of the family — included,
  never assumed.**
- **Learning = exact BMA.** For each observed `(X, response)`: every structure updates *its* cell for
  X via the axiom-constrained `condition()` (Beta-Bernoulli conjugate, `src/conjugate.jl:23`), and the
  structure posterior is reweighted *only* by that cell's predictive likelihood. Sequentially, the
  product of one-step predictives **is** the structure's marginal likelihood (chain rule) ⇒ exact
  Bayesian model averaging.
- **Occam / pooling emerges.** A fine structure fragments the evidence, so each observation lands in a
  near-prior cell and its predictive stays mediocre ⇒ the marginal likelihood down-weights it early; a
  novel cell is therefore predicted by the *pooled* (coarse) structures until evidence justifies the
  split. This is pooling at the *predictive* level — exactly what decisions consume — so **no
  hierarchical primitive is needed.**

What the edge prior does **not** capture is the *parameterisation of A's CPD*: the `{tool,rep}`
structure jumps straight to the full interaction table, skipping "both matter, no interaction / merged
cells" (context-specific independence). That ladder is the **staged-tree / CEG** refinement = the
fine-pooling frontier (Open Q2), out of scope for Move 3.

## The declared shape, and the one genuinely-missing substrate piece

The G1 spike proved the *arithmetic* is exact (max discrepancy `2.2e-16` vs an exact-fraction
oracle; structure posterior S0=0.0985, S1=0.1773, **S2=0.4138, S3=0.3103**; predictives 0.5→**0.323**
for repeated `bash`, 0.5→**0.670** for novel `read`). But it did so as a **host-side fold** (a
`Dict` of cells + a hand loop over structure weights). For shipped `apps/` code that fold is a second
belief-update path outside `condition()` (Invariant-1 topological) and undeclared structure
(Invariant-2). The constitutional shape must express the *same* computation through the canalised
path. Working it out precisely surfaced a correction to the approved plan:

**Correct declared shape:**
- **Top level** = `MixturePrevision` over the (≤32) structures. This is a *genuine* mixture —
  structures are mutually exclusive hypotheses ("the true parent set is S"). `condition(::MixturePrevision,
  k, obs)` (`src/ontology.jl:1121`) already reweights each component by its `_predictive_ll` and
  conditions it — this *is* the chain-rule structure reweight, for free.
- **Each structure** = a `ProductPrevision` over its cells, each cell a `TaggedBetaPrevision` with a
  **globally-unique cell tag**. A product (not a mixture): the cells are independent parameters for
  different contexts, **not** competing hypotheses.
- **Per observed context X**, build one kernel with `likelihood_family =
  FiringByTag(fires = F(X), when_fires = BetaBernoulli(), when_not = Flat())`, where `F(X)` = the set
  of the matching cell-tags, **one per structure** (globally-unique tags make a single kernel route
  correctly through the whole nested object: within each structure exactly one cell's tag is in
  `F(X)`).

**Why the spike-verifier's "structure = mixture over cells" was wrong (and why this needs a new
method).** If a structure were a `MixturePrevision` over its cells, then
`_predictive_ll(::MixtureMeasure)` (`src/ontology.jl:1060`) blends by weight — it returns
`Σ wₖ·predₖ`, not the firing cell's predictive — so the structure's marginal likelihood would be
corrupted by the non-firing cells. The cells must be a **product**, and the existing product path
does not do what we need:
- `condition(::ProductMeasure, k, obs)` with **no `FactorSelector`** falls to **particle sampling**
  (`src/ontology.jl:1213`) — approximate, and it does not route by tag.
- there is **no exact `_predictive_ll(::ProductPrevision, …)`** — it falls to the generic
  sampling estimator (`src/ontology.jl:1066`).

So the spike's `src_change_needed: false` was true **only for the host-fold**. The constitutional
declared shape needs a small, exact addition (so the approved plan's "Code-1 skipped" is **revised
to "Code-1 required, minimal"**):

**Code-1 (substrate — minimal, mutable execution layer, NOT frozen):** a per-factor-routed product
conditioning + exact product predictive.
- `condition(p::ProductPrevision, k, obs)` for a `FiringByTag`/per-factor kernel (no `FactorSelector`):
  resolve the leaf family **per factor** by the factor's tag (`_resolve_likelihood_family`,
  `src/kernels.jl:80`), update each factor independently via its conjugate (firing factor → Beta
  update; `Flat` factors → the registered no-op, `src/conjugate.jl:33`), and **return a
  `ProductPrevision`** (no mixture, so the top-level `condition(::MixturePrevision)` does not flatten).
- `_predictive_ll(p::ProductPrevision, k, obs)` = `Σ_factors logpred(factor, resolve(k, factor), obs)`
  computed **family-awarely**: `BetaBernoulli` factor → its closed-form Beta-Bernoulli predictive;
  `Flat` factor → `0` (log-predictive of a no-op observation). The sum collapses to exactly the
  firing factor's predictive — which is the mathematically-correct structure marginal-likelihood term.

This is squarely "how `condition` dispatches" — the part CLAUDE.md marks **mutable** — and it stays
canalised (every weight-change still goes through the conjugate registry). The frozen four types do
not change. It is the kind of "implement the missing part of the Prevision machine" the author
sanctioned; it is *not* a host reimplementation and *not* a second learning mechanism.

With Code-1 in place, learning is a single `condition(top_mixture, kernel_F(X), obs)`; the
model-averaged readout is a `LinearCombination` Functional over structures of `NestedProjection`s into
each structure's cell-for-X, fed to `expect` — **not** a host loop.

## Decision + utility

`decide-action` consumes the model-averaged predictive at the current X and runs EU-max over
`{proceed, ask, block}` with `voi` gating `ask` — the Pass-1 mechanism (`bdsl/decide.bdsl`), now
per-context. **Utility is linear in dollars**: EU-max then literally maximises expected dollars
saved, so the optimised objective equals the headline KPI. Concavity, if wanted, comes later from an
explicit *budget-state*, not `log(cost)` (Open Q3).

## Architecture (body / sensor / skin / brain) — restated, and the two fixes

Bright line: **the body sends raw percepts; every probability and routing decision is brain-side.**
- **Fix 1 — daemon forwards features.** Today `handle_sensor_event` calls `decide(posterior)`
  (`apps/credence-pi/daemon/server.jl:283`). It must forward the feature dict verbatim into the DSL
  (transport, skin-legal); the cell-mapping / `F(X)` construction happens in the brain
  (`decide-action`/`observe-response`), never in the daemon.
- **Fix 2 — replay rejoins context.** `init_state` replays `user-responded` through `observe-response`
  (`server.jl:152`), but a feature-conditioned update needs *which cell* — the features of the
  matching `tool-proposed` (by `in_response_to`). `replay_user_responses`
  (`daemon/observation_log.jl`) yields only `obs` today; it must yield `(context, obs)`.
- **Embedding split** (for the later gated move): embedding = perception (body); category-from-embedding
  = inference (brain). The body never sees a category; the brain never computes an embedding.

## What this lands (files)

- **Code-1 (substrate):** `src/ontology.jl` (per-factor-routed `condition(::ProductMeasure/::ProductPrevision,…)`
  + exact `_predictive_ll(::ProductPrevision,…)`); possibly a thin helper to build the
  globally-unique-tag `FiringByTag` kernel from a context. `test/test_*.jl` oracle tests.
- **Code-2 (brain):** rewrite `apps/credence-pi/bdsl/prior.bdsl` (the structure mixture + tagged
  cells), `kernel.bdsl`, `decide.bdsl` (per-context EU-max + linear cost utility, readout via
  `LinearCombination`); wire features into `decide-action`/`observe-response`; fix the replay
  context-join (`daemon/observation_log.jl`, `daemon/server.jl`). `apps/credence-pi/tests/julia/`.
- **Code-3 (demo + savings):** extend `demo/governance_demo.jl` to show the surgical win — block the
  repeated loop while still proceeding/asking on a novel call. `savings.jl` unchanged.

## Honesty

- The surgical win is *demonstrated on synthetic data with exact oracle tests*; real-world efficacy
  still needs the user-gated deploy. No efficacy claim is made from synthetic data.
- `p = 0.5` is a chosen prior, stated as such; the marginal likelihood (not the prior) does the
  Occam work for the first cut.
- Cost is per-turn (pi exposes no per-tool usage); the linear-cost utility uses the per-turn estimate,
  flagged where it surfaces.

## Verification

- **Oracle tests (tightest-invariant):** the declared shape must reproduce the host-fold oracle to
  `<1e-9` — structure posterior `{S0..S3}`, the two predictives, and the prior predictives `= 0.5`
  (the exact fractions: marginal likelihoods S0=1/105, S1=3/175, S2=1/25, S3=3/100). The host-fold
  (`/tmp/g1_spike/g1_spike.jl`) is captured as the oracle.
- **Degenerate reductions (exact):** point-mass on `∅` ≡ global Beta; point-mass on all-edges ≡
  independent cross-product (the spike verified both to `|Δ| = 0`).
- **Per-factor primitive:** `condition` updates exactly the firing factor and no-ops the rest;
  `_predictive_ll(product)` equals the firing factor's predictive (Flat factors contribute 0).
- `voi` cold-start still asks; cost-denominated loop-halt is EU-optimal *by computation*.
- `python3 tools/credence-lint/credence_lint.py --repo-root . check apps/credence-pi/` → **zero
  violations**; Julia + TS suites green.

## Open design questions

1. **`p < 0.5` later.** When and how do we reintroduce a sparsity bias as a *documented* belief?
   Trigger = having watched the 32-structure fragmentation on real data. Until then, `p = 0.5`.
2. **Staged-tree / CEG (CPD parameterisation).** The edge prior forces saturated CPDs within a parent
   set. The context-specific-independence ladder (value-level cell merges) is the fine-pooling
   frontier — a later move, and the home for the `analyse_posterior_subtrees → propose_nonterminal`
   machinery (constitutional structure growth).
3. **Utility concavity.** Confirmed linear-now; budget-state-later. Is the budget a per-session or a
   rolling/org budget (affects when it becomes belief state)?
4. **Embeddings tool-categorisation.** Gated on the discrete `tool-name` partition *saturating* (the
   `other` bucket dominating). Reuses paper1 Phase B (`category_inference.jl`, `WeightedBernoulli`,
   `marginal_reliability`). Not built in Move 3.
5. **Per-factor primitive scope.** Should Code-1's product condition be specified only for
   `FiringByTag`-of-leaf factors (what Move 3 needs), or generalised to any per-factor family routing
   now? Lean: implement exactly what Move 3 needs + leave a typed seam, per `feedback_semantics_first`.
6. **Structure enumeration cost.** 32 structures × up-to-4608 cells is fine eager, but most cells are
   never visited — do we instantiate cells lazily (on first visit) to bound memory, and is lazy
   instantiation still "declared structure"? (Lean: lazy by cell-tag, the tag set is declared.)
