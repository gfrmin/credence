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
  `p = 0.5` ⇒ a **uniform prior over the 2⁵ = 32 structures**. Framed precisely (the loose claim
  "no bias in the prior" is wrong): independent inclusion at 0.5 gives **neutral per-edge marginals**
  (`P(edge) = 0.5`, the decision-relevant quantity — so decisions are clean), but it induces a
  `Binomial(5, 0.5)` prior on the *number of parents* (20/32 of the mass on 2–3-parent structures,
  1/32 each on ∅ and all-edges). So there *is* an implicit complexity prior favouring medium
  complexity; it is immaterial at five features and the *marginal likelihood* (not the prior) does the
  Occam work. It stops being immaterial at scale — independent-inclusion priors do not self-correct
  for multiplicity — where the **Scott–Berger** fix (a `Beta` hyperprior on `p`, i.e.
  uniform-on-model-size) becomes the right move. `p < 0.5` is reintroduced later as a documented
  belief, once we have watched whether the 32 structures fragment as expected (Open Q1).
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

## Architecture: where the brain lives (Route B — settled via expert review)

The Pass-1 brain is five `.bdsl` files. This move asked: should the structure-BMA brain stay in the
DSL (expand the S-expression surface with `mixture`/`product`/tagged-cell/firing-kernel constructors —
**Route A**), or become a typed Julia brain-side module that *declares* the structure family +
readout Functionals and *calls* the Tier-1 ops, with the `.bdsl` keeping the declared data —
**Route B**? Settled on **Route B** via a written expert exchange. The decisive reasons:

1. **There is almost no orchestration here.** The per-decision path is three calls — `condition`,
   `expect` over a `LinearCombination`, `optimise`/`voi` — Pass-1's exact shape. What looked like
   "orchestration" (enumerate 32 structures, allocate globally-unique tags, assemble the
   mixture-of-products, build the per-context `FiringByTag`) is **construction of a typed belief
   object too large to transcribe by hand** (a 32-component mixture of up-to-4608-cell products), not
   reasoning over it. Route B accepts that this *declaration* is generated, not typed out — which is
   exactly the constitution's canonical shape: applications **declare** data (Spaces, Measures,
   Kernels, **Functionals**) and **call** Tier-1 primitives. The DSL's keep was never "every belief
   object is hand-written in `.bdsl`"; it was "every belief *change* and every *decision* routes
   through the primitives + stdlib." Route B preserves that whole.
2. **A 32-structure brain is not human-auditable in *either* language.** Nobody reads it to verify
   it. What a governance buyer can audit is the *invariant*: the policy provably cannot do
   un-Bayesian arithmetic. At Pass-1 size "read the source" was a credible trust story; at 32
   structures it has to become "the source is machine-checked to route through four primitives" — and
   that story is *stronger* under Route-B-plus-lint than under a sprawling generated S-expression.

**The guardrail Route B requires (non-optional).** The thin DSL surface was a *forcing function*:
Invariant 1 held because there was no syntax for raw probability arithmetic — you could not extract a
predictive and threshold it host-side because you could not say it. Julia can say it. So Route B
**replaces the lost forcing function with an enforced one**: a `credence-lint` rule that flags raw
arithmetic on belief-derived scalars (`weights`/`mean`/`expect`/predictive outputs) inside `brain/`.
The canonical violation it must catch: pulling the scalar `P(approve | X)` out and doing the
`{proceed, ask, block}` argmax in host arithmetic instead of handing the belief + the typed cost
Functional to `optimise`. Without the lint, Invariant 1 is prose.

**Library / wiring split (keeps `brain/` from rotting into a God-module).**
- **(i) The typed belief-object constructors** — `MixturePrevision`, `ProductPrevision`,
  `TaggedBetaPrevision`, `FiringByTag` — already live in `src/` and earn a *contract test*
  (`test/test_product_bma_routing.jl` is that contract; it exercises construction → routing →
  reweight end-to-end). These recur in Pass-3 CEG and the embedding extension.
- **(ii) The credence-pi assembly** (`apps/credence-pi/brain/`) names the five features, the edge
  prior `p`, and the cost utility, and wires them. The generic structure-BMA assembly functions
  (edge-subset enumeration, tag allocation, mixture-of-products build, per-context kernel, `belief_at_X`
  view) are factored as their own functions, *liftable* to `src/` when the CEG/embedding second
  consumer concretely needs them (deferred per the extract-at-second-consumer rule — the assembly API
  for staged trees is not yet known).

**The sharper constitutional line this settles (durable).** DSL surface = constructors of the four
**frozen types** only (`space`/`measure`/`kernel`). Compositions *within* a frozen type — and
`Mixture`/`Product` *are* Previsions, not new types — are **stdlib**. Stdlib that **builds typed
objects** is Julia (the execution layer); stdlib that **composes primitives** is `.bdsl` (`voi`,
`optimise`). This permanently answers "should `mixture` be a DSL builtin": **no** — it is a Prevision
composition, hence stdlib, hence (object-building) Julia.

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

With Code-1 in place, learning is a single `condition(top_mixture, kernel_F(X), obs)`.

**The decision side runs on a transient view, not the top object** (this refines the original
"`LinearCombination` of `NestedProjection`s" sketch, which fails because the cell-for-X sits at a
*different factor index* in each structure, so no single index-projection reads it across the
heterogeneous mixture). Instead, for context X build

    belief_at_X = MixturePrevision([ structure_s's cell-for-X : s ∈ structures ], top.log_weights)

— **pure construction**: select each structure's cell-for-X (a `TaggedBetaPrevision`) and copy the
*structure posterior weights* verbatim. No arithmetic on probabilities; the weights came from
`condition`. Then the model-averaged readout is `expect(belief_at_X, Identity())` (closed-form
`α/(α+β)` per component, weighted by the mixture), and the decision is `optimise`/`voi` over
`belief_at_X` with typed linear-utility Functionals.

**Equivalence lemma (why the view is exact for `voi` too).** Conditioning `belief_at_X` with a plain
`BetaBernoulli` kernel updates each component (= each structure's cell-for-X) and reweights the
structures by that cell's predictive — which is *identical* to Code-1's reweighting of `top` by
`condition(top, FiringByTag_F(X), obs)` (the firing cell's predictive *is* the structure
marginal-likelihood term). So `value`/`voi` computed on the view equal those for the full decision.
Consequence: the whole decision path needs only pre-existing mixture/Beta `expect`/`condition` — **no
new `Functional` type** (an earlier `TagProjection` idea is unnecessary).

## Decision + utility

`decide-action(top, features, cost)` builds `belief_at_X` and runs EU-max over `{proceed, ask, block}`
with `voi` gating `ask` — the Pass-1 mechanism, now per-context.

**Utility is linear in dollars.** With approval latent θ = `P(approve | X)`, the call's cost `c` (the
per-turn USD estimate the body reports via `turn-cost`), and a false-block penalty `L` (config — the
opportunity cost of blocking a call the user wanted), measured relative to a proceed baseline:

    EU(proceed) = 0
    EU(block)   = (1−θ)·c − θ·L           # save c on a wasteful call; lose L on a wanted one
    EU(ask)     = voi − q                  # EVPI of one yes/no, minus interruption cost q

EU-max then literally maximises expected dollars saved, so the optimised objective equals the headline
KPI. `block` wins iff θ < c/(c+L): a cost-sensitive threshold, not a fixed one. At cold-start (every
cell `Beta(2,2)`, θ=0.5 everywhere) with sensible `c < L`, `EU(block) < 0` and `voi > 0`, so **ask
wins by computation** (calibration-friendly; matches Pass-1). Concavity, if wanted, comes later from
an explicit *budget-state*, not `log(cost)` (Open Q3).

**The typed-decision substrate (the only new piece beyond Code-1).** `EU(proceed)`/`EU(block)` are
typed `LinearCombination`s over `Identity` — `expect(belief_at_X, LinearCombination([(−(c+L),
Identity())], c))` is closed-form, never opaque-closure quadrature. `voi` is computed by a **typed
`optimise`/`value`/`voi`/`net_voi` in `src/stdlib.jl`** that takes a *functional-per-action* preference
(the same shape the skin's `handle_optimise` already dispatches inline — this **canonicalises and
dedupes** that logic; it is not new capability). `ask`'s voi-gated EU enters `optimise` as a **constant
`LinearCombination`** (`voi − q`), so all three actions are compared through the *one* canonical argmax
— no host-side action selection (Invariant 1). Three exact Prevision-level `_predictive_ll` methods
(`BetaPrevision`/`TaggedBetaPrevision`/`MixturePrevision`, mirroring the existing Measure-level ones)
complete the gap that lets the typed `voi` reweight exactly.

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

- **Code-1 (substrate — DONE, committed `e45e9e8`):** `src/ontology.jl` per-factor-routed
  `condition(::ProductMeasure, FiringByTag)` + exact `_predictive_ll(::ProductMeasure)`;
  `test/test_product_bma_routing.jl` (18-check oracle; also the constructor contract test).
- **Code-1b (typed-decision substrate):** `src/stdlib.jl` typed `optimise`/`value`/`eu`/`voi`/`net_voi`
  over a functional-per-action preference (canonicalises the skin's inline `functional_per_action`
  loop); `src/ontology.jl` three exact Prevision-level `_predictive_ll` methods; `src/Credence.jl`
  exports. `test/test_typed_decision.jl`.
- **Code-2 (brain — Route B):** `apps/credence-pi/brain/feature_brain.jl` — generic structure-BMA
  assembly (edge enumeration, tag allocation, mixture-of-products build, per-context `FiringByTag`,
  `belief_at_X` view, linear-cost Functionals) + credence-pi wiring (`make-prior`/`decide-action`/
  `observe-response` injected into the env from the declared feature spaces + prior `p` + cost
  utility; daemon call-site unchanged). The Pass-1 `prior.bdsl`/`kernel.bdsl`/`decide.bdsl` reasoning
  is retired in favour of the Julia brain; `features.bdsl`/`capabilities.bdsl` stay as declared data.
  Daemon: forward features into `decide-action` (`server.jl` Fix 1), per-turn cost plumbing, and the
  replay context-join `replay_user_responses → (context, obs)` (`observation_log.jl`/`server.jl`
  Fix 2). `apps/credence-pi/tests/julia/test_feature_brain.jl` (oracle reproduction + decision-margin
  instrumentation + cold-start-asks + degenerate reductions).
- **Lint (the Route-B guardrail):** `tools/credence-lint/credence_lint.py` rule flagging raw
  arithmetic on belief-derived scalars inside `apps/credence-pi/brain/`; corpus self-test entry.
- **Code-3 (demo + savings):** extend `demo/governance_demo.jl` to show the surgical win — block the
  repeated loop while still proceeding/asking on a novel call. `savings.jl` unchanged.

## Feature cardinality vs recurrence (the real over-pooling risk)

The "complex structures never split / over-pool forever" worry is real but it is **not** a `p` or a
marginal-likelihood pathology — it is **feature cardinality vs context recurrence**. A feature earns
its edge only through *recurrence of its values*: BMA learns "knowing F helps" by predicting a *repeat*
of an F-value better than the pool does, and that reward requires a *second visit* to a cell. A feature
whose values are effectively unique per observation can never accumulate per-cell evidence — every
cell sees its first and only observation, predicts the prior mean, and the structure including it is
permanently penalised for fragmentation while never being rewarded. So **the cardinality of each
declared `Space` is a brain-relevant design parameter** that decides whether each edge is even
learnable.

For Move 3's five features this is **already handled** and verified:
- All five spaces are modest finite buckets (8·4·9·4·4 = 4608 cells at the all-edges vertex).
- `time-since-last-user-message` — the one in the danger zone if it were ever continuous/fine-binned —
  is declared as 4 coarse buckets (`lt-30s lt-2m lt-10m gt-10m`) **and** the body emits exactly those
  buckets (`openclaw-plugin/src/features/.../time_since_user.ts` returns one of the four strings, never
  a raw number — verified). The bucketing directive is satisfied at *both* the declaration and the
  perception layer.
- `recent-repetition-count` recurs by construction — which is *why* the G1 spike fired, and a caveat
  on the spike: its clean result is partly an artefact of exercising the one feature immune to the
  recurrence problem. The remaining three (`tool-name`, `working-directory-relative`,
  `parent-tool-call-name`) recur naturally.

## Honesty

- The surgical win is *demonstrated on synthetic data with exact oracle tests*; real-world efficacy
  still needs the user-gated deploy. No efficacy claim is made from synthetic data.
- `p = 0.5` gives **neutral per-edge marginals** (not "no prior bias"; it is `Binomial(5,0.5)` on
  parent count — see "Prior over edges"); the marginal likelihood does the Occam work for the first cut.
- Cost is per-turn (pi exposes no per-tool usage); the linear-cost utility uses the per-turn estimate,
  flagged where it surfaces.
- **Decision margin must be instrumented, not assumed.** The spike reached 0.72 structure posterior →
  predictive 0.5→0.32 → block on six observations. That margin *shrinks* as features that fragment
  without informing are added (each dilutes the model-averaged predictive back toward the pool). The
  oracle test reports the structure-weight at which EU-max flips `proceed→block` and the head-room
  above it at 0.72 — the win is only "surgical" if `block` fires at a posterior reachable on the
  handful of calls a real loop gives.

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
