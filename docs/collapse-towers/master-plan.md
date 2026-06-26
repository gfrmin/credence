# Credence — Core-library engine arc: *Collapse the towers* (master plan)

> Durable, in-repo master plan for the `collapse-towers` branch family. A thematic engine
> arc (matching the `decouple` precedent), unnumbered, that *precedes* Posture-6 body work
> (`docs/posture-6-prep/`). Each phase lands design-doc-before-code (seven-section template at
> `docs/collapse-towers/DESIGN-DOC-TEMPLATE.md`), each commit green + bisectable, **stop-and-report
> at every phase boundary.**

## Context

The constitution already commits Credence to two unifications that the **code** does not yet honour:

1. **One complexity log-prior** (SPEC §1.3, the Occam/Solomonoff weighting `P(program) = 2^{-|program|}`).
   Today three bespoke sites each open-code `−λ·(description length)`: the structure-BMA edge prior
   (`src/structure_bma.jl:96`), the program node-count prior (`src/program_space/enumeration.jl:170`
   and `src/program_space/agent_state.jl:131`), and the new-rule compression gate
   (`propose_nonterminal`, `src/program_space/perturbation.jl:130`).
2. **One net-expected-value functional** (SPEC's meta-action passage + CLAUDE.md Invariant 1's
   heuristics clause). Today `net_voi` (`src/stdlib.jl:202`), the routing EU (`src/routing.jl:54`),
   and the `decide_with_voi` ask-gate (`src/stdlib.jl:231`) each express `E[Δvalue|action] − cost(action)`
   independently, and **`perturb_grammar` selects its meta-action with `rand`** (`perturbation.jl:153`)
   — a live breach of Invariant 1, which lists `perturb_grammar` as a *canalised* composition.

**This arc is conformance, not new doctrine.** Make the code instance the two ur-templates the
constitution already declares; two capabilities fall out as consequences — **Family-BMA** (a
posterior over likelihood families, replacing the hand-declared `:family` surface) and
**compute-as-decision** (a compute-cost coordinate + a Value-of-Computation gate that retires the
`rand`). Outcome: two new engine files (`src/complexity.jl`, `src/net_value.jl`); the six bespoke
sites route through them; the `rand` in `perturb_grammar` is gone; the metalevel is the same
`optimise` as the object level.

**Hard constraints:** spec-first; stop-and-report at every phase boundary; design-doc-before-code;
**no new constitutional text** (exactly the one `average-not-collapse` slug + the one SPEC cross-ref
block, both already landed — Phase 0 verifies, never authors); `λ` is **per-axis**, never shared;
no silent fallbacks; no non-EU shortcuts; tolerance inside the boolean; no `using Test`.

## Phase 0 — Verify constitution preconditions (no edits) — **PASSES**

Verified on branch creation (2026-06-26): `average-not-collapse` at `CLAUDE.md:264` (index) +
`docs/precedents.md:53` (Legal/Illegal prose); SPEC cross-ref block at `CLAUDE.md:60`; SPEC §1.3
`P(program) = 2^{-|program|}` at `SPEC.md:56` ("Each symbol costs 1 bit") ⇒ program-axis `λ = ln 2`,
not free. If at execution any is absent, **STOP and report** — do not add them.

## Phase 1 — Extract the complexity log-prior (refactor + generalisation)

`src/complexity.jl`: `complexity_logprior(L; λ, offset = 0.0) = -λ*L + offset`. Recover the two forms:
- **Edge axis** (`structure_bma._structure_logweights`): `k·log(p) + (n−k)·log(1−p) = −k·log((1−p)/p)
  + n·log(1−p)` ⇒ `complexity_logprior(|parents|; λ=log((1−p_edge)/p_edge), offset=n·log(1−p_edge))`;
  at `p_edge=0.5`, `λ=0` ⇒ uniform. Up-to-`offset` (renormalised away).
- **Program axis** (`enumeration.jl:170`, `agent_state.jl:131`): express as the *sum of two* calls
  (`complexity_logprior(g.complexity; λ=log(2)) + complexity_logprior(p.complexity; λ=log(2))`) to
  preserve **bit-exactness**. `λ = log(2)` pinned by §1.3 — do not overwrite §1.3's form.

Design-doc decisions (Phase 1 design doc, RESOLVED): program-axis `L` = `g+p complexity` — the
**two-part MDL code** (`g` = dictionary definition `length(features)+Σ(1+|body|)`; `p` = program-given-
dictionary, each nonterminal ref costing 1), the correct §1.3 instance. `expanded_complexity` (the
degenerate one-part code, no reuse discount) is **rejected on the merits, not deferred**: it zeroes
`propose_nonterminal`'s `savings_per_use` and falsifies Phase 5's `net_payoff` recovery, and is a
test-only savings-verification helper, never a prior. The move prompt's `L = expanded_complexity` is an
**error**, superseded (the prompt is not authoritative over the code). Keep `p_edge` a fixed
hyperparameter (a hyperprior = another BMA axis, a separate move). Grep for absolute-weight oracles
first. Tests: differences-between-structures (not absolutes), `λ=0` uniform, monotonicity directional,
program-axis bit-identical; existing structure/program/sparse tests stay green.

## Phase 2 — Family-BMA (new capability via the Phase-1 prior on a new axis)

> **LANDED (full suite 40/40 green). Two deviations from the original plan, both recorded in
> `docs/collapse-towers/phase-2-design.md`:** (1) added **exact closed-form conjugate predictives**
> (`GaussianMeasure+NormalNormal`, `NormalGammaMeasure+NormalGammaLikelihood`) + a Lanczos `_loggamma`
> — family reweighting needs exact per-family marginal likelihoods, which Gaussian/NormalGamma lacked
> (only the approximate MC fallback); approved scope expansion. (2) the **mixture-condition dedup is
> deferred to `measure-as-view`** — collapsing `condition(MixtureMeasure)` to a Prevision facade is
> unsafe (drops carrier-space context, changes consumer-visible component type; broke test_flat_mixture
> + test_host). Phase 2 added the per-component routing **only** to `condition(MixturePrevision)`.

A `MixturePrevision` of **different leaf families over the same observation space**, prior-weighted by
`complexity_logprior` on the family index, conditioned by the **existing** chain-rule reweighting.
Feasibility verified: `condition(p::MixturePrevision, k, obs)` (`ontology.jl:1610`) already does the
per-component marginal-likelihood reweighting; `_resolve_likelihood_family` (`kernels.jl:323`) already
routes heterogeneous per-component families via `DispatchByComponent(classify)`. So the Family-BMA
kernel is a `DispatchByComponent` whose `classify(component)` returns each component's declared family.
**No new frozen type, no new axiom-constrained function.**

Commensurability guard (loud, no fallback): all candidates score the same obs space (same kernel
`target`); each carries its honest within-family conjugate prior. Design-doc decisions: declared
candidate subset per leaf (not the whole `FAMILY_REGISTRY`); `L_family`/`λ_family` default uniform
(faithful to structure-BMA's `p_edge=0.5`; **honesty note** — at uniform the family-axis prior is a
no-op, the Bayesian evidence does all the Occam work, so the unification is structural-and-available
there, not load-bearing). `average-not-collapse` binding: deliverable is the *posterior over families*
— no "select the family" step; a test asserts mixture-not-`argmax`; pragma at the carrying site.

## Phase 3 — Extract the net-value functional (refactor) — **LANDED** (suite 41/41 green)

`src/net_value.jl`: `net_value(delta_value, cost) = delta_value - cost`. `net_voi =
net_value(voi(...), cost)` (bit-identical). **Reframed in review (the headline-protecting insight):**
the net-value *semantics* `E[value] − cost` is **already unified across all four** — the routing EU
(`_eu_functional`, `routing.jl`) and the `decide_with_voi` block payoff are the **general
Functional-offset representation** (value integrated over the joint by `expect`, cost in the offset);
the scalar `net_value` is its **reduction** for already-scalar value (`net_voi`, `net_voc`). Two
representations of one semantics — not "one unification split in two," and forcing routing through the
scalar would be a regression. So when Phase 5 lands, "does EU subsume every lever?" answers "yes, in one
of two representations." A **paired-comment guardrail** at `net_value.jl` + `_eu_functional` states the
invariant both hold: pure linear `value − cost`, no clamp/nonlinearity — if either gains a nonlinearity
the unification breaks and must be revisited.

## Phase 4 — Compute-cost coordinate (additive, degenerate-reducing) — **LANDED** (suite 42/42 green)

`compute_cost::Float64 = 0.0` added to `decide_with_voi`; `eu_ask = net_voi(…, interrupt_cost) -
compute_cost`. A known scalar ⇒ a constant subtraction (like `tcost`/`time_cost`), **not** a
belief-weighted coefficient (the master plan's "additively, like harm_cost" was loose — `harm_cost` is a
coefficient on `Projection(2)` *because* harm is uncertain; this means "additive in the one currency").
`compute_cost` (agent inference) and `interrupt_cost` (user attention) are distinct currencies that
**sum** (two separate subtractions; the decision depends only on their total — tested split-invariant).

**Rationale corrected in review (load-bearing for Phase 5): `compute_cost` prices FORWARD, not sunk,
inference.** `:ask` bears it because `:ask` is the only action that commits to *further* inference
(interrupt → await → condition → re-decide); `:proceed`/`:block` terminate. It is **NOT** the EVPI
look-ahead (`net_voi`/`eu_ask` is computed unconditionally before `optimise`, so that is *sunk and common
to all three* — and pricing "is it worth computing the VOI?" is the meta-decision *above*
`decide_with_voi`, i.e. Phase 5's `net_voc` territory). The forward reading is what makes the directional
test sound (a sunk common cost cancels in the argmax) and what makes Phase 4 genuinely "the object-level
half of `net_voc`" — both price forward, not-yet-incurred inference. Skin wire unchanged (kwarg defaults
to 0.0; `server.jl:1320` caller untouched — no protocol bump).

## Phase 5 — VOC gate: retire the `rand` breach in `perturb_grammar`

`net_voc = E[value(belief after the computation)] − value(belief now) − compute_cost`, depth-one.
Replace the `rand`-based selections with `optimise` over candidate concrete perturbations ranked by
`net_voc`; the *selection* becomes an `argmax`, the *surgery* stays a meta-action. The outer
"perturb or not" decision is already EU-max at the hosts; the breach is the inner "which perturbation"
choice. Signature unchanged ⇒ skin wire unaffected.

**The 7 `rand` calls split 5 + 2.** Five (outer `rand(ops)` `:153` + structure-preserving inner
selections `:remove_rule` `:177`, `:modify_threshold` `:183`/`:187`/`:190`) dissolve into the
structure-preserving concrete-perturbation enumeration ranked by `net_voc` (`:add_rule` is already a
deterministic `argmax` in `propose_nonterminal`; its `net_payoff` folds in as the description-length
term). The remaining **two are the alphabet-op rands** (`:add_feature` `:158`, `:remove_feature`
`:164`).

**Alphabet ops are unreachable by depth-one VOC — by construction, not by difficulty.** `:add_feature`
enlarges the space to admit structure the current posterior *cannot represent*; its value is the value
of a hypothesis **not yet entertained** — the Cromwell / escape-mass frontier — which myopic depth-one
lookahead cannot see in principle. Recommendation (the answer, not a fallback): **scope `net_voc` to
the structure-preserving ops; govern alphabet ops by a separate, explicitly-deferred mechanism** —
either the feature-inclusion prior the complexity prior already supplies (Phase 1's edge axis *is* a
feature-inclusion prior) or a small explicit exploration budget.

**Close the inconsistent triple** ("retire all `rand`" + "defer alphabet selection" + "two runs ⇒ same
grammar") in the Phase 5 design doc by picking, for the two alphabet rands: **(a)** exclude
`:add_feature`/`:remove_feature` from the Phase-5 op set, or **(b)** select them deterministically from
the feature-inclusion prior. Both close the breach. **Never leave them random.** The determinism test
scopes to whichever op set survives.

**OQ-voc-estimator** is the genuine crux; the design doc is the gate. Recommended estimator
(structure-preserving ops): posterior-weighted `Δ complexity_logprior` read off the
`SubprogramFrequencyTable` already passed in (recovers `net_payoff` for `:add_rule`). Confirm the
candidate set is **enumerable** (else: constrain it — recommended — not sample-and-argmax, which
reintroduces randomness). **Stall gate stands:** if no cheap estimator survives, Phase 5 stalls at the
doc and reports — do not implement a guess.

## Files

**Create:** `src/complexity.jl`, `src/net_value.jl` (and likely `src/family_bma.jl`);
`test/test_complexity.jl`, `test/test_family_bma.jl`, `test/test_net_value.jl`,
`test/test_compute_cost.jl`, `test/test_voc_gate.jl`; `docs/collapse-towers/phase-1..5-design.md`.
**Modify:** `src/ontology.jl` (includes), `src/structure_bma.jl`, `src/program_space/enumeration.jl`
+ `agent_state.jl`, `src/stdlib.jl`, `src/program_space/perturbation.jl`, `src/Credence.jl` (exports);
possibly grid_world / email_agent meta-action tests (Phase 5 behaviour shift).

## Verification (per phase, from repo root; Julia tests are not CI-gated)

- Phase 1: `julia test/test_complexity.jl && julia test/test_structure_bma.jl && julia test/test_program_space.jl && julia test/test_sparse_structure_equivalence.jl`
- Phase 2: `julia test/test_family_bma.jl && julia test/test_family_registry.jl && julia test/test_product_bma_routing.jl`
- Phase 3: `julia test/test_net_value.jl && julia test/test_decide_with_voi.jl`
- Phase 4: `julia test/test_compute_cost.jl && julia test/test_decide_with_voi.jl`
- Phase 5: `julia test/test_voc_gate.jl && julia test/test_program_space.jl && julia test/test_email_agent.jl`
- Each phase boundary: full `test/test_*.jl` suite green; lint self-test + `check apps/`; stop and report.
- Whole-arc end-to-end: `uv run python apps/skin/test_skin.py` + Python workspace pytest.

## Key risks (carried into the per-phase design docs)
1. **Phase 1 FP-associativity** — program axis bit-exact (two-call sum); edge axis up-to-`offset`.
2. **Phase 1 program-axis `L` = `g+p` (two-part MDL); `expanded_complexity` rejected on merits** — not a deferred correction (resurrecting it breaks `propose_nonterminal` + Phase 5). The move prompt's `L = expanded_complexity` is an error, superseded.
3. **Phase 5 alphabet ops unreachable by depth-one VOC (by construction)** — 5+2 rand split; resolve the two (exclude vs prior-select), never random.
4. **Phase 5 candidate-set tractability vs determinism** — enumerate-and-constrain (recommended) vs sample-and-argmax (reintroduces randomness); decide in the doc.
5. **Phase 4 currency double-count** — `compute_cost` (inference) sums with, not duplicates, `interrupt_cost` (attention).
6. **Phase 5 benchmark drift** — greedy vs random perturbation changes app outputs (intended).

## Follow-on arc: `measure-as-view` (decided 2026-06-26; runs AFTER collapse-towers)

A duplication audit (triggered during Phase 2) found the engine drifts from the constitutional
"Measure is a declared view over Prevision" (`prevision-not-measure`). All 11 Measure facades hold a
`.prevision` field, yet:
1. **Full duplication** — mixture `condition`/`prune`/`truncate`/`draw` reimplement the Prevision
   logic verbatim (Measure ↔ Prevision twins). *Class 1 stays HERE in full:* Phase 2 attempted the
   `condition(MixtureMeasure)`→facade collapse and **reverted** it — a naive facade drops per-component
   carrier-space context and changes the consumer-visible component type (Measure→Prevision), breaking
   `wrap_in_measure`/consumer code. The dedup is real but must thread the carrier space (the
   `measure-as-view` job), so it lands here, not as a one-liner. Phase 2 added per-component routing
   only to `condition(MixturePrevision)`.
2. **Backwards delegation** — the non-conjugate `condition`/`_predictive_ll` fallbacks for
   `BetaPrevision`/`GaussianPrevision`/`GammaPrevision`/`ProductPrevision` delegate *up* to the
   Measure facade, inverting the constitution.
3. **A latent CORRECTNESS bug (highest priority)** — `expect(m::BetaMeasure, f)` uses exact
   Gauss-Jacobi quadrature (~1e-13) but `expect(p::BetaPrevision, f)` uses the *old uniform-grid
   Riemann sum* (~1e-4) the Measure path explicitly "replaces"; `TaggedBetaPrevision`/
   `GaussianPrevision` generic-`f` share the inferior grid. The constitutionally-primary path is
   silently less accurate than its facade. (Structured Functionals are closed-form on both sides and
   agree exactly — only the generic-closure fallback diverges.)

The arc: (a) invert backwards delegation to Prevision-primary + make the facades thin views;
(b) collapse `draw`/`prune`/`truncate`; (c) fix the `expect` asymmetry as an explicit change with a
test asserting `expect(p, f) ≈ expect(wrap_in_measure(p), f)` to ~1e-12; (d) capture-before-refactor
bit-exactness throughout. Keeps collapse-towers scoped to the §1.3 / meta-action towers.
