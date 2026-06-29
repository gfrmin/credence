# Move 4 design — feature discovery (`:add_feature` / `:remove_feature`)

> Exploration-budget arc, Move 4. Design-doc-before-code; ratify before any code lands.
> Master plan: `docs/exploration-budget/master-plan.md` (§3.1 the selection/generation seam — what this
> move may *claim*; §4 Move-4 scope; the fine-before-coarse escalation). Predecessors on master: Move 1
> (`:remove_rule` + sound reference count), Move 2 (saturation signal), Move 3 (threshold-refinement
> lookahead VOI + host wiring). Authored 2026-06-29.

---

## 1. Purpose

Move 4 as scoped in the master plan (§4): grow the agent's **feature** alphabet — the next rung above
thresholds in the fine-before-coarse escalation — by **reusing Move 3's lookahead VOI** over a feature
candidate set, gated on threshold-refinement saturating. `:remove_feature` reuses Move 1's
reference-count soundness.

Two realisations sharpen the scope over the master-plan framing:

- **Base-feature discovery needs ZERO host-extraction changes — it is pure EU-max *selection* over a
  host-provided candidate set.** The hosts already extract the *full* feature superset every step
  (`entity_features` computes all 8 of `ALL_GW_FEATURES`; `extract_features` the email superset),
  regardless of which grammar is active. A grammar's `feature_set` is a *subset*. So `:add_feature` =
  add an available-but-unused feature (`available_features \ g.feature_set`) to the grammar — and its
  value is *already in the features Dict*, so the new predicates over it work immediately. The
  `available_features` argument `perturb_grammar` has carried as a dead placeholder since collapse-towers
  is exactly this candidate source. This is the **fully-closed selection half** of §3.1: the host
  *provides* the candidates, the lookahead *ranks* them — no proposer, no construction.

- **Composed/novel features are the deferred frontier, and the line is the §3.1 seam.** Conjunctions and
  feature×threshold predicates are *already grammar-expressible* (`AndExpr`/`OrExpr` over `GTExpr`/`LTExpr`
  at depth ≥ 3), so they need no new mechanism — the grammar already composes selected features.
  *Products* and other arithmetic combinations are **not** expressible (the AST has no feature arithmetic),
  and a brain that *proposes* them is doing hypothesis **construction** — the creative floor. Move 4
  therefore claims **"EU-max selection among available features"** without hedging, and names
  composed-feature synthesis as the standing frontier (Move 4b or later), exactly as the master plan's
  selection/generation seam requires.

What unblocks: with thresholds (Move 3) and base features (Move 4) both EU-max, Move 5 attempts the
combined single-currency `argmax` over the whole meta-action space.

## 2. Files touched

**`src/program_space/exploration.jl`** — modification. The Move-3 lookahead generalises to feature
candidates with no new valuation logic:
- `_feature_candidates(g, available_features)` — the candidate set `available_features \ g.feature_set`
  (deterministic, sorted). The host-provided analogue of `_threshold_candidates`.
- `_add_feature(g, feat)` / `_remove_feature(g, feat)` — the grammar surgery (a fresh-id `Grammar` with
  `feat` added to / removed from `feature_set`; thresholds default for a new feature, drop for a removed
  one). Complexity *does* change here (|features| is in `compute_grammar_complexity`) — unlike thresholds,
  a feature is a genuine description-length unit (see §5 Q2).
- `explore_features(g, observations, available_features, max_depth; action_space, compute_cost)` — the
  belief-aware feature meta-action: full-eval argmax of `net_value(Δℓ, compute_cost)` over the candidate
  set, reusing `_grammar_marginal_log_loss` verbatim. Sibling of `explore_grammar` (§5 Q3 weighs
  unify-vs-sibling).
- `collect_feature_refs!(acc::Set{Symbol}, e::ProgramExpr)` — the sound full-depth feature-reference walk
  (the Move-1 `collect_nonterminal_refs!` pattern, one method per expr type, no generic fallback), for
  `:remove_feature` soundness.

**`src/program_space/perturbation.jl`** — modification. `analyse_posterior_subtrees` additionally
threads a `referenced_features::Set{Symbol}` (via `collect_feature_refs!`) alongside the existing
`referenced_nonterminals`, so `:remove_feature` has a *sound* count (a feature referenced by no
posterior-support program is dead). Mirrors Move 1 exactly.

**`src/program_space/types.jl`** — modification. `SubprogramFrequencyTable` gains
`referenced_features::Union{Nothing, Set{Symbol}}` (same `nothing`-sentinel discipline as Move 1's
`referenced_nonterminals` — fail-closed: un-analysed ⇒ no removal).

**`src/Credence.jl`** — export `explore_features`, `collect_feature_refs!`.

**Host wiring** (`apps/julia/grid_world/host.jl`, `apps/julia/email_agent/host.jl`): add
`:gw_add_feature`/`:gw_remove_feature` (and email equivalents) as meta-actions, gated on
*threshold-saturation* (§5 Q4); call `explore_features` with the host's `available_features`
(`ALL_GW_FEATURES` / `ALL_EMAIL_FEATURES_EXTENDED`); reset the residual regime on a feature change (an
alphabet expansion, like explore — §9.4 of Move 3).

**New test** `test/test_feature_discovery.jl` — candidate set, `_add_feature`/`_remove_feature`,
the discovery test (a grammar missing the relevant feature acquires it), the no-op, `collect_feature_refs!`
soundness (depth-1 reference seen), `:remove_feature` only drops dead features, determinism.

## 3. Behaviour preserved

- **Move 1/2/3 untouched.** `explore_grammar` (thresholds), `perturb_grammar` (compression), the
  saturation signal, the Grammar.thresholds field — all unchanged. `test_threshold_explore`,
  `test_voc_gate`, `test_saturation`, `test_program_space` stay green.
- **`SubprogramFrequencyTable` extension is additive** (the new `referenced_features` field defaults to
  `nothing` via a convenience constructor; existing 3-arg/4-arg call sites unaffected) — the Move-1
  precedent, asserted by re-running `test_voc_gate`/`test_perturb_consumption` `==`.
- **Host trajectories bit-stable until a feature meta-action fires** (gated on threshold-saturation, which
  is post-plateau), exactly as Move 3's explore was. Capture-before-refactor pins the pre-feature
  trajectory; the post-feature drift is the restored capability.

## 4. Worked end-to-end example

A grid_world grammar `g` with `feature_set = {:red, :green, :blue}` (a colour grammar). The true rule is
"enemy iff `wall_dist < 0.3`" — a feature `g` **does not use**, though the host extracts it every step.
The colour predicates never separate the classes, so after thresholds saturate (Move 3 finds no colour
split that helps) the residual sits at a non-zero floor (plateaued).

1. **Saturation escalation (host).** Threshold-refinement is exhausted (`explore_grammar` returns the
   input unchanged — no colour threshold clears VOI) **and** the residual is plateaued
   (`plateau_probability` high). Both ⇒ `:gw_add_feature` becomes admissible (§5 Q4 / §8.4 — the
   threshold-exhausted baseline makes the feature's VOI un-confounded, not a fine-before-coarse ordering gate).
2. **Candidate set (`_feature_candidates`).** `ALL_GW_FEATURES \ {:red,:green,:blue}` =
   `{:x_norm, :y_norm, :speed, :wall_dist, :agent_dist}`. Small and host-provided — full-eval, no screen.
3. **Lookahead (`explore_features` → `_grammar_marginal_log_loss`, reused verbatim).** For candidate
   `:wall_dist`: `g′ = g` with `:wall_dist` added to `feature_set` (fresh id; default grid for the new
   feature; complexity rises by 1). `enumerate_programs(g′, depth)` now includes
   `IF((lt :wall_dist 0.3), enemy, food)` etc.; replay the buffer; `Δℓ = mll(buffer|g) − mll(buffer|g′)`
   is large (the `wall_dist < 0.3` split predicts the data the colour grammar could not). `net_voi =
   Δℓ − compute_cost > 0`.
4. **Apply + reset (host).** `add_programs_to_state!(state, g′, depth)`; `reset_learning_regime!` + clear
   the buffer (the feature alphabet expanded). The ensemble can now represent the true rule; the residual
   drops off the floor.

Result: a feature the grammar *ignored* (but the host always extracted) is acquired by EU-max selection.
The colour-only grammar provably could not reach this rule. Owner of each step: host (gate, apply, reset)
↔ engine (candidates, lookahead, VOI).

## 5. Open design questions

### Q1 — Scope: base-feature selection only, or also composed-feature proposal? (the headline scope call)

> **Ratified (§8.1): base-feature selection only.** The master-plan §4 "combinations of existing features"
> tension dissolves on a *reading*, not an amendment — conjunctive/disjunctive combinations are already
> grammar-expressible (the program-space's job, compression abstracting a recurring `And(...)` into a
> nonterminal via `:add_rule`); only arithmetic *products* are genuinely new (the deferred §3.1 construction
> floor). So §4's "combinations" means the compositions the grammar already reaches, not the products it
> doesn't — and base-feature-only is *consistent* with the master plan, not a narrowing of it.

**Recommendation: base-feature selection only for Move 4; composed/product features deferred to a named
frontier.** Base-feature selection is *fully* EU-max (the host provides the candidate set and their
values — no construction), reuses Move 3 with zero new valuation logic, and needs no host-extraction or
AST changes. Composed features split two ways: conjunctions/feature×threshold are *already
grammar-expressible* (so not a feature-discovery concern at all — the grammar composes selected features
via And/Or once they're in `feature_set`), and *products/arithmetic* are **not** expressible and require
(a) a feature-arithmetic AST extension and (b) a brain *proposer* — hypothesis construction, the §3.1
creative floor. Folding products into Move 4 would drag the AST extension + the proposer into a move whose
clean claim is *selection*. Counter to weigh: the master plan §4 text says "proposals are combinations of
existing features," which reads as including products — so if you want the combination-proposer in Move 4,
say so and I will scope the AST extension; my read is that base-feature selection is the honest
"EU-max selection among proposed features" and products are the next rung.

### Q2 — Feature complexity: a feature IS a description-length unit (unlike a threshold)

> **Ratified (§8.2): keep the prior-Occam.** Feature discovery is priced on *both* axes (marginal-likelihood
> Δℓ + the §1.3 prior penalty), where a threshold was priced on the likelihood axis alone. The asymmetry
> *is* fine-before-coarse, made endogenous — and it is the exact converse of Move 3 Q1(b): there the
> fineness-Occam rode the *likelihood* (a threshold adds no symbol); here it rides the *prior* (a feature
> adds a symbol). Same Occam, correct axis each time.

Thresholds were complexity-invariant (Q1(b) of Move 3 — the fineness-Occam rode the marginal likelihood).
**Features are different: `compute_grammar_complexity` already counts `length(feature_set)`**, so adding
a feature raises `|G|` by 1 — a real prior penalty, and the *correct* one (a feature is a genuine new
symbol in the description, not a finer grid point of an existing one). **Recommendation: keep it — do not
make features complexity-invariant.** This means feature discovery is priced on *both* axes: the marginal
likelihood (the lookahead Δℓ, as for thresholds) *and* the prior complexity penalty (the `−log 2` per
feature already in the program log-prior). The two are consistent — the prior penalty is the honest
Occam cost of a new symbol, and the lookahead must overcome it. Confirm this is the intended asymmetry
(threshold = likelihood-Occam only; feature = likelihood-Occam + prior-Occam), because it is the precise
content of "fine-before-coarse": a finer threshold is *cheap* (no prior cost), a new feature is *dearer*
(a prior symbol), so the agent exhausts the cheap rung first.

### Q3 — One unified `explore` meta-action, or sibling `explore_grammar` / `explore_features`?

> **Ratified (§8.3): sibling now, unify at Move 5.** Unifying here would pre-empt the decision Move 5 exists
> to make; the siblings keep the different saturation-gates legible (plateau ∧ compression-exhausted for
> grammar; that ∧ threshold-exhausted for features) and line Move 5 up against the *currency frontier*
> (`explore_features` spans nats + a prior-Occam penalty, `explore_grammar` spans nats, `perturb_grammar`
> spans prior nats, the object level spans utility) rather than closing it early.

Both reuse `_grammar_marginal_log_loss` and the full-eval-argmax shell; they differ only in the candidate
generator (`_threshold_candidates` vs `_feature_candidates`) and the surgery (`_refine_grammar` vs
`_add_feature`). **Recommendation: sibling functions now, unify at Move 5.** A sibling `explore_features`
keeps the candidate-type and the saturation-gate (thresholds gate on compression-exhausted; features gate
on *threshold*-exhausted — a different escalation rung) legible and independently testable. Move 5's whole
job is the combined `argmax`, so premature unification here would pre-empt it. Counter: a single
`explore(g, observations, candidates, ...)` parameterised by a candidate generator is barely more code —
if you prefer the unified form now, it is a clean generalisation. Decide the seam.

### Q4 — The escalation gate: how does the host know thresholds have saturated?

> **Ratified (§8.4): the three-level lazy ladder — but banked on *attribution fidelity, not ordering*.** The
> gate is NOT a fine-before-coarse ordering device (Q2's pricing already orders; an ordering-gate would be
> redundant, or — if it deferred a correctly-measured positive-EU feature — a Move-2-forbidden cap). It is a
> *confound* guard: a feature's Δℓ measured against a coarse-grid baseline is inflated by residual that
> threshold-refinement would *also* have captured, so feature *evaluation* is deferred until the
> threshold-exhausted baseline exists and the feature is scored against what thresholds alone *cannot* reach.
> A sound deferral (unreliable measurement, bounded until exhaustion), not a cap. The ladder is *cyclic* — an
> added feature re-opens threshold refinement on its own grid — so `threshold_exhausted` must be lazily
> recomputed (rec a), never carried.

The original framing below was fine-before-coarse *ordering*; the **ratified reason is confound-avoidance
(§8.4)** — the mechanism recommendation (option a, lazy) is unchanged, but the *why* is attribution fidelity,
not ordering. Feature discovery should wait until *threshold* refinement is exhausted. Move 3 gated explore
on `compression_exhausted ∧ plateau`. Move 4's feature gate needs a **threshold-exhausted** signal. Two
options: **(a)** run `explore_grammar` and check it returns the input unchanged (no threshold clears VOI)
— *exact* but it pays the full threshold lookahead just to gate; **(b)** a cheaper proxy (e.g. the last
explore meta-action was a no-op, cached host-side). **Recommendation: (a), reusing the lazy pattern** —
the feature gate computes the threshold-exhausted check only when the belief-side EU (plateau-scaled) is
already positive (the Move-3 lazy-`compression_exhausted` shape), so the expensive check runs only when
features are even viable. The escalation is then: `plateau ∧ compression_exhausted ∧ threshold_exhausted`
⇒ features. Confirm the three-level ladder (compress → refine thresholds → add features) is gated this way,
or whether `threshold_exhausted` should be a carried signal rather than recomputed.

## 6. Risk + mitigation

1. **`:remove_feature` unsoundness (the Move-1 risk, redux).** A feature referenced only in a
   complexity-1 predicate that `extract_subtrees` drops would be misread as dead → removal corrupts a
   support program. Mitigation: `collect_feature_refs!` is a *separate* full-depth walk (NOT routed
   through `extract_subtrees`), one method per expr type, no generic fallback — exactly Move 1's
   `collect_nonterminal_refs!`. Tested by `test_feature_discovery.jl` (a depth-1 `GTExpr(:f, t)` reference
   is seen). The `nothing`-sentinel keeps un-analysed tables removal-free (fail-closed).
2. **Feature-removal vs feature-addition fighting (the Move-3 review risk, redux).** Like
   `perturb_grammar` dropping a refined grid, a feature op must thread `g.thresholds` through (drop only
   the removed feature's grid). Mitigation: `_add_feature`/`_remove_feature` build via the 4-arg
   constructor with an explicitly-adjusted threshold Dict; a test asserts a refined *other* feature's grid
   survives an add/remove.
3. **Escalation gate too eager / too costly (Q4).** If features fire before thresholds saturate, the
   agent skips the cheap rung; if the gate recomputes the threshold lookahead every step, it is slow.
   Mitigation: the lazy three-level gate (Q4 rec a); a test that feature discovery does NOT fire while a
   threshold refinement still clears VOI.
4. **Pre-emptive grep** for `referenced_nonterminals` / `SubprogramFrequencyTable(` call sites before the
   field addition (the Move-1 blast radius), each dispositioned.

## 7. Verification cadence

```
julia test/test_feature_discovery.jl     # candidates, add/remove, discovery, soundness, determinism
julia test/test_threshold_explore.jl     # Move 3 untouched (incl. the refined-grid-survives-compression pin)
julia test/test_voc_gate.jl              # Move 1 reference-count + compression untouched
julia test/test_program_space.jl         # enumeration + the freq-table extension bit-stable
julia test/test_grid_world.jl            # host wiring + the three-level escalation gate
julia test/test_email_agent.jl           # second host
```

Full `test/test_*.jl` green before commit; lint self-test + `check apps/`. Skin smoke is **optional** (no
wire change expected — features, like thresholds, are host-orchestrated; `:add_feature` adds no wire
verb), but run it if the `SubprogramFrequencyTable` field touches a serialised path. Halt-the-line on any
failure.

## 8. Ratification (2026-06-29)

All four ratified. Two refinements change the doc's *reasoning* (not its mechanism): Q1 resolves by a
reading rather than an amendment (§8.1), and Q4 is re-banked from *ordering* onto *attribution fidelity*
(§8.4). Recorded here as the authoritative outcome.

### 8.1 — Q1: base-feature selection only. Ratified.

The master-plan §4 "combinations of existing features" tension **dissolves on a reading, not an amendment**.
"Combinations" splits cleanly in two, and the split is the whole answer:

- *Conjunctive / disjunctive* combinations — `And`/`Or` over `GT`/`LT` — are **already grammar-expressible**,
  so they are the *program-space's* job: compression abstracts a recurring `And(...)` into a nonterminal via
  `:add_rule`. They were never feature-discovery's work.
- *Arithmetic* combinations — `red × blue` — are the **only genuinely new dimension**, and proposing one is
  hypothesis *construction*, the deferred §3.1 floor.

So §4's "combinations" means the compositions the grammar already reaches, not the products it doesn't —
and under that reading base-feature-only is **consistent with the master plan, not a narrowing of it**. No
master-plan edit; the clarification lives here. Move 4's honest headline is exactly what the thesis licensed:
**EU-max *selection* over host-furnished features**, stopping one rung short of proposing the features that
aren't there yet.

### 8.2 — Q2: keep the prior-Occam. Ratified.

Charging `length(feature_set)` prices feature discovery on **both** axes — the marginal-likelihood `Δℓ` and
the §1.3 prior penalty — where a threshold was priced on the likelihood axis alone. That asymmetry is not a
wart; it **is fine-before-coarse, made endogenous**: a feature carries a strictly higher bar (it must repay a
prior symbol a threshold never owed), so thresholds clear while they pay and features wait until they don't —
the ordering *falls out of EU-max pricing* rather than being imposed on top of it. This is the satisfying
**converse of Move 3 Q1(b)**: there the fineness-Occam rode the *likelihood* because a threshold adds no
symbol; here it rides the *prior* because a feature does. Same Occam, correct axis each time.

### 8.3 — Q3: sibling now, unify at Move 5. Ratified.

Unifying `explore_grammar` and `explore_features` here would **pre-empt the decision Move 5 exists to make**;
the siblings keep their different saturation-gates legible (plateau ∧ compression-exhausted for grammar; that
∧ threshold-exhausted for features) and independently testable. Let the unification move own the unification —
and note what staying siblings *buys*: it lines Move 5 up against the **currency frontier** rather than
closing it prematurely. After Move 4, `explore_features` spans nats *plus* a prior-Occam penalty,
`explore_grammar` spans nats, `perturb_grammar` spans prior nats, the object level spans utility — and forcing
those into one currency is precisely the thing that defeats cheap compression if pushed. Keep them siblings;
let Move 5 reach that wall on purpose.

### 8.4 — Q4: the three-level lazy ladder. Ratified as a *mechanism*; re-banked on *attribution fidelity*.

The mechanism is unchanged — `plateau ∧ compression_exhausted ∧ threshold_exhausted ⇒ features`, with
`threshold_exhausted` computed lazily (rec a, the Move-3 pattern). But the **justification is load-bearing and
was wrong in the question as posed**: it is *not* fine-before-coarse *ordering*.

- A `threshold_exhausted` gate looks like a **cap** — it defers a possibly-positive-EU feature, which Move 2's
  one-sidedness forbids. And "ordering" cannot rescue it: Q2's pricing already does the ordering, so an
  ordering-gate would be either redundant (pricing already defers the dear rung) or, if it blocked a
  *correctly-measured* positive-EU feature, a genuine cap.
- The real reason is **confound**. Evaluated against a coarse-grid baseline, a feature's `Δℓ` is *inflated*,
  because it captures residual that threshold-refinement would *also* have captured. The gate defers feature
  *evaluation* until the threshold-exhausted baseline exists, so the feature is scored against **what
  thresholds alone cannot reach**. That is the project's own contingency law in another guise — exploration's
  value is contingent on attribution fidelity; horizon-aware VOI bled out under the fair condition precisely
  through soft-credit *misattribution*. The gate is the same lesson: *don't act on a confounded VOI*.
- So it is a **sound deferral** (defer because the measurement is unreliable, bounded until exhaustion), not a
  block of a correctly-measured positive-EU explore — consistent with Move 2 Q3, because you cannot cheaply
  tell an orthogonal feature from a confounded one without the exhausted baseline, which makes **uniform
  deferral the resource-rational response**, not a gratuitous cap.

Two corollaries to carry into code:

1. **Q2 and Q4 are complementary, not redundant.** Q2 is a *value* adjustment (features are dearer); Q4 is an
   *attribution* adjustment (features are measured against the right baseline). Pricing alone will not fix a
   confound that survives the prior penalty — both are needed.
2. **The ladder is cyclic, not one-shot.** Adding a feature **re-opens** threshold refinement on that
   feature's own grid: refine-to-exhaustion → add-feature → new thresholds re-open → refine again. Move 2's
   regime-reset on grammar change is exactly what re-enters `:improving` to drive the next pass. This is why
   `threshold_exhausted` must be **lazily recomputed, never carried** — a cached signal would go stale across
   the cycle.

### Net

Base-feature only, with "combinations" clarified as the compositions the grammar already expresses; keep the
feature prior-Occam (fine-before-coarse is now endogenous); the three-level ladder banked as *attribution
fidelity*, not ordering; siblings until Move 5. Move 4 claims **EU-max selection over host-furnished
features** and stops, correctly, one rung short of proposing the features that aren't there yet.
