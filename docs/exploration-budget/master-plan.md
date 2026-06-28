# Credence — Core-library engine arc: *exploration-budget* (master plan)

> Durable, in-repo master plan for the `exploration-budget` branch family. A thematic engine arc
> (unnumbered, like `collapse-towers`, `decouple`, `measure-as-view`). It **discharges the two named
> successors collapse-towers deferred** (`docs/collapse-towers/master-plan.md` §"Named successors"):
> **Scope B (`:remove_rule`)** and the **EU-priced exploration budget**. Phase-5 of collapse-towers
> retired the `rand` breach by shipping *Scope A* (`:add_rule` only, depth-one prior VOC) and explicitly
> named this arc as the *immediate, adjacent* successor — because Scope A leaves the agent unable to
> discover **any** new feature or threshold until it lands. Each move: design-doc-before-code
> (`docs/exploration-budget/DESIGN-DOC-TEMPLATE.md`), each commit green + bisectable, **stop-and-report at
> every move boundary.**
>
> **STATUS: DRAFT — thesis pending ratification (2026-06-28).** Per the thesis-first discipline, the move
> sequence below is *provisional*; it is not finalised until §3 (the thesis) and §5 (open questions) are
> ratified. Do not open Move 1's design doc until then.

## 1. Context — what Scope A left undone, and why it is hard

After collapse-towers, `perturb_grammar(g, freq_table, available_features; compute_cost)` is **prior-only
and depth-one** (`perturbation.jl`). It does exactly one thing: `:add_rule` — define a frequent posterior
subtree as a nonterminal when `net_voc = Δcomplexity_logprior − compute_cost > 0`. This is the
**compression class** — it re-describes the *same* hypotheses more compactly (a prior effect), and
depth-one VOC over `(g, freq_table)` can value it exactly (`net_payoff`).

Everything that grows the agent's *expressiveness* was deferred, for a reason that is structural, not
incidental (collapse-towers Phase-5 R2, ratified): the **generative-change class** —
`:modify_threshold`, `:add_feature`, `:remove_feature` — changes *which* hypotheses the grammar
generates. Its value is a **likelihood effect over programs not in the current ensemble** — the
escape-mass / Cromwell frontier — and is **invisible to depth-one prior-only VOC by construction**. You
cannot price the value of a hypothesis you have not yet entertained from the prior alone; the prior over
the un-entertained region is exactly what the complexity prior truncates so the tower does not regress
(SPEC §1.3). Phase-5's resolution was honest: exclude the three, defer them, *never* re-introduce `rand`
or an arbitrary deterministic tiebreak (that only launders the breach).

Concretely, today (Explore audit, 2026-06-28):
- **Features** are `Set{Symbol}` in `Grammar`, fixed for the grammar's life; **no path adds one.**
- **Thresholds** are a hard-coded `const THRESHOLDS = [0.1,0.3,0.5,0.7,0.9]` (`enumeration.jl:9`),
  applied uniformly; **no path refines them.** A threshold constant is complexity-invariant
  (`expr_complexity(::GTExpr)=1`), so `:modify_threshold` leaves the prior *identically* unchanged — it
  is purely a fit effect.
- `available_features` is **accepted but unused** in Scope A — the placeholder hook for this arc.
- The `freq_table` is **lossy at `min_complexity=2`** (drops bare-reference programs), which is the
  precise reason Scope B (`:remove_rule`) is blocked: it cannot supply a *sound* nonterminal reference
  count, only an estimable-not-provably-sound one (the stall gate's concern).

## 2. The architectural crux (R1, redux) — exploration is belief-aware; compression is not

Phase-5 R1 established that `perturb_grammar` is prior-only *by signature*: it receives no belief, no
utilities, no programs, no re-conditioning — so it can only price the prior. **That is exactly why the
generative-change class cannot live inside it.** The value of expanding the alphabet is realised only
against the **belief's predictive residual** — where the current posterior-weighted ensemble
*mispredicts the data* — and measuring it requires re-enumeration + re-conditioning (the very forward
inference the metalevel is deciding whether to spend). So the load-bearing structural decision of this
arc:

> **The exploration budget is a *belief-aware* meta-action, categorically distinct from the prior-only
> compression op.** Compression (Scope A/B) stays in `perturb_grammar` (prior currency, depth-one,
> wire-stable). Exploration (thresholds, features) is a *new* belief-aware entry that does
> **compute-budgeted lookahead** against the residual. The host orchestrates the two; they are never
> conflated.

This is the `measure-as-view` lesson transposed: just as carrier-bound ops are Measure-resident (not a
wart but the view relationship working), **belief-dependent value is belief-level (host/belief-aware),
not prior-level** — and the bright line between them is the architecture, not a defect.

## 3. The thesis (the falsifiable claim — RATIFY FIRST)

> **Generative-change grammar discovery can be made EU-max — not random, not capped — by pricing
> alphabet expansion as *compute-budgeted lookahead value-of-information against the belief's predictive
> residual*, gated on *saturation* of the current discrete partition.**

Mechanism. Depth-one prior-only VOC cannot see generative-change value (R2). So the agent spends a
**compute budget** to do **depth-≥1 lookahead**: for a candidate threshold/feature, *actually* expand
the grammar, re-enumerate, re-condition against the (residual of the) belief, and measure the **realised
value gain**. How much lookahead to spend — how many candidates, how deep — is itself `argmax EU` with
**compute in the utility** (resource-rational, graceful degradation, *no hard cap* — the standing
metareason-not-cliff direction). **Saturation is the precondition**: explore only when compression has
exhausted *and* the residual has plateaued within the current alphabet, because that is exactly when
expansion has positive expected VOI. The escalation is **fine-before-coarse** (the discrete partition
must demonstrably saturate before continuous features are admissible): compress → saturate → refine
thresholds (finer partition *within* features) → add features (new dimensions) → (far future) continuous
features / embeddings.

This is not new doctrine. It is the heuristics-inside-EU-max clause (CLAUDE.md Invariant 1: *"when
computational cost enters the utility … it is the optimal strategy"*) and the complexity-prior /
escape-mass framing (SPEC §1.3/§1.4) applied to the *metalevel's own* discovery problem. Exploration
strategies (UCB/Thompson/ε-greedy) are *not forbidden as concepts* — but they must **emerge** as
EU-optimal under the compute-priced VOI, never be hard-coded.

**Falsifiable predictions (the empirical gates):**
1. **Discovery:** on a task whose optimal predicate is *outside* the initial alphabet, the
   saturation-gated residual-driven explorer finds it; Scope A alone provably cannot (it never expands).
2. **Dominance:** it beats both (a) the retired *random* exploration and (b) a *fixed-schedule /
   fixed-budget* baseline, on sample-efficiency or final realised value.
3. **Graceful degradation:** under a tighter compute budget it explores *less* (fewer/shallower
   lookaheads) and the explore/exploit split moves *smoothly* with the budget — never a cliff, never a
   crash, never a hard cap.

If the thesis does not survive review — in particular if no cheap-enough residual signal or lookahead
makes prediction 3 hold without a cap — the arc **stalls at this master plan**; we do not ship a guess
(the same stall gate Phase 5 honoured).

## 4. Provisional move skeleton (pending §3/§5 ratification)

Scope B and the exploration budget are **one arc** (coupled: they share the perturbation machinery, and
Scope B completes the compression pair that makes *saturation* precisely definable — the precondition for
exploration). Five provisional moves:

- **Move 1 — Complete the compression pair: Scope B (`:remove_rule`) + a sound reference count.**
  Discharge the Scope-B deferral. Fix the lossy-`freq_table` blocker so the nonterminal reference count
  is *provably sound* (OQ-4). With add+remove, the dictionary is hygiene-complete and **saturation gets a
  precise prior-side definition** (no compression op improves the prior). Prior-only, depth-one — extends
  the existing `net_voc`. Foundational; no belief needed yet.
- **Move 2 — The saturation signal (the belief-aware meta-level entry).** Define + implement saturation:
  compression exhausted (Move 1) **and** the belief's predictive residual plateaued. This is the first
  belief-aware meta computation; it establishes the entry point exploration will hang off, and it makes
  *explore-vs-exploit* an EU comparison (expected compression value vs expected exploration VOI).
- **Move 3 — Compute-budgeted lookahead VOI for threshold refinement.** The cheapest, safest
  generative-change op: adaptive thresholds (the hard-coded grid becomes refinable *within* existing
  features). Establishes the lookahead-VOC mechanism (re-enumerate/re-condition a candidate, measure
  realised gain, EU-max over how much lookahead, saturation-gated). Fine-before-coarse: refine before
  adding.
- **Move 4 — Feature discovery (`:add_feature` / `:remove_feature`).** Add a feature from
  `available_features`, priced by the Move-3 lookahead-VOC, gated on threshold-refinement saturating.
  `:remove_feature` reuses Move 1's reference-count soundness.
- **Move 5 — The combined single-currency `argmax` (close the collapse-towers headline).** Fold
  compression (exploit, prior currency) and exploration (expand, lookahead-utility currency) into **one**
  `argmax EU` with compute in the utility — the *"combined single-currency `argmax` over object-and-meta
  space"* Phase-5 named as "the next escape-mass frontier." This makes the metalevel literally one
  `optimise`, completing collapse-towers' headline. May fold into Move 4 or stand alone (OQ-5).

Each move lands design-doc-then-code; the empirical predictions (§3) are checked at Moves 3–5 on a
purpose-built task (a paper-style figure, per the paper-as-gating-artifact discipline if this feeds a
publication).

## 5. Open design questions (ratify before finalising the move sequence)

1. **The valuation mechanism (the thesis's heart).** Compute-budgeted **lookahead** (actually
   re-enumerate/re-condition candidates — honest VOI, exact, expensive, budget-bounded) **vs** a cheaper
   **surrogate** (a hyper-prior over feature relevance + an emergent-from-EU bandit). *Recommendation:
   lookahead* — it is the honest value-of-information, the budget bounds the cost, and a surrogate risks
   smuggling in a non-EU exploration rule (the thing Invariant 1 forbids unless it *emerges*). Counter to
   weigh: lookahead may be too expensive to make prediction-3 (graceful degradation) hold cheaply.
2. **The architectural home (R1 redux, §2).** A *new belief-aware entry* (`explore_grammar(belief, g,
   available_features; compute_budget)`, separate from prior-only `perturb_grammar`) **vs** extending
   `perturb_grammar` to optionally carry the belief/residual. *Recommendation: separate entry* — keeps
   compression prior-only and wire-stable, and reflects the compression-is-prior / exploration-is-belief
   distinction. Counter: two entry points the host must orchestrate vs one.
3. **The saturation definition / residual metric.** Compression-exhausted *alone* (prior-only, simple)
   **vs** compression-exhausted *+ residual-plateau* (belief-aware, the real precondition per the
   saturation-precondition direction). If the latter: *which* residual metric — predictive log-loss
   plateau, posterior-entropy plateau, or held-out fit? *Recommendation: the belief-aware definition*
   (saturation must be *demonstrated*, not assumed); metric TBD in Move 2's design doc, leaning
   predictive-log-loss-plateau on recent observations.
4. **The Scope-B soundness fix (OQ-4).** Lower `min_complexity` for reference-scanning (cheap, but
   changes `freq_table` semantics for everyone) **vs** thread an explicit nonterminal reference count
   through enumeration (cleaner, more work, sound by construction). *Recommendation: explicit reference
   count* — Scope B was deferred *precisely* because the estimate wasn't provably sound; a real count
   removes the stall-gate concern rather than papering it.
5. **Does Move 5 actually close the headline?** Is "the metalevel is one `argmax EU`" achievable once
   exploration has a (lookahead-derived) utility-currency value commensurable with compression's prior
   value — or does the currency gap Phase-5 flagged remain, leaving Move 5 a *named* frontier rather than
   a *closed* one? *Recommendation: attempt it at Move 5; if the commensurability doesn't hold honestly,
   stop and name it* (do not force a fake common currency — the Phase-5 precedent).

## 6. Hard constraints (inherited)

Spec-first; design-doc-before-code; stop-and-report at every move boundary; **no new constitutional
text** (if a move seems to need it, stop and report); **no `rand` / no arbitrary tiebreak in action
selection** (the breach Phase 5 closed stays closed — exploration is EU-max or it does not ship); **no
hard compute cap** (metareason fidelity against a budget — graceful degradation); heuristics live
*inside* EU-max, never alongside it; no silent approximations (an approximation ships only if a strict
Bayesian/EU argument validates it — else stall); tolerance inside the boolean; no `using Test`;
**capture-before-refactor** on every behaviour-preserving step (canonical values pinned PRE-change,
asserted `==`); seeded-RNG `==` for any sampling lookahead. `λ` per-axis (program axis pinned `ln 2`).

## 7. Key risks

1. **Lookahead too expensive / prediction-3 fails (the stall risk).** If compute-budgeted lookahead
   cannot degrade gracefully without a cap, the thesis fails — *stall at this plan*, do not cap.
   Mitigation: Move 2's saturation gate sharply bounds *when* lookahead runs (only post-saturation), and
   the budget bounds *how much*; if even that is too costly, escalate rather than approximate.
2. **Scope-B reference-count unsoundness (OQ-4).** A rule referenced only in low-complexity contexts
   misread as dead → `:remove_rule` corrupts the dictionary. Mitigation: a *sound* count (recommendation
   3 above), or do not ship Scope B (keep the deferral honest).
3. **Benchmark drift / capability-vs-cleanliness.** Restoring exploration changes grid_world /
   email_agent trajectories (intended — it restores a capability Scope A removed). Capture-before-refactor
   pins the *mechanism*; the *behavioural* change is the point and is documented per move.
4. **Over-scoping (the thesis-first risk this plan guards against).** Five moves is the *ceiling*; if the
   thesis ratifies a cheaper mechanism (OQ-1 surrogate, OQ-5 no combined argmax), the arc shrinks. The
   skeleton is provisional precisely so it can.
