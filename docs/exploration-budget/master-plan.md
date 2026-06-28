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
> **STATUS: RATIFIED 2026-06-28 (author).** Thesis ratified — and sharpened: the selection/generation
> seam (§3.1) scopes what each move may *claim*. All five open questions ratified as recommended, with the
> reasoning strengthened from "preferred" to "forced" where it is forced (§5). The move sequence is final.
> Move 1's design doc may open.

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

### 3.1 What is EU-max here, and what is the irreducible residue (selection vs generation)

The thesis makes candidate **selection** EU-max — lookahead VOI *ranks* candidates. It presupposes a
candidate *set* to rank, and where that set comes from — candidate **generation** — is not itself priced
by the lookahead. The two halves of the arc differ exactly here, and the victory must be scoped to match:

- **Thresholds — EU-max *generative* discovery, claimed without hedging.** A threshold only matters at an
  *observed* value, so the observed values **are** the complete, finite candidate set. Generation is
  trivial and exhaustive; lookahead ranks all of them; the selection *is* the generation. For thresholds
  the frontier closes fully — this is the move that earns the arc's headline.
- **Features — EU-max *selection over a residual-proposed set*; novel-feature synthesis is the standing
  frontier.** A feature is an arbitrary function of the observations, so the candidate set is unbounded
  and something must **propose** which features to look ahead over. That proposer is doing hypothesis
  *construction* — and (as established in the founding exchange) metareasoning *allocates inference over a
  hypothesis space; it does not build one*. A proposer that enumerates **combinations of existing
  features** (products, conjunctions, feature×threshold predicates) stays tractable and near-complete and
  is legitimately within EU-max selection. The moment a proposer must synthesise a *genuinely novel
  dimension*, it is at the **creative floor**, and no pricing makes that step EU-derived. **Move 4 may
  therefore claim "EU-max selection among proposed features," never "EU-max feature generation"**; novel
  synthesis is named as a permanent frontier — the same floor the project's first exchange identified, now
  made architectural. (This is *why* fine-before-coarse: threshold refinement is fully closable, feature
  combination is closable, novel features are not — the escalation is ordered by how much of generation
  the lookahead can absorb.)

### 3.2 The two-tier resource-rational structure (why the regress terminates)

The expensive lookahead is made tractable by a cheap belief-aware *screen*. The four mechanisms divide
the labour and truncate the metalevel regress at the right place:

> **residual screens *where* · saturation gates *when* · the compute budget bounds *how many* · lookahead
> evaluates the survivors.**

Without the residual screen the lookahead would range over everything and the regress would not terminate;
the residual is the cheap signal that nominates a *small* survivor set for the expensive exact evaluation.

### 3.3 Gate 3 is the one that will bite — and the cap is forbidden

Predictions 1–2 (discovery, dominance) are almost certainly demonstrable. Prediction 3 (smooth
degradation without a cap) depends on the candidate-VOI distribution having **no cliff**, which cannot be
guaranteed a priori — so the stall gate is *genuine* and *will be tested under pressure*: the agent cannot
explore **at all** until this lands, so there will be a strong pull to ship a *capped* version and call it
graceful. **Do not.** A hard cap is the non-EU lever the constitution forbids (metareason fidelity against
a budget, never a cliff). "We shipped something" is *worse* than "we stalled honestly" here, because a
capped explorer that *looks* like it works is harder to dislodge than an absent one. If gate 3 cannot hold
without a cap, stall and escalate.

## 4. Move sequence (ratified 2026-06-28)

Scope B and the exploration budget are **one arc** — and the coupling is causal, not cosmetic: **Scope B
*enables* Move 2.** "Compression is exhausted" — half of the saturation gate — is *undefinable* until the
compression pair (add + remove) is complete, so Scope B is a genuine precondition, not a rider. Five
moves:

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
  generative-change op, and the one that **fully closes EU-max *generative* discovery** (§3.1): a threshold
  matters only at *observed* values, so the observed values *are* the complete finite candidate set —
  generation is exhaustive, lookahead ranks all of it. Establishes the lookahead-VOC mechanism
  (re-enumerate/re-condition a candidate, measure realised gain, EU-max over how much lookahead,
  saturation-gated). This move earns the arc's headline; claim it without hedging.
- **Move 4 — Feature discovery (`:add_feature` / `:remove_feature`).** Add a feature from a
  **residual-proposed candidate set** — proposals are *combinations of existing features* (products,
  conjunctions, feature×threshold predicates), which keeps generation tractable and near-complete — priced
  by the Move-3 lookahead-VOC, gated on threshold-refinement saturating. **Claim: "EU-max *selection* among
  proposed features," never "EU-max feature *generation*"** (§3.1); novel-dimension synthesis is named as
  the standing creative frontier, not implemented. `:remove_feature` reuses Move 1's reference-count
  soundness.
- **Move 5 — The combined single-currency `argmax`: attempt, expect to stop (close-or-name the headline).**
  Exploration's lookahead VOI is *already* in utility (realised value gain), so it unifies with the object
  level for free — the lone outlier is **compression**, priced in prior nats. So the combined `argmax`
  faces *one* question, not two currencies: can compression's nats become utility? The prediction is **no,
  and stopping is correct**: compression is cheap *because* it is depth-one prior-priced; pricing it in
  utility means pricing it by lookahead, which makes it as expensive as exploration and **destroys the
  cheap-screen-first ordering the whole design rests on**. So the currency gap is not a representational
  accident to convert away — it is the **signature of the two compute tiers** (cheap prior screen vs
  expensive utility lookahead). Move 5 attempts the unification, and on hitting this, **names it a permanent
  frontier with that reasoning** (Phase-5 precedent — do not force a fake common currency). May fold into
  Move 4 (OQ-5).

Each move lands design-doc-then-code; the empirical predictions (§3) are checked at Moves 3–5 on a
purpose-built task (a paper-style figure, per the paper-as-gating-artifact discipline if this feeds a
publication).

## 5. Open design questions

> **All five RATIFIED as recommended 2026-06-28 (author), reasoning strengthened:**
> - **Q1 — lookahead, *forced* not merely preferred.** The surrogate is not just "risks a non-EU rule" —
>   it is **unsound by the same R2 argument that motivates the arc**: a hyper-prior over the relevance of
>   features *outside* the current alphabet is a prior over the un-entertained region, exactly what the
>   complexity prior truncates so the tower does not regress. There is no sound surrogate there *by
>   construction*; lookahead is the only honest valuation. If it is too costly → **stall**, never fall back
>   to the unsound surrogate.
> - **Q2 — new `explore_grammar`, *forced* by R1.** Threading a belief into prior-only `perturb_grammar`
>   breaks the wire and collapses the prior/belief seam the design rests on. Two entry points is not a cost
>   — it is **the seam made visible in the types** (the measure-as-view transposition: carrier-free
>   Prevision / carrier-bound Measure ↦ prior-only compression / belief-aware exploration).
> - **Q3 — compression-exhausted *and* residual-plateau.** Tie "compression-exhausted" to the existing
>   `net_voc ≤ 0` so the **prior-side half is free**; Move 2's real work is the *belief-side* plateau metric.
>   Compression-exhausted-alone assumes what must be shown.
> - **Q4 — explicit reference count.** This is precisely what "blocked on a sound reference count" was
>   waiting for. The lossy `freq_table` cannot supply soundness; lowering `min_complexity` changes
>   `freq_table` semantics for *every* consumer (non-local blast radius for a local need). Thread the count
>   — sound by construction.
> - **Q5 — attempt, expect to stop; the currency gap *is* the architecture** (see Move 5, §4). Name it a
>   permanent frontier with the two-compute-tiers reasoning, rather than forcing a conversion that would
>   quietly defeat cheap compression.
>
> The prose below is retained as the rationale of record.

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
