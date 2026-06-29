# Move 3 design — compute-budgeted lookahead VOI for threshold refinement (the headline)

> Exploration-budget arc, Move 3. Design-doc-before-code; ratify before any code lands.
> Master plan: `docs/exploration-budget/master-plan.md` (§3.1 scopes what this move may *claim*;
> §3.2 the two-tier screen/gate/budget/lookahead division; §3.3 the gate-3 stall risk).
> Predecessors on master: Move 1 (`:remove_rule` + sound reference count, #166), Move 2 (the
> saturation signal, #168). Authored 2026-06-29.

---

## 1. Purpose

Move 3 as scoped in the master plan (§4): the **first belief-aware generative-change op** — refine a
feature's threshold grid against the belief's predictive residual, priced by **compute-budgeted
lookahead VOI**, gated on saturation. It earns the arc's headline because thresholds are the op for
which EU-max *generative discovery* **fully closes** (§3.1): a threshold only matters at an *observed*
feature value, so the observed values **are** the complete finite candidate set — generation is
exhaustive, lookahead ranks all of it, the selection *is* the generation. No proposer, no creative
floor.

Two refinements/additions over the master-plan framing:

- **Threshold refinement is the *purest* generative-change op, and that is why it is the clean test of
  the mechanism.** A threshold constant is complexity-invariant at the program level
  (`expr_complexity(::GTExpr) = 1`, independent of the threshold value), so refining the grid has
  **zero prior-side signal** — it is *all* fit effect. Prior-only depth-one `net_voc` (Move 1) is
  therefore *identically blind* to it by construction; the only honest valuation is lookahead against
  the belief (Q1 of the master plan, *forced*). Thresholds isolate the lookahead mechanism with no
  prior-side confound.
- **Move 3 is the first *consumer* of the Move-2 saturation signal, so it lands the host wiring Move 2
  deferred.** Move 2 shipped the signal mechanism (the regime BMA, `compression_exhausted`,
  `update_learning_regime`, `reset_learning_regime!`) but nothing fed it the residual or read its
  verdict. Move 3 wires the residual feed at the host conditioning sites and reads `plateau_probability`
  as the soft saturation gate on `explore_grammar` — closing the loop the deferral opened.

What unblocks: with thresholds closed, Move 4 (features) reuses the identical lookahead-VOC mechanism
over a *residual-proposed* candidate set; Move 5 attempts the combined single-currency `argmax`.

## 2. Files touched

**New engine file `src/program_space/exploration.jl`** (included in `ontology.jl` after
`saturation.jl`; the belief-aware sibling of prior-only `perturbation.jl`):
- `explore_grammar(belief, g, observations; compute_cost)` — the belief-aware entry (Q2-master,
  *forced* separate from `perturb_grammar`). Returns a refined `Grammar` (new id) or the input `g`
  unchanged (the no-op, when no candidate clears net VOI).
- `_threshold_candidates(g, observations)` — the finite candidate set: residual-screened split-points
  from observed feature values (§5 Q2).
- `_lookahead_voi(belief, g, candidate, observations; compute_cost)` — re-enumerate the candidate
  grammar, re-condition against `observations`, return net VOI in predictive-log-loss nats (§5 Q3).
- the sequential cap-free budget loop (evaluate the next screened candidate while expected gain >
  `compute_cost`; §5 Q3).

**`src/program_space/types.jl` (`Grammar`, ~141–156)** — modification. Add a per-feature threshold
field (§5 Q1); default it to the global grid so every existing `Grammar(feature_set, rules, id)` call
(34 sites) enumerates **bit-identically**. The exact field shape is Q1.

**`src/program_space/enumeration.jl` (~9, ~62–144)** — modification. `enumerate_programs` reads the
grammar's threshold field instead of the global `const THRESHOLDS` (which becomes the default seed).
Bit-exact for any grammar carrying the default grid.

**`src/Credence.jl`** — export `explore_grammar`.

**Host wiring (the Move-2 deferral):**
- `apps/julia/grid_world/host.jl` — feed the already-computed `surprise` (host.jl:398 — it *is*
  `ℓ = −log predictive`) into `update_learning_regime` around the conditioning site (host.jl:417);
  call `reset_learning_regime!` at the grammar-change sites (`execute_gw_meta_action!`, ~356); add
  `:gw_explore` to `GW_META_ACTIONS` gated on `plateau_probability` (the integration).
- `apps/julia/email_agent/host.jl` — the analogous residual is `−log action_probs[observed]` from
  `build_predictive` (host.jl:124); same feed + reset + meta-action wiring at its sites.

**Skin (`apps/skin/server.jl`, ~631–644)** — *conditional*. If the refined grammar must round-trip the
wire (Q1 decides whether the threshold field is wire-visible), `grammar_from_spec` and the grammar
serialisation carry it. If `explore_grammar` is host-internal (like the meta-action loop), no new verb.
Decided in Q1/Q4.

**New test file `test/test_threshold_explore.jl`** — discovery, degenerate no-op, cap-free budget
monotonicity, the capture-before-refactor enumeration pin (§3, §7).

## 3. Behaviour preserved (capture-before-refactor)

The arc's standing discipline: canonical values pinned **PRE-change**, asserted `==`.

- **Enumeration bit-exactness (the load-bearing pin).** For a representative grammar carrying the
  *default* (global) threshold grid, `enumerate_programs(g, depth)` returns a byte-identical
  `Vector{Program}` before and after the `Grammar` threshold-field change. Captured pre-change,
  asserted `==` (not `isapprox`). This is what guarantees the 34 `Grammar(` sites and every existing
  `test_program_space.jl` / host-benchmark trajectory are untouched until a grammar is *actually*
  refined. Strata-1 equivalence, `==`.
- **`perturb_grammar` (compression) untouched.** `explore_grammar` is a separate entry; the prior-only
  compression path is not edited. `test_voc_gate.jl`, `test_program_space.jl` stay green unchanged.
- **Move-2 saturation signal untouched.** `test_saturation.jl` stays green `==`. The host wiring only
  *feeds* the signal; `initial/update/plateau` are not edited.
- **Conditioning bit-exactness.** The host residual feed is **additive** — `surprise` was already
  computed (grid_world host.jl:398); we route a copy into `update_learning_regime`. The
  `condition(belief, k, 1.0)` call is bit-identical; belief trajectories up to the first *explore* are
  unchanged.
- **Behavioural change is gated and intended.** Once `:gw_explore` fires (post-saturation), grid_world /
  email_agent trajectories move — that is the restored capability (Scope A removed all generative
  change). Per the master plan §7.3, capture-before-refactor pins the *mechanism*; the *behavioural*
  shift at saturation is the point and is documented, not suppressed.

## 4. Worked end-to-end example

A grid_world grammar `g` with feature `:dist` and the default grid `[0.1, 0.3, 0.5, 0.7, 0.9]`. The
true class boundary is `dist ≈ 0.62` — **off-grid**. The nearest on-grid splits (0.5, 0.7) both
misclassify the 0.5–0.7 band, so after the ensemble has absorbed all on-grid structure the predictive
residual `ℓ` settles at a **non-zero floor** (the band is permanently mispredicted) — a plateau, not at
zero.

1. **Residual feed (host, grid_world host.jl ~398–417).** Each step the host computes
   `surprise = −log p_obs` (already there) and calls
   `state.learning_regime = update_learning_regime(state.learning_regime, state.last_residual, surprise)`,
   then `state.last_residual = surprise`. Owner: host → `Ontology.update_learning_regime` (Tier-1
   `condition` on the regime BMA).
2. **Saturation gate (host meta-loop).** `compression_exhausted(g, freq_table)` is `true` (Move 1: no
   rule add/remove improves the prior) **and** `plateau_probability(state.learning_regime) > τ_sat`
   (Move 2: the residual decrements look like inferred noise). Both ⇒ `:gw_explore` is *admissible*.
   Per Q3-master this is a **soft prior into the explore EU, never a hard gate** — it raises explore's
   EU, it does not veto the alternatives.
3. **Candidate generation (`_threshold_candidates`, exploration.jl).** Owner: engine. From the
   observation buffer, the observed `:dist` values bracketing the mispredicted band are e.g.
   `{0.58, 0.61, 0.65}`; the residual-screened split-points are their midpoints `{0.595, 0.63}`. Finite,
   exhaustive over observed values (§3.1).
4. **Lookahead VOI (`_lookahead_voi`, exploration.jl).** Owner: engine. For candidate `0.63`: build
   `g′ = g` with `0.63` inserted into `:dist`'s grid (new id), `enumerate_programs(g′, depth)`,
   re-condition the resulting belief against the observation buffer via Tier-1 `condition`, and measure
   `Δℓ = ℓ̄(buffer | g) − ℓ̄(buffer | g′)` (mean predictive-log-loss reduction — the realised fit gain,
   in the *same nats* the residual plateau is measured in). `net_voi = Δℓ − compute_cost`.
5. **Cap-free budget (the sequential loop).** Owner: engine. Evaluate screened candidates in
   residual-screen-rank order while expected `net_voi > 0`; stop when the next candidate's expected gain
   falls below `compute_cost`. The winner (`0.63`, `Δℓ` largest, `net_voi > 0`) is applied; if none
   clears, return `g` unchanged (no-op).
6. **Apply + reset (host).** `add_programs_to_state!(state, g′, depth)` injects the new split's programs;
   `reset_learning_regime!(state)` starts the residual Measure afresh (Q1b — the alphabet changed, the
   plateau question resets). Owner: host.

Result: the ensemble can now represent the `dist > 0.63` boundary; the next steps' residual drops off
the floor; the regime re-enters `:improving`. Scope A (compression-only) provably could never reach
this grammar — it never adds a threshold.

## 5. Open design questions

### Q1 — Threshold residence, and the complexity-of-fineness tension (the structural crux)

Two coupled sub-questions; this is the deepest decision in the move.

**(a) Where do per-grammar refined thresholds live?** `THRESHOLDS` is a global `const`
(`enumeration.jl:9`) shared by every grammar; `Grammar` carries `feature_set` + `rules` only. For a
refined grammar to *be* a distinct grammar (new id, its own enumeration — the shape `perturb_grammar`
already uses), it must carry its own grid. **Recommendation: a `thresholds::Dict{Symbol, Vector{Float64}}`
field on `Grammar`, per-feature, defaulted to the global grid for every feature** via an inner
constructor, so all 34 existing `Grammar(` sites are bit-exact (§3). Per-feature (not one shared
vector) because refinement is local — the band that needs splitting is one feature's. Counter to weigh:
a flat shared `Vector{Float64}` is simpler and smaller wire-surface, but couples unrelated features
(refining `:dist` would needlessly branch `:energy`).

**(b) Does a finer grid cost grammar complexity?** *This is the question that may touch the master
plan.* The plan (§1) states a threshold is "complexity-invariant … purely a fit effect" — but that
claim is about **moving** a threshold (`:modify_threshold`, the deferred op: same split-count,
relocated). Move 3 **adds** split-points (5 → 6), which raises the grammar's *branching factor* — more
enumerable programs of the same per-program complexity. So "complexity-invariant" is true at the
*program* node-count level and **false** at the *grammar* level. The open question: do we
**(i)** charge `Grammar.complexity` for each added split-point (a principled Occam penalty on partition
fineness — the §1.3 truncation actively bounding the escape mass, so an over-fine grid is prior-penalised
and the refinement self-limits), or **(ii)** leave `Grammar.complexity` threshold-count-invariant and
rely on the saturation gate (*when*) + the lookahead net-VOI gate (*whether*, `Δℓ > compute_cost`) to
bound refinement (simpler; no new λ to pin; but no prior Occam on fineness — a finer grid dilutes the
normalised per-program prior only through enumeration count, not through `|G|`)? **Recommendation: (ii)
for this move, with (b) flagged.** The VOI gate + saturation already bound refinement to "only where the
residual demands it, only if the fit gain pays for the compute," so unbounded refinement is checked
without a complexity charge; and (i) would need a *new* per-threshold λ that §1.3 does not pin (§1.3
pins the *program* axis to `ln 2`, not a threshold-count axis), which risks new doctrine. But I want
pushback here: if leaving fineness un-penalised in the prior is judged to weaken the escape-mass
truncation, (i) is the principled alternative — and if (i) requires asserting a threshold-axis λ the
constitution does not sanction, **that is a stop-and-report**, not a quiet choice.

### Q2 — Candidate generation and the observation buffer (where "exhaustive generation" actually happens)

§3.1 says "the observed values are the candidate set," but two specifics are undetermined.

**(a) Split-points: observed values, or midpoints, residual-screened how?** A threshold at an observed
value `v` vs the midpoint between adjacent observed values are the two classic decision-stump
conventions; midpoints generalise better (no tie on `v` itself). **Recommendation: midpoints between
adjacent sorted observed feature-values, screened to the residual-bearing region** — only propose splits
inside the band where the current ensemble mispredicts (the master plan §3.2 "residual screens *where*").
This keeps the candidate set small *and* exhaustive over the values that can matter. Open: the screen
threshold (which observations count as "mispredicted"), and whether to screen at all vs rank-all (rank-all
is more faithful to "lookahead ranks all of them" but costs more lookahead).

**(b) The lookahead needs the data — a new observation buffer.** Re-conditioning a candidate grammar
(step 4 of §4) requires the accumulated `(features, outcome)` evidence; the host currently conditions
incrementally and **discards** it. Move 3 must retain an observation buffer and pass it into
`explore_grammar`. **Recommendation: a bounded host-side buffer** (the host already owns world state and
provides observations — the buffer is *data*, not belief, so it is host-side by the brain/body split and
does **not** violate state-is-measure). Open: buffer bound (a sliding window vs all-history-since-last-reset
— the saturation cadence already bounds how often explore runs, so all-history-since-reset may be fine),
and whether the buffer rides on `AgentState` (convenient, but `AgentState` is the belief bundle) or stays
a separate host structure (cleaner brain/body separation — recommended).

### Q3 — The VOI currency, and the cap-free budget (gate-3, the stall risk)

**(a) In what currency is the gain measured?** The thesis says "realised *value* gain." Two readings:
**(i)** belief-intrinsic — mean predictive-log-loss reduction `Δℓ` (marginal-likelihood / evidence gain
of the buffer under `g′` vs `g`), exact, domain-agnostic, and **measured in the exact same nats as the
residual plateau** (the saturation signal *is* predictive-log-loss decrements, so explore is valued in
the units saturation is detected in — a pleasing coherence); **(ii)** realised domain-utility gain, which
needs the domain preference threaded into `explore_grammar`, coupling it to the object-level utility and
to Move 5's currency question. **Recommendation: (i), predictive-log-loss nats.** It keeps
`explore_grammar` belief-aware but domain-agnostic (one entry serves grid_world and email_agent without
threading each one's utility), and it is the honest value-of-information for a *belief* (the metalevel is
improving the belief; the object level spends the improved belief). Counter: if a threshold improves
predictive fit but not *decision* quality (a split in a region utility doesn't care about), (i) over-values
it — (ii) would not. I lean (i) and name the (ii) coupling as Move 5's concern; argue if you want the
utility currency now.

**(b) The budget must be cap-free by construction (the §3.3 stall gate).** Prediction 3 (graceful
degradation, *no hard cap*) is the gate the master plan says *will* bite. **Recommendation: a sequential
greedy net-VOI stopping rule** — evaluate residual-screened candidates in rank order, applying the budget
as `compute_cost` *per lookahead*, and **stop when the next candidate's expected `net_voi ≤ 0`** — never
a fixed candidate count `m`. This is cap-free by construction (a cost-driven stop that scales smoothly
with `compute_cost`, exactly `decide_with_voi`'s shape), and a test asserts evaluations rise *monotonically*
as `compute_cost` falls, with no cliff (§7). **The hard line (master plan §3.3): if the screened-candidate
VOI distribution turns out to have a cliff that forces a fixed cap to stay tractable, Move 3 STALLS at this
doc — we do not ship a capped explorer dressed as graceful.** I believe the sequential rule holds because
the residual screen keeps the survivor set small, but this is the genuine risk, not a formality.

### Q4 — Scope: vertical slice vs mechanism + wiring (and the empirical figure)

Move 3 could be **(a)** the full vertical slice — engine mechanism + host wiring + integration + the
master plan's three empirical gates (§3: discovery, dominance, graceful-degradation curves on a
purpose-built task with baselines) — or **(b)** mechanism + the Move-2 host wiring + integration as a
saturation-gated meta-action + the **discovery** test only (gate 1: finds an off-grid optimum Scope A
provably cannot), with the **dominance** and **degradation** *figures* (gates 2–3) deferred to a dedicated
empirical step (the paper artifact — it needs a purpose-built task and the random / fixed-schedule
baselines stood up). **Recommendation: (b).** Mirrors Move 2's mechanism-first cut; keeps the code PR
reviewable; and the dominance/degradation figures are a measurement exercise that wants its own scope. The
**non-negotiable caveat (§3.3):** deferring the *empirical curve* must not defer the *mechanism's*
cap-freeness — the sequential-stopping rule (Q3b) and its monotonicity test are **in this move**, so the
stall gate is honoured at the mechanism level even though the figure is later. Argue (a) if you judge the
thesis unproven until the dominance figure exists — that is the legitimate counter (paper-as-gating-artifact).

## 6. Risk + mitigation

1. **`Grammar` field blast radius (34 sites + the skin wire).** Failure: a non-defaulted field breaks
   construction everywhere; a wire-invisible field makes refined grammars fail to round-trip. Blast: all
   of `src/program_space`, the hosts, `apps/skin/server.jl:644`, persistence fixtures. Mitigation:
   defaulted inner constructor (every existing site bit-exact, §3); **pre-emptive grep** `grep -rn
   'Grammar(' src/ apps/julia/ apps/skin/` with each hit dispositioned in the code PR; persistence
   round-trip fixture (a refined grammar serialises + reloads) per the commit-pinned-fixtures convention;
   skin smoke if Q1 makes the field wire-visible.
2. **Enumeration drift (the capture-before-refactor pin).** Failure: routing `THRESHOLDS` through the
   grammar field silently reorders or perturbs enumeration for default grammars. Blast: `test_program_space.jl`,
   every host benchmark, the skin enumeration verb. Mitigation: the `==` enumeration pin (§3), captured
   pre-change.
3. **Gate-3 cap temptation (the stall risk, §3.3).** Failure: the sequential rule proves intractable and
   a fixed cap sneaks in as "graceful." Mitigation: the cap-free sequential rule is the *only* sanctioned
   form (Q3b); the monotonicity test catches a cap (a capped explorer's evaluation count plateaus instead
   of rising with budget); **if the rule cannot hold, STALL** — do not cap.
4. **Observation-buffer unboundedness.** Failure: retaining all history blows memory on long runs. Blast:
   host memory, lookahead cost. Mitigation: the saturation cadence bounds explore frequency; buffer bound
   decided in Q2b; if a window is used, document the exactness lost (a windowed re-condition is an
   approximation — surface it per no-silent-approximations).
5. **Lookahead cost (re-enumerate + re-condition per candidate).** Failure: the lookahead is too
   expensive even post-screen → gate-3 fails. Blast: the thesis. Mitigation: the residual screen nominates
   few candidates; `compute_cost` is priced into every evaluation; if still too costly, this *is* the
   stall (risk 3), escalate not approximate.
6. **Benchmark trajectory drift (intended).** Failure mode is mis-reading the intended capability change
   as a regression. Mitigation: trajectories up to first-explore are pinned `==` (§3); post-explore drift
   is documented as the restored capability, not suppressed.

## 7. Verification cadence

Code-PR end-of-PR suite (Julia tests are not CI-gated — run locally; halt-the-line on any failure):

```
julia test/test_threshold_explore.jl      # discovery, degenerate no-op, cap-free monotonicity, enum pin
julia test/test_program_space.jl          # enumeration bit-exact for default grammars
julia test/test_saturation.jl             # Move-2 signal untouched
julia test/test_voc_gate.jl               # Move-1 compression untouched
julia test/test_grid_world.jl             # host wiring + integration (or the named host test)
julia test/test_email_agent.jl            # second host wiring
julia test/test_persistence.jl            # refined-grammar round-trip (if Grammar field is serialised)
```

Full `test/test_*.jl` suite green before the commit; lint self-test (`python
tools/credence-lint/credence_lint.py`) + `check apps/`.

**Skin smoke (`uv run python apps/skin/test_skin.py`): mandatory iff Q1 makes the threshold field
wire-visible or `explore_grammar` gets a wire verb; optional if `explore_grammar` stays host-internal.**
The template's blanket "Moves 3/4/6/7 run the skin smoke" is keyed to *wire changes*, not the move
number — Q1/Q4 decide whether this move changes the wire.

Test-tolerance classes (§3): enumeration pin `==` (strata-1); the discovery test is directional (the
off-grid optimum is found, its posterior mass *exceeds* any on-grid split — no magic threshold); the
cap-free test is monotone-directional (evaluations non-decreasing as `compute_cost` falls); any seeded
sampling in the lookahead asserts `==` at ~1e-12 under a fixed seed.

## 8. Ratification (2026-06-29, author)

All four questions ratified, two with corrections that change the *reasoning* (not the verdict). The §5
prose is retained as the questions-as-posed; this section is the decision of record.

### Q1(b) — take (ii), but the dichotomy's premise on (i) was false (the correction)

**(i) does not need a new threshold-axis λ, and is not unsanctioned by §1.3 — so neither branch
escalates.** A threshold is "complexity-1" only because `expr_complexity(GTExpr) = 1` **omits** the bits
to say *which* grid point it is. In a proper two-part code a threshold drawn from an `n`-point grid costs
`log₂(n)` bits in the program (which point) + the grid's definition in the grammar, so the honest
`|program|` **already rises with fineness**, priced by §1.3's *existing* `λ = ln 2`. Fineness lives on
the **program axis, not a new one**; (i) is a *correction to `|program|`*, not an amendment. `= 1` is an
approximation — exact and harmless on a fixed grid (the `log₂(n)` term is constant and cancels under
renormalisation), an **under-count** the moment Move 3 changes the grid size. Recorded in SPEC §1.3's
margin (this PR), because the next person to refine a grid will have the same worry and the answer is
"the prior already charges it."

**Decision: (ii) — leave `Grammar.complexity` threshold-count-invariant — for the principled reason, not
the doc's vague one.** The fineness-Occam that (i) would put in the *prior* is **already implemented by
the marginal likelihood** in the chosen VOI currency. Because `Δℓ` is measured on the **predictive**
(`ℓ = −log_predictive`, Q3a), a finer grid wins only if its *marginal* likelihood beats the coarser
grid's — and the marginal likelihood automatically penalises the extra split's parameter (Bayesian Occam
= the parameter integration). A split that merely fits noise does not improve the predictive, so
refinement **self-limits**. (ii) therefore does *not* weaken escape-mass truncation; it **relocates** the
truncation from the prior's description length to the likelihood's evidence — two routes to the same
Occam, and the predictive route is the one we are already on.

**The load-bearing dependency — Q1(b) ≡ Q3(a) are ONE decision.** This soundness holds *because* Q3a is
the predictive (marginal) log-loss and **not** a max-likelihood or point-estimate loss. A point-estimate
`Δℓ` would have no Occam and would chase infinite refinement — at which point (i) becomes **mandatory**.
**If anyone later "optimises" Q3a to a cheaper non-marginal loss, (ii) silently becomes unsound.** This
coupling is recorded here, in the SPEC §1.3 margin, and must be an `# executable-documentation`-style
cross-reference at both the `explore_grammar` VOI site and the `expr_complexity(GTExpr)` site in code.

### Q1(a) — ratified

Per-feature `thresholds::Dict{Symbol, Vector{Float64}}` on `Grammar`, defaulted to the global grid. It is
the right home **and** forward-compatible with (i): should the predictive Occam ever prove empirically
insufficient, the correction is to `expr_complexity` *per-feature* — exactly where the Dict lives.

### Q2(a) — ratified, and it confirms the thesis claim

Midpoints between adjacent observed values are the **complete, finite** candidate set — a threshold
**cannot matter except where it crosses an observation** — which is *precisely why thresholds are
EU-max-complete and need no heuristic proposer* (unlike features, Move 4). On the screen sub-question:
**do not introduce a screen cutoff (that is a magic number).** Use the residual as the **evaluation
order** — evaluate candidates in descending residual, stop at the net-VOI boundary — so the residual
*ranks* rather than *gates*, and there is no threshold to defend.

### Q2(b) — ratified

Host-side observation buffer (data is *body*, not *brain* — correct by the split).
**All-history-since-reset is the principled conditioning set** — it is the full evidence under the current
alphabet, and Move 2's `reset_learning_regime!` already clears the residual on grammar change. A bounded
buffer is a *sound* approximation **only if the bound is tied to where the VOI estimate stabilises**, not
an arbitrary window (a windowed re-condition is otherwise a silent approximation).

### Q3(a) — ratified (predictive-log-loss nats)

For the §5 reasons **plus** the Q1(b) coupling above. Note it correctly **side-steps the Q5 currency
gap**: Move 3 lives entirely in predictive nats, so the utility-coupling is deferred to Move 5 where it
belongs.

### Q3(b) — ratified, with a completeness guard (the Move-2 one-sidedness edge, again)

Cap-free sequential net-VOI stop + the monotonicity test + the stall line — all ratified; the value-driven
stop with `compute_cost` inside `net_voi` gives graceful degradation by construction, and the cliff-check
is the honest falsifier. **Completeness guard to add:** stopping at the *first* `net_voi ≤ 0` **in residual
order** assumes residual-order tracks VOI-order; if a low-residual candidate carries high VOI, the early
stop silently blocks a positive-EU explore — a soft cap. **Test that the ordered early-stop finds the same
positives as full evaluation** on the move's cases. If it misses any, the stop must become
**budget-bounded** (evaluate-until-`compute_cost`-budget-exhausted, take *all* positives) rather than
first-negative. Budget exhaustion is a **provable** defer (one-sided by construction); the residual-order
stop is a **heuristic** defer — only the former is guaranteed one-sided. The test decides which ships.

### Q4 — ratified (b)

The thesis is ratified, so the dominance and degradation **figures** are paper evidence, not preconditions
for the mechanism — defer them to the purpose-built empirical step where the baselines live. The
**cap-freeness mechanism and its monotonicity test stay in-move** (Q3b). Ship the discovery test now; the
figures travel with the paper (paper-as-gating-artifact).

### Net for the code PR

(ii) with the predictive-Occam justification recorded as **load-bearing on Q3a**; residual-as-order not
cutoff; predictive nats; the cap-free sequential stop **with the completeness guard**; scope (b). No
escalation. The Q1(b)≡Q3(a) coupling and the SPEC §1.3 margin note are part of the design-doc PR.

## 9. Code-time refinements (Move 3 code, ratified 2026-06-29)

Surfaced during implementation and ratified before the host integration. Each is forced by a constraint,
not chosen.

1. **`explore_grammar` is self-contained in `src/`, not a host replay closure — forced by Invariant 1.**
   The lookahead's marginal-log-loss accumulation (`Σ −log_predictive`) feeds the explore decision, so it
   cannot live in `apps/` (the spatial face forbids apps/ summing log-probs to influence behaviour). The
   replay therefore lives in `src/program_space/exploration.jl`, which required lifting a generic
   `program_space_observation_kernel` into `src/` — grid_world's + email_agent's `build_observation_kernel`
   (and email_agent's `build_step_kernel`) are the **same closure**, so the lift is a latent dedup. The
   host copies are left untouched (a NOTE'd DRY follow-up) to keep the live conditioning trajectories
   bit-stable.

2. **No live `belief` argument.** Signature: `explore_grammar(g, observations, max_depth; action_space,
   compute_cost)`. Belief-awareness flows through the buffer's per-observation `residual` (the screen
   *order*) + the **counterfactual replay** (mll under each candidate grammar) — the lookahead reconstructs
   beliefs per grammar, so the *current* belief object is not an input. `ExploreObservation(features,
   temporal_state, correct_actions, residual)` is the host-side buffer (Q2b).

3. **Full-eval-argmax, not a residual-order early-stop — Q3b's provable form.** Residual is the evaluation
   *order* (Q2a); the result is the global argmax over the finite candidate set → one-sided by
   construction. The completeness-guard test proves it (0.625 wins even as the lowest-residual candidate).
   `compute_cost` prices each lookahead (graceful degradation via the cost-gated boundary, which flips
   continuously at `Δℓ`); budget-*limited* early termination is a future hook (the residual order is its
   seam).

4. **Reset on alphabet expansion (explore) only — NOT on perturb/deepen/enumerate.** A precise reading of
   Q1b ("superseded alphabet"): compression is a *prior* effect (fit unchanged), and deepen/enumerate are
   *within-alphabet* — their effects surface in *subsequent* data residuals, which the regime tracks
   without reset. Only threshold/feature expansion supersedes the alphabet and resets the regime + clears
   the buffer. This makes the integration far less invasive (perturb/deepen/enumerate untouched) and is
   *more* correct than resetting on every grammar-id change.

5. **email_agent scope: the single-decision `run_agent` path (default) is fully wired; the episode path is
   safely gated off.** `:explore`'s meta-EU returns `-Inf` when `state.last_residual === nothing` — true on
   the episode path (which does not feed the regime), so `:explore` is never chosen there and
   `execute_meta_action!`'s empty-buffer default makes it a no-op regardless. The episode-path residual
   feed is a NOTE'd follow-up; the single-decision path is the canonical email integration, mirroring
   grid_world.

6. **The saturation gate in the host meta-EU.** `:explore` is admissible only when **both** halves hold
   (master plan §3.2): `compression_exhausted` (prior-side, lazy — computed only when the belief-side EU is
   already positive) **and** the residual-plateau, where `plateau_probability` is the **soft prior** (Q3 —
   it scales the EU continuously, never a hard threshold). The two are orthogonal (prior vs fit), so both
   are required.

7. **Skin: no wire change.** The `Grammar.thresholds` field is defaulted and not serialised outbound; the
   3-arg `Grammar(::Set,::Vector,::Int)` constructor is unchanged; `explore` is a host-orchestrated
   meta-action with no wire verb (a wire-parity follow-up if external apps ever drive exploration). The
   skin smoke is unaffected.
