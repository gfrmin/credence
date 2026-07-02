# Belief-derived meta-action valuation — design doc

> Exploration-budget arc, post-coherent-injection (PR #187). Master plan:
> `docs/exploration-budget/master-plan.md` §3.2 (dominance, still undischarged after the
> coherent-injection gate re-run). Ratified premise (author, 2026-07-02): *"the engine should be
> coming up with the model that best fits the situation"* — at the meta level as well as the
> domain level. This move removes the last hand-written numbers at the selection seam and
> replaces them with posterior expectations. Design-doc-before-code; this doc is the review
> surface.

**STATUS: DRAFT — §5 open questions need answers before code.**

## 1. Purpose

After #187, the dominance gate fails on exactly two grounds, both instrumented
(`coherent-injection-design.md` §6), both places where a scoring number comes from the author
instead of from the agent's beliefs:

1. **Growth VOI is horizon-myopic.** `Δℓ` is window-total *past-fit* nats; a growth op's value
   accrues per *future* conditioning event over the remaining horizon. Brute-force eager growers
   (fixed_k5: final-window 0.899 vs eu_max 0.565) reap the multiple the score never counts.
2. **The escape tier is a heuristic that has now failed empirically.** `entropy − log 2` fires
   `enumerate_more` ~3×/step forever (posterior entropy asymptotes just above one bit; the op's
   realised effect is zombie resurrection at evidence-crushed weight). dominance-design §0 named
   this exact trigger; its ratified fallback was a separate track. This move supersedes the
   fallback with something stronger: **learn the returns**.

The unifying change: **meta-action value = a posterior expectation, computed by the engine.**
(a) Growth ops get the horizon completion — per-event gain × expected persistent future events —
from beliefs and declared task data the agent already carries. (b) Escape ops get a **learned
returns-to-growth model**: the realised yield of every executed op is an observation; a conjugate
belief over per-op yields is conditioned through the one learning mechanism; the score is its
posterior mean. No entropy proxy, no hand-priced `log 2` value claim, no separate track — escape
ops compete in the one argmax at honest, learned values.

The regress terminates where SPEC §1.3 says: the returns model's hypothesis space and the utility
remain declared data — one meta-level, no tower.

## 2. What retires, what stays

Retires:
- `entropy(belief) − escape_cost` as the escape score (the `entropy` accessor itself stays —
  it is a legitimate read).
- The saturation-ordering *eligibility gate* (escape scored `-Inf` unless the exact tier ≤ 0).
  It was a guard against heuristic unreliability; with learned returns the argmax needs no
  ordering rule. Breadth-before-depth becomes a *prior statement* (deepen's returns prior sits
  below enumerate's; see §5 Q4).
- `GW_ESCAPE_COST_DEFAULT` as a *value* claim. A declared compute price may persist as utility
  data (§5 Q6), but nothing hand-prices the value side.

Stays (bit-stable, out of scope):
- The transition (coherent injection, #187), all lookaheads (`exploration_voi`,
  `feature_discovery_voi`, `_grammar_marginal_log_loss`), `perturbation_voc`, the plateau
  machinery, `condition`/`expect` paths, the hard `voi_explore > 0` attribution gate on
  `add_feature` (it is an attribution-confound argument, not a valuation).
- The benchmark harness and gate assertions (one metrics amendment, §2c).

### 2a. Horizon-completed growth valuation

Current: `score(op) = plateau · net_value(Δℓ_window [+ prior_term], compute_cost)`.

Proposed:

    score(op) = plateau · (Δℓ_window / n_buf) · H  +  prior_term  −  compute_cost

- `Δℓ_window / n_buf` — the measured per-conditioning-event predictive gain (the lookahead's
  window total, normalised by the window it was measured on).
- `H` — expected remaining conditioning events: `(events so far / steps so far) × (max_steps −
  step)`. Declared task data + host bookkeeping counts (the episode length is the host's to
  declare; counting events is data, not probability arithmetic). Open-ended hosts: §5 Q2.
- `plateau` keeps its Move-2 semantics — P(the measured gain is a persistent plateau, not
  transient) — and does **not** double-count with `H`: plateau is *whether* the gain is real,
  `H` is *how long* it pays. `E[value] = P(real) · gain/event · E[events]`.
- `prior_term` (the `Δcomplexity · log 2` Occam charge for `add_feature`) stays **one-time** —
  it is a prior over grammars, paid once, never multiplied by the horizon.
- The arithmetic is canalised in a new engine accessor (`growth_value(Δℓ, n_buf, plateau, H;
  prior_term, compute_cost)` in the stdlib, the `net_value` pattern); hosts pass declared data
  and never multiply.

### 2b. Learned returns-to-growth (escape ops)

**The yield observable.** With coherent injection, every executed op has an exact, instantaneous,
engine-computable yield: the **posterior mass captured by the injected components** after their
window replay — `probability(belief, TagSet(injected tags))`, a Tier-1 read. It is zero when
`n_added = 0` (dedup no-op), near-zero when zombies re-enter at evidence-crushed weight (the
churn self-reports as worthless), and large exactly when the op admitted hypotheses the evidence
favours. Convert to nats as `−log(1 − mass)` (the log-evidence the op claimed). Alternatives and
their trade-offs: §5 Q1.

**The returns model.** Per op type (and context, §5 Q3), a conjugate belief over expected yield —
brain state, a Measure on AgentState (state-is-measure), conditioned on each executed op's
observed yield through `condition` with a declared kernel. Cold-start prior: weakly favourable
(ops are worth trying until evidence says otherwise — this replaces the old always-fire entropy
score with bounded initial optimism that *decays under evidence*, which the entropy score never
did).

**The score.** `score(op) = E_posterior[yield] − compute_cost`. After a handful of zero-yield
`enumerate_more` firings the posterior concentrates near zero and the op loses to the
`do_nothing` floor — the zombie churn kills itself in ~3 observations instead of never. If the
state changes (depth raised, grammar grown), the context shifts and the op earns fresh
consideration (§5 Q3).

**Constitutional shape.** One learning mechanism: yields are observations, the returns belief is
conditioned, never decayed or reset ad hoc. One decision mechanism: the score joins the same
argmax. This is Russell–Wefald metareasoning *as inference* — the "On metareasoning" clause made
mechanical — and it supersedes dominance-design §0's separate-track fallback (that fallback was
scoped for "entropy proves unreliable"; unreliability is now proven, and a learned value is
strictly more principled than a quarantined heuristic).

### 2c. Benchmark metrics amendment

`steps_to_own_half` is self-relative and rewards early collapse (never_explore is "fastest" to
half of a trajectory that ends at 0.244 final-window rate). Report final-window rate as a
co-primary alongside AUC in the gate tables; the efficiency gate assertion moves to a
fixed-reference variant (steps to a shared absolute level, e.g. half the *best* policy's total).
Metrics arithmetic is world-outcome data (no beliefs).

## 3. Behaviour preserved

This move deliberately changes selection-seam behaviour; equivalence pins apply to everything
beneath it:

- All Tier-1 paths, lookaheads, and the coherent-injection transition: bit-stable (existing
  suites — `test_coherent_injection.jl` ==/1e-12 pins, `test_threshold_explore.jl`,
  `test_feature_discovery.jl`, golden arithmetic lift — must pass untouched).
- `score_gw_meta_actions` with a fresh returns prior and `H` matching the old window-total
  (i.e. `H = n_buf`, plateau unchanged) reproduces the old *growth* scores exactly — asserted
  `==` in the new test as a nested-special-case pin.
- `test_grid_world_meta.jl` re-baselines for the escape tier (the entropy pins are replaced by
  returns-model pins: prior score, post-zero-yield collapse, context reset).
- New `test_growth_returns.jl`: conjugate update exactness (tightest-invariant α/β), yield
  observable correctness (mass of injected tags, zero on dedup no-op), score decay to below the
  floor after k zero yields, and the argmax integration (op stops firing).

## 4. Worked end-to-end example

Seed-0 step 165 of the gate run (the pathological step: old code fires `enumerate_more` on a
4e-5-nat entropy margin, 3×, forever):

1. Host calls `score_gw_meta_actions`. Growth tier: `exploration_voi = 0.0` (no candidate
   clears), `feature_discovery_voi = 0.0` → horizon completion multiplies zero — growth scores
   0, as before.
2. Escape tier: the returns belief for `enumerate_more` in context "hypothesis space unchanged
   since last firing" has been conditioned on ~40 realised yields, all `−log(1−mass) ≈ 0` (every
   firing dedup-no-oped or resurrected mass ≈ 1e-9 zombies). Posterior mean yield ≈ 0.001 nats.
   `score = 0.001 − compute_cost < 0`.
3. `default_eu_max_policy`: nothing exceeds the `do_nothing` floor → `:gw_do_nothing`, break.
   The meta loop costs one score evaluation instead of 3 executions + 3 prune/truncate passes.
4. Counterfactual: at step 141 (regime change), `add_feature`'s horizon-completed score is
   `plateau · (Δℓ/30) · H≈70·rate + log2-term` — the per-event gain multiplied by ~70 remaining
   events instead of a 30-obs window total: growth that the old score undervalued ~2–3× now
   clears earlier, which is precisely the fixed_k5 advantage the gate measured.

Ownership: yields and `H` are host-provided data; every expectation, update, and the mass read
are engine calls (`condition`, `expect`/`mean`, `probability(·, TagSet)`, `growth_value`).

## 5. Open design questions

1. **Yield observable: injected posterior mass vs realised next-window predictive delta.** Mass
   (`−log(1−P(TagSet))`) is instantaneous, exact, and free — but it measures *posterior
   attention*, not realised predictive improvement (a newcomer can claim mass and then predict
   poorly; conversely the mass it claims already reflects window fit under coherent injection).
   The predictive delta is the true currency but delayed, confounded by world-regime changes,
   and requires attributing a shared trajectory to one op. Proposed: mass, with the delta named
   as the finer fidelity (Move-5 fidelity-frontier pattern). Push back if attention-vs-value is
   a real wedge on this task.
2. **`H` for open-ended hosts.** grid_world's benchmark is episodic (declared `max_steps`);
   email_agent and the skin have no horizon. Options: (a) declared-horizon only, ship now,
   open-ended hosts keep window-total scoring (H = n_buf) until they declare one; (b) derive H
   from a learned persistence hazard (plateau-machinery extension). Proposed: (a), naming (b)
   as follow-up — hazard learning is its own move.
3. **Returns-model context granularity.** (op) alone under-fits (deepen-after-depth-change ≠
   deepen-at-stale-depth); (op × changed-since-last-fire bit) is the minimal honest context;
   (op × depth) finer still. Fine-parameterisation preference says finer, cold-start says
   coarser. Proposed: (op × changed-bit), four cells, conjugate per cell.
4. **Does breadth-before-depth survive as a prior?** The old tie order (enumerate before deepen)
   was load-bearing. With learned returns, encode it as deepen's lower prior-mean yield (or
   higher declared compute price)? Or keep the tie order in the argmax as-is (it only bites on
   exact ties, which learned scores make measure-zero)? Proposed: keep the argmax tie order,
   set equal priors — let evidence differentiate.
5. **Escape eligibility: fully free competition, or keep the exact-tier-≤-0 gate one more
   round?** Free competition is the clean end-state; the gate is a belt worn during the
   heuristic era. Risk of removing now: early-run (optimistic prior) escape ops could outbid a
   genuinely positive but small growth VOI. Proposed: remove the gate (the returns prior's
   optimism is bounded and decays; the old gate never protected against the actual failure
   mode anyway).
6. **Does a declared compute price stay?** The returns score is a value estimate; EU-max still
   wants `value − cost`. Proposed: keep `escape_cost`-style declared prices as utility data
   (renamed `op_compute_cost`, per-op, wire/kwarg-overridable, default the measured-elapsed
   pattern deferred) — prices are declarations, never learned value substitutes.

## 6. Risk + mitigation

- **Optimistic cold-start churn** (prior fires ops before evidence kills them): bounded by
  `max_meta_per_step` and by yield observations arriving on every firing; the first gate re-run
  measures it directly. If cold-start dominates 210-step runs, the prior tightens (a prior is
  data — tuning it against the *mechanism* is legal; tuning against the *gate outcome* is the
  overfitting the halt-the-line discipline exists to catch: one prior choice, stated in the doc,
  before the run).
- **Horizon completion over-fires growth late-window** (small Δℓ × large H): plateau
  multiplication and the one-time Occam charge remain; the gate's per-seed inversions will show
  any pathological late growth.
- **Scope creep into hazard learning**: fenced by §5 Q2 (a) — declared horizons only.

## 7. Verification

- `test_growth_returns.jl` (new) + re-baselined `test_grid_world_meta.jl` + untouched passes of
  the full local suite (55 files) and skin smoke.
- The dominance gate re-run, same 20 seeds — the falsifiable claims: (i) eu_max's meta-action
  count collapses from ~625 to O(growth ops + a few escape probes); (ii) the never_explore
  headline separates again (growth's isolated value is no longer masked by valuation noise);
  (iii) the AUC gaps vs tuned schedules close or invert. Halt-the-line on failure, as before.
