# Coherent injection — the growth-op transition redesign

> Exploration-budget arc, post-dominance-gate (a halt-the-line response). Master plan:
> `docs/exploration-budget/master-plan.md` §3.2 (the dominance gate, NOT discharged by the first
> run — `apps/julia/dominance_benchmark/results/summary.md`). This move fixes the *model*, not the
> benchmark: the gate failed because the meta-action score and the meta-action transition were two
> different functions. Design-doc-with-code (author-directed fix; the conversation is the review).

**STATUS: implemented on this branch; gate re-run results in
`apps/julia/dominance_benchmark/results/` — see §6 for the outcome and the two named
follow-up decisions.**

## 0. The diagnosis (what the gate + priced-VOI experiment proved)

The first full gate run held the headline (`eu_max − never_explore = +6.06` AUC, CI > 0) but lost
to tuned sparse baselines on AUC (`random_p005 +7.75`, `fixed_k50 +10.58`). The priced-exploration
sweep (branch `experiment/priced-exploration`, `0f3265a`) then eliminated the obvious suspect:
pricing lookahead compute into scoring AND execution reduces growth-op firing 10.6 → 3.6 but
*worsens* both gaps monotonically. At matched growth budgets, VOI-selected growth was losing to
schedule-triggered growth by ~13 AUC. The unpriced factor was never compute — it was the
**transition**:

1. **Ignorant injection.** `add_programs_to_state!` appends new components as `Beta(1,1)` with bare
   complexity-prior log-weights. Incumbents carry evidence-earned posterior weights; newcomers
   enter as if the past never happened and take mass they have not paid for.
2. **Evidence destruction.** `execute_gw_meta_action!` runs `empty!(explore_buffer)` on both
   alphabet-expanding ops, destroying the very evidence that could have informed the newcomers.
3. **The score never promised any of this.** `exploration_voi`/`feature_discovery_voi` measure
   `Δℓ = mll(buffer|g) − mll(buffer|g′)` by replaying the buffer through `condition` — a lookahead
   whose simulated post-growth belief is *informed*. The executor delivered an *ignorant* one.

That is a score/transition divergence: `argmax EU` was maximising the value of a transition the
agent does not make. A4 forbids exactly this shape at the belief level ("if condition and some
second function can both modify beliefs, the implementation can disagree with itself"); here the
two disagreeing implementations were the lookahead's simulated growth and the executor's actual
growth. By definition `argmax EU` dominates *when the score is the EU of what will happen* — the
gate falsified the setup, not the theorem.

## 1. The fix: injection commutes with conditioning

**The property.** Bayes does not care when you thought of a hypothesis. The posterior over an
enlarged hypothesis space given evidence `o₁..oₙ` must not depend on whether the new hypotheses
were present from the start or injected at step n. The target injection weight for a newcomer is

    lw(newcomer) = complexity_logprior(grammar) + complexity_logprior(program)
                 + Σₜ pred_llₜ            (prequential, its own Beta replayed through condition)
                 (+ the same shared offset the incumbents carry)

`MixturePrevision` **normalises on every construction** (including inside `condition`), so the
cross-group constant that normalisation discards must be restored by two ledgers, both available
without new machinery:

- **De-normalisation ledger** (replay-side): the newcomers' prior normaliser
  `logsumexp(priors)` plus `Σₜ log_predictive(nm, kₜ, 1.0)` accumulated prequentially during the
  replay — undoes the replay's own normalisations.
- **Incumbent ledger** (window-side): `Σₜ obs.residual`. The buffer's `residual` field is the
  live trajectory's per-step surprise `−log_predictive` — *exactly* the normaliser the incumbents'
  weights absorbed at each conditioning step. A field that previously only ordered candidate
  evaluation becomes load-bearing for coherence (and its contract sharpens: the window must
  contain every observation the live belief conditioned on since window start).

With `new_lw .+ (denorm + ledger)`, a mixture built by *enumerate-union-then-condition* and one
built by *condition-then-inject* are equal over the shared window — exactly, up to float
summation order (different `logsumexp` groupings), pinned at `≤ 1e-12` on log-weights with Beta
states and tags exactly `==` in `test/test_coherent_injection.jl` §1. Host-side the equality is
additionally modulo `sync_prune!`/`sync_truncate!` mass drops (`≤ e⁻³⁰` relative per prune).

**Mechanism.** `add_programs_to_state!` gains a **required** `observations` keyword
(`Vector{ExploreObservation}`; `ExploreObservation` moves to `types.jl` — declared data, and
`agent_state.jl` loads before `exploration.jl`). Newcomers are assembled as a local
newcomers-only `MixturePrevision` and the window is replayed through Tier-1 `condition` with
`program_space_observation_kernel` — the same kernel, the same learning mechanism the live loop
uses (Invariant 1: no second conditioning path; the ledgers read Tier-1 accessors —
`log_predictive`, `logsumexp` — never reimplement them; the same accumulation shape
`_grammar_marginal_log_loss` already canalises in `src/`). Tags are local during replay, then
re-tagged to global positions on append (the established `sync_prune!` re-tag pattern).

**Why required, not defaulted.** A default-empty kwarg is the misspecification re-armed: the next
call site silently injects ignorant components again. Making the caller state the evidence window
turns "forgot the evidence" from a silent default into an explicit, greppable lie
(`observations = ExploreObservation[]` where a buffer exists). The signature is the enforcement —
the add-surface-not-default pattern. At t=0 (initial enumeration) the honest window is genuinely
empty.

**Buffer retention (the Q2b amendment).** `empty!(explore_buffer)` is removed from both
alphabet-expanding ops. Move 2 Q1b's staleness argument was over-applied: what an alphabet change
makes stale is the *learning-regime residual history* (`reset_learning_regime!` — which stays,
unchanged, on both ops) and, mildly, the `residual` ordering field (harmless: it orders candidate
evaluation, never gates). The raw `(features, correct_actions)` records are world data —
alphabet-independent — and are exactly what coherent injection needs. `explore_window` remains the
sole aging mechanism (it already superseded the clear for world-regime changes; the gate memo
flagged the clear as "possibly redundant destruction" — it was worse than redundant).

**All five injection sites.** The same dilution applies to `:gw_enumerate_more`,
`:gw_perturb_grammar`, and `:gw_deepen` (they never cleared the buffer, but injected ignorant
components all the same). Every `add_programs_to_state!` call in the grid_world and email_agent
hosts passes its buffer; the skin's two call sites pass an explicitly empty window (the wire has
no buffer concept yet — an honest declaration, and the pre-change behaviour).

## 2. What does NOT change

- **The scores.** `exploration_voi`, `feature_discovery_voi`, `perturbation_voc`, the plateau soft
  gate, the escape tier, the argmax policy, `GW_ESCAPE_COST_DEFAULT` — untouched. The fix makes the
  transition deliver what the score already prices; it does not re-price.
- **`reset_learning_regime!` on alphabet expansion.** The plateau meta-belief legitimately restarts
  after a caused change-point (Q1b's actual content).
- **The VOI memo cache.** Lookaheads stay pure in `(grammar, buffer, depth)`; the epoch key
  discipline (bump on growth execution and window trims) is unchanged and remains sound.
- **Dedup semantics.** `(grammar_id, expr)` keying as before.

## 3. Residual honesty (what coherence does not buy)

- **Incumbent head-start.** Incumbents carry evidence older than the window; newcomers can only be
  conditioned on what was retained. The asymmetry is honest — it favours incumbents by exactly the
  evidence newcomers never saw — and shrinks as `explore_window` grows.
- **The score is still a window-evidence contrast**, not a future-reward oracle. `Δℓ` compares
  fresh replays of `g` vs `g′`; the live mixture also carries pre-window history. The catastrophic
  divergence (promise informed, deliver ignorant, destroy evidence) is gone; the remaining gap is
  the ordinary gap between model evidence and realised future utility, which the plateau
  marginalisation already prices in expectation.
- **Repeated within-window growth.** With the buffer retained, a second refinement can clear
  immediately after the first (the same window can justify two thresholds). That is coherent
  EU-max, not a bug; `max_meta_per_step` bounds the per-step rate and the plateau reset soft-gates
  the sequence. Watched empirically in the gate re-run.

## 4. Test plan

- **Commutation (`test_coherent_injection.jl` §1, the constitutional pin):** union-from-start vs
  inject-at-n over the same window ⇒ log-weights and `weights()` equal `≤ 1e-12` (float
  summation-order is the only slack — the deterministic-arithmetic tolerance, not 1e-6), Beta
  states and tags exactly `==`.
- **Ignorance regression (§2):** injecting with a non-empty window ≠ injecting with
  `ExploreObservation[]` (the old behaviour is representable only by explicitly declaring an empty
  window) — pins that the evidence actually flows.
- **Host seam (§3):** after `:gw_explore`/`:gw_add_feature` execute, the buffer is intact
  (length preserved) and the learning regime is reset — the split Q1b actually asked for.
- **Suite:** full `test/test_*.jl` locally (not CI-gated); the touched suites
  (`test_program_space`, `test_perturb_consumption`, `test_grid_world*`, `test_email_agent`,
  `test_threshold_explore`, `test_feature_discovery`) updated for the required kwarg.
- **The gate re-run** (`apps/julia/dominance_benchmark/run.jl`, same 20 seeds × configs): the
  falsifiable claim is that eu_max's AUC deficits vs `random_p005` (−7.75) and `fixed_k50`
  (−10.58) close or invert once the transition is coherent. Halt-the-line again if they do not.

## 5. Open design questions

1. **Should the buffer window used for injection be the full retained buffer or the since-last-
   grammar-change suffix?** Shipped: full retained buffer. The newcomers' kernels evaluate current
   features on old observations identically to how the union-from-start would have — commutation
   holds either way — but a *world*-regime change inside the window conditions newcomers on
   mixed-regime evidence, same as it does incumbents. Symmetric, therefore chosen; flagged because
   the window semantics interact with `explore_window` tuning.
2. **Skin parity.** The wire has no explore-buffer verb, so skin injections declare an empty
   window. If a wire consumer ever drives program-space growth seriously, the buffer becomes wire
   state (opaque server-side, like Measures) — its own design pass; noted, not built.
3. **Does `email_agent` want the same benchmark treatment?** It gets coherent injection (the
   engine change is host-agnostic) and passes its buffer at all four sites, but its episode loop
   (`condition_step!`) conditions without buffer records, so its ledger is PARTIAL: newcomers are
   informed by the witnessed subset only and retain a bounded over-weight from unwitnessed
   normalisations — strictly better than the old ignorant injection, exactly coherent only when
   the buffer witnesses every condition (as grid_world's now does). Closing it means buffering
   episode steps; deferred with this note (no dominance gate on email_agent).

## 6. Gate re-run outcome (2026-07-02, 20 seeds)

**The transition fix is real and large — and the gate still fails, on new grounds.** Every
policy improved (the dilution was engine-wide: `enumerate_more` resurrects `sync_prune!`-dropped
components, so even never_explore's ~300 escape ops were injecting ignorantly — its mean AUC
moved 34.9 → 46.7). eu_max moved 40.96 → 46.49, and the statistically significant deficits
became noise: vs random_p005 −7.75 [CI < 0] → −3.47 [−9.24, +2.78]; vs best-fixed −10.58
[CI < 0] → −4.74 [−9.41, +0.17]. Efficiency vs random_p005 flipped decisively positive
(+56.1 [24.0, 89.7]). But the headline collapsed (eu_max − never_explore = −0.2 [−3.53, +2.99])
and tuned schedules stay nominally ahead — §3.2 remains undischarged.

Per-step score instrumentation (seed 0) localises the residual mis-specifications, both in the
valuation, neither in the transition:

1. **The escape tier is mis-priced, twice over.** `enumerate_more` fires ~3/step for entire
   runs: its `entropy − log 2` score claims up to 3.5 nats while the op's only actual effect is
   zombie resurrection (dedup blocks everything not pruned), and late-run it keeps firing on
   ~4e-5-nat scores because posterior entropy asymptotes just above one bit (irreducible
   near-tie uncertainty no enumeration can reduce). This is precisely the dominance-design §0
   trigger — "if entropy proves unreliable in the benchmark" — whose ratified fallback is
   moving the escape ops to a separate, honestly-labelled track outside the unified argmax.
2. **Growth VOI is horizon-myopic.** `add_feature` behaves correctly (fires ~5×/seed, including
   after the step-140 regime change; `explore` is worth 0.0 on this task throughout), but its
   score is window-total past-fit nats (`Δℓ − log 2`), while a feature's realised value accrues
   per future step over the remaining horizon. With dilution eliminated, growth is nearly free,
   and brute-force eager growers reap the horizon value the score never counts (fixed_k5:
   42 ops, final-window rate 0.899 vs eu_max's 0.565). The principled fix is a horizon factor —
   value ≈ per-step predictive gain × expected persistence (the plateau machinery already
   carries persistence beliefs) — but that rewrites the ratified Move 3/4/5 valuation semantics
   and needs ratification before code.

Also observed: the self-relative efficiency metric (steps-to-own-half) rewards early collapse
(never_explore is "fastest" to half of a trajectory that ends at 0.244 final-window rate);
worth a metrics note before the next gate.
