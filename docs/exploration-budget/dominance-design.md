# Dominance benchmark + real-VOI selection — design doc

> Exploration-budget arc, post-Move-5 (a deliberate re-open). Master plan:
> `docs/exploration-budget/master-plan.md` §3.2 (dominance, OQ-5.3, deferred) + §4 (one currency,
> two fidelities). This move deploys the real single-currency `argmax` at the grid_world selection
> seam — the end-state Move 5 *named* but explicitly did not build — and then proves the deployed
> policy dominates random and fixed-schedule exploration. Design-doc-before-code; per-phase commits
> green + bisectable; stop-and-report at each phase boundary. Julia tests are not CI-gated: run the
> full suite locally.

## 0. Ratification

Ratified in conversation; §5 retains the reasoning of record.

- **Re-open is completion, not contradiction (RATIFIED).** Deploying real VOI at selection replaces
  *proxy-for-exact* with *actual-exact* on the exploration ops only. Compression stays a prior-only
  surrogate, never re-conditioned at selection; the cheap saturation screen still gates when the
  expensive exact lookahead runs. This finishes Move 5's cascade; it does not flatten it. Move 5 §6
  is the citation for why flattening would be the error.
- **Escape-mass ops (`:gw_enumerate_more` / `:gw_deepen`) — the load-bearing decision (RATIFIED).**
  Their value is un-entertained-hypothesis value, myopic-unreachable *by construction*, so there is
  no exact VOI. Score them by the **posterior entropy of the program-mixture** (nats), named as a
  **heuristic proxy — not a bound** — constitutional under resource-rationality (a principled
  heuristic for a quantity one cannot afford to measure exactly *is* the EU-max play once evaluation
  cost enters the utility). Saturation-ordered strictly below any positive exact exploration VOI so
  it is the fallback, never a peer. **Fallback:** if entropy proves unreliable in the benchmark, move
  these two ops to a separate, honestly-labelled track *outside* the unified `argmax` rather than
  manufacture a bound. Never a magic scalar.
- **`:gw_do_nothing = 0.0` (RATIFIED).** The act-now reference, replacing the current `−Inf`. Any op
  with `net_value ≤ 0` must lose to it. Interlocks with the escape-mass score: `entropy − compute_cost`
  measured against `0.0` searches only when the belief is uncertain enough to pay, and stops when it
  has concentrated.
- **Dominance attribution (RATIFIED).** Promote `never_explore` from bracket to **headline result**:
  the gap `eu_max − never_explore` isolates the value of exploration with the escape-mass heuristic
  held constant on both sides — the one comparison the softest score in the policy cannot contaminate.
  Report it beside the vs-random and vs-fixed-schedule gates, paired with behaviour-verified
  inversions.
- **Behaviour shift is intended (RATIFIED).** grid_world meta-action sequences move (proxy →
  principled). `test_grid_world_meta.jl` expectations are *updated to the real VOI values*, not
  preserved; the tightest-invariant form asserts the numbers, not "no error".

## 1. Purpose

Move 5 proved the metalevel is **one currency — Δ log-joint — at two fidelities**, and that the
single currency is already in the *engine* (`explore_features` sums its two terms in one `net_value`).
It deliberately shipped zero behaviour change, leaving the deployed grid_world selection ranking
meta-actions by hand-tuned proxy EUs (`compute_gw_meta_eu`, host.jl:172-228). This move deploys the
genuine combined single-currency `argmax` at that seam — each meta-action scored by real `net_value`
at its correct fidelity, every proxy constant retired, `do_nothing = 0.0` — and then discharges the
one outstanding empirical gate the thesis named and Move 5 deferred (§3.2 dominance, OQ-5.3) as a
runnable, checked-in, self-asserting in-repo benchmark. What unblocks: the arc's capstone lands, and
the claim "the principled explorer dominates" stops resting on assertion and starts resting on
bracketed, CI-bounded evidence.

## 2. Files touched

**Engine — expose the scalar VOI/VOC (behaviour-preserving extraction).**
- `src/program_space/exploration.jl:273` (`explore_grammar`) — **modify.** Extract
  `_best_threshold_refinement(g, obs, d; …) :: Tuple{Grammar,Float64}` returning `(winning grammar,
  its net VOI)`; `explore_grammar` becomes `first(_best_threshold_refinement(…))` (bit-exact wrapper);
  add `exploration_voi(…) = last(_best_threshold_refinement(…))`.
- `src/program_space/exploration.jl:351` (`explore_features`) — **modify.** Same shape:
  `_best_feature_addition` → `explore_features` / `feature_discovery_voi`. VOI is
  `net_value(Δℓ + complexity_logprior(Δc; λ=log2), cc)` (the exact-general instance, both terms).
- `src/program_space/perturbation.jl:418,446` (`perturb_grammar`) — **modify.** Extract
  `perturbation_voc(…) :: Float64` = best `net_voc` over compression candidates (prior-only surrogate);
  `perturb_grammar` unchanged in return.
- `src/Credence.jl` — **modify.** Export `exploration_voi`, `feature_discovery_voi`,
  `perturbation_voc`.

**Host — the selection `argmax` goes real (the capstone landing).**
- `apps/julia/grid_world/host.jl:172-228` (`compute_gw_meta_eu`) — **modify.** Replace each proxy
  branch with the real-`net_value` score from the cascade table (§4); score `enumerate`/`deepen` by
  program-mixture entropy per §0; delete the retired constants (`GW_EXPLORE_BASE`, `GW_EXPLORE_COST`,
  `GW_EXPLORE_VOI_FLOOR`, `GW_ADD_FEATURE_BASE/COST/VOI_FLOOR`, `GW_ENUMERATE_COST`, `GW_PERTURB_COST`,
  `GW_DEEPEN_COST`, and the inline `/5.0`, `·0.5/·0.6/·0.4`, `0.1`).
- `apps/julia/grid_world/host.jl:180` — **modify.** `:gw_do_nothing → 0.0`.
- `apps/julia/grid_world/host.jl:440-446` (selection seam) — **modify.** (i) prefer `do_nothing` over
  any `≤ 0` op; (ii) add one keyword `meta_policy::Function = default_eu_max_policy`, and route the
  chosen op through `chosen = meta_policy(scored)` where `scored::Dict{Symbol,Float64}`.
  `default_eu_max_policy(scored) = argmax(scored)` reproduces the plain argmax bit-exactly.
- `test/test_grid_world_meta.jl` — **modify.** Assertions updated to the principled expectations
  (assert the real VOI values).

**Benchmark harness (new), modelled on `apps/julia/qa_benchmark/`.**
- `apps/julia/dominance_benchmark/policies.jl` — **new.** Five policies over `scored::Dict`:
  - `eu_max` — `argmax(scored)` (the agent; no pragma).
  - `random` — uniform over non-`do_nothing` ops, `MersenneTwister(seed+offset)` (*retired random
    explorer*).
  - `fixed_schedule` — explore every *k* steps regardless of VOI; **swept over *k*, best-tuned
    reported**.
  - `never_explore` — Scope-A floor: object-level + `enumerate`/`deepen` only, never grammar/feature
    growth (the *same* entropy-scored escape-mass ops as `eu_max`).
  - `clairvoyant` — ceiling: grows the task's known-predictive feature/threshold at the first step it
    helps (discovery-task ground truth + regime schedule).
  Each research baseline carries the sanctioned pragma
  (`# credence-lint: allow — precedent:baseline-comparison — <name>: <why non-Bayesian>`).
- `apps/julia/dominance_benchmark/host.jl` — **new.** Loop `policies × 20 seeds` via
  `run_agent(rng_seed=seed, meta_policy=…)` on a **non-stationary** task (the file's existing dual
  `run_agent` regime-change staging). Reuse `qa_benchmark/metrics.jl`'s `SeedResult` / `summary_table`
  / SQLite `save_results`; persist to `apps/julia/dominance_benchmark/results/`.
- `apps/julia/dominance_benchmark/metrics.jl` — **new (thin).** Per (policy, seed): energy trajectory
  → **AUC + final-window mean** (realised value, no magic threshold); **steps-to-own-asymptote**
  (sample-efficiency, relative); `n_meta_actions`; discovered grammar.
- `apps/julia/dominance_benchmark/run.jl` — **new.** Entry point + the gate (§7). Manually-run
  (heavy), documented as out-of-fast-suite.
- `apps/julia/dominance_benchmark/README.md`, `.../results/` — **new.**

**Update at close.** `docs/exploration-budget/master-plan.md` status (arc re-opened past Move 5; §3.2
discharged) + memory.

## 3. Behaviour preserved (and the one intended change)

Two behaviour classes, held apart:

- **Phase 2 (engine extraction) — behaviour-preserving.** Capture-before-refactor: pin the exact
  grammars returned by `explore_grammar` / `explore_features` / `perturb_grammar` on the existing
  fixtures pre-refactor; the wrappers reproduce them with `==`. Strata-1 unit equivalence,
  `isapprox(atol=1e-14)` on any float path; the returned grammar is `==`. `test_threshold_explore.jl`,
  `test_feature_discovery.jl`, `test_program_space.jl` stay green unchanged; scalar-accessor asserts
  added (`exploration_voi ≡ the Δℓ−cc the doc names`).
- **Phase 3 (host upgrade) — the deliberate change.** The selection ranks meta-actions differently
  (proxy → real). This is *not* `==`; it is the move's point. Capture current `run_agent` outputs,
  then re-baseline `test_grid_world_meta.jl` to the real VOI values (tightest-invariant: assert the
  numbers). The one equivalence that *does* hold: `default_eu_max_policy(scored) = argmax(scored)`
  reproduces Phase 3's argmax bit-exactly, so Phase 4's harness is `==` to Phase 3 for the `eu_max`
  policy — the seam parameterisation adds no behaviour.

No `net_evidence_voc` merge (Move 5 rejected it; it would blur the prior/belief seam). The
strata-2/3 tolerance ladder applies only to Phase 2's extraction; Phase 3 has no behaviour to preserve
by construction.

## 4. Worked end-to-end example — one tick through the real-VOI `argmax`

Grid `g` with features `{food, enemy}`, one rule `R`, residual buffer `obs` on which the posterior
mispredicts; belief carried as a program mixture with `log_weights`. One tick, `scored::Dict`:

| Op | Score (post-upgrade) | Fidelity | Owner |
|---|---|---|---|
| `:gw_explore` | `net_value(Δℓ, cc)` = `exploration_voi(g, obs, d)` | exact re-conditioned lookahead | `exploration.jl` |
| `:gw_add_feature` | `net_value(Δℓ + complexity_logprior(Δc; λ=log2), cc)` = `feature_discovery_voi(…)` | exact re-conditioned lookahead | `exploration.jl` |
| `:gw_perturb_grammar` | `perturbation_voc(g, freq_table)` = best `net_voc` (prior-only) | prior-only surrogate | `perturbation.jl` |
| `:gw_enumerate_more`, `:gw_deepen` | `H(posterior) − compute_cost`, `H = −Σ wᵢ log wᵢ` from `log_weights` | escape-mass **heuristic** | host |
| `:gw_do_nothing` | `0.0` | act-now reference | host |

Trace. The host reads `plateau_probability(state.learning_regime)` (Move 2, soft). It computes the
escape-mass score directly from the mixture's normalized `log_weights` — O(n), no re-conditioning. If
`plateau` is high and no cheap surrogate-positive compression remains, it pays for the exact
exploration lookahead (`exploration_voi`, `feature_discovery_voi`), which re-enumerates and
re-conditions on `obs` and returns the real `Δℓ`; `:gw_add_feature` additionally charges
`complexity_logprior(+1; λ=log2) = −log2` explicitly (the mll cancels the grammar prior — Finding 1,
compression-removal). `feature_discovery_voi` stays gated by `threshold_exhausted` (hard, attribution)
and `plateau` (soft). The seam takes `argmax(scored)`; because `do_nothing = 0.0`, an all-negative
`scored` selects *do nothing* rather than the least-bad op. Every score is nats or the `0.0`
reference; there is one currency and one argmax, and the only heuristic (entropy) is named as such and
ranks below any positive exact VOI.

## 5. Open design questions (ratified — reasoning of record)

1. **The escape-mass score — the load-bearing question.** `:gw_enumerate_more` / `:gw_deepen` enlarge
   the entertained hypothesis set; their worth is the worth of the *un-entertained* tail, which cannot
   be exactly valued without enumerating it. Options: (a) an upper bound = complexity-prior tail mass
   × best-possible fit — *rejected*: far too slack, it assumes a perfect program hiding in the tail
   and so always says "search more"; (b) **posterior program-mixture entropy** — tighter, in nats,
   read from the live belief, but a *heuristic proxy* (it measures uncertainty among *entertained*
   programs, correlated with — not equal to — the tail's value); (c) a separate track outside the
   argmax. **Ratified: (b), named as a heuristic**, saturation-ordered strictly below any positive
   exact exploration VOI, with **(c) as the standing fallback** if the benchmark shows the entropy→
   search direction unreliable. Constitutional basis: resource-rationality — a principled heuristic
   for an unaffordable-exact quantity is the EU-max strategy once evaluation cost is in the utility.
   Entropy is confirmed cheap (the mixture carries `log_weights`). Named refinement, deferred: entropy
   does not distinguish breadth (`enumerate`) from depth (`deepen`); split only if the benchmark shows
   they need it.
2. **`do_nothing = 0.0`, not `−Inf`.** With real `net_value`, `0.0` is the principled act-now
   reference and any `net_value ≤ 0` op must lose to it. `−Inf` forced a meta-action even into a
   settled belief. **Ratified.** It interlocks with (1): `entropy − cc` vs `0.0` stops search when the
   belief concentrates.
3. **Cascade-consistency (the re-open risk).** Deploying real VOI must not be read as licence to
   flatten. **Ratified argument:** each op scored at its own fidelity (exact lookahead / prior-only
   surrogate / entropy heuristic); the cheap saturation screen still gates the expensive exact eval;
   real VOI replaces proxy-for-exact on the *exploration* ops only and never touches compression's
   fidelity. A flat one-shot argmax is not more unified — it is the cascade with its EU-optimal
   evaluation-cost ordering deleted (Move 5 §4/§6).
4. **Behaviour shift is intended.** `test_grid_world_meta.jl` is re-baselined to the real VOI values,
   not preserved. **Ratified.**

## 6. Risk + mitigation

- **Over-claiming the unification / flattening (headline risk, one level down from Move 5).** Failure:
  reading "real VOI at selection" as "flatten to one exact argmax," re-conditioning compression and
  destroying the cheap screen. Mitigation: §5.3's cascade argument; the code re-conditions only the
  exploration ops; `perturbation_voc` stays prior-only. No pragma site — no code asserts a flattening.
- **The escape-mass heuristic contaminating the dominance claim.** Failure: a reviewer cannot tell
  whether `eu_max` beats `random` because it *explores* well or because it *searches* (entropy) well —
  entropy is the softest number in the policy. Mitigation (the de-confounder): `never_explore` is
  `eu_max` with grammar/feature growth off but the *same* entropy-scored escape-mass ops, so
  `eu_max − never_explore` isolates exploration's value with the heuristic held constant on both
  sides — the one gap entropy cannot contaminate. Report it as a headline number; pair with
  behaviour-verified inversions (concrete steps where `eu_max` grows a feature and a baseline does
  not). If entropy proves unreliable, §5.1(c): separate track.
- **Fixed-schedule strawman.** Failure: beating an untuned schedule proves nothing. Mitigation: sweep
  *k*, report the baseline's best-tuned configuration.
- **`never_explore ≤ eu_max` misread as a sanity assertion.** It is a *hypothesis under test*: on a
  task that genuinely rewards exploration it should hold, but if it fails, interrogate whether the
  non-stationarity is strong enough to reward exploration *before* concluding the policy is broken.
  The regime-shift magnitude is load-bearing for the gate meaning anything — report and diagnose the
  task first. `eu_max ≤ clairvoyant` is a true sanity check and must always hold.
- **Benchmark runtime.** 5 policies × 20 seeds × full `run_agent` + 10 000-resample bootstrap may run
  minutes — acceptable for a manually-run gate. If a run hangs or exact inference is too slow, STOP
  and report — no silent approximation.

## 7. Verification cadence

- **Phase 2:** `julia test/test_feature_discovery.jl && julia test/test_threshold_explore.jl &&
  julia test/test_program_space.jl` — green; returned grammars `==` the pre-refactor capture;
  scalar-accessor asserts pass.
- **Phase 3:** full `test/test_*.jl` suite green; `test_grid_world_meta.jl` asserts the real VOI
  values; `grep` shows the retired constants gone; `python tools/credence-lint/credence_lint.py test`
  + `check apps/` clean; `uv run python apps/skin/test_skin.py` (wire smoke — the engine refactor must
  not disturb the skin surface).
- **Phase 4-5 — the gate (`run.jl` asserts, so running it *is* the check):**
  - **Paired-seed bootstrap CIs** on the per-seed gap `eu_max − baseline` (resample seed indices,
    10 000 resamples, 2.5/97.5 percentile — reuse `papers/paper1/scripts/paper1-bootstrap.jl`).
  - **Gate:** CI on `eu_max − random` and `eu_max − best-tuned fixed_schedule` excludes 0 on **both**
    AUC and sample-efficiency; `eu_max − never_explore` reported and its CI excludes 0 (exploration's
    isolated value); `never_explore ≤ eu_max ≤ clairvoyant`; **minimax-regret** — worst-seed gap ≥ 0
    (dominance survives the worst seed, not just the mean).
  - **Behaviour-verified inversions** extracted and reported (decision-level grounding, not just
    aggregates).
  - Writes a results table + `README.md` (what is compared, what dominance means). Documented as
    out-of-fast-suite, like `test_live.py`.
- **Halt-the-line:** any test failure, any non-`==` on Phase 2's captured grammars, or any gate
  assertion failing on a task whose non-stationarity has been confirmed adequate, is a halt to
  investigate — not to patch forward.
