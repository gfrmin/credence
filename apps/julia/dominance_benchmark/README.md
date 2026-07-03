# Dominance benchmark

The exploration-budget arc's empirical capstone (dominance-design.md, Phases 4–5): does the
deployed EU-max meta-selection policy — every meta-action scored by its real `net_value` in
the one Δ log-evidence currency (grid_world Phase 3) — dominate non-Bayesian exploration
policies on a non-stationary task?

## What is compared

Five policies over the same seam (`run_agent(meta_policy=…)`), same seeds, same task:

| policy | selection rule | role |
|---|---|---|
| `eu_max` | `default_eu_max_policy` — argmax of the real scores, act-now floor at 0 | the agent |
| `random_p{.05,.15,.4}` | with rate p, a uniform non-idle op; score-blind; **best-tuned p reported** | retired random explorer |
| `fixed_k{5,10,25,50}` | one growth op every k steps, VOI-blind; **best-tuned k reported** | the hand-tuned schedule |
| `never_explore` | eu_max with growth ops vetoed; same learned-returns search ops | Scope-A floor / de-confounder |
| `clairvoyant` | eu_max + eager growth in ground-truth regime windows (masks nothing) | adaptation-timing ceiling |

Score-blind baselines (`random`, `fixed`) are declared as such (`ScoreBlind`), so the seam
skips computing the exact VOI lookaheads they would never read — behaviour-neutral, and it
keeps the score-blind cells' wall-clock bounded (an always-act random's growth executions
bloat the enumeration without bound; the swept-rate family subsumes it as p→1).

Task: `colour_typed → motion_typed → territorial` (changes at steps 70/140, 210 steps),
entity **respawn on** so encounters recur and beliefs keep conditioning within each regime.
Each regime moves the predictive feature (colour → speed → wall distance): staying good
requires re-discovery.

## What dominance means

Paired-seed percentile bootstrap (10 000 resamples) on per-seed gaps. The gate, asserted by
`run.jl` (running it IS the check), on the measure set fixed by dominance-design.md **§8**
(measure–utility alignment, ratified 2026-07-03 — the primary realised-value measure is the
**mean per-step energy rate** `ce[end]/n`, the uniform-weight statistic the agent's declared
utility maximises; AUC of the cumulative trajectory is front-loaded and reported-only):

- CI of `eu_max − random` and `eu_max − best fixed` excludes 0 on ALL of the **mean rate**
  (primary), the **final-window rate** (co-primary), and **shared-level sample efficiency**;
  best-tuned baselines are selected on the primary measure (anti-strawman follows the
  asserted measure);
- **headline:** CI of `eu_max − never_explore` excludes 0 on the mean rate and the
  final-window rate — both sides share the identical learned-returns escape ops, so this
  gap is exploration's isolated value; the final-regime exact sign test is reported beside
  it;
- bracket `never_explore ≤ eu_max ≤ clairvoyant` on the mean rate (the left inequality is a
  hypothesis under test; the right is a sanity check that must always hold);
- minimax regret: the worst-seed mean-rate gap vs random and vs best fixed is ≥ 0 (the
  winner's-curse pricing move's target, with the q10 gap reported beside it);
- behaviour-verified inversions: concrete steps where `eu_max` grows grammar/feature and a
  baseline does not.

`summary.md` also carries a reported-only panel (win rate + exact sign test, median gap,
q10 gap, AUC, final-regime rate) locating the claim without moving the gate.

## Running

```
julia apps/julia/dominance_benchmark/run.jl
```

Manually run and **out of the fast suite** (minutes of wall clock, like credence_router's
`test_live.py`). Writes `results/results.tsv` and `results/summary.md`; exits non-zero if
any gate assertion fails (halt-the-line: investigate, never patch forward).

Persistence is a plain TSV + markdown summary rather than qa_benchmark's SQLite schema —
that schema is question-shaped (per-question records); this harness stays stdlib-only so it
runs exactly like the test files, no project environment needed.
