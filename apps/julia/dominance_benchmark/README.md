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
`run.jl` (running it IS the check):

- CI of `eu_max − random` and `eu_max − best fixed` excludes 0 on **both** realised value
  (AUC of cumulative interaction energy) and sample efficiency (steps to the run's own
  half-total);
- **headline:** CI of `eu_max − never_explore` excludes 0 — both sides share the identical
  learned-returns escape ops, so this gap is exploration's isolated value, uncontaminated
  by the policy's one heuristic score;
- bracket `never_explore ≤ eu_max ≤ clairvoyant` on mean AUC (the left inequality is a
  hypothesis under test; the right is a sanity check that must always hold);
- minimax regret: the worst-seed AUC gap vs random and vs best fixed is ≥ 0;
- behaviour-verified inversions: concrete steps where `eu_max` grows grammar/feature and a
  baseline does not.

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
