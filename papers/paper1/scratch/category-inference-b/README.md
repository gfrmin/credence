# Phase B2a scratch — category inference option (b) prototypes

Throwaway evidence for §3 of `docs/paper1/move-2-design.md`. Will be
deleted after Paper 1 ships.

## Files

- `gaussian_nb_prototype.jl` — option **(b-NB)**: Gaussian Naive Bayes
  on embeddings with Dirichlet class prior. Runs end-to-end against the
  existing Credence DSL primitives. No new `LikelihoodFamily`, no new
  `ConjugatePrevision` pair, no new `update()` method.

  Run: `julia --project=. papers/paper1/scratch/category-inference-b/gaussian_nb_prototype.jl`

  On synthetic 5-cat × 8-dim well-separated embeddings (5 train + 5 test
  per category), achieves 25/25 accuracy and mean P(true) ≈ 0.998. The
  100% is a property of synthetic separation, not a forecast for the
  qa_benchmark — the prototype's job is to verify the inference plumbing
  (`condition` calls dispatch, posteriors normalise, Student-t
  predictives compose), not to evaluate model fit.

- `polya_gamma_gap.jl` — option **(b-PG)**: Pólya-Gamma multinomial
  logistic. *Intentionally non-running.* Spells out, in concrete
  signatures, what would have to be added to `src/` for PG to become
  expressible. See its file-level comment for the missing-pieces list.

## What this does NOT establish

- That (b-NB) is the right model for the qa_benchmark. The synthetic
  data is generated to be perfectly Gaussian Naive Bayes-shaped; real
  embeddings of qa_benchmark questions almost certainly are not.
- That accuracy on synthetic data predicts calibration on real data.
- That Gaussian Naive Bayes is what the master plan's OQ2(b) sub-bullet
  was actually aiming at. (The design doc takes a position on this; the
  prototype is one piece of the supporting evidence.)
