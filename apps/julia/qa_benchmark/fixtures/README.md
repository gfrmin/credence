# QA benchmark fixtures — question bank + embeddings

Provenance for the offline embedding fixture (Paper 1, B2c). Committed so
the Bayesian benchmark runs **fully offline**: sentence-transformers is
needed only to *regenerate* these files, never to *run* the benchmark.

## Files

- `question_bank.json` — `(id, text, category)` for the 50-question bank,
  exported from `apps/julia/qa_benchmark/environment.jl::QUESTION_BANK` by
  `scripts/export_question_bank.jl`. Re-export if `QUESTION_BANK` changes.
- `question_embeddings.json` — per-question sentence embedding keyed by id.
  - model: `sentence-transformers/all-MiniLM-L6-v2`
  - sentence-transformers: 5.5.1
  - dim 384, n 50, **raw** (un-normalised)
  - generator: `scripts/paper1-embed-questions.py`
  - generated: 2026-05-31

## Why raw, not normalised

The classifier (`category_inference.jl`) standardises per dimension at fit
time — on the training fold only, so it is leak-free under LOO. Raw
embeddings are anisotropic (per-dimension scales vary widely), which
swamps the NormalGamma prior and collapses the classifier to the majority
class (raw LOO ≈ 0.30 ≈ chance); per-dim standardisation lifts it to
≈ 0.78. Committing raw embeddings keeps "what perception produces"
separate from "how the model conditions it".

## Regenerating (only if the question bank changes)

    julia scripts/export_question_bank.jl
    python3 -m venv --system-site-packages /tmp/embed-venv
    /tmp/embed-venv/bin/pip install sentence-transformers
    /tmp/embed-venv/bin/python scripts/paper1-embed-questions.py

Per the repo's commit-pinned-fixtures convention: do **not** regenerate to
fix a load bug — fix the loader. Regenerate only when the bank or the
chosen embedding model changes, and update this file's provenance block.
