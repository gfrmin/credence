#!/usr/bin/env python3
"""scripts/paper1-embed-questions.py — offline embedding of the QA question bank.

Reads ``apps/julia/qa_benchmark/fixtures/question_bank.json`` (produced by
``scripts/export_question_bank.jl``), embeds each question with
sentence-transformers ``all-MiniLM-L6-v2`` (384-dim), and writes
``apps/julia/qa_benchmark/fixtures/question_embeddings.json``.

Run ONCE. The committed embedding fixture makes the Bayesian benchmark
fully offline — sentence-transformers is needed only to (re)generate the
fixture, never to run the benchmark.

Requires ``sentence-transformers`` (+ torch). Reusing a system torch:
    python3 -m venv --system-site-packages /tmp/embed-venv
    /tmp/embed-venv/bin/pip install sentence-transformers
    /tmp/embed-venv/bin/python scripts/paper1-embed-questions.py

Raw (un-normalised) embeddings are written deliberately: the Gaussian
Naive Bayes classifier models each dimension as an independent Gaussian,
so L2-normalisation (which couples dimensions onto the unit sphere) is
avoided.
"""
import json
import os

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "apps", "julia", "qa_benchmark", "fixtures")


def main() -> None:
    with open(os.path.join(FIX, "question_bank.json")) as f:
        bank = json.load(f)

    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    texts = [q["text"] for q in bank]
    embs = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
    dim = int(embs.shape[1])

    by_id = {q["id"]: [float(x) for x in embs[i]] for i, q in enumerate(bank)}
    out = {
        "model": MODEL,
        "sentence_transformers_version": sentence_transformers.__version__,
        "normalize_embeddings": False,
        "dim": dim,
        "n": len(bank),
        "embeddings": by_id,
    }

    path = os.path.join(FIX, "question_embeddings.json")
    with open(path, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"wrote {len(by_id)} x {dim} embeddings ({MODEL}) to {path}")


if __name__ == "__main__":
    main()
