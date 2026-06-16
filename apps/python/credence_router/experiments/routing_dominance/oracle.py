# Role: eval
"""Real-model oracle: measure per-model MCQ correctness for the routing-dominance proof.

Turns the synthetic toy into a REAL headline: query a cost-ladder of real models on a
labelled benchmark (credence_agents' 50-question bank — `correct_index` is ground truth)
and record, per (model, question), whether the model chose the right answer. The result
is a frozen grid `dominance.py --real` routes over, so the dominance claim stands on real
model accuracies, not a synthetic generator.

This module only MEASURES (queries models, compares to the labelled answer) and CACHES.
No probability arithmetic, no routing decision here (that is the skin's job in
routing_state.py / dominance.py). `# Role: eval`.

Determinism + thrift: temperature 0, tiny max_tokens, and a per-(model,question) cache
on disk — a re-run re-spends nothing. ~50 questions × 3 models ≈ 150 calls.

Run (key from the keyring, per the machine's secret convention):
    ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) \
        uv run python apps/python/credence_router/experiments/routing_dominance/oracle.py --test
    ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) \
        uv run python apps/python/credence_router/experiments/routing_dominance/oracle.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

from credence_agents.environment.questions import QUESTION_BANK
from credence_router.tools.llm.provider import ALL_MODELS, model_cost

# The cost ladder, with CURRENT model IDs (provider.py's claude-opus-4-6 is stale → 4-8).
# Costs are the proxy's own per-call estimate (provider.model_cost = input + 0.5·output per
# 1k); the exp tier reuses the opus-4-6 price row (same tier) since 4-8 isn't in the table.
MODELS = [
    ("cheap", "claude-haiku-4-5", "claude-haiku-4-5"),
    ("mid", "claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("exp", "claude-opus-4-8", "claude-opus-4-6"),  # (api id, price-row id)
]
ENDPOINT = "https://api.anthropic.com/v1/chat/completions"
LETTERS = ("A", "B", "C", "D")

CACHE_PATH = Path(__file__).with_name("oracle_grid.json")


def costs() -> list[float]:
    return [round(model_cost(ALL_MODELS[price_id]), 6) for _, _, price_id in MODELS]


def prompt_for(q) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(q.candidates))
    return (
        f"Answer this multiple-choice question. Reply with ONLY the single letter "
        f"(A, B, C, or D) of the correct option — no explanation.\n\n"
        f"{q.text}\n{opts}\n\nAnswer:"
    )


def parse_choice(text: str) -> int | None:
    """First standalone A/B/C/D in the reply → 0..3, else None."""
    m = re.search(r"\b([ABCD])\b", text.upper())
    return LETTERS.index(m.group(1)) if m else None


def query(api_id: str, q, api_key: str) -> tuple[int | None, str]:
    """One MCQ call. Returns (chosen_index_or_None, raw_reply).

    Sends temperature=0 for determinism; some newer models (e.g. opus-4-8) reject the
    param ("temperature is deprecated for this model") — retry once without it.
    """
    base = {"model": api_id, "messages": [{"role": "user", "content": prompt_for(q)}], "max_tokens": 16}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for body in ({**base, "temperature": 0.0}, base):
        r = httpx.post(ENDPOINT, headers=headers, json=body, timeout=60.0)
        if r.status_code == 400 and "temperature" in r.text and body.get("temperature") is not None:
            continue  # retry without temperature
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return parse_choice(text), text.strip()
    raise RuntimeError(f"query failed for {api_id}")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"raw": {}}


def save(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def build_grid(cache: dict) -> dict:
    """Assemble the dominance.py-consumable grid from the per-(model,question) cache."""
    cats = sorted({q.category for q in QUESTION_BANK})
    grid = {}
    for mi, (_, api_id, _) in enumerate(MODELS):
        for q in QUESTION_BANK:
            rec = cache["raw"].get(f"{api_id}|{q.id}")
            if rec is not None:
                grid[f"{mi}|{q.id}"] = bool(rec["correct"])
    return {
        "models": [name for name, _, _ in MODELS],
        "model_ids": [api_id for _, api_id, _ in MODELS],
        "costs": costs(),
        "categories": cats,
        "questions": [{"id": q.id, "category": q.category, "difficulty": q.difficulty} for q in QUESTION_BANK],
        "grid": grid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="one call per model (verify IDs/endpoint) then exit")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set — run with: ANTHROPIC_API_KEY=$(secret-tool lookup service env key ANTHROPIC_API_KEY) ...")

    if args.test:
        q = QUESTION_BANK[0]
        print(f"test question [{q.id}/{q.category}]: {q.text}  (correct={LETTERS[q.correct_index]})")
        for name, api_id, _ in MODELS:
            try:
                idx, raw = query(api_id, q, api_key)
                ok = idx == q.correct_index
                print(f"  {name:6} {api_id:22} → {LETTERS[idx] if idx is not None else '??':2} "
                      f"({'correct' if ok else 'wrong'})   raw={raw!r}")
            except httpx.HTTPStatusError as e:
                print(f"  {name:6} {api_id:22} → HTTP {e.response.status_code}: {e.response.text[:200]}")
        print("\ncosts (per-call $):", dict(zip([m[0] for m in MODELS], costs())))
        return

    cache = load_cache()
    raw = cache["raw"]
    n_calls = 0
    for name, api_id, _ in MODELS:
        correct_n = total_n = 0
        for q in QUESTION_BANK:
            key = f"{api_id}|{q.id}"
            if key not in raw:
                idx, reply = query(api_id, q, api_key)
                raw[key] = {"answer": LETTERS[idx] if idx is not None else None,
                            "correct": idx == q.correct_index, "raw": reply}
                n_calls += 1
                if n_calls % 10 == 0:
                    save(cache)  # checkpoint
            total_n += 1
            correct_n += int(raw[key]["correct"])
        print(f"{name:6} {api_id:22} accuracy {correct_n}/{total_n} = {correct_n/total_n:.2%}")
    save(cache)
    save_grid = build_grid(cache)
    Path(__file__).with_name("oracle_grid.json").write_text(json.dumps({**cache, **save_grid}, indent=2))
    print(f"\n{n_calls} live calls made; grid → {CACHE_PATH}")
    # Per-category accuracy table (the routing signal).
    cats = save_grid["categories"]
    print("\nper-(model, category) accuracy:")
    print("  ", "model ".ljust(8), "".join(c[:8].ljust(10) for c in cats))
    for mi, (name, api_id, _) in enumerate(MODELS):
        cells = []
        for c in cats:
            qs = [q for q in QUESTION_BANK if q.category == c]
            k = sum(int(raw[f"{api_id}|{q.id}"]["correct"]) for q in qs)
            cells.append(f"{k}/{len(qs)}".ljust(10))
        print("  ", name.ljust(8), "".join(cells))


if __name__ == "__main__":
    main()
