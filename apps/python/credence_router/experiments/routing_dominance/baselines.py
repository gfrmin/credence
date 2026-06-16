# Role: eval
"""Competitor routing systems — the "other systems out there" the proof must beat.

Each is a profile-agnostic fixed rule (or a clairvoyant upper bound), standing in for
a class of deployed router:

  - make_always(k)        single-model policies (cost floor / quality ceiling bounds)
  - make_argmax_accuracy  "just use the most accurate model per category" (cost-blind)
  - make_best_fixed_table the best single category→model table serving ALL profiles —
                          the strongest profile-agnostic static router (Wald foil)
  - make_threshold_router RouteLLM-style 2-bin cheap/strong split at a difficulty cut
  - make_oracle_cascade   FrugalGPT-style cheapest-first escalation, charged the
                          CUMULATIVE cost of every rung tried; clairvoyant (stops at
                          the first actually-correct rung) — an UPPER BOUND on any real
                          cascade, which needs a fallible verifier to decide to stop

None of these touch the belief; they are declared non-Bayesian foils
(baseline-comparison precedent). Every arm — including EU-max — maps a question to a
realised (correct: bool, cost: float) outcome against the same frozen oracle grid, so
all arms are scored on identical ground truth.
"""

from __future__ import annotations

from collections.abc import Callable

# An arm maps a question (any object with .id and .category) to a realised outcome.
Arm = Callable[[object], "tuple[bool, float]"]

Grid = "dict[tuple[int, str], bool]"  # (model_idx, question_id) -> was_correct


def empirical_accuracy(grid, train_ids, questions, n_models, categories):
    """Per-(model, category) train-split accuracy: dict[(m, cat)] -> (acc, n)."""
    ids_by_cat = {
        c: [q.id for q in questions if q.id in train_ids and q.category == c]
        for c in categories
    }
    acc = {}
    for m in range(n_models):
        for c in categories:
            ids = ids_by_cat[c]
            if ids:
                k = sum(1 for qid in ids if grid[(m, qid)])
                acc[(m, c)] = (k / len(ids), len(ids))
            else:
                acc[(m, c)] = (0.5, 0)  # credence-lint: allow — precedent:baseline-comparison — no-data fallback for a non-Bayesian foil
    return acc


# credence-lint: allow — precedent:baseline-comparison — single-model policy (cost/quality bound)
def make_always(grid, costs, k: int) -> Arm:
    return lambda q: (grid[(k, q.id)], costs[k])


def make_argmax_accuracy(grid, costs, acc, categories, n_models) -> Arm:
    """Most accurate model per category (ties → cheaper). Cost-blind."""
    # credence-lint: allow — precedent:baseline-comparison — cost-blind argmax-of-means foil
    best = {
        c: min(range(n_models), key=lambda m: (-acc[(m, c)][0], costs[m]))
        for c in categories
    }
    return lambda q: (grid[(best[q.category], q.id)], costs[best[q.category]])


def make_best_fixed_table(grid, costs, acc, categories, n_models, profiles) -> Arm:
    """Single category→model table maximising mean expected welfare across profiles.

    The best a profile-agnostic static router can do: it must serve every profile with
    one table, so it cannot be the per-profile argmax for more than one (Wald).
    """
    def mean_welfare(m, c):  # credence-lint: allow — precedent:baseline-comparison — profile-blind static-table foil
        return sum(r * acc[(m, c)][0] - costs[m] for r in profiles.values()) / len(profiles)

    best = {c: max(range(n_models), key=lambda m: mean_welfare(m, c)) for c in categories}
    return lambda q: (grid[(best[q.category], q.id)], costs[best[q.category]])


def make_threshold_router(grid, costs, acc, categories, n_models) -> Arm:
    """RouteLLM-style 2-bin router: cheapest vs most-capable, split at a difficulty cut.

    The cut is the midpoint of the cheapest model's per-category accuracy range — a
    tuned difficulty threshold. Binary by construction: cannot express a 3-way ladder.
    """
    cheap = min(range(n_models), key=lambda m: costs[m])
    strong = max(range(n_models), key=lambda m: costs[m])
    cheap_acc = {c: acc[(cheap, c)][0] for c in categories}
    # credence-lint: allow — precedent:baseline-comparison — tuned fixed difficulty threshold (RouteLLM foil)
    cut = (min(cheap_acc.values()) + max(cheap_acc.values())) / 2.0
    pick = {c: (cheap if cheap_acc[c] >= cut else strong) for c in categories}
    return lambda q: (grid[(pick[q.category], q.id)], costs[pick[q.category]])


def make_oracle_cascade(grid, costs, order) -> Arm:
    """Cheapest-first escalation, charged the cumulative cost of every rung tried.

    Clairvoyant: stops at the first actually-correct rung. A real cascade needs a
    fallible verifier to decide when to stop, so this is an upper bound on any cascade.
    """
    def arm(q):  # credence-lint: allow — precedent:baseline-comparison — clairvoyant FrugalGPT-cascade upper bound
        spent = 0.0
        for m in order:
            spent += costs[m]
            if grid[(m, q.id)]:
                return (True, spent)
        return (False, spent)

    return arm
