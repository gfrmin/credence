# Role: eval
"""Routing-dominance proof.

Claim: for any set of ≥2 profiles, per-profile EU-max routing weakly dominates every
competitor system and is the only arm optimal across all profiles (Wald complete
class). Offline + deterministic against a frozen oracle grid (toy: synthetic; real:
measured per-model MCQ correctness from oracle.py).

The EU-max arm's routing decisions are made by the skin (skin.optimise over the
Beta-Bernoulli belief in routing_state.py). Competitors are declared foils
(baselines.py). Welfare is scored on identical ground truth for every arm:

    welfare_P(arm) = Σ_q [ reward_P · 1{correct} − cost ]

so a lower bill at equal correctness, or more correct answers at equal bill, both show
up as higher welfare. The belief is shared across profiles (Savage: one posterior,
many utilities); only the per-profile `reward` differs.

Run the zero-spend proof of the mechanism:
    uv run python apps/python/credence_router/experiments/routing_dominance/dominance.py --toy
"""

from __future__ import annotations

import argparse
import random
from types import SimpleNamespace

import baselines as B
import routing_state as RS


def train_state(skin, grid, train_ids, questions, n_models, categories):
    """Condition the Beta-Bernoulli belief on every model's correctness over the train
    split. All arms train on this identical split (the cache holds every model's answer
    to every question — the offline counterfactual advantage)."""
    cat_index = {c: i for i, c in enumerate(categories)}
    s = RS.build_state(skin, n_models, len(categories))
    for q in questions:
        if q.id in train_ids:
            ci = cat_index[q.category]
            for m in range(n_models):
                s = RS.observe_correct(skin, s, m, ci, grid[(m, q.id)])
    return s


def eu_max_arm(skin, state, grid, costs, categories, reward):
    cat_index = {c: i for i, c in enumerate(categories)}

    def arm(q):
        m = RS.route(skin, state, costs, cat_index[q.category], reward, len(categories))
        return (grid[(m, q.id)], costs[m])

    return arm


def welfare(arm, test_qs, reward):
    return sum(reward * (1.0 if correct else 0.0) - cost for correct, cost in map(arm, test_qs))


def eu_routing_table(skin, state, costs, categories, reward):
    """The category→model table EU-max picks under `reward` (for display)."""
    return {
        c: RS.route(skin, state, costs, i, reward, len(categories))
        for i, c in enumerate(categories)
    }


def run(grid, questions, model_names, costs, categories, profiles, seeds, train_frac=0.6):
    ids = [q.id for q in questions]
    n_models = len(model_names)
    cheapest = min(range(n_models), key=lambda m: costs[m])
    priciest = max(range(n_models), key=lambda m: costs[m])
    cascade_order = sorted(range(n_models), key=lambda m: costs[m])

    # regrets[arm][profile] -> list over seeds; also keep raw welfare for the first seed.
    regrets: dict[str, dict[str, list[float]]] = {}
    first_seed_tables: dict[str, dict] = {}
    first_seed_welfare: dict = {}

    skin = RS.SkinClient(project=RS.REPO_ROOT)
    skin.initialize()
    try:
        for si, seed in enumerate(seeds):
            rng = random.Random(seed)
            shuffled = ids[:]
            rng.shuffle(shuffled)
            n_train = int(len(ids) * train_frac)
            train_ids = set(shuffled[:n_train])
            test_ids = set(shuffled[n_train:])
            test_qs = [q for q in questions if q.id in test_ids]

            acc = B.empirical_accuracy(grid, train_ids, questions, n_models, categories)
            state = train_state(skin, grid, train_ids, questions, n_models, categories)

            competitors = {
                f"always-{model_names[cheapest]}": B.make_always(grid, costs, cheapest),
                f"always-{model_names[priciest]}": B.make_always(grid, costs, priciest),
                "argmax-accuracy": B.make_argmax_accuracy(grid, costs, acc, categories, n_models),
                "best-fixed-table": B.make_best_fixed_table(grid, costs, acc, categories, n_models, profiles),
                "threshold-router": B.make_threshold_router(grid, costs, acc, categories, n_models),
                "oracle-cascade": B.make_oracle_cascade(grid, costs, cascade_order),
            }

            for p, reward in profiles.items():
                eu = eu_max_arm(skin, state, grid, costs, categories, reward)
                w_eu = welfare(eu, test_qs, reward)
                for name, arm in competitors.items():
                    w_arm = welfare(arm, test_qs, reward)
                    regrets.setdefault(name, {}).setdefault(p, []).append(w_eu - w_arm)
                    if si == 0:
                        first_seed_welfare.setdefault(p, {})["EU-max (credence)"] = w_eu
                        first_seed_welfare[p][name] = w_arm

            if si == 0:
                for p, reward in profiles.items():
                    first_seed_tables[p] = eu_routing_table(skin, state, costs, categories, reward)
    finally:
        skin.shutdown()

    _report(model_names, categories, profiles, regrets, first_seed_tables, first_seed_welfare, len(seeds))
    return regrets


def _report(model_names, categories, profiles, regrets, tables, welfare_by_profile, n_seeds):
    print("\n" + "=" * 72)
    print("ROUTING-DOMINANCE PROOF  (welfare = reward·correct − cost; higher is better)")
    print("=" * 72)

    print(f"\nProfiles (reward = $ value of a correct answer): {dict(profiles)}")
    print("\nEU-max routing table (category → model) — SAME belief, per-profile decision:")
    for p in profiles:
        named = {c: model_names[tables[p][c]] for c in categories}
        print(f"  {p:>14}: {named}")
    diverge = any(
        tables[p1][c] != tables[p2][c]
        for c in categories
        for p1 in profiles
        for p2 in profiles
    )
    print(f"  → per-profile routing diverges: {diverge}  "
          f"(no single fixed table is optimal for all profiles — Wald)")

    print(f"\nRealised welfare, first seed (test split):")
    for p in profiles:
        row = welfare_by_profile[p]
        best = max(row.values())
        print(f"  [{p}]")
        for name, w in sorted(row.items(), key=lambda kv: -kv[1]):
            tag = "  ← optimal" if w == best else ""
            print(f"      {name:>22}: {w:+.4f}{tag}")

    print(f"\nMean regret vs EU-max over {n_seeds} seed(s)  "
          f"(regret = welfare[EU-max] − welfare[arm]; ≥0 means EU-max ≥ arm):")
    header = "  " + " " * 22 + "".join(f"{p:>16}" for p in profiles)
    print(header)
    for name in regrets:
        cells = ""
        for p in profiles:
            rs = regrets[name][p]
            mean = sum(rs) / len(rs)
            winrate = sum(1 for r in rs if r >= -1e-12) / len(rs)
            cells += f"{mean:>+10.4f}({winrate:.0%})"
        print(f"  {name:>22}{cells}")

    # Verdict — honest per-profile beats/ties/loses (EU-max can LOSE: with a belief learned
    # from finite data it is the per-profile argmax over the BELIEF, which equals the true
    # optimum only when the belief is well-calibrated; on thin data it can misrank).
    # Comparisons below are on already-computed welfare regrets, for the human-read verdict
    # line only — non-causal display, the same inline pattern as the regret table above.
    print("\nVERDICT  (per profile — does EU-max beat / tie / lose to the arm?):")
    for name in regrets:
        beats = [p for p in profiles if sum(regrets[name][p]) / len(regrets[name][p]) > 1e-6]
        loses = [p for p in profiles if sum(regrets[name][p]) / len(regrets[name][p]) < -1e-6]
        ties = [p for p in profiles if -1e-6 <= sum(regrets[name][p]) / len(regrets[name][p]) <= 1e-6]
        parts = []
        if beats:
            parts.append(f"beats on {beats}")
        if ties:
            parts.append(f"ties on {ties}")
        if loses:
            parts.append(f"LOSES on {loses}")
        print(f"  {name:>22}: EU-max " + "; ".join(parts))

    # Wald admissibility: no single FIXED router should beat EU-max on EVERY profile. The
    # clairvoyant oracle-cascade is excluded — it is an unattainable upper bound, not a rule.
    fixed = [n for n in regrets if n != "oracle-cascade"]
    dominators = [n for n in fixed if all(sum(regrets[n][p]) / len(regrets[n][p]) < -1e-6 for p in profiles)]
    if dominators:
        print(f"\n  ⚠ a fixed router beats EU-max on ALL profiles: {dominators} — EU-max is DOMINATED here.")
    else:
        print("\n  No fixed router beats EU-max on all profiles ⇒ EU-max is UNDOMINATED across "
              "profiles (admissible).")
        print("  (Per-profile dominance is stronger and holds where the belief is well-calibrated; "
              "with thin data EU-max can trail the per-profile winner — see ROUTING_DOMINANCE.md.)")
    print("=" * 72 + "\n")


def build_toy_oracle():
    """Controlled 3×3 oracle: 3 models (cheap/mid/exp) × 3 difficulties.

    Correctness pattern (homogeneous per category, so the result is split-invariant —
    the cleanest demonstration of the theorem, no sampling noise):
        easy   : all models correct      → cheapest suffices
        medium : cheap fails, mid+exp ok  → mid is the cheapest adequate model
        hard   : only exp correct         → must pay for exp to be right
    """
    categories = ["easy", "medium", "hard"]
    model_names = ["cheap", "mid", "exp"]
    costs = [0.001, 0.005, 0.02]  # rough haiku/sonnet/opus ratio
    pattern = {"easy": [1, 1, 1], "medium": [0, 1, 1], "hard": [0, 0, 1]}
    questions = [
        SimpleNamespace(id=f"{c}{i}", category=c) for c in categories for i in range(6)
    ]
    grid = {(m, q.id): bool(pattern[q.category][m]) for q in questions for m in range(len(model_names))}
    # Profiles = value (in $) of a correct answer. cost-hawk: a correct answer is worth
    # about one model-call; quality-hawk: worth far more than any call.
    profiles = {"cost-hawk": 0.01, "quality-hawk": 1.0}
    return grid, questions, model_names, costs, categories, profiles


def build_real_oracle(path="oracle_grid.json"):
    """Load the real-model oracle grid produced by oracle.py.

    Same shape as build_toy_oracle, so run() is byte-identical on real data — only the
    grid (measured per-model MCQ correctness) and the real per-call costs differ. Profiles
    are the dollar value of a correct answer: cost-hawk ≈ one expensive call; quality-hawk
    far more. Keeps only questions every model answered (a complete grid row), so a partial
    oracle run still yields a sound proof on its completed subset.
    """
    import json
    import os
    from types import SimpleNamespace

    p = path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), path)
    data = json.loads(open(p).read())
    model_names, costs, categories = data["models"], data["costs"], data["categories"]
    n = len(model_names)
    grid = {}
    for k, v in data["grid"].items():
        mi, qid = k.split("|", 1)
        grid[(int(mi), qid)] = bool(v)
    questions = [
        SimpleNamespace(id=q["id"], category=q["category"], difficulty=q["difficulty"])
        for q in data["questions"]
        if all((mi, q["id"]) in grid for mi in range(n))
    ]
    profiles = {"cost-hawk": 0.02, "quality-hawk": 1.0}
    return grid, questions, model_names, costs, categories, profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="run the controlled synthetic proof (no spend)")
    ap.add_argument("--real", action="store_true", help="run against the real-model oracle grid (oracle.py output)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.toy:
        oracle = build_toy_oracle()
    elif args.real:
        oracle = build_real_oracle()
    else:
        raise SystemExit("choose --toy (synthetic, no spend) or --real (oracle.py grid)")
    grid, questions, model_names, costs, categories, profiles = oracle
    run(grid, questions, model_names, costs, categories, profiles, seeds=list(range(args.seeds)))


if __name__ == "__main__":
    main()
