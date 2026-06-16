# Role: eval
"""Skin-backed Beta-Bernoulli routing belief for the dominance proof.

θ_{model,category} = P(model answers a question of this category correctly) ~ Beta.
The belief is one product state — models × categories × Beta(1,1) — updated by
Bernoulli conditioning on observed correctness. A routing decision is EU-max via
`skin.optimise` over a `functional_per_action` preference: a LinearCombination of
NestedProjections to each Beta's mean (= E[θ]), with the per-profile value of a
correct answer as `reward` and the real per-model dollar cost as the action offset:

    EU(model a | category c) = reward · E[θ_{a,c}] − cost_a

Why Beta-Bernoulli and not the production router's continuous-quality (θ, k) kernel:
MCQ correctness is *binary*, so Beta-Bernoulli is the honest conjugate model
(credence_agents/CLAUDE.md: "Beta(1,1) uniform — we genuinely don't know tool
reliability"). Forcing binary data through the continuous quality kernel would
require an arbitrary correct→0.8 / wrong→0.2 map — an arbitrary constant. Here a
correct answer is observation 1.0 (increments α), a wrong answer is 0.0 (increments
β); no constants.

All inference runs in the skin (Julia). This module only *declares* structure and
*calls* Tier-1 primitives (create_state / condition / expect / optimise / factor /
replace_factor) — no probability arithmetic in Python (Invariant 1, topological).
The preference mirrors credence_router.routing_domain._router_preference; the only
difference is the leaf is a bare Beta (indices [model, cat]) rather than a
(Beta, Gamma) product (indices [provider, cat, 0]).
"""

from __future__ import annotations

from credence_router.routing_domain import _ensure_skin_importable

REPO_ROOT = _ensure_skin_importable()
from skin.client import SkinClient  # noqa: E402


def build_state(skin: SkinClient, n_models: int, n_categories: int) -> str:
    """Create the routing belief: models × categories × Beta(1,1)."""
    leaf = {"type": "beta", "alpha": 1.0, "beta": 1.0}
    model = {"type": "product", "factors": [leaf for _ in range(n_categories)]}
    state = {"type": "product", "factors": [model for _ in range(n_models)]}
    return skin.create_state(**state)


def observe_correct(
    skin: SkinClient, state_id: str, model_idx: int, cat_idx: int, correct: bool
) -> str:
    """Condition θ_{model,cat} on one outcome. Returns the new top-level state_id.

    factor → condition(bernoulli) → replace_factor, mirroring
    routing_domain._apply_outcome but with the binary-correctness Bernoulli kernel.
    """
    model_id = skin.factor(state_id, model_idx)
    cat_id = skin.factor(model_id, cat_idx)
    skin.condition(cat_id, kernel={"type": "bernoulli"}, observation=1.0 if correct else 0.0)
    model_updated = skin.replace_factor(model_id, cat_idx, cat_id)
    return skin.replace_factor(state_id, model_idx, model_updated)


def learned_accuracy(skin: SkinClient, state_id: str, model_idx: int, cat_idx: int) -> float:
    """E[θ_{model,cat}] — the posterior-mean accuracy. Read through `expect`."""
    return skin.expect(
        state_id, function={"type": "nested_projection", "indices": [model_idx, cat_idx]}
    )


def _preference(
    n_models: int,
    n_categories: int,
    cat_weights: list[float],
    costs: list[float],
    reward: float,
) -> dict:
    """functional_per_action spec: EU(a) = reward · Σ_c w_c · E[θ_{a,c}] − cost_a."""
    actions: dict[str, dict] = {}
    for a in range(n_models):
        terms = [
            [reward * cat_weights[c], {"type": "nested_projection", "indices": [a, c]}]
            for c in range(n_categories)
        ]
        actions[str(a)] = {"type": "linear_combination", "terms": terms, "offset": -costs[a]}
    return {"type": "functional_per_action", "actions": actions}


def route(
    skin: SkinClient,
    state_id: str,
    costs: list[float],
    cat_idx: int,
    reward: float,
    n_categories: int,
) -> int:
    """EU-max model index for a question of category `cat_idx` under `reward`.

    `reward` is the profile: the value (in dollars) of a correct answer. Low reward
    → cost dominates → route cheap; high reward → quality dominates → route capable.
    The belief is shared across profiles (Savage: one posterior, many utilities).
    """
    n_models = len(costs)
    cat_weights = [0.0] * n_categories
    cat_weights[cat_idx] = 1.0
    pref = _preference(n_models, n_categories, cat_weights, costs, reward)
    actions_spec = {"type": "finite", "values": list(range(n_models))}
    action, _eu = skin.optimise(state_id, actions_spec, pref)
    return int(action) if not isinstance(action, int) else action


def _smoke() -> None:
    """Controlled toy: 2 models × 2 categories, known counts, known answer.

    Verifies (a) the skin wiring — Bernoulli conditioning + nested_projection read
    back exact analytic Beta means — and (b) the headline phenomenon: on the SAME
    input, a quality profile and a cost profile route to DIFFERENT models. No spend.
    """
    # Fixtures (test-oracle): two profiles and two model costs.
    CHEAP, EXP = 0.001, 0.01  # model 0 cheap/weak-ish, model 1 expensive/capable
    QUALITY_REWARD = 1.0  # a correct answer worth $1   (quality-hawk profile)
    COST_REWARD = 0.005  # a correct answer worth ~half a cent (cost-hawk profile)
    costs = [CHEAP, EXP]

    skin = SkinClient(project=REPO_ROOT)
    try:
        skin.initialize()
        s = build_state(skin, n_models=2, n_categories=2)

        # model 0: good at cat 0 (5 correct), bad at cat 1 (5 wrong).
        # model 1: good at both (5 correct each).
        for _ in range(5):
            s = observe_correct(skin, s, 0, 0, True)
            s = observe_correct(skin, s, 0, 1, False)
            s = observe_correct(skin, s, 1, 0, True)
            s = observe_correct(skin, s, 1, 1, True)

        # (a) exact posterior means — Beta(1+correct, 1+wrong).
        acc = {(m, c): learned_accuracy(skin, s, m, c) for m in (0, 1) for c in (0, 1)}
        assert acc[(0, 0)] == 6 / 7, acc[(0, 0)]  # Beta(6,1)  # credence-lint: allow — precedent:test-oracle — Beta(6,1) mean exact
        assert acc[(0, 1)] == 1 / 7, acc[(0, 1)]  # Beta(1,6)  # credence-lint: allow — precedent:test-oracle — Beta(1,6) mean exact
        assert acc[(1, 0)] == 6 / 7, acc[(1, 0)]  # Beta(6,1)  # credence-lint: allow — precedent:test-oracle — Beta(6,1) mean exact
        assert acc[(1, 1)] == 6 / 7, acc[(1, 1)]  # Beta(6,1)  # credence-lint: allow — precedent:test-oracle — Beta(6,1) mean exact

        # (b) routing. cat 0 (both models good): both profiles pick the cheap model.
        assert route(skin, s, costs, 0, QUALITY_REWARD, 2) == 0
        assert route(skin, s, costs, 0, COST_REWARD, 2) == 0
        # cat 1 (only the expensive model is good): the profiles DIVERGE.
        q1 = route(skin, s, costs, 1, QUALITY_REWARD, 2)
        c1 = route(skin, s, costs, 1, COST_REWARD, 2)
        assert q1 == 1, q1  # quality-hawk pays for the capable model
        assert c1 == 0, c1  # cost-hawk eats the error rather than pay
        assert q1 != c1, "per-profile divergent routing on the same input"

        print("PASS routing_state smoke:")
        print(f"  learned accuracy (model,cat)->E[θ]: {acc}")
        print(f"  cat0: quality->{route(skin, s, costs, 0, QUALITY_REWARD, 2)} "
              f"cost->{route(skin, s, costs, 0, COST_REWARD, 2)}  (agree: cheap suffices)")
        print(f"  cat1: quality->{q1} cost->{c1}  (DIVERGE: no single fixed rule matches both)")
    finally:
        skin.shutdown()


if __name__ == "__main__":
    _smoke()
