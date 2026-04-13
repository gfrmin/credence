"""Smoke tests for the brain server via the Python client."""

from pathlib import Path

import pytest
from client import BrainClient, BrainError


def test_basic_inference():
    """Test basic Beta-Bernoulli conditioning."""
    brain = BrainClient()
    repo_root = Path(__file__).parent.parent

    try:
        brain.initialize()

        # Create a Beta(1,1) prior
        sid = brain.create_state(type="beta", alpha=1.0, beta=1.0)
        print(f"Created state: {sid}")

        # Mean should be 0.5 — Beta(1,1) mean = 1/2 is bit-exact.
        m = brain.mean(sid)
        print(f"Prior mean: {m}")
        assert m == 0.5, f"Expected 0.5 (exact), got {m}"

        # Condition on observation=1 (success)
        result = brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        print(f"Condition result: {result}")

        # Beta(2,1) mean = 2/3 — Python float division of integer-accumulated
        # α,β is bit-exact, so == not abs() < tol.
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1: {m}")
        assert m == 2 / 3, f"Expected 2/3 (exact), got {m}"

        # Condition on another observation=1
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1,1: {m}")
        assert m == 3 / 4, f"Expected 3/4 (exact), got {m}"

        # Condition on observation=0 (failure)
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=0.0)
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1,1,0: {m}")
        # Beta(3,2) → mean = 3/5.
        assert m == 3 / 5, f"Expected 3/5 (exact), got {m}"

        brain.destroy_state(sid)
        print("PASS: basic inference")
    finally:
        brain.shutdown()


def test_categorical():
    """Test CategoricalMeasure operations."""
    brain = BrainClient()

    try:
        brain.initialize()

        sid = brain.create_state(
            type="categorical",
            space={"type": "finite", "values": [0, 1, 2, 3]},
        )
        w = brain.weights(sid)
        print(f"Uniform weights: {w}")
        assert len(w) == 4
        # Uniform normalisation of 4 equal log-weights is bit-exact.
        assert all(wi == 0.25 for wi in w)

        # Optimise: identity preference (h == a → 1, else 0)
        action, eu = brain.optimise(
            sid,
            actions={"type": "finite", "values": [0, 1, 2, 3]},
            preference={
                "type": "tabular_2d",
                "matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
        )
        print(f"Optimise: action={action}, eu={eu}")
        # All actions are equally good under uniform prior: EU = 0.25 exactly.
        assert eu == 0.25, f"Expected 0.25 (exact), got {eu}"

        brain.destroy_state(sid)
        print("PASS: categorical")
    finally:
        brain.shutdown()


def _make_router_state(brain, n_providers: int, n_categories: int) -> str:
    """Build nested ProductMeasure: providers x categories x (Beta, Gamma).

    Shape: ProductMeasure of n_providers providers, each a ProductMeasure
    of n_categories joint (theta, k) beliefs.
    """
    prior = {
        "type": "product",
        "factors": [
            {"type": "beta", "alpha": 1.0, "beta": 1.0},
            {"type": "gamma", "alpha": 2.0, "beta": 0.5},
        ],
    }
    provider = {
        "type": "product",
        "factors": [prior for _ in range(n_categories)],
    }
    state = {
        "type": "product",
        "factors": [provider for _ in range(n_providers)],
    }
    return brain.create_state(**state)


def _router_preference(
    n_providers: int,
    n_categories: int,
    cat_weights: list[float],
    costs: list[float],
    reward: float,
) -> dict:
    """Build a functional_per_action preference spec for router decide.

    EU(provider a) = reward * sum_c cat_weights[c] * E[theta_{a,c}] - costs[a]

    Each action's functional is a LinearCombination of NestedProjections.
    NestedProjection([a, c, 0]) navigates: state -> providers[a] -> categories[c]
    -> (theta, k)[0] = theta factor. Identity() at the leaf yields mean(Beta).
    """
    actions = {}
    for a in range(n_providers):
        terms = []
        for c in range(n_categories):
            terms.append([
                reward * cat_weights[c],
                {"type": "nested_projection", "indices": [a, c, 0]},
            ])
        actions[str(a)] = {
            "type": "linear_combination",
            "terms": terms,
            "offset": -costs[a],
        }
    return {"type": "functional_per_action", "actions": actions}


def test_router_roundtrip():
    """Router decide + observe using only brain protocol primitives.

    No DSL wrappers. State is a nested ProductMeasure. Preference is a
    functional_per_action spec with LinearCombination of NestedProjections.
    Learning is factor -> condition -> replace_factor chain.
    """
    brain = BrainClient()
    repo_root = Path(__file__).parent.parent

    router_path = repo_root / "examples" / "router.bdsl"
    if not router_path.exists():
        print("SKIP: router.bdsl not found")
        return

    try:
        # Load router.bdsl — gives us a DSL env for the quality kernel.
        brain.initialize(dsl_files={"router": str(router_path)})

        n_providers, n_categories = 3, 5
        cat_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        costs = [0.01, 0.02, 0.005]
        reward = 1.0

        state_id = _make_router_state(brain, n_providers, n_categories)
        print(f"Router state: {state_id}")

        # With uniform Beta(1,1) priors, all E[theta] = 0.5.
        # EU(a) = 1.0 * 5 * 0.2 * 0.5 - costs[a] = 0.5 - costs[a]
        # Provider 2 (cost 0.005) has highest EU; provider 1 (cost 0.02) lowest.
        expected_eu = [0.5 - c for c in costs]

        pref = _router_preference(n_providers, n_categories, cat_weights, costs, reward)
        actions_spec = {"type": "finite", "values": list(range(n_providers))}

        action, eu = brain.optimise(state_id, actions_spec, pref)
        print(f"Decide: action={action}, eu={eu}")

        # EU must hit the closed-form expect(BetaMeasure, Identity) dispatch
        # path — no fallback to Monte Carlo / quadrature.
        best_expected = max(expected_eu)
        best_action = expected_eu.index(best_expected)
        assert action == best_action, f"Expected action {best_action}, got {action}"
        # Safe: JSON3.write preserves full Float64 precision; stdlib json
        # round-trips exactly. Re-verify if either encoder changes.
        assert eu == best_expected, f"Expected EU {best_expected} (exact), got {eu} (non-closed-form?)"
        print(f"PASS: closed-form EU matches exactly (action={action}, eu={eu:.12f})")

        # Determinism: calling optimise again gives the bit-identical EU.
        action2, eu2 = brain.optimise(state_id, actions_spec, pref)
        assert eu == eu2, f"EU not deterministic: {eu} vs {eu2}"
        print("PASS: deterministic dispatch (closed-form only)")

        # Observe high quality for provider `action`, category 2.
        # Chain: factor(state, action) -> factor(provider, 2) -> condition -> replace back.
        provider_id = brain.factor(state_id, action)
        category_id = brain.factor(provider_id, 2)
        brain.condition(category_id, kernel={"type": "quality"}, observation=0.9)
        provider_updated = brain.replace_factor(provider_id, 2, category_id)
        state_updated = brain.replace_factor(state_id, action, provider_updated)
        print(f"Updated state: {state_updated}")

        # E[theta_{action, 2}] should have shifted above 0.5 after observing 0.9.
        new_theta = brain.expect(
            brain.factor(brain.factor(state_updated, action), 2),
            function={"type": "projection", "index": 0},
        )
        print(f"E[theta_{{{action},2}}] after obs=0.9: {new_theta:.4f}")
        assert new_theta > 0.5, f"Expected theta mean > 0.5 after obs 0.9, got {new_theta}"

        # Other factors unchanged — Beta(1,1) mean is bit-exact 0.5.
        other_theta = brain.expect(
            brain.factor(brain.factor(state_updated, (action + 1) % n_providers), 0),
            function={"type": "projection", "index": 0},
        )
        assert other_theta == 0.5, f"Other factor should be unchanged at 0.5 (exact), got {other_theta}"
        print("PASS: only the targeted factor was updated")

        # Re-decide: EU for the updated provider should have shifted up.
        action3, eu3 = brain.optimise(state_updated, actions_spec, pref)
        # updated EU for `action` = reward * (4*0.2*0.5 + 0.2*new_theta) - costs[action]
        updated_eu_for_action = reward * (4 * 0.2 * 0.5 + 0.2 * new_theta) - costs[action]
        assert abs(eu3 - max(updated_eu_for_action, *(expected_eu[i] for i in range(n_providers) if i != action))) < 1e-10
        print(f"Re-decide: action={action3}, eu={eu3:.12f} (provider {action} shifted)")

        print("PASS: router roundtrip")
    finally:
        brain.shutdown()


def test_snapshot_restore():
    """Test state persistence via snapshot/restore."""
    brain = BrainClient()

    try:
        brain.initialize()

        # Create and condition a Beta measure
        sid = brain.create_state(type="beta", alpha=1.0, beta=1.0)
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        m_before = brain.mean(sid)

        # Snapshot
        data = brain.snapshot_state(sid)
        print(f"Snapshot size: {len(data)} bytes")

        # Restore
        sid2 = brain.restore_state(data)
        m_after = brain.mean(sid2)
        print(f"Mean before: {m_before}, after restore: {m_after}")
        # Serialisation round-trip of a BetaMeasure must be bit-exact —
        # any drift would mean precision loss in the snapshot encoder.
        assert m_before == m_after, f"Snapshot round-trip drift: {m_before} vs {m_after}"

        brain.destroy_state(sid)
        brain.destroy_state(sid2)
        print("PASS: snapshot/restore")
    finally:
        brain.shutdown()


def test_unknown_state_id():
    """Operations on a never-registered state_id return StateNotFound (-32000)."""
    brain = BrainClient()
    try:
        brain.initialize()
        bogus = "state_99999"
        with pytest.raises(BrainError) as info:
            brain.mean(bogus)
        assert info.value.code == -32000, f"Expected -32000 (StateNotFound), got {info.value.code}"
        assert bogus in str(info.value), f"Error message should name the bogus id: {info.value}"

        with pytest.raises(BrainError) as info:
            brain.weights(bogus)
        assert info.value.code == -32000

        with pytest.raises(BrainError) as info:
            brain.condition(bogus, kernel={"type": "bernoulli"}, observation=1.0)
        assert info.value.code == -32000
        print("PASS: unknown state_id surfaces StateNotFound (-32000)")
    finally:
        brain.shutdown()


def test_unknown_method():
    """Unknown JSON-RPC method returns MethodNotFound (-32601)."""
    brain = BrainClient()
    try:
        brain.initialize()
        with pytest.raises(BrainError) as info:
            brain._call("nonexistent_method")
        assert info.value.code == -32601, f"Expected -32601 (MethodNotFound), got {info.value.code}"
        print("PASS: unknown method surfaces MethodNotFound (-32601)")
    finally:
        brain.shutdown()


def test_factor_on_non_product_measure():
    """factor on a plain BetaMeasure errors with an informative typed message."""
    brain = BrainClient()
    try:
        brain.initialize()
        sid = brain.create_state(type="beta", alpha=2.0, beta=3.0)
        with pytest.raises(BrainError) as info:
            brain.factor(sid, 0)
        # Non-ProductMeasure triggers a bare error() at the handler site, which
        # lands on the generic -32603 code today. The important invariant is
        # that it is NOT a silent success, and that the message mentions the
        # offending type so clients can diagnose.
        assert "ProductMeasure" in str(info.value), \
            f"Error message should name ProductMeasure: {info.value}"
        brain.destroy_state(sid)
        print("PASS: factor on non-ProductMeasure raises informative BrainError")
    finally:
        brain.shutdown()


def test_replace_factor_identity_pin():
    """replace_factor at index 1 leaves factors 0 and 2 bit-exact unchanged.

    Tightest invariant: mean(factor(new, i)) == mean(factor(orig, i)) for i ≠ replaced.
    Both sides are Python floats from α/(α+β) divisions on integer-accumulated
    α,β — bit-exact under ==, no tolerance needed.
    """
    brain = BrainClient()
    try:
        brain.initialize()

        # 3-factor ProductMeasure with distinct priors.
        prior_spec = {
            "type": "product",
            "factors": [
                {"type": "beta", "alpha": 2.0, "beta": 3.0},
                {"type": "beta", "alpha": 4.0, "beta": 1.0},
                {"type": "beta", "alpha": 1.0, "beta": 5.0},
            ],
        }
        orig_id = brain.create_state(**prior_spec)

        # Means pre-replace.
        m0_orig = brain.mean(brain.factor(orig_id, 0))
        m2_orig = brain.mean(brain.factor(orig_id, 2))
        # Sanity: these are α/(α+β) exactly.
        assert m0_orig == 2.0 / 5.0, f"m0_orig bit-exact check: {m0_orig}"
        assert m2_orig == 1.0 / 6.0, f"m2_orig bit-exact check: {m2_orig}"

        # Replace factor 1 with a fresh Beta(7,2).
        new_f1 = brain.create_state(type="beta", alpha=7.0, beta=2.0)
        new_id = brain.replace_factor(orig_id, 1, new_f1)

        # Means of factors 0 and 2 in the new state must match orig exactly.
        m0_new = brain.mean(brain.factor(new_id, 0))
        m2_new = brain.mean(brain.factor(new_id, 2))
        assert m0_new == m0_orig, f"factor 0 drifted: {m0_orig} → {m0_new}"
        assert m2_new == m2_orig, f"factor 2 drifted: {m2_orig} → {m2_new}"

        # Factor 1 reflects the replacement.
        m1_new = brain.mean(brain.factor(new_id, 1))
        assert m1_new == 7.0 / 9.0, f"factor 1 replacement: expected 7/9, got {m1_new}"

        print("PASS: replace_factor preserves sibling factors bit-exactly")
    finally:
        brain.shutdown()


if __name__ == "__main__":
    test_basic_inference()
    print()
    test_categorical()
    print()
    test_router_roundtrip()
    print()
    test_snapshot_restore()
    print()
    test_unknown_state_id()
    print()
    test_unknown_method()
    print()
    test_factor_on_non_product_measure()
    print()
    test_replace_factor_identity_pin()
    print()
    print("All tests passed!")
