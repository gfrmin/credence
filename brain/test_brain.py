"""Smoke tests for the brain server via the Python client."""

from pathlib import Path

from client import BrainClient


def test_basic_inference():
    """Test basic Beta-Bernoulli conditioning."""
    brain = BrainClient()
    repo_root = Path(__file__).parent.parent

    try:
        brain.initialize()

        # Create a Beta(1,1) prior
        sid = brain.create_state(type="beta", alpha=1.0, beta=1.0)
        print(f"Created state: {sid}")

        # Mean should be 0.5
        m = brain.mean(sid)
        print(f"Prior mean: {m}")
        assert abs(m - 0.5) < 0.01, f"Expected 0.5, got {m}"

        # Condition on observation=1 (success)
        result = brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        print(f"Condition result: {result}")

        # Mean should now be 2/3 (Beta(2,1))
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1: {m}")
        assert abs(m - 2 / 3) < 0.01, f"Expected 0.667, got {m}"

        # Condition on another observation=1
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1,1: {m}")
        assert abs(m - 3 / 4) < 0.01, f"Expected 0.75, got {m}"

        # Condition on observation=0 (failure)
        brain.condition(sid, kernel={"type": "bernoulli"}, observation=0.0)
        m = brain.mean(sid)
        print(f"Posterior mean after obs=1,1,0: {m}")
        # Beta(3,2) → mean = 3/5 = 0.6
        assert abs(m - 0.6) < 0.01, f"Expected 0.6, got {m}"

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
        assert all(abs(wi - 0.25) < 0.01 for wi in w)

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
        # All actions are equally good under uniform prior
        assert abs(eu - 0.25) < 0.01

        brain.destroy_state(sid)
        print("PASS: categorical")
    finally:
        brain.shutdown()


def test_dsl_call():
    """Test calling DSL functions (router.bdsl)."""
    brain = BrainClient()
    repo_root = Path(__file__).parent.parent

    router_path = repo_root / "examples" / "router.bdsl"
    if not router_path.exists():
        print("SKIP: router.bdsl not found")
        return

    try:
        brain.initialize(dsl_files={"router": str(router_path)})

        # Create router state: 2 providers, 3 categories
        # Returns an opaque state (nested list of measures)
        state_id = brain.call_dsl("router", "make-router-state", [2, 3])
        print(f"Router state: {state_id}")
        assert isinstance(state_id, str), f"Expected state_id string, got {type(state_id)}"

        # Make a routing decision
        cat_weights = [0.5, 0.3, 0.2]
        costs = [0.01, 0.02]
        reward = 1.0
        provider = brain.call_dsl(
            "router",
            "router-decide",
            [{"ref": state_id}, cat_weights, costs, reward],
        )
        print(f"Router decision: provider {provider}")
        # With uniform priors and lower cost, should prefer provider 0
        assert provider == 0, f"Expected provider 0, got {provider}"

        # Observe quality from provider 0
        new_state_id = brain.call_dsl(
            "router",
            "router-observe",
            [{"ref": state_id}, 0, cat_weights, 0.9],
        )
        print(f"Updated state: {new_state_id}")
        assert isinstance(new_state_id, str)

        print("PASS: DSL call")
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
        assert abs(m_before - m_after) < 0.001

        brain.destroy_state(sid)
        brain.destroy_state(sid2)
        print("PASS: snapshot/restore")
    finally:
        brain.shutdown()


if __name__ == "__main__":
    test_basic_inference()
    print()
    test_categorical()
    print()
    test_dsl_call()
    print()
    test_snapshot_restore()
    print()
    print("All tests passed!")
