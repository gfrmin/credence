# Role: skin
"""Smoke tests for the skin server via the Python client."""

from pathlib import Path

import pytest
from client import SkinClient, SkinError


def test_basic_inference():
    """Test basic Beta-Bernoulli conditioning."""
    skin = SkinClient()
    repo_root = Path(__file__).parent.parent

    try:
        skin.initialize()

        # Create a Beta(1,1) prior
        sid = skin.create_state(type="beta", alpha=1.0, beta=1.0)
        print(f"Created state: {sid}")

        # Mean should be 0.5 — Beta(1,1) mean = 1/2 is bit-exact.
        m = skin.mean(sid)
        print(f"Prior mean: {m}")
        assert m == 0.5, f"Expected 0.5 (exact), got {m}"

        # Condition on observation=1 (success)
        result = skin.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        print(f"Condition result: {result}")

        # Beta(2,1) mean = 2/3 — Python float division of integer-accumulated
        # α,β is bit-exact, so == not abs() < tol.
        m = skin.mean(sid)
        print(f"Posterior mean after obs=1: {m}")
        assert m == 2 / 3, f"Expected 2/3 (exact), got {m}"

        # Condition on another observation=1
        skin.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        m = skin.mean(sid)
        print(f"Posterior mean after obs=1,1: {m}")
        assert m == 3 / 4, f"Expected 3/4 (exact), got {m}"

        # Condition on observation=0 (failure)
        skin.condition(sid, kernel={"type": "bernoulli"}, observation=0.0)
        m = skin.mean(sid)
        print(f"Posterior mean after obs=1,1,0: {m}")
        # Beta(3,2) → mean = 3/5.
        assert m == 3 / 5, f"Expected 3/5 (exact), got {m}"

        skin.destroy_state(sid)
        print("PASS: basic inference")
    finally:
        skin.shutdown()


def test_categorical():
    """Test CategoricalMeasure operations."""
    skin = SkinClient()

    try:
        skin.initialize()

        sid = skin.create_state(
            type="categorical",
            space={"type": "finite", "values": [0, 1, 2, 3]},
        )
        w = skin.weights(sid)
        print(f"Uniform weights: {w}")
        assert len(w) == 4
        # Uniform normalisation of 4 equal log-weights is bit-exact.
        assert all(wi == 0.25 for wi in w)

        # Optimise: identity preference (h == a → 1, else 0)
        action, eu = skin.optimise(
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

        skin.destroy_state(sid)
        print("PASS: categorical")
    finally:
        skin.shutdown()


def _make_router_state(skin, n_providers: int, n_categories: int) -> str:
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
    return skin.create_state(**state)


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
    """Router decide + observe using only skin protocol primitives.

    No DSL wrappers. State is a nested ProductMeasure. Preference is a
    functional_per_action spec with LinearCombination of NestedProjections.
    Learning is factor -> condition -> replace_factor chain.
    """
    skin = SkinClient()
    repo_root = Path(__file__).parent.parent

    router_path = repo_root / "examples" / "router.bdsl"
    if not router_path.exists():
        print("SKIP: router.bdsl not found")
        return

    try:
        # Load router.bdsl — gives us a DSL env for the quality kernel.
        skin.initialize(dsl_files={"router": str(router_path)})

        n_providers, n_categories = 3, 5
        cat_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        costs = [0.01, 0.02, 0.005]
        reward = 1.0

        state_id = _make_router_state(skin, n_providers, n_categories)
        print(f"Router state: {state_id}")

        # With uniform Beta(1,1) priors, all E[theta] = 0.5.
        # EU(a) = 1.0 * 5 * 0.2 * 0.5 - costs[a] = 0.5 - costs[a]
        # Provider 2 (cost 0.005) has highest EU; provider 1 (cost 0.02) lowest.
        expected_eu = [0.5 - c for c in costs]

        pref = _router_preference(n_providers, n_categories, cat_weights, costs, reward)
        actions_spec = {"type": "finite", "values": list(range(n_providers))}

        action, eu = skin.optimise(state_id, actions_spec, pref)
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
        action2, eu2 = skin.optimise(state_id, actions_spec, pref)
        assert eu == eu2, f"EU not deterministic: {eu} vs {eu2}"
        print("PASS: deterministic dispatch (closed-form only)")

        # Observe high quality for provider `action`, category 2.
        # Chain: factor(state, action) -> factor(provider, 2) -> condition -> replace back.
        provider_id = skin.factor(state_id, action)
        category_id = skin.factor(provider_id, 2)
        skin.condition(category_id, kernel={"type": "quality"}, observation=0.9)
        provider_updated = skin.replace_factor(provider_id, 2, category_id)
        state_updated = skin.replace_factor(state_id, action, provider_updated)
        print(f"Updated state: {state_updated}")

        # E[theta_{action, 2}] should have shifted above 0.5 after observing 0.9.
        new_theta = skin.expect(
            skin.factor(skin.factor(state_updated, action), 2),
            function={"type": "projection", "index": 0},
        )
        print(f"E[theta_{{{action},2}}] after obs=0.9: {new_theta:.4f}")
        assert new_theta > 0.5, f"Expected theta mean > 0.5 after obs 0.9, got {new_theta}"

        # Other factors unchanged — Beta(1,1) mean is bit-exact 0.5.
        other_theta = skin.expect(
            skin.factor(skin.factor(state_updated, (action + 1) % n_providers), 0),
            function={"type": "projection", "index": 0},
        )
        assert other_theta == 0.5, f"Other factor should be unchanged at 0.5 (exact), got {other_theta}"
        print("PASS: only the targeted factor was updated")

        # Re-decide: EU for the updated provider should have shifted up.
        action3, eu3 = skin.optimise(state_updated, actions_spec, pref)
        # updated EU for `action` = reward * (4*0.2*0.5 + 0.2*new_theta) - costs[action]
        updated_eu_for_action = reward * (4 * 0.2 * 0.5 + 0.2 * new_theta) - costs[action]
        assert abs(eu3 - max(updated_eu_for_action, *(expected_eu[i] for i in range(n_providers) if i != action))) < 1e-10
        print(f"Re-decide: action={action3}, eu={eu3:.12f} (provider {action} shifted)")

        print("PASS: router roundtrip")
    finally:
        skin.shutdown()


def test_snapshot_restore():
    """Test state persistence via snapshot/restore."""
    skin = SkinClient()

    try:
        skin.initialize()

        # Create and condition a Beta measure
        sid = skin.create_state(type="beta", alpha=1.0, beta=1.0)
        skin.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        skin.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
        m_before = skin.mean(sid)

        # Snapshot
        data = skin.snapshot_state(sid)
        print(f"Snapshot size: {len(data)} bytes")

        # Restore
        sid2 = skin.restore_state(data)
        m_after = skin.mean(sid2)
        print(f"Mean before: {m_before}, after restore: {m_after}")
        # Serialisation round-trip of a BetaMeasure must be bit-exact —
        # any drift would mean precision loss in the snapshot encoder.
        assert m_before == m_after, f"Snapshot round-trip drift: {m_before} vs {m_after}"

        skin.destroy_state(sid)
        skin.destroy_state(sid2)
        print("PASS: snapshot/restore")
    finally:
        skin.shutdown()


def test_mixture_roundtrip():
    """MixtureMeasure of 3 TaggedBeta components: create, snapshot, restore, bit-exact."""
    skin = SkinClient()
    try:
        skin.initialize()

        sid = skin.create_state(
            type="mixture",
            components=[
                {"type": "tagged_beta", "tag": 0, "alpha": 2.0, "beta": 3.0},
                {"type": "tagged_beta", "tag": 1, "alpha": 5.0, "beta": 1.0},
                {"type": "tagged_beta", "tag": 2, "alpha": 1.0, "beta": 4.0},
            ],
            log_weights=[0.0, 0.0, 0.0],
        )
        w_before = skin.weights(sid)
        print(f"Pre-snapshot weights: {w_before}")

        data = skin.snapshot_state(sid)
        sid2 = skin.restore_state(data)

        w_after = skin.weights(sid2)
        # Weights are normalised via logsumexp — bit-exact round-trip
        # holds when the snapshot encoder doesn't reorder pairwise sums.
        assert w_before == w_after, f"weights drifted: {w_before} vs {w_after}"

        skin.destroy_state(sid)
        skin.destroy_state(sid2)
        print("PASS: mixture roundtrip (3 TaggedBeta components, weights bit-exact)")
    finally:
        skin.shutdown()


def test_normal_gamma_roundtrip():
    """NormalGammaMeasure with explicit κ, μ, α, β: create, snapshot, restore, mean survives."""
    skin = SkinClient()
    try:
        skin.initialize()

        sid = skin.create_state(
            type="normal_gamma",
            kappa=2.0,
            mu=1.5,
            alpha=3.0,
            beta=4.0,
        )
        m_before = skin.mean(sid)
        assert m_before == 1.5, f"NormalGamma mean is μ=1.5, got {m_before}"

        data = skin.snapshot_state(sid)
        sid2 = skin.restore_state(data)

        m_after = skin.mean(sid2)
        assert m_before == m_after, f"NormalGamma round-trip drift: {m_before} vs {m_after}"

        skin.destroy_state(sid)
        skin.destroy_state(sid2)
        print("PASS: normal_gamma roundtrip (mean = μ, bit-exact)")
    finally:
        skin.shutdown()


def test_gamma_roundtrip():
    """GammaMeasure with shape α and rate β: create, snapshot, restore, bit-exact."""
    skin = SkinClient()
    try:
        skin.initialize()

        sid = skin.create_state(type="gamma", alpha=3.0, beta=2.0)
        # Gamma(α, β) mean = α/β = 1.5 — exact.
        m_before = skin.mean(sid)
        assert m_before == 1.5, f"Gamma(3,2) mean = 3/2, got {m_before}"

        data = skin.snapshot_state(sid)
        sid2 = skin.restore_state(data)

        m_after = skin.mean(sid2)
        assert m_before == m_after, f"Gamma round-trip drift: {m_before} vs {m_after}"

        skin.destroy_state(sid)
        skin.destroy_state(sid2)
        print("PASS: gamma roundtrip (mean = α/β, bit-exact)")
    finally:
        skin.shutdown()


def test_dirichlet_roundtrip():
    """DirichletMeasure with concentration α: create, snapshot, restore, weights bit-exact."""
    skin = SkinClient()
    try:
        skin.initialize()

        sid = skin.create_state(type="dirichlet", alpha=[2.0, 3.0, 5.0])
        w_before = skin.weights(sid)
        # Dirichlet mean = α / sum(α) — exact under integer α.
        assert w_before == [0.2, 0.3, 0.5], f"expected [0.2, 0.3, 0.5], got {w_before}"

        data = skin.snapshot_state(sid)
        sid2 = skin.restore_state(data)

        w_after = skin.weights(sid2)
        assert w_before == w_after, f"Dirichlet weights drifted: {w_before} vs {w_after}"

        skin.destroy_state(sid)
        skin.destroy_state(sid2)
        print("PASS: dirichlet roundtrip (weights = α/sum(α), bit-exact)")
    finally:
        skin.shutdown()


def test_v1_snapshot_fails_loudly():
    """A v1 snapshot blob (pre-Move-3 struct layout) must fail loudly on restore.

    The fixture `apps/skin/test_skin_fixtures/beta_v1.b64` is a BetaMeasure
    snapshot from master SHA bf74f98 (pre-Move-3). Post-Move-3, the struct
    layout changed (BetaMeasure now wraps BetaPrevision); Julia's
    Serialization on v1 bytes raises TypeError because `new(Interval,
    Float64, Float64)` doesn't match the new `new(BetaPrevision, Interval)`
    layout. That TypeError must surface as a SkinError to the client —
    silent corruption or silent success would be worse than a loud
    failure.

    Users with v1 snapshots reinitialise. The skin server is not a
    persistence-migration layer. See `apps/skin/test_skin_fixtures/
    README.md` for the fixture protocol.
    """
    fixture_path = Path(__file__).parent / "test_skin_fixtures" / "beta_v1.b64"
    assert fixture_path.exists(), f"v1 fixture missing: {fixture_path}"
    v1_blob = fixture_path.read_text().strip()

    skin = SkinClient()
    try:
        skin.initialize()

        with pytest.raises(SkinError) as info:
            skin.restore_state(v1_blob)

        # Must be a loud failure — not silent corruption. The specific
        # code today is -32603 (generic internal error); the important
        # invariants are (a) it raises, (b) the raise names something
        # informative pointing at the struct change / package mismatch.
        # Julia's Serialization surfaces this as either TypeError (direct
        # struct-new mismatch) or KeyError (package identity hash lookup
        # failure); both are legitimate "loud failure" shapes.
        err_msg = str(info.value)
        informative_markers = [
            "BetaPrevision", "TypeError", "type", "KeyError",
            "Credence", "PkgId", "deserialize",
        ]
        assert any(m in err_msg for m in informative_markers), \
            f"v1 snapshot failure didn't mention any expected shape marker: {err_msg}"
        print(f"PASS: v1 snapshot fails loudly with informative error: {err_msg[:120]}")
    finally:
        skin.shutdown()


def test_beta_bernoulli_conjugate():
    """Beta-Bernoulli conjugate: _dispatch_path == 'conjugate'; posterior bit-exact.

    Stratum-2 contract (Move 4): dispatch-path assertion FIRST, then value —
    a silent registry miss would still produce the correct value via the
    particle path, so pinning the path is load-bearing.
    """
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="beta", alpha=2.0, beta=3.0)
        kernel = {"type": "bernoulli"}

        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        skin.condition(sid, kernel=kernel, observation=1.0)
        m = skin.mean(sid)
        assert m == 3 / 6, f"Beta(3,3) mean = 3/6 (exact), got {m}"

        skin.destroy_state(sid)
        print("PASS: beta-bernoulli conjugate (dispatch-path pinned, α+1 exact)")
    finally:
        skin.shutdown()


def test_flat_likelihood_no_op():
    """BetaMeasure + Flat likelihood: posterior equals prior bit-exactly."""
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="beta", alpha=4.0, beta=7.0)
        m_before = skin.mean(sid)
        assert m_before == 4.0 / 11.0, f"Beta(4,7) mean = 4/11, got {m_before}"

        kernel = {"type": "flat"}
        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        # Flat is obs-agnostic; any obs leaves α, β untouched.
        skin.condition(sid, kernel=kernel, observation=1.0)
        m_after = skin.mean(sid)
        assert m_after == m_before, f"Flat no-op drifted: {m_before} → {m_after}"

        skin.destroy_state(sid)
        print("PASS: flat likelihood no-op (dispatch-path pinned, prior preserved)")
    finally:
        skin.shutdown()


def test_gaussian_normal_conjugate():
    """Gaussian-NormalNormal conjugate: closed-form posterior mean.

    Prior: N(0, 1). Obs: 2.0 with σ_obs = 1. Posterior precision τ_post = 2;
    posterior mean μ_post = (1·0 + 1·2) / 2 = 1.0 — bit-exact integer ratio.
    """
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="gaussian", mu=0.0, sigma=1.0)

        # Skin's gaussian_known_var kernel declares params[:sigma_obs]; the
        # registry matches on that legacy pattern and produces a
        # ConjugatePrevision{GaussianPrevision, NormalNormal}.
        kernel = {"type": "gaussian_known_var", "variance": 1.0}
        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        skin.condition(sid, kernel=kernel, observation=2.0)
        m = skin.mean(sid)
        assert m == 1.0, f"Gaussian posterior μ = (τ_prior·0 + τ_obs·2)/2 = 1.0 exact, got {m}"

        skin.destroy_state(sid)
        print("PASS: gaussian-normal conjugate (dispatch-path pinned, μ_post exact)")
    finally:
        skin.shutdown()


def test_dirichlet_categorical_conjugate():
    """Dirichlet-Categorical conjugate: α at observed idx increments by 1."""
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="dirichlet", alpha=[2.0, 3.0, 5.0])
        w_before = skin.weights(sid)
        assert w_before == [0.2, 0.3, 0.5], f"expected [0.2, 0.3, 0.5], got {w_before}"

        # Observe category index 1 (label 1.0). Posterior α = [2, 4, 5].
        kernel = {"type": "categorical", "categories": [0.0, 1.0, 2.0]}
        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        skin.condition(sid, kernel=kernel, observation=1.0)
        w_after = skin.weights(sid)
        # α/sum(α) = [2, 4, 5]/11
        expected = [2.0 / 11.0, 4.0 / 11.0, 5.0 / 11.0]
        assert w_after == expected, f"Dirichlet α increment drifted: {expected} vs {w_after}"

        skin.destroy_state(sid)
        print("PASS: dirichlet-categorical conjugate (dispatch-path pinned, α[1]+=1 exact)")
    finally:
        skin.shutdown()


def test_normal_gamma_conjugate():
    """NormalGamma + NormalGammaLikelihood conjugate: closed-form κ/μ/α/β update.

    Prior: κ=1, μ=0, α=2, β=2. Obs: r=2.0.
    Posterior: κ_n = 2, μ_n = (1·0 + 2)/2 = 1.0, α_n = 2.5,
               β_n = 2 + 1·(2−0)²/(2·2) = 2 + 1 = 3.0. All exact.
    """
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="normal_gamma", kappa=1.0, mu=0.0, alpha=2.0, beta=2.0)
        m_before = skin.mean(sid)
        assert m_before == 0.0, f"NormalGamma mean = μ = 0.0, got {m_before}"

        kernel = {"type": "normal_gamma"}
        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        skin.condition(sid, kernel=kernel, observation=2.0)
        m_after = skin.mean(sid)
        # Posterior mean is μ_n = 1.0 exact.
        assert m_after == 1.0, f"NormalGamma posterior μ_n = 1.0 exact, got {m_after}"

        skin.destroy_state(sid)
        print("PASS: normal-gamma conjugate (dispatch-path pinned, μ_n exact)")
    finally:
        skin.shutdown()


def test_gamma_exponential_conjugate():
    """Gamma-Exponential conjugate (net-new in Move 4): α+1, β+obs.

    Prior: Gamma(2, 3) — mean α/β = 2/3 exact. Obs: λ_obs = 4.0.
    Posterior: Gamma(3, 7) — mean 3/7 exact.
    """
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="gamma", alpha=2.0, beta=3.0)
        m_before = skin.mean(sid)
        assert m_before == 2.0 / 3.0, f"Gamma(2,3) mean = 2/3, got {m_before}"

        kernel = {"type": "exponential"}
        path = skin._dispatch_path(sid, kernel)
        assert path == "conjugate", f"Expected 'conjugate', got {path!r}"

        skin.condition(sid, kernel=kernel, observation=4.0)
        m_after = skin.mean(sid)
        # Gamma(3, 7) mean = 3/7 exact.
        assert m_after == 3.0 / 7.0, f"Gamma posterior mean = 3/7 exact, got {m_after}"

        skin.destroy_state(sid)
        print("PASS: gamma-exponential conjugate (dispatch-path pinned, net-new fast-path)")
    finally:
        skin.shutdown()


def test_unknown_state_id():
    """Operations on a never-registered state_id return StateNotFound (-32000)."""
    skin = SkinClient()
    try:
        skin.initialize()
        bogus = "state_99999"
        with pytest.raises(SkinError) as info:
            skin.mean(bogus)
        assert info.value.code == -32000, f"Expected -32000 (StateNotFound), got {info.value.code}"
        assert bogus in str(info.value), f"Error message should name the bogus id: {info.value}"

        with pytest.raises(SkinError) as info:
            skin.weights(bogus)
        assert info.value.code == -32000

        with pytest.raises(SkinError) as info:
            skin.condition(bogus, kernel={"type": "bernoulli"}, observation=1.0)
        assert info.value.code == -32000
        print("PASS: unknown state_id surfaces StateNotFound (-32000)")
    finally:
        skin.shutdown()


def test_unknown_method():
    """Unknown JSON-RPC method returns MethodNotFound (-32601)."""
    skin = SkinClient()
    try:
        skin.initialize()
        with pytest.raises(SkinError) as info:
            skin._call("nonexistent_method")
        assert info.value.code == -32601, f"Expected -32601 (MethodNotFound), got {info.value.code}"
        print("PASS: unknown method surfaces MethodNotFound (-32601)")
    finally:
        skin.shutdown()


def test_factor_on_non_product_measure():
    """factor on a plain BetaMeasure errors with an informative typed message."""
    skin = SkinClient()
    try:
        skin.initialize()
        sid = skin.create_state(type="beta", alpha=2.0, beta=3.0)
        with pytest.raises(SkinError) as info:
            skin.factor(sid, 0)
        # Non-ProductMeasure triggers a bare error() at the handler site, which
        # lands on the generic -32603 code today. The important invariant is
        # that it is NOT a silent success, and that the message mentions the
        # offending type so clients can diagnose.
        assert "ProductMeasure" in str(info.value), \
            f"Error message should name ProductMeasure: {info.value}"
        skin.destroy_state(sid)
        print("PASS: factor on non-ProductMeasure raises informative SkinError")
    finally:
        skin.shutdown()


def test_replace_factor_identity_pin():
    """replace_factor at index 1 leaves factors 0 and 2 bit-exact unchanged.

    Tightest invariant: mean(factor(new, i)) == mean(factor(orig, i)) for i ≠ replaced.
    Both sides are Python floats from α/(α+β) divisions on integer-accumulated
    α,β — bit-exact under ==, no tolerance needed.
    """
    skin = SkinClient()
    try:
        skin.initialize()

        # 3-factor ProductMeasure with distinct priors.
        prior_spec = {
            "type": "product",
            "factors": [
                {"type": "beta", "alpha": 2.0, "beta": 3.0},
                {"type": "beta", "alpha": 4.0, "beta": 1.0},
                {"type": "beta", "alpha": 1.0, "beta": 5.0},
            ],
        }
        orig_id = skin.create_state(**prior_spec)

        # Means pre-replace.
        m0_orig = skin.mean(skin.factor(orig_id, 0))
        m2_orig = skin.mean(skin.factor(orig_id, 2))
        # Sanity: these are α/(α+β) exactly.
        assert m0_orig == 2.0 / 5.0, f"m0_orig bit-exact check: {m0_orig}"
        assert m2_orig == 1.0 / 6.0, f"m2_orig bit-exact check: {m2_orig}"

        # Replace factor 1 with a fresh Beta(7,2).
        new_f1 = skin.create_state(type="beta", alpha=7.0, beta=2.0)
        new_id = skin.replace_factor(orig_id, 1, new_f1)

        # Means of factors 0 and 2 in the new state must match orig exactly.
        m0_new = skin.mean(skin.factor(new_id, 0))
        m2_new = skin.mean(skin.factor(new_id, 2))
        assert m0_new == m0_orig, f"factor 0 drifted: {m0_orig} → {m0_new}"
        assert m2_new == m2_orig, f"factor 2 drifted: {m2_orig} → {m2_new}"

        # Factor 1 reflects the replacement.
        m1_new = skin.mean(skin.factor(new_id, 1))
        assert m1_new == 7.0 / 9.0, f"factor 1 replacement: expected 7/9, got {m1_new}"

        print("PASS: replace_factor preserves sibling factors bit-exactly")
    finally:
        skin.shutdown()


if __name__ == "__main__":
    test_basic_inference()
    print()
    test_categorical()
    print()
    test_router_roundtrip()
    print()
    test_snapshot_restore()
    print()
    test_mixture_roundtrip()
    print()
    test_normal_gamma_roundtrip()
    print()
    test_gamma_roundtrip()
    print()
    test_dirichlet_roundtrip()
    print()
    test_v1_snapshot_fails_loudly()
    print()
    test_beta_bernoulli_conjugate()
    print()
    test_flat_likelihood_no_op()
    print()
    test_gaussian_normal_conjugate()
    print()
    test_dirichlet_categorical_conjugate()
    print()
    test_normal_gamma_conjugate()
    print()
    test_gamma_exponential_conjugate()
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
