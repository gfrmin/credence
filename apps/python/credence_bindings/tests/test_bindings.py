# Role: body
"""Tests for the Python bindings to the Credence DSL."""

import math
from pathlib import Path

import pytest

from credence import (
    Space, Measure, Kernel,
    condition, expect, push, density, draw,
    run_dsl, load_dsl,
)


# ── Space tests ──

def test_finite_space_creation():
    s = Space.finite([0.1, 0.3, 0.5, 0.7, 0.9])
    vals = s.support()
    assert len(vals) == 5
    assert abs(float(vals[0]) - 0.1) < 1e-10
    assert abs(float(vals[4]) - 0.9) < 1e-10


# ── Measure tests ──

def test_uniform_prior():
    s = Space.finite([0.1, 0.3, 0.5])
    m = Measure.uniform(s)
    w = m.weights()
    assert len(w) == 3
    for wi in w:
        assert abs(wi - 1 / 3) < 1e-10


def test_categorical_prior():
    s = Space.finite([0.0, 1.0])
    m = Measure.categorical(s, [0.3, 0.7])
    w = m.weights()
    assert abs(w[0] - 0.3) < 1e-6
    assert abs(w[1] - 0.7) < 1e-6


def test_beta_measure():
    m = Measure.beta(8.0, 2.0)
    assert abs(m.mean() - 0.8) < 1e-10  # credence-lint: allow — precedent:test-oracle — Beta(8,2) mean = 8/10 = 0.8
    expected_var = 8 * 2 / (10**2 * 11)
    assert abs(m.variance() - expected_var) < 1e-10  # credence-lint: allow — precedent:test-oracle — Beta variance αβ/((α+β)²(α+β+1))


def test_gaussian_measure():
    m = Measure.gaussian(5.0, 2.0)
    assert abs(m.mean() - 5.0) < 1e-10  # credence-lint: allow — precedent:test-oracle — Gaussian(5, 2) mean = 5


# ── Kernel tests ──

def test_kernel_density():
    H = Space.finite([0.3, 0.7])
    O = Space.finite([0, 1])

    def bernoulli_ld(h, o):
        h = float(h)
        o = float(o)
        if o == 1.0:
            return math.log(h)
        return math.log(1.0 - h)

    k = Kernel(H, O, log_density=bernoulli_ld)
    assert abs(density(k, 0.7, 1) - math.log(0.7)) < 1e-10  # credence-lint: allow — precedent:test-oracle — Bernoulli(0.7) density at 1 = log(0.7)
    assert abs(density(k, 0.7, 0) - math.log(0.3)) < 1e-10  # credence-lint: allow — precedent:test-oracle — Bernoulli(0.7) density at 0 = log(0.3)
    assert abs(density(k, 0.3, 1) - math.log(0.3)) < 1e-10  # credence-lint: allow — precedent:test-oracle — Bernoulli(0.3) density at 1 = log(0.3)


# ── Condition tests ──

def test_condition_coin():
    H = Space.finite([0.1, 0.3, 0.5, 0.7, 0.9])
    O = Space.finite([0, 1])
    prior = Measure.uniform(H)

    def bernoulli_ld(h, o):
        h, o = float(h), float(o)
        return math.log(h) if o == 1.0 else math.log(1.0 - h)

    k = Kernel(H, O, log_density=bernoulli_ld)

    # Observe heads (1) — should shift toward higher theta
    posterior = condition(prior, k, 1)
    w = posterior.weights()
    # Weights should increase with theta
    for i in range(len(w) - 1):
        assert w[i] < w[i + 1], f"weight {i} should be < weight {i+1}"


def test_condition_chain():
    """Multiple sequential conditions: H H T H (same as coin.bdsl)."""
    H = Space.finite([0.1, 0.3, 0.5, 0.7, 0.9])
    O = Space.finite([0, 1])
    prior = Measure.uniform(H)

    def bernoulli_ld(h, o):
        h, o = float(h), float(o)
        return math.log(h) if o == 1.0 else math.log(1.0 - h)

    k = Kernel(H, O, log_density=bernoulli_ld)

    posterior = condition(condition(condition(condition(prior, k, 1), k, 1), k, 0), k, 1)
    w = posterior.weights()
    # 3 heads 1 tail: theta=0.7 should have highest weight
    assert w[3] == max(w), "theta=0.7 should have highest posterior weight"


# ── Expect tests ──

def test_expect_identity():
    s = Space.finite([0.3, 0.7])
    m = Measure.uniform(s)
    result = expect(m, lambda h: float(h))
    assert abs(result - 0.5) < 1e-10  # credence-lint: allow — precedent:test-oracle — uniform on {0.3, 0.7} mean = 0.5


# ── Draw tests ──

def test_draw():
    s = Space.finite([10.0, 20.0, 30.0])
    m = Measure.uniform(s)
    vals = s.support()
    val_set = {float(v) for v in vals}
    for _ in range(20):
        sample = float(draw(m))
        assert sample in val_set, f"draw returned {sample}, not in {val_set}"


# ── Product measure ──

def test_product_measure():
    m1 = Measure.beta(2.0, 3.0)
    m2 = Measure.beta(5.0, 1.0)
    prod = Measure.product([m1, m2])
    # Should be constructable and drawable
    sample = draw(prod)
    assert len(sample) == 2


# ── Mixture measure ──

def test_mixture_measure():
    s = Space.finite([0.0, 1.0])
    c1 = Measure.categorical(s, [0.9, 0.1])
    c2 = Measure.categorical(s, [0.1, 0.9])
    mix = Measure.mixture([c1, c2], [0.5, 0.5])
    w = mix.weights()
    assert len(w) == 2
    assert abs(sum(w) - 1.0) < 1e-10


# ── DSL interop ──

def test_run_dsl_coin():
    coin_path = Path(__file__).resolve().parents[4] / "examples" / "coin.bdsl"
    source = coin_path.read_text()
    # coin.bdsl's final expression is (voi ...) which returns a number
    result = run_dsl(source)
    assert isinstance(result, (int, float))


def test_load_dsl():
    source = """
    (define H (space :finite 0.3 0.7))
    (define prior (measure H :uniform))
    """
    env = load_dsl(source)
    assert "H" in env
    assert "prior" in env
    # The prior should be a Measure
    assert isinstance(env["prior"], Measure)
    w = env["prior"].weights()
    assert len(w) == 2
    assert abs(w[0] - 0.5) < 1e-10
