"""Startup smoke test.

Guards against the exact regression that shipped in v0.3.1: a missing
DSL symbol caused FastAPI lifespan to crash, which meant the
container was unhealthy forever. This test runs startup end-to-end
and asserts /ready / /health respond 200. Any host-to-brain contract
drift breaks this test.
"""

from __future__ import annotations

import os

import pytest

try:
    import juliacall  # noqa: F401
    _JULIA_AVAILABLE = True
except Exception:
    _JULIA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _JULIA_AVAILABLE, reason="Julia runtime not available",
)


@pytest.fixture
def test_env(monkeypatch):
    """Ensure at least one LLM provider is considered available.

    `available_models()` keys off env vars — a dummy placeholder is enough
    to trigger LLM domain initialisation. No network calls are made at
    startup; the brain runs locally.
    """
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY",
        os.environ.get("ANTHROPIC_API_KEY", "sk-ant-test-smoke"),
    )


def test_startup_reaches_ready(test_env):
    """Lifespan must complete, /ready must answer 200."""
    from fastapi.testclient import TestClient

    from credence_router.server import app

    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text

        health = client.get("/health")
        assert health.status_code == 200, health.text


def test_state_endpoint_populated(test_env):
    """/state must include a populated `llm` section when an API key is set."""
    from fastapi.testclient import TestClient

    from credence_router.server import app

    with TestClient(app) as client:
        resp = client.get("/state")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "llm" in body, body
        # Presence of provider entries confirms RoutingDomain constructed and
        # the brain subprocess answered make-state / reliability calls.
        assert body["llm"], body
