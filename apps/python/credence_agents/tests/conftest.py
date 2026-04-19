# Role: body
"""Shared fixtures for credence-engine tests."""

import pytest
from credence_agents.julia_bridge import CredenceBridge


@pytest.fixture(scope="session")
def bridge():
    """Session-scoped CredenceBridge (Julia loads once for all tests)."""
    return CredenceBridge()
