# Role: body
"""Constructing priors, kernels, and Problems is declarative — no arithmetic."""
from skin.client import SkinClient


def build_coin_problem(skin: SkinClient) -> str:
    prior_id = skin.create_state(type="beta", alpha=1.0, beta=1.0)
    kernel = {
        "type": "kernel",
        "likelihood_family": "BetaBernoulli",
    }
    # Numeric literals as kernel / prior parameters are declarative data,
    # not arithmetic on DSL returns. Lint does not flag construction.
    return prior_id
