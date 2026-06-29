# Role: eval
"""Capability smoke test — the engine's ONLY routing-experiment assertion after the slim.

Proves the language primitive and nothing competitive: per-profile EU-max routing over a
Beta-Bernoulli belief expressed entirely through skin primitives (create_state / factor /
condition / replace_factor / expect / optimise), and that the SAME belief routes to
DIFFERENT models under a quality profile vs a cost profile — so no single fixed rule is
optimal for both. There is not one competitor, profile sweep, persona, or dollar-accounting
fact here: the approach-dominance evaluation (foils, profiles, personas, win/tie/loss
verdicts, the loss map) lives in credence-governor, reached over the skin wire.
"""

import routing_state as RS


def test_routing_capability_smoke():
    # _smoke() conditions a toy 2×2 Beta-Bernoulli belief, checks exact analytic posterior
    # means, and asserts per-profile divergent routing on the same input. Raises on failure.
    RS._smoke()
