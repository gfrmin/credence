# Role: Python-side action EUs + bridge to Julia decide-action.
"""Compute action EUs from Beta posteriors and dispatch the action choice
to the tool_decider DSL via juliacall.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


class Action(enum.IntEnum):
    EXECUTE = 0
    SUBSTITUTE = 1
    STOP = 2
    ASK = 3


@dataclass(frozen=True)
class DecideInputs:
    action_eus: list[float]
    voi_ask: float
    ask_cost: float


def compute_action_eus(
    *,
    proposed_alpha: float,
    proposed_beta: float,
    best_alt_alpha: float,
    best_alt_beta: float,
    stop_alpha: float,
    stop_beta: float,
    llm_cost: float,
) -> list[float]:
    """Return [eu_execute, eu_substitute, eu_stop, eu_ask_floor].

    Approval rate = α / (α + β); EU = approval - llm_cost (the LLM call has
    already been made; llm_cost is the dollar cost we want to remember when
    comparing actions of similar approval). Ask floor = 0.0 (the ask itself
    is rated only via its VOI - cost gate, not via the action_eus vector).
    """
    eu_execute = _mean(proposed_alpha, proposed_beta) - llm_cost
    eu_substitute = _mean(best_alt_alpha, best_alt_beta) - llm_cost
    eu_stop = _mean(stop_alpha, stop_beta)
    return [eu_execute, eu_substitute, eu_stop, 0.0]


def _mean(alpha: float, beta: float) -> float:
    s = alpha + beta
    if s <= 0.0:
        return 0.5
    return alpha / s


# ---- Julia bridge ----

_julia_lock = Lock()
_decide_action_callable = None


def _julia_decide_action():
    global _decide_action_callable
    with _julia_lock:
        if _decide_action_callable is not None:
            return _decide_action_callable
        from juliacall import Main as jl  # type: ignore
        host_path = (
            Path(__file__).resolve().parents[6]
            / "apps" / "julia" / "tool_decider" / "host.jl"
        )
        jl.include(str(host_path))
        _decide_action_callable = jl.decide_action
        return _decide_action_callable


def decide(inputs: DecideInputs) -> Action:
    from juliacall import Main as jl  # type: ignore
    fn = _julia_decide_action()
    eus_jl = jl.Vector[jl.Float64](list(inputs.action_eus))
    idx = int(fn(eus_jl, float(inputs.voi_ask), float(inputs.ask_cost)))
    return Action(idx)
