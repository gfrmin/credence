"""Baseline agents: Random, AllTools, Oracle, SingleBestTool.

These provide reference points for the benchmark. The OracleAgent uses the
same Julia DSL backend as BayesianAgent but with true reliabilities pre-loaded,
giving an upper bound on what EU maximisation can achieve.
"""

from __future__ import annotations

import random
from collections import Counter

from credence_agents.agents.common import DecisionStep
from credence_agents.inference.decision import Action, ActionType
from credence_agents.inference.voi import ToolConfig
from credence_agents.environment.tools import SimulatedTool
from credence_agents.julia_bridge import CredenceBridge

# Reuse BayesianAgent's full decision machinery for the oracle
from credence_agents.agents.bayesian_agent import BayesianAgent


class RandomAgent:
    """Picks a random available tool, submits its answer. Lower bound."""

    def __init__(self, num_tools: int, seed: int = 0, name: str = "random"):
        self.name = name
        self._num_tools = num_tools
        self._rng = random.Random(seed)
        self._candidates: tuple[str, ...] = ()
        self._tool_responses: dict[int, int | None] = {}
        self._queried: bool = False
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._tool_responses = {}
        self._queried = False
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if not self._queried:
            self._queried = True
            tool_idx = self._rng.randrange(self._num_tools)
            action = Action(ActionType.QUERY, tool_idx=tool_idx)
        else:
            answer = None
            for resp in self._tool_responses.values():
                if resp is not None:
                    answer = resp
                    break
            if answer is None:
                answer = self._rng.randrange(len(self._candidates))
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._tool_responses[tool_idx] = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"tool_responses": dict(self._tool_responses)}


class AllToolsAgent:
    """Queries all tools, majority-votes on the answers, always submits."""

    def __init__(self, num_tools: int, name: str = "all_tools"):
        self.name = name
        self._num_tools = num_tools
        self._candidates: tuple[str, ...] = ()
        self._tool_responses: dict[int, int | None] = {}
        self._next_tool: int = 0
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._tool_responses = {}
        self._next_tool = 0
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if self._next_tool < self._num_tools:
            action = Action(ActionType.QUERY, tool_idx=self._next_tool)
            self._next_tool += 1
        else:
            answer = self._majority_vote()
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def _majority_vote(self) -> int:
        """Return the most common non-None response, or 0 as fallback."""
        votes = [r for r in self._tool_responses.values() if r is not None]
        if not votes:
            return 0
        counter = Counter(votes)
        return counter.most_common(1)[0][0]

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._tool_responses[tool_idx] = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"tool_responses": dict(self._tool_responses)}


class OracleAgent(BayesianAgent):
    """Knows true tool reliabilities. Upper bound on EU maximisation.

    Uses exactly the same Julia DSL backend as BayesianAgent, but pre-loads
    the reliability states with the true values (as if it had observed
    infinitely many questions). No learning occurs.
    """

    def __init__(
        self,
        bridge: CredenceBridge,
        tools: list[SimulatedTool],
        tool_configs: list[ToolConfig],
        category_names: tuple[str, ...] = ("factual", "numerical", "recent_events", "misconceptions", "reasoning"),
        name: str = "oracle",
    ):
        super().__init__(bridge=bridge, tool_configs=tool_configs, categories=category_names, name=name)

        # Pre-load reliability states with true values using tight Beta priors
        for t_idx, tool in enumerate(tools):
            reliabilities = [
                tool.reliability_by_category.get(cat, 0.0)
                for cat in category_names
            ]
            self.rel_states[t_idx] = bridge.make_oracle_rel_state(reliabilities)

    def on_question_end(self, was_correct: bool | None) -> None:
        # Oracle does NOT update — it already knows the true reliabilities
        pass


class SingleBestToolAgent:
    """Always queries Tool A (cheapest reliable tool), always submits its answer."""

    def __init__(self, tool_idx: int = 0, name: str = "single_best_tool"):
        self.name = name
        self._tool_idx = tool_idx
        self._candidates: tuple[str, ...] = ()
        self._response: int | None = None
        self._queried: bool = False
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._response = None
        self._queried = False
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if not self._queried:
            self._queried = True
            action = Action(ActionType.QUERY, tool_idx=self._tool_idx)
        else:
            answer = self._response if self._response is not None else 0
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._response = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"response": self._response}


def _action_str(action: Action) -> str:
    if action.action_type == ActionType.SUBMIT:
        return f"submit({action.answer_idx})"
    elif action.action_type == ActionType.ABSTAIN:
        return "abstain"
    return f"query({action.tool_idx})"
