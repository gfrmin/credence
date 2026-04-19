# Role: body
"""Bayesian decision-theoretic agent backed by the Julia Credence DSL.

All inference (VOI, EU maximisation, Bayesian conditioning) runs in Julia via
the CredenceBridge. Python manages the host loop: tool queries, per-question
state tracking, and post-question reliability updates.
"""

from __future__ import annotations

from credence_agents.agents.bayesian_selector import BayesianSelector
from credence_agents.agents.common import AgentResult, DecisionStep
from credence_agents.inference.decision import (
    Action,
    ActionType,
    compute_reliability_updates,
)
from credence_agents.inference.voi import ScoringRule, ToolConfig
from credence_agents.julia_bridge import CredenceBridge

_DEFAULT_SCORING = ScoringRule()


class BayesianAgent(BayesianSelector):
    """EU-maximising agent with per-category Beta reliability tracking.

    Extends BayesianSelector with MCQ-specific logic: answer measures,
    tool responses, submit/abstain decisions, and the VOI query loop.

    Conforms to the benchmark's Agent protocol (on_question_start / choose_action /
    on_tool_response / on_question_end) AND provides solve_question for direct use.

    All Bayesian computation runs in the Julia Credence DSL via the bridge.
    Python tracks per-question bookkeeping (which tools were queried, what
    responses came back) and maps post-question feedback to reliability updates.
    """

    def __init__(
        self,
        bridge: CredenceBridge,
        tool_configs: list[ToolConfig],
        categories: tuple[str, ...] | None = None,
        num_categories: int = 5,
        forgetting: float = 1.0,
        name: str = "bayesian",
        scoring: ScoringRule = _DEFAULT_SCORING,
    ):
        resolved_categories = categories or tuple(
            str(i) for i in range(num_categories)
        )
        super().__init__(
            bridge=bridge,
            option_configs=tool_configs,
            categories=resolved_categories,
            scoring=scoring,
        )
        self.name = name
        self.tool_configs = tool_configs
        self.forgetting = forgetting
        self.num_tools = len(tool_configs)
        self._num_categories = self._n_cats

        # Per-question state (set in on_question_start)
        self._answer_measure = None  # Julia CategoricalMeasure
        self._n_answers = 0
        self._available = []  # indices into tool_configs
        self._tool_responses: dict[int, int | None] = {}
        self._trace: list[DecisionStep] = []
        self._step = 0
        self._question_text = ""
        self._submitted_answer: int | None = None

    # --- Benchmark Agent protocol ---

    def on_question_start(
        self,
        question_id: str,
        candidates: tuple[str, ...],
        num_tools: int,
        question_text: str = "",
    ) -> None:
        self._n_answers = len(candidates)
        self._answer_measure = self.bridge.make_answer_measure(self._n_answers)
        self._available = list(range(self.num_tools))
        self._tool_responses = {}
        self._trace = []
        self._step = 0
        self._question_text = question_text
        self._submitted_answer = None

    def choose_action(self) -> Action:
        assert self._answer_measure is not None
        self._step += 1

        bridge = self.bridge

        # All tools exhausted — decide submit vs abstain from current posterior
        if not self._available:
            ans_w = bridge.weights(self._answer_measure)
            best_idx = max(range(len(ans_w)), key=lambda i: ans_w[i])
            p_best = ans_w[best_idx]
            eu_submit = (
                p_best * self.scoring.reward_correct + (1 - p_best) * self.scoring.penalty_wrong
            )
            eu_abstain = self.scoring.reward_abstain

            if eu_submit >= eu_abstain:
                action = Action(ActionType.SUBMIT, answer_idx=best_idx)
                chosen = f"submit({best_idx})"
            else:
                action = Action(ActionType.ABSTAIN)
                chosen = "abstain"

            self._trace.append(
                DecisionStep(
                    step=self._step,
                    eu_submit=eu_submit,
                    eu_abstain=eu_abstain,
                    eu_query={},
                    chosen_action=chosen,
                )
            )
            return action

        cat_w = bridge.weights(self.cat_belief)

        # Build per-tool reliability measures and coverage probs for available tools
        rel_measures = [
            bridge.marginalize_betas(self.rel_states[t], cat_w) for t in self._available
        ]
        cov_probs = [
            bridge.expect_identity(bridge.marginalize_betas(self.cov_states[t], cat_w))
            for t in self._available
        ]
        costs = [self.tool_configs[t].cost for t in self._available]

        # Call DSL agent-step
        action_type, action_arg = bridge.agent_step(
            self._answer_measure,
            rel_measures,
            costs,
            cov_probs,
            self.scoring.reward_correct,
            self.scoring.reward_abstain,
            self.scoring.penalty_wrong,
        )

        # Convert DSL result to Action
        if action_type == 2:  # query
            tool_local_idx = action_arg
            tool_idx = self._available[tool_local_idx]
            action = Action(ActionType.QUERY, tool_idx=tool_idx)
        elif action_type == 0:  # submit
            action = Action(ActionType.SUBMIT, answer_idx=action_arg)
        else:  # abstain
            action = Action(ActionType.ABSTAIN)

        # Log decision trace
        if action.action_type == ActionType.SUBMIT:
            chosen = f"submit({action.answer_idx})"
        elif action.action_type == ActionType.ABSTAIN:
            chosen = "abstain"
        else:
            chosen = f"query({action.tool_idx})"

        # Approximate EU values for the trace
        ans_w = bridge.weights(self._answer_measure)
        p_best = max(ans_w)
        eu_sub = p_best * self.scoring.reward_correct + (1 - p_best) * self.scoring.penalty_wrong
        eu_abs = self.scoring.reward_abstain

        self._trace.append(
            DecisionStep(
                step=self._step,
                eu_submit=eu_sub,
                eu_abstain=eu_abs,
                eu_query={},  # VOI details are inside Julia
                chosen_action=chosen,
            )
        )

        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        assert self._answer_measure is not None
        bridge = self.bridge

        if response is not None:
            self._tool_responses[tool_idx] = response

            # Update coverage state (responded = 1.0)
            self.cov_states[tool_idx], self.cat_belief = bridge.update_beta_state(
                self.cov_states[tool_idx],
                self.cat_belief,
                1.0,
            )

            # Update answer belief via DSL
            cat_w = bridge.weights(self.cat_belief)
            rel_m = bridge.marginalize_betas(self.rel_states[tool_idx], cat_w)
            k = bridge.answer_kernel(rel_m, self._n_answers)
            self._answer_measure = bridge.update_on_response(
                self._answer_measure,
                k,
                float(response),
            )
        else:
            self._tool_responses[tool_idx] = None

            # Update coverage state (not responded = 0.0)
            self.cov_states[tool_idx], self.cat_belief = bridge.update_beta_state(
                self.cov_states[tool_idx],
                self.cat_belief,
                0.0,
            )

        # Remove from available
        if tool_idx in self._available:
            self._available.remove(tool_idx)

    def on_question_end(self, was_correct: bool | None) -> None:
        # Determine what we submitted
        submitted = self._submitted_answer
        if submitted is None:
            for step in reversed(self._trace):
                if step.chosen_action.startswith("submit("):
                    submitted = int(step.chosen_action[7:-1])
                    break

        # Map feedback to per-tool correctness
        updates = compute_reliability_updates(
            submitted,
            was_correct,
            self._tool_responses,
        )

        # Update reliability states via Julia
        bridge = self.bridge
        for tool_idx, was_tool_correct in updates.items():
            if was_tool_correct is None:
                continue
            obs = 1.0 if was_tool_correct else 0.0
            self.rel_states[tool_idx], self.cat_belief = bridge.update_beta_state(
                self.rel_states[tool_idx],
                self.cat_belief,
                obs,
            )

    def get_belief_snapshot(self) -> dict | None:
        if self._answer_measure is None:
            return None
        return {
            "answer_posterior": self.bridge.weights(self._answer_measure),
            "used_tools": [t for t in range(self.num_tools) if t not in self._available],
        }

    # --- Properties for backward compatibility (used by Router) ---

    @property
    def answer_posterior(self) -> list[float]:
        """Current answer posterior as Python list."""
        if self._answer_measure is None:
            return []
        return self.bridge.weights(self._answer_measure)

    @property
    def tool_responses(self) -> dict[int, int | None]:
        """Responses received this question."""
        return self._tool_responses

    # --- High-level solve_question interface ---

    def solve_question(
        self,
        question_text: str,
        candidates: tuple[str, ...],
        category_hint: str | None,
        tool_query_fn,
    ) -> AgentResult:
        """Solve a single question using EU maximisation."""
        self.on_question_start("", candidates, self.num_tools, question_text)
        tools_used: list[int] = []

        while True:
            action = self.choose_action()

            if action.action_type == ActionType.QUERY:
                assert action.tool_idx is not None
                response = tool_query_fn(action.tool_idx)
                self.on_tool_response(action.tool_idx, response)
                tools_used.append(action.tool_idx)

            elif action.action_type == ActionType.SUBMIT:
                self._submitted_answer = action.answer_idx
                ans_w = self.bridge.weights(self._answer_measure)
                confidence = max(ans_w)
                return AgentResult(
                    answer=action.answer_idx,
                    tools_queried=tuple(tools_used),
                    confidence=confidence,
                    decision_trace=tuple(self._trace),
                )

            elif action.action_type == ActionType.ABSTAIN:
                ans_w = self.bridge.weights(self._answer_measure)
                confidence = max(ans_w)
                return AgentResult(
                    answer=None,
                    tools_queried=tuple(tools_used),
                    confidence=confidence,
                    decision_trace=tuple(self._trace),
                )
