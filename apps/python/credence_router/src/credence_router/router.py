"""Router: transparent, cost-optimal tool routing via EU maximisation.

Wraps credence's BayesianAgent (backed by the Julia Credence DSL) to provide
a clean API for routing multiple-choice questions to the cheapest reliable tool.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from credence_agents.agents.bayesian_agent import BayesianAgent
from credence_agents.agents.common import DecisionStep
from credence_agents.environment.categories import CATEGORIES
from credence_router.categories import make_keyword_category_infer_fn
from credence_agents.inference.voi import ScoringRule, ToolConfig
from credence_agents.julia_bridge import CredenceBridge

from credence_router.answer import Answer
from credence_router.tool import Tool

DEFAULT_CATEGORIES = CATEGORIES
DEFAULT_SCORING = ScoringRule(reward_correct=0.01, penalty_wrong=-0.005, reward_abstain=0.0)


class Router:
    """Transparent, cost-optimal tool routing via EU maximisation.

    Replaces LangGraph's opaque LLM-based routing with principled VOI
    calculations. Zero routing cost, <1ms routing latency, learns from feedback.
    """

    name = "credence-router"

    def __init__(
        self,
        tools: list[Tool],
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        scoring: ScoringRule | None = None,
        forgetting: float = 1.0,
        latency_weight: float = 0.0,
        category_infer_fn=None,
        bridge: CredenceBridge | None = None,
    ):
        self._tools = tools
        self._categories = categories
        self._scoring = scoring or DEFAULT_SCORING
        self._latency_weight = latency_weight

        # Create bridge (lazy-loads Julia on first use)
        self._bridge = bridge or CredenceBridge()

        # Convert tools to ToolConfig list with effective costs
        self._category_infer_fn = category_infer_fn or make_keyword_category_infer_fn(categories)

        self._tool_configs = [
            ToolConfig(
                cost=t.cost + latency_weight * t.latency,
                coverage_by_category=t.coverage(categories),
            )
            for t in tools
        ]

        self._agent = BayesianAgent(
            bridge=self._bridge,
            tool_configs=self._tool_configs,
            categories=categories,
            forgetting=forgetting,
            scoring=self._scoring,
            name="credence-router",
        )

        # State from last solve
        self._last_trace: tuple[DecisionStep, ...] = ()
        self._last_tools_queried: tuple[int, ...] = ()

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer:
        """Route a question to the optimal tool(s) and return an answer.

        Args:
            question: The question text.
            candidates: Possible answers (2-10 options).
            category_hint: Optional category override (e.g. "numerical").

        Returns:
            Answer with choice, confidence, cost breakdown, and reasoning trace.
        """
        t_start = time.monotonic()

        # Track per-query costs
        monetary_costs: list[float] = []

        def tool_query_fn(tool_idx: int) -> int | None:
            tool = self._tools[tool_idx]
            monetary_costs.append(tool.cost)
            return tool.query(question, candidates)

        result = self._agent.solve_question(
            question_text=question,
            candidates=candidates,
            category_hint=category_hint,
            tool_query_fn=tool_query_fn,
        )

        wall_time = time.monotonic() - t_start

        # Store for explain_last_decision
        self._last_trace = result.decision_trace
        self._last_tools_queried = result.tools_queried

        # Build answer
        monetary_cost = sum(monetary_costs)
        effective_cost = sum(self._tool_configs[t_idx].cost for t_idx in result.tools_queried)
        tools_used = tuple(self._tools[t_idx].name for t_idx in result.tools_queried)
        choice_text = candidates[result.answer] if result.answer is not None else None

        reasoning = self._format_reasoning(result.decision_trace)
        trace_dicts = tuple(
            {
                "step": s.step,
                "eu_submit": s.eu_submit,
                "eu_abstain": s.eu_abstain,
                "eu_query": {self._tools[k].name: v for k, v in s.eu_query.items()},
                "chosen": s.chosen_action,
            }
            for s in result.decision_trace
        )

        answer_posterior = tuple(self._agent.answer_posterior)

        tool_responses = tuple(
            (t_idx, self._agent.tool_responses.get(t_idx)) for t_idx in result.tools_queried
        )

        return Answer(
            choice=result.answer,
            choice_text=choice_text,
            confidence=result.confidence,
            tools_used=tools_used,
            monetary_cost=monetary_cost,
            effective_cost=effective_cost,
            wall_time=wall_time,
            reasoning=reasoning,
            answer_posterior=answer_posterior,
            tool_responses=tool_responses,
            decision_trace=trace_dicts,
        )

    @property
    def scoring(self) -> ScoringRule:
        """The scoring rule used by this Router."""
        return self._scoring

    def refresh_tool_coverage(self, tool_idx: int) -> None:
        """Re-read tool.coverage() and update cached ToolConfig for a tool."""
        tool = self._tools[tool_idx]
        old = self._tool_configs[tool_idx]
        self._tool_configs[tool_idx] = ToolConfig(
            cost=old.cost,
            coverage_by_category=tool.coverage(self._categories),
        )

    def save_state_dict(self) -> dict:
        """Return learned state as a plain dict (for embedding in larger state files).

        Extracts per-tool per-category reliability and coverage means from Julia state.
        """
        bridge = self._bridge
        # Extract approximate reliability table from Julia MixtureMeasure state
        reliability_means = []
        for t_idx in range(len(self._tools)):
            means = bridge.extract_reliability_means(self._agent.rel_states[t_idx])
            reliability_means.append(means)

        # Extract approximate coverage table from Julia MixtureMeasure state
        coverage_means = []
        for t_idx in range(len(self._tools)):
            means = bridge.extract_reliability_means(self._agent.cov_states[t_idx])
            coverage_means.append(means)

        return {
            "reliability_means": reliability_means,
            "coverage_means": coverage_means,
            "tool_names": [t.name for t in self._tools],
            "categories": list(self._categories),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore learned state from a dict.

        Reconstructs Julia rel_states and cov_states from saved means using
        tight Beta priors. For backward compat, legacy coverage_alpha/coverage_beta
        are converted to means if coverage_means is absent.
        """
        bridge = self._bridge
        if "reliability_means" in state:
            for t_idx, means in enumerate(state["reliability_means"]):
                self._agent.rel_states[t_idx] = bridge.make_oracle_rel_state(means)

        # Prefer coverage_means; fall back to legacy alpha/beta
        coverage_means = state.get("coverage_means")
        if coverage_means is None:
            cov_alpha = state.get("coverage_alpha", {})
            cov_beta = state.get("coverage_beta", {})
            if cov_alpha:
                name_to_idx = {t.name: i for i, t in enumerate(self._tools)}
                coverage_means = [None] * len(self._tools)
                for name in cov_alpha:
                    idx = name_to_idx.get(name)
                    if idx is not None:
                        a = cov_alpha[name]
                        b = cov_beta[name]
                        coverage_means[idx] = [ai / (ai + bi) for ai, bi in zip(a, b)]

        if coverage_means:
            for t_idx, means in enumerate(coverage_means):
                if means is not None:
                    self._agent.cov_states[t_idx] = bridge.make_oracle_rel_state(means)

    def report_outcome(self, correct: bool) -> None:
        """Report whether the last answer was correct. Updates reliability table."""
        self._agent.on_question_end(correct)

    def explain_last_decision(self) -> str:
        """Human-readable VOI trace from the last solve() call."""
        return self._format_reasoning(self._last_trace)

    @property
    def learned_reliability(self) -> dict[str, dict[str, float]]:
        """Current learned reliability per tool per category.

        Returns e.g. {"web_search": {"factual": 0.83, "reasoning": 0.31}}.
        """
        bridge = self._bridge
        result: dict[str, dict[str, float]] = {}
        for t_idx, tool in enumerate(self._tools):
            means = bridge.extract_reliability_means(self._agent.rel_states[t_idx])
            per_cat: dict[str, float] = {}
            for c_idx, cat in enumerate(self._categories):
                per_cat[cat] = means[c_idx]
            result[tool.name] = per_cat
        return result

    def save_state(self, path: str | Path) -> None:
        """Persist learned state (reliability + coverage) to disk."""
        Path(path).write_text(json.dumps(self.save_state_dict()))

    def load_state(self, path: str | Path) -> None:
        """Restore learned state (reliability + coverage) from disk."""
        state = json.loads(Path(path).read_text())
        self.load_state_dict(state)

    def _format_reasoning(self, trace: tuple[DecisionStep, ...]) -> str:
        """Format decision trace as human-readable text."""
        lines: list[str] = []
        for step in trace:
            lines.append(f"Step {step.step}:")
            lines.append(f"  EU(submit)={step.eu_submit:.4f}  EU(abstain)={step.eu_abstain:.4f}")
            for t_idx, net_voi in sorted(step.eu_query.items()):
                name = self._tools[t_idx].name
                lines.append(f"  VOI({name})={net_voi:+.4f}")
            lines.append(f"  -> {step.chosen_action}")
        return "\n".join(lines)
