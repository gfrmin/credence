# Role: body
"""IFAgent — composes BayesianAgent with IF domain logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from credence_agents import BayesianAgent, ScoringRule
from credence_agents.julia_bridge import CredenceBridge

from bayesian_if.categories import CATEGORIES, infer_category_hint, make_if_category_infer_fn
from bayesian_if.reward import attribute_reward
from bayesian_if.tools import DEFAULT_TOOLS, IFTool, LLMAdvisorTool
from bayesian_if.world import Observation, World

# Initial prior — adapted online from observed score deltas.
IF_SCORING = ScoringRule(reward_correct=1.0, penalty_wrong=-0.5, reward_abstain=-0.05)


@dataclass
class StepRecord:
    """Record of a single game step."""

    step: int
    observation_text: str
    valid_actions: list[str]
    chosen_action: str
    tools_queried: tuple[int, ...]
    confidence: float
    reward: float
    cumulative_score: int
    tool_recommendations: dict[int, int | None] = field(default_factory=dict)
    category_hint: str | None = None
    was_correct: bool | None = None


@dataclass
class GameResult:
    """Result of playing a full game."""

    final_score: int
    steps_taken: int
    steps: list[StepRecord] = field(default_factory=list)
    reliability_means: list[list[float]] | None = None


class IFAgent:
    """Bayesian decision-theoretic IF agent.

    Uses BayesianAgent as the information-gathering controller: each game step,
    VOI decides which info sources to consult before committing to an action.
    The reliability table persists across steps, learning which sources work
    in which situations.
    """

    def __init__(
        self,
        world: World,
        tools: list[IFTool] | None = None,
        categories: tuple[str, ...] = CATEGORIES,
        category_infer_fn: Callable[[str], list[float]] | None = None,
        scoring: ScoringRule = IF_SCORING,
        forgetting: float = 0.85,
        verbose: bool = False,
        bridge: CredenceBridge | None = None,
    ) -> None:
        self.world = world
        self.if_tools = tools if tools is not None else list(DEFAULT_TOOLS)
        self.categories = categories
        self.scoring = scoring
        self.verbose = verbose
        self._history: list[tuple[str, str]] = []
        self._failed_actions: dict[str | None, set[str]] = {}

        self._bridge = bridge or CredenceBridge()

        if category_infer_fn is None:
            category_infer_fn = make_if_category_infer_fn(categories)
        self._category_infer_fn = category_infer_fn

        tool_configs = [t.to_tool_config(categories) for t in self.if_tools]

        self.bayesian = BayesianAgent(
            bridge=self._bridge,
            tool_configs=tool_configs,
            categories=categories,
            forgetting=forgetting,
            scoring=self.scoring,
        )

        # Warm-start LLM reliability to r_eff=0.7 so it has nonzero VOI
        for i, tool in enumerate(self.if_tools):
            if isinstance(tool, LLMAdvisorTool):
                self.bayesian.rel_states[i] = self._bridge.make_warm_rel_state(
                    self.bayesian._num_categories, alpha=7.0, beta=3.0
                )

        # EMA trackers for online scoring rule adaptation
        self._ema_reward: float = 1.0
        self._ema_penalty: float = -0.5

    def play_step(self, observation: Observation) -> tuple[str, StepRecord]:
        """One game step: gather info via VOI, choose action, return action string."""
        valid_actions = self.world.valid_actions()

        if not valid_actions:
            return "look", StepRecord(
                step=0,
                observation_text=observation.text,
                valid_actions=[],
                chosen_action="look",
                tools_queried=(),
                confidence=0.0,
                reward=0.0,
                cumulative_score=observation.score,
            )

        # Filter out actions that failed at this location
        location = observation.location
        failed_here = self._failed_actions.get(location, set())
        effective = [a for a in valid_actions if a not in failed_here] or valid_actions

        # If only one action remains, skip VOI — nothing to decide
        if len(effective) == 1:
            chosen = effective[0]
            return chosen, StepRecord(
                step=0,
                observation_text=observation.text,
                valid_actions=valid_actions,
                chosen_action=chosen,
                tools_queried=(),
                confidence=1.0,
                reward=0.0,
                cumulative_score=observation.score,
            )

        # Build the tool query function that BayesianAgent will call
        recent_history = self._history or None
        tool_recommendations: dict[int, int | None] = {}

        def tool_query_fn(tool_idx: int) -> int | None:
            result = self.if_tools[tool_idx].query(
                self.world,
                observation,
                effective,
                history=recent_history,
                failed_actions=failed_here,
            )
            tool_recommendations[tool_idx] = result
            return result

        # Infer category hint from structured state
        category_hint = infer_category_hint(observation)

        result = self.bayesian.solve_question(
            question_text=observation.text,
            candidates=tuple(effective),
            category_hint=category_hint,
            tool_query_fn=tool_query_fn,
        )

        if result.answer is not None and result.answer < len(effective):
            if _is_uniform_posterior(result.confidence, len(effective)):
                chosen_action = _exploration_tiebreak(
                    effective,
                    failed_here,
                    self._history,
                )
            else:
                chosen_action = effective[result.answer]
        else:
            # Abstain → take a safe action
            chosen_action = _safe_action(effective, failed=failed_here)

        record = StepRecord(
            step=0,  # filled in by play_game
            observation_text=observation.text,
            valid_actions=valid_actions,
            chosen_action=chosen_action,
            tools_queried=result.tools_queried,
            confidence=result.confidence,
            reward=0.0,  # filled in after step
            cumulative_score=observation.score,
            tool_recommendations=tool_recommendations,
            category_hint=category_hint,
        )

        return chosen_action, record

    def play_game(self, max_steps: int = 100) -> GameResult:
        """Play a full game, returning trace and final score."""
        obs = self.world.reset()
        self._history = []
        self._failed_actions = {}
        steps: list[StepRecord] = []

        for step_num in range(1, max_steps + 1):
            action, record = self.play_step(obs)
            prev_obs = obs
            obs, reward, done = self.world.step(action)

            # Track history for LLM context
            self._history.append((action, obs.text[:100]))

            # Update failed-action memory
            if reward > 0:
                self._failed_actions.pop(prev_obs.location, None)
            elif reward <= 0 and obs.score == prev_obs.score and obs.intermediate_reward <= 0:
                self._failed_actions.setdefault(prev_obs.location, set()).add(action)

            # Adapt scoring rule from observed score deltas
            score_delta = obs.score - prev_obs.score
            if score_delta > 0:
                self._ema_reward = 0.7 * self._ema_reward + 0.3 * score_delta
            elif score_delta < 0:
                self._ema_penalty = 0.7 * self._ema_penalty + 0.3 * score_delta

            self.bayesian.scoring = ScoringRule(
                reward_correct=max(self._ema_reward, 0.1),
                penalty_wrong=min(self._ema_penalty, -0.1),
                reward_abstain=min(-0.05 * abs(self._ema_reward), -0.01),
            )

            # Attribute reward for reliability learning
            was_correct = attribute_reward(reward, prev_obs, obs)
            self.bayesian.on_question_end(was_correct)

            record.step = step_num
            record.reward = reward
            record.cumulative_score = obs.score
            record.was_correct = was_correct
            steps.append(record)

            if self.verbose:
                tools_str = ", ".join(self.if_tools[t].name for t in record.tools_queried)
                print(
                    f"[Step {step_num}] "
                    f"Action: {action!r}  "
                    f"Tools: [{tools_str}]  "
                    f"Confidence: {record.confidence:.2f}  "
                    f"Reward: {reward:+.0f}  "
                    f"Score: {obs.score}"
                )

            if done:
                break

        return GameResult(
            final_score=obs.score,
            steps_taken=len(steps),
            steps=steps,
            reliability_means=self._get_reliability_means(),
        )

    def _get_reliability_means(self) -> list[list[float]]:
        return [self._bridge.extract_reliability_means(rs) for rs in self.bayesian.rel_states]


def _is_uniform_posterior(confidence: float, n_candidates: int, tol: float = 1e-6) -> bool:
    """True when the posterior is effectively uniform (no tool moved the needle)."""
    return abs(confidence - 1.0 / n_candidates) < tol


def _exploration_tiebreak(
    effective: list[str],
    failed_here: set[str],
    history: list[tuple[str, str]] | None,
) -> str:
    """Break ties among EU-equal actions: prefer novel movement, then novel interactions."""
    import random

    tried = {act for act, _ in history} if history else set()

    # Untried movement commands (highest exploration value)
    untried_moves = [
        a for a in effective if a.startswith("go ") and a not in tried and a not in failed_here
    ]
    if untried_moves:
        return random.choice(untried_moves)

    # Untried object-interaction commands (examine, take, open, etc.)
    interaction_verbs = (
        "examine",
        "take",
        "open",
        "unlock",
        "use",
        "push",
        "pull",
        "eat",
        "drink",
    )
    untried_interact = [
        a
        for a in effective
        if any(a.startswith(v) for v in interaction_verbs)
        and a not in tried
        and a not in failed_here
    ]
    if untried_interact:
        return random.choice(untried_interact)

    # Any untried action
    untried = [a for a in effective if a not in tried and a not in failed_here]
    if untried:
        return random.choice(untried)

    # All tried — random from effective
    return random.choice(effective)


def _safe_action(valid_actions: list[str], failed: set[str] | None = None) -> str:
    """Pick a safe fallback action when the agent abstains."""
    import random

    failed = failed or set()
    # Prefer untried movement — highest option value when local actions failed
    moves = [a for a in valid_actions if a.startswith("go ") and a not in failed]
    if moves:
        return random.choice(moves)
    # Then standard safe verbs, excluding failed
    for verb in ("look", "inventory", "wait"):
        for action in valid_actions:
            if action.lower().startswith(verb) and action not in failed:
                return action
    # Absolute fallback
    unfailed = [a for a in valid_actions if a not in failed]
    return random.choice(unfailed) if unfailed else random.choice(valid_actions)
