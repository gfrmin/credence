"""Benchmark harness: runs agents through the question bank and records results.

Manages the question loop, enforces protocol invariants (no re-querying, bounded
tool calls), and computes scoring per SPEC §2.3.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

from credence_agents.inference.decision import Action, ActionType
from credence_agents.inference.voi import PENALTY_WRONG, REWARD_ABSTAIN, REWARD_CORRECT
from credence_agents.environment.questions import Question
from credence_agents.environment.tools import SimulatedTool, query_tool, ResponseType


@runtime_checkable
class Agent(Protocol):
    name: str

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None: ...

    def choose_action(self) -> Action: ...

    def on_tool_response(self, tool_idx: int, response: int | None) -> None: ...

    def on_question_end(self, was_correct: bool | None) -> None: ...


class QuestionRecord(NamedTuple):
    question_id: str
    category: str
    difficulty: str
    tools_queried: tuple[int, ...]
    tool_responses: dict[int, int | None]
    action_type: str                      # "submit" | "abstain"
    submitted_answer: int | None
    was_correct: bool | None
    reward: float
    tool_cost: float
    belief_snapshot: dict | None


class BenchmarkResult(NamedTuple):
    agent_name: str
    seed: int
    records: tuple[QuestionRecord, ...]
    total_score: float
    total_tool_cost: float
    total_reward: float
    wall_time_s: float = 0.0


def run_benchmark(
    agent: Agent,
    tools: Sequence[SimulatedTool],
    questions: Sequence[Question],
    seed: int = 42,
) -> BenchmarkResult:
    """Run an agent through the question bank and return scored results."""
    rng = np.random.default_rng(seed)
    num_tools = len(tools)
    records: list[QuestionRecord] = []
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time.perf_counter()

    for question in questions:
        agent.on_question_start(question.id, question.candidates, num_tools,
                                question_text=question.text)

        used_tools: list[int] = []
        tool_responses: dict[int, int | None] = {}
        question_tool_cost = 0.0

        while True:
            action = agent.choose_action()

            if action.action_type == ActionType.QUERY:
                tool_idx = action.tool_idx
                assert tool_idx is not None

                if tool_idx in tool_responses:
                    raise RuntimeError(
                        f"Agent '{agent.name}' queried tool {tool_idx} twice "
                        f"on question '{question.id}'"
                    )
                if len(used_tools) >= num_tools:
                    raise RuntimeError(
                        f"Agent '{agent.name}' exceeded max tool queries ({num_tools}) "
                        f"on question '{question.id}'"
                    )

                resp = query_tool(tools[tool_idx], question, rng)
                candidate_idx = resp.candidate_idx if resp.response_type == ResponseType.ANSWER else None
                question_tool_cost += tools[tool_idx].cost
                used_tools.append(tool_idx)
                tool_responses[tool_idx] = candidate_idx
                agent.on_tool_response(tool_idx, candidate_idx)

            elif action.action_type == ActionType.SUBMIT:
                submitted = action.answer_idx
                was_correct = submitted == question.correct_index
                reward = REWARD_CORRECT if was_correct else PENALTY_WRONG
                total_reward += reward
                total_tool_cost += question_tool_cost

                belief_snapshot = getattr(agent, "get_belief_snapshot", lambda: None)()

                agent.on_question_end(was_correct)

                records.append(QuestionRecord(
                    question_id=question.id,
                    category=question.category,
                    difficulty=question.difficulty,
                    tools_queried=tuple(used_tools),
                    tool_responses=dict(tool_responses),
                    action_type="submit",
                    submitted_answer=submitted,
                    was_correct=was_correct,
                    reward=reward,
                    tool_cost=question_tool_cost,
                    belief_snapshot=belief_snapshot,
                ))
                break

            elif action.action_type == ActionType.ABSTAIN:
                total_reward += REWARD_ABSTAIN
                total_tool_cost += question_tool_cost

                belief_snapshot = getattr(agent, "get_belief_snapshot", lambda: None)()

                agent.on_question_end(None)

                records.append(QuestionRecord(
                    question_id=question.id,
                    category=question.category,
                    difficulty=question.difficulty,
                    tools_queried=tuple(used_tools),
                    tool_responses=dict(tool_responses),
                    action_type="abstain",
                    submitted_answer=None,
                    was_correct=None,
                    reward=REWARD_ABSTAIN,
                    tool_cost=question_tool_cost,
                    belief_snapshot=belief_snapshot,
                ))
                break

    wall_time_s = time.perf_counter() - t_start

    return BenchmarkResult(
        agent_name=agent.name,
        seed=seed,
        records=tuple(records),
        total_score=total_reward - total_tool_cost,
        total_tool_cost=total_tool_cost,
        total_reward=total_reward,
        wall_time_s=wall_time_s,
    )
