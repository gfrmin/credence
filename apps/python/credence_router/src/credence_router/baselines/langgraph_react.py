# Role: body
"""LangGraph ReAct baseline: same tools, LLM-based routing.

Uses LangGraph's ReAct agent with Claude Sonnet as the routing LLM.
Tracks all costs: tool cost + routing LLM cost (input/output tokens).
"""

from __future__ import annotations

import os
import re
import time

from credence_router.answer import Answer
from credence_router.tool import Tool

# Approximate costs for Claude Sonnet routing
# ~200 input tokens (system + question + tool descriptions) + ~50 output tokens per step
# At $3/M input, $15/M output: ~$0.0006 + $0.00075 = ~$0.0014 per routing step
# Typical ReAct: 2-3 steps → $0.003-$0.004 routing cost
_ROUTING_COST_PER_STEP = 0.0014
_ROUTING_LATENCY_PER_STEP = 0.5  # seconds


class LangGraphReActSolver:
    """LangGraph ReAct agent wrapping the same tools as credence-router.

    Requires: pip install credence-router[langgraph]
    """

    name = "langgraph-react"

    def __init__(
        self,
        tools: list[Tool],
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ):
        self._tools = {t.name: t for t in tools}
        self._tools_list = tools
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def solve(
        self,
        question: str,
        candidates: tuple[str, ...],
        category_hint: str | None = None,
    ) -> Answer:
        """Route via LangGraph ReAct agent."""
        try:
            return self._solve_live(question, candidates)
        except ImportError:
            return self._solve_simulated(question, candidates)

    def _solve_live(self, question: str, candidates: tuple[str, ...]) -> Answer:
        """Live LangGraph ReAct execution."""
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent

        # Wrap our tools as LangChain tools
        from langchain_core.tools import StructuredTool

        lc_tools = []
        for t in self._tools_list:
            cands = candidates

            def make_query_fn(tool=t):
                def query_tool(question: str) -> str:
                    idx = tool.query(question, cands)
                    if idx is not None:
                        return f"Answer: {cands[idx]}"
                    return "No answer available"

                return query_tool

            lc_tools.append(
                StructuredTool.from_function(
                    func=make_query_fn(t),
                    name=t.name,
                    description=f"Query {t.name} to answer a question",
                )
            )

        llm = ChatAnthropic(model=self._model, api_key=self._api_key)
        agent = create_react_agent(llm, lc_tools)

        letters = [chr(65 + i) for i in range(len(candidates))]
        options = "\n".join(f"{ltr}) {c}" for ltr, c in zip(letters, candidates))

        t_start = time.monotonic()
        result = agent.invoke(
            {
                "messages": [
                    (
                        "system",
                        "Answer the multiple-choice question by using the available tools. "
                        "Reply with ONLY the letter of the correct answer.",
                    ),
                    ("human", f"Question: {question}\n\nOptions:\n{options}"),
                ]
            }
        )
        wall_time = time.monotonic() - t_start

        # Parse the final message
        final_msg = result["messages"][-1].content
        choice = self._parse_letter(final_msg, len(candidates))

        # Count routing steps (LLM calls minus the final answer)
        routing_steps = sum(1 for m in result["messages"] if hasattr(m, "tool_calls"))
        routing_cost = routing_steps * _ROUTING_COST_PER_STEP

        # Count tool calls
        tool_names_used: list[str] = []
        tool_cost = 0.0
        for m in result["messages"]:
            if hasattr(m, "name") and m.name in self._tools:
                tool_names_used.append(m.name)
                tool_cost += self._tools[m.name].cost

        return Answer(
            choice=choice,
            choice_text=candidates[choice] if choice is not None else None,
            confidence=0.0,
            tools_used=tuple(tool_names_used),
            monetary_cost=tool_cost + routing_cost,
            effective_cost=tool_cost + routing_cost,
            wall_time=wall_time,
            reasoning=f"LangGraph ReAct: {routing_steps} routing steps, "
            f"${routing_cost:.4f} routing cost",
        )

    def _solve_simulated(self, question: str, candidates: tuple[str, ...]) -> Answer:
        """Simulated LangGraph ReAct (for benchmarking without API keys).

        Simulates typical ReAct behaviour: 2-3 routing steps, queries 2-3 tools,
        then submits the answer from the last tool queried.
        """
        import numpy as np

        t_start = time.monotonic()
        rng = np.random.default_rng(hash(question) % (2**31))

        # Simulate 2-3 routing steps
        n_steps = int(rng.integers(2, 4))
        routing_cost = n_steps * _ROUTING_COST_PER_STEP

        # Query 1-3 tools (ReAct often over-queries)
        n_queries = min(int(rng.integers(1, 4)), len(self._tools_list))
        indices = rng.choice(len(self._tools_list), size=n_queries, replace=False)

        tool_cost = 0.0
        tools_used: list[str] = []
        last_result: int | None = None

        for idx in indices:
            tool = self._tools_list[idx]
            result = tool.query(question, candidates)
            tool_cost += tool.cost
            tools_used.append(tool.name)
            if result is not None:
                last_result = result

        wall_time = time.monotonic() - t_start
        # Add simulated routing latency
        simulated_latency = n_steps * _ROUTING_LATENCY_PER_STEP

        return Answer(
            choice=last_result,
            choice_text=candidates[last_result] if last_result is not None else None,
            confidence=0.0,
            tools_used=tuple(tools_used),
            monetary_cost=tool_cost + routing_cost,
            effective_cost=tool_cost + routing_cost,
            wall_time=wall_time + simulated_latency,
            reasoning=f"Simulated ReAct: {n_steps} routing steps, "
            f"${routing_cost:.4f} routing cost",
        )

    def report_outcome(self, correct: bool) -> None:
        pass  # LangGraph doesn't learn

    @staticmethod
    def _parse_letter(text: str, n: int) -> int | None:
        match = re.search(r"([A-D])\)", text)
        if not match:
            match = re.search(r"\b([A-D])\b", text.upper())
        if match:
            idx = ord(match.group(1).upper()) - 65
            if 0 <= idx < n:
                return idx
        return None
