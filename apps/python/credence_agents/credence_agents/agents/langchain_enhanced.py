# Role: body
"""Enhanced LangChain agent with reliability-aware prompting (SPEC §4.2).

Same as standard but with strategy guidance and full conversation history
so the LLM can track past tool performance. This is the strongest possible
LangChain baseline.
"""

from __future__ import annotations

from credence_agents.agents.common import DecisionStep
from credence_agents.agents.langchain_agent import (
    SYSTEM_PROMPT,
    TOOL_NAMES,
    LangChainAgent,
    _action_str,
    _get_llm,
)
from credence_agents.inference.decision import Action, ActionType

# SPEC §4.2 enhanced prompt — appended to the standard system prompt
STRATEGY_PROMPT = """
STRATEGY GUIDANCE:
- Track which tools have been reliable for which types of questions.
- The calculator is perfect for maths but useless otherwise.
- Web search can return popular but incorrect answers for common misconceptions.
- If two tools disagree, consider which has been more reliable for this question type.
- The knowledge base returning "no result" means the question may be outside \
common factual territory.
- Be especially careful with questions that SEEM simple but might be tricky.
- Only say "I don't know" if you genuinely can't determine the answer with \
reasonable confidence. The threshold: if you think there's less than a 1 in 3 \
chance you're right, abstain.

TOOL SELECTION:
- Don't query tools that are unlikely to help with this question type.
- Don't cross-verify unless the first result seems uncertain.
- Use the cheapest applicable tool first.
"""

ENHANCED_SYSTEM_PROMPT = SYSTEM_PROMPT + STRATEGY_PROMPT


class LangChainEnhancedAgent(LangChainAgent):
    """Enhanced LangChain agent with strategy guidance and conversation history."""

    def __init__(self, name: str = "langchain_enhanced"):
        super().__init__(name=name)
        self._history: list[dict] = []  # past question results for context

    def _ask_llm(self) -> Action:
        """Query the LLM with enhanced prompt and conversation history."""
        llm = _get_llm()

        parts = []

        # Include recent history so LLM can track tool reliability
        if self._history:
            parts.append("PAST RESULTS (most recent):")
            for entry in self._history[-10:]:  # last 10 questions
                parts.append(
                    f"  Q: {entry['question_id']} | "
                    f"Tools: {entry['tools']} | "
                    f"Result: {entry['result']}"
                )
            parts.append("")

        parts.append(f"Question: {self._question_text}")
        candidate_lines = "\n".join(
            f"  {i}: {c}" for i, c in enumerate(self._candidates)
        )
        parts.append(f"Candidates:\n{candidate_lines}")

        if self._tool_responses:
            parts.append("\nTool results so far:")
            for t_idx, resp in self._tool_responses.items():
                tool_name = TOOL_NAMES[t_idx] if t_idx < len(TOOL_NAMES) else f"tool_{t_idx}"
                if resp is None:
                    parts.append(f"  {tool_name}: no result / not applicable")
                else:
                    parts.append(f"  {tool_name}: candidate {resp} ({self._candidates[resp]})")

        available = [i for i in range(self._num_tools) if i not in self._tool_responses]
        if available:
            tool_list = ", ".join(
                TOOL_NAMES[i] if i < len(TOOL_NAMES) else f"tool_{i}"
                for i in available
            )
            parts.append(f"\nAvailable tools (not yet queried): {tool_list}")
            parts.append(
                "\nRespond with EXACTLY one of:\n"
                "- QUERY <tool_name>\n"
                "- SUBMIT <index> (the candidate number 0-3)\n"
                "- ABSTAIN"
            )
        else:
            parts.append("\nAll tools have been queried. You must now decide.")
            parts.append(
                "\nRespond with EXACTLY one of:\n"
                "- SUBMIT <index> (the candidate number 0-3)\n"
                "- ABSTAIN"
            )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=ENHANCED_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(parts)),
        ]

        response = llm.invoke(messages)
        return self._parse_llm_response(response.content)

    def on_question_end(self, was_correct: bool | None) -> None:
        """Record question outcome for history."""
        tools_used = [
            TOOL_NAMES[i] if i < len(TOOL_NAMES) else f"tool_{i}"
            for i in self._tools_queried
        ]
        if was_correct is None:
            result_str = "abstained"
        elif was_correct:
            result_str = "correct"
        else:
            result_str = "wrong"

        self._history.append({
            "question_id": self._question_id,
            "tools": ", ".join(tools_used) if tools_used else "none",
            "result": result_str,
        })
