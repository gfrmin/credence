# Role: body
"""Environment layer: simulated tools, question bank, and benchmark harness."""

from credence_agents.environment.tools import (
    ResponseType,
    ToolResponse,
    SimulatedTool,
    query_tool,
    make_spec_tools,
    tool_config_for,
)
from credence_agents.environment.questions import (
    Question,
    QUESTION_BANK,
    get_questions,
)
from credence_agents.environment.benchmark import (
    Agent,
    QuestionRecord,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    "ResponseType",
    "ToolResponse",
    "SimulatedTool",
    "query_tool",
    "make_spec_tools",
    "tool_config_for",
    "Question",
    "QUESTION_BANK",
    "get_questions",
    "Agent",
    "QuestionRecord",
    "BenchmarkResult",
    "run_benchmark",
]
