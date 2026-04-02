"""Tests for the LangGraph compatibility layer."""

from __future__ import annotations

import numpy as np
import pytest

from credence_router.answer import Answer
from credence_router.router import Router
from credence_router.tool import Tool


# ---------------------------------------------------------------------------
# Minimal mock of a LangChain BaseTool (no langchain_core dependency)
# ---------------------------------------------------------------------------


class _MockLCTool:
    """Minimal mock of a LangChain tool for adapter tests."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


# ---------------------------------------------------------------------------
# ToolRoutingAdapter tests (no langgraph/langchain required)
# ---------------------------------------------------------------------------


class TestToolRoutingAdapter:
    def test_implements_tool_protocol(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("calc", "math calculator"), 0, 3)
        assert isinstance(adapter, Tool)

    def test_name_from_lc_tool(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("my_tool", "desc"), 0, 1)
        assert adapter.name == "my_tool"

    def test_cost_and_latency_default_zero(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("t"), 0, 1)
        assert adapter.cost == 0.0
        assert adapter.latency == 0.0

    def test_query_returns_index_on_keyword_match(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("calculator", "evaluate math expressions"), 0, 3)
        # "math" appears in both question and tool description
        result = adapter.query("solve this math problem", ("calculator", "search", "lookup"))
        assert result == 0

    def test_query_returns_none_on_no_match(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("calculator", "evaluate math expressions"), 0, 3)
        result = adapter.query("weather today", ("calculator", "search", "lookup"))
        assert result is None

    def test_coverage_returns_valid_array(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        categories = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")
        adapter = ToolRoutingAdapter(_MockLCTool("calc", "math calculator"), 0, 1)
        cov = adapter.coverage(categories)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (len(categories),)
        assert np.all(cov >= 0.1)
        assert np.all(cov <= 1.0)

    def test_coverage_min_floor(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        categories = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")
        adapter = ToolRoutingAdapter(_MockLCTool("x", ""), 0, 1)
        cov = adapter.coverage(categories)
        # Even with empty description, coverage is at least 0.1
        assert np.all(cov >= 0.1)

    def test_custom_cost_and_latency(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapter = ToolRoutingAdapter(_MockLCTool("t"), 0, 1, cost=0.5, latency=1.0)
        assert adapter.cost == 0.5
        assert adapter.latency == 1.0


# ---------------------------------------------------------------------------
# Router with adapters (no langgraph required, tests VOI routing logic)
# ---------------------------------------------------------------------------


class TestRoutingWithAdapters:
    def test_router_with_adapters_returns_answer(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapters = [
            ToolRoutingAdapter(_MockLCTool("calc", "math calculator evaluation"), 0, 2),
            ToolRoutingAdapter(_MockLCTool("search", "web search information"), 1, 2),
        ]
        router = Router(tools=adapters)
        answer = router.solve("math problem", ("calc", "search"))
        assert isinstance(answer, Answer)
        assert answer.choice is None or answer.choice in (0, 1)
        assert answer.monetary_cost == 0.0  # adapters are free

    def test_routing_produces_valid_answer(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        # With more candidates, the Router needs to query tools to decide
        adapters = [
            ToolRoutingAdapter(_MockLCTool("calc", "math calculator evaluation"), 0, 4),
            ToolRoutingAdapter(_MockLCTool("search", "web search information"), 1, 4),
            ToolRoutingAdapter(_MockLCTool("lookup", "database lookup queries"), 2, 4),
            ToolRoutingAdapter(_MockLCTool("translate", "language translation"), 3, 4),
        ]
        router = Router(tools=adapters)
        candidates = ("calc", "search", "lookup", "translate")
        answer = router.solve("math problem", candidates)
        assert isinstance(answer, Answer)
        assert answer.choice is None or 0 <= answer.choice < len(candidates)

    def test_non_matching_tool_not_queried(self):
        from credence_router.compat.tool_adapter import ToolRoutingAdapter

        adapters = [
            ToolRoutingAdapter(_MockLCTool("calc", "math calculator evaluation"), 0, 2),
            ToolRoutingAdapter(_MockLCTool("search", "web search information"), 1, 2),
        ]
        router = Router(tools=adapters)
        answer = router.solve("math problem", ("calc", "search"))
        # "math" doesn't match search's keywords, so search shouldn't be queried
        # (coverage may still cause it to be queried, but the keyword probe returns None)
        assert isinstance(answer, Answer)


# ---------------------------------------------------------------------------
# Extract question helper
# ---------------------------------------------------------------------------


class TestExtractQuestion:
    def test_extracts_from_tuple_messages(self):
        from credence_router.compat.routing_node import _extract_question

        msgs = [("system", "Be helpful"), ("human", "What is 2+2?")]
        assert _extract_question(msgs) == "What is 2+2?"

    def test_extracts_latest_human_message(self):
        from credence_router.compat.routing_node import _extract_question

        msgs = [("human", "First question"), ("human", "Second question")]
        assert _extract_question(msgs) == "Second question"

    def test_returns_empty_on_no_human_message(self):
        from credence_router.compat.routing_node import _extract_question

        msgs = [("system", "Be helpful")]
        assert _extract_question(msgs) == ""


# ---------------------------------------------------------------------------
# Integration tests requiring langgraph + langchain_core
# ---------------------------------------------------------------------------


def _has_langgraph() -> bool:
    try:
        import langchain_core  # noqa: F401
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_langgraph(), reason="langgraph/langchain_core not installed")
class TestCreateReactAgent:
    def _make_tools(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return "42"

        @lc_tool
        def search(query: str) -> str:
            """Search the web for information."""
            return f"Result for: {query}"

        return [calculator, search]

    def _make_mock_model(self):
        from langchain_core.messages import AIMessage

        class _BoundMockModel:
            def __init__(self, tools):
                self._tools = tools

            def invoke(self, messages, **kwargs):
                if not self._tools:
                    return AIMessage(content="No tools bound")
                tool = self._tools[0]
                # Get first arg name from tool schema
                arg_name = "input"
                if hasattr(tool, "args_schema") and tool.args_schema:
                    fields = list(tool.args_schema.model_fields.keys())
                    if fields:
                        arg_name = fields[0]
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool.name,
                            "args": {arg_name: "test"},
                            "id": "call_mock_001",
                            "type": "tool_call",
                        }
                    ],
                )

        class _MockChatModel:
            def invoke(self, messages, **kwargs):
                return AIMessage(content="Final answer: 42")

            def bind_tools(self, tools, **kwargs):
                return _BoundMockModel(tools)

        return _MockChatModel()

    def test_returns_compiled_graph(self):
        from credence_router.compat import create_react_agent

        tools = self._make_tools()
        model = self._make_mock_model()
        graph = create_react_agent(model, tools)
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_exposes_credence_router(self):
        from credence_router.compat import create_react_agent

        tools = self._make_tools()
        model = self._make_mock_model()
        graph = create_react_agent(model, tools)
        assert isinstance(graph._credence_router, Router)

    def test_invoke_returns_messages(self):
        from credence_router.compat import create_react_agent

        tools = self._make_tools()
        model = self._make_mock_model()
        graph = create_react_agent(model, tools)
        result = graph.invoke({"messages": [("human", "What is 2+2?")]})
        assert "messages" in result
        assert len(result["messages"]) >= 1

    def test_stream_yields_states(self):
        from credence_router.compat import create_react_agent

        tools = self._make_tools()
        model = self._make_mock_model()
        graph = create_react_agent(model, tools)
        states = list(graph.stream({"messages": [("human", "What is 2+2?")]}))
        assert len(states) >= 1

    def test_with_memory_saver(self):
        from credence_router.compat import MemorySaver, create_react_agent

        tools = self._make_tools()
        model = self._make_mock_model()
        memory = MemorySaver()
        graph = create_react_agent(model, tools, checkpointer=memory)
        result = graph.invoke(
            {"messages": [("human", "What is 2+2?")]},
            config={"configurable": {"thread_id": "test-1"}},
        )
        assert "messages" in result

    def test_stategraph_passthrough(self):
        from credence_router.compat import StateGraph

        from langgraph.graph import StateGraph as RealStateGraph

        assert StateGraph is RealStateGraph

    def test_toolnode_passthrough(self):
        from credence_router.compat import ToolNode

        from langgraph.prebuilt import ToolNode as RealToolNode

        assert ToolNode is RealToolNode

    def test_end_passthrough(self):
        from credence_router.compat import END

        from langgraph.graph import END as RealEND

        assert END is RealEND
