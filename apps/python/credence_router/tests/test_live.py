# Role: body
"""Live API smoke tests — skipped unless API keys are set.

Run manually:
    ANTHROPIC_API_KEY=... pytest tests/test_live.py -v
    PERPLEXITY_API_KEY=... pytest tests/test_live.py -v
"""

from __future__ import annotations

import os

import pytest

has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
has_perplexity_key = bool(os.environ.get("PERPLEXITY_API_KEY"))


@pytest.mark.skipif(not has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
class TestClaudeToolLive:
    def test_simple_factual(self):
        from credence_router.tools.claude import ClaudeTool

        tool = ClaudeTool("haiku")
        result = tool.query(
            "What is the capital of France?",
            ("London", "Paris", "Berlin", "Madrid"),
        )
        assert result == 1  # Paris

    def test_simple_numerical(self):
        from credence_router.tools.claude import ClaudeTool

        tool = ClaudeTool("haiku")
        result = tool.query(
            "What is 2 + 2?",
            ("3", "4", "5", "6"),
        )
        assert result == 1  # 4

    def test_returns_valid_index(self):
        from credence_router.tools.claude import ClaudeTool

        tool = ClaudeTool("haiku")
        result = tool.query(
            "Which planet is closest to the Sun?",
            ("Venus", "Mercury", "Mars", "Earth"),
        )
        assert result is None or 0 <= result < 4

    def test_cost_and_latency(self):
        from credence_router.tools.claude import ClaudeTool

        tool = ClaudeTool("haiku")
        assert tool.cost == 0.0003
        assert tool.latency == 0.200
        assert tool.name == "claude_haiku"


@pytest.mark.skipif(not has_perplexity_key, reason="PERPLEXITY_API_KEY not set")
class TestPerplexityToolLive:
    def test_simple_factual(self):
        from credence_router.tools.perplexity import PerplexityTool

        tool = PerplexityTool()
        result = tool.query(
            "What is the capital of France?",
            ("London", "Paris", "Berlin", "Madrid"),
        )
        assert result == 1  # Paris

    def test_returns_valid_index(self):
        from credence_router.tools.perplexity import PerplexityTool

        tool = PerplexityTool()
        result = tool.query(
            "What year did World War II end?",
            ("1943", "1944", "1945", "1946"),
        )
        assert result is None or 0 <= result < 4

    def test_cost_and_latency(self):
        from credence_router.tools.perplexity import PerplexityTool

        tool = PerplexityTool()
        assert tool.cost == 0.005
        assert tool.latency == 0.800
        assert tool.name == "web_search"
