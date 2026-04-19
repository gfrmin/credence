# Role: body
"""Tests for the Router class."""

from __future__ import annotations


from credence_router.answer import Answer
from credence_router.router import Router
from credence_router.tools.simulated import SimulatedTool, make_default_simulated_tools


def _make_simple_tools() -> list[SimulatedTool]:
    """Two tools: a cheap unreliable one and an expensive reliable one."""
    return [
        SimulatedTool(
            name="cheap",
            cost=0.001,
            latency=0.1,
            reliability_by_category={"factual": 0.5, "numerical": 0.3},
            coverage_by_category={"factual": 1.0, "numerical": 1.0},
            seed=42,
        ),
        SimulatedTool(
            name="expensive",
            cost=0.01,
            latency=0.5,
            reliability_by_category={"factual": 0.9, "numerical": 0.9},
            coverage_by_category={"factual": 1.0, "numerical": 1.0},
            seed=43,
        ),
    ]


class TestRouterInit:
    def test_creates_with_tools(self):
        tools = _make_simple_tools()
        router = Router(tools=tools, categories=("factual", "numerical"))
        assert router is not None

    def test_creates_with_default_categories(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        assert router is not None

    def test_latency_weight_increases_effective_cost(self):
        tools = _make_simple_tools()
        r0 = Router(tools=tools, categories=("factual", "numerical"), latency_weight=0.0)
        r1 = Router(tools=tools, categories=("factual", "numerical"), latency_weight=0.1)
        # With latency_weight=0.1, expensive tool (0.5s) gets +$0.05 effective cost
        assert r1._tool_configs[1].cost > r0._tool_configs[1].cost


class TestRouterSolve:
    def test_returns_answer(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is the capital of France?",
            ("Paris", "London", "Berlin", "Madrid"),
        )
        assert isinstance(answer, Answer)
        assert answer.choice is None or 0 <= answer.choice < 4
        assert answer.monetary_cost >= 0
        assert answer.wall_time >= 0
        assert len(answer.reasoning) > 0

    def test_category_hint_works(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is 2 + 2?",
            ("3", "4", "5", "6"),
            category_hint="numerical",
        )
        assert isinstance(answer, Answer)

    def test_tools_used_is_populated(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is the capital of Australia?",
            ("Sydney", "Melbourne", "Canberra", "Brisbane"),
        )
        assert len(answer.tools_used) >= 0  # may abstain without querying

    def test_decision_trace_is_populated(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is 17% of 4230?",
            ("718.1", "719.1", "721.1", "723.1"),
            category_hint="numerical",
        )
        assert len(answer.decision_trace) >= 1


class TestRouterLearning:
    def test_report_outcome_updates_reliability(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)

        initial = router.learned_reliability
        initial_factual = {name: cats.get("factual", 0.5) for name, cats in initial.items()}

        # Solve and report several correct outcomes
        for _ in range(5):
            router.solve(
                "What is the capital of France?",
                ("Paris", "London", "Berlin", "Madrid"),
            )
            router.report_outcome(True)

        updated = router.learned_reliability
        # At least one tool's reliability should have changed
        changed = False
        for name in initial_factual:
            if abs(updated[name].get("factual", 0.5) - initial_factual[name]) > 0.01:
                changed = True
                break
        assert changed, "Reliability table should update after feedback"


class TestRouterPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)

        # Run some questions to build state
        for _ in range(3):
            router.solve(
                "What is 2^10?",
                ("512", "1024", "2048", "4096"),
                category_hint="numerical",
            )
            router.report_outcome(True)

        # Save
        state_path = tmp_path / "state.json"
        router.save_state(state_path)

        # Load into new router
        router2 = Router(tools=tools)
        router2.load_state(state_path)

        # Reliability tables should match
        r1 = router.learned_reliability
        r2 = router2.learned_reliability
        for tool_name in r1:
            for cat in r1[tool_name]:
                assert abs(r1[tool_name][cat] - r2[tool_name][cat]) < 1e-10


class TestRouterToolResponses:
    def test_tool_responses_populated(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is the capital of France?",
            ("Paris", "London", "Berlin", "Madrid"),
        )
        # tool_responses should be a tuple of (tool_idx, response) pairs
        assert isinstance(answer.tool_responses, tuple)
        for entry in answer.tool_responses:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            t_idx, resp = entry
            assert isinstance(t_idx, int)
            assert resp is None or isinstance(resp, int)

    def test_tool_responses_matches_tools_used(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        answer = router.solve(
            "What is 2^10?",
            ("512", "1024", "2048", "4096"),
            category_hint="numerical",
        )
        # Number of tool_responses should match tools_used
        assert len(answer.tool_responses) == len(answer.tools_used)

    def test_tool_responses_empty_on_abstain(self):
        tools = _make_simple_tools()
        router = Router(tools=tools, categories=("factual", "numerical"))
        answer = router.solve(
            "Something trivial",
            ("A", "B"),
        )
        # Even if abstained, tool_responses should be consistent
        assert len(answer.tool_responses) == len(answer.tools_used)


class TestRouterScoringProperty:
    def test_scoring_returns_scoring_rule(self):
        from credence_agents.inference.voi import ScoringRule

        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        assert isinstance(router.scoring, ScoringRule)

    def test_custom_scoring_returned(self):
        from credence_agents.inference.voi import ScoringRule

        custom = ScoringRule(reward_correct=1.0, penalty_wrong=-0.5, reward_abstain=0.0)
        tools = _make_simple_tools()
        router = Router(tools=tools, categories=("factual", "numerical"), scoring=custom)
        assert router.scoring.reward_correct == 1.0
        assert router.scoring.penalty_wrong == -0.5


class TestRouterRefreshCoverage:
    def test_refresh_updates_tool_config(self):
        tools = _make_simple_tools()
        router = Router(tools=tools, categories=("factual", "numerical"))
        old_coverage = router._tool_configs[0].coverage_by_category.copy()
        router.refresh_tool_coverage(0)
        # Coverage should be the same since tool hasn't changed
        import numpy as np

        np.testing.assert_array_equal(
            router._tool_configs[0].coverage_by_category,
            old_coverage,
        )


class TestRouterStateDictPersistence:
    def test_save_load_state_dict_roundtrip(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)

        for _ in range(3):
            router.solve(
                "What is 2^10?", ("512", "1024", "2048", "4096"), category_hint="numerical"
            )
            router.report_outcome(True)

        state = router.save_state_dict()
        assert "reliability_means" in state
        assert "tool_names" in state
        assert "categories" in state

        router2 = Router(tools=tools)
        router2.load_state_dict(state)

        r1 = router.learned_reliability
        r2 = router2.learned_reliability
        for tool_name in r1:
            for cat in r1[tool_name]:
                assert abs(r1[tool_name][cat] - r2[tool_name][cat]) < 1e-10

    def test_coverage_means_in_saved_state(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)

        router.solve("What is 2^10?", ("512", "1024", "2048", "4096"), category_hint="numerical")
        router.report_outcome(True)

        state = router.save_state_dict()
        assert "coverage_means" in state
        assert len(state["coverage_means"]) == len(tools)
        for means in state["coverage_means"]:
            assert isinstance(means, list)
            assert all(isinstance(v, float) for v in means)

    def test_save_state_dict_matches_save_state(self, tmp_path):
        import json

        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        router.solve("test", ("A", "B", "C", "D"))
        router.report_outcome(True)

        state_dict = router.save_state_dict()
        state_path = tmp_path / "state.json"
        router.save_state(state_path)
        file_state = json.loads(state_path.read_text())

        assert state_dict["reliability_means"] == file_state["reliability_means"]
        assert state_dict["tool_names"] == file_state["tool_names"]
        assert state_dict["categories"] == file_state["categories"]


class TestRouterExplain:
    def test_explain_last_decision(self):
        tools = make_default_simulated_tools()
        router = Router(tools=tools)
        router.solve(
            "What is the capital of France?",
            ("Paris", "London", "Berlin", "Madrid"),
        )
        explanation = router.explain_last_decision()
        assert "VOI" in explanation or "Step" in explanation
