# Role: body
"""Tests for benchmark infrastructure."""

from __future__ import annotations

from bayesian_if.benchmark import (
    BenchmarkResult,
    GameRecord,
    GameSpec,
    OracleBaseline,
    RandomBaseline,
    LookOnlyBaseline,
    compare_results,
)
from tests.mock_world import MockWorld


# ---------------------------------------------------------------------------
# Metrics computation (no TextWorld needed)
# ---------------------------------------------------------------------------

def _make_spec(**kwargs) -> GameSpec:
    defaults = dict(path="/tmp/test.z8", world_size=1, nb_objects=2, quest_length=1, seed=42)
    defaults.update(kwargs)
    return GameSpec(**defaults)


def test_normalized_score():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=5, max_score=10, steps_taken=3,
                   game_won=False, tools_queried_per_step=0.0),
        GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=5,
                   game_won=True, tools_queried_per_step=1.5),
    ])
    assert br.normalized_scores == [0.5, 1.0]
    assert br.mean_normalized_score == 0.75


def test_win_rate():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=5,
                   game_won=True, tools_queried_per_step=0.0),
        GameRecord(_make_spec(), final_score=0, max_score=10, steps_taken=10,
                   game_won=False, tools_queried_per_step=0.0),
        GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=7,
                   game_won=True, tools_queried_per_step=0.0),
    ])
    assert abs(br.win_rate - 2.0 / 3.0) < 1e-9


def test_mean_steps_to_win():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=5,
                   game_won=True, tools_queried_per_step=0.0),
        GameRecord(_make_spec(), final_score=0, max_score=10, steps_taken=100,
                   game_won=False, tools_queried_per_step=0.0),
        GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=3,
                   game_won=True, tools_queried_per_step=0.0),
    ])
    assert br.mean_steps_to_win == 4.0  # (5 + 3) / 2


def test_mean_steps_to_win_no_wins():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=0, max_score=10, steps_taken=100,
                   game_won=False, tools_queried_per_step=0.0),
    ])
    assert br.mean_steps_to_win == float("inf")


def test_tool_calls_per_step():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=0, max_score=10, steps_taken=10,
                   game_won=False, tools_queried_per_step=2.0),
        GameRecord(_make_spec(), final_score=0, max_score=10, steps_taken=10,
                   game_won=False, tools_queried_per_step=4.0),
    ])
    assert br.tool_calls_per_step == 3.0


def test_empty_result():
    br = BenchmarkResult()
    assert br.mean_normalized_score == 0.0
    assert br.win_rate == 0.0
    assert br.tool_calls_per_step == 0.0


def test_zero_max_score():
    br = BenchmarkResult(game_records=[
        GameRecord(_make_spec(), final_score=0, max_score=0, steps_taken=5,
                   game_won=False, tools_queried_per_step=0.0),
    ])
    assert br.normalized_scores == [0.0]


def test_compare_results_formatting():
    results = {
        "random": BenchmarkResult(game_records=[
            GameRecord(_make_spec(), final_score=2, max_score=10, steps_taken=50,
                       game_won=False, tools_queried_per_step=0.0),
        ]),
        "bayesian": BenchmarkResult(game_records=[
            GameRecord(_make_spec(), final_score=10, max_score=10, steps_taken=8,
                       game_won=True, tools_queried_per_step=2.5),
        ]),
    }
    table = compare_results(results)
    assert "random" in table
    assert "bayesian" in table
    assert "NormScore" in table


# ---------------------------------------------------------------------------
# Baselines on MockWorld (no TextWorld needed)
# ---------------------------------------------------------------------------

def test_random_baseline_runs():
    world = MockWorld()
    baseline = RandomBaseline(world, seed=42)
    result = baseline.play_game(max_steps=20)
    assert result.steps_taken > 0
    assert result.final_score >= 0


def test_look_only_baseline_runs():
    world = MockWorld()
    baseline = LookOnlyBaseline(world)
    result = baseline.play_game(max_steps=20)
    assert result.steps_taken > 0
    assert result.final_score >= 0


def test_oracle_baseline_with_mock_policy():
    """Oracle follows policy_commands to win."""
    world = MockWorld()
    # Patch MockWorld to have policy_commands (the optimal solution)
    world.policy_commands = ["take key", "go north", "go north", "open chest"]  # type: ignore[attr-defined]
    baseline = OracleBaseline(world)
    result = baseline.play_game(max_steps=20)
    assert result.final_score == 17  # 5 (key) + 1 + 1 (explore) + 10 (chest)
