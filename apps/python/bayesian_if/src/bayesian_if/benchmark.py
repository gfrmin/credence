"""TextWorld benchmark suite: game generation, runner, baselines, and metrics."""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameSpec:
    """Specification for a single benchmark game."""

    path: str
    world_size: int
    nb_objects: int
    quest_length: int
    seed: int


@dataclass
class GameRecord:
    """Per-game benchmark results."""

    game_spec: GameSpec
    final_score: int
    max_score: int
    steps_taken: int
    game_won: bool
    tools_queried_per_step: float


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results across a game suite."""

    game_records: list[GameRecord] = field(default_factory=list)

    @property
    def normalized_scores(self) -> list[float]:
        return [
            r.final_score / r.max_score if r.max_score > 0 else 0.0
            for r in self.game_records
        ]

    @property
    def mean_normalized_score(self) -> float:
        scores = self.normalized_scores
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def win_rate(self) -> float:
        if not self.game_records:
            return 0.0
        return sum(1 for r in self.game_records if r.game_won) / len(self.game_records)

    @property
    def mean_steps_to_win(self) -> float:
        won = [r for r in self.game_records if r.game_won]
        return sum(r.steps_taken for r in won) / len(won) if won else float("inf")

    @property
    def tool_calls_per_step(self) -> float:
        if not self.game_records:
            return 0.0
        return sum(r.tools_queried_per_step for r in self.game_records) / len(
            self.game_records
        )


# ---------------------------------------------------------------------------
# Game suite generation
# ---------------------------------------------------------------------------

# Difficulty grid: world_size x nb_objects x quest_length
# Exclude clearly impossible configs (tiny world + long quest).
_DIFFICULTY_CONFIGS: list[tuple[int, int, int]] = [
    (ws, no, ql)
    for ws in [1, 3, 5]
    for no in [2, 5]
    for ql in [1, 3, 5]
    if not (ws == 1 and ql == 5)
]


def generate_game_suite(output_dir: str, seed: int = 42) -> list[GameSpec]:
    """Generate a suite of TextWorld games for benchmarking.

    Calls ``tw-make custom`` with a grid of difficulty configs.
    Writes ``.z8`` files and a ``manifest.json`` recording params per game.
    """
    os.makedirs(output_dir, exist_ok=True)
    games: list[GameSpec] = []

    for i, (ws, no, ql) in enumerate(_DIFFICULTY_CONFIGS):
        game_seed = seed + i
        filename = f"tw_ws{ws}_no{no}_ql{ql}_s{game_seed}.z8"
        path = os.path.join(output_dir, filename)

        subprocess.run(
            [
                "tw-make", "custom",
                "--world-size", str(ws),
                "--nb-objects", str(no),
                "--quest-length", str(ql),
                "--seed", str(game_seed),
                "--output", path,
            ],
            check=True,
            capture_output=True,
        )

        games.append(
            GameSpec(
                path=path, world_size=ws, nb_objects=no,
                quest_length=ql, seed=game_seed,
            )
        )

    manifest = [asdict(g) for g in games]
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return games


def load_game_suite(output_dir: str) -> list[GameSpec]:
    """Load a previously generated game suite from its manifest."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path) as f:
        entries = json.load(f)
    return [GameSpec(**e) for e in entries]


# ---------------------------------------------------------------------------
# Player protocol
# ---------------------------------------------------------------------------

class Player(Protocol):
    """Anything with a play_game method."""

    def play_game(self, max_steps: int = 100) -> Any: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@dataclass
class _SimpleResult:
    """Minimal result for baselines (compatible with run_benchmark)."""

    final_score: int
    steps_taken: int
    steps: list[Any] = field(default_factory=list)
    reliability_table: Any = None


class RandomBaseline:
    """Uniform random action selection."""

    def __init__(self, world: Any, seed: int = 0) -> None:
        self.world = world
        self.rng = random.Random(seed)

    def play_game(self, max_steps: int = 100) -> _SimpleResult:
        obs = self.world.reset()
        steps_taken = 0
        done = False
        for _ in range(max_steps):
            actions = self.world.valid_actions()
            if not actions or done:
                break
            action = self.rng.choice(actions)
            obs, _, done = self.world.step(action)
            steps_taken += 1
        return _SimpleResult(final_score=obs.score, steps_taken=steps_taken)


class LookOnlyBaseline:
    """Always consult LookTool, follow its recommendation."""

    def __init__(self, world: Any) -> None:
        self.world = world

    def play_game(self, max_steps: int = 100) -> _SimpleResult:
        from bayesian_if.tools import LookTool

        look = LookTool()
        obs = self.world.reset()
        steps_taken = 0
        done = False
        for _ in range(max_steps):
            actions = self.world.valid_actions()
            if not actions or done:
                break
            idx = look.query(self.world, obs, actions)
            action = actions[idx] if idx is not None else actions[0]
            obs, _, done = self.world.step(action)
            steps_taken += 1
        return _SimpleResult(final_score=obs.score, steps_taken=steps_taken)


class OracleBaseline:
    """Follow TextWorld's policy_commands for a perfect playthrough (upper bound)."""

    def __init__(self, world: Any) -> None:
        self.world = world

    def play_game(self, max_steps: int = 100) -> _SimpleResult:
        obs = self.world.reset()
        commands = list(getattr(self.world, "policy_commands", []))
        steps_taken = 0
        done = False
        cmd_idx = 0
        for _ in range(max_steps):
            if done or cmd_idx >= len(commands):
                break
            actions = self.world.valid_actions()
            cmd = commands[cmd_idx]
            if cmd in actions:
                obs, _, done = self.world.step(cmd)
            else:
                obs, _, done = self.world.step(cmd)
            cmd_idx += 1
            steps_taken += 1
        return _SimpleResult(final_score=obs.score, steps_taken=steps_taken)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    agent_factory: Callable[[Any], Player],
    game_suite: list[GameSpec],
    max_steps: int = 100,
) -> BenchmarkResult:
    """Run a player across every game in the suite, collecting metrics."""
    from bayesian_if.textworld_world import TextWorldWorld

    records: list[GameRecord] = []
    for spec in game_suite:
        world = TextWorldWorld(spec.path)
        agent = agent_factory(world)
        result = agent.play_game(max_steps=max_steps)

        max_score = getattr(world, "max_score", 0) or 1
        game_won = getattr(world, "game_won", False)

        steps = getattr(result, "steps", [])
        tools_per_step = (
            sum(len(s.tools_queried) for s in steps) / len(steps)
            if steps
            else 0.0
        )

        records.append(
            GameRecord(
                game_spec=spec,
                final_score=result.final_score,
                max_score=max_score,
                steps_taken=result.steps_taken,
                game_won=game_won,
                tools_queried_per_step=tools_per_step,
            )
        )

    return BenchmarkResult(game_records=records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compare_results(results: dict[str, BenchmarkResult]) -> str:
    """Format a comparison table of benchmark results."""
    header = f"{'Agent':<20} {'NormScore':>10} {'WinRate':>10} {'StepsToWin':>12} {'Tools/Step':>12}"
    sep = "-" * len(header)
    lines = [header, sep]

    for name, br in results.items():
        steps_str = (
            f"{br.mean_steps_to_win:.1f}"
            if br.mean_steps_to_win < float("inf")
            else "N/A"
        )
        lines.append(
            f"{name:<20} {br.mean_normalized_score:>10.3f} "
            f"{br.win_rate:>10.1%} {steps_str:>12} "
            f"{br.tool_calls_per_step:>12.2f}"
        )

    return "\n".join(lines)
