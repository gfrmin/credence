"""CLI entry point for TextWorld benchmarks."""

from __future__ import annotations

import argparse
import os

from bayesian_if.agent import IFAgent
from bayesian_if.benchmark import (
    BenchmarkResult,
    LookOnlyBaseline,
    OracleBaseline,
    RandomBaseline,
    compare_results,
    generate_game_suite,
    load_game_suite,
    run_benchmark,
)
from bayesian_if.tools import DEFAULT_TOOLS

_DEFAULT_SUITE_DIR = os.path.join("benchmarks", "games")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian IF benchmark suite")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate game suite in benchmarks/games/",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run all agents (bayesian + baselines) on suite",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print comparison table from last run",
    )
    parser.add_argument(
        "--suite-dir", type=str, default=_DEFAULT_SUITE_DIR,
        help="Directory for game suite",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Max steps per game",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for suite gen")
    args = parser.parse_args(argv)

    if args.generate:
        print(f"Generating game suite in {args.suite_dir} ...")
        games = generate_game_suite(args.suite_dir, seed=args.seed)
        print(f"Generated {len(games)} games.")
        for g in games:
            print(f"  ws={g.world_size} no={g.nb_objects} ql={g.quest_length} → {g.path}")

    if args.run:
        suite = load_game_suite(args.suite_dir)
        print(f"Running benchmark on {len(suite)} games (max_steps={args.max_steps})")

        all_results: dict[str, BenchmarkResult] = {}

        # Baselines
        print("\n[Random baseline]")
        all_results["random"] = run_benchmark(
            lambda w: RandomBaseline(w), suite, max_steps=args.max_steps,
        )

        print("[LookOnly baseline]")
        all_results["look_only"] = run_benchmark(
            lambda w: LookOnlyBaseline(w), suite, max_steps=args.max_steps,
        )

        print("[Oracle baseline]")
        all_results["oracle"] = run_benchmark(
            lambda w: OracleBaseline(w), suite, max_steps=args.max_steps,
        )

        # Bayesian agent
        print("[Bayesian agent]")
        all_results["bayesian"] = run_benchmark(
            lambda w: IFAgent(world=w, tools=list(DEFAULT_TOOLS)),
            suite,
            max_steps=args.max_steps,
        )

        # Print report
        print("\n" + compare_results(all_results))

        # Save results
        results_path = os.path.join(args.suite_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write(compare_results(all_results) + "\n")
        print(f"\nResults saved to {results_path}")

    if args.report:
        results_path = os.path.join(args.suite_dir, "results.txt")
        if os.path.exists(results_path):
            with open(results_path) as f:
                print(f.read())
        else:
            print(f"No results found at {results_path}. Run --run first.")


if __name__ == "__main__":
    main()
