"""CLI entry point: play an IF game with the Bayesian agent."""

from __future__ import annotations

import argparse

from bayesian_if.agent import IFAgent
from bayesian_if.ollama import ollama_available
from bayesian_if.tools import DEFAULT_TOOLS, LLMAdvisorTool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian IF agent")
    parser.add_argument("--game", type=str, help="Path to a Z-machine ROM (.z5/.z8)")
    parser.add_argument("--textworld", action="store_true", help="Use TextWorld")
    parser.add_argument("--tw-game", type=str, help="Path to a pre-generated TextWorld game file")
    parser.add_argument("--tw-difficulty", type=int, default=3, help="TextWorld difficulty")
    parser.add_argument("--model", type=str, default="llama3.1", help="Ollama model name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM advisor tool")
    parser.add_argument("--max-steps", type=int, default=100, help="Max game steps")
    parser.add_argument("--forgetting", type=float, default=0.85, help="Forgetting factor (0-1)")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step trace")
    args = parser.parse_args(argv)

    # Create world
    if args.game:
        from bayesian_if.jericho_world import JerichoWorld

        world = JerichoWorld(args.game)
        print(f"Loaded Jericho game: {args.game}")
    elif args.textworld:
        from bayesian_if.textworld_world import TextWorldWorld

        if args.tw_game:
            game_file = args.tw_game
        else:
            import tempfile
            import subprocess

            tmpdir = tempfile.mkdtemp(prefix="tw_")
            game_file = f"{tmpdir}/tw_game.z8"
            difficulty = args.tw_difficulty
            subprocess.run(
                [
                    "tw-make",
                    "custom",
                    "--world-size",
                    str(difficulty),
                    "--nb-objects",
                    str(difficulty + 2),
                    "--quest-length",
                    str(difficulty),
                    "--seed",
                    "42",
                    "--output",
                    game_file,
                ],
                check=True,
            )
            print(f"Generated TextWorld game: {game_file}")
        world = TextWorldWorld(game_file)
        print(f"Loaded TextWorld game: {game_file}")
    else:
        parser.error("Specify --game <rom_path> or --textworld")
        return

    # Set up tools
    tools = list(DEFAULT_TOOLS)
    if not args.no_llm and ollama_available():
        tools.append(LLMAdvisorTool(model=args.model))
        print(f"LLM advisor enabled (model: {args.model})")
    elif not args.no_llm:
        print("Ollama not available — running without LLM advisor")

    # Create and run agent
    agent = IFAgent(world=world, tools=tools, forgetting=args.forgetting, verbose=args.verbose)
    result = agent.play_game(max_steps=args.max_steps)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Game finished after {result.steps_taken} steps")
    print(f"Final score: {result.final_score}")

    if result.reliability_means is not None and args.verbose:
        print("\nLearned reliability (mean per category):")
        from bayesian_if.categories import CATEGORIES

        for t_idx, tool in enumerate(tools):
            reliabilities = []
            for c_idx, cat in enumerate(CATEGORIES):
                r = result.reliability_means[t_idx][c_idx]
                reliabilities.append(f"{cat}={r:.2f}")
            print(f"  {tool.name}: {', '.join(reliabilities)}")


if __name__ == "__main__":
    main()
