# Role: body
"""CLI entry points: bench and route subcommands."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _make_real_tools() -> list | None:
    """Build real tool list if API keys are available. Returns None if none available."""
    from credence_router.tools.calculator import CalculatorTool

    tools = [CalculatorTool()]
    has_api = False

    if os.environ.get("ANTHROPIC_API_KEY"):
        from credence_router.tools.claude import ClaudeTool

        tools.append(ClaudeTool("haiku"))
        has_api = True

    if os.environ.get("PERPLEXITY_API_KEY"):
        from credence_router.tools.perplexity import PerplexityTool

        tools.append(PerplexityTool())
        has_api = True

    return tools if has_api else None


def cmd_bench(args: argparse.Namespace) -> None:
    """Run the benchmark comparing routing strategies."""
    from credence_router.analysis import (
        format_learning_curve,
        format_reliability_table,
    )
    from credence_router.baselines.simple import (
        AlwaysBestSolver,
        AlwaysCheapestSolver,
        RandomSolver,
    )
    from credence_router.benchmark import (
        BenchmarkResult,
        format_comparison_table,
        run_benchmark,
    )
    from credence_router.questions import get_questions
    from credence_router.router import DEFAULT_SCORING, Router
    from credence_router.tools.simulated import make_default_simulated_tools

    seed = args.seed
    latency_weight = args.latency_weight
    questions = get_questions(seed=seed)

    # Create simulated tools with ground truth from question bank
    sim_tools = make_default_simulated_tools(seed=seed, questions=questions)

    # Use real tools for credence-router if --live; baselines always use simulated
    if args.live:
        live_tools = _make_real_tools()
        if live_tools is None:
            print(
                "Error: --live requires ANTHROPIC_API_KEY or PERPLEXITY_API_KEY",
                file=sys.stderr,
            )
            sys.exit(1)
        cr_tools = live_tools
    else:
        cr_tools = sim_tools

    all_results: list[BenchmarkResult] = []

    if args.run:
        mode = "LIVE" if args.live else "simulated"
        print(f"Running benchmark: {len(questions)} questions, seed={seed} ({mode})")
        print(f"Latency weight: ${latency_weight}/sec")
        print()

        # credence-router
        router = Router(
            tools=cr_tools,
            scoring=DEFAULT_SCORING,
            latency_weight=latency_weight,
        )
        cr_result = run_benchmark(router, questions)
        all_results.append(cr_result)

        # Baselines
        for solver_cls, solver_args in [
            (AlwaysCheapestSolver, {"tools": sim_tools}),
            (AlwaysBestSolver, {"tools": sim_tools}),
            (RandomSolver, {"tools": sim_tools, "seed": seed}),
        ]:
            solver = solver_cls(**solver_args)
            result = run_benchmark(solver, questions)
            all_results.append(result)

        # LangGraph baseline (simulated)
        if not args.no_langgraph:
            from credence_router.baselines.langgraph_react import LangGraphReActSolver

            lg_solver = LangGraphReActSolver(tools=sim_tools)
            lg_result = run_benchmark(lg_solver, questions)
            all_results.append(lg_result)

        print(format_comparison_table(all_results))
        print()

    if args.explain:
        router = Router(
            tools=cr_tools,
            scoring=DEFAULT_SCORING,
            latency_weight=latency_weight,
        )
        indices = [int(x) - 1 for x in args.explain.split(",")]
        for idx in indices:
            if 0 <= idx < len(questions):
                q = questions[idx]
                answer = router.solve(q.text, q.candidates, q.category)
                router.report_outcome(answer.choice == q.correct_index)
                print(f"\nQ{idx + 1}: {q.text}")
                print(f"Category: {q.category}, Correct: {q.candidates[q.correct_index]}")
                print(answer.reasoning)
                if answer.choice is not None:
                    mark = "CORRECT" if answer.choice == q.correct_index else "WRONG"
                    print(f"Submitted: {answer.choice_text} — {mark}")
                else:
                    print("ABSTAINED")
                print(f"Cost: ${answer.monetary_cost:.4f}")

    if args.show_learning and all_results:
        print(format_learning_curve(all_results))

    if args.show_reliability:
        router = Router(
            tools=cr_tools,
            scoring=DEFAULT_SCORING,
            latency_weight=latency_weight,
        )
        # Run all questions to build reliability
        for q in questions:
            answer = router.solve(q.text, q.candidates, q.category)
            router.report_outcome(answer.choice == q.correct_index)
        print("\nLearned reliability table:")
        print(format_reliability_table(router.learned_reliability))


def _build_search_tools(providers: list[str] | None = None) -> list:
    """Build search tool list from available providers."""
    from credence_router.tool import SearchTool

    if providers is None:
        providers = ["duckduckgo", "brave", "perplexity", "tavily"]

    search_tools: list[SearchTool] = []

    # DuckDuckGo: always available (no API key)
    if "duckduckgo" in providers:
        from credence_router.tools.web.duckduckgo import DuckDuckGoSearchTool
        search_tools.append(DuckDuckGoSearchTool())
    if "brave" in providers and os.environ.get("BRAVE_API_KEY"):
        from credence_router.tools.web.brave import BraveSearchTool
        search_tools.append(BraveSearchTool())
    if "perplexity" in providers and os.environ.get("PERPLEXITY_API_KEY"):
        from credence_router.tools.web.perplexity import PerplexitySearchTool
        search_tools.append(PerplexitySearchTool())
    if "tavily" in providers and os.environ.get("TAVILY_API_KEY"):
        from credence_router.tools.web.tavily import TavilySearchTool
        search_tools.append(TavilySearchTool())

    return search_tools


def cmd_eval_search(args: argparse.Namespace) -> None:
    """Run the search provider evaluation benchmark."""
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from credence_router.benchmarks.search_eval import (
        QUERY_BANK,
        RawEvalData,
        collect_raw_data,
        format_category_table,
        format_frontier_table,
        format_provider_table,
        simulate_routing,
        sweep_preferences,
    )

    # Analyse mode: read cached data, sweep preferences
    if args.analyse:
        logging.getLogger(__name__).info("Loading cached data from %s", args.analyse)
        raw_data = RawEvalData.load(args.analyse)
        seeds = list(range(args.seeds))
        frontier = sweep_preferences(raw_data, seeds=seeds)
        print(format_frontier_table(frontier))
        return

    # Collect mode: need real search tools
    providers = args.providers.split(",") if args.providers else None
    search_tools = _build_search_tools(providers)

    if len(search_tools) < 2:
        print(
            "Error: need at least 2 search providers for routing evaluation.\n"
            "DuckDuckGo is always available (no key). Also set one of:\n"
            "  BRAVE_API_KEY, PERPLEXITY_API_KEY, TAVILY_API_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY required for LLM judge.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = args.output or "eval_raw_data.json"

    logging.getLogger(__name__).info(
        "Providers: %s, Queries: %d", [t.name for t in search_tools], len(QUERY_BANK),
    )

    # Phase 1: Collect
    raw_data = collect_raw_data(
        search_tools, verbose=not args.quiet, partial_save_path=output_path,
    )

    output_path = args.output or "eval_raw_data.json"
    raw_data.save(output_path)
    print(f"\nRaw data saved to {output_path}")

    if args.collect_only:
        print("Done (collect only). Re-run with --analyse to explore preferences.")
        return

    # Phase 2: Analyse
    print("\nPhase 2: Sweeping preference frontier...")
    seeds = list(range(args.seeds))
    frontier = sweep_preferences(raw_data, seeds=seeds)
    print(format_frontier_table(frontier))

    # Also show detailed breakdown for the balanced point
    print("\n" + "=" * 60)
    print("Detailed breakdown (balanced: reward=0.25, latency_weight=0.05)")
    print("=" * 60 + "\n")
    balanced = simulate_routing(raw_data, reward=0.25, latency_weight=0.05, seed=42)
    print(format_category_table(balanced))
    print()
    print(format_provider_table(balanced))


def cmd_eval_llm(args: argparse.Namespace) -> None:
    """Evaluate LLM model routing."""
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    from credence_router.benchmarks.llm_eval import (
        LLM_QUERY_BANK,
        collect_llm_data,
    )
    from credence_router.tools.llm.provider import available_models

    models = available_models()
    if not models:
        print("Error: no LLM API keys found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY required for LLM judge.", file=sys.stderr)
        sys.exit(1)

    model_names = [m.name for m in models]
    logging.getLogger(__name__).info("Models: %s, Queries: %d", model_names, len(LLM_QUERY_BANK))

    raw_data = collect_llm_data(
        model_names, verbose=not args.quiet, partial_save_path=args.output,
    )
    raw_data.save(args.output)
    logging.getLogger(__name__).info("Saved to %s", args.output)

    # Summary
    print(f"\n{'Model':<35s} {'Quality':>8s} {'Cost$':>8s} {'Latency':>8s}")
    print("-" * 65)
    for model in model_names:
        scores = [
            qd.model_results[model].scores.get("composite", 0.0)
            for qd in raw_data.queries if model in qd.model_results
        ]
        costs = [qd.model_results[model].cost_usd for qd in raw_data.queries if model in qd.model_results]
        latencies = [qd.model_results[model].total_seconds for qd in raw_data.queries if model in qd.model_results]
        n = len(scores)
        if n > 0:
            print(
                f"{model:<35s} {sum(scores)/n:>7.2f} "
                f"${sum(costs):>6.3f} "
                f"{sum(latencies)/n:>6.1f}s"
            )


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the HTTP server."""
    from credence_router.server import serve
    serve(host=args.host, port=args.port)


def cmd_route(args: argparse.Namespace) -> None:
    """Route a single question interactively."""
    from credence_router.router import Router

    if args.simulate:
        from credence_router.tools.simulated import make_default_simulated_tools

        tools = make_default_simulated_tools()
        print("(using simulated tools)", file=sys.stderr)
    else:
        real_tools = _make_real_tools()
        if real_tools is not None:
            tools = real_tools
        else:
            from credence_router.tools.simulated import make_default_simulated_tools

            tools = make_default_simulated_tools()
            print(
                "Warning: no API keys found (ANTHROPIC_API_KEY, PERPLEXITY_API_KEY). "
                "Using simulated tools. Pass --simulate to suppress this warning.",
                file=sys.stderr,
            )

    router = Router(tools=tools, latency_weight=args.latency_weight)

    candidates = tuple(args.options)
    answer = router.solve(args.question, candidates)

    print(f"Question: {args.question}")
    print(f"Options: {candidates}")
    print()
    print(answer.reasoning)
    print()
    if answer.choice is not None:
        print(f"Answer: {answer.choice_text} (confidence: {answer.confidence:.2f})")
    else:
        print("ABSTAINED")
    print(f"Tools used: {answer.tools_used}")
    print(f"Cost: ${answer.monetary_cost:.4f} (effective: ${answer.effective_cost:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="credence-router",
        description="Transparent, cost-optimal tool routing via EU maximisation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # bench subcommand
    bench = subparsers.add_parser("bench", help="Run benchmark")
    bench.add_argument("--run", action="store_true", help="Run the benchmark")
    bench.add_argument("--simulate", action="store_true", help="Use simulated tools only")
    bench.add_argument("--explain", type=str, help="Explain questions (comma-separated indices)")
    bench.add_argument("--show-learning", action="store_true", help="Show learning curves")
    bench.add_argument("--show-reliability", action="store_true", help="Show reliability table")
    bench.add_argument("--latency-weight", type=float, default=0.01, help="$/second")
    bench.add_argument("--seed", type=int, default=42, help="Random seed")
    bench.add_argument("--no-langgraph", action="store_true", help="Skip LangGraph baseline")
    bench.add_argument("--live", action="store_true", help="Use real API tools (needs API keys)")

    # route subcommand
    route = subparsers.add_parser("route", help="Route a single question")
    route.add_argument("question", help="The question to route")
    route.add_argument("-o", "--options", nargs="+", required=True, help="Answer options")
    route.add_argument("--latency-weight", type=float, default=0.01, help="$/second")
    route.add_argument(
        "--simulate", action="store_true", help="Force simulated tools (no API calls)"
    )

    # eval-search subcommand
    eval_search = subparsers.add_parser("eval-search", help="Evaluate search provider routing")
    eval_search.add_argument("--seeds", type=int, default=3, help="Number of seeds for preference sweep")
    eval_search.add_argument("--providers", type=str, default=None, help="Comma-separated providers")
    eval_search.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    eval_search.add_argument("--collect-only", action="store_true", help="Only collect data, don't analyse")
    eval_search.add_argument("--analyse", type=str, default=None, help="Analyse cached JSON data file")
    eval_search.add_argument("--output", type=str, default=None, help="Output path for raw data JSON")
    eval_search.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    # eval-llm subcommand
    eval_llm = subparsers.add_parser("eval-llm", help="Evaluate LLM model routing")
    eval_llm.add_argument("--output", type=str, default="llm_eval_data.json", help="Output path")
    eval_llm.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    eval_llm.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    # serve subcommand
    serve = subparsers.add_parser("serve", help="Start HTTP server")
    serve.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    serve.add_argument("--port", type=int, default=8377, help="Bind port")

    args = parser.parse_args()
    if args.command == "bench":
        cmd_bench(args)
    elif args.command == "route":
        cmd_route(args)
    elif args.command == "eval-search":
        cmd_eval_search(args)
    elif args.command == "eval-llm":
        cmd_eval_llm(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
