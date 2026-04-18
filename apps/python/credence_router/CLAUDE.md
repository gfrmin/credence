# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

credence-router: transparent, cost-optimal tool routing via EU maximisation.
Replaces LangChain/LangGraph's opaque LLM-based routing with principled VOI
calculations. Zero routing cost, <1ms routing latency, learns from feedback.

## Development Commands

```bash
# From monorepo root (credence/):
uv sync                                                    # Install workspace
uv run pytest apps/python/credence_router/tests/                # Run all tests
uv run pytest apps/python/credence_router/tests/test_router.py  # Run one test file
ruff check apps/python/credence_router/                         # Lint
ruff format apps/python/credence_router/                        # Format
```

```bash
# Simulated benchmark (no API keys needed)
credence-router bench --run --simulate

# Full benchmark (needs ANTHROPIC_API_KEY, PERPLEXITY_API_KEY)
credence-router bench --run

# Interactive routing
credence-router route "What is 2+2?" -o "3" "4" "5" "6"
```

## Architecture

All source lives under `src/credence_router/`. The core loop is:

1. **Router.solve()** receives a question + candidates
2. Router delegates to **BayesianAgent** (from `credence-agents`) which runs a VOI loop
3. Each iteration: compute EU(submit), EU(abstain), net-VOI for each tool. Query the tool with highest positive net-VOI, or submit/abstain if no tool is worth querying
4. Tool responses update the posterior via Bayesian updating
5. Returns a frozen **Answer** dataclass with choice, confidence, costs, and full reasoning trace

Key dependency: `credence-agents` is a workspace sibling in the credence monorepo (configured in `[tool.uv.sources]` as `{ workspace = true }`). It provides `BayesianAgent`, `ScoringRule`, `ToolConfig`, `DecisionStep`, and the category inference system.

### Tool protocol

`Tool` is a `@runtime_checkable Protocol` with: `name`, `cost` ($/query), `latency` (seconds), `query(question, candidates) -> int | None`, and `coverage(categories) -> NDArray`. Tools return a candidate index or None (can't answer). Coverage is P(returns an answer | category).

### LangGraph compatibility layer (`compat/`)

Drop-in replacement for `langgraph.prebuilt.create_react_agent`. Builds a real LangGraph StateGraph but swaps the LLM routing node with VOI-based selection. The LLM is still used for argument extraction and answer synthesis — only the routing decision is replaced. Access the underlying router via `graph._credence_router`.

## Design Principles

Everything is EU maximisation, no hacks, LLM outputs are data. Prefer functional programming. Ruff: line-length = 99.
