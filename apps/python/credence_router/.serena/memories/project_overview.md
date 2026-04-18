# credence-router: Project Overview

## Purpose
credence-router is a transparent, cost-optimal tool routing library that uses Expected Utility (EU) maximisation and Value of Information (VOI) calculations to select the best tool for answering multiple-choice questions. It replaces LangChain/LangGraph's opaque LLM-based routing with principled Bayesian decision theory — zero routing cost, <1ms routing latency, learns from feedback.

## Tech Stack
- **Language**: Python 3.11+
- **Build system**: Hatchling
- **Package manager**: uv
- **Core dependency**: `credence-agents` (Bayesian inference engine, local editable install from `../credence`)
- **Numerical**: numpy
- **Optional**: anthropic (Claude API), httpx (Perplexity API), langgraph + langchain-anthropic (baseline comparisons), langchain-core (compat layer)
- **Dev tools**: pytest, ruff
- **License**: AGPL-3.0-only

## Architecture
- `src/credence_router/router.py` — `Router` class wrapping credence's `BayesianAgent`, orchestrates tool selection via VOI
- `src/credence_router/tool.py` — `Tool` protocol: name, cost, latency, query(), coverage()
- `src/credence_router/answer.py` — Frozen `Answer` dataclass with choice, confidence, costs, reasoning trace
- `src/credence_router/tools/` — Built-in tools: calculator (safe AST eval), claude, perplexity, simulated
- `src/credence_router/baselines/` — Comparison agents: random, always-cheapest, always-best, langgraph ReAct
- `src/credence_router/compat/` — LangGraph drop-in compatibility layer (routing_node, prebuilt create_react_agent)
- `src/credence_router/benchmark.py` — Benchmark runner with cost + latency tracking, comparison tables
- `src/credence_router/categories.py` — 5 question categories + keyword inference
- `src/credence_router/questions.py` — 50-question evaluation bank
- `src/credence_router/cli.py` — CLI: `bench` and `route` subcommands
- `tests/` — pytest tests for router, tools, baselines, benchmark, compat, and live integration

## Public API
- `Router` — main entry point for tool routing
- `Tool` — protocol for implementing tools
- `Answer` — result dataclass
- `create_react_agent` — LangGraph-compatible wrapper

## Design Principles
Everything is EU maximisation, no hacks, LLM outputs are data. Follows same philosophy as the parent `credence` project.
