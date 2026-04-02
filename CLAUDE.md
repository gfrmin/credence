# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Bayesian decision-theoretic Interactive Fiction agent that uses the `credence` library
for information-gathering decisions. The agent uses VOI (Value of Information) to decide
which sources to consult (look, examine, inventory, LLM) before committing to an action.

## Development Commands

```bash
# Install (editable, with dev deps)
uv sync --all-groups

# Run tests
pytest tests/
pytest tests/test_agent.py                # single file
pytest tests/test_agent.py::test_name -v  # single test

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Run against a Z-machine game
bayesian-if --game path/to/game.z5 --max-steps 100 --verbose

# Run against TextWorld (procedural)
bayesian-if --textworld --max-steps 50

# Disable LLM advisor (run with just look/examine/inventory tools)
bayesian-if --game path/to/game.z5 --no-llm
```

## Architecture

- `BayesianAgent` from credence is the **information-gathering controller** — it decides
  which sources to consult, not which action to take directly.
- Each game step is a fresh "question" — answer posteriors reset per step, but the
  reliability table persists across steps, learning which sources work in which situations.
- Score deltas provide ground truth: `delta > 0` → correct, `delta < 0` → wrong, `delta == 0` → no update.
- Info tools use save/restore to peek at game state without consuming a turn.

**Per-step data flow:** observation → VOI tool selection → tool queries via save/restore → action → reward → reliability update.

**Key modules:**
- `agent.py` — `IFAgent` wraps credence's `BayesianAgent`, orchestrates play loop
- `tools.py` — `IFTool` base class + LookTool, ExamineTool, InventoryTool, LLMAdvisorTool; each defines per-category coverage probabilities
- `categories.py` — keyword-based classifier into 5 situation types (exploration, puzzle, inventory, dialogue, combat)
- `world.py` — `World` protocol (structural subtyping); implementations in `jericho_world.py` and `textworld_world.py`
- `ollama.py` — Ollama HTTP client for LLM advisor queries
- `reward.py` — maps score deltas to correctness signals for reliability updates
- `play.py` — CLI entry point (`bayesian-if` command)

**Testing:** `tests/mock_world.py` provides a 3-room deterministic game (key→chest objective) used by all tests.

## Design Principles

Same as credence: everything is EU maximisation, no hacks, LLM outputs are data.

## Dependencies

- `credence-agents` — Bayesian inference layer ([PyPI](https://pypi.org/project/credence-agents/))
- `numpy` — numerical computation
- `jericho` — Z-machine IF interpreter (optional)
- `textworld` — Procedural IF environments (optional)
- `httpx` — Ollama HTTP client (optional, for LLM advisor tool)

Ruff: line-length = 99.
