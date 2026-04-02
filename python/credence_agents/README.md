# Credence

**Bayesian decision-theoretic agents — backed by the [Credence DSL](https://github.com/gfrmin/credence).**

All Bayesian inference (conditioning, expected utility, value of information) runs in
the Julia [Credence DSL](https://github.com/gfrmin/credence) via `juliacall`. Python
handles host concerns: tool queries, benchmark loop, persistence, and LangChain
comparison baselines.

The library provides domain-agnostic Bayesian agents that learn tool reliability from
experience, use VOI to decide when to query and when to abstain, and adapt when
reliability changes. No heuristics, no hardcoded routing — just expected utility
maximisation.

## Prerequisites

- Python ≥ 3.11
- Julia ≥ 1.10 (loaded automatically by `juliacall`)
- The [Credence DSL](https://github.com/gfrmin/credence) cloned at `~/git/credence/`

## Setup

```bash
# Core library (numpy + juliacall)
uv sync

# With benchmark dependencies (matplotlib, langchain, etc.)
uv sync --extra benchmark

# With dev tools (pytest, ruff, etc.)
uv sync --extra dev
```

## Using Credence as a Library

```python
import numpy as np
from credence import BayesianAgent, CredenceBridge, ToolConfig, ScoringRule

bridge = CredenceBridge()  # lazy-loads Julia on first use

categories = ("plot", "character", "setting", "theme")
tools = [
    ToolConfig(cost=1.0, coverage_by_category=np.ones(len(categories))),
    ToolConfig(cost=3.0, coverage_by_category=np.ones(len(categories))),
]

agent = BayesianAgent(
    bridge=bridge,
    tool_configs=tools,
    categories=categories,
)
```

Categories, tools, scoring rules, and the category inference function are all
injected — the agent itself is domain-agnostic.

## The Benchmark

A head-to-head comparison between a Bayesian decision-theoretic agent and LangChain
ReAct agents on a multi-tool question-answering task.

### Running experiments

```bash
# Run tests
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python -m pytest tests/ -v

# Run all experiments (stationary + drift + ablation)
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python -m experiments.run_stationary --seeds 20
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python -m experiments.run_drift --seeds 20
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python -m experiments.run_ablation --seeds 20
```

LangChain agents require a local Ollama instance with `llama3.1` (default).
Set `CREDENCE_LLM_PROVIDER=openai` or `CREDENCE_LLM_PROVIDER=anthropic` for API-based models.

## Architecture

```
Python (this repo)                    Julia (credence DSL)
┌──────────────────────┐              ┌──────────────────────────┐
│ BayesianAgent        │              │ credence_agent.bdsl      │
│   choose_action      │─juliacall──→ │   agent-step (VOI + EU)  │
│   on_tool_response   │              │   answer-kernel           │
│   on_question_end    │              │   update-on-response      │
│                      │              ├──────────────────────────┤
│ CredenceBridge       │              │ host helpers             │
│   (lazy juliacall)   │              │   update_beta_state      │
│                      │              │   marginalize_betas      │
│ Benchmark harness    │              │   initial_rel/cov_state  │
│ LangChain baselines  │              └──────────────────────────┘
└──────────────────────┘
```

```
credence/
├── julia_bridge.py          # CredenceBridge: lazy juliacall wrapper
├── inference/               # Types only (computation in Julia)
│   ├── voi.py               # ScoringRule, ToolConfig
│   └── decision.py          # ActionType, Action, feedback mapping
├── agents/
│   ├── bayesian_agent.py    # BayesianAgent (Julia DSL backend)
│   ├── baselines.py         # Random, AllTools, Oracle, SingleBest
│   ├── langchain_agent.py   # Standard LangChain ReAct
│   └── langchain_enhanced.py
├── environment/             # Benchmark: tools, questions, harness
└── analysis/                # Metrics and visualisation
```

## Design Principles

1. **Everything is expected utility maximisation** — no heuristics
2. **No hacks** — if it doesn't work, fix the model
3. **LLM outputs are data** — with quantified uncertainty
4. **The benchmark must be fair** — LangChain gets every advantage
5. **Be honest about parameters** — every number is justified

See `CLAUDE.md` for development guidelines. See `SPEC.md` for the mathematical specification.
