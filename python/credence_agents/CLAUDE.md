# CLAUDE.md — Credence

## What This Project Is

**Credence** is a reusable Bayesian decision-theoretic agent library and an empirical
benchmark demonstrating that decision theory outperforms prompt engineering for
tool-using agents.

All Bayesian inference (conditioning, expected utility, value of information) runs in
the Julia Credence DSL via `juliacall`. Python keeps host concerns: tool queries,
benchmark loop, persistence, and LangChain comparison baselines.

The library provides domain-agnostic Bayesian agents: categories, tools, scoring rules,
and category inference are all injected — the agent has no hardcoded domain knowledge.

---

## Design Principles (NON-NEGOTIABLE)

### 1. Everything is Expected Utility Maximisation

Every decision the Bayesian agent makes — which tool to query, whether to cross-verify,
whether to submit or abstain — is:

    a* = argmax_a E[U | a, beliefs]

No special cases. No "if confused, query all tools." Just EU.

### 2. No Hacks

If the agent behaves badly, the solution is better modelling, not bolted-on fixes.

**Forbidden:**
- Exploration bonuses
- Hardcoded tool routing ("always use calculator for maths")
- Retry logic that isn't derived from VOI
- "If stuck, try all tools"
- Any rule that isn't derived from EU maximisation

**Instead:** Figure out why EU maximisation gives the wrong answer. Fix the model.

### 3. LLM Outputs Are Data

When the Bayesian agent uses an LLM (e.g. to classify question category), the output
is an observation with uncertainty, not a command. The agent maintains beliefs about
LLM reliability per category and updates from experience.

### 4. The Benchmark Must Be Fair

- Use LangChain idiomatically — recommended patterns, good prompts, good model
- Don't sandbag the LangChain agent
- Include enhanced LangChain variants (self-reflection, reliability-aware prompts)
- Report all results honestly, including cases where LangChain does well

### 5. Be Honest About Parameters

Every parameter in the Bayesian agent must be justified:
- **Tool cost:** Reflects real API cost asymmetry
- **Priors:** Beta(1,1) uniform — we genuinely don't know tool reliability
- **Forgetting rate λ:** Only used in drift variant; justified by non-stationarity prior

---

## Architecture Overview

```
Python (credence-engine)              Julia (credence DSL)
┌──────────────────────┐              ┌──────────────────────────┐
│ BayesianAgent        │              │ credence_agent.bdsl      │
│   choose_action      │─juliacall──→ │   agent-step             │
│   on_tool_response   │              │   answer-kernel           │
│   on_question_end    │              │   update-on-response      │
│                      │              ├──────────────────────────┤
│ CredenceBridge       │              │ host helpers (Julia)     │
│   (lazy juliacall)   │              │   update_beta_state      │
│                      │              │   marginalize_betas      │
│ Benchmark harness    │              │   initial_rel_state      │
│ LangChain baselines  │              │   initial_cov_state      │
│ Metrics              │              │   extract_reliability_means│
└──────────────────────┘              └──────────────────────────┘
```

Everything lives under the `credence` package (PyPI: `credence-agents`).

```
credence/
├── credence/                    # The package
│   ├── __init__.py              # Public API re-exports
│   ├── julia_bridge.py          # CredenceBridge: lazy juliacall wrapper
│   ├── inference/               # Types and feedback logic (computation in Julia)
│   │   ├── voi.py               # ScoringRule, ToolConfig (types only)
│   │   └── decision.py          # ActionType, Action, reliability update mapping
│   ├── agents/
│   │   ├── bayesian_agent.py    # BayesianAgent (Julia DSL backend)
│   │   ├── common.py            # Shared interface (AgentResult, DecisionStep)
│   │   ├── baselines.py         # Random, all-tools, oracle, single-best agents
│   │   ├── langchain_agent.py   # Standard LangChain ReAct agent
│   │   └── langchain_enhanced.py # LangChain with best-effort prompting
│   ├── environment/             # Benchmark-specific
│   │   ├── benchmark.py         # The benchmark harness
│   │   ├── tools.py             # Simulated tools with known reliability
│   │   ├── questions.py         # Question bank with ground truth
│   │   └── categories.py        # CATEGORIES tuple, make_keyword_category_infer_fn()
│   └── analysis/
│       ├── metrics.py           # Score, calibration, cost metrics
│       └── visualisation.py     # Plots and dashboards
├── tests/
├── experiments/
│   ├── run_stationary.py        # Main experiment: stationary reliabilities
│   ├── run_drift.py             # Extension: tool reliability drift
│   └── run_ablation.py          # Ablation studies
├── results/                     # Generated plots and data
└── pyproject.toml               # juliacall + numpy; [benchmark] extras for LangChain
```

### How it works

The `CredenceBridge` lazily loads Julia, the Credence module, and the BDSL agent spec
on first use. `BayesianAgent` maintains Julia objects (CategoricalMeasure for answer
beliefs, MixtureMeasure of ProductMeasure of BetaMeasure for per-tool reliability/coverage
state) and calls DSL functions through the bridge:

- `agent-step` — VOI + EU maximisation → action selection
- `answer-kernel` — observation model from reliability measure
- `update-on-response` — Bayesian conditioning on tool response
- `update_beta_state` — structured FactorSelector conditioning for reliability/coverage
- `marginalize_betas` — effective per-tool reliability from mixture state

Python handles the host loop: mapping post-question feedback to per-tool correctness
labels (`compute_reliability_updates`), then calling `update_beta_state` for each tool.

### Decoupled Architecture

The `BayesianAgent` constructor accepts injected domain knowledge:

```python
BayesianAgent(
    bridge: CredenceBridge,                    # Julia bridge (lazy-loads on first use)
    tool_configs: list[ToolConfig],            # Tool costs + coverage vectors
    categories: tuple[str, ...] | None,        # Domain categories (injected)
    category_infer_fn: Callable | None,        # Question text → category prior (injected)
    forgetting: float = 1.0,                   # Exponential forgetting rate
)
```

`ScoringRule` and `ToolConfig` are the only configuration types needed to parameterise
the entire decision loop. `make_keyword_category_infer_fn()` in `categories.py` is a
convenience for the benchmark; custom domains provide their own `category_infer_fn`.

---

## Common Mistakes to Avoid

### WRONG: Reimplementing inference in Python
```python
# BAD — duplicating Julia's exact Bayesian computation
posterior = prior * likelihood / marginal
```

### RIGHT: Call the DSL through the bridge
```python
# GOOD — all inference in Julia
answer_measure = bridge.update_on_response(answer_measure, kernel, response)
```

### WRONG: Hardcoding domain knowledge in the agent
```python
# BAD — domain-specific routing inside the agent
if question.looks_numerical():
    tool = "calculator"
```

### RIGHT: Inject domain knowledge, decide via EU
```python
# GOOD — domain categories and inference injected at construction
agent = BayesianAgent(
    bridge=bridge,
    tool_configs=tool_configs,
    categories=("factual", "numerical", "recent", "misconception", "reasoning"),
    category_infer_fn=my_category_infer_fn,
)
# Tool selection happens via EU maximisation in the Julia DSL
```

### WRONG: Making the LangChain agent deliberately bad
```python
# BAD — strawman
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### RIGHT: Give LangChain every advantage
```python
# GOOD — use the best available patterns with a carefully crafted system prompt
agent = initialize_agent(
    tools, llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)
```

---

## Testing Strategy

1. **Julia DSL tests pass first** — `julia ~/git/credence/test/test.jl` and `test_host.jl`
2. **Python type/feedback tests** — `test_decision.py` (no Julia needed)
3. **Agent integration tests** — `test_bayesian_agent.py` (needs Julia via bridge)
4. **Baseline comparison tests** — `test_baselines.py` (Oracle, Random, AllTools, SingleBest)
5. **Benchmark harness tests** — `test_benchmark.py` (mock agents, no Julia)

Run all: `PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run python -m pytest tests/ -v`

---

## Key Invariants

- VOI is always non-negative (information can't hurt in expectation)
- All Bayesian computation runs in Julia — Python never modifies measure weights
- EU(submit) + EU(abstain) + EU(query) computed over the SAME belief state
- Tool costs are subtracted from EU(query), not from score retrospectively
- Ground truth for reliability updates: ONLY from final answer feedback
- The Bayesian agent never "peeks" at true tool reliability
- The agent has NO domain-specific constants
- PythonCall uses 0-based indexing for Julia arrays/tuples accessed from Python

## Dependencies

- **juliacall** — Python-Julia bridge (lazy-loads Julia on first use)
- **numpy** — used by benchmark/environment for coverage vectors and category priors
- **Julia Credence DSL** — at `~/git/credence/` (or pass paths to CredenceBridge)
