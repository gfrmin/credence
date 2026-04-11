# Credence

A Bayesian decision-making DSL (Julia) and the tools built on it: an AI gateway that learns to route LLM requests, a domain-agnostic agent library, and research environments for testing the theory.

Everything reduces to three axioms: beliefs are probability measures, rational action maximises expected utility, learning is conditioning on evidence. Three types (Space, Measure, Kernel), axiom-constrained functions, and nothing else.

## Use credence-proxy for smarter LLM routing

[credence-proxy](python/credence_router/) is a drop-in OpenAI-compatible gateway that learns which model works best for each type of query:

| Metric | Always Sonnet | Credence routing | Change |
|--------|--------------|-----------------|--------|
| Quality (0-10) | 6.56 | 7.80 | **+1.24** |
| Avg latency | 8.4s | 4.0s | **-52%** |
| Avg cost/request | $0.024 | $0.001 | **-96%** |

```bash
docker run -p 8377:8377 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  -v credence-data:/data \
  credence-proxy
```

Point any OpenAI-compatible client at `http://localhost:8377/v1`. The proxy picks the best model, streams the response, and updates its beliefs from the outcome.

### Using with OpenClaw

Add a custom provider to `~/.openclaw-dev/openclaw.json`:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "credence": {
        "baseUrl": "http://localhost:8377/v1",
        "apiKey": "not-needed",
        "api": "openai-completions",
        "models": [{"id": "auto", "name": "Credence (auto-routed)"}]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "credence/auto"}
    }
  }
}
```

See [credence-proxy docs](python/credence_router/README.md) for all endpoints, search routing, monitoring, and configuration.

## Explore the DSL

The DSL has three primitives — no more, no less:

```scheme
(belief h1 h2 ...)              ;; weighted hypotheses (probability measure)
(update <belief> <obs> <lik>)   ;; Bayesian conditioning (the only learning mechanism)
(decide <belief> <acts> <util>) ;; expected utility maximisation (the only decision mechanism)
```

### Quick start

Requires Julia >= 1.9 (stdlib only, no packages).

```bash
julia -e 'push!(LOAD_PATH, "src"); using Credence; run_dsl(read("examples/coin.bdsl", String))'
```

### Example: learning a biased coin

```scheme
(let prior (belief 0.1 0.3 0.5 0.7 0.9)
  (let lik (lambda (theta obs)
             (if (= obs 1) (log theta) (log (- 1.0 theta))))
    (let posterior (update (update prior 1 lik) 0 lik)
      (decide posterior (list 1 0)
        (lambda (theta action)
          (if (= action 1)
            (if (> theta 0.5) 1.0 -1.0)
            (if (< theta 0.5) 1.0 -1.0)))))))
```

### Run tests

```bash
julia test/test_core.jl             # Core DSL (42 tests)
julia test/test_flat_mixture.jl     # Flat mixture conditioning
julia test/test_program_space.jl    # Tier 2: enumeration, compilation, perturbation
julia test/test_grid_world.jl       # Tier 3: full agent + regime change
```

## What's in this repo

```
src/                        DSL core: parser, evaluator, ontology (Space, Measure, Kernel)
src/program_space/          Program-space inference (grammars, enumeration, perturbation)
domains/                    Research environments (grid world, email, RSS, QA)
python/
  credence_bindings/        Python bindings via juliacall (Space, Measure, Kernel)
  credence_agents/          Domain-agnostic Bayesian agent library
  credence_router/          credence-proxy: AI gateway for LLM/search routing
  bayesian_if/              Interactive Fiction agent
julia/pomdp_agent/          POMDP agent (Thompson MCTS, state abstraction)
examples/                   DSL examples (coin, agent, grid)
```

| Component | README | What it does |
|-----------|--------|-------------|
| [credence-proxy](python/credence_router/) | [README](python/credence_router/README.md) | LLM/search routing gateway |
| [credence-agents](python/credence_agents/) | [README](python/credence_agents/README.md) | Bayesian agent library + benchmark |
| [bayesian-if](python/bayesian_if/) | [README](python/bayesian_if/README.md) | Interactive Fiction agent |
| [POMDP agent](julia/pomdp_agent/) | [CLAUDE.md](julia/pomdp_agent/CLAUDE.md) | Planning under uncertainty (Jericho IF) |

## Install everything

```bash
# Python packages (all four, via uv workspace)
uv sync

# Run Python tests
PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest python/

# Run credence-proxy from source
PYTHON_JULIACALL_HANDLE_SIGNALS=yes credence-router serve
```

## Architecture

```
Three types (frozen)           Space, Measure, Kernel
Axiom-constrained functions    condition, expect, push, density
Standard library               optimise, value, voi, model, problem
Program-space inference         grammars, enumeration, compilation
Domain applications             grid world, email, RSS, QA benchmark
Python ecosystem                bindings → agents → router, bayesian-if
```

All Bayesian inference runs in the Julia DSL. Python handles host concerns: API calls, tool queries, persistence. The DSL is pure — no side effects, no IO, no mutation.

## References

- Cox (1946) — probability from consistency
- Savage (1954) — utility + probability from preferences
- Jaynes (2003) — *Probability Theory: The Logic of Science*
- McCarthy (1960) — why S-expressions

## License

AGPL-3.0
