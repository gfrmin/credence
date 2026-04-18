# Bayesian IF

A Bayesian decision-theoretic agent for Interactive Fiction, built on [credence](https://github.com/gfrmin/credence).

The agent uses **Value of Information** (VOI) to decide which sources to consult — looking around, examining objects, checking inventory, or asking an LLM — before committing to an action. Every decision is expected-utility maximisation; there are no hardcoded heuristics.

## Install

```bash
pip install bayesian-if
# or with optional backends:
pip install bayesian-if[jericho]   # Z-machine games
pip install bayesian-if[ollama]    # LLM advisor via Ollama
```

## Usage

```python
from bayesian_if import IFAgent
from bayesian_if.jericho_world import JerichoWorld

world = JerichoWorld("zork1.z5")
agent = IFAgent(world)
result = agent.play(max_steps=100)
print(f"Score: {result.final_score}")
```

## How It Works

Each game step is treated as a fresh decision problem:

1. The agent observes the current game state
2. VOI determines which **information-gathering tools** to query (look, examine, inventory, LLM)
3. Once the expected value of further queries drops below their cost, the agent commits to the best action
4. Score deltas provide ground truth for updating tool reliability beliefs

The reliability table persists across steps, so the agent learns which sources are informative for which types of situations.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
