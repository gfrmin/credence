# CLAUDE.md

Practical guidance for Claude Code working in this package.

This is the POMDP agent package within the credence monorepo.
Location: `credence/apps/julia/pomdp_agent/`

## Build & Run

Julia >= 1.9 required. Run from this directory (`credence/apps/julia/pomdp_agent/`).

    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    julia --project=. -e 'using Pkg; Pkg.test()'
    julia --project=. -e 'using BayesianAgents'

## Playing Interactive Fiction Games

Requires: `pip install jericho`.

    # Default: LLM enabled, auto-detected via Ollama /api/tags
    julia --project=. examples/jericho_agent.jl /path/to/game.z3

    # Fast testing without LLM
    julia --project=. examples/jericho_agent.jl /path/to/game.z3 --no-llm

    # All options
    --no-llm              # Disable LLM sensor (LLM is ON by default)
    --episodes N          # Number of episodes (default 5)
    --steps N             # Max steps per episode (default 300)
    --model NAME          # Ollama model name (default llama3.2)
    --mcts-iter N         # MCTS iterations per decision (default 60)
    --mcts-depth N        # Planning horizon (default 12)
    --quiet               # Suppress verbose output
    --debug               # Enable Julia debug logging

Game files: `~/yo/Games/if/jericho-game-suite/`
Ollama setup: `ollama serve` then `ollama pull llama3.2`

## Architecture

Core module: `src/BayesianAgents.jl` — defines 5 abstract interfaces and the agent loop.

| Interface | Methods | Implementations |
|---|---|---|
| `World` | `reset!`, `step!`, `actions` | `GridWorld`, `JerichoWorld` |
| `WorldModel` | `update!`, `sample_dynamics`, `transition_dist`, `reward_dist` | `TabularWorldModel`, `FactoredWorldModel` |
| `Planner` | `plan` | `ThompsonMCTS`, `FactoredMCTS` |
| `StateAbstractor` | `abstract_state`, `record_transition!`, `check_contradiction`, `refine!` | `IdentityAbstractor`, `BisimulationAbstractor`, `MinimalStateAbstractor` |
| `Sensor` | `query`, `tpr`, `fpr`, `update_reliability!` | `BinarySensor` (Beta posteriors for TPR/FPR) |

**Two model paths**:
- **TabularWorldModel** — Dirichlet-Categorical transitions, Normal-Gamma rewards, hash-based states
- **FactoredWorldModel** — factored CPDs over (location, inventory, hidden vars), used by Jericho agent

**Data flow**: World → StateAbstractor → WorldModel → Planner → Decision. Sensors provide VOI-gated observations.

**Agent loop** (`act!` in BayesianAgents.jl): observe → abstract state → infer hidden variables → update model → plan via Thompson MCTS → execute action. Cumulative learning: model persists across episodes, only world state resets.

## Key Source Files

    src/BayesianAgents.jl              # Main module: interfaces, agent loop, exports
    src/probability/cpd.jl             # DirichletCategorical conjugate pair
    src/state/minimal_state.jl         # Factored state: location + inventory + hidden vars
    src/state/state_belief.jl          # StateBelief with Thompson sampling
    src/models/factored_world_model.jl # Action-conditional CPDs, selfloop tracking
    src/models/tabular_world_model.jl  # Dirichlet-Categorical tabular model
    src/models/binary_sensor.jl        # BinarySensor + LLMSensor with Beta posteriors
    src/inference/hidden_variable_inference.jl  # Spell/object/knowledge extraction
    src/inference/bayesian_update.jl   # LLM likelihood queries, belief updates
    src/planning/factored_mcts.jl      # MCTS in sampled factored MDP
    src/planners/thompson_mcts.jl      # MCTS with posterior-sampled dynamics
    src/worlds/jericho.jl              # Jericho IF game interface (PyCall)
    src/worlds/gridworld.jl            # Testing environment
    examples/jericho_agent.jl          # CLI entry point for IF games
    test/runtests.jl                   # 55 tests

## Design Principles

1. **Unified decision space** — act, ask, plan, refine all compared by expected utility; never separate "should I ask?" from "what should I do?"
2. **Trajectory-level planning** — MCTS plans 12+ steps ahead; VOI is trajectory-level, not myopic
3. **Hierarchical uncertainty** — state (level 0), dynamics (level 1), structure (level 2), meta-parameters (level 3)
4. **No exploration bonuses** — exploration emerges from Thompson Sampling over posterior uncertainty
5. **No loop detection hacks** — if the agent loops, fix state abstraction or beliefs, not symptoms
6. **LLM is a sensor** — provides observations with learned reliability; Bayesian machinery makes decisions
7. **Ground truth = reward > 0** — never use "state changed" as ground truth for sensor learning

## Forbidden Patterns

| Pattern | Why it's wrong | Correct approach |
|---------|---------------|-----------------|
| `eu + exploration_bonus` | Exploration emerges from Thompson Sampling | `eu = belief * reward` |
| `if should_ask(): ask()` | Violates unified decision space | `argmax EU over {ask, act}` |
| `llm.choose_action(state)` | LLM is a sensor, not decision-maker | Update beliefs from LLM, then argmax EU |
| `if action in recent: random()` | Masks model deficiency | Fix state abstraction |
| `helped = (new_state != old_state)` | State change != progress | `helped = (reward > 0)` |
| `prior_helps = 0.5` | Most actions don't help | `prior_helps = 1/n_actions` |

## Critical Implementation Lesson: observable_key

Hidden variables (spells_known, object_states, knowledge_gained) make `MinimalState` unique each step because heuristic inference adds to `knowledge_gained`. Any `Set` or `Dict` keyed on `MinimalState` must use `observable_key(s)` which returns `(location, sorted_inventory)` — the observable portion only.

Affected code:
- Selfloop detection in `FactoredWorldModel` — uses observable_key
- Oscillation detection in agent loop — compares observable_key to recent states
- `action_belief_cache` — keyed on `(observable_key(s), action)`

## Specs & Docs

- `MASTER_SPEC.md` — complete mathematical framework (equations, proofs, algorithms)
- `UNIFIED_SPEC.md` — additional theory (bisimulation, meta-learning, VOI)
- `ARCHITECTURE.md` — system design and component relationships

## Dependencies

Distributions.jl, StatsBase.jl, DataStructures.jl, LinearAlgebra (stdlib), PyCall.jl (for Jericho).

## Known deviations

Uses `Distributions.jl` directly in a few places (factored_world_model.jl, goal_planning.jl, gridworld.jl) instead of going through the credence DSL. Tracked as technical debt — the deviation is contained and doesn't violate axioms (used for sampling/mean-extraction, not belief updates).
