# Bayesian Agent Framework: Design Specification

## Philosophy

This framework implements **Bayes-Adaptive POMDPs** — agents that maintain uncertainty over both world state and world dynamics, and plan in a way that naturally balances exploration and exploitation.

### Core Principles (Non-Negotiable)

1. **All behaviour derives from expected utility maximisation** — no ad-hoc exploration bonuses, no loop detection hacks, no arbitrary thresholds.

2. **Uncertainty is first-class** — beliefs are probability distributions, not point estimates. Planning considers the *value of information*.

3. **The agent learns the world** — transition dynamics, observation models, and reward functions are uncertain and updated from experience.

4. **Sensors are fallible** — any information source (including LLMs) has learned reliability (TPR/FPR), and observations update beliefs via Bayes' rule.

5. **Simplicity is cost, not prior** — we don't say "simple models are more likely true"; we say "simple models cost less to use."

---

## Mathematical Foundation

### The Bayes-Adaptive POMDP

A standard POMDP is a tuple $(S, A, Z, T, O, R, \gamma)$:
- $S$ — state space
- $A$ — action space  
- $Z$ — observation space
- $T(s'|s,a)$ — transition model
- $O(z|s',a)$ — observation model
- $R(s,a)$ — reward function
- $\gamma$ — discount factor

A **Bayes-Adaptive POMDP** (BA-POMDP) augments the state with model parameters:

$$\bar{S} = S \times \Theta$$

where $\Theta$ is the space of possible models. The agent maintains a belief:

$$b(\bar{s}) = P(s, \theta | h)$$

where $h$ is the action-observation history.

### The Key Insight

In a BA-POMDP, **information-gathering actions have intrinsic value** because reducing uncertainty about $\theta$ improves future decisions. The optimal policy naturally explores when uncertain and exploits when confident — no exploration bonus needed.

### Belief Update

Given action $a$, observation $z$, and reward $r$:

$$b'(s', \theta') \propto O(z|s',a,\theta') \cdot T(s'|s,a,\theta') \cdot P(\theta'|\theta, s, a, s', r) \cdot b(s, \theta)$$

The term $P(\theta'|\theta, s, a, s', r)$ captures Bayesian updating of the model posterior.

### Planning

The optimal value function satisfies:

$$V^*(b) = \max_a \left[ \sum_{\bar{s}} b(\bar{s}) R(\bar{s}, a) + \gamma \sum_z P(z|b,a) V^*(b'_{a,z}) \right]$$

In practice, we use **Thompson Sampling over trajectories**:
1. Sample a world model $\theta \sim P(\theta|h)$
2. Plan optimally under the sampled model (via MCTS or similar)
3. Execute the first action of the optimal plan
4. Observe outcome, update beliefs, repeat

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           WORLD INTERFACE                           │
│  (Abstract — implemented by Jericho, GridWorld, real data, etc.)    │
│                                                                     │
│  • reset!() → initial observation                                   │
│  • step!(action) → (observation, reward, done)                      │
│  • actions(observation) → available actions                         │
│  • render() → human-readable state (optional)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STATE ABSTRACTOR                            │
│  (Learns equivalence classes over observations)                     │
│                                                                     │
│  • abstract(observation) → abstract state                           │
│  • expand!(contradiction) → refine abstraction                      │
│  • signature(state, action) → behavioural fingerprint               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          WORLD MODEL                                │
│  (Bayesian beliefs over dynamics)                                   │
│                                                                     │
│  • P(s'|s,a) — transition posterior (Dirichlet or GP)               │
│  • P(r|s,a) — reward posterior (Beta or Normal-Gamma)               │
│  • sample() → sampled dynamics for planning                         │
│  • update!(s, a, r, s') → Bayesian update                          │
│  • entropy() → current uncertainty                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           SENSOR BANK                               │
│  (Queryable information sources with learned reliability)           │
│                                                                     │
│  For each sensor (LLM, heuristic, oracle, etc.):                    │
│  • TPR: P(positive | true) — Beta posterior                         │
│  • FPR: P(positive | false) — Beta posterior                        │
│  • query(question) → answer                                         │
│  • posterior(prior, answer) → updated belief via Bayes             │
│  • update_reliability!(prediction, ground_truth)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            PLANNER                                  │
│  (Thompson Sampling + MCTS)                                         │
│                                                                     │
│  1. Sample world model from posterior                               │
│  2. Build search tree via MCTS under sampled model                  │
│  3. Use sensor priors for action expansion (if VOI > cost)          │
│  4. Return best action by visit count                               │
│                                                                     │
│  Planning depth: configurable (default K=10 steps)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DECISION MAKER                              │
│  (Unified EU maximisation over all options)                         │
│                                                                     │
│  Options:                                                           │
│    • take(action) — execute in world                                │
│    • ask(sensor, question) — query sensor (if VOI > cost)           │
│    • think(depth) — run more MCTS iterations                        │
│                                                                     │
│  Decision: argmax E[U]                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Value of Information (VOI)

For a binary sensor with learned TPR/FPR, the VOI of asking question $Q$ about proposition $P$ is:

$$\text{VOI}(Q) = \mathbb{E}[\max_a \text{EU}(a) | \text{after asking}] - \max_a \text{EU}(a) | \text{now}$$

Expanded:

$$\text{VOI}(Q) = P(\text{yes}) \cdot \max_a \text{EU}(a | \text{yes}) + P(\text{no}) \cdot \max_a \text{EU}(a | \text{no}) - \max_a \text{EU}(a | \text{now})$$

where:
- $P(\text{yes}) = \text{TPR} \cdot P(P) + \text{FPR} \cdot (1 - P(P))$
- $P(\text{no}) = 1 - P(\text{yes})$

**Decision rule:** Ask if $\text{VOI}(Q) > \text{cost}(Q)$

**Stopping criterion:** Stop asking when no question has positive net VOI.

---

## State Abstraction via Bisimulation

Two states $s_1, s_2$ are **bisimilar** if for all actions $a$:

$$P(r | s_1, a) = P(r | s_2, a) \quad \text{and} \quad P(\phi(s') | s_1, a) = P(\phi(s') | s_2, a)$$

where $\phi$ is the abstraction function.

In practice, we:
1. Start with identity abstraction (each observation is its own class)
2. Track behavioural signatures: $(a_1 \to r_1, s'_1), (a_2 \to r_2, s'_2), ...$
3. Cluster states with identical signatures
4. When contradiction detected (same abstract state, different outcomes), refine

This ensures the agent doesn't loop on functionally equivalent states.

---

## Handling Sparse Rewards

### Option 1: Intrinsic Motivation via Information Gain

Define intrinsic reward as entropy reduction in the world model:

$$r_{\text{intrinsic}}(s, a, s') = H[\theta | h] - H[\theta | h, (s, a, s')]$$

This is principled: information about the world improves future decisions.

### Option 2: Hindsight Credit Assignment

When reward is received, trace back the trajectory and update Q-values for contributing actions:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \gamma^{T-t} \cdot r_T$$

### Option 3: LLM-based Progress Assessment

Query the LLM sensor: "Does this state seem closer to the goal than the previous state?"

Use the answer to construct shaped rewards, with reliability learned from eventual ground truth.

---

## World Interface Contract

Any world must implement:

```julia
abstract type World end

# Required
reset!(world::World) → observation
step!(world::World, action) → (observation, reward, done, info)
actions(world::World, observation) → Vector{Action}

# Optional
render(world::World) → String
seed!(world::World, seed::Int) → nothing
```

This allows the same agent to connect to:
- **Jericho** (Interactive Fiction)
- **GridWorld** (your existing simulation)
- **Gymnasium** (OpenAI Gym environments)
- **Real data streams** (financial markets, sensor data, etc.)

---

## Sensor Interface Contract

Any sensor must implement:

```julia
abstract type Sensor end

# Query the sensor
query(sensor::Sensor, state, question) → answer

# Sensor reliability (learned)
tpr(sensor::Sensor) → Float64  # P(positive | true)
fpr(sensor::Sensor) → Float64  # P(positive | false)

# Update reliability from ground truth
update!(sensor::Sensor, predicted::Bool, actual::Bool)

# Bayesian belief update
posterior(sensor::Sensor, prior::Float64, answer::Bool) → Float64
```

An LLM is just one sensor. Others might include:
- **Heuristics** (domain-specific rules)
- **Human feedback** (crowd-sourced labels)
- **Other ML models** (classifiers, regressors)

---

## Configuration

```julia
@kwdef struct AgentConfig
    # Planning
    planning_depth::Int = 10
    mcts_iterations::Int = 100
    discount::Float64 = 0.99
    
    # Exploration (Thompson sampling handles this, but tunable)
    ucb_c::Float64 = 2.0  # for MCTS tree policy
    
    # Sensors
    sensor_cost::Float64 = 0.01
    max_queries_per_step::Int = 10
    
    # Priors
    transition_prior_strength::Float64 = 0.1  # Dirichlet concentration
    reward_prior_mean::Float64 = 0.0
    reward_prior_variance::Float64 = 1.0
    
    # State abstraction
    abstraction_threshold::Float64 = 0.95  # confidence for merging states
end
```

---

## Success Criteria

1. **No loops** — agent never repeats futile action sequences (bisimulation handles this)
2. **Sensor learning** — TPR/FPR converge to true reliability over episodes
3. **Natural exploration** — agent tries diverse actions when uncertain, exploits when confident
4. **Score improvement** — later episodes outperform earlier ones
5. **Principled** — every behaviour explainable via expected utility maximisation

---

## Implementation Roadmap

### Phase 1: Core Framework (Julia)
- [ ] World interface + GridWorld implementation
- [ ] Bayesian world model (Dirichlet transitions, Normal-Gamma rewards)
- [ ] Thompson Sampling planner
- [ ] Basic MCTS

### Phase 2: State Abstraction
- [ ] Behavioural signature tracking
- [ ] Bisimulation-based clustering
- [ ] Automatic refinement on contradiction

### Phase 3: Sensor Integration
- [ ] Sensor interface
- [ ] Binary sensor with learned reliability
- [ ] VOI calculation
- [ ] LLM sensor (Ollama integration)

### Phase 4: World Connections
- [ ] Jericho adapter for IF games
- [ ] Gymnasium adapter for RL benchmarks
- [ ] Real data adapters

### Phase 5: Advanced Planning
- [ ] POMCP / DESPOT integration
- [ ] Belief compression for scalability
- [ ] Parallel MCTS

---

## References

- Ross, Chaib-draa, Pineau (2008). "Bayes-Adaptive POMDPs"
- Silver & Veness (2010). "Monte-Carlo Planning in Large POMDPs" (POMCP)
- Katt et al. (2018). "Bayesian Reinforcement Learning in Factored POMDPs"
- Egorov et al. (2017). "POMDPs.jl: A Framework for Sequential Decision Making"
