# UNIFIED_SPEC.md — Complete Bayesian Agent Framework

## Overview

This document captures the complete specification for a **general-purpose Bayesian agent framework** that:

1. Maintains principled uncertainty over world dynamics, state, and sensor reliability
2. Plans over **trajectories**, not just single actions
3. Learns **state equivalence classes** to avoid loops on functionally identical states
4. Treats **asking questions** and **taking actions** in a **unified decision space**
5. Supports **meta-learning**: learning how to learn, adapting priors across tasks
6. Connects to arbitrary worlds (IF games, grid worlds, real data streams)

**All behaviour derives from expected utility maximisation. No hacks.**

---

## Part I: Foundational Principles

### Principle 1: Bayesian Everything

Uncertainty is first-class throughout:

| Component | Representation | Update Rule |
|-----------|---------------|-------------|
| World dynamics | Dirichlet-Categorical / GP | Conjugate Bayesian |
| Rewards | Normal-Gamma | Conjugate Bayesian |
| Sensor reliability | Beta (TPR, FPR) | Conjugate Bayesian |
| State abstraction | Posterior over partitions | Evidence from contradictions |
| Meta-parameters | Hierarchical priors | Empirical Bayes / full Bayes |

### Principle 2: Expected Utility Maximisation

Every decision is:

$$d^* = \arg\max_{d \in \mathcal{D}} \mathbb{E}[U \mid d, \text{beliefs}]$$

where $\mathcal{D}$ includes both actions and information-gathering queries.

### Principle 3: No Ad-Hoc Components

| Forbidden | Principled Alternative |
|-----------|----------------------|
| Exploration bonuses | Thompson Sampling (exploration from posterior sampling) |
| Loop detection | Bisimulation abstraction (equivalent states merged) |
| Hardcoded heuristics | Learned from data via Bayesian updates |
| LLM as decision-maker | LLM as sensor with learned reliability |

### Principle 4: Trajectories, Not Single Steps

The agent plans over **sequences of actions**, not just the next action. One-step lookahead fails in domains requiring multi-step solutions (IF games need ~20 steps to solve puzzles).

### Principle 5: Hierarchical Uncertainty

Uncertainty exists at multiple levels:

```
Level 0: State          — Where am I? P(s | observations)
Level 1: Dynamics       — How does the world work? P(θ | history)
Level 2: Structure      — What state representation is correct? P(φ | contradictions)
Level 3: Meta           — What priors work well? P(hyperparameters | tasks)
```

---

## Part II: The World Model Hierarchy

### Level 0: State Belief

**Problem**: The agent observes text/pixels, not the true state.

**Solution**: Maintain belief over current state given observation history.

$$P(s_t \mid o_1, a_1, \ldots, o_t) \propto P(o_t \mid s_t) \sum_{s_{t-1}} P(s_t \mid s_{t-1}, a_{t-1}) P(s_{t-1} \mid \ldots)$$

For deterministic IF games, this often collapses to a point mass once we've identified our location.

### Level 1: Dynamics Belief

**Problem**: We don't know transition probabilities or reward distributions.

**Solution**: Bayesian model over dynamics parameters.

#### Transition Model (Dirichlet-Categorical)

For discrete states:

$$P(s' \mid s, a) \sim \text{Categorical}(\theta_{s,a})$$
$$\theta_{s,a} \sim \text{Dirichlet}(\alpha_{s,a})$$

**Prior**: $\alpha_{s,a,s'} = \alpha_0$ (e.g., 0.1 — weak, easily overridden)

**Update**: After observing $(s, a) \to s'$:
$$\alpha_{s,a,s'} \leftarrow \alpha_{s,a,s'} + 1$$

**Posterior predictive**:
$$P(s' \mid s, a, \mathcal{H}) = \frac{\alpha_{s,a,s'}}{\sum_{s''} \alpha_{s,a,s''}}$$

#### Reward Model (Normal-Gamma)

$$r \mid s, a \sim \mathcal{N}(\mu_{s,a}, \sigma^2_{s,a})$$

Maintain sufficient statistics $(n, \bar{r}, S)$ for online updates.

### Level 2: Structure Belief (State Abstraction)

**Problem**: Raw state representations are too fine-grained. "Wearing trousers" vs "not wearing trousers" may be strategically identical.

**Solution**: Learn equivalence classes via **bisimulation**.

#### Bisimulation Definition

States $s_1, s_2$ are **bisimilar** iff:

$$\forall a \in \mathcal{A}: \quad P(r \mid s_1, a) = P(r \mid s_2, a) \quad \land \quad P(\phi(s') \mid s_1, a) = P(\phi(s') \mid s_2, a)$$

where $\phi: S \to S_{\text{abstract}}$ maps to equivalence classes.

#### Behavioural Signature

Each state has a signature:

$$\text{sig}(s) = \left\{ a \mapsto \left( \mathbb{E}[r \mid s, a], \text{Var}[r \mid s, a], P(\phi(\cdot) \mid s, a) \right) \right\}_{a \in \mathcal{A}}$$

Two states are equivalent iff their signatures match (within tolerance).

#### Contradiction Detection

When the same abstract state yields different outcomes:

$$\exists a: (s_1 \sim s_2) \land (r_1 \neq r_2 \text{ or } s'_1 \not\sim s'_2)$$

This triggers **refinement**: split the equivalence class.

#### Posterior Over Partitions

Formally, maintain:

$$P(\phi \mid \text{history}) \propto P(\text{history} \mid \phi) P(\phi)$$

In practice, use greedy refinement when contradictions detected.

### Level 3: Meta-Parameters (Learning to Learn)

**Problem**: What priors should we use for a new world?

**Solution**: Hierarchical Bayesian model learns priors from experience across tasks.

#### Hierarchical Prior for Dynamics

$$\alpha_0 \sim \text{Gamma}(a, b)$$
$$\theta_{s,a} \mid \alpha_0 \sim \text{Dirichlet}(\alpha_0 \cdot \mathbf{1})$$
$$s' \mid \theta_{s,a} \sim \text{Categorical}(\theta_{s,a})$$

After many tasks, the posterior $P(\alpha_0 \mid \text{tasks})$ concentrates on the right sparsity level.

#### Hierarchical Prior for Sensor Reliability

$$\text{TPR}_k \sim \text{Beta}(\alpha_{tp}, \beta_{tp})$$

where $(\alpha_{tp}, \beta_{tp})$ are learned across games:
- If LLM consistently has 70% TPR, the prior sharpens around 0.7
- New games start with this informed prior, not vague (2,1)

#### Hierarchical Prior for Action Success

Instead of hardcoding $P(\text{action helps}) = 1/n$:

$$p_{\text{success}} \sim \text{Beta}(\alpha_s, \beta_s)$$

Learn $(\alpha_s, \beta_s)$ from experience across games.

---

## Part III: Trajectory-Level Planning

### The Problem with Myopic Planning

One-step lookahead computes:

$$a^* = \arg\max_a \mathbb{E}[r(s, a) + \gamma V(s')]$$

This fails when:
- Rewards are sparse (most actions give 0)
- Solutions require sequential dependencies (key → door → treasure)
- Information value extends beyond immediate decisions

### Trajectory Definition

A trajectory is a sequence:

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$$

The value of a trajectory:

$$V(\tau) = \sum_{t=0}^{T} \gamma^t r(s_t, a_t)$$

### Optimal Trajectory

$$\tau^* = \arg\max_\tau \mathbb{E}_{\theta \sim P(\theta \mid \mathcal{H})} \left[ V(\tau) \mid \theta \right]$$

### Thompson Sampling Over Trajectories

```
function thompson_trajectory(state, model, horizon):
    # 1. Sample world model from posterior
    θ ~ P(θ | history)
    
    # 2. Plan optimal trajectory under sampled θ
    τ* = tree_search(state, θ, horizon)  # MCTS, A*, or beam search
    
    # 3. Return first action of optimal trajectory
    return τ*[0].action
```

**Key insight**: Different posterior samples yield different optimal trajectories. This naturally explores promising directions.

### Trajectory-Level Value of Information

Myopic VOI asks: "Will this answer change my next action?"

Trajectory VOI asks: "Will this answer improve my trajectory?"

$$\text{TVOI}(q) = \mathbb{E}_{\text{answer}}\left[\max_\tau V(\tau) \mid \text{answer}\right] - \max_\tau V(\tau) \mid \text{current beliefs}$$

#### Computing TVOI

```
function trajectory_voi(state, question, sensor, model, horizon):
    # Current best trajectory value
    θ_current = sample_dynamics(model)
    τ_current = tree_search(state, θ_current, horizon)
    V_current = trajectory_value(τ_current, θ_current)
    
    # Expected value after asking
    p_yes = P(sensor says yes | state, question)
    
    # Value if sensor says yes
    model_if_yes = update_beliefs(model, question, yes)
    θ_yes = sample_dynamics(model_if_yes)
    τ_yes = tree_search(state, θ_yes, horizon)
    V_yes = trajectory_value(τ_yes, θ_yes)
    
    # Value if sensor says no
    model_if_no = update_beliefs(model, question, no)
    θ_no = sample_dynamics(model_if_no)
    τ_no = tree_search(state, θ_no, horizon)
    V_no = trajectory_value(τ_no, θ_no)
    
    # Expected value after asking
    V_after = p_yes * V_yes + (1 - p_yes) * V_no
    
    return V_after - V_current
```

### Subgoal Discovery

Long trajectories can be decomposed into subgoals:

$$\tau = \tau_1 \circ \tau_2 \circ \ldots \circ \tau_k$$

where each $\tau_i$ achieves a subgoal.

**Subgoal detection**: States that are visited on many successful trajectories are likely subgoals.

$$P(\text{subgoal} \mid s) \propto \frac{\text{# successful trajectories through } s}{\text{# total trajectories through } s}$$

**Hierarchical planning**: First plan subgoal sequence, then plan within each subgoal.

---

## Part IV: The Unified Decision Space

### Decision Types

```julia
abstract type Decision end

struct ActDecision <: Decision
    action::Any
end

struct AskDecision <: Decision
    question::String
    sensor::Sensor
end

struct PlanDecision <: Decision
    # Request more planning computation
    additional_iterations::Int
end

struct AbstractDecision <: Decision
    # Request abstraction refinement
    states_to_split::Vector{Any}
end
```

### The Unified Decision Problem

At each step, choose from:

$$\mathcal{D} = \mathcal{A} \cup \mathcal{Q} \cup \{\text{plan\_more}\} \cup \{\text{refine\_abstraction}\}$$

All evaluated by expected utility:

$$d^* = \arg\max_{d \in \mathcal{D}} \mathbb{E}[U \mid d]$$

### Expected Utility Calculations

#### EU of Acting

$$\mathbb{E}[U \mid \text{take}(a)] = \mathbb{E}_{\theta, \tau}\left[V(\tau) \mid \tau_0 = (s, a), \theta\right]$$

With Thompson Sampling over trajectories:

$$\mathbb{E}[U \mid \text{take}(a)] \approx V(\tau^*_{\hat{\theta}})$$

where $\hat{\theta} \sim P(\theta \mid \mathcal{H})$ and $\tau^*_{\hat{\theta}}$ starts with action $a$.

#### EU of Asking

$$\mathbb{E}[U \mid \text{ask}(q, k)] = \mathbb{E}_{\text{answer}}\left[\max_\tau V(\tau) \mid \text{answer}\right] - c_k$$

This is trajectory-level VOI minus query cost.

#### EU of Planning More

$$\mathbb{E}[U \mid \text{plan}(n)] = \mathbb{E}\left[\max_\tau V(\tau) \mid n \text{ more iterations}\right] - c_{\text{compute}} \cdot n$$

Diminishing returns: more planning helps less as we've already explored.

#### EU of Refining Abstraction

$$\mathbb{E}[U \mid \text{refine}(\phi')] = \mathbb{E}\left[\max_\tau V(\tau) \mid \phi'\right] - c_{\text{refine}}$$

Useful when contradictions suggest current abstraction is wrong.

### The Decision Algorithm

```julia
function unified_decide(agent)
    state = agent.current_state
    model = agent.world_model
    sensors = agent.sensors
    config = agent.config
    
    # 1. Plan trajectories under current beliefs
    trajectories = thompson_trajectory_search(
        state, model, config.horizon, config.n_samples
    )
    
    best_trajectory = trajectories[argmax(v -> v.value, trajectories)]
    best_action_eu = best_trajectory.value
    
    # 2. Compute trajectory-VOI for each question
    best_question = nothing
    best_question_eu = -Inf
    
    for sensor in sensors
        for question in generate_questions(state, best_trajectory, model)
            tvoi = trajectory_voi(state, question, sensor, model, config.horizon)
            
            if tvoi > sensor.cost
                eu_ask = best_action_eu + tvoi - sensor.cost
                if eu_ask > best_question_eu
                    best_question_eu = eu_ask
                    best_question = (question, sensor)
                end
            end
        end
    end
    
    # 3. Check if more planning would help
    planning_voi = estimate_planning_voi(trajectories, config)
    eu_plan = best_action_eu + planning_voi - config.compute_cost
    
    # 4. Check if abstraction refinement would help
    contradiction = check_contradiction(agent.abstractor)
    if !isnothing(contradiction)
        refinement_voi = estimate_refinement_voi(contradiction, model)
        eu_refine = best_action_eu + refinement_voi - config.refine_cost
    else
        eu_refine = -Inf
    end
    
    # 5. Unified decision: pick best option
    options = [
        (ActDecision(best_trajectory.first_action), best_action_eu),
        (AskDecision(best_question...), best_question_eu),
        (PlanDecision(config.extra_iterations), eu_plan),
        (AbstractDecision(contradiction), eu_refine)
    ]
    
    filter!(x -> x[2] > -Inf, options)
    
    return options[argmax(x -> x[2], options)][1]
end
```

---

## Part V: Sensor System

### Binary Sensor Model

```julia
mutable struct BinarySensor
    name::String
    
    # Reliability: TPR ~ Beta(α_tp, β_tp), FPR ~ Beta(α_fp, β_fp)
    α_tp::Float64
    β_tp::Float64
    α_fp::Float64
    β_fp::Float64
    
    # Query function
    query_fn::Function
    
    # Cost per query (in utility units)
    cost::Float64
    
    # Statistics
    n_queries::Int
    n_correct::Int
end

tpr(s) = s.α_tp / (s.α_tp + s.β_tp)
fpr(s) = s.α_fp / (s.α_fp + s.β_fp)
```

### Bayesian Belief Update from Sensor Response

Given prior $P(\text{prop})$ and sensor response:

$$P(\text{prop} \mid \text{yes}) = \frac{\text{TPR} \cdot P(\text{prop})}{\text{TPR} \cdot P(\text{prop}) + \text{FPR} \cdot (1 - P(\text{prop}))}$$

$$P(\text{prop} \mid \text{no}) = \frac{(1-\text{TPR}) \cdot P(\text{prop})}{(1-\text{TPR}) \cdot P(\text{prop}) + (1-\text{FPR}) \cdot (1 - P(\text{prop}))}$$

### Sensor Reliability Learning

**Ground truth** (critical!): 
$$\text{actually\_helped} = (\text{reward} > 0)$$

NOT state change. NOT new location. Only positive reward.

**Update**:

| Sensor said | Ground truth | Update |
|-------------|--------------|--------|
| yes | helped | $\alpha_{tp} \mathrel{+}= 1$ |
| no | helped | $\beta_{tp} \mathrel{+}= 1$ |
| yes | didn't help | $\alpha_{fp} \mathrel{+}= 1$ |
| no | didn't help | $\beta_{fp} \mathrel{+}= 1$ |

### LLM Sensor Implementation

```julia
struct LLMSensor <: Sensor
    name::String
    client::LLMClient
    model_name::String
    
    # Reliability (learned)
    α_tp::Float64
    β_tp::Float64
    α_fp::Float64
    β_fp::Float64
    
    # Cost
    cost::Float64
    
    # Prompt engineering
    system_prompt::String
end

function query(sensor::LLMSensor, state, question::String)::Bool
    prompt = format_prompt(sensor.system_prompt, state, question)
    response = generate(sensor.client, sensor.model_name, prompt)
    return parse_yes_no(response)
end
```

### Question Generation

Questions should be about things that matter for trajectory planning:

```julia
function generate_questions(state, trajectory, model)
    questions = String[]
    
    # About immediate action
    a = trajectory.first_action
    push!(questions, "Will '$a' help make progress?")
    push!(questions, "Is '$a' safe to try?")
    
    # About trajectory structure
    push!(questions, "Am I on the right track to solve this?")
    push!(questions, "Is there something I need to do before '$a'?")
    
    # About state understanding
    push!(questions, "Have I missed examining something important?")
    push!(questions, "Is there a hidden exit or item here?")
    
    # About subgoals
    if trajectory has waypoints
        for waypoint in trajectory.waypoints
            push!(questions, "Do I need to reach $waypoint?")
        end
    end
    
    return questions
end
```

---

## Part VI: Complete Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORLD INTERFACE                                │
│                                                                             │
│  Jericho (IF)  │  GridWorld  │  Gymnasium  │  RealData  │  Custom          │
│                                                                             │
│  Provides: observations, valid_actions, rewards, done                       │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE ABSTRACTION (Level 2)                        │
│                                                                             │
│  BisimulationAbstractor: Learn equivalence classes from behaviour          │
│                                                                             │
│  observation → signature → equivalence class → abstract state              │
│                                                                             │
│  Refines when contradictions detected (same abstract, different outcome)   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WORLD MODEL (Level 1)                              │
│                                                                             │
│  TabularWorldModel:                                                         │
│    • Transitions: Dirichlet-Categorical P(s'|s,a)                          │
│    • Rewards: Normal-Gamma P(r|s,a)                                        │
│                                                                             │
│  Supports:                                                                  │
│    • sample_dynamics() → θ for Thompson Sampling                           │
│    • update!(s, a, r, s') → posterior update                               │
│    • entropy() → uncertainty measure                                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENSOR BANK                                       │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   LLM Sensor    │  │  Heuristic      │  │    Oracle       │             │
│  │                 │  │  Sensor         │  │   (testing)     │             │
│  │  TPR/FPR        │  │                 │  │                 │             │
│  │  learned        │  │  TPR/FPR        │  │  Perfect        │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  Each: query(state, question) → answer, learned reliability                │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAJECTORY PLANNER                                     │
│                                                                             │
│  Thompson MCTS:                                                             │
│    1. Sample θ ~ P(θ | history)                                            │
│    2. Build search tree under θ                                            │
│    3. Return best trajectory                                                │
│                                                                             │
│  Computes: trajectory values, subgoal structure, uncertainty               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       UNIFIED DECISION ENGINE                               │
│                                                                             │
│  Options:                                                                   │
│    • take(a): Execute game action                                          │
│    • ask(q, k): Query sensor k with question q                             │
│    • plan(n): Request more planning iterations                             │
│    • refine(φ'): Refine state abstraction                                  │
│                                                                             │
│  Decision: argmax EU(option)                                               │
│                                                                             │
│  All options compete on equal footing!                                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        META-LEARNER (Level 3)                               │
│                                                                             │
│  Learns across tasks:                                                       │
│    • Prior concentration α₀ for dynamics                                   │
│    • Prior (α_tp, β_tp) for sensor reliability                             │
│    • Prior success rate for actions                                        │
│    • Useful state features for abstraction                                 │
│                                                                             │
│  Updates after each episode/task via empirical Bayes                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part VII: Mathematical Summary

### Trajectory Value

$$V(\tau) = \sum_{t=0}^{T} \gamma^t r(s_t, a_t)$$

### Expected Utility of Acting

$$\mathbb{E}[U \mid \text{take}(a)] = \mathbb{E}_{\theta \sim P(\theta|\mathcal{H})} \left[ V(\tau^*_\theta) \mid \tau^*_\theta[0] = a \right]$$

### Expected Utility of Asking

$$\mathbb{E}[U \mid \text{ask}(q,k)] = \mathbb{E}_{\text{ans}} \left[ \max_\tau V(\tau) \mid \text{ans} \right] - c_k$$

### Trajectory VOI

$$\text{TVOI}(q,k) = \mathbb{E}_{\text{ans}} \left[ \max_\tau V(\tau) \mid \text{ans} \right] - \max_\tau V(\tau) \mid \text{current}$$

### Bayesian Sensor Update

$$P(\text{prop} \mid y) = \frac{P(y \mid \text{prop}) P(\text{prop})}{P(y)}$$

### Bisimulation Condition

$$s_1 \sim s_2 \iff \forall a: P(r|s_1,a) = P(r|s_2,a) \land P(\phi(s')|s_1,a) = P(\phi(s')|s_2,a)$$

### Hierarchical Prior

$$\alpha_0 \sim P(\alpha_0 \mid \text{previous tasks})$$
$$\theta_{s,a} \sim \text{Dir}(\alpha_0)$$

---

## Part VIII: Implementation Checklist

### Core Types
- [ ] `Decision` hierarchy (Act, Ask, Plan, Refine)
- [ ] `World` interface
- [ ] `WorldModel` interface with Thompson sampling support
- [ ] `Sensor` interface with reliability learning
- [ ] `StateAbstractor` interface with refinement

### World Model
- [ ] Dirichlet-Categorical transitions
- [ ] Normal-Gamma rewards
- [ ] `sample_dynamics()` for Thompson
- [ ] `update!()` for posterior updates
- [ ] `entropy()` for uncertainty

### Sensors
- [ ] `BinarySensor` with Beta posteriors
- [ ] `LLMSensor` wrapping language model
- [ ] VOI computation (trajectory-level)
- [ ] Ground truth learning (`reward > 0` only)

### State Abstraction
- [ ] `IdentityAbstractor` (baseline)
- [ ] `BisimulationAbstractor`
- [ ] Signature computation
- [ ] Contradiction detection
- [ ] Refinement

### Planning
- [ ] Thompson Sampling infrastructure
- [ ] MCTS with sampled dynamics
- [ ] Trajectory representation
- [ ] Subgoal detection (optional)

### Decision Engine
- [ ] Unified EU computation for all decision types
- [ ] Trajectory-level VOI
- [ ] Proper comparison (all options equal)

### Meta-Learning
- [ ] Hierarchical priors for dynamics
- [ ] Hierarchical priors for sensors
- [ ] Cross-task prior updates

### Worlds
- [ ] `GridWorld` for testing
- [ ] `JerichoWorld` for IF
- [ ] Generic `World` adapter pattern

### Testing
- [ ] VOI computation correctness
- [ ] Sensor learning convergence
- [ ] Bisimulation merges equivalent states
- [ ] Thompson explores then exploits
- [ ] No loops on equivalent states
- [ ] Trajectory planning beats myopic

---

## Part IX: Forbidden Patterns

| ❌ DO NOT | ✅ DO INSTEAD |
|-----------|---------------|
| `eu = belief * reward + exploration_bonus` | `eu = belief * reward` (Thompson explores) |
| `if should_ask(): ask() else: act()` | `argmax EU(options)` over all options |
| `action = llm.choose_action(state)` | `belief = update(belief, llm.answer(question))` |
| `if action in recent: random_action()` | Fix abstraction so equivalent states merge |
| `helped = (state != prev_state)` | `helped = (reward > 0)` |
| `prior_success = 0.5` | `prior_success = 1/n_actions` (or learn it) |
| `voi > threshold` (hardcoded) | `voi > sensor.cost` (principled) |

---

## Part X: Configuration

```julia
Base.@kwdef struct AgentConfig
    # Trajectory planning
    horizon::Int = 20
    n_trajectory_samples::Int = 10
    mcts_iterations::Int = 100
    discount::Float64 = 0.99
    
    # Unified decision
    compute_cost::Float64 = 0.001      # Cost per planning iteration
    refine_cost::Float64 = 0.01        # Cost of abstraction refinement
    
    # Priors
    dynamics_prior_α::Float64 = 0.1    # Dirichlet concentration
    action_success_prior::Float64 = 0.05  # P(random action helps)
    sensor_tpr_prior::Tuple = (2.0, 1.0)
    sensor_fpr_prior::Tuple = (1.0, 2.0)
    
    # Abstraction
    use_bisimulation::Bool = true
    signature_similarity_threshold::Float64 = 0.95
    
    # Meta-learning
    meta_learning::Bool = true
    meta_update_frequency::Int = 10    # Episodes between meta-updates
end
```

---

## Part XI: Success Criteria

A correct implementation should:

1. **Ask when it helps**: TVOI > cost triggers questions
2. **Stop asking when it doesn't**: Converged beliefs → VOI → 0 → pure acting
3. **Never loop**: Bisimulation detects equivalent states
4. **Learn sensor reliability**: TPR/FPR converge to true values
5. **Plan trajectories**: Multi-step dependencies handled
6. **Improve across tasks**: Meta-learning sharpens priors
7. **Score improves**: Later episodes better than early ones
8. **Principled throughout**: Every decision is argmax EU, no hacks

---

## Part XII: Debugging Guide

### Agent always asks
- Check: Is sensor cost too low?
- Check: Is trajectory planning working? (Need trajectory value, not just immediate)
- Fix: Ensure TVOI compares trajectory values

### Agent never asks
- Check: Is sensor cost too high?
- Check: Are questions informative? (TPR > FPR)
- Fix: Start with reasonable sensor priors

### Agent loops
- Check: Is bisimulation enabled?
- Check: Are signatures being computed?
- Fix: Verify signature similarity threshold

### Poor trajectory planning
- Check: Is horizon sufficient?
- Check: Are MCTS iterations enough?
- Fix: Increase horizon, iterations; check reward signal

### Sensor reliability doesn't converge
- Check: Is ground truth being recorded?
- Check: Are updates happening?
- Fix: Only use `reward > 0` as ground truth; log updates

### Meta-learning not helping
- Check: Enough tasks completed?
- Check: Prior updates implemented?
- Fix: Verify hierarchical updates; need sufficient task diversity
