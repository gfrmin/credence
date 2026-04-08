# SPECIFICATION.md — Mathematical Details

## 1. The Unified Decision Problem

At each timestep, the agent faces a decision from a unified option space:

$$\mathcal{O} = \mathcal{A} \cup \mathcal{Q}$$

where:
- $\mathcal{A}$ = game actions (move north, take lamp, etc.)
- $\mathcal{Q}$ = sensor queries (ask LLM questions)

The agent selects:

$$o^* = \arg\max_{o \in \mathcal{O}} \mathbb{E}[U | o, s, \theta]$$

where $s$ is the current state and $\theta$ represents beliefs.

---

## 2. Expected Utility of Game Actions

For action $a \in \mathcal{A}$ in state $s$:

### Case 1: Known Outcome (Deterministic World)

If we've observed $(s, a)$ before:

$$\mathbb{E}[U | \text{take}(a), s] = r_{\text{observed}}$$

### Case 2: Unknown Outcome

$$\mathbb{E}[U | \text{take}(a), s] = \mathbb{E}_{\theta \sim P(\theta|\mathcal{H})} \left[ \mathbb{E}_{s', r \sim P(\cdot|s,a,\theta)} [r + \gamma V(s', \theta)] \right]$$

With Thompson Sampling, we approximate by sampling $\hat{\theta} \sim P(\theta|\mathcal{H})$:

$$\mathbb{E}[U | \text{take}(a), s] \approx \mathbb{E}_{s', r \sim P(\cdot|s,a,\hat{\theta})} [r + \gamma V(s', \hat{\theta})]$$

### Prior Belief About Action Success

For an untried action with $n$ total actions available:

$$P(\text{action helps}) = \frac{1}{n}$$

**Rationale**: In IF games, typically only 1-2 actions out of ~20-50 lead to progress.

---

## 3. Expected Utility of Asking Questions

For question $q$ to sensor $k$:

$$\mathbb{E}[U | \text{ask}(q, k), s] = \mathbb{E}_{\text{answer}}[\max_a \mathbb{E}[U | \text{take}(a), s, \text{answer}]] - c_k$$

where $c_k$ is the cost of querying sensor $k$.

### Expansion

Let $Y$ be the sensor's response (yes/no). Then:

$$\mathbb{E}[U | \text{ask}(q, k)] = P(Y=\text{yes}) \cdot \max_a \mathbb{E}[U|a, Y=\text{yes}] + P(Y=\text{no}) \cdot \max_a \mathbb{E}[U|a, Y=\text{no}] - c_k$$

### Computing $P(Y=\text{yes})$

$$P(Y=\text{yes}) = \text{TPR}_k \cdot P(\text{prop}) + \text{FPR}_k \cdot (1 - P(\text{prop}))$$

where "prop" is the proposition the question is about (e.g., "action A helps").

---

## 4. Value of Information

$$\text{VOI}(q, k) = \mathbb{E}[\max_a \mathbb{E}[U|a] \mid \text{after asking}] - \max_a \mathbb{E}[U|a] \mid \text{before asking}$$

**Properties:**
- $\text{VOI} \geq 0$ always (information cannot hurt)
- $\text{VOI} = 0$ if the answer wouldn't change the decision
- $\text{VOI}$ is highest when we're uncertain and the question discriminates

### Decision Criterion

Ask if and only if:

$$\text{VOI}(q, k) > c_k$$

Equivalently:

$$\mathbb{E}[U | \text{ask}(q, k)] > \max_a \mathbb{E}[U | \text{take}(a)]$$

---

## 5. Sensor Model

### Binary Sensor with Learned Reliability

A sensor answers yes/no questions with unknown reliability:

- **TPR** (True Positive Rate): $P(\text{yes} | \text{true})$
- **FPR** (False Positive Rate): $P(\text{yes} | \text{false})$

### Beta-Bernoulli Model

$$\text{TPR} \sim \text{Beta}(\alpha_{tp}, \beta_{tp})$$
$$\text{FPR} \sim \text{Beta}(\alpha_{fp}, \beta_{fp})$$

Point estimates:
$$\hat{\text{TPR}} = \frac{\alpha_{tp}}{\alpha_{tp} + \beta_{tp}}$$
$$\hat{\text{FPR}} = \frac{\alpha_{fp}}{\alpha_{fp} + \beta_{fp}}$$

### Prior

Start with weak belief that sensor is somewhat reliable:
- TPR prior: $\text{Beta}(2, 1)$ → mean 0.67
- FPR prior: $\text{Beta}(1, 2)$ → mean 0.33

### Posterior Update from Ground Truth

When we observe ground truth (reward > 0 means action helped):

| Sensor said | Ground truth | Update |
|-------------|--------------|--------|
| yes | true | $\alpha_{tp} \leftarrow \alpha_{tp} + 1$ |
| no | true | $\beta_{tp} \leftarrow \beta_{tp} + 1$ |
| yes | false | $\alpha_{fp} \leftarrow \alpha_{fp} + 1$ |
| no | false | $\beta_{fp} \leftarrow \beta_{fp} + 1$ |

### Belief Update from Sensor Response

Given prior $P(\text{prop})$ and sensor response $Y$:

$$P(\text{prop} | Y=\text{yes}) = \frac{\text{TPR} \cdot P(\text{prop})}{\text{TPR} \cdot P(\text{prop}) + \text{FPR} \cdot (1 - P(\text{prop}))}$$

$$P(\text{prop} | Y=\text{no}) = \frac{(1-\text{TPR}) \cdot P(\text{prop})}{(1-\text{TPR}) \cdot P(\text{prop}) + (1-\text{FPR}) \cdot (1 - P(\text{prop}))}$$

---

## 6. World Model

### Tabular Dirichlet-Categorical

For discrete states, model transitions as:

$$P(s' | s, a) \sim \text{Categorical}(\theta_{s,a})$$
$$\theta_{s,a} \sim \text{Dirichlet}(\alpha_{s,a,1}, \ldots, \alpha_{s,a,|S|})$$

### Prior

Symmetric Dirichlet with concentration $\alpha_0$ (e.g., 0.1):

$$\alpha_{s,a,s'} = \alpha_0 \quad \forall s, a, s'$$

### Posterior Update

After observing transition $(s, a) \to s'$:

$$\alpha_{s,a,s'} \leftarrow \alpha_{s,a,s'} + 1$$

### Posterior Predictive

$$P(s' | s, a, \mathcal{H}) = \frac{\alpha_{s,a,s'}}{\sum_{s''} \alpha_{s,a,s''}}$$

### Thompson Sampling

Sample $\theta_{s,a} \sim \text{Dirichlet}(\alpha_{s,a})$, then use $\theta_{s,a}$ for planning.

---

## 7. Reward Model

### Normal-Gamma Conjugate

$$r | s, a \sim \mathcal{N}(\mu_{s,a}, \sigma^2_{s,a})$$

With Normal-Gamma prior, maintain sufficient statistics:
- $n_{s,a}$: count
- $\bar{r}_{s,a}$: sample mean
- $S_{s,a}$: sum of squared deviations

### Online Update

For new reward $r$:

$$n \leftarrow n + 1$$
$$\delta = r - \bar{r}$$
$$\bar{r} \leftarrow \bar{r} + \delta / n$$
$$S \leftarrow S + \delta \cdot (r - \bar{r})$$

### Point Estimates

$$\hat{\mu} = \bar{r}$$
$$\hat{\sigma}^2 = S / (n - 1)$$

---

## 8. Planning: Thompson MCTS

### Algorithm

```
function thompson_mcts(state, model, iterations, depth):
    root = MCTSNode(state)
    
    for i in 1:iterations:
        # 1. Sample world model from posterior
        θ ~ P(θ | history)
        
        # 2. Simulate under sampled model
        simulate(root, θ, depth)
    
    # 3. Return most-visited action
    return argmax_visits(root.children)
```

### Simulation

```
function simulate(node, θ, depth):
    if depth == 0 or node.terminal:
        return 0
    
    if node.not_fully_expanded:
        action = select_unexpanded(node)
        child = expand(node, action, θ)
        value = reward(θ, node.state, action) + γ * rollout(child, θ, depth-1)
    else:
        action, child = select_ucb(node)
        value = reward(θ, node.state, action) + γ * simulate(child, θ, depth-1)
    
    node.visits += 1
    node.value += value
    return value
```

### UCB Selection (within single Thompson sample)

$$\text{UCB}(a) = \bar{V}(a) + c \sqrt{\frac{\ln N}{n(a)}}$$

---

## 9. State Abstraction: Bisimulation

### Definition

States $s_1$ and $s_2$ are bisimilar if:

$$\forall a: P(r | s_1, a) = P(r | s_2, a) \land P(\phi(s') | s_1, a) = P(\phi(s') | s_2, a)$$

where $\phi$ maps states to equivalence classes.

### Behavioural Signature

For state $s$, its signature is:

$$\text{sig}(s) = \{a \mapsto (\text{reward\_dist}(s,a), \text{transition\_dist}(s,a))\}_{a \in \mathcal{A}}$$

### Equivalence Criterion

$$s_1 \sim s_2 \iff \text{sig}(s_1) \approx \text{sig}(s_2)$$

where $\approx$ uses a similarity threshold on reward means and transition distributions.

### Refinement

When contradiction detected (same abstract state, different outcomes):
1. Split the equivalence class
2. Reassign states based on signature similarity
3. Re-cluster

---

## 10. Ground Truth Definition

**Critical**: Ground truth for sensor learning is:

$$\text{action\_helped} = (\text{reward} > 0)$$

**NOT**:
- ❌ State changed
- ❌ New location reached
- ❌ Item acquired

**Rationale**: Many state changes are neutral (moving between equivalent rooms) or negative (walking into danger). Only positive reward indicates genuine progress.

---

## 11. Complete Decision Algorithm

```julia
function decide(state, actions, sensors, model, config)
    # 1. Compute EU for all game actions
    action_eus = Dict()
    for a in actions
        if is_known(model, state, a)
            action_eus[a] = known_reward(model, state, a)
        else
            θ = sample_dynamics(model)
            action_eus[a] = simulate_eu(state, a, θ, config.depth)
        end
    end
    
    best_action = argmax(action_eus)
    best_action_eu = action_eus[best_action]
    
    # 2. Compute VOI for all questions to all sensors
    best_question = nothing
    best_question_eu = -Inf
    
    for sensor in sensors
        questions = generate_questions(state, actions)
        for q in questions
            prior = get_proposition_prior(q, model, state)
            voi = compute_voi(sensor, prior, action_eus)
            
            if voi > sensor.cost
                eu_ask = best_action_eu + voi - sensor.cost
                if eu_ask > best_question_eu
                    best_question_eu = eu_ask
                    best_question = (q, sensor)
                end
            end
        end
    end
    
    # 3. Unified decision
    if best_question !== nothing && best_question_eu > best_action_eu
        return AskDecision(best_question...)
    else
        return ActDecision(best_action)
    end
end
```

---

## 12. Convergence Properties

### Sensor Reliability

As observations $n \to \infty$:

$$\hat{\text{TPR}} \to \text{TPR}_{\text{true}}$$
$$\hat{\text{FPR}} \to \text{FPR}_{\text{true}}$$

Posterior variance: $O(1/n)$

### World Model

For tabular MDP with enough exploration:

$$P(s'|s,a,\mathcal{H}) \to P(s'|s,a)_{\text{true}}$$

### VOI Decay

As beliefs converge:
- VOI decreases (less uncertainty to resolve)
- Agent asks fewer questions over time
- Transitions to pure exploitation

---

## 13. Computational Complexity

| Operation | Complexity |
|-----------|------------|
| VOI computation | $O(|\mathcal{A}|)$ per question |
| Thompson sample | $O(|S|^2 \cdot |\mathcal{A}|)$ for tabular |
| MCTS iteration | $O(\text{depth})$ |
| Bisimulation check | $O(|S|^2 \cdot |\mathcal{A}|)$ worst case |

For IF games with $|S| \approx 1000$, $|\mathcal{A}| \approx 50$:
- VOI: fast
- Thompson: moderate
- MCTS: main bottleneck (tune iterations)
