# Credence: A Bayesian Agent Architecture

## Research Spec

---

## 1. Axioms

The architecture rests on five axioms — the true bottom of the system, from which everything else is derived or learned. Each is grounded in a foundational result establishing its necessity.

### 1.1 Bayes' rule (`condition`)

Beliefs are updated by conditioning on observations:

$$b_{t+1}(\theta) \propto P(o_t \mid \theta, a_t) \cdot b_t(\theta)$$

This is the unique update rule satisfying diachronic coherence (Lewis 1999 — no sequence of bets can extract guaranteed money from an agent that updates this way). Amarante (2022) proved a stronger result: Bayes' rule is the unique rule for which updating and computing the predictive commute. Any other update rule either loses information or introduces inconsistency.

The DSL primitive `condition(belief, kernel, observation)` implements this. The kernel encodes P(o | θ, a) — the observation model.

### 1.2 Expected utility maximisation (`optimise`)

Actions are selected to maximise expected utility under the current belief:

$$a^* = \arg\max_{a \in A} \mathbb{E}_{\theta \sim b}[u_\theta(s, a)]$$

Savage's representation theorem (1954) establishes that if an agent's preferences over acts satisfy six axioms — completeness, the sure-thing principle, state-independence, comparative probability, non-triviality, and continuity — then there exists a unique probability measure P and a utility function u (unique up to positive affine transformation) such that the agent acts as if maximising E_P[u]. Six axioms suffice (Abdellaoui and Wakker 2020 showed P3 is redundant).

This grounds the entire enterprise: utility functions exist because preferences satisfying basic rationality constraints imply their existence. The agent doesn't need to be told what utility is — it's a consequence of coherent preference.

The DSL primitive `optimise(belief, action_space, utility)` implements this.

### 1.3 The complexity prior

The prior over programs is weighted by description length:

$$P(\text{program}) = 2^{-|\text{program}|}$$

This is Solomonoff's universal prior (1964) — the maximum-entropy coding over the terminal alphabet. Each symbol costs 1 bit regardless of what it does. It is the unique prior that dominates all computable priors up to a constant factor, implementing Occam's Razor: simpler hypotheses are favoured unless the data overwhelms the prior.

### 1.4 The alignment commitment

The agent's utility function IS the user's utility function. The agent does not know what the user's utility function is. This is formalised as a Cooperative Inverse Reinforcement Learning (CIRL) game (Hadfield-Menell, Russell, Abbeel, Dragan 2016):

$$M = \langle S, \{A_H, A_R\}, T, \{\Theta, R(\cdot;\theta)\}, P_0, \gamma \rangle$$

Both players receive the same reward R(s, a_H, a_R; θ), parameterised by θ which only the human knows. The agent maintains a belief b(θ) and maximises:

$$EU(a_R \mid s, b) = \mathbb{E}_{\theta \sim b}\left[R(s, a_H, a_R; \theta) + \gamma \cdot V^*(s', b')\right]$$

The alignment commitment is structural — it defines the objective, not the solution. It cannot be manipulated by the agent because the ground truth (user behaviour) comes from outside.

Russell's three principles (Human Compatible, 2019) articulate this:
1. The machine's only objective is to maximise the realisation of human preferences.
2. The machine is initially uncertain about what those preferences are.
3. The ultimate source of information about human preferences is human behaviour.

Shah et al. (2020) generalised CIRL into assistance games and proved that the optimal strategy in any assistance game reduces to solving a POMDP where b(θ) is a sufficient statistic — preserving tractability.

### 1.5 Prediction and integration (`expect`)

Predictions are computed by integrating over the belief:

$$\mathbb{E}_b[f(\theta)] = \int f(\theta) \, b(\theta) \, d\theta$$

This bridges beliefs and decisions: expected utility is `expect(belief, λθ. u_θ(action))`. The DSL primitive `expect(belief, function)` implements this.

### 1.6 Why these axioms are necessary (not merely sufficient)

Wald's complete class theorem (1950) establishes that under mild regularity conditions, any admissible decision procedure — any procedure not dominated by another across all possible states of nature — must be a Bayes rule with respect to some prior and utility function. This means our architecture is not one principled approach among many. It is the only admissible approach, up to choice of prior and hypothesis space.

Harsanyi's aggregation theorem (1955) provides additional confirmation: if the agent's preferences satisfy expected utility axioms and respect Pareto indifference over possible θ values, the agent's utility must be a weighted sum of the u_θ — exactly E_{θ~b}[u_θ], with weights given by the belief.

---

## 2. The Agent's Utility Function

### 2.1 Definition

The agent's utility for taking action a in state s given belief b is:

$$U(a, s, b) = \mathbb{E}_{\theta \sim b}[u_\theta(s, a)]$$

This is the expected user utility under the agent's current beliefs about what the user wants. The agent doesn't know u_θ directly — it knows only its belief b(θ), the posterior over programs representing hypotheses about the user's preferences.

### 2.2 Why this produces aligned behaviour

**Deference under uncertainty.** When b(θ) has high entropy, E_{θ~b}[u_θ(a)] is washed out — averaging over diverse preference hypotheses yields low expected utility for any specific action. The off-switch theorem (Hadfield-Menell, Dragan, Abbeel, Russell 2017) makes this precise: in a game where the agent can act, defer, or shut down, the incentive to defer is:

$$\Delta = \mathbb{E}[\pi_H(U_a) \cdot U_a] - \max\{\mathbb{E}[U_a], 0\}$$

When the agent has non-zero belief mass on both positive and negative utility actions, Δ > 0 — deference is strictly preferred. This is the theorem of non-negative expected value of information: an agent certain of its objectives has no reason to defer; an agent uncertain about them naturally seeks human guidance.

**Autonomy under confidence.** When b(θ) is concentrated around the true θ*, E_{θ~b}[u_θ(a)] closely approximates u_{θ*}(a). No query would significantly improve decisions. The agent acts autonomously.

**Re-engagement after preference change.** If the user's preferences change, the agent's predictions fail, surprise increases (low marginal likelihood), b(θ) disperses, and the agent becomes consultative again. This requires no explicit change-point detection — it's a consequence of the posterior dynamics. Programs with temporal structure that model non-stationarity will outperform stable programs when preferences genuinely shift.

### 2.3 Revealed preference and Savage's connection

Savage's framework provides the normative foundation for why u_θ exists at all. Each observed user choice (preferring action f over action g in situation s) yields a constraint:

$$\mathbb{E}_P[u_\theta(f, s)] > \mathbb{E}_P[u_\theta(g, s)]$$

These constraints progressively narrow the feasible set of utility functions consistent with observed behaviour. The agent's posterior b(θ) concentrates on utility functions that satisfy all observed constraints — this is Bayesian IRL, which we discuss in §3.

The Ellsberg paradox and prospect theory demonstrate that real humans violate Savage's axioms. This raises the preference laundering problem (§8): should the agent learn the user's revealed preferences (what they actually do, including biases) or their idealised preferences (what they would do if fully rational)? Our framework learns revealed preferences by default. Whether to "launder" them is a design choice at the observation model level.

---

## 3. Preference Inference: How the Agent Learns θ

The agent infers the user's preference parameter θ from observed behaviour. Eight frameworks formalise different aspects of this inference, and they converge on the same mathematical structure.

### 3.1 Inverse reinforcement learning

Classical IRL (Ng & Russell 2000) recovers a reward function from an observed optimal policy π*. The fundamental insight: the observed policy contains information about the reward function being optimised. The reward is constrained so that π* is optimal under it, with ℓ₁ regularisation resolving the degeneracy that R = 0 explains any policy.

**Bayesian IRL** (Ramachandran & Amir 2007) replaces the LP with posterior inference:

$$P(R \mid \text{demonstrations}) \propto \exp\left(\alpha \sum_i Q^*(s_i, a_i; R)\right) \cdot P(R)$$

The parameter α controls assumed expert optimality. The posterior mean E[R | data] is the optimal reward estimator under squared loss, and the optimal apprenticeship policy is optimal for the MDP with this mean reward. This directly maps to our architecture: the prior P(R) is the complexity-weighted mixture, the demonstrations are observed user actions, and the posterior is what `condition` produces.

**Maximum Entropy IRL** (Ziebart et al. 2008) assigns trajectory probabilities proportional to exponentiated reward:

$$P(\zeta \mid \theta) = \frac{1}{Z(\theta)} \exp(\theta^\top f_\zeta)$$

This is the maximum-entropy distribution matching empirical feature expectations. The resulting log-likelihood is convex, guaranteeing a unique optimum. MaxEnt IRL avoids the "label bias" problem of purely local models by providing globally normalised distributions. In our context, the Boltzmann-rational human model in CIRL is essentially MaxEnt IRL applied to single actions rather than trajectories.

### 3.2 Why passive observation is suboptimal

Shah et al. (2020) proved that separating reward inference from action selection is strictly suboptimal. In CIRL, the agent's actions affect what the human does (the human responds pedagogically — acting suboptimally to convey more information about θ). This is the key insight that distinguishes CIRL from classical IRL:

The human may choose to accept less reward on a particular action in order to convey more information to the agent. The demonstration-by-expert assumption (that the human acts optimally in isolation) is provably suboptimal in the cooperative setting. Merged estimation and control enables plans conditional on future feedback and relevance-aware active learning.

Our architecture handles this naturally. The user's response to ASK_USER is a cooperative signal — the user reveals their preference precisely because doing so helps the agent help them. Classical IRL treats the human as a passive demonstrator; our framework treats them as a cooperative partner.

### 3.3 Reward modelling and comparison-based learning

The RLHF framework (Christiano et al. 2017) learns reward functions from pairwise human comparisons using the Bradley-Terry preference model:

$$P(\sigma^1 \succ \sigma^2) = \sigma\left(\sum_t r(o^1_t, a^1_t) - \sum_t r(o^2_t, a^2_t)\right)$$

This is relevant because our agent may receive comparison feedback ("this was better than what you did last time") in addition to binary approval. The Bradley-Terry model provides the likelihood function for comparison observations. In our framework, this would be a program in the mixture that predicts user comparisons — it competes with simpler models under the complexity prior.

RLHF as practiced is not Bayesian — it produces a point estimate, not a posterior. It has no formal prior, no principled uncertainty quantification, and no VOI mechanism. Our framework provides all of these, with the Bradley-Terry likelihood as one possible observation model among many.

### 3.4 Multi-attribute utility and structural constraints

Keeney and Raiffa's (1976) Multi-Attribute Utility Theory provides structural constraints that make preference learning tractable. Under mutual preferential independence, utility decomposes additively:

$$U(x) = \sum_{i=1}^n w_i \cdot u_i(x_i)$$

This reduces the learning problem from an arbitrary function over the full outcome space to n weight parameters plus n single-attribute utility functions. In our framework, an MAUT decomposition is a program hypothesis — a program that predicts user behaviour by evaluating independent attribute dimensions and combining them linearly. Whether the user's actual preferences decompose this way is an empirical question that the mixture resolves: if MAUT programs predict well, they gain weight; if the user's preferences are non-decomposable, richer programs dominate.

The Bayesian formulation places a Dirichlet prior over the weight simplex and updates via observed choices. This is directly implementable in our DSL using DirichletMeasure with condition.

---

## 4. The Observation Model

### 4.1 Theoretical framework

The user's true satisfaction is a hidden state θ. The agent observes user behaviour — overrides, acceptances, timing, ratings, silence, subsequent actions — which are noisy signals about θ. The observation model P(o | θ, a) is a kernel from (preference hypothesis, action taken) to observations.

Different feedback channels contribute to the posterior through their likelihood functions:

$$b_{t+1}(\theta) \propto P(o^{explicit}_t \mid \theta, a_t) \cdot P(o^{implicit}_t \mid \theta, a_t) \cdot b_t(\theta)$$

In a correctly specified model, each channel's contribution to the posterior is automatically calibrated by its Fisher information:

$$I_k(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log P(o_k \mid \theta)}{\partial \theta^2}\right]$$

Signals with higher Fisher information shift the posterior more. Explicit feedback (ratings, overrides) has sharply peaked likelihood functions and high Fisher information. Implicit feedback (dwell time, silence) has broad, confounded likelihood functions and low Fisher information. No manual weighting is needed — the mathematics of conditioning handles calibration.

However — and this is the critical subtlety — this guarantee holds only for correctly specified likelihood models. The challenge is not the weighting but the specification. Real implicit signals are confounded: dwell time depends on task difficulty, attention, and interruptions, not just satisfaction. Marginalising over confounders:

$$P(o^{implicit} \mid \theta) = \int P(o^{implicit} \mid \theta, c) \, P(c) \, dc$$

When likelihood models are misspecified (inevitable in practice), three remedies exist within the Bayesian framework: tempered likelihoods (P(o|θ)^α with α < 1 to downweight unreliable channels), hierarchical models (priors on noise parameters, learning calibration from data), and robust Bayesian methods (sets of priors and likelihoods rather than point specifications).

### 4.2 The observation model is learned

In principle, the observation model should itself be learned — "user override means disapproval" is a hypothesis, a program in the mixture, competing under the complexity prior. The agent should discover which signals are informative through the same inference that discovers everything else.

There is a bootstrapping issue: to `condition` on an observation, you need a kernel; to learn a kernel, you need to condition on observations. The resolution is that the initial observation model is part of the prior — the simplest hypothesis about what feedback means. "Override = bad, accept = good" is a very short program with high prior weight. It is almost certainly correct. Richer models ("silence for 5 minutes then override = strong disapproval," "quick accept without reading = weak approval") are longer programs with lower prior weight that earn their complexity cost if they predict feedback patterns better.

The theoretical landscape from §3 describes what the agent is learning toward: Bayesian IRL models that infer reward from behaviour patterns, Bradley-Terry models that learn from comparisons, MAUT decompositions that identify independent utility dimensions. These are all programs in the hypothesis space, each representing a different observation model. The complexity prior governs which are worth maintaining. The agent begins with simple models and discovers richer ones as data accumulates.

### 4.3 Practical bootstrapping

For the initial implementation, explicit binary feedback (user approves or overrides) provides the cleanest bootstrapping signal. Its interpretation is nearly unambiguous — minimal observation model learning required. The Beta-Bernoulli conjugate (`TaggedBetaMeasure`) tracks P(approve | θ, action) directly.

Richer signal types — timing, override specifics (which action the user chose instead), subsequent actions, comparative feedback — enter as additional programs encoding those observation models. Each new signal type is a program that predicts user behaviour from θ; when it predicts well, it gains posterior weight and begins influencing decisions.

The information-theoretic connection: the most informative query maximises expected information gain:

$$\text{EIG}(a) = \mathbb{E}_o\left[D_{KL}\left[P(\theta \mid o, a) \;\|\; P(\theta)\right]\right] = I(\theta; o \mid a)$$

This is the mutual information between θ and the observation given the action. It equals the VOI of the action as an information-gathering operation. Explicit feedback (asking the user directly) has high EIG because the response is highly correlated with θ. Implicit observation (watching what the user does unprompted) has lower EIG but zero user-attention cost.

---

## 5. Active Preference Elicitation: When to Ask vs Act

### 5.1 The POMDP formulation

Boutilier (2002) and Chajewska, Koller and Parr (2000) established the formal framework for active preference learning. The agent maintains a prior P(u) over utility functions. Expected utility under uncertainty is:

$$EU(d, P) = \mathbf{p}_d \cdot \mathbb{E}_P[\mathbf{u}]$$

which depends only on the mean of the posterior — a consequence of EU's linearity in utilities. The Expected Value of Information of a query q is:

$$\text{EVOI}(q) = \sum_{r \in R_q} P(r \mid q) \cdot \text{MEU}(P_r) - \text{MEU}(P)$$

where P_r is the posterior after observing response r, and MEU(P) = max_d EU(d, P). The agent asks when EVOI(q) exceeds the cost of querying; otherwise, it acts.

Boutilier embeds this in a full POMDP where the hidden state is the user's utility function, actions include both queries and terminal decisions, and the Bellman equation balances information gathering against action:

$$V^*(P) = \max_{a \in Q \cup D} Q^*_a(P)$$

The myopic EVOI approximation can underestimate value when no single question changes the optimal decision but a sequence would. The full POMDP handles this correctly but is computationally harder.

### 5.2 Connection to our architecture

Our architecture implements this framework directly:

- P(u) is the MixtureMeasure over programs (each program is a utility hypothesis).
- The decision between asking and acting is made by `optimise` over an action space that includes ASK_USER alongside all domain actions. There is no separate VOI computation — ASK_USER is just another action whose EU is computed uniformly. Its EU happens to be high when the belief is uncertain (because asking guarantees correct handling) and low when the belief is concentrated (because the agent can act correctly without asking). The ask-vs-act transition emerges from EU comparison, not from a VOI side-channel.
- The posterior update after receiving the user's response is `condition(belief, response_kernel, response)`.

When the agent asks, the user reveals their preferred action. This is maximally informative: every firing program whose predicted action matches gets positive evidence, every other firing program gets negative evidence. Many programs updated simultaneously, which is why EU for asking is high early on — asking guarantees correct handling at a known cost (user attention), while acting risks incorrect handling.

Chajewska et al.'s key practical insight: when utility variables are independent, the optimal query targets the "intersection points" where two decision strategies achieve equal expected utility. Population-level data (utility functions from similar users) can initialise the prior, enabling useful behaviour before any individual-specific queries. This maps directly to our multi-user meta-learning: the population grammar provides a structured prior that concentrates on preference hypotheses that have worked for other users.

---

## 6. Embodiment: Brain, Body, Environment

### 6.1 The formal boundary

Ay (2015) formalises two ways to draw the agent boundary:

**(a)** Agent = brain + body, interacting with environment as a black box.
**(b)** Agent = brain, interacting with world (= environment + body) as a black box.

Our Bayesian core is a brain. The host is the body. The email server, the user, and the LLM APIs are the environment. The brain receives sensor signals (feature vectors) and sends effector signals (action symbols). It is causally independent of the world given its sensors, and the world is causally independent of the brain given its effectors. Crucially: the boundary between body and environment is not visible to the brain — it must be discovered through interaction.

### 6.2 What the brain does

The brain (the Bayesian inference core) operates exclusively on:

- **Observations:** feature vectors, preprocessed by the body (like the retina preprocesses light).
- **Actions:** symbols from a vocabulary (like motor commands — "extend arm", not "contract muscle fibre #4721").
- **Outcomes:** user reactions (the proprioceptive feedback — did my action achieve what I intended?).

The brain decides *what* to do. It does not know *how* the body executes its decisions. It learns what each action *achieves* (the affordance) through conditioning on outcomes.

### 6.3 What the body does

The body (the host) handles:

- **Preprocessing (sensors):** raw email → feature vector. Like the eye performing edge detection before signals reach the visual cortex. The LLM can be part of this pipeline — extracting urgency, classifying topics — just as the retina does contrast enhancement.
- **Execution (effectors):** action symbol → world effects. `:draft_response` → LLM generates text → sends email. `:archive` → API call moves email. The brain doesn't know how archiving works; it knows that `:archive` in context X tends to produce user approval.
- **Feedback routing (proprioception):** user reactions → observation delivered to brain. The body translates world events into the format the brain can process.

The body is not a dumb relay. It does real computational work. But it doesn't make decisions about *what* to do — it makes decisions about *how to do what the brain decided.*

### 6.4 LLMs as prosthetics

An LLM is a prosthetic that extends the agent's sensory or effector capabilities — like glasses, or a calculator. The agent discovers that using the prosthetic (calling the LLM) helps it see better (extract richer features) or act more effectively (generate better drafts) in certain situations. It decides when to use the prosthetic via EU, like any other action. The LLM is part of the body, not the brain.

### 6.5 Embodiment constraints and cheap design

Ay's key result: the extrinsic constraints of the body (what it can and cannot do) reduce the effective dimensionality of the control problem. The agent doesn't need to explore all possible programs — only programs consistent with what its body can actually execute. Failed actions (API errors, user confusion from malformed output) provide negative feedback, and programs that violate body constraints lose posterior weight.

This is "cheap design" — the body does some of the brain's work for free. Physical constraints reduce the hypothesis space, making inference tractable. The email API constrains what action sequences are possible; the agent discovers these constraints through use rather than being told.

### 6.6 Learning the body

A baby doesn't know what its hands can do. It learns through motor babbling — trying random actions and observing effects. Our agent does the same. The action vocabulary is given (like muscles), but what each action achieves (the affordance) is learned through conditioning on outcomes. The agent's programs are hypotheses about affordances: "when I send `:flag_urgent` in the context of an urgent-from-manager email, the effect is user approval."

If the primitives are low-level enough, higher-level actions emerge as compositions through grammar evolution — the same mechanism that discovers predicate abstractions. `:archive` could be a nonterminal compressing `(begin (move-to-folder "archive") (mark-read))`. The principled position: primitives should be as low-level as computationally feasible. The lower the primitives, the more the agent discovers through composition, and the less we smuggle in through design.

---

## 7. Programs as DSL Expressions

The credence DSL is a Lisp. The agent's hypothesis space is the space of programs in this language, weighted by description length.

### 7.1 The terminal alphabet

The terminal alphabet includes the inference primitives, control flow, predicates, and the body's action vocabulary:

```scheme
;; Axioms (the bedrock — cannot be defined in terms of each other)
(condition belief kernel observation)     cost: 1
(expect belief function)                  cost: 1
(optimise belief action-space utility)    cost: 1

;; Convenience (derived from axioms, included to keep programs short)
(voi belief kernel actions utility obs)   cost: 1

;; Body actions (effectors — given by the body, effects learned through use)
(archive)                                 cost: 1
(flag-urgent)                             cost: 1
(ask-user question)                       cost: 1
(ask-llm prompt)                          cost: 1

;; Control flow
(if predicate then else)                  cost: 1
(begin expr ...)                          cost: 1
(let ((var expr)) body)                   cost: 1

;; Predicates (sensors — domain-provided)
(gt channel threshold)                    cost: 1
(lt channel threshold)                    cost: 1
(and p q)  (or p q)  (not p)             cost: 1 each

;; Temporal operators
(changed p)  (persists p n)               cost: 1 each
```

Every symbol costs 1 in description length — the universal prior is uniform over the alphabet. The body's action primitives are at whatever level the body exposes. If the body exposes low-level operations (`move-to-folder`, `add-label`, `set-flag`), the agent composes higher-level actions through grammar evolution. If the body exposes high-level operations (`archive`, `flag-urgent`), the agent selects among them but can't compose new ones. The level of the primitives is a design choice about where to draw the body boundary.

### 7.2 Programs encode preference hypotheses

Each program in the mixture is a hypothesis about what the user wants. A simple program: `(if (gt urgency 0.7) flag-urgent archive)` — "the user flags urgent emails and archives the rest." A complex program might invoke the LLM, condition on its analysis, and decide based on multiple factors. The complexity prior governs the tradeoff: simple hypotheses are favoured unless the data demands complexity.

### 7.3 There are no "stages"

A depth-1 program is reactive. A depth-3 program is conditional. A deeper program might do deliberative reasoning — querying the LLM, conditioning on the response, computing expected utility. The agent explores program depth as part of its normal operation. The complexity prior and the learned observation model determine what depth is useful. The designer does not choose.

### 7.4 Grammar evolution and action composition

Nonterminals compress frequently useful subexpressions. Grammar perturbation extracts recurring subtrees from high-posterior programs and promotes them to named primitives. This applies uniformly to predicates and actions:

- Predicate compression: `(and (gt urgency 0.7) (gt sender_is_manager 0.5))` → `URGENT-FROM-BOSS`
- Action composition: `(begin (move-to-folder "priority") (add-label "urgent") (notify-user))` → `TRIAGE-URGENT`

The grammar evolves the agent's vocabulary of both concepts and skills — its language of thought and action.

---

## 8. The Five-Layer Stack

Concepts emerge from raw observations through five layers, each from inference under the complexity prior at a different timescale. The stack applies uniformly to perceptual abstractions and action compositions:

**Layer 0 (given): Raw primitives.** Feature predicates (`gt`, `lt`) and body actions (`move-to-folder`, `add-label` or higher-level equivalents). The terminal alphabet.

**Layer 1 (emerges): Abstractions.** Frequent patterns promoted to grammar nonterminals. "Urgent" compresses a predicate pattern. "Archive" (if low-level primitives are used) compresses an action sequence. Perceptual and motor abstractions emerge through the same mechanism.

**Layer 2 (emerges): Preference models.** Combinations of abstractions into conditional programs that predict user behaviour. Programs that predict accurately gain posterior weight.

**Layer 3 (emerges): Meta-regularities.** Across preference changes or users, certain abstractions persist. The grammar evolves a stable core alongside a mutable periphery.

**Layer 4 (emerges): Inductive bias.** The stable core IS the agent's inductive bias. New situations start with a prior concentrated on proven abstractions — both perceptual and motor.

---

## 9. Meta-Actions: The Strange Loop

The agent uses its current beliefs to decide whether to modify its own hypothesis space, and the modified hypothesis space changes its future beliefs. This self-reference — Hofstadter's "strange loop" — is not architecturally confused; it is the principled treatment of computation as action.

Russell and Wefald's metareasoning framework (1991) formalises this: computational operations are actions in a meta-level decision problem, evaluated by expected improvement in decision quality. The meta-level MDP has states (the agent's current belief/computational state), actions (computational operations or "stop and act"), and a reward for stopping equal to the value of the best available action.

In our architecture, the action space includes alongside domain actions:

- `:enumerate_more` — expand the hypothesis space at current depth
- `:perturb_grammar` — propose grammar modifications from posterior analysis
- `:deepen` — re-enumerate at greater depth where nonterminals enable it

These are evaluated by the same `compute_eu` as domain actions. The agent decides whether to think more or act now through one `argmax EU` over the combined space. The Bayesian machinery underneath — `condition`, `expect`, `optimise`, the complexity prior — is immutable. What meta-actions modify is the hypothesis space the machinery operates over: the set of programs, grammars, and depth. This is analogous to changing what you're thinking about, not how you think.

Meta-actions terminate naturally: each one reduces posterior entropy (more hypotheses → more weight on the best), which reduces the EU of further meta-actions. Cost accumulation (the user is waiting) provides additional pressure. The agent thinks as long as thinking is more valuable than acting, and stops when it isn't.

---

## 10. What Is Given vs What Is Learned

### 10.1 Given (the agent's embodiment)

- **The inference primitives:** `condition`, `expect`, `optimise`. These are immutable — the laws of thought. (`voi` is derived, in the stdlib.)
- **The complexity prior:** `2^(-|program|)`. The only inductive bias.
- **The alignment commitment:** the agent's utility equals the user's utility, inferred from behaviour.
- **The body's action vocabulary:** a set of primitive effector operations the body exposes. The agent discovers what they do and what they cost through use.
- **The body's sensor channels:** observation streams preprocessed by the body. The agent discovers which channels are informative.
- **Embodiment constraints:** what the body can and cannot do. Discovered through failed actions, not specified in advance.

### 10.2 Learned (everything else)

- What the user wants (θ — the preference parameter).
- What each action achieves (affordances — learned through conditioning on outcomes).
- What each action costs (learned through user reactions to latency, resource use).
- Which features are informative (grammar evolution over sensor configurations).
- What abstractions are useful — both perceptual and motor (grammar evolution).
- When to think vs act (meta-actions evaluated by EU).
- When to ask the user vs act autonomously (asking is an action with EU).
- When preferences have changed (temporal programs).
- How to interpret feedback signals (observation models as programs in the mixture).
- Action compositions (higher-level skills from lower-level primitives, via grammar evolution).
- The structure of the user's preferences (MAUT decomposition, if it exists, discovered not assumed).

---

## 11. Open Problems

### 11.1 Preference dynamics

All major frameworks assume static θ. Real preferences change. Our architecture handles this partially through temporal programs, but this is limited by enumeration depth. Time-Varying CIRL (Mayer 2024) extends the framework by letting θ evolve stochastically, but the computational implications are largely unexplored.

### 11.2 Human irrationality

CIRL assumes Boltzmann-rational users. Real users exhibit systematic biases — prospect theory, framing effects, loss aversion. The preference laundering problem: should the agent learn revealed preferences (what the user actually does, including biases) or idealised preferences (what they would do if fully rational)? Laundered preferences are the normative ideal but impossible to observe directly. Our framework learns revealed preferences by default. Whether and how to "launder" them is philosophically unresolved and practically consequential.

### 11.3 Convergence and safety

After convergence to confident beliefs, the uncertainty that drives safe behaviour disappears. If the agent has converged on wrong preferences, it acts confidently on wrong beliefs. The off-switch theorem's protection is proportional to belief variance — it vanishes at convergence. Ongoing monitoring and periodic re-elicitation may be necessary. This is an open problem in the CIRL literature.

### 11.4 Observation model bootstrapping

The initial observation model (explicit feedback interpreted as approval/disapproval) is the simplest viable prior. But early learning is entirely dependent on the quality of this bootstrap. If the initial model is wrong (e.g., the user approves by default and only overrides when strongly displeased), the agent may learn incorrect preferences before it has enough data to discover better observation models. Robust bootstrapping strategies — starting with high ASK_USER frequency, using multiple signal types from the beginning — may be necessary.

### 11.5 Dynamic depth and metareasoning

Nonterminal promotion becomes meaningful at depth 3+, but depth 3 is intractable under sparse grammars. The `:deepen` meta-action (§9) provides the principled mechanism: the agent decides when to increase depth based on EU over the expanded hypothesis space. Currently implemented with entropy-based EU approximation. The fully principled version would have the agent learn meta-action value from experience — meta-action EU formulas as programs in the mixture.

### 11.6 LLMs as prosthetics

An LLM extends the agent's sensory and effector capabilities (§6.4). As a sensor prosthetic, it enriches feature extraction (topic classification, urgency detection from raw text). As an effector prosthetic, it generates natural language output (drafts, explanations, questions). The agent decides when to use the prosthetic via EU. The prompt is part of the program — longer prompts cost more in description length. Both the complexity prior and learned user reactions push toward concise, targeted use.

---

## 12. Architecture

### 12.1 Three tiers

**Tier 1 (`src/`):** DSL core. Measures, kernels, `condition`, `expect`, `optimise`, `TaggedBetaMeasure`, evaluator, stdlib (which includes `voi` as a derived computation). Domain-independent.

**Tier 2 (`src/program_space/`):** Program-space inference. Grammars, enumeration, compilation, perturbation, `AgentState`. Domain-independent.

**Tier 3 (`domains/`):** Applications. Each domain provides features, terminals, actions, feedback, host driver.

### 12.2 One agent per user

Domains are event sources, not agents. A single `AgentState` per user. Every observation from every domain conditions the same belief.

### 12.3 Design invariant enforcement

**Type-level (primary):** `CompiledKernel` has no AST field. `propose_nonterminal` requires `SubprogramFrequencyTable`.

**Tests (secondary):** Observable consequences of correct implementation. 10K kernel evals < 1ms. Nonterminals are real posterior subtrees. Compression payoff is measurable.

**CLAUDE.md (tertiary):** Documents the reasoning behind the constraints.

### 12.4 Testing philosophy

**Mechanism tests:** deterministic, agent-free where possible. Failure = code is wrong.

**Emergent-behaviour tests:** statistical, directional, controlled confounds. Failure = code or causal attribution may be wrong.

**Enforcement tests:** structural. Failure = design invariant bypassed.

---

## 13. Domain: Grid World (Complete)

72 tests passing. Validated: convergence, Occam's Razor, regime-change adaptation, meta-learning (controlled comparison: perturbation agent 0.700 vs no-perturbation 0.667 in regime 3), grammar evolution (23 perturbed grammars, 4 with positive weight), scale (46K → 2K at depth 2).

---

## 14. Domain: Email Agent

### 14.1 Preference inference via action prediction

Each program is a hypothesis about what the user prefers: "for emails matching this predicate, the user chooses this action." The Beta tracks P(approve | predicate fires, action taken). The agent recommends whichever action has highest expected approval.

### 14.2 Features, actions, user profiles

13 feature channels (sender, urgency, topic, etc.). 7 actions including ASK_USER. 4 hidden test profiles (urgency_responsive, delegator, hands_on, selective). See implementation instructions for details.

### 14.3 Multi-user meta-learning

Grammar pool from one user initialises the next. Population-level structural regularities transfer; individual details don't. Chajewska et al.'s result directly applies: population data initialises the prior, enabling useful behaviour before individual-specific queries.

---

## 15. Resolved Questions

- ~~Kernel compilation~~ — `CompiledKernel` type, precompiled closures.
- ~~Temporal operators~~ — kernel closures capture temporal window.
- ~~Precompile predicates~~ — enforced by no AST field.
- ~~Program enumeration~~ — prior-weighted floor (`min_log_prior`).
- ~~Scale~~ — validated at depth 2 (46K → 2K).
- ~~Base-rate sentinel~~ — originally `log(0.5)` for non-firing predicates. Now eliminated: expression-tree programs always produce a recommendation, so every program is scored. No non-firing case.
- ~~Flat vs nested~~ — flat by design.
- ~~Per-component dispatch~~ — `TaggedBetaMeasure.tag`.
- ~~One vs many agents~~ — one agent per user, domains are event sources.
- ~~Action-predicate explosion~~ — the complexity prior handles it.
- ~~Setting the utility function~~ — CIRL: agent utility = expected user utility under preference uncertainty. Grounded by Savage (utility exists), complete class theorem (Bayesian EU is the only admissible procedure), Harsanyi (linear aggregation).
- ~~Feedback weighting~~ — Bayesian updating calibrates automatically once observation models are learned. Fisher information governs signal contribution. The observation models themselves are programs in the mixture, learned through the same inference.
- ~~Computational cost~~ — only matters insofar as it affects user satisfaction. The user's reactions encode their tolerance for cost. No separate cost model.
- ~~Brain/body boundary~~ — the brain (Bayesian core) receives features and sends action symbols. The body (host) preprocesses raw data and executes actions. The brain doesn't know how the body works; it learns what the body can do through conditioning on outcomes. Formalised by Ay (2015).
- ~~Meta-actions~~ — computational operations (enumerate, perturb, deepen) are actions in the action space, evaluated by the same EU as domain actions. The strange loop (agent modifying its own hypothesis space) is principled — the Bayesian machinery is immutable; what changes is the set of hypotheses it operates over.
- ~~Action composition~~ — higher-level actions emerge through grammar evolution, the same mechanism as predicate abstractions. The grammar evolves skills alongside concepts. Primitives should be as low-level as computationally feasible.
- ~~LLM role~~ — the LLM is a prosthetic (glasses, not a brain implant). Part of the body's sensor/effector system. The brain decides when to use it via EU.
- ~~Non-firing programs~~ — eliminated. Programs are expression trees that always produce a recommendation. Every program is scored. No `log(0.5)` sentinel needed.
