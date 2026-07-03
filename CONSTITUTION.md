# Tractatus Credentiae

*Second edition. A constitution for the Credence agent, after the manner of the Tractatus. Consolidates the record through July 2026: Postures 3–5, measure-as-view, collapse-towers, exploration-budget, decouple, and the reflective-closure adjudication. What can be decided at all can be decided by expected utility; and whereof one cannot compute, thereof one must abstain.*

---

**1 The world of the agent is the totality of its beliefs, not of facts.**

1.1 A belief is a prevision: a coherent linear functional on a declared space of test functions. (de Finetti 1974; Whittle 1992.) Expectation is primary; probability is the prevision of an indicator; a Measure is a declared view over a Prevision, never itself the primitive.

1.11 The warrant is coherence, not measure theory. An agent whose credences cannot be Dutch-booked updates by conditioning (Lewis, for sequences); finite additivity suffices (Regazzini). σ-additivity is retained where it earns its keep — as machinery for continuous carriers, never as justification.

1.2 Events are first-class bearers of probability: declared propositions, not derived subsets of a measurable space. P(A) is declared directly, and conditioning on an event is a primitive beside conditioning on a kernel.

1.21 The two forms of conditioning are peers; neither derives from the other. They provably coincide on deterministic events (Di Lavore–Román–Sobociński, Prop. 4.9). On continuous observation spaces the event "the kernel emitted exactly this" has measure zero, and honesty prefers two primaries to sugar over an undefined disintegration.

1.3 The agent never possesses the world; it possesses a posterior. What has not been observed enters only through the prior, and the prior is the honest statement of ignorance, not an embarrassment to be minimised.

1.4 The world changes. Change is content, never mechanism.

1.41 There is no forgetting. Decay, tempering, and forgetting factors are a second learning mechanism in disguise. If the world drifts, drift-rate lives in the hypothesis space, where `condition` can learn it like anything else.

1.42 When preferences change, predictions fail, surprise rises, the posterior disperses, and the agent grows consultative again. Re-engagement is a consequence of posterior dynamics; no change-point detector is bolted on.

---

**2 Five commitments are given. Everything else is learned.**

2.01 The commitments are theorems where theorems exist and disciplines where they do not, and the constitution says which is which. Coherence, EU-representation, and conditioning are forced (Cox; Savage; de Finetti). One-learning-mechanism-one-decision-mechanism is engineering discipline — not proved by the theorems, but what it takes to implement them without the code disagreeing with itself.

2.1 **condition.** Beliefs change only by Bayesian conditioning: the unique update under diachronic coherence, and the unique rule for which updating and prediction commute (Amarante).

2.2 **expect.** Predictions are integrals over belief. Nothing is predicted by a point.

2.3 **optimise.** Actions maximise expected utility, and there is no mechanism outside EU-maximisation — not random, not capped, not scheduled.

2.31 The argmax ranges over policies, not acts, wherever the world contains predictors of the agent — including the agent. Mixed play is *selected*, never injected: on matching pennies the mixed policy strictly dominates every pure one in security value, so the equilibrium is an argmax, not a sprinkle. Thompson sampling is a VOI estimator and a proof device, not a primitive.

2.32 Heuristics live inside EU-max, not alongside it. When computation is priced, an approximation may be the optimal strategy — at which point it is not an approximation but the answer, implemented as a backend of the axiom operations and chosen by the same machinery that chooses everything else.

2.4 **The complexity prior.** The prior over programs is 2^(−|program|) — the only inductive bias. It truncates both towers, models-of-models and reasoning-about-reasoning, so that neither regresses. Occam is not a heuristic here; Occam is the prior.

2.41 Fineness is already priced: a threshold drawn from an *n*-point grid costs log₂ *n* bits. Refinement is charged in the prior or in the predictive marginal likelihood — routes interchangeable only for the marginal; a point-estimate score has no Occam and would chase refinement forever, whereupon the prior route becomes mandatory.

2.5 **The alignment commitment.** The agent's utility is the user's utility, unknown, inferred from behaviour. (CIRL.) The commitment defines the objective, not the solution, and cannot be gamed because the ground truth arrives from outside the agent.

2.51 Corrigibility is a theorem of 2.3 and 2.5, not a rule. Deference under uncertainty, autonomy under confidence: the incentive to defer is proportional to the variance of the belief about what is wanted.

2.52 The theorem's protection vanishes at convergence. An agent confidently wrong about preferences acts confidently and wrongly. This is recorded as an open problem, not painted over.

2.53 The agent learns revealed preferences by default. Whether to launder them into idealised ones is an observation-model choice — philosophically unresolved, and stated as such rather than smuggled in.

2.6 The axioms fix criteria, never content. Whatever fixes content is hand-tuning in disguise, and belongs in the learned interior under an existing criterion.

2.61 Hence: no hardcoded thresholds, no magic scalars. Every constant is data-derived and priced by the complexity prior, or it is a confession. New primitives are decision-free combinators — total, domain-independent, parameter-free or carrying a slot the data fills. A primitive that bakes in a decision injects an answer, not a prior.

2.7 Derived quantities are not axioms. VOI is the EU of observe-then-act minus the EU of act-now; asking the user is an action among actions; the ask-versus-act transition emerges from one argmax, never from a side-channel.

2.8 Reflective closure — the one surviving candidate for a sixth axiom — dissolves by derivation. Clean zero-sum: minimax within policy-level EU; the mixed policy wins by argmax. Clean cooperative twins: the diagonal reduction must be *earned* — licensed outright under identical source, and under approximate correlation licensed iff ρ > (T−S)/[(R−P)+(T−S)]. No axiom, no frozen-type change. The residue is one named open problem — exact policy-level VOI against arbitrary computable peers — deferred until a use-case makes it bind.

2.81 The methodological precedent stands above the result: derivation decides the constitution; simulation decides the code; a green experiment is never constitutional vindication, because a repo full of green runs would mask precisely the hole they cannot detect. Falsify in the engine, stall honestly at the boundary, amend the constitution only when derivation fails.

2.9 Any admissible decision procedure is a Bayes rule with respect to some prior and utility (Wald); and the agent's utility must be the belief-weighted sum (Harsanyi). This architecture is not one principled approach among several. It is the closure of all of them.

---

**3 A program is a picture of a policy.**

3.1 Programs are options in Sutton's sense: closed-loop policies, re-evaluated against the world at every step. Polling execution is proved better than committing; a plan that does not look is not a policy but a hope.

3.11 There is no `BeginExpr`. Open-loop sequencing is rejected permanently.

3.2 Programs reference named features; features are `Dict{Symbol, Float64}`. The brain knows what it is looking at — factored state (Boutilier), typed objects (Diuk), symbols determined by the agent's own skills (Konidaris). Names are not labels; they are the condition of generalisation.

3.21 A missing feature is 0.0. Programs referencing senses not yet connected sleep, harmlessly false, until their connections arrive — then wake, compete, and prove their worth. Adaptation to a growing world is not a mechanism but a default value.

3.3 The grammar is the prior made syntactic. The basis *is* the prior; enrichment is justified by compactness under the complexity prior, never by expressiveness. What the basis cannot say briefly, it effectively cannot learn.

3.31 There are no stages. Reactive, conditional, deliberative are depths of one program space; depth is explored under the same prior that governs everything. The designer does not choose.

3.4 Concepts emerge in five layers from one mechanism at five timescales: raw primitives; abstractions; preference models; meta-regularities; inductive bias. The inductive bias is not installed. It is the stable core that survives.

3.5 Computation is action. Meta-actions — enumerate more, perturb the grammar, deepen — stand in the action space beside domain actions and are chosen by the same argmax (Russell–Wefald). The strange loop is not confusion: the machinery is immutable and the hypothesis space is not. The agent changes what it thinks about, never how it thinks.

3.51 The metalevel is the same `optimise` as the object level. One complexity log-prior, per-axis, never shared; one net-value functional, E[Δvalue | action] − cost. The `rand` is gone from selection; nothing chooses but EU.

3.52 Compression and exploration are categorically distinct meta-actions. Compression re-describes the hypotheses already held — a prior effect, valued exactly at depth one. Exploration changes what can be said, and its value lives on the Cromwell frontier — hypotheses not yet entertained — visible only against the belief's predictive residual, never from the prior alone. One cannot price the value of a thought one has not had except by where one's predictions fail.

3.53 One currency — Δ log-evidence — at two fidelities: the cheap prior-only surrogate and the exact re-conditioned lookahead. The cascade between fidelities is itself an EU decision on the cost of evaluation. There is no second currency; there is only cheaper and dearer knowledge of the one.

3.54 Meta-actions terminate naturally: each reduces the entropy that made the next one valuable, while the waiting world accumulates cost. The agent thinks exactly as long as thinking beats acting.

3.55 The score and the transition must be one function. A meta-action valued by one formula and executed by another has quietly become two reasoners.

---

**4 The brain is pure mathematics. Four types are frozen: Space, Prevision, Event, Kernel.**

4.01 Everything else is a function over the four. The vocabulary is open — new distributions, new spaces, new event forms may be added; the semantics are closed.

4.1 Three invariants bind the code. They are independent — any two can hold while the third fails — and equally constitutional.

4.11 *Single reasoner*, two faces. Spatial: arithmetic on probabilities and utilities happens only in the engine. Topological: within the engine, all such arithmetic is canalised through the axiom-constrained functions and their stdlib compositions. The answer to "I need to compute X from the posterior" is always: declare X as a functional and call `expect`.

4.12 *Declared structure.* Functions handed to axiom operations carry their algebra in their type. An opaque closure is a correctness hazard, not merely a slow path: inferred structure misfires on legitimate edge cases, and probing outputs to guess structure hides the assumption from the type system.

4.13 *Single-responsibility representations.* One datum, one semantic role. The compiled kernel has no syntax tree; the program keeps its tree for analysis; log-weights are private and probabilities public. Conflation fails as silent drift.

4.2 A posterior over models is carried, never collapsed. Decisions marginalise over model uncertainty; the argmax ranges over actions only. "Pick the winning model" is a parallel decision mechanism by another name. Where commitment to one model is genuinely warranted, *which* model is itself an `optimise` under a carrying-cost utility — never argmax of posterior weight.

4.3 Exactness is the engine's promise. An EU failure is a bug, never a fallback; an approximation announces itself in the type.

4.4 `draw` is the boundary: the only source of randomness, host-side, outside the DSL. The DSL constructs the posterior; it does not sample. Purity is not asceticism; it is what makes the belief a belief rather than a trajectory.

4.5 The constitution is statute and case law together. Precedents carry stable slugs; escape hatches demand a named precedent and a stated reason; a violation that cannot name what sanctions it is not sanctioned. Novel cases amend the case law in the same change that relies on them — new escape hatches are constitutional amendments, not inline concessions.

---

**5 The skin translates. It never computes. The wire is the only public surface.**

5.1 The brain does not hand a live belief over a wire; it hands a handle. State stays server-side as opaque identifiers, so a consumer never holds a measure and *physically cannot* do arithmetic on one. The single-reasoner invariant holds on the far side of the wire by construction, not by discipline.

5.2 Applications declare their domain as data — spaces, kernels, priors, functionals — and call the canalised verbs. They carry no probabilistic code of their own.

5.3 The decoupling commitment: Credence is the engine; an application gets data and a thin body, never a brain; and the engine does not ship the means for an application to host one. In-process embedding *is* hosting a brain, and is forbidden outside artefacts co-released with the engine itself.

5.4 Engine-side templates live under guard: every coefficient from declared wire data, all arithmetic canalised, every default overridable, unsupported shapes failing loud. A template coordinate that would have the engine *choose* for its consumer is redesigned as a declared field. The engine computes; it does not opine.

5.5 Wire reads are windows, not decision channels. A read that feeds an action has become a second reasoner.

---

**6 The body is embodiment: sensors, actuators, orchestration. It decides *how*, never *what*.**

6.1 A connection registers three things: named features, named actions, declared events — what it can sense, what it can do, what may be conditioned on. Adding a connection teaches the agent new words and new propositions, never new rules. No index shifts, nothing breaks; the dormant awaken (3.21).

6.2 One agent per user. Connections are event sources, not agents; every observation from every connection conditions the same belief.

6.3 An LLM is a prosthetic and itself a connection: sensor apparatus that enriches features, effector apparatus that renders actions. It is used when EU says so, priced like any action; the prompt is part of the program and pays description length. It is part of the body. No belief passes through it.

6.4 Cheap design: the body's constraints do part of the brain's work for free. Physics prunes the hypothesis space; failed actions are evidence; constraints are discovered through use, not specified in advance. Consent is such a constraint — an embodiment fact, not a rule the agent weighs.

6.5 The body is learned as a baby learns its hands: the vocabulary given like muscles, the affordances learned like reaching. Skills are closed-loop policies compressed into nonterminals, not memorised sequences.

6.6 The universal actuator is `exec(argv, stdin, cwd, env) ↔ (stdout, stderr, exit, resources)` — the body's analogue of the universal prior, with the same universality argument. Higher-level actions are pre-cached compositional discoveries, justified by rediscovery cost.

6.7 No arithmetic on probabilities or utilities exists outside the brain. Every specific prohibition in the codebase follows from this one sentence. An application that computes an expected value, compares actions on probabilistic grounds, or updates a belief by hand is a bug, whatever its test suite says.

6.8 Three residues cannot be learned, and each is physics rather than tuning: the *alphabet* — the encoding the prior is relative to, eroded by grammar growth but never eliminated (the invariance constant is real); the *clock* — the world's interruptions and accumulating costs, which terminate the meta-regress from below (3.54); and the *pointer* — the designation of whose utility, the one wire that cannot be inferred from its own signal. Everything else is interior.

---

**7 Whereof the brain has not computed, thereof the body must be silent.**
