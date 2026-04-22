# A de Finettian foundation for Bayesian agent architectures

**Status:** draft. Foundations sections (§1.1–§1.6) drafted in prose; operational consequences (§2), comparison (§3), implementation (§4), and conclusion (§5) appear as structured placeholders that name the specific claims each section will defend. Each subsequent move PR (Moves 1–8 of the Posture 3 reconstruction) updates this draft with whatever the move now justifies. Move 8's completion gate is "every section has prose, even rough."

**Target venue:** AISTATS 2027 or NeurIPS workshop 2026 fast-track. Submission expected ~October 2026.

---

## Abstract

Bayesian agent frameworks justify their existence by appealing to coherence: an agent that updates by Bayes' rule and acts to maximise expected utility cannot be Dutch-booked. But the operational substrate of those frameworks — programming languages, type systems, runtime architectures — typically inherits the *measure-theoretic* foundational story: probability spaces, σ-algebras, integration against measures. The two stories agree when they coincide operationally, but they diverge in three places that matter for agent code: conjugate update is forced into case-analytic dispatch rather than type-structural composition, mixture conditioning is implemented as duct tape on top of indicator-routing helpers rather than as a coherent operation on combinations of previsions, and exchangeability — the type-system fact that justifies de Finetti's representation theorem and underwrites partial-mixture inference — is inexpressible as anything richer than a comment in the code.

We present a reconstruction of the Credence agent architecture in which the *prevision*, in the sense of de Finetti and Whittle, is the foundational type. `expect` is defined as the action of a coherent linear functional on a declared test function space, and Measure is recovered as the restriction of a prevision to indicator functions. Conjugate update becomes a type-structural registry of `ConjugatePrevision{Prior, Likelihood}` pairs, dispatched on parametric type rather than likelihood-family case analysis. Mixture conditioning becomes a coherent operation on combinations of previsions, with `ExchangeablePrevision` carrying de Finetti's representation theorem as a constructive method. The reconstruction is operationally equivalent to the measure-theoretic implementation it replaces — every existing test passes under floating-point reassociation tolerance — but it expresses three things the measure-theoretic implementation could not.

We give three worked examples (conjugate dispatch as registry, programs-as-exchangeable-hypotheses for an email-routing agent, the prevision treatment of particle-method execution), compare against four other operationalisations of probabilistic computation (Staton's measure-categorical school, Jacobs' effectus framework, Hakaru's disintegration-first treatment, MonadBayes' score-primitive monad), and report on a reference implementation in Julia with 359 tests passing on the reconstruction.

---

## 1. Introduction

The dominant framework for justifying Bayesian agents is coherence. Cox's theorem (Cox 1946; Jaynes 2003) recovers probability from logical consistency on degrees of plausibility. Savage's representation theorem (Savage 1954; Abdellaoui and Wakker 2020) recovers probability and utility jointly from coherence on preferences over acts. de Finetti's Dutch-book argument (de Finetti 1937, 1974) recovers probability directly from no-arbitrage on betting prices — and, crucially, recovers *expectation* on the same coherence principle, applied to bets on numerical quantities rather than only on events.

The dominant *implementation substrate* for Bayesian agents — probabilistic programming languages, agent frameworks, decision-theoretic libraries — is measure-theoretic. Probability is encoded as a measure on a measurable space; expectation is encoded as integration against that measure; conditional probability is encoded as a Radon-Nikodym derivative or, in modern categorical treatments, as a kernel in a Markov category (Fritz 2020; Staton 2017). This is not a mistake: the measure-theoretic story is mathematically equivalent to the coherence story on the things they both treat. But it is a *foundational mismatch*. The agent is justified by coherence; the substrate is justified by measure theory. When the two coincide operationally, the mismatch is harmless. When they diverge, the substrate wins by default — and the agent ends up implemented in a vocabulary that obscures the things coherence would have made obvious.

This paper reports on a reconstruction. Credence is a Bayesian decision-making domain-specific language, embedded in Julia, that drives an agent architecture used in production for LLM/search routing and tested across several research environments (interactive fiction, RSS preference learning, email triage). The previous version of Credence took `Measure` as a primitive type and defined `expect` as a derived integration operation against it. The reconstruction inverts this: `Prevision` is the primitive, `expect` is its action, and `Measure` is recovered as a *view* on a prevision restricted to indicator functions. We retain Measure as a user-facing surface so existing code keeps working; the foundational status changes underneath.

Three operational pains motivated the reconstruction. **First**, conjugate update was implemented by case-analytic dispatch in `condition`, with each conjugate pair adding a new branch and the dispatch logic scattered across the codebase. **Second**, mixture conditioning was implemented by duct-taping a `FiringByTag` likelihood-family routing helper onto a `_predictive_ll` mixture-update loop, with the per-component routing semantics living on the kernel side and the mixture flattening living on the measure side — a dual residency that made each subsequent extension fragile. **Third**, exchangeability — the property that justifies treating $N$ programs as ergodic components of a mixture rather than $N$ independent hypotheses — was inexpressible in the type system; the agent's 22-program email-routing belief was implemented as a mixture-of-tagged-components by hand, with the exchangeability-within-tag-class encoded only as a comment.

The reconstruction dissolves all three. Conjugate update becomes a type-structural registry of `ConjugatePrevision{Prior, Likelihood}` pairs; adding a new pair adds a row, not a branch. Mixtures become coherent convex combinations of previsions, with `MixturePrevision` as a single primitive that owns both component-wise update and per-component routing. Exchangeability becomes `ExchangeablePrevision`, a declared subtype carrying de Finetti's representation theorem as a `decompose` method that returns the mixture-of-ergodic-components structure native to the architecture rather than reconstructed at the application layer.

The contribution is operational, not just philosophical. The reconstruction passes the existing test suite under bit-equivalence tolerance for conjugate paths and floating-point-reassociation tolerance (1e-12) for particle paths under deterministic seeding. It opens three new pieces of expressive power that the measure-theoretic implementation could not state: (i) the `ConjugatePrevision` registry as a single source of truth for fast paths, (ii) `ExchangeablePrevision` as a first-class type that gives the email agent's parallel-program belief its true category-theoretic shape, and (iii) `condition` on events as the primitive form, with parametric Bayes update derived as one instance.

The remainder of the paper is organised as follows. §1.1–§1.6 set out the foundational reconstruction: coherence as the single rationality axiom, prevision as the operator-valued primitive, conditional prevision as the primary form of `condition`, exchangeability and de Finetti's representation theorem as a constructive method, the complexity prior as a prevision over programs, and the alignment commitment recast as prevision over utility functions. §2 walks through three operational consequences in worked detail. §3 compares against four other operationalisations of probabilistic computation. §4 reports on the reference implementation. §5 concludes.

---

## 1.1 Coherence

The foundational axiom is coherence. The agent's degrees of belief and its utilities are jointly *coherent* if no finite system of bets the agent is willing to accept yields a sure loss (de Finetti 1937; Walley 1991). Coherence is the Dutch-book axiom; it is the unique justification we appeal to.

Two things follow from coherence that we use throughout the architecture. First, an agent's degrees of belief on events are probabilities (Cox 1946; de Finetti 1937): non-negative, additive on disjoint events, normalised to one on the whole sample space. Second — and this is the move that distinguishes the de Finettian view from the Kolmogorov view — an agent's *previsions* on numerical quantities (the fair prices it is willing to pay for gambles paying $f$ at resolution) are coherent linear functionals on the space of test functions over which they are declared (de Finetti 1974, vol. 1, §3; Whittle 1992, ch. 1). Probability is the prevision restricted to indicator functions; expectation is the prevision on bounded measurable test functions; the higher moments are previsions on polynomial test functions; and so on.

The Dutch-book theorem is the same theorem in both presentations: it says coherence of the betting prices is the necessary and sufficient condition for the prices to admit *any* representation as expectations against a measure. The two presentations differ in *which side they take as primitive*. The Kolmogorov side takes the measure as primitive and derives the prevision as integration against it; the de Finettian side takes the prevision as primitive and derives the measure as the restriction of the prevision to indicators. Operationally, this difference does not matter — coherence picks out the same set of admissible belief states either way. Foundationally, it changes which type appears at the bottom of the agent's source code.

We strengthen coherence with σ-continuity (Whittle 1992, Axiom A5: $p(\inf_n f_n) = \inf_n p(f_n)$ for $f_n \downarrow$) only when the application requires it. Finite coherence is the default; σ-continuous subclasses (the bounded-measurable test function space, the standard setting for Kolmogorov measures) are declared explicitly. This avoids inheriting the technical apparatus of measure theory for applications that do not need it (e.g. discrete program-space inference, where the test functions are the polynomial features over an enumerated grammar).

---

## 1.2 Prevision

A *prevision* $p$ is a coherent linear functional on a declared test function space. The test function space, $\mathcal{T}_S$, is itself a piece of declared structure over a base space $S$: it has a base set of generators (indicator functions for events in some Boolean algebra; bounded measurable functions; polynomial features; etc.) and a closure property (linear combinations, products up to some order, σ-continuous limits).

The contract on $p$ is the four coherence axioms: non-negativity ($p(f) \geq 0$ when $f \geq 0$), normalisation ($p(\mathbf{1}) = 1$), linearity ($p(\alpha f + \beta g) = \alpha p(f) + \beta p(g)$), and (optionally) σ-continuity. These axioms suffice to characterise $p$ uniquely up to its action on the test function space; the Riesz-style representation results recover a measure $\mu$ such that $p(f) = \int f \, d\mu$ when the test function space is rich enough, but the prevision is the primary object — the measure is the secondary one.

In the reconstruction, `Prevision` is an abstract type and each prevision class is a concrete subtype:

- `CategoricalPrevision{T}` — prevision over a finite space, parametrised by log-weights. Its action on `Identity()` returns the mean; its action on `Tabular(values)` returns the weighted sum.
- `BetaPrevision(α, β)` — prevision over $[0, 1]$ with the Beta density as its representing measure. Its action on `Identity()` returns $\alpha / (\alpha + \beta)$.
- `GaussianPrevision(μ, σ)`, `GammaPrevision(α, β)`, `DirichletPrevision(α)`, `NormalGammaPrevision(κ, μ, α, β)` — analogous closed-form previsions.
- `MixturePrevision(components, log_weights)` — coherent convex combination.
- `ExchangeablePrevision(component_space, prior_on_components)` — see §1.4.
- `ConjugatePrevision{Prior, Likelihood}(prior, hyperparameters)` — see §1.3.
- `ParticlePrevision(samples, log_weights, rng_seed)` — empirical prevision; its action on $f$ is the importance-weighted sample mean of $f$.
- `QuadraturePrevision(grid, weights)` — prevision approximated on a finite quadrature grid.

The test function hierarchy generalises the existing functional hierarchy of the measure-theoretic implementation. `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination` (the existing Functional subtypes) become `TestFunction` subtypes directly. `Indicator(e::Event)` is added: it is the test function that takes value 1 on the event $e$ and 0 elsewhere, and is the bridge between previsions and probabilities (i.e. between the de Finettian and Kolmogorov idioms). `OpaqueClosure(f)` remains as the escape hatch from the declared-structure discipline; using it forfeits the type-structural fast paths the rest of the hierarchy enables.

Two consequences of taking the prevision as primitive are worth stating. **First**, `expect` is no longer derived: a prevision *is* its action as `expect`. The implementation reflects this directly — there is no longer a generic `expect(::Measure, ::Functional)` dispatch with per-Measure-subtype methods; each `Prevision` subtype implements `expect(::P, ::TestFunction)` as the definition of what that prevision is. **Second**, Measure is no longer a primitive type but a *view* on a prevision restricted to indicator functions:

$$\mu(A) \;=\; \mathrm{expect}(p, \mathrm{Indicator}(A))$$

We retain `Measure` as a thin wrapper over `Prevision` so that the considerable amount of consumer code reading `weights(m)`, `mean(m)`, `m.alpha`, etc. continues to work unchanged. The de Finettian primitive surface is `expect(p, f)` and `probability(μ, e) = expect(μ.prevision, Indicator(e))`; the Kolmogorov-familiar surface is `weights(m)`, `mean(m)`, etc. Both work; they are equivalent views on a single underlying object.

---

## 1.3 Conditional prevision

Bayesian update is, in the de Finettian view, *conditional prevision* — a primitive two-place operation:

$$p(f \mid e) \;=\; \frac{p(f \cdot \mathbf{1}_e)}{p(\mathbf{1}_e)}, \quad \text{when } p(\mathbf{1}_e) > 0.$$

The reconstruction implements this with two peer primary forms of `condition` at the Prevision level:

```
condition(p::Prevision, e::Event) → Prevision       # event-form primary
condition(p::Prevision, k::Kernel, obs) → Prevision # parametric-form primary
```

The conditioning object is either an event (a declared structural predicate over the hypothesis space) or a kernel–observation pair. Both forms are primitive at the Prevision level; neither derives from the other. The reconstruction's Move 7 explicitly considered whether parametric-form could be derived from event-form via an `ObservationEvent(k, obs)` construction — unifying both paths under a single event-primary primitive — and rejected the reduction on two grounds.

First, the Event hierarchy's members (`TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`) are *structural predicates over a space*: each declares a subset of the hypothesis space, and `p(\mathbf{1}_e)` is well-defined as an expectation against that indicator. An `ObservationEvent(k, obs)` would carry a kernel and an observation value — a likelihood-structured object, categorically different from a subset-of-space predicate. Forcing it into the Event hierarchy strains the declared-structure discipline.

Second, and load-bearing: for continuous observation spaces with stochastic kernels — the common case for real-valued likelihoods — the event $\{k \text{ emits exactly } o\}$ has Lebesgue measure zero. Event-conditioning requires $p(\mathbf{1}_e) > 0$ to be well-defined; continuous observations violate this. Reducing parametric-form to event-form in that regime requires **disintegration**, which gives a coherent conditional prevision even when the conditioning event has zero mass. Disintegration is out of scope for this reconstruction (Narayanan and Shan 2020 survey the implementation approaches; its integration into the axiom layer is a separate axiom extension). Claiming parametric-form derives from event-form absent disintegration ships sugar over a mathematically undefined reduction.

The peer-primary framing is the honest alternative. The two forms are **provably equivalent on deterministic events** (Di Lavore, Román, Sobociński 2025, Proposition 4.9: Pearl's and Jeffrey's updates coincide for deterministic observations); in that regime the parametric-form $\mathrm{condition}(p, k, o)$ can be computed by constructing the indicator kernel of $\{k \text{ emits } o\}$ and conditioning on it via the event-form. The reconstruction's code path exploits this equivalence for deterministic cases (Posture 2's `indicator_kernel(e)` expansion); continuous cases go through the parametric-form primary directly, consulting the conjugate registry (§2.1) for closed-form updates or the particle method (§2.2) for non-conjugate fallbacks.

What this framing buys: conditioning on measure-zero events acquires clean failure semantics — the event-form primary raises a structured error pointing at the disintegration extension, rather than silently returning NaN or masking the boundary. The parametric-form handles continuous-observation conditioning where it is mathematically well-defined. Full unification under one primitive awaits the disintegration extension.

---

## 1.4 Exchangeability and the representation theorem

A sequence of random quantities $(X_1, X_2, \ldots, X_N)$ is *exchangeable* under a prevision $p$ if $p$ is invariant under permutations of the sequence. Exchangeability is weaker than independence (independent sequences are exchangeable, but exchangeable sequences need not be independent) and is the natural condition for an agent that is uncertain about an underlying generative process and willing to treat repeated observations symmetrically.

de Finetti's representation theorem (de Finetti 1937; Hewitt and Savage 1955; Aldous 1985) is a constructive result of striking power: every exchangeable prevision over a sequence of observations from a common space admits a canonical decomposition as a mixture of independent ergodic components. Concretely, if $p$ is exchangeable on $(X_1, X_2, \ldots) \in S^{\infty}$, there is a (possibly uncountable) prior $\pi$ over a parameter space $\Theta$ and a family of i.i.d. previsions $\{p_\theta\}_{\theta \in \Theta}$ on $S$ such that

$$p(f(X_1, \ldots, X_n)) \;=\; \int p_\theta\big(f(X_1, \ldots, X_n)\big) \, d\pi(\theta).$$

This theorem is the operational justification for "treat 22 programs as a mixture indexed by their tag class": the programs are exchangeable within tag class, and the representation theorem says the mixture-of-ergodic-components decomposition is the canonical way to express that.

In the reconstruction, `ExchangeablePrevision` is a declared subtype:

```
struct ExchangeablePrevision <: Prevision
    component_space::Space
    prior_on_components::Prevision   # prevision over ergodic component previsions
end
```

and `decompose(p::ExchangeablePrevision)::MixturePrevision` is the constructive instance of the representation theorem. For the email-agent case, where exchangeability is *tag-indexed* (programs are exchangeable within their predicate-firing tag class but not across tag classes), `decompose` returns a `MixturePrevision` whose components are the per-tag-class ergodic previsions; the existing `FiringByTag` accessor becomes the `component_prevision(p, tag)` method on `ExchangeablePrevision`, and `condition(p, TagSet(...))` (the form Posture 2 introduced) decomposes natively through the mixture rather than requiring application-level reconstruction.

What we get from declaring exchangeability in the type system: (i) the representation theorem is a method, callable when needed and guaranteed to produce the canonical decomposition; (ii) per-component routing has a typed home (`component_prevision`), eliminating the dual residency between LikelihoodFamily-side `FiringByTag` and Measure-side mixture flattening that the previous architecture papered over with the `_predictive_ll` helper; (iii) the partial-mixture machinery from chain event graph (CEG) inference (Smith and Anderson 2008; Collazo, Görgen and Smith 2018) — which is the natural framework for the email-agent's program-as-tag-class structure — has a category to attach to. The CEG connection is not a contribution of this paper, but having `ExchangeablePrevision` in the type system makes the future connection straightforward; without it, the connection would require a rewrite of the agent's belief representation.

---

## 1.5 Complexity prior

The agent's prior over programs is Solomonoff's universal prior (Solomonoff 1964; Hutter 2005), weighted by description length:

$$P(\mathrm{program}) \;\propto\; 2^{-|\mathrm{program}|}.$$

This is the maximum-entropy prior over the terminal alphabet of the DSL grammar. It is the unique computable prior dominating all other computable priors up to a constant factor; it implements Occam's razor structurally.

Under the Posture 3 reconstruction, the complexity prior is described as a *prevision over program ASTs*: the test function space is the declared set of subprogram-frequency features (used by `analyse_posterior_subtrees` and `perturb_grammar`) and the complexity scoring functions (used in enumeration). Operationally, this is identical to the previous formulation as a measure over ASTs — every existing test under `test/test_program_space.jl` passes unchanged. Ontologically, it is cleaner: the test functions that grammar perturbation analyses and the complexity scores that enumeration computes are both `expect` evaluations of declared features against the prior prevision, so they sit in the same axiom-constrained operation rather than as separate code paths.

The change is largely cosmetic and lands in Move 8 of the reconstruction. We mention it here because the consistency of the foundation matters for the paper's claim: every belief object in the architecture is a prevision (with measure-as-view available where the Kolmogorov idiom is more familiar), including the complexity prior over the program space. There is no second probabilistic primitive the framework reaches for when the measure-theoretic story becomes awkward.

---

## 1.6 Alignment

The agent's utility function is the user's utility function; the agent does not know what the user's utility function is. This is the alignment commitment, formalised as a Cooperative Inverse Reinforcement Learning (CIRL) game (Hadfield-Menell, Russell, Abbeel and Dragan 2016) in which the agent maintains a belief over the unknown utility parameter $\theta$ and acts to maximise expected utility under that belief.

Under the prevision-first reconstruction, the agent's belief over $\theta$ is a prevision $p$ on the utility-parameter space $\Theta$, and the test function space is the closure of the per-action utility functions $u_\theta(s, a)$ under linear combinations. The agent's expected utility for action $a$ in state $s$ is

$$U(a, s, p) \;=\; \mathrm{expect}\big(p, \, \theta \mapsto u_\theta(s, a)\big).$$

This is the same equation as in the measure-theoretic formulation; the change is what type the belief object is. Three properties of the alignment commitment carry over directly:

- **Deference under uncertainty.** When $p$ has high entropy on $\Theta$, the prevision of any specific action's utility is washed out by the per-$\theta$ disagreement, and the value of querying the user (computed as an expected change in the prevision after observing the user's response) is correspondingly high.
- **Autonomy under confidence.** When $p$ concentrates on a narrow region of $\Theta$, the prevision of the optimal action's utility approximates the true utility, querying yields little, and the agent acts.
- **Re-engagement after preference change.** A genuine preference shift produces low predictive prevision for observed user choices; this disperses $p$, and the agent re-enters the deferential regime without explicit change-point detection.

The off-switch theorem (Hadfield-Menell, Dragan, Abbeel and Russell 2017) and the assistance-game generalisation (Shah et al. 2020) carry over verbatim, since the only operational change is the type of the belief object.

---

## 2. Operational consequences

This section presents three worked examples of how the prevision-first reconstruction expresses things the measure-theoretic implementation could not. Each example is supported by a corresponding Move in the reference implementation.

**Section to be drafted as Moves 4, 5, 6 land.** The structure and the specific claims-to-defend are listed below.

### 2.1 Conjugate dispatch as a type-structural registry

*Claim:* In the previous implementation, conjugate Bayesian update for the canonical pairs (Beta-Bernoulli, Normal-Normal, Dirichlet-Categorical, NormalGamma-NormalGamma) was implemented by case-analytic dispatch in the `condition` method, with each pair adding a new branch and the dispatch logic scattered across nine measure subtypes and five likelihood-family routing types. Under the prevision-first reconstruction, conjugate update becomes a `ConjugatePrevision{Prior, Likelihood}` parametric type with a single `update` method per pair; dispatch is type-structural; adding a new pair adds a row to a registry, not a branch to a method.

*Worked example to land in this section after Move 4:* trace `condition(BetaPrevision(2, 3), bernoulli_kernel, 1)` step by step through the registry lookup, the `update(::ConjugatePrevision{BetaPrevision, Bernoulli}, ::Bool)` method, and the resulting `BetaPrevision(3, 3)`. Contrast with the pre-refactor dispatch path through `condition(::TaggedBetaMeasure, k, obs)` and its `likelihood_family` case analysis.

### 2.2 Mixture conditioning through exchangeability representation

*Claim:* In the previous implementation, mixture conditioning was implemented by component-wise update plus mixture flattening (in `condition(::MixtureMeasure, k, obs)`), with per-component routing handled by a separate `FiringByTag` likelihood-family type whose semantics lived on the kernel side. This was a dual residency: the kernel knew how to route observations, the measure knew how to flatten the resulting mixture, and the per-component routing logic had to coordinate across both. Under the prevision-first reconstruction, mixtures are coherent convex combinations of previsions (`MixturePrevision`), per-component routing is an accessor on `MixturePrevision` (`component_prevision(p, tag)`), and the email-agent's tag-indexed exchangeability is declared via `ExchangeablePrevision` whose `decompose` method returns the canonical mixture decomposition.

*Worked example to land in this section after Move 5:* the email agent's 22-program belief is constructed as `ExchangeablePrevision(ProgramSpace, prior_on_program_classes)`. Conditioning on a TagSet event `condition(p, TagSet({3, 7}))` decomposes through the mixture: the representation theorem returns the per-tag-class ergodic components, the conditioning event picks out the components whose tag class matches, and the resulting prevision is the renormalised conjunction. Contrast with the pre-refactor email-agent code that constructed the same posterior via application-level iteration over the measure's components.

### 2.3 Particle methods as previsions

*Claim:* Particle filtering — the standard fallback when no conjugate path applies — is, in the de Finettian view, just another prevision: the empirical mean against an importance-weighted sample is a coherent linear functional on bounded measurable test functions. The previous implementation represented particle sets as `CategoricalMeasure(Finite(samples), log_weights)`, conflating the empirical-prevision representation with the categorical-measure-on-a-finite-space representation. Under the prevision-first reconstruction, `ParticlePrevision` is its own subtype with its own `update` method (importance reweighting + optional resampling) and its own deterministic-seeding contract.

*Worked example to land in this section after Move 6:* trace a non-conjugate `condition` call from prior `ExchangeablePrevision` through the `_condition_particle` fallback, into the resulting `ParticlePrevision`, and verify that the mean (computed via `expect(p, Identity())`) matches the closed-form posterior to within `rtol = 1e-12` under the canonical RNG seed.

---

## 3. Comparison

This section compares the de Finettian foundation against four other operationalisations of probabilistic computation. The point of comparison is the foundational primitive each framework takes; *no existing PPL or categorical-probability framework takes prevision as primitive*. This makes the position genuinely novel.

**Section to be drafted as Move 7 lands and the Move 6 particle treatment is concrete.** The structure and the specific claims-to-defend are listed below.

### 3.1 Staton: measure-categorical PPL semantics

*Claim:* Staton (2017) and the surrounding measure-categorical school (Heunen et al. 2017; Vákár, Kammar and Staton 2019) take the measurable space as primitive and define probabilistic computation as morphisms in a Markov category over standard Borel spaces. This is operationally equivalent to our framework on the things they both treat — bounded measurable test functions, σ-finite measures — but it inherits the measure-theoretic vocabulary throughout. Conjugate update is not a first-class concept; exchangeability is expressible only through the measure-on-products formulation; the agent-architectural concerns (deference, dispatch, registry) are not addressed because the framework is for PPL semantics, not agent runtimes. We *do not displace* Staton's framework — the two are compatible — but we *demonstrate a different foundational starting point* with operational consequences for agent code that the measure-categorical formulation does not surface.

### 3.2 Jacobs: effectus theory

*Claim:* Jacobs (2015; Jacobs, Kissinger and Zanasi 2019) develops effectus theory as a categorical framework for both classical and quantum probability. Effectuses are categories where distributions, sub-distributions, and their conditioning operations live as structural morphisms. The framework is more abstract than ours and more ambitious in scope; it is also non-operationalised as an agent framework. The de Finettian reconstruction shares Jacobs' commitment to making the conditional operation a first-class structural object; it differs by taking a coherence-justified operator (the prevision) as primitive rather than a categorical effectus.

### 3.3 Hakaru and disintegration-first treatments

*Claim:* Hakaru (Narayanan, Carette, Romano, Shan and Zinkov 2016; Narayanan and Shan 2020) takes disintegration as the primitive operation for Bayesian inference, with measure-zero conditioning treated explicitly via a declared base measure. This is the right move for continuous-feature conditioning and is orthogonal to our reconstruction. Our Move 7 raises a structured error pointing at the disintegration extension when measure-zero conditioning is attempted; integrating Hakaru-style disintegration into Credence is a future direction. The de Finettian reconstruction is foundationally compatible with Hakaru: a disintegration kernel is just a parameterised conditional prevision in our framework, and the existing `condition(p, e::Event)` primitive extends naturally.

### 3.4 MonadBayes and score-primitive monads

*Claim:* MonadBayes (Ścibior, Ghahramani and Gordon 2015; Ścibior et al. 2018) implements probabilistic programming in Haskell with a `score` primitive that accumulates likelihood weights inside a monadic computation. The score-primitive treatment is operationally close to our particle-method path (Move 6) but is foundationally derived from the categorical-monad story rather than from coherence. The MonadBayes framework is complementary to ours: where MonadBayes targets first-class probabilistic programs as Haskell values, Credence targets a DSL-as-data-with-a-Julia-evaluator architecture suited to agent runtimes. The de Finettian reconstruction is straightforward to embed in a MonadBayes-style monadic framework; the converse (taking score as the primitive in an agent runtime that needs declared structure for fast paths) is awkward.

---

## 4. Implementation

The reference implementation is Credence: a Bayesian decision-making DSL embedded in Julia, with applications spanning LLM/search routing (the production credence-proxy gateway), interactive fiction agents, RSS preference learning, and email triage. The Posture 3 reconstruction is on the `de-finetti/migration` branch (this paper's branch); it is composed of eight implementation moves landing as a sequence of design-doc-then-code PR pairs.

**Section to be drafted as Moves 1–8 land.** Specific artefact pointers to include:

- Reference implementation: GitHub link to the merged branch, commit SHA at acceptance.
- Test suite: 359+ assertions across 7 core test files plus the POMDP-agent package's separate test suite. Three strata: unit equivalence (1e-14), composition equivalence (1e-12), end-to-end (1e-10).
- Worked example for §2.2: the email agent's 22-program belief, with before/after code snippets showing the application-level iteration in the pre-refactor code vs the `ExchangeablePrevision` declaration in the post-refactor code.
- Skin layer: the JSON-RPC API surface (`apps/skin/server.jl`) is preserved bit-for-bit; the Python-side `Prevision.beta(α, β)` idiom is a follow-up branch.

The skin smoke test surface (`apps/skin/test_skin.py`) is extended in Moves 3, 4, 6, 7 to cover the JSON-RPC paths each move changes; the Move-0 audit (`docs/posture-3/move-0-skin-surface-audit.md`) enumerates the gaps and attaches each to the corresponding move.

---

## 5. Conclusion

**Section to be drafted in Move 8.** Two sentences of structure for now:

The de Finettian foundation gives Bayesian agent architectures a coherence-justified primitive (prevision) at the bottom of the type system, with three operational consequences — type-structural conjugate dispatch, native exchangeability with a constructive representation theorem, and event-primary conditioning with parametric Bayes update derived — that the measure-theoretic foundation could not express. The reconstruction is operationally equivalent to the measure-theoretic implementation it replaces and demonstrates that no operational ground is given up by the foundational shift.

---

## References (partial — to be completed in Move 8)

- Aldous, D. (1985). Exchangeability and related topics. *École d'Été de Probabilités de Saint-Flour XIII*, Lecture Notes in Mathematics 1117. Springer.
- Abdellaoui, M. and Wakker, P. P. (2020). Savage for dummies and experts. *Journal of Economic Theory* 186.
- Collazo, R. A., Görgen, C. and Smith, J. Q. (2018). *Chain Event Graphs.* Chapman & Hall/CRC.
- Cox, R. T. (1946). Probability, frequency, and reasonable expectation. *American Journal of Physics* 14(1).
- de Finetti, B. (1937). La prévision: ses lois logiques, ses sources subjectives. *Annales de l'Institut Henri Poincaré* 7.
- de Finetti, B. (1974). *Theory of Probability* (2 vols). Wiley.
- Di Lavore, E., Román, M. and Sobociński, P. (2025). Partial Markov categories. *arXiv preprint* arXiv:2502.03477.
- Fritz, T. (2020). A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. *Advances in Mathematics* 370.
- Hadfield-Menell, D., Russell, S., Abbeel, P. and Dragan, A. (2016). Cooperative inverse reinforcement learning. *NeurIPS*.
- Hadfield-Menell, D., Dragan, A., Abbeel, P. and Russell, S. (2017). The off-switch game. *IJCAI*.
- Hewitt, E. and Savage, L. J. (1955). Symmetric measures on Cartesian products. *Trans. AMS* 80.
- Heunen, C., Kammar, O., Staton, S. and Yang, H. (2017). A convenient category for higher-order probability theory. *LICS*.
- Hutter, M. (2005). *Universal Artificial Intelligence.* Springer.
- Jacobs, B. (2015). New directions in categorical logic, for classical, probabilistic and quantum logic. *Logical Methods in Computer Science* 11(3).
- Jacobs, B., Kissinger, A. and Zanasi, F. (2019). Causal inference by string diagram surgery. *FoSSaCS*.
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science.* Cambridge.
- Lewis, D. (1999). Why conditionalize? In *Papers in Metaphysics and Epistemology.* Cambridge.
- Narayanan, P., Carette, J., Romano, W., Shan, C.-c. and Zinkov, R. (2016). Probabilistic inference by program transformation in Hakaru (system description). *FLOPS*.
- Narayanan, P. and Shan, C.-c. (2020). Symbolic disintegration with a variety of base measures. *ACM TOPLAS* 42(2).
- Russell, S. (2019). *Human Compatible.* Viking.
- Savage, L. J. (1954). *The Foundations of Statistics.* Wiley.
- Ścibior, A., Ghahramani, Z. and Gordon, A. D. (2015). Practical probabilistic programming with monads. *Haskell Symposium*.
- Ścibior, A., Kammar, O., Vákár, M., Staton, S., Yang, H., Cai, Y., Ostermann, K., Moss, S. K., Heunen, C. and Ghahramani, Z. (2018). Denotational validation of higher-order Bayesian inference. *POPL*.
- Shah, R., Krasheninnikov, D., Alexander, J., Abbeel, P. and Russell, S. (2020). Preferences implicit in the state of the world. *ICLR*.
- Smith, J. Q. and Anderson, P. E. (2008). Conditional independence and chain event graphs. *Artificial Intelligence* 172.
- Solomonoff, R. (1964). A formal theory of inductive inference, parts I and II. *Information and Control* 7.
- Staton, S. (2017). Commutative semantics for probabilistic programming. *ESOP*.
- Vákár, M., Kammar, O. and Staton, S. (2019). A domain theory for statistical probabilistic programming. *POPL*.
- Wald, A. (1950). *Statistical Decision Functions.* Wiley.
- Walley, P. (1991). *Statistical Reasoning with Imprecise Probabilities.* Chapman & Hall.
- Whittle, P. (1992). *Probability via Expectation* (3rd ed.). Springer.
