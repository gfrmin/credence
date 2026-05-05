# Move 2 design — Phase B2: category inference (OQ1 + OQ2 evidence surface)

Status: design doc (B2a). Implementation lands in B2b.

Template reference: `docs/posture-3/DESIGN-DOC-TEMPLATE.md` (adapted —
this is a methodology design doc, not a code-refactor design doc; section
shape follows the B2a session prompt).

Master plan reference: `docs/paper1/master-plan.md` § "B2 — Category
inference".

This doc surfaces the evidence Guy needs to resolve OQ1 (architectural
location of category inference) and OQ2 (form of the Bayesian
classifier) jointly, in conversation. It does not recommend.

---

## 1. Context

The Phase B master plan operationalises the new Paper 1 thesis: that
Bayesian VOI tool selection occupies a non-empty region of the
cost-performance Pareto frontier *under fair conditions*. "Fair"
includes condition (a): categories are not given to any agent — every
agent must infer the category from question content. Move B2's
deliverable is the category-inference component that emits a
per-question category posterior (or hard label, depending on OQ5)
consumable by the Bayesian agent and the LLM agents in equivalent form.

OQ1 and OQ2 fix two architectural decisions that ripple through B2b's
implementation, B3's slice construction (whose questions need labels for
calibration), and B4's LLM-prompt symmetry audit:

- **OQ1.** Where does category inference live — environment-side
  deterministic, environment-side learned-and-frozen, or body-side with
  identical implementation per agent?
- **OQ2.** What is the parametric form of the Bayesian classifier —
  full Bayesian via HMC, conjugate via DSL primitives, or MAP +
  Laplace?

This document surfaces the evidence to let those resolutions happen in
conversation. It splits OQ2's option (b) ("conjugate") into two
sub-options based on the B1 pre-flight finding that the master plan's
sub-bullet is ambiguous between a generative and a discriminative
model — see §3.

The master plan and Paper 1 prose currently use the phrase "Bayesian
multinomial logistic" generically. §2 below names the four candidate
models specifically. Whichever option is resolved, the paper's prose
needs to change to the specific name; reviewers reading "multinomial
logistic" with Gaussian Naive Bayes in the code will not be charitable.

---

## 2. The four options, mathematically

For all four: the agent sees an embedding `e ∈ R^d` of the question and
infers a posterior over categories `c ∈ {c_1, …, c_K}` (K = 5 in the
qa_benchmark: factual, numerical, recent_events, misconceptions,
reasoning).

### 2(a) Bayesian multinomial logistic via HMC over softmax coefficients

Model:

```
P(C = c_k | e, β)  =  exp(β_k · e) / Σ_j exp(β_j · e)
β_k ~ N(0, σ²_β · I)         for k = 1 … K
```

Posterior `P(β_1..K | calibration_data)` has no closed form (the softmax
is not conjugate to a Gaussian prior on β). HMC samples
`β^(s) ∼ P(β | data)` with the unnormalised log-density

```
log P(β | data) = Σ_i log P(c_i | e_i, β) + Σ_k log N(β_k; 0, σ²_β·I) + const
```

At inference time, `P(C | e_new) = (1/S) Σ_s softmax(β^(s) · e_new)` —
Monte Carlo over the posterior samples. Calibration evaluation needs
enough HMC samples per question to make MC noise small relative to
between-question signal: typically S = 1000–4000 with
no-U-turn / dynamic-HMC, post-warmup.

### 2(b-NB) Gaussian Naive Bayes on embeddings with Dirichlet class prior

Model (with conditionally independent dimensions — the "naive"
assumption):

```
P(C = c_k | π)            =  π_k                 with  π ~ Dirichlet(α_0)
P(e_j | C = c_k, μ_kj, τ_kj)  =  N(μ_kj, 1/τ_kj)  with  (μ_kj, τ_kj) ~ NormalGamma(κ_0, μ_0, α_0_NG, β_0)
```

Conjugate updates land independently on each (k, j):

```
NormalGamma(κ, μ, α, β)  +  obs r  →
    κ' = κ + 1
    μ' = (κ μ + r) / κ'
    α' = α + 1/2
    β' = β + κ (r − μ)² / (2 κ')
```

(`src/conjugate.jl:85-93`, NormalGamma + NormalGammaLikelihood pair.)

Class prior updates independently:

```
Dirichlet(α)  +  obs c_k  →  Dirichlet(α + e_k)
```

(`src/conjugate.jl:66-72`, DirichletPrevision + Categorical pair.)

At inference time, the per-dimension marginal predictive is Student-t
(integrating the joint over μ, τ):

```
P(e_j_new | data, C = c_k)  =  t_{2α'_kj}( e_j_new ; μ'_kj , β'_kj (κ'_kj+1)/(α'_kj κ'_kj) )
```

and the posterior is

```
P(C = c_k | e_new, data)  ∝  (α'_k / Σ α') · ∏_{j=1}^d t-pdf( e_j_new ; … )
```

Closed form throughout. No external dependency, no HMC, no PG
augmentation.

### 2(b-PG) Discriminative multinomial logistic via Pólya-Gamma augmentation

Model: same softmax as (a), but with stick-breaking (Linderman,
Johnson, Adams 2015) turning the K-class problem into K−1 binary
logistic problems. Each stage k = 1 … K−1:

```
P(C ≥ c_{k+1} | C ≥ c_k, e, β_k)  =  σ(β_k · e)
β_k ~ N(0, Σ_β)
```

Pólya-Gamma augmentation (Polson, Scott, Windle 2013) introduces
auxiliary variables `ω_ik ~ PG(1, β_k · e_i)` such that, conditional on
ω, the posterior of `β_k` is Gaussian:

```
β_k | ω, data ~ N(m_k, V_k)
    V_k = (Σ_β^{-1} + Σ_i ω_ik e_i e_i')^{-1}
    m_k = V_k (Σ_β^{-1} m_β + Σ_i (y_ik − 1/2) e_i)
```

where `y_ik = 1[C_i ≥ c_{k+1}]` is the binary outcome at stage k.
Inference is Gibbs alternating between PG draws and Gaussian β
updates — closed-form *conditionally* on ω, but requires a PG sampler.
At test time, posterior predictive is computed by averaging
`σ(β_k^(s) · e_new)` across Gibbs samples and composing the K−1
stick-break stages.

### 2(c) MAP estimate of multinomial-logistic coefficients with Laplace approximation

Model: same softmax + Gaussian prior as (a). Optimisation rather than
sampling:

```
β̂  =  argmax_β  [ Σ_i log P(c_i | e_i, β)  +  Σ_k log N(β_k; 0, σ²_β·I) ]
```

The objective is concave (log-concave likelihood × log-concave prior →
concave log-posterior), so Newton with a closed-form Hessian
converges in O(K · d²) per iteration; typical convergence in <50
iterations.

Laplace approximation around `β̂`:

```
P(β | data) ≈ N(β̂, H^{-1})
    where  H = -∇² log P(β | data) | β=β̂
```

At inference time, `P(C | e_new)` either uses the plug-in
`softmax(β̂ · e_new)` (MAP point estimate) or integrates via Gauss-Hermite
quadrature on the Laplace Gaussian; the latter gives the "Bayesian"
posterior the option name advertises. The "uncertainty" in the
Bayesian sense is the Gaussian curvature at the mode — accurate when
the true posterior is approximately Gaussian, which holds for
multinomial logistic with reasonable data sizes.

What (c) gives up vs. (a): the Laplace approximation's tails are
Gaussian; the true posterior's are not. This biases tail-predictive
quantities (e.g. P(C = c_k) for low-probability k); for the *modal*
prediction it agrees with (a) up to MC noise.

What (c) gives up vs. (b-NB): the discriminative form does not model
`P(e | C)`, so embeddings drawn from a different distribution than
calibration produce no out-of-distribution warning. (b-NB)'s Student-t
likelihoods drop sharply outside the training support and produce
visibly diffuse posteriors there — a free OOD signal.

---

## 3. The DSL-stdlib reduction question for (b)

The B1 master plan §4 OQ2(b) sub-bullet sketched the conjugate
reduction as

> a `MixtureMeasure` of `CategoricalMeasure` over category-conditional
> feature distributions plus a `Kernel` from features to category —
> `condition`/`expect`/`voi` exist already and would do the inference
> without new primitives.

**That description points at (b-NB), not at (b-PG).** The plumbing it
sketches — category-conditional feature distributions plus a kernel
back to categories — is generative classification, not discriminative
multinomial logistic. Posterior over class given features by
Bayes-flipping the generative likelihood is a different model from
softmax over linear features.

The B2a session ran the plumbing for both:

### 3.1 (b-NB): the reduction works

Working prototype at
`papers/paper1/scratch/category-inference-b/gaussian_nb_prototype.jl`,
74 lines of inference code (excluding driver and pretty-printing).
Calibration uses two existing conjugate paths:

- Per-(class, dimension) `NormalGammaMeasure` updates dispatch through
  `maybe_conjugate(::NormalGammaPrevision, ::Kernel)` with
  `likelihood_family = NormalGammaLikelihood()`. Closed-form posterior.
- Class-prior `DirichletMeasure` updates dispatch through
  `maybe_conjugate(::DirichletPrevision, ::Kernel)` with
  `likelihood_family = Categorical(Finite(CATS))`. Closed-form
  posterior.

Inference of `P(C | e_new)` builds a `Kernel` from `Finite(CATS)` to
`Euclidean(d)` whose `log_density` evaluates the per-dimension Student-t
marginal predictives (closed-form NormalGamma marginal). The
`condition(::CategoricalMeasure, ::Kernel, e_new)` path
(`src/ontology.jl:763-771`) dispatches on `log_density` directly — it
does not consult `maybe_conjugate`, so the Kernel's
`likelihood_family = Flat()` is correct.

The full inference flow in DSL idiom:

```julia
# Calibration — every belief update is condition() on existing primitives.
class_prior = DirichletMeasure(Simplex(K), Finite(CATS), ones(K))
params = [NormalGammaMeasure(1.0, 0.0, 2.0, 2.0) for _ in 1:K, _ in 1:D]

for (c, e) in train_data
    class_prior = condition(class_prior, CLASS_OBS_KERNEL, c)
    cat_idx = findfirst(==(c), CATS)
    for j in 1:D
        params[cat_idx, j] = condition(params[cat_idx, j], SCALAR_OBS_KERNEL, e[j])
    end
end

# Inference — one condition() call per question.
cat_prior = CategoricalMeasure(Finite(CATS),
    log.(class_prior.alpha ./ sum(class_prior.alpha)))
nb_kernel = Kernel(Finite(CATS), Euclidean(D),
    c -> error("generate not exercised"),
    (c, e) -> sum(ng_predictive_logpdf(params[findfirst(==(c), CATS), j], e[j])
                  for j in 1:D);
    likelihood_family = Flat())
posterior = condition(cat_prior, nb_kernel, e_new)
```

Every primitive in this fragment exists in `src/`. No additions.

On synthetic 5-cat × 8-dim well-separated data the prototype achieves
25/25 accuracy with mean P(true) = 0.998. **The 100% is not the point.**
The point is: every belief update is a `condition()` call into the
existing primitives; no host-side weight surgery, no new
`LikelihoodFamily`, no new `ConjugatePrevision` pair, no new `update()`
method. The reduction IS as clean as the master plan sub-bullet
implied — for the (b-NB) reading.

### 3.2 (b-PG): the reduction does not work as primitives currently stand

Sketch at `papers/paper1/scratch/category-inference-b/polya_gamma_gap.jl`
(intentionally non-running). The missing pieces:

1. **A new `LeafFamily`** `PolyaGammaBinary <: LeafFamily` (and a
   stick-breaking variant `PolyaGammaStick(stage::Int)`). The
   `LikelihoodFamily` hierarchy is part of the constitutional
   declared-structure surface (Invariant 2): every `Kernel` carries one
   at construction, condition() routes on it. Adding one is a public
   addition to a frozen-ish hierarchy.
2. **A multivariate-Gaussian Prevision.** `β_k ∈ R^d` posterior under PG
   augmentation is genuinely multivariate Gaussian with off-diagonal
   covariance. The existing `GaussianPrevision` is scalar; a
   `ProductPrevision` of d scalar Gaussians discards the off-diagonal,
   producing a wrong posterior. A new
   `MultivariateGaussianPrevision <: Prevision` with Cholesky-form
   storage (~200 lines) is the honest path.
3. **A new `ConjugatePrevision` pair**
   `(MultivariateGaussianPrevision, PolyaGammaBinary)` plus its
   `update()` method. The PG-augmented update is conjugate
   *conditional on the auxiliary ω draw* — `update()` becomes
   stochastic, breaking the deterministic-closed-form contract that
   other registry pairs maintain (see `src/conjugate.jl:21-29` for
   BetaBernoulli — pure deterministic update). The honest framing is
   that PG-Gaussian inference is a Gibbs sampler over (β, ω) and lives
   as a `ParticlePrevision`, not in the closed-form registry — but
   that means the registry entry is misleading and the model is
   structurally outside the conjugate fast path.
4. **A Pólya-Gamma sampler.** Not in `Distributions.jl`. The standard
   Julia implementation is a 3rd-party package
   (PolyaGammaHybridSamplers.jl, ~500 lines, depends on
   Distributions.jl) which would either become a new dependency in
   `Project.toml` or be vendored into `src/`.
5. **Multinomial composition.** Stick-breaking turns K-class into K−1
   binary PG problems. Either a `MixturePrevision` of K−1
   `MultivariateGaussianPrevision`-conditioned-on-PolyaGammaStick
   components, or a new `StickBreakingPrevision` type. More structural
   surgery either way.

The master plan §5 says "DSL primitive additions — none expected. If
category inference cannot be expressed in existing primitives that is
itself a finding worth separate discussion." That separate discussion
is what (b-PG) would require.

### 3.3 Implication for paper prose

The B1 pre-flight read "Bayesian multinomial logistic" from the master
plan as a single option labelled (b). The B2a evidence shows it
actually denotes two materially different models, of which only (b-NB)
fits the existing primitives. **Paper 1's §3 prose currently says
"Bayesian multinomial logistic" without disambiguation; that string
will need to change** to whichever specific model OQ2 resolves to. The
naming distinction is load-bearing — a reviewer who sees "multinomial
logistic" and finds Gaussian Naive Bayes in the code is correct that
those are different models. Phase D's LaTeX rewrite already in scope;
the rename rides along.

---

## 4. The OQ1 architectural-location options

For each option below: *where the inference module lives*, *how each
agent invokes it*, *how fairness is audited*, *what doors it leaves
open or closes for Paper 3 / Paper 6*.

### 4(a) Environment-side, deterministic per question

**Where:** category labels become a per-question attribute.
`Question.category` already exists at
`apps/julia/qa_benchmark/environment.jl:66-73`; this option keeps it
authoritative — no inference happens at evaluation time, ever.

**Bayesian agent invocation:** unchanged from v1
(`apps/julia/qa_benchmark/host.jl:58`: `cat_idx = findfirst(==(q.category), CATEGORIES)`).

**LLM agent invocation:** the LLM's user message includes the category
label.
`build_user_message` at `apps/julia/qa_benchmark/llm_agent.jl:206-209`
becomes "Category: factual\n\nQuestion: …" prefixed.

**Fairness audit:** trivially symmetric — both agents read the same
exact-string label.

**Future-proofing for Paper 3 / Paper 6:** *closes the door on online
category learning entirely.* The label is a deterministic lookup, not
a learned inference; nothing in the architecture supports updating
categories from data. Paper 3's joint category + reliability
inference would need to discard this entire module and reintroduce
inference at that point.

**Compatibility with the paper's claim:** the paper's "Bayesian
multinomial logistic" prose evaporates — there is nothing Bayesian
about a hardcoded label. The Phase B thesis's condition (a) ("agents
infer category from content") is also violated: agents do not infer,
they receive. This option is faithful to fairness but not to the
methodology.

### 4(b) Environment-side, learned once on a held-out calibration set, frozen and shared

**Where:** a new module `apps/julia/qa_benchmark/category_inference.jl`
holds the trained classifier and exposes
`infer_category(question_text) → CategoricalMeasure`. Loaded at
host-startup time; calibrated against a held-out (per OQ3) labelled
question set; the resulting parameters are saved to disk and shipped in
the repo.

**Bayesian agent invocation:** the host calls
`category_posterior = infer_category(q.text)` at the top of the
per-question loop and passes it to `agent.bdsl` (either as a hard-label
substitute for `cat_idx` or, with a new agent.bdsl signature, as a
posterior to propagate).

**LLM agent invocation:** `build_user_message` reads the same
`category_posterior` and serialises it into the prompt per OQ5.

**Fairness audit:** symmetric — both agents read from a single shared
`category_posterior`. Auditable by code review (one module, called
twice). Behavioural parity test: assert that both agents see byte-equal
serialisations of the posterior for the same question + seed.

**Future-proofing for Paper 3 / Paper 6:** *partial.* The module exists
and is named; future work can replace its frozen-at-train-time
posterior with an online-updating one. But the `apps/julia/qa_benchmark/`
location is *brain-side* per `SPEC.md` §12.1 / §6, which means the
inference is *brain-internal*. SPEC §4.2 says perception lives in the
body. So this option violates §6 (perception in the body) at the
location level even though it preserves it at the inference level
(perception is still computed; it's just done at calibration time).
Paper 3 / Paper 6 would inherit the location violation if it doesn't
refactor.

**Compatibility with the paper's claim:** the paper's "Bayesian"
language stands; the model is genuinely Bayesian. The Phase B thesis's
condition (a) is satisfied at the methodology level: agents see
category information that is *inferred*, not stipulated, even though
the inference happened at calibration time. The Phase B thesis's
condition (c) — agents see equivalent information — is the strongest
under (b).

### 4(c) Body-side, identical implementation across agents

**Where:** a new module `apps/python/credence_router/category.py` or
similar in the *body* (per `SPEC.md` §12.1: "Body — user-facing
surfaces, prosthetics, connections — talks to the skin, never to
Measures directly"). For the qa_benchmark specifically this is awkward
because qa_benchmark is currently brain-side
(`apps/julia/qa_benchmark/`); putting category inference body-side
means either (i) splitting the qa_benchmark across the
brain/skin/body boundary, with the host calling a Python body module
through some mechanism, or (ii) re-locating the qa_benchmark to
follow the credence_router architecture. Both are larger
re-architectures than the master plan scopes.

**Bayesian agent invocation:** through the body-side module, returning
a posterior per question.

**LLM agent invocation:** through the same body-side module — *the same
function call*. This is the option's strongest fairness story: identity
of code path, not just identity of output.

**Fairness audit:** strongest of the three. Both agents call the
same function; behavioural parity is structural rather than
verified-per-test.

**Future-proofing for Paper 3 / Paper 6:** strongest. The body-side
module is the natural home for online inference; Paper 3 / Paper 6
would extend rather than replace.

**Compatibility with the paper's claim:** the paper's "Bayesian"
language stands. SPEC alignment is the cleanest of the three — perception
in the body matches §6 and §12.1 exactly. The Phase B thesis is
satisfied on all conditions.

**Cost issue.** The qa_benchmark currently has no body. The skin layer
exists but the qa_benchmark doesn't use it (no `SkinClient`, no
JSON-RPC traffic). Option (c) requires either standing up the
brain↔skin↔body wiring for qa_benchmark for the first time, or
choosing a different — easier but architecturally weaker — placement
that calls itself "body-side" while living in the same Julia process
(e.g. a new `apps/julia/qa_benchmark/category_inference.jl` that's
notionally body-side but in-process). The latter is a fudge that
amounts to (b) with a different file name; the former is a move
larger than B2 was scoped for.

---

## 5. Joint OQ1 × OQ2 cost matrix

Implementation effort estimates in person-days, against the existing
codebase. Assumes embedding dimension ~384 (sentence-transformers
default), embedding model frozen and pre-computed for the question bank
once, ~50–100 calibration questions. Per-cell estimate convention:
median expected days; ranges where the median is genuinely uncertain.

| | (a) HMC | (b-NB) Gaussian Naive Bayes | (b-PG) Pólya-Gamma | (c) MAP + Laplace |
|---|---|---|---|---|
| **(a) env, deterministic** | n/a — model irrelevant under fixed labels | n/a | n/a | n/a |
| **(b) env, learned & frozen** | 4–6 d (incl. dep) | 1–2 d | 8–14 d (incl. PG sampler + new prim types) | 2–3 d |
| **(c) body-side identical** | 5–8 d (incl. dep + body wiring) | 2–3 d (incl. body wiring) | 10–17 d (incl. PG + body wiring) | 3–5 d (incl. body wiring) |

**Per-cell notes.**

- `(a)·env`: 1–2 d for the HMC integration (`AdvancedHMC.jl` +
  `LogDensityProblems.jl`, lighter than Turing.jl). Plus 1–2 d to wrap
  the resulting samples as a Measure consumable by the agents.
  Calibration eval ~1 d on top.
- `(b-NB)·env`: directly enabled by the working scratch prototype.
  Most of the cost is replacing synthetic embeddings with a real
  sentence-transformers model (Python or Julia bindings — half a day),
  plus the calibration-set construction (OQ3-dependent), plus the
  per-agent invocation wiring, plus calibration-quality evaluation.
- `(b-PG)·env`: the missing-pieces list in §3.2 dominates. New
  `LeafFamily` (1 d), `MultivariateGaussianPrevision` with Cholesky
  storage (3–4 d), conjugate-pair entry + update (2 d), PG sampler
  decision (vendor or dep, 1–2 d), stick-breaking composition (1 d),
  test coverage at the strata-2 level for any new primitive type per
  the Posture 3 conventions (2–4 d). The lower bound assumes the
  Posture-3 strata-2 conventions can be relaxed for an scratch-status
  primitive (they probably cannot). Larger uncertainty than the other
  cells — risk that a new substrate type (`MultivariateGaussianPrevision`)
  triggers a Posture-5-shaped review process is real.
- `(c)·env`: ~150–300 lines of Julia for Newton + Hessian + Laplace.
  No external deps. The Laplace integration at inference time is a
  small Gauss-Hermite quadrature in K dimensions (or plug-in MAP).
- All `·body-side` cells add 1–2 d of brain↔skin↔body wiring for the
  qa_benchmark — currently absent. This includes: SkinClient
  integration test, protocol additions to surface category posteriors,
  build/run the body container.

**Robustness to estimate error.**

- `(b-NB)·env` is the most reliable estimate — the prototype already
  runs, and the path from prototype to integration is mechanical
  (replace synthetic embeddings with real ones, wire to the agents).
- `(c)·env` is the second-most reliable — pure Julia, well-precedented
  in the literature.
- `(a)·env` has medium uncertainty — adding HMC infrastructure is the
  largest unknown; if `AdvancedHMC.jl` integrates cleanly the lower
  bound holds.
- `(b-PG)·*` is the most uncertain. The estimate assumes new primitive
  types can be added without escalating to a Posture-3-style move-doc
  review process. If the master plan's "no new primitives" line is
  enforced strictly (and it should be — see `feedback_dsl_optimization_invisible`,
  `feedback_no_hack_shortcuts`), (b-PG) becomes effectively infeasible
  inside Phase B and either falls out of consideration or warrants its
  own pre-Phase-B substrate move. Cell value should not be read as
  "expensive but doable in B2b" — read it as "this many days IF
  primitive surgery is licensed; not licensed under the current master
  plan."

---

## 6. Recommended next-decision surface

This section summarises trade-offs without ranking. Guy resolves OQ1
and OQ2 in conversation after reading.

### What the evidence shows

- **(b-PG) is the heaviest option by a lot, and it requires DSL
  surgery the master plan rules out.** §3.2 spells out five missing
  pieces; §5 puts the lower-bound cost at >8 days even if the
  primitive-additions are licensed. If they are not licensed,
  (b-PG) requires a separate substrate move (analog to a Posture-N
  move) before B2b can even start.
- **(b-NB) fits the existing primitives cleanly.** §3.1's working
  prototype shows the inference plumbing dispatches through the
  conjugate registry without modification.
- **(a) HMC is a one-time substrate cost.** Adding HMC infrastructure
  is 1–2 days but it is *real* substrate work — the dependency lives
  forever, and it changes Credence's axiomatic-function strategy
  surface. Worth doing if HMC will be needed for other models too;
  speculative substrate addition if (b-NB) or (c) suffices for B2.
- **(c) MAP+Laplace is the cheapest model in implementation cost** but
  the weakest "Bayesian story" — Laplace is an approximation, not the
  true posterior. The paper's prose can describe it as "Gaussian
  approximation to the posterior" honestly without losing the
  decision-theoretic argument; the methodological tension is whether
  Phase B's "principled Bayesian inference" claim survives the word
  "approximation".

### Trade-offs to weigh in conversation

- **(a) gets the strongest Bayesian story** (full HMC posterior
  sampling), but **costs** an external dependency and substrate
  expansion work that the master plan did not budget. 4–6 days is
  not an off-cuff estimate; it includes wrapping HMC samples as a
  Measure consumable by the agents.
- **(b-NB) gets a real conjugate Bayesian story that fits existing
  primitives exactly**, with **the lowest cost of any honest option
  (1–2 days under env-frozen, 2–3 under body-side)**. The cost is that
  the model is *generative* — it learns `P(e | C)` per class. Real
  text embeddings under sentence-transformers may not be even
  approximately Gaussian within a class, so calibration may underperform
  what (a) or (c) would give on the same data. The prototype's 100% on
  synthetic data does not predict real-world performance. The
  paper's prose needs to change from "multinomial logistic" to
  "Gaussian Naive Bayes" — a load-bearing rename.
- **(b-PG) gets the discriminative Bayesian story the paper currently
  describes**, at a cost the master plan rules out. If Guy decides
  the paper's argument *requires* the discriminative form, the
  consequence is that B2 cannot proceed under the current master plan
  and either the master plan changes or B2 falls back to (a) or (c).
- **(c) gets a cheap, defensible model with substantive Bayesian
  language** ("Laplace approximation to the posterior") at the cost of
  the "principled Bayesian inference" string in the paper getting
  qualified. The MAP point estimate at decision time is not Bayesian;
  the Gauss-Hermite integration over the Laplace is.

### OQ1 × OQ2 interaction

- **(a)·env+deterministic doesn't make sense** — there is nothing to
  do. Eliminated.
- **OQ1·(a) (env-side deterministic) ALSO makes the OQ2 question
  vacuous.** No model is fitted. The Phase B thesis's condition (a)
  fails ("agents infer category from content"); the methodology
  collapses. Eliminated as a coherent option.
- **The remaining live options are six:** OQ1 ∈ {env-frozen,
  body-side} × OQ2 ∈ {(a), (b-NB), (b-PG), (c)}. (b-PG) cells under
  current master-plan rules are infeasible.

So the practical decision surface is **2 × 3 = 6 cells** (or 2 × 4 = 8
if (b-PG) is brought back via a separate substrate move).

The 6 live cells:

|         | (a) HMC      | (b-NB) Gaussian NB | (c) MAP + Laplace |
|---------|--------------|--------------------|-------------------|
| env-frozen   | 4–6 d        | 1–2 d              | 2–3 d             |
| body-side    | 5–8 d        | 2–3 d              | 3–5 d             |

---

## 7. Out of scope for B2b (forward-looking)

- **Online category learning.** B2b's classifier is calibrated once and
  frozen for the evaluation pass. Updating the classifier from
  evaluation-time embeddings is Paper 3 territory.
- **Joint inference of category and tool-reliability.** B2b's category
  posterior does not feed back into `rel_betas` indexing; reliability
  remains category-indexed and the matrix shape stays
  `(n_tools, n_cats)`. Joint inference is Paper 3 / Paper 6.
- **Embedding-into-reliability.** If category inference produces or
  consumes embeddings, those embeddings do not appear anywhere in the
  reliability update path. Paper 3 / Paper 6.
- **Hierarchical priors over category-aware reliability.** Paper 3.
- **Cross-domain transfer of the inference model.** Paper 3 / Paper 5.
- **Learning the embedding model.** B2b uses a frozen pre-trained
  embedding model (sentence-transformers default unless OQ-resolution
  specifies otherwise). No fine-tuning, no embedding-model selection
  across multiple options.
- **Per-question confidence weighting beyond the category posterior.**
  Whatever confidence representation OQ5 lands on (hard label, soft
  distribution, plus reliability profile) is what B4 audits; richer
  confidence schemas are deferred.
- **Anything that touches the four frozen types or their semantics.**
  Per `CLAUDE.md` the four frozen types and the axiom-constrained
  function semantics do not change in B2b. (b-PG) would have stretched
  this — explicitly out of scope at the master-plan level.
- **DSL grammar changes.** No new `.bdsl` syntax. If a chosen option
  needs DSL surface that doesn't exist, that's a feature-finding worth
  separate discussion, not a B2b deliverable.
- **Paper 1 LaTeX rewrite (including the multinomial-logistic →
  specific-model-name rename).** Phase D, post-B5.

---

## 8. Open design questions

The whole document IS B2a's open-questions surface — OQ1 and OQ2
remain explicitly unresolved, by design. Section §6 summarises the
trade-offs Guy will resolve in conversation. This section names the
sub-questions B2b's design doc may need to surface depending on the
resolution:

### 8.1 If OQ2 resolves to (b-NB): per-class covariance structure

(b-NB) as scratched assumes per-class diagonal covariance — the "naive"
assumption that embedding dimensions are conditionally independent
given the class. With sentence-transformer embeddings, this is
known false: dimensions are correlated. The conjugate update extends to
per-class full covariance with a Normal-Wishart prior (Murphy 2007),
but Normal-Wishart is **not** in the existing conjugate registry. So
(b-NB) splits *again* into:

- (b-NB-naive): diagonal covariance, fits existing primitives
  (NormalGamma per (class, dim)).
- (b-NB-full): full covariance, requires adding a NormalWishart
  conjugate pair (lighter than the b-PG additions but non-trivial,
  ~3–4 days).

B2b's design doc resolves this if (b-NB) is the OQ2 answer.

### 8.2 If OQ2 resolves to (a) HMC: dependency choice

`AdvancedHMC.jl` (lighter, hand-write the log-density) vs `Turing.jl`
(heavier, declarative model spec). B2b's design doc resolves.

### 8.3 If OQ1 resolves to (c) body-side: skin protocol additions

The skin protocol (`apps/skin/protocol.md`) currently has no
category-inference operation. A new method `infer_category(question_id)
→ CategoricalMeasure` would need to land. This is not large but it is
visible (changes the JSON-RPC surface), so B2b's design doc names the
addition explicitly per the `docs/posture-3/` skin-smoke-test
discipline.

---

## 9. Risk + mitigation

**R1 (medium).** OQ2 resolves to (b-NB) but real sentence-transformer
embeddings are sufficiently non-Gaussian within-class that the
posterior is poorly calibrated, even if classification accuracy is
fine. Posterior probability ≠ expected accuracy. **Caught by:** B2b's
calibration evaluation — reliability diagrams over predicted P(true)
quantiles. **Mitigation now:** the design doc names this risk; if it
fires in B2b, (c) is the natural fallback (Laplace gives better tail
behaviour for the discriminative form on these embeddings) and
re-resolution of OQ2 is a B2b finding, not a Phase-B reset.

**R2 (low).** OQ2 resolves to (b-NB) and the diagonal-covariance
assumption (8.1) sinks calibration. **Caught by:** the same
reliability-diagram check. **Mitigation now:** flagged in §8.1; if it
fires, B2b either upgrades to (b-NB-full) at the ~3–4 day cost, or
falls back to (c).

**R3 (low).** OQ1 resolves to body-side (c), and the brain↔skin↔body
wiring for qa_benchmark turns out larger than the 1–2 day estimate
because no precedent exists in this domain. **Caught by:** B2b's
implementation start — first day reveals the actual wiring cost.
**Mitigation now:** §4(c) flags the wiring as the option's main risk;
B2b is licensed to fall back to env-frozen if the wiring exceeds 3 days
without re-opening OQ1.

**R4 (low).** Calibration-set construction (OQ3) interacts with the
chosen model in non-trivial ways — e.g. (a) HMC needs more calibration
data than (b-NB) for a comparable posterior, (c) MAP needs less.
B2b's design doc surfaces the model–calibration coupling as a
joint OQ3–OQ2 resolution rather than treating OQ3 as fully
independent.

**R5 (medium).** The paper's prose says "multinomial logistic" and the
chosen model is (b-NB). Phase D's rename is straightforward in
isolation, but reviewers reading current arXiv preprints may see
mismatched language. **Mitigation:** any pre-Phase-D arXiv revision
omits the (b)-specific paragraph entirely or footnotes it as "the
specific Bayesian classifier is detailed in §X" — Phase D handles the
substantive rewrite.

---

## 10. Verification cadence

This is a docs-only PR (the design doc + the scratch prototype). End
of B2a's PR:

```bash
# Scratch prototype runs and produces sensible output.
julia --project=. papers/paper1/scratch/category-inference-b/gaussian_nb_prototype.jl
```

No code under `src/`, `apps/`, or `test/` changes; no `julia test/*.jl`
required. Lint pass: no slug changes expected (the scratch prototype
imports Credence and calls condition() — a lint-clean usage pattern).

B2b's verification cadence is itself an OQ for B2b's design doc.
