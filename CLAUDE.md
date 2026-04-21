# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Constitution for Credence

> Read this entire document before writing any code.

## What this project is

A monorepo containing the Credence Bayesian decision-making DSL (Julia)
and all Python packages that build on it. S-expression syntax. Compiles
to Julia. Grounded in axiomatic decision theory.

### Python workspace (uv)

The `python/` directory is a uv workspace with four packages:

| Directory | PyPI package | Import path | Description |
|-----------|-------------|-------------|-------------|
| `apps/python/credence_bindings/` | `credence` | `import credence` | Low-level Python bindings (Space, Measure, Kernel) |
| `apps/python/credence_agents/` | `credence-agents` | `import credence_agents` | Agent library + Julia bridge + benchmark |
| `apps/python/credence_router/` | `credence-router` | `import credence_router` | Tool routing via EU maximisation |
| `apps/python/bayesian_if/` | — | `import bayesian_if` | Interactive fiction agent application |

Install all: `uv sync` from repo root. `credence_agents`, `credence_router`,
and `bayesian_if` each have their own CLAUDE.md; read the relevant one when
working inside that package.

### The product surface

The repo has two framings. CLAUDE.md is DSL-focused (axioms, frozen layer,
forbidden patterns). README.md is product-focused: the public face is
**credence-proxy** (in `apps/python/credence_router/`), a drop-in
OpenAI-compatible gateway that Bayesian-routes requests across LLM
providers, packaged as a Docker image. The DSL program that drives it
lives at `examples/router.bdsl`. When in doubt about user intent on the
`apps/python/credence_router/` package, the product framing is the gateway, not
the library.

## The axioms (mathematical truths, not design choices)

    A1. Beliefs are probability measures                  (Cox)
    A2. Rational action maximises expected utility        (Savage)
    A3. Learning is conditioning on evidence              (Bayes, de Finetti)

A1–A3 are theorems: each forces a unique answer under mild
consistency requirements. The next item is engineering
discipline — not proved by A1–A3, but what it takes to
implement them without creating internal incoherence:

    A4. One learning mechanism, one decision mechanism.

Rationale: if condition and some second function can both
modify beliefs, the implementation can disagree with itself
— the situation Dutch book arguments rule out at the agent
level. A4 is the invariant that prevents the theorems from
being violated in code.

## The frozen layer: three types

The DSL has exactly three kinds of object. These are the
ontology of Bayesian decision theory. They do not change.

    Space   — a set of possibilities
    Measure — a probability distribution over a space
    Kernel  — a conditional distribution between two spaces

Everything else — every operation, every combinator, every
named concept — is a function over these three types.

What is frozen: the three types and their semantics. What
is NOT frozen: the constructor roster below. Named
distributions and space types may be added (and are),
provided the added item respects the semantics of its type.
The lists below are current vocabulary, not a closed set.

### Space constructors

    (space :finite a b c ...)       finite discrete set
    (space :interval lo hi)         bounded real interval
    (space :product S1 S2 ...)      cartesian product
    (space :euclidean n)            R^n (honest domain for Gaussians)
    (space :positive-reals)         R+ (domain for precision/variance)

### Measure constructors

    (measure S :uniform)            maximum entropy
    (measure S :categorical w1 w2)  explicit weights
    (measure S :beta α β)           Beta distribution on [0,1]
    (measure S :gaussian μ σ)       Gaussian on Euclidean space
    (measure S :gamma α β)          Gamma on positive reals
    (measure S :exponential rate)   Exponential on positive reals

### Kernel constructors

    (kernel H O generator)          H → Measure(O)
        H: source space (hypothesis space)
        O: target space (observation space)
        generator: (lambda (h) distribution-spec)

A kernel is a morphism in the Markov category. It specifies,
for each hypothesis, what distribution over observations the
agent expects. It is the agent's theory of how hypotheses
generate data.

## The axiom-constrained functions

These functions are in the standard library, not the frozen layer.
Their INTERFACE may change. Their BEHAVIOUR is constrained by the
axioms and must not be violated.

    condition : Measure(H) × Kernel(H,O) × O → Measure(H)
        Bayesian inversion. The unique coherent way to update
        a belief given evidence. Must implement Bayes' rule.
        No other function may modify a measure's weights.

    expect : Measure(S) × (S → ℝ) → ℝ
        Integration against a measure. This is what a measure IS:
        a thing that assigns expected values to functions. EU, VOI,
        predictive probability — all are expectations.

    push : Measure(H) × Kernel(H,T) → Measure(T)
        Pushforward / composition. Given a distribution over
        hypotheses and a kernel to another space, produce the
        induced distribution. expect is push to ℝ.

    density : Kernel(H,O) × H × O → ℝ
        The kernel's density at a point. What condition needs
        to compute the likelihood ratio.

## The derived functions (stdlib)

These are compositions of the axiom-constrained functions with
ordinary computation. They are convenience, not capability.
Their interfaces are negotiable and will evolve.

    optimise   = argmax over actions of expect(measure, pref)
    value      = max over actions of expect(measure, pref)
    predictive = expect(measure, density(kernel, ·, obs))
    voi        = expect over obs of [value after condition] - value before
    model      = packaging of measure + kernel (convenience)
    problem    = packaging of model + actions + preference (convenience)

## What Claude Code may change

The standard library: function signatures, new combinators,
packaging of types into convenient bundles.

The Julia implementations: how condition dispatches (conjugate,
particle, quadrature). How expect computes (summation, quadrature,
Monte Carlo). How push constructs new measures. The computational
strategy is determined by the types of the spaces involved and
is invisible to the DSL user.

The set of named distributions and space types.

Tests, examples, documentation, host drivers.

## What Claude Code may NOT change

The three types: Space, Measure, Kernel.

The axiom constraints: condition must be Bayesian inversion,
expect must be integration, no other function may modify
a measure's weights.

The purity of the DSL: no side effects, no IO, no mutation.
The host provides observations and executes actions.

The S-expression grammar: expr = atom | '(' expr* ')'

## Three invariants

The constitution binds three invariants. They are independent — you can satisfy any two while violating the third, and each fails in a different way. Treat all three as equally constitutional: declared-structure is not a softer "style" sibling of single-reasoner. CI catches what it can; the rest is on the author.

**The invariants bind computations whose outputs cause the agent to act or update.** Computation that is non-causal by construction — display formatting, sorted logging, diagnostic telemetry, test oracles — is out of scope. The burden is on the author to show the computation cannot feed back into decisions or beliefs. If in doubt, push it into `src/` or declare it structurally.

### Invariant 1: Single reasoner

Tier 1 is the only reasoner. Two faces, and they both bite:

- **Spatial.** Arithmetic on probabilities or utilities happens only in `src/`. Applications (`apps/`), tests, and hosts never multiply, sum, integrate, compare, or condition raw probability values to influence behaviour.
- **Topological.** Within `src/`, the arithmetic is canalised through the axiom-constrained functions (`condition`, `expect`, `push`, `density`) and their stdlib compositions (`optimise`, `value`, `voi`, `perturb_grammar`, `model`, `problem`, …). No other path modifies weights or selects actions.

What applications do instead: **declare** data (Spaces, Measures, Kernels, Functionals, Problems) and **call** Tier 1 primitives. The answer to "I need to compute X from the posterior" is always "declare X as a Functional and call `expect`". The arithmetic lives in `expect`; the application lives in `apps/`.

Direct consequences of Invariant 1:

- *No second learning mechanism.* Only `condition` changes beliefs. Forget, decay, exploration bonuses, ad-hoc reweighting violate the topological face even if written inside `src/`. If the world changes, encode drift-rate in the hypothesis space so `condition` can learn it.
- *No second decision mechanism.* Only EU-maximisation selects actions. Epsilon-greedy, UCB, Thompson sampling are not forbidden as *concepts* — they may emerge as EU-optimal strategies when computational cost enters the utility function — but they cannot be hard-coded as mechanisms outside EU-max.
- *No random mutation in place of subprogram extraction.* Changes to the hypothesis space are weight-changes in disguise and must derive from `src/`-computed posterior analysis. `propose_nonterminal` requires a `SubprogramFrequencyTable`, only constructable by `analyse_posterior_subtrees`; the type system enforces the dependency.
- *No host-side reimplementation.* `condition`, `expect`, `optimise`, `value`, `push`, `density` have one implementation each, in the ontology module. Hosts call; hosts do not reimplement.
- *`draw` is the boundary, not an exception.* `draw : Measure(S) → S` is the only source of randomness. It lives in the ontology module for Julia callers but is **not** in the DSL's `default_env` — the DSL is pure. Host code calls `draw` after the DSL has constructed the posterior; the DSL does not sample. "Construct the posterior in the DSL, call `draw()` in the host" is the canonical shape.
- *Heuristics live inside EU-max, not alongside it.* When computational cost enters the utility function, an approximate strategy may have higher EU than the exact Bayesian computation — that is not an approximation, it is the optimal strategy. Heuristics are implemented as alternative backends of `expect`/`condition`/`push` inside the Julia execution layer, selected by the same EU machinery the agent uses for domain actions. The DSL specification does not change.

### Invariant 2: Declared structure

Tier 1's dispatch quality depends on first-class type access to the structure of its inputs. Opaque functions are a correctness hazard, not just a performance one — they force `condition`/`expect` to infer structure at runtime or fall back to generic computation, and inferred structure is unreliable: legitimate edge cases misfire.

The rule: functions passed to axiom-constrained operations carry their algebraic structure in their type.

- **Kernels** declare their `likelihood_family` at construction: `BetaBernoulli`, `Flat`, `FiringByTag`, `DispatchByComponent`, etc. Omitting the keyword raises `UndefKeywordError` at construction — not later at dispatch. `condition()` rejects `PushOnly` and unrecognised families with a remediation error. Probing a kernel's output at chosen inputs to infer structure (e.g. treating `log_density == 0.0` as "flat") is forbidden: it misfires on legitimate zero-density points and hides the assumption from the type system.
- **Functionals passed to `expect`** declare their structure: `Identity`, `Projection`, `NestedProjection`, `Tabular`, `LinearCombination`. `OpaqueClosure` is the fallback — it works but forfeits fast paths.
- **Composability is part of structure.** `LinearCombination` carries `Vector{Tuple{Float64, Functional}}`, not flat coefficient arrays. Each sub-functional navigates its own structure. Flat indexing schemes encode stride conventions invisible to the type system — forbidden.
- **Per-component routing in mixtures has a declarative vocabulary.** `FiringByTag(fires::Set{Int}, when_fires, when_not)` covers the dominant "some predicates fire, some don't" pattern. `DispatchByComponent` takes a `classify(measure) -> LikelihoodFamily` closure; it is the typed-return escape hatch (analogue of `OpaqueClosure` for Functional) and should be reached for only when no declarative subtype fits. Add a new declarative subtype before reaching for `DispatchByComponent`. *Declaration stays kernel-side* (`Kernel(..., likelihood_family = FiringByTag(...))` at construction is how kernels announce their routing structure per Invariant 2). *Routing semantics live on the prevision side* as of Move 5: `condition(p::MixturePrevision, k, obs)` resolves each component's LikelihoodFamily via the internal `_resolve_likelihood_family` helper, then dispatches through Move 4's `maybe_conjugate` registry on the resolved leaf family. Consumer code unchanged; the dual residency is real but narrow, with construction and routing in non-overlapping blast radii.
- **Events declare structure too.** `Event` is a first-class declared type alongside Space/Measure/Kernel (see the `Event` hierarchy: `TagSet`, `FeatureEquals`, `FeatureInterval`, `Conjunction`, `Disjunction`, `Complement`). Each constructor witnesses an `indicator_kernel` — a declared kernel into a Boolean Space — and `condition(m, e::Event)` is the sibling form that expands to `condition(m, indicator_kernel(e), true)`. No opaque predicate closures at the axiom layer: `TagSet` carries a typed `Set{Int}`; `FeatureEquals`/`FeatureInterval` reach the hypothesis through `feature_value(h, name)` method dispatch, overridable per hypothesis type.

Invariant 2 also bears on DSL wrappers. A domain file that defines `(defun choose-x (s) (optimise s as pref))` funnels its arguments through an opaque closure that defeats Functional dispatch. Domain files contain data (spaces, kernels, priors); preferences are protocol-level Functional specs (e.g. `functional_per_action` with `LinearCombination` of `NestedProjections`), not DSL lambdas. Wrapping forces an opaque bottleneck around an axiom-constrained op — a violation of Invariant 2, and a violation of Invariant 1's topological face (the wrapper creates a hidden path to the axiom-constrained op that CI can't see).

### Invariant 3: Single-responsibility representations

Each piece of data has one semantic role. Representations used for computation and representations used for structural analysis are kept separate, even when they describe the same thing.

Canonical example: programs.

- **`CompiledKernel`** is for arithmetic. Its type has no AST field, enforced. Kernel evaluation calls a closure, never walks a tree.
- **`Program`** retains the AST, for structural analysis (grammar perturbation, subprogram frequency, complexity scoring).

The two representations serve different purposes and must not be conflated. Conflation typically appears as a single type trying to support both cheap evaluation and tree traversal; the failure mode is silent drift (cached closure and AST diverging) or dispatch ambiguity (runtime type-checks to decide which path to take).

Second example: weights. Measures store weights in log-space internally (`logw` field). Consumer code reads probabilities via `weights(m)`; direct access to `logw`, whether by exponentiation or otherwise, betrays assumptions about normalisation that only the Measure type knows. `logw` is a private representation; `weights(m)` is the public accessor. Mixing the two — caching `logw` somewhere and then treating it as a probability, or normalising it by hand in consumer code — is a Single-responsibility violation.

## Precedents

Case law. Each entry names which invariant it follows from and why. Weight is on grey-zone cases — the bright-line violations are caught by the constitution and (eventually) by CI; what merits human-readable reasoning are the judgement calls where mechanical enforcement can't distinguish causal from non-causal.

Every precedent carries a stable **slug** — a short identifier used by the lint escape-hatch pragma. When a grey-zone case sanctions code that would otherwise violate the invariants, the author marks the line with:

```
# credence-lint: allow — precedent:<slug> — <one-line reason>
```

Both the slug and the reason are mandatory. Unknown slugs and missing reasons fail the lint. Novel cases unblock via a new precedent entry in this document (with its own slug) in the same PR — new escape hatches are constitutional amendments, not inline concessions. `grep -r 'credence-lint:' .` is a usable audit surface.

### Grey zones

#### Reading vs. computing on weights
**Slug:** `compute-on-weights`.
**Legal:** `weights(m)` for logging, telemetry, display. `mean(m)` passed to a non-causal dashboard.
**Illegal:** any arithmetic on the result that feeds back into a decision or belief — summation, multiplication, comparison-in-branch, threshold checks that gate behaviour.
**Follows from Invariant 1** because the public accessor is sanctioned access; what makes it a violation is the subsequent causal arithmetic. `weights()` itself does no reasoning; what you do with the return value can. There is no escape hatch — arithmetic that needs the posterior must flow through `expect` with a declared Functional. The slug exists for cross-referencing.

#### Sort-for-display vs. compare-to-branch
**Slug:** `sort-for-display`.
**Legal:** `sort(pairs, by=last)` for a top-K log line. The ordering is non-causal — the display is read by a human, not by the agent.
**Illegal:** `if w1 > w2 then action_a else action_b` — that comparison *is* the decision, and it lives outside `optimise`.
**Follows from Invariant 1 (topological face)** because action selection must flow through EU-max. A weight comparison in application code is a parallel decision mechanism. Escape hatch with this slug permits comparison/sort when the author can assert the result is consumed only by display/logging, not by subsequent logic.

#### Display arithmetic
**Slug:** `display-arithmetic`.
**Legal with escape hatch:** `f"{round(w * 100, 1)}%"` for a progress bar or report.
**Required pragma:** `# credence-lint: allow — precedent:display-arithmetic — <reason>` on the line, reviewed per commit.
**Follows from Invariant 1** because the rule binds causal arithmetic; display arithmetic is non-causal by construction. CI cannot distinguish display from causation mechanically, so the author carries the burden of marking it.

#### Stdlib compositions calling each other
**Slug:** `stdlib-composition`.
**Legal:** `voi` calls `expect`; `optimise` calls `expect` + `argmax`; `model`/`problem` constructors compose kernels and priors; `perturb_grammar` takes posterior analysis as input.
**Follows from Invariant 1 (topological face)** because the canalised path is the axiom-constrained functions **and their stdlib compositions**. Stdlib members calling each other stays on the sanctioned path. New stdlib operations are added by composing existing ones plus ordinary computation, not by creating a new arithmetic path. Slug is documentation-only — stdlib code lives in `src/`, which is out of scope for the lint.

#### Application constructing a `Problem`
**Slug:** `declarative-construction`.
**Legal.** `Problem(state, actions, preference)` is a struct constructor — declarative data. `initial_rel_state(...)`, `CategoricalMeasure(Finite(vals))`, `Kernel(H, O, gen, likelihood_family=…)` — all declarative.
**Contrast with:** a DSL wrapper like `(defun solve-email (state) (optimise state email-actions email-pref))` — that is a callable re-exporting an axiom-constrained op with hidden structure (see Invariant 2 violation in the Historical rejections). Slug is documentation-only — constructors don't trigger the lint in the first place.

#### Iterating a posterior's support
**Slug:** `posterior-iteration`.
**Almost always illegal in consumer code.** If you're writing a loop over `zip(support(m), weights(m))` — or over a mixture's components to sum weighted quantities — the "something" is probability arithmetic.
**Rewrite:** declare the computation as a `Functional` (`Projection`, `NestedProjection`, `Tabular`, composed via `LinearCombination`) and call `expect(m, f)`. If the loop is a conditional aggregation ("sum over components where predicate fires"), the right primitive is *event-conditioning*: `expect(condition(m, TagSet(fires)), inner)` or a typed `FeatureEquals` / `FeatureInterval`. See the `event-conditioning` precedent below.
**Last-resort escape for deferred rewrites.** Inline iteration that predates the Functional / Event invariants may be kept via `# credence-lint: allow — precedent:posterior-iteration — tracked in issue #<N>`. The reason must reference a tracking issue for the rewrite; the pragma lives until the rewrite lands. Reach for this only when neither a Functional nor an Event constructor fits — now that events are first-class, most mixture-filter cases have a declarative path.
**Follows from Invariant 1 and Invariant 2** jointly: the spatial rule rejects the loop; the declared-structure rule points to the rewrite.

#### Event-conditioning
**Slug:** `event-conditioning`.
**Preferred idiom when the conditioning object is an event.** `condition(m, e::Event)` is provably equivalent to `condition(m, indicator_kernel(e), true)` for deterministic events (Di Lavore–Román–Sobociński Prop. 4.9). The sibling form is the natural shape when the conditioning object is a declared predicate (`TagSet`, `FeatureEquals`, `FeatureInterval`, or Boolean compositions thereof); the parametric form remains primary for genuine observation-with-likelihood conditioning.
**Mechanical bridge.** Every `Event` constructor witnesses an `indicator_kernel` into a Boolean Space. That witness is how Invariant 2 is preserved at the axiom layer: events reach `condition` through declared kernels, not opaque predicate closures.
**Follows from Invariants 1 and 2.** The topological face is preserved — `condition(m, e)` is on the canalised path, just through the event surface syntax. Declared structure is preserved because every Event carries its data in typed fields. No escape hatch; this is the legal path.

#### Baseline comparison
**Slug:** `baseline-comparison`.
**Legal with escape hatch.** Research baselines deliberately implement non-Bayesian decision mechanisms — argmax-of-means, fixed-threshold, cheapest-first — to empirically contrast against the principled EU-max agent. The paper depends on having these baselines; they are not bugs to be fixed.
**Required pragma:** `# credence-lint: allow — precedent:baseline-comparison — <which baseline, why non-Bayesian>`. Each baseline's violation is tagged so `grep -r 'baseline-comparison'` produces an inventory of what the paper compares against.
**Follows from Invariant 1 (topological face)** being scoped to the agent. The invariant forbids parallel decision mechanisms *that the agent uses*; a baseline is, by construction, not the agent. This precedent makes that distinction explicit.

#### Test code computing expected values manually
**Slug:** `test-oracle`.
**Legal with escape hatch.** Tests *of* the reasoner legitimately need an independent oracle: `assert expect(m, f) == pytest.approx(0.7)  # computed by hand from Beta(3,7)`.
**Required pragma:** `# credence-lint: allow — precedent:test-oracle — <reason>` on the comparison line. The manual computation is the test's ground truth; it is causal *within the test*, but the test is non-causal with respect to the agent (it doesn't feed back into agent behaviour).

### Specific derivations

#### Indifference implies exploration
When `EU(interact) == EU(wait)` (both zero), interact. `select_action` threshold is `>= 0`, not `> 0`. Follows from correct EU accounting: at indifference, VOI from the interaction outcome is still positive (you'll learn something), which is part of EU. If the threshold is strict, a correct EU computation will have already broken the tie. Listed here because it's the kind of edge case that gets "fixed" back to strict inequality under perceived instability, and the fix is wrong.

#### Non-firing predicates predict the base rate
Programs whose predicates don't fire return `log(0.5)`, not `0.0`. A non-firing program is implicitly predicting "I don't know, 50/50"; that prediction is scored against the observation. Ranking: *informed-and-right* > *uninformed* > *informed-and-wrong*. Returning `0.0` makes non-firing programs unbeatable (no information → no penalty), creating weight rigidity after regime changes. Listed here because the bug manifests long after the change (the posterior stops adapting) and the fix looks like a tunable constant; it is not — it is scoring-rule calibration.

### Historical rejections

#### PROPOSED: `(sample measure)` in the DSL for Thompson sampling.
**REJECTED.** Sampling is randomness; randomness is a side effect; the DSL is pure. Construct the posterior in the DSL; call `draw()` in the host. **Invariant 1 (spatial)** — DSL stays non-executing.

#### PROPOSED: `host_decide()` / `host_optimise()` in host drivers.
**REJECTED.** `optimise` and `value` belong in the ontology alongside `expect` and `condition`. One implementation per operation. The host driver is pure orchestration — it calls ontology functions. **Invariant 1 (topological)** — one canonical path per operation.

#### PROPOSED: `(thompson-sample m actions pref)` in stdlib that calls sample.
**REJECTED.** Compounds both errors above. Thompson sampling is `draw` (host) + `argmax` (ordinary computation on the drawn value). The DSL constructs the posterior; its job is done. **Invariant 1 (both faces).**

#### PROPOSED: Bare lambda as a kernel's likelihood.
**REJECTED.** Kernels declare `likelihood_family` at construction; bare lambdas defeat conjugate dispatch and force probing. **Invariant 2.**

#### PROPOSED: Flat coefficient arrays for `LinearCombination` instead of `Vector{Tuple{Float64, Functional}}`.
**REJECTED.** Flat indexing encodes stride conventions invisible to the type system. Each sub-functional must navigate its own structure. **Invariant 2** — composition is part of structure.

#### PROPOSED: Inferring kernel family by probing `log_density == 0.0` for flat likelihoods.
**REJECTED.** Legitimate kernels can return zero log-density at specific points without being flat; the probe misfires and hides the assumption from the type system. **Invariant 2** — declared at construction, not dispatch.

#### PROPOSED: DSL wrapper functions in domain files (e.g., `(defun choose-email (s) (optimise s email-actions email-pref))`).
**REJECTED.** Wrappers force the preference through an opaque closure that defeats Functional dispatch AND hide the causal arithmetic path from CI. A domain file contains data; axiom-constrained ops are called at the protocol level, not wrapped. **Invariants 1 and 2** jointly.

## Development commands

Run tests:
    julia test/test_core.jl             # Core DSL tests
    julia test/test_flat_mixture.jl     # Flat mixture tests
    julia test/test_host.jl             # Host helper tests
    julia test/test_program_space.jl    # Program-space tests (grammars, enumeration, perturbation)
    julia test/test_grid_world.jl       # Grid-world domain tests
    julia test/test_email_agent.jl      # Email agent domain tests
    julia test/test_rss.jl              # RSS domain tests

Run POMDP agent tests:
    cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

Run the Jericho IF agent:
    cd apps/julia/pomdp_agent && julia --project=. examples/jericho_agent.jl /path/to/game.z3

Run an example:
    julia -e 'push!(LOAD_PATH, "src"); using Credence; run_dsl(read("examples/coin.bdsl", String))'

Run the grid-world agent:
    julia apps/julia/grid_world/host.jl

Run the credence agent (host-driven):
    julia examples/host_credence_agent.jl

Use the module from Julia REPL:
    push!(LOAD_PATH, "src")
    using Credence

Load DSL and get callable closures (host-driver pattern):
    env = load_dsl(read("examples/credence_agent.bdsl", String))
    agent_step = env[Symbol("agent-step")]

Requires Julia >=1.9 (stdlib only for the DSL core). CI pins Julia 1.11.
External deps (for full workspace): HTTP, JSON3, Serialization.

Python workspace:
    uv sync                                         # install all 4 packages
    uv sync --extra server --extra search           # what CI installs
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/

Run a single Python test file (example):
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
      uv run pytest apps/python/credence_router/tests/test_routing.py -x

`apps/python/credence_router/tests/test_live.py` is excluded from CI (hits
real provider APIs); run it manually when changing live paths.

Skin server (language-agnostic host interface — JSON-RPC wire layer):
    julia apps/skin/server.jl                       # holds Measures, evaluates DSL via JSON-RPC
    python -m skin.test_skin                        # smoke tests (from repo root)

credence-proxy (production gateway):
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes credence-router serve
    docker build -t credence-proxy .                # same image CI publishes

CI: `.github/workflows/publish-image.yml` runs core Julia tests,
`uv sync --extra server --extra search --no-dev`, then
`credence_router` + `credence_agents` pytest (excluding test_live.py),
and publishes the Docker image.

## Repo conventions for Claude Code sessions

This repo is single-maintainer. The author/Claude conversation loop *is* the review — there is no separate human reviewer role. The conventions below codify how Claude Code operates across sessions so that future sessions (including ones on different machines without access to the author's `~/.claude/`) can pick up where prior sessions left off.

### Merge authority

Claude Code merges PRs on this repo when two gates are satisfied:
1. CI is green.
2. The author (Guy) has approved the PR content in conversation.

Both gates together constitute the review. When both are satisfied, merge via `gh pr merge <N> --rebase`. This has happened seven times during Posture 2 and was exercised on Posture 3 Move 0 (PR #16); it is the normal path, not an escalation. An explicit "merge it" or equivalent from the author counts as the in-conversation approval.

### Rebase-merge for linear master history

Use `--rebase` (not `--merge` or `--squash`) so individual commits from the branch land on master unchanged. The Posture 2 sequence (7 gates) and Posture 3 Move 0 (3 commits) both preserve their commit histories this way. Squash-merge only when the branch's commit history is genuinely noise (rare for this repo).

### Multi-move branches: design-doc before code

When a branch structures work as a sequence of moves (Posture 3's 8-move plan is the canonical example), each move lands as a design-doc PR followed by a code PR. The mandatory design-doc template lives at `docs/<branch-family>/DESIGN-DOC-TEMPLATE.md` (see `docs/posture-3/DESIGN-DOC-TEMPLATE.md`). Reviewers reject design docs that omit "Open design questions" or fill it with boilerplate.

The branch's master plan lives at `docs/<branch-family>/master-plan.md` as the durable, in-repo copy. Historical drafts may also be kept under the author's session notes (`~/.claude/plans/`) but that location is session-external and not guaranteed available to future sessions on other machines. **If you are working on Posture 3 (branch `de-finetti/migration`), read `docs/posture-3/master-plan.md` before touching code.**

### Test fixtures are commit-pinned

When a refactor changes a serialised schema (or any reference state), capture fixtures from a specific named SHA and record that SHA in `test/fixtures/README.md`. Fixtures are never regenerated to fix loading bugs — fix the load code. See `test/fixtures/README.md` for the provenance protocol.

### Session memory is user-level, not in-repo

Claude Code sessions maintain persistent memory at `~/.claude/projects/<hash>/memory/` with `MEMORY.md` as the index. This is *user-level* and private to the author; it is not in the repo. Conventions derived from session feedback are saved there for the author's own future sessions; durable conventions that future Claude Code sessions (including first-time sessions) must know about are lifted into this CLAUDE.md file. This section is where those lifted conventions live.

### Lint pragmas and precedent slugs

Inline `# credence-lint: allow — precedent:<slug> — <reason>` pragmas sanction grey-zone violations. Unknown slugs and missing reasons fail the lint; the slug catalogue lives in the Precedents section above. This is *not* a new convention — it predates the Posture 2/3 work — but is worth naming here because it is one of the session-level mechanisms a new Claude Code session needs to know exists.

## Project structure

    src/                          Tier 1: DSL core
      Credence.jl                 Module entry point
      parse.jl                    S-expression parser
      ontology.jl                 Three types + axiom-constrained functions
      eval.jl                     Evaluator / compiler (DSL → Julia calls)
      stdlib.bdsl                 Standard library (optimise, value, voi, etc.)
      persistence.jl              Save/load agent state across sessions
      host_helpers.jl             Host-level reliability/coverage helpers
      program_space/              Program-space extensions (grouped for cohesion)
        ProgramSpace.jl           Module entry point
        types.jl                  AST types, Grammar, Program, CompiledKernel
        enumeration.jl            Bottom-up enumeration, complexity scoring
        compilation.jl            AST → closure compilation
        perturbation.jl           Posterior subtree analysis, grammar perturbation
        agent_state.jl            AgentState, sync_prune!, sync_truncate!
    examples/                     Runnable DSL programs
      coin.bdsl                   Biased coin learning
      credence_agent.bdsl         Agent DSL (pure functions, host-driven)
      grid_agent.bdsl             Grid agent DSL
      router.bdsl                 Bayesian LLM/search routing (drives credence-proxy)
      host_credence_agent.jl      Julia host driver for credence agent
    test/
      test_core.jl                Core DSL tests (42 tests)
      test_flat_mixture.jl        Flat mixture conditioning tests
      test_host.jl                Host helper tests
      test_program_space.jl       Program-space tests (enumeration, compilation, perturbation)
      test_grid_world.jl          Grid-world tests (full agent, regime change, meta-learning)
      test_email_agent.jl         Email agent domain tests
      test_rss.jl                 RSS domain tests
    apps/                         Everything built on top of the DSL — three sub-layers:
      julia/                      Brain-side applications (in-process DSL callers)
        DOMAIN_INTERFACE.md       Contract for apps/julia domains
        grid_world/               Grid-world domain (simulation, host, terminals, metrics)
        email_agent/              Email domain (JMAP integration, LLM prosthetic)
        qa_benchmark/             QA benchmark domain (LLM comparison harness)
        rss/                      RSS article ranking domain (Postgres-backed)
        pomdp_agent/              POMDP agent package (MCTS, factored models)
          Project.toml            Separate Julia package depending on Credence
          src/, examples/, test/, CLAUDE.md
      skin/                       JSON-RPC translation layer (opaque Measure handles)
        protocol.md               JSON-RPC protocol spec
        server.jl                 Julia server (holds Measures, evaluates DSL)
        client.py                 Python client (spawns subprocess, sends RPC)
        test_skin.py              Smoke tests
      python/                     Body — user-facing surfaces, prosthetics, connections (uv workspace; Python >=3.11)
        credence_bindings/        Low-level Python bindings
        credence_agents/          Agent library + Julia bridge + benchmark
        credence_router/          credence-proxy (LLM/search routing gateway)
        bayesian_if/              Interactive fiction agent
    docs/                         Additional documentation
      rss-preference-learning.md  RSS preference learning design
    papers/                       Publication (credence.tex)
    data/                         Eval output artefacts (gitignored)
    Dockerfile                    credence-proxy container (published by CI)
    SPEC.md                       Authoritative three-tier architecture spec
    pyproject.toml                uv workspace root (4 members under apps/python/)
    .github/workflows/            CI: publish-image.yml (tests + Docker publish)

DSL source files use the `.bdsl` extension.

Weights are stored in log-space internally (`logw` field). Use
`weights(m)` to get normalized probabilities. This is a cross-file
invariant — never exponentiate manually.

Constructors: `CategoricalMeasure(Finite(vals))` for uniform prior,
`CategoricalMeasure{T}(Finite{T}(vals), logw)` for explicit log-weights
(auto-normalises). `BetaMeasure(α, β)` for Beta on [0,1].

## Architecture

Two-tier architecture. See SPEC.md for details.

- Tier 1 (src/): DSL core — Space/Measure/Kernel, condition/expect/push, the
  stdlib (optimise, voi, perturb-grammar, agent-state, …), and their
  program-space extensions (Grammar as a Space constructor, CompiledKernel as
  a Kernel variant, enumeration/compilation as execution strategies).
  Program-related files are grouped under src/program_space/ for cohesion,
  not as a separate tier.
- Tier 2 (apps/): Applications, in three explicit sub-layers relative to the wire:
  - **Brain-side applications** (`apps/julia/*`) — in-process DSL callers; domains
    (grid_world, email_agent, rss, qa_benchmark) and the pomdp_agent package.
    See `apps/julia/DOMAIN_INTERFACE.md`.
  - **Skin** (`apps/skin/`) — JSON-RPC translation layer. The `SkinClient`
    Python handle talks to `server.jl`; Measures stay server-side as opaque IDs.
  - **Body** (`apps/python/*`) — user-facing surfaces, prosthetics, connections:
    credence_bindings, credence_agents, credence_router (credence-proxy gateway),
    bayesian_if. The body talks to the skin, never to Measures directly.

    ┌─────────────────────────────┐
    │  Three types                │  FROZEN
    │  Space, Measure, Kernel     │  (Grammar = a discrete Space constructor)
    ├─────────────────────────────┤
    │  Axiom-constrained fns      │  Behaviour frozen,
    │  condition, expect, push    │  interface negotiable
    ├─────────────────────────────┤
    │  Standard library           │  MUTABLE
    │  optimise, value, voi,      │  (perturb_grammar and agent-state
    │  model, problem,            │   live here too — peers of voi, not
    │  perturb_grammar,           │   a layer above it)
    │  agent-state                │
    ├─────────────────────────────┤
    │  Julia execution layer      │  MUTABLE
    │  Conjugate, quadrature,     │  (enumeration + compilation live
    │  particle, heuristic,       │   here alongside conjugate dispatch)
    │  enumeration, compilation   │
    ├─────────────────────────────┤
    │  Applications (apps/)       │  MUTABLE
    │  brain-side  apps/julia/    │  (in-process DSL callers)
    │  skin        apps/skin/     │  (JSON-RPC translation)
    │  body        apps/python/   │  (user-facing surfaces)
    └─────────────────────────────┘

    Host (Julia): provides observations, executes actions,
    drives loops, manages persistent state. The DSL is pure.

## Authoritative spec

credence-spec.md (in repo root)

## On metareasoning

There is no separate metareasoning layer. The choice of how
much to compute is itself a decision problem: beliefs about
computational cost, expected improvement from further
computation, EU maximisation over the choice of strategy.
This is expressible in the same DSL the agent uses for
everything else. The Julia layer implements the strategies;
the DSL specifies the choice among them.

## Key references

    Cox (1946)                — probability from consistency
    Savage (1954)             — utility + probability from preferences
    Jaynes (2003)             — Probability Theory: The Logic of Science
    Hutter (2005)             — AIXI: the Platonic limit
    Russell & Wefald (1991)   — rational metareasoning
    Lieder & Griffiths (2020) — resource-rationality
    Fritz (2020)              — Markov categories
    Staton (2017)             — measure-theoretic PPL semantics
    McCarthy (1960)           — why S-expressions

## Performance problems

If a test run is too slow, if the agent hangs, or if exact inference
is computationally expensive: STOP. Do not replace exact inference
with an approximation to fix a performance problem. Instead:

1. Report the performance problem and where it occurs.
2. Propose the approximation explicitly, explaining what
   exactness is lost.
3. Wait for approval before implementing.

Approximations are sometimes necessary. Undocumented approximations
are always bugs. The distinction is consent, not capability.

## One-line summary

Three types, axiom-constrained functions, everything else is stdlib.
