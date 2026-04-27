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

## The frozen layer: four types

The DSL has exactly four kinds of object. These are the
ontology of Bayesian decision theory under the de Finettian
framing Posture 3 landed. They do not change.

    Space      — a set of possibilities
    Prevision  — a coherent linear functional on a declared test
                 function space (de Finetti 1974; Whittle 1992)
    Event      — a structural predicate over a Space
    Kernel     — a conditional distribution between two spaces

Measure is a declared view over Prevision, not a frozen primitive
(Move 3 wrapped Measures around Previsions; Move 7 made the view
relationship constitutional). The Measure surface is preserved for
consumer-facing API; internally, Prevision is the object beliefs
are coherent linear functionals on.

Everything else — every operation, every combinator, every
named concept — is a function over these four types.

What is frozen: the four types and their semantics. What
is NOT frozen: the constructor roster below. Named
distributions, space types, and Event subtypes may be added
(and are), provided the added item respects the semantics of
its type. The lists below are current vocabulary, not a closed set.

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
    eu         = expect(measure, pref(·, action))                    — EU of one action
    predictive = expect(measure, density(kernel, ·, obs))
    voi        = expect over obs of [value after condition] - value before
    net-voi    = voi minus the cost of observing
    model      = packaging of measure + kernel (convenience; Julia struct)
    problem    = packaging of model + actions + preference (convenience; Julia struct)

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

The four frozen types: Space, Prevision, Event, Kernel. Measure as a
declared view over Prevision is preserved but not itself frozen.

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

- *No second learning mechanism.* Only `condition` changes beliefs. Forget, decay, exploration bonuses, ad-hoc reweighting violate the topological face even if written inside `src/`. If the world changes, encode drift-rate in the hypothesis space so `condition` can learn it. Under Move 7's event-primary elevation, `condition` has two primary forms at the Prevision level — event-form (`condition(p, e::Event)`) and parametric-form (`condition(p, k::Kernel, obs)`) — as peer primitives; neither derives from the other. They are provably equivalent on deterministic events (Di Lavore–Román–Sobociński "Partial Markov Categories", Proposition 4.9, arXiv:2502.03477). Full reduction via disintegration is out of scope per the Posture 3 master plan.
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

Case law. Each precedent has a stable **slug** used by the lint escape-hatch pragma:

```
# credence-lint: allow — precedent:<slug> — <one-line reason>
```

Both slug and reason are mandatory. The pragma is recognised on the same line as the violation or on the immediately preceding comment-only line. Unknown slugs and missing reasons fail the lint. Novel cases unblock via a new precedent entry — in this slug index AND in `docs/precedents.md` — in the same PR. `grep -r 'credence-lint:' .` is the audit surface.

The slug index below is the lint's source of truth for valid slugs (regex `^\*\*Slug:\*\*\s*\`...\``). Full prose for each precedent — Legal/Illegal cases, failure modes, escape-hatch templates, specific derivations, and historical rejections — lives in `docs/precedents.md`. Read that file before invoking a slug you haven't used before, or before proposing a new one.

### Slug index

**Slug:** `compute-on-weights` — Reading `weights(m)` is sanctioned; arithmetic on the result that feeds a decision is not. No escape hatch; route through `expect` with a Functional. (Invariant 1)

**Slug:** `sort-for-display` — Sort/compare weights for a human-read log line is fine; comparing weights to branch action selection is not. Escape hatch sanctions display/logging only. (Invariant 1, topological)

**Slug:** `display-arithmetic` — Arithmetic for display strings (percentages, progress bars). Pragma required per line; CI cannot tell display from causation. (Invariant 1)

**Slug:** `stdlib-composition` — Stdlib functions calling each other (`voi`→`expect`, `optimise`→`expect`+`argmax`) stay on the canalised path. Documentation-only; `src/` is out of lint scope. (Invariant 1, topological)

**Slug:** `declarative-construction` — Struct constructors that build declarative data (`Problem`, `Kernel(..., likelihood_family=…)`, `CategoricalMeasure(Finite(vals))`) are legal. Documentation-only. (Invariant 2)

**Slug:** `posterior-iteration` — Looping over a mixture's support to do weighted arithmetic is almost always wrong; rewrite as a `Functional` + `expect`, or as event-conditioning. Last-resort escape requires a tracking issue for the rewrite. (Invariants 1 + 2)

**Slug:** `event-conditioning` — `condition(m, e::Event)` is the preferred form when the conditioning object is a declared predicate. Provably equivalent to `condition(m, indicator_kernel(e), true)` on deterministic events (DLRS Prop. 4.9). No escape hatch; this is the legal path. (Invariants 1 + 2)

**Slug:** `prevision-not-measure` — Prevision is the frozen primitive; Measure is a declared view. Extensions (new conjugate pairs, mixture routing) declare at the Prevision level; Measure-level methods are thin facades. (Invariant 2)

**Slug:** `event-primary-condition` — `condition(p::Prevision, e::Event)` is a primary primitive (Move 7 §5.1 Option B), NOT sugar for parametric-form via `ObservationEvent(k, obs)`. No escape hatch. (Invariants 1 + 2)

**Slug:** `parametric-form-sibling` — `condition(p::Prevision, k::Kernel, obs)` is a peer primary alongside event-form, NOT derived from it. Required for continuous observation spaces where event-form needs disintegration. (Invariants 1 + 2)

**Slug:** `baseline-comparison` — Research baselines (argmax-of-means, fixed-threshold) deliberately implement non-Bayesian decision mechanisms for empirical contrast. Pragma names the baseline. Scope of Invariant 1 is the agent, not its baselines. (Invariant 1, topological)

**Slug:** `test-oracle` — Tests of the reasoner need an independent manual oracle (`assert expect(m, f) == approx(0.7)`). Pragma marks the comparison line. Causal within the test, non-causal w.r.t. the agent. (Invariant 1)

**Slug:** `expect-through-accessor` — Reading a Prevision's parameter fields (`.alpha`, `.log_weights`, `.mu`, `.sigma`, `.kappa`) to compute a probabilistic property. Rewrite as `mean(p)`, `variance(p)`, `probability(p, e)`, `weights(p)`, or `expect(p, f)`. Not flagged: `.beta` (ambiguous — TaggedBetaMeasure navigation vs BetaPrevision parameter; `.alpha` catches the same violations), `.components`/`.factors` (containers, not parameters). File-scope exclusion for `src/previsions.jl` and `src/conjugate.jl` (legitimate internal reads). (Invariant 2)

**Slug:** `untyped-mixture-construction` — Untyped container literals (`Any[]`, bare `[]`, `Vector{Any}(...)`, `convert(Vector{Any}, ...)`) passed to Mixture/Product/Particle/Enumeration constructors. Use a typed Vector literal (e.g. `TaggedBetaPrevision[]`). Escape hatch for justified heterogeneous construction (e.g. deserialisation). (Invariant 3)

## Development commands

Julia tests (one file at a time; `ls test/test_*.jl` for the catalogue):
    julia test/test_core.jl                         # canonical pattern

POMDP agent (separate package):
    cd apps/julia/pomdp_agent && julia --project=. -e 'using Pkg; Pkg.test()'

Host-driver pattern (load DSL, extract callable closures):
    env = load_dsl(read("examples/credence_agent.bdsl", String))
    agent_step = env[Symbol("agent-step")]

Requires Julia >=1.9 (stdlib only for the DSL core); CI pins 1.11. Full
workspace deps: HTTP, JSON3, Serialization.

Python workspace (uv):
    uv sync                                         # install all 4 packages
    uv sync --extra server --extra search           # what CI installs
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/

`apps/python/credence_router/tests/test_live.py` is excluded from CI (hits
real provider APIs); run manually when changing live paths.

Skin server (JSON-RPC wire layer):
    julia apps/skin/server.jl
    python -m skin.test_skin                        # smoke tests from repo root

credence-proxy (production gateway):
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes credence-router serve
    docker build -t credence-proxy .

CI (`.github/workflows/publish-image.yml`) runs three jobs.
**unit-tests** instantiates the Julia project (`Pkg.instantiate` +
`Pkg.precompile` — no `julia test/…`), runs `uv sync --extra server --extra
search --no-dev`, then the credence-lint corpus self-test and `check
apps/` pass, then `credence_router` + `credence_agents` pytest (excluding
test_live.py).
**smoke-build** builds amd64 and curls `/ready` against a running container
before anything ships.
**publish** (master + version-tag) builds multi-arch and pushes to GHCR.
**Julia tests are NOT CI-gated** — run `test/*.jl` locally before pushing
DSL-core changes.

Lint at `tools/credence-lint/credence_lint.py` enforces the precedent
slugs. Two passes per file: pass one is a same-line regex over DSL
return-value names; pass two is taint analysis (Python via stdlib `ast`,
Julia via a stateful line scanner that propagates taint through
assignments, tuple unpacking, and `for`-loop targets with `zip` positional
precision). Both passes share the seed rule (DSL call returns are tainted)
and stop at opaque function boundaries. `apps/julia/pomdp_agent/` is
excluded (own `src/`, own invariants).

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

The full file tree is discoverable by `ls` / `find`. The annotations below
are the architecturally load-bearing facts that aren't obvious from
filenames.

    src/                          Tier 1: DSL core
      ontology.jl                 Space / Measure / Event / Kernel types +
                                  axiom-constrained functions (condition,
                                  expect, push, density)
      prevision.jl                Prevision primitive + TestFunction
                                  hierarchy. Dispatch target of Move 4–7
                                  routing — extensions go here, not at
                                  Measure level.
      stdlib.bdsl                 optimise, value, eu, voi, net-voi, etc.
      program_space/              Folder grouping (loaded directly from
                                  Credence.jl). NOT a separate module
                                  entry point.
    examples/                     Runnable DSL programs.
      router.bdsl                 Drives credence-proxy.
    test/                         `julia test/test_*.jl` to run any one.
      fixtures/                   Commit-pinned; see README.md for SHA
                                  provenance protocol. Never regenerate
                                  to fix a loading bug — fix the load code.
    apps/                         Tier 2 — see Architecture section for
                                  the brain/skin/body sub-layer split.
      julia/                      Brain-side; per-domain CLAUDE.md inside.
        DOMAIN_INTERFACE.md       Contract for apps/julia domains.
        pomdp_agent/              Separate Julia package (own Project.toml,
                                  own src/test/CLAUDE.md) — depends on
                                  Credence.
      skin/                       JSON-RPC; protocol.md is the spec.
      python/                     Body; uv workspace, Python >=3.11. Per-
                                  package CLAUDE.md for the three with one.
    docs/
      precedents.md               Full prose for every precedent slug
                                  named in the slug index above.
      posture-3/, posture-4/      Per-branch master-plan.md +
                                  DESIGN-DOC-TEMPLATE.md + per-move docs.
    papers/                       Publication drafts + PAPERS-STRATEGY.md.
    tools/credence-lint/          Precedent-slug lint; corpus self-test
                                  + `check apps/` pass run in CI.
    data/                         Eval output artefacts (gitignored).
    SPEC.md                       Authoritative architecture spec.
    pyproject.toml                uv workspace root (4 members under
                                  apps/python/).

DSL source files use the `.bdsl` extension.

Weights are stored in log-space internally (`logw` field). Use
`weights(m)` to get normalised probabilities. Cross-file invariant —
never exponentiate manually.

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

    ┌──────────────────────────────────┐
    │  Four types                      │  FROZEN
    │  Space, Prevision, Event, Kernel │  (Grammar = a discrete Space
    │                                  │   constructor; Measure = view
    │                                  │   over Prevision)
    ├──────────────────────────────────┤
    │  Axiom-constrained fns           │  Behaviour frozen,
    │  condition, expect, push         │  interface negotiable
    ├──────────────────────────────────┤
    │  Standard library                │  MUTABLE
    │  optimise, value, voi,           │  (perturb_grammar and agent-state
    │  model, problem,                 │   live here too — peers of voi, not
    │  perturb_grammar,                │   a layer above it)
    │  agent-state                     │
    ├──────────────────────────────────┤
    │  Julia execution layer           │  MUTABLE
    │  Conjugate, quadrature,          │  (enumeration + compilation live
    │  particle, heuristic,            │   here alongside conjugate dispatch)
    │  enumeration, compilation        │
    ├──────────────────────────────────┤
    │  Applications (apps/)            │  MUTABLE
    │  brain-side  apps/julia/         │  (in-process DSL callers)
    │  skin        apps/skin/          │  (JSON-RPC translation)
    │  body        apps/python/        │  (user-facing surfaces)
    └──────────────────────────────────┘

    Host (Julia): provides observations, executes actions,
    drives loops, manages persistent state. The DSL is pure.

## Authoritative spec

SPEC.md (in repo root)

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

Four types, axiom-constrained functions, everything else is stdlib.
