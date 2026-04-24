# Precedents

Constitutional case law for Credence. Each entry names which invariant it follows from and why. Weight is on grey-zone cases — the bright-line violations are caught by the constitution and (eventually) by CI; what merits human-readable reasoning are the judgement calls where mechanical enforcement can't distinguish causal from non-causal.

Every precedent carries a stable **slug** — a short identifier used by the lint escape-hatch pragma. When a grey-zone case sanctions code that would otherwise violate the invariants, the author marks the line with:

```
# credence-lint: allow — precedent:<slug> — <one-line reason>
```

Both the slug and the reason are mandatory. The pragma is recognised on the same line as the violation or on the immediately preceding comment-only line — whichever reads better at the call site. Unknown slugs and missing reasons fail the lint. Novel cases unblock via a new precedent entry in this document (with its own slug) in the same PR — new escape hatches are constitutional amendments, not inline concessions. `grep -r 'credence-lint:' .` is a usable audit surface.

The compact slug index lives in `CLAUDE.md` — that's what the lint reads to discover valid slugs. Full prose for each precedent (Legal/Illegal cases, failure modes, escape-hatch templates) is below.

## Grey zones

### Reading vs. computing on weights
**Slug:** `compute-on-weights`.
**Legal:** `weights(m)` for logging, telemetry, display. `mean(m)` passed to a non-causal dashboard.
**Illegal:** any arithmetic on the result that feeds back into a decision or belief — summation, multiplication, comparison-in-branch, threshold checks that gate behaviour.
**Follows from Invariant 1** because the public accessor is sanctioned access; what makes it a violation is the subsequent causal arithmetic. `weights()` itself does no reasoning; what you do with the return value can. There is no escape hatch — arithmetic that needs the posterior must flow through `expect` with a declared Functional. The slug exists for cross-referencing.

### Sort-for-display vs. compare-to-branch
**Slug:** `sort-for-display`.
**Legal:** `sort(pairs, by=last)` for a top-K log line. The ordering is non-causal — the display is read by a human, not by the agent.
**Illegal:** `if w1 > w2 then action_a else action_b` — that comparison *is* the decision, and it lives outside `optimise`.
**Follows from Invariant 1 (topological face)** because action selection must flow through EU-max. A weight comparison in application code is a parallel decision mechanism. Escape hatch with this slug permits comparison/sort when the author can assert the result is consumed only by display/logging, not by subsequent logic.

### Display arithmetic
**Slug:** `display-arithmetic`.
**Legal with escape hatch:** `f"{round(w * 100, 1)}%"` for a progress bar or report.
**Required pragma:** `# credence-lint: allow — precedent:display-arithmetic — <reason>` on the line, reviewed per commit.
**Follows from Invariant 1** because the rule binds causal arithmetic; display arithmetic is non-causal by construction. CI cannot distinguish display from causation mechanically, so the author carries the burden of marking it.

### Stdlib compositions calling each other
**Slug:** `stdlib-composition`.
**Legal:** `voi` calls `expect`; `optimise` calls `expect` + `argmax`; `model`/`problem` constructors compose kernels and priors; `perturb_grammar` takes posterior analysis as input.
**Follows from Invariant 1 (topological face)** because the canalised path is the axiom-constrained functions **and their stdlib compositions**. Stdlib members calling each other stays on the sanctioned path. New stdlib operations are added by composing existing ones plus ordinary computation, not by creating a new arithmetic path. Slug is documentation-only — stdlib code lives in `src/`, which is out of scope for the lint.

### Application constructing a `Problem`
**Slug:** `declarative-construction`.
**Legal.** `Problem(state, actions, preference)` is a struct constructor — declarative data. `initial_rel_state(...)`, `CategoricalMeasure(Finite(vals))`, `Kernel(H, O, gen, likelihood_family=…)` — all declarative.
**Contrast with:** a DSL wrapper like `(defun solve-email (state) (optimise state email-actions email-pref))` — that is a callable re-exporting an axiom-constrained op with hidden structure (see Invariant 2 violation in the Historical rejections). Slug is documentation-only — constructors don't trigger the lint in the first place.

### Iterating a posterior's support
**Slug:** `posterior-iteration`.
**Almost always illegal in consumer code.** If you're writing a loop over `zip(support(m), weights(m))` — or over a mixture's components to sum weighted quantities — the "something" is probability arithmetic.
**Rewrite:** declare the computation as a `Functional` (`Projection`, `NestedProjection`, `Tabular`, composed via `LinearCombination`) and call `expect(m, f)`. If the loop is a conditional aggregation ("sum over components where predicate fires"), the right primitive is *event-conditioning*: `expect(condition(m, TagSet(fires)), inner)` or a typed `FeatureEquals` / `FeatureInterval`. See the `event-conditioning` precedent below.
**Last-resort escape for deferred rewrites.** Inline iteration that predates the Functional / Event invariants may be kept via `# credence-lint: allow — precedent:posterior-iteration — tracked in issue #<N>`. The reason must reference a tracking issue for the rewrite; the pragma lives until the rewrite lands. Reach for this only when neither a Functional nor an Event constructor fits — now that events are first-class, most mixture-filter cases have a declarative path.
**Follows from Invariant 1 and Invariant 2** jointly: the spatial rule rejects the loop; the declared-structure rule points to the rewrite.

### Event-conditioning
**Slug:** `event-conditioning`.
**Preferred idiom when the conditioning object is an event.** `condition(m, e::Event)` is provably equivalent to `condition(m, indicator_kernel(e), true)` for deterministic events (Di Lavore–Román–Sobociński Prop. 4.9). The sibling form is the natural shape when the conditioning object is a declared predicate (`TagSet`, `FeatureEquals`, `FeatureInterval`, or Boolean compositions thereof); the parametric form remains primary for genuine observation-with-likelihood conditioning.
**Mechanical bridge.** Every `Event` constructor witnesses an `indicator_kernel` into a Boolean Space. That witness is how Invariant 2 is preserved at the axiom layer: events reach `condition` through declared kernels, not opaque predicate closures.
**Follows from Invariants 1 and 2.** The topological face is preserved — `condition(m, e)` is on the canalised path, just through the event surface syntax. Declared structure is preserved because every Event carries its data in typed fields. No escape hatch; this is the legal path.

### Prevision, not Measure, is primitive
**Slug:** `prevision-not-measure`.
**Rule.** Prevision is the frozen primitive (Move 7 elevation); Measure is a declared view over Prevision (Move 3 wrapping; Move 7 constitutional). Beliefs are coherent linear functionals on a declared test function space — what `expect` realises — not probability mass distributions over a measurable space. The Measure surface is preserved for consumer-facing API; internally, belief-changing operations work with Prevision.
**When it applies.** Code that extends axiom-constrained functions with new beliefs (new conjugate pairs, new mixture routing, new fallback strategies) should declare its work at the Prevision level. `maybe_conjugate` dispatches on (Prior, Likelihood) type pairs where Prior is a Prevision subtype; `_dispatch_path`, `condition`, `update` are all Prevision-level. The Measure-level methods stay as thin facades delegating to the Prevision primary.
**Failure mode.** Code that patches Measure-specific behaviour (e.g. `condition(m::SomeMeasure, …)` with arithmetic inline) without extending the corresponding Prevision surface creates a dispatch surface that bypasses Move 4's registry and Move 5's routing. Post-Move-7 the Measure surface is a view; arithmetic lives on the prevision side. Inline arithmetic at the Measure level silently forks behaviour.
**Follows from Invariant 2** (declared structure: Prevision is the dispatch target for axiom-constrained functions) and Move 7's frozen-types edit.

### Event-form `condition` is a primary, not sugar
**Slug:** `event-primary-condition`.
**Rule.** `condition(p::Prevision, e::Event)` is a primary primitive at the Prevision level (Move 7 §5.1 Option B). It is NOT sugar for `condition(p, k::Kernel, obs)` via a synthesised `ObservationEvent(k, obs)`. The two forms are peer primary primitives; neither derives from the other.
**When it applies.** When the conditioning object is a declared structural predicate over a Space — `TagSet`, `FeatureEquals`, `FeatureInterval`, or Boolean compositions (`Conjunction`, `Disjunction`, `Complement`). Each Event witnesses an `indicator_kernel` into BOOLEAN_SPACE; deterministic-event equivalence to the parametric form is DLRS Prop. 4.9 (arXiv:2502.03477).
**Failure mode.** Attempting to derive the event-form from the parametric-form via `ObservationEvent(k, obs)` at the axiom layer. This requires `p(1_e) > 0` for the event `{k emits exactly obs}`; continuous observation spaces violate this (Lebesgue measure zero), requiring disintegration which is out of scope per the master plan. Sugar over an undefined reduction is the exact failure mode Option B was committed to avoid.
**Follows from Invariants 1 and 2.** Topologically, `condition(p, e::Event)` is on the canalised path as a primary form. Declared structure: the Event hierarchy carries typed data in fields; `ObservationEvent` would carry a kernel and observation — a likelihood-structured object, categorically different. No escape hatch; this is the legal path.

### Parametric-form `condition` is a sibling primary, not derived
**Slug:** `parametric-form-sibling`.
**Rule.** `condition(p::Prevision, k::Kernel, obs)` is a peer primary at the Prevision level alongside the event-form (Move 7 §5.1 Option B). It is NOT derived from event-form via any `ObservationEvent` construction. Parametric Bayes update — the familiar `p(θ | o) ∝ p(o | θ) p(θ)` shape — has its own mathematically-well-defined semantics for continuous observation kernels that event-form conditioning cannot cover without disintegration.
**When it applies.** When the conditioning object is a kernel–observation pair, especially for continuous observation spaces (GaussianMeasure + NormalNormal kernel; BetaMeasure + grid-quadrature kernel; anything Move 6 routes through `_condition_particle` or `_condition_by_grid`). Move 4's conjugate registry dispatches through this form.
**Failure mode.** Framing parametric-form as "the real condition" with event-form as a sibling-that-expands-to-parametric, OR framing event-form as primary with parametric-form as sugar. Both framings mask the peer-primary structure. Downstream consumers reading such a framing misunderstand whether `condition(p, e::Event)` is a first-class primitive (it is, under Option B) or a derived form (it is not).
**Follows from Invariants 1 and 2** jointly with the §1.0 continuous-kernel measure-zero constraint. The topological face treats both forms as on the canalised path; the declared-structure face keeps `Event` and `Kernel` as structurally-distinct frozen types (Move 7 promoted both). Provable equivalence on deterministic events (DLRS Prop. 4.9) is a bridge, not a reduction.

### Baseline comparison
**Slug:** `baseline-comparison`.
**Legal with escape hatch.** Research baselines deliberately implement non-Bayesian decision mechanisms — argmax-of-means, fixed-threshold, cheapest-first — to empirically contrast against the principled EU-max agent. The paper depends on having these baselines; they are not bugs to be fixed.
**Required pragma:** `# credence-lint: allow — precedent:baseline-comparison — <which baseline, why non-Bayesian>`. Each baseline's violation is tagged so `grep -r 'baseline-comparison'` produces an inventory of what the paper compares against.
**Follows from Invariant 1 (topological face)** being scoped to the agent. The invariant forbids parallel decision mechanisms *that the agent uses*; a baseline is, by construction, not the agent. This precedent makes that distinction explicit.

### Test code computing expected values manually
**Slug:** `test-oracle`.
**Legal with escape hatch.** Tests *of* the reasoner legitimately need an independent oracle: `assert expect(m, f) == pytest.approx(0.7)  # computed by hand from Beta(3,7)`.
**Required pragma:** `# credence-lint: allow — precedent:test-oracle — <reason>` on the comparison line. The manual computation is the test's ground truth; it is causal *within the test*, but the test is non-causal with respect to the agent (it doesn't feed back into agent behaviour).

## Specific derivations

### Indifference implies exploration
When `EU(interact) == EU(wait)` (both zero), interact. `select_action` threshold is `>= 0`, not `> 0`. Follows from correct EU accounting: at indifference, VOI from the interaction outcome is still positive (you'll learn something), which is part of EU. If the threshold is strict, a correct EU computation will have already broken the tie. Listed here because it's the kind of edge case that gets "fixed" back to strict inequality under perceived instability, and the fix is wrong.

### Non-firing predicates predict the base rate
Programs whose predicates don't fire return `log(0.5)`, not `0.0`. A non-firing program is implicitly predicting "I don't know, 50/50"; that prediction is scored against the observation. Ranking: *informed-and-right* > *uninformed* > *informed-and-wrong*. Returning `0.0` makes non-firing programs unbeatable (no information → no penalty), creating weight rigidity after regime changes. Listed here because the bug manifests long after the change (the posterior stops adapting) and the fix looks like a tunable constant; it is not — it is scoring-rule calibration.

## Historical rejections

### PROPOSED: `(sample measure)` in the DSL for Thompson sampling.
**REJECTED.** Sampling is randomness; randomness is a side effect; the DSL is pure. Construct the posterior in the DSL; call `draw()` in the host. **Invariant 1 (spatial)** — DSL stays non-executing.

### PROPOSED: `host_decide()` / `host_optimise()` in host drivers.
**REJECTED.** `optimise` and `value` belong in the ontology alongside `expect` and `condition`. One implementation per operation. The host driver is pure orchestration — it calls ontology functions. **Invariant 1 (topological)** — one canonical path per operation.

### PROPOSED: `(thompson-sample m actions pref)` in stdlib that calls sample.
**REJECTED.** Compounds both errors above. Thompson sampling is `draw` (host) + `argmax` (ordinary computation on the drawn value). The DSL constructs the posterior; its job is done. **Invariant 1 (both faces).**

### PROPOSED: Bare lambda as a kernel's likelihood.
**REJECTED.** Kernels declare `likelihood_family` at construction; bare lambdas defeat conjugate dispatch and force probing. **Invariant 2.**

### PROPOSED: Flat coefficient arrays for `LinearCombination` instead of `Vector{Tuple{Float64, Functional}}`.
**REJECTED.** Flat indexing encodes stride conventions invisible to the type system. Each sub-functional must navigate its own structure. **Invariant 2** — composition is part of structure.

### PROPOSED: Inferring kernel family by probing `log_density == 0.0` for flat likelihoods.
**REJECTED.** Legitimate kernels can return zero log-density at specific points without being flat; the probe misfires and hides the assumption from the type system. **Invariant 2** — declared at construction, not dispatch.

### PROPOSED: DSL wrapper functions in domain files (e.g., `(defun choose-email (s) (optimise s email-actions email-pref))`).
**REJECTED.** Wrappers force the preference through an opaque closure that defeats Functional dispatch AND hide the causal arithmetic path from CI. A domain file contains data; axiom-constrained ops are called at the protocol level, not wrapped. **Invariants 1 and 2** jointly.
