# Escalate-arithmetic-depth design — the meta-action that raises `max_num_depth` by residual VOI

> **STATUS: DRAFT — awaiting ratification.** No code in this PR; design-doc-before-code per the arc
> discipline. Master plan: `docs/exploration-budget/master-plan.md` (§3.1 the selection/generation seam;
> the fine-before-coarse ladder `refine thresholds → add features → construct arithmetic-derived features
> → (far future) continuous`; §3.2 dominance). Discharges the **named successor** the feature-arithmetic
> move parked (`feature-arithmetic-design.md` §1 "Depth is earned, not toggled," §6.1 risk 1, §5 Q4). The
> const-free NumExpr sublayer this move drives is already on the branch
> (`09bab94 feat(program-space): the NumExpr numeric sublayer`). Authored 2026-07-02.

---

## 1. Purpose

The feature-arithmetic move shipped the numeric sublayer parameterised by `max_num_depth`
(`enumeration.jl:140-163`), with `max_num_depth = 1` the behaviour-preserving floor (only `FeatureRef`
atoms — the lifted image of the pre-arithmetic grammar). It **deliberately did not** ship the thing that
raises the lever: the doc framed depth as *brain-earned from the start* ("a host-toggled depth would make
this the one rung of the arc where a discovery is *configured* rather than *earned*," §1) and named the
**escalate-arithmetic-depth meta-action** as the immediate successor (§6.1). This is that move.

**The op.** A VOI-scored exploration meta-action that raises `max_num_depth` for the agent's top grammar by
one, admitting the next depth-tier of arithmetic `NumExpr`s (`Times(A,B)`, `A−B`, `A/B`, …) into the
enumerable predicate alphabet — scored against the belief's predictive residual **exactly as**
`explore_grammar` (threshold refinement) and `explore_features` (feature addition) are: enumerate the
richer grammar, re-condition the buffer, measure the realised marginal-log-loss reduction `Δℓ`, price the
compute spent. It generalises `:gw_deepen` — but *not* by copying it. `:gw_deepen` raises the **boolean**
program depth (`current_max_depth`) and is scored by the **escape-mass entropy heuristic** because a
deeper program tree is myopic-unreachable un-entertained mass (dominance-design §0). Arithmetic-depth
escalation is the **opposite case**: the candidate `NumExpr`s at depth `d+1` are enumerable, their
predicates evaluate on the *existing* buffer, and their fit is *exactly measurable* by the same lookahead
`explore_features` uses. So this op belongs on the **exact-VOI tier**, not the escape-mass tier — and that
placement is the doc's load-bearing claim (§4, §5 Q1).

**Why now, and what unblocks.** The feature-arithmetic move made products *expressible*; without this
op they are expressible only at a **host-set** `max_num_depth`, which is precisely the "configured, not
earned" failure the arc is built to avoid. This move closes the "depth is earned" promissory note: it
makes the depth a **decision** the agent takes when its residual cannot be explained without a product,
priced in the one Δ log-evidence currency (Move 5). It also makes the const-free arithmetic sublayer
*usable in the meta-loop* rather than a dormant kwarg. What it does **not** do: it does not add
`ConstSlot` (a separate parked successor), and it does not resolve the breadth-wall benchmark (named,
deferred — §6).

**The cost caveat that reshapes this move (the dominance finding).** The dominance benchmark on the
`exploration-budget/dominance` branch (`apps/julia/dominance_benchmark/results/summary.md`) **just measured
that unpriced exact VOI over-fires**: with `compute_cost = 0` and only the plateau soft-gate, `eu_max`
fired ~630 meta-actions per run and **lost** on AUC to a fixed-`k50` schedule (4 meta-actions) and to
`random_p005` (12 meta-actions) — the paired AUC gaps were `−10.58 [−17.17, −4.09]` vs fixed-k50 and
`−7.75 [−15.48, −0.92]` vs random-p005 (the dominance gate FAILS; only `eu_max − never_explore =
6.06 [0.89, 10.97]`, exploration's isolated value with the escape-mass held constant, is positive). The
lesson is not "exact VOI is wrong" — it is "**an exact VOI with no compute price over-fires**," and
escalation is the **most expensive op in the roster** (`O((features·ops)^d)` enumeration + a full buffer
re-conditioning at the richer depth, feature-arithmetic §6.1). So this move cannot ship escalation on the
exact tier without engaging how its compute cost enters the score (§3, the genuine open question). A
zero-cost escalation would be the worst offender in exactly the failure the benchmark just surfaced.

## 2. Files touched (the follow-up code PR)

Exhaustive; the shape mirrors `explore_features` (the exact-VOI sibling), not `:gw_deepen`.

**`src/program_space/exploration.jl`** — modification. A new exact-VOI accessor pair + shared core, in the
established `_best_*` idiom (the Invariant-3 one-computation-two-projections pattern of
`_best_threshold_refinement` / `_best_feature_addition`, `exploration.jl:297-322, 399-424`):

```julia
# The candidate "grammar edit" here is not a grammar mutation — max_num_depth is an ENUMERATION
# parameter, not a Grammar field (§2 decision). So the lookahead varies the depth, not g.
function _best_depth_escalation(g::Grammar, observations::Vector{ExploreObservation},
                                max_depth::Int, current_num_depth::Int;
                                action_space::Vector{Symbol} = Symbol[:classify],
                                compute_cost::Float64 = 0.0)::Tuple{Int, Float64}
    isempty(observations) && return (current_num_depth, 0.0)
    baseline = _grammar_marginal_log_loss(g, observations, max_depth, action_space;
                                          max_num_depth = current_num_depth)
    cand_depth = current_num_depth + 1
    mll = _grammar_marginal_log_loss(g, observations, max_depth, action_space;
                                     max_num_depth = cand_depth)
    # Δℓ only: raising max_num_depth does NOT change g.complexity (§4 — the numeric depth is a
    # per-NumExpr description-length unit paid by expr_complexity at enumeration, not a grammar
    # symbol), so the prior term is carried inside the mll exactly as a threshold refinement is,
    # NOT charged separately like a feature. This is explore_grammar's Δprior = 0 shape, one rung up.
    voi = net_value(baseline - mll, compute_cost)
    voi > 0.0 ? (cand_depth, voi) : (current_num_depth, 0.0)
end

escalate_arithmetic_depth(g, observations, max_depth, current_num_depth; kw...) =
    first(_best_depth_escalation(g, observations, max_depth, current_num_depth; kw...))
depth_escalation_voi(g, observations, max_depth, current_num_depth; kw...) =
    last(_best_depth_escalation(g, observations, max_depth, current_num_depth; kw...))
```

This requires `_grammar_marginal_log_loss` (`exploration.jl:198-216`) to thread a `max_num_depth` kwarg
through to its `enumerate_programs` call (`:200`) — a one-line change, defaulting to `1` so every existing
caller is bit-identical.

**`src/Credence.jl`** — modification. Export `escalate_arithmetic_depth`, `depth_escalation_voi` (peers of
`exploration_voi` / `feature_discovery_voi`).

**`src/program_space/agent_state.jl`** — modification (**pending §2 decision below**). Add
`current_num_depth::Int` to `AgentState` (`agent_state.jl:15-35`), defaulting to `1`, alongside the
existing `current_max_depth::Int` (`:21`). The escalation op increments it, exactly as `:gw_deepen`
increments `current_max_depth`. See §2's open question for the Grammar-field alternative and the
recommendation.

**`apps/julia/grid_world/host.jl`** — modification.
- `GW_META_ACTIONS` (`host.jl:51`) gains `:gw_escalate_depth`.
- `score_gw_meta_actions` (`host.jl:183-249`): add `:gw_escalate_depth` to the **exact-VOI tier** (§4),
  scored `plateau * depth_escalation_voi(g_top, explore_buffer, current_max_depth, state.current_num_depth;
  …)`, memoised on the same `voi_cache` shape with a fresh key band (`3000 + cache_epoch`), and
  compute-priced per §3. It is **not** an escape-mass op — it does not join the `entropy − escape_cost`
  block (`:245-247`).
- The `execute` dispatch (the `elseif action == :gw_deepen` block, `host.jl:336-347`) gains an
  `elseif action == :gw_escalate_depth` branch: `state.current_num_depth += 1`, re-add the top grammars'
  programs at the new `max_num_depth`, reset the residual regime (an alphabet expansion, Move 2 Q1b —
  same as every other growth op).
- Enumeration call sites that build the live belief (`host.jl:436` and the `add_programs_to_state!` path)
  thread `max_num_depth = state.current_num_depth`.

**`src/program_space/enumeration.jl`** — no signature change (the `max_num_depth` kwarg already exists,
`:144`); only `_grammar_marginal_log_loss`'s pass-through (listed under exploration.jl).

**New test** `test/test_escalate_depth.jl` —
- the exact-VOI shape: `depth_escalation_voi` on a buffer generated by a depth-2 rule (`(A×B)>t`) is
  strictly positive; on a buffer generated by a depth-1 rule it is `0.0` (no fit gain — the no-op floor);
- Invariant-3 agreement: the depth `escalate_arithmetic_depth` returns == the depth whose VOI
  `depth_escalation_voi` reports (they share `_best_depth_escalation`);
- **the acquired-only-via-the-meta-action test (§5, the centrepiece):** a grid_world task whose true rule
  needs a depth-2 product, run through the full meta-loop, acquires the correct predicate — and asserts
  the depth was raised **by the escalation op firing**, with `state.current_num_depth == 1` at the start
  and no host toggle anywhere in the harness;
- compute-price monotonicity: raising `compute_cost` suppresses the escalation smoothly (the `net_value`
  no-cliff property), and at a high enough price the op no-ops even when Δℓ > 0.

## 3. Behaviour preserved

The escalation op adds a meta-action; it changes no existing arithmetic. The preservation spine:

- **The `max_num_depth = 1` floor is untouched.** Every existing `enumerate_programs` /
  `_grammar_marginal_log_loss` caller passes (or defaults to) `max_num_depth = 1`, so the pre-change
  enumeration, complexity, prior log-weights, and posteriors are `==` (strata-1 unit,
  `isapprox(atol = 1e-14)`). The feature-arithmetic move's own `max_num_depth = 1` `==` pin
  (`test_feature_arithmetic.jl`) is the backstop — this move adds no new atoms at the floor.
- **`_grammar_marginal_log_loss`'s new kwarg defaults to `1`.** The existing `explore_grammar` /
  `explore_features` lookaheads call it without the kwarg and stay bit-identical; capture-before-refactor
  pins their returned grammars `==` on the existing fixtures (strata-1).
- **`:gw_deepen` (boolean depth) is a separate op and is not touched.** This move adds a numeric-depth
  op beside it; the two increment different `AgentState` fields (`current_max_depth` vs
  `current_num_depth`) and sit on different scoring tiers.

The one **intended** behaviour change is the grid_world meta-action sequence on tasks that need a product:
those runs will now fire `:gw_escalate_depth` where before the depth was pinned at 1. Following the
dominance-design precedent (§0 "behaviour shift is intended"), the meta-loop test asserts the **real**
escalation behaviour (the op fires, the depth rises, the rule is acquired), not "no error."

Strata-3 end-to-end `isapprox(rtol = 1e-10)` on any energy trajectory the meta-loop test pins; halt-the-line
at greater drift.

## 4. Worked end-to-end example — the escalation through the exact-VOI tier

**Setup (the dual residency this move introduces: depth lives in `AgentState`, is *read* by the
enumeration/lookahead).** Grid_world grammar `g` over `{:x_norm, :y_norm}`. `state.current_num_depth = 1`
(the floor). The true rule is *"enemy iff `x_norm × y_norm > 0.25`"* — the far-corner hyperbola the
axis-aligned depth-1 grammar provably cannot name (feature-arithmetic §4). The agent has accrued a buffer
of ~80 observations under the depth-1 alphabet; its posterior mispredicts in the corner (high residual
mass there).

**Score (module `program_space`, `depth_escalation_voi`; called from `apps/julia/grid_world` host
`score_gw_meta_actions`).**
1. `baseline = _grammar_marginal_log_loss(g, buffer, max_depth; max_num_depth = 1)` — the evidence for the
   buffer under the depth-1 (product-free) alphabet. The corner residual is unexplained, so `baseline` is
   high.
2. `mll = _grammar_marginal_log_loss(g, buffer, max_depth; max_num_depth = 2)` — the same lookahead with
   the depth-2 `NumExpr` set enumerated, which now contains `Times(FeatureRef(:x_norm),
   FeatureRef(:y_norm))`; the atom `GTExpr(Times(...), 0.25)` (threshold from the observed products' grid
   — §5 Q3) enters, and the program `IfExpr(GTExpr(Times(...), 0.25), enemy, food)` gets Beta(1,1) prior
   weight. Re-conditioning the buffer, the product program earns weight (it names the corner); `mll` drops.
3. `voi = net_value(baseline − mll, compute_cost)` — `Δℓ` is large and positive (the product explains what
   nothing at depth 1 could), minus the escalation's compute price (§3). **Δcomplexity is not charged
   separately** — this is `explore_grammar`'s `Δprior = 0` shape, one rung up: raising `max_num_depth` adds
   no *grammar* symbol (it is not a `feature_set` member), and the depth-2 predicate's own
   description-length is already paid by `expr_complexity(GTExpr(Times(...))) = 3` *inside* the enumerated
   program's prior, which rides the normalized `mll`. So, like a threshold refinement, the prior term is
   carried by the marginal likelihood, not added at the argmax. (Contrast `explore_features`, which
   charges `−log2·Δcomplexity` explicitly because a new *feature* is a grammar symbol whose per-grammar
   constant cancels inside the mll — `exploration.jl:340-357`. A numeric-depth increment is not a grammar
   symbol, so there is nothing to un-cancel.)

**Select (module `program_space` host `score_gw_meta_actions` → the `argmax`).** `:gw_escalate_depth`
scores `plateau · Δℓ − compute_cost`. If it clears `0.0` (the act-now reference) and beats the other exact
ops (threshold refinement and feature addition have no positive VOI here — no threshold or base feature
names the hyperbola), the meta-loop selects it.

**Execute (module `apps/julia/grid_world` host `execute`).** `state.current_num_depth += 1` → `2`; the top
grammars' programs are re-added at `max_num_depth = 2` (the product program now in the live belief);
residual regime reset. On subsequent data the product program both *fits* and is *short*, and
`expect(posterior, u_action)` (module `ontology`) selects the correct action in the corner.

**Dual-residency ledger (mandatory per template §4).** The numeric depth has two homes and the authority
is clean at each step: it is **stored** in `AgentState.current_num_depth` (host-owned mutable meta-state,
the authoritative value); it is **read** by `enumerate_programs` / `_grammar_marginal_log_loss` as the
`max_num_depth` enumeration parameter (never stored there — the kwarg is transient). The lookahead
`_best_depth_escalation` reads `current_num_depth` and probes `current_num_depth + 1` **without mutating
state** (it passes the candidate depth as a kwarg); only `execute` writes back. This mirrors exactly how
`current_max_depth` already works for `:gw_deepen` (`host.jl:337`), so it introduces no new residency
*pattern* — it is the second instance of an existing one, which is why AgentState (not Grammar) is the
recommended home (§2 Q1).

## 5. Open design questions (the genuine forks)

### Q1 — the exact-VOI tier vs the escape-mass tier: is depth escalation *really* myopically measurable? (the load-bearing placement)
`:gw_deepen` (boolean depth) is scored by the entropy escape-mass heuristic (`host.jl:245-247`) with the
ratified justification (dominance-design §0) that a deeper program tree grows **un-entertained hypotheses
that are myopic-unreachable by construction** — there is *no exact VOI* because the value is in mass the
belief does not yet carry. The whole architecture of this doc rests on the claim that **arithmetic-depth
escalation is category-different: it *is* myopically measurable**, because the depth-`d+1` `NumExpr` set is
finite and enumerable *now*, its predicates evaluate on the *existing* buffer, and re-conditioning gives a
real `Δℓ` — exactly `explore_features`'s situation, not `:gw_deepen`'s. **The counter to weigh honestly:**
this is *also* what `:gw_deepen` could claim — deepening the boolean tree also enumerates a finite set and
re-conditions — so what actually separates them? The answer I'll defend: `:gw_deepen`'s value is
overwhelmingly in trees whose *predicate atoms don't yet exist in the grammar's alphabet at all* (it is a
proxy for "the grammar is too small"), whereas arithmetic escalation's value is in a **specific, small,
enumerable** set of new atoms over the *existing* features — the lookahead is affordable and the fit is
real. But this is a *quantitative* distinction (affordable-enough lookahead), and if the depth-2 `NumExpr`
count is large the lookahead itself may be unaffordable, collapsing escalation back toward the escape-mass
case. **Recommendation: exact-VOI tier, with a fidelity fallback** — score it exactly (like
`explore_features`) *when the depth-`d+1` enumeration is within a compute budget*, and if it is not, fall
back to the escape-mass entropy heuristic (the honest `:gw_deepen` treatment) rather than pay an
unaffordable exact lookahead. That fallback is itself the Move-5 fidelity cascade (cheap surrogate ↔
exact), one rung deeper. **This is the question I most want pushback on**, because it decides whether the
op is a peer of `explore_features` or a peer of `:gw_deepen`, and the honest answer may be "both, keyed on
affordability."

### Q2 — where the depth lives: `AgentState.current_num_depth` vs a `Grammar.num_depth` field
`max_num_depth` is today *only* an `enumerate_programs` kwarg — no persistence anywhere. Two homes:
- **(a) `AgentState.current_num_depth`** (recommended). The precedent is exact: `current_max_depth`
  (boolean depth) already lives on `AgentState` and is incremented by `:gw_deepen` (`host.jl:337`). Numeric
  depth is the same *kind* of thing — a per-agent enumeration-budget lever the meta-loop raises — so it
  belongs in the same place, and the dual-residency pattern (§4) is already established for its sibling.
  `Grammar` stays immutable and thresholds-only; the fresh-id-per-refinement pattern (`_refine_grammar`,
  `_add_feature`) is untouched.
- **(b) `Grammar.num_depth`** (the alternative to argue against). Depth *feels* grammar-structural (like
  `thresholds`, which live on Grammar). But it is not: `thresholds` is a per-*feature* grid the grammar
  carries because different features have different observed-value grids; `max_num_depth` is a single
  scalar over the *whole* enumeration, not per-anything, and putting it on Grammar would force a fresh
  Grammar id (and a fresh grammar in `state.grammars`) on every escalation — conflating "which grammar"
  (the BMA mixture is over grammars) with "how deep to enumerate it," a Single-responsibility smell
  (Invariant 3). And escalation would then have to re-key the grammar mixture, not just bump a scalar.
- **Recommendation: (a) AgentState.** Depth is enumeration-budget meta-state, not grammar-structural data;
  its sibling already lives there; Grammar immutability + the fresh-id pattern stay clean. **Open
  sub-point for pushback:** whether `current_max_depth` (boolean) and `current_num_depth` (numeric) should
  be one field or two — they are semantically distinct axes (tree depth vs arithmetic depth), so two; but
  a reviewer may argue for a single `depths::Dict{Symbol,Int}` if more depth axes are coming.

### Q3 — per-NumExpr threshold-grid attachment (feature-arithmetic §5 Q4's deferred half)
Escalated depth-`d` compound expressions currently threshold over the **default seed grid**
(`_num_threshold_grid(::Grammar, ::NumExpr) = THRESHOLDS`, `enumeration.jl:138`) — a bare `FeatureRef`
reads the grammar's refined per-feature grid, but a `Times(A,B)` falls back to the coarse
`[0.1,0.3,0.5,0.7,0.9]`. That is a real gap: the observed values of `A×B` are not the observed values of
`A` or `B`, so the seed grid may miss the split (`0.25` in §4 is not a seed point). The explore path should
refine **per-`NumExpr`** observed-value grids, generalising `_threshold_candidates`
(`exploration.jl:65-82`, which today keys on `feat::Symbol`). The key question: `Grammar.thresholds` is
`Dict{Symbol, Vector{Float64}}` — a compound `NumExpr` has no `Symbol` key.
- **(a)** Extend the key type to `Dict{NumExpr, Vector{Float64}}` (or a union) — but this makes Grammar
  carry arithmetic structure, re-opening the Invariant-3 question (`thresholds` becomes non-trivially
  keyed) and forcing `NumExpr` to be hashable/`==`-comparable as a dict key.
- **(b)** Don't store compound grids on the Grammar at all — compute them **on the fly** in the lookahead
  from the buffer's observed `NumExpr` values (evaluate the compiled `NumExpr` over the buffer, take
  residual-midpoints, exactly Move 3's mechanism but on the compiled expression rather than the raw
  feature). The grid is transient, never persisted; the grammar stays `Symbol`-keyed. This matches the
  feature-arithmetic move's own recommendation ("per-`NumExpr` observed-value grid … evaluated on the
  compiled `NumExpr`," feature-arithmetic §5 Q4) and inherits Move 3's complexity-invariance (the grid is
  not charged).
- **Recommendation: (b), transient per-`NumExpr` grids computed in the lookahead.** It keeps Grammar
  `Symbol`-keyed (no Invariant-3 pressure, no `NumExpr`-as-dict-key machinery), and the grid *is* data-fit
  (observed midpoints) exactly as the bare-feature grid is. **Sequencing sub-question for pushback:**
  should (b) land *in this move* (escalation is much weaker without it — depth-2 products thresholded on a
  coarse seed grid may not fit, so the escalation VOI is under-measured), or as a fast-follow? I lean
  **fold it into this move** — an escalation whose new atoms threshold on the wrong grid would measure a
  spuriously low Δℓ and the op would under-fire, undercutting the very demonstration §5 needs. But it
  widens the move; the reviewer decides whether the demonstration can stand on the seed grid first.

### Q4 — compute-cost mechanism: how the escalation's price enters the one currency (the dominance-finding question)
The dominance benchmark proved unpriced exact VOI over-fires (§1), and escalation is the most expensive op
(`O((features·ops)^d)`). Its compute price is a **genuine open question** — three candidate mechanisms:
- **(a) A declared cost kwarg** (like `escape_cost = log(2)`, `host.jl:61`). Simplest: a
  `depth_escalation_cost` constant the host passes as `compute_cost`. Honest (it *is* a declared price the
  caller owns) but arbitrary — the number is hand-set, and the dominance failure is *exactly* what
  hand-set exploration constants produce.
- **(b) A measured enumeration-size cost.** Price the escalation by the *actual* size of the depth-`d+1`
  enumeration it triggers — `compute_cost ∝ |programs at d+1| − |programs at d|`, the real compute the op
  will spend, in the same log-evidence nats (a bit-count of the enumeration is a description-length, which
  is nats). This makes the price **endogenous and resource-rational** — escalation pays more when it opens
  more, which is exactly when it should hesitate — and it needs no hand-tuned constant. It is more work
  (the host must estimate the enumeration size before committing), and the units-bridge (enumeration bits
  → the Δℓ currency) needs care, but it is the principled answer.
- **(c) Defer to the priced-VOI experiment's evidence.** The dominance branch has a queued follow-up (the
  priced-VOI dominance experiment) that will measure whether a *general* compute price on all exact-VOI ops
  fixes the over-firing. If it lands first, escalation inherits whatever pricing that experiment validates,
  rather than inventing its own.
- **Recommendation: (b) measured enumeration-size cost, informed by (c).** The dominance finding is a
  direct mandate against (a)'s hand-set constants; (b) is the resource-rational form the arc's own thesis
  demands ("heuristics live inside EU-max," and a measured compute price *is* EU-max over compute); and (c)
  de-risks the units-bridge by grounding the coefficient in the experiment's evidence rather than a guess.
  **The honest caveat I'll flag:** if the priced-VOI experiment shows a *uniform* per-op price suffices,
  (b)'s per-op-size refinement may be over-engineering — so this recommendation is explicitly *contingent
  on (c)'s result*, and the fallback is the uniform declared price the experiment validates. This is the
  second question I most want pushback on, because it couples this move to an in-flight empirical result.

## 6. Risk + mitigation

1. **The op over-fires (the dominance finding, applied to the most expensive op).** Failure mode: shipped
   on the exact tier with `compute_cost = 0`, escalation fires whenever *any* product has positive Δℓ,
   pays `O((features·ops)^d)` every time, and reproduces the ~630-meta-action over-firing the benchmark
   caught — worse, because escalation is the costliest op. *Blast radius:* the dominance benchmark's AUC
   (escalation would join `eu_max` and drag its meta-action count further up); grid_world wall-clock.
   *Mitigation:* the compute price (§3, §5 Q4) is **not optional** — the op ships priced, and the meta-loop
   test asserts compute-price monotonicity (raising the price suppresses the op smoothly, no cliff). The
   escalation is added to the dominance benchmark's `eu_max` policy and the gate re-run; if `eu_max`'s AUC
   gap does not improve (or worsens), **halt and report** — do not ship an op that makes the failing gate
   fail harder.
2. **Unaffordable lookahead (the exact-VOI-tier bet fails).** Failure mode: the depth-`d+1` enumeration is
   so large that scoring the op (the lookahead itself) hangs or dominates the step budget — the
   `breadth^depth` wall (feature-arithmetic §6.1). *Blast radius:* `depth_escalation_voi` wall-clock; the
   meta-level lookahead cost. *Mitigation:* the §5 Q1 fidelity fallback — if the depth-`d+1` enumeration
   exceeds a compute budget, fall back to the escape-mass entropy heuristic (the honest `:gw_deepen`
   treatment) rather than pay the unaffordable exact lookahead. Per the master plan's performance clause:
   if a run hangs or exact enumeration is too slow, **STOP and report** (no silent approximation) — the
   fallback is a *designed* fidelity tier, ratified, not an ad-hoc cap.
3. **The prior term double-charged or dropped (the §4 subtlety).** Failure mode: treating depth escalation
   like `explore_features` and charging `−log2·Δcomplexity` (there is no grammar-symbol Δcomplexity to
   charge — the numeric depth is not a `feature_set` member), or treating a compound predicate's own
   `expr_complexity` as uncharged (it *is* charged, inside the enumerated program's prior, riding the
   normalized mll). *Blast radius:* `_best_depth_escalation`'s VOI is wrong → the op fires at the wrong
   boundary. *Mitigation:* the test asserts the boundary sits at `Δℓ` (the `explore_grammar` Δprior=0
   shape), *not* `Δℓ − log2` (the `explore_features` shape) — a direct analogue of
   `test_feature_discovery.jl §4`'s boundary assertion, one rung up.
4. **Grid mismatch under-measures the VOI (§5 Q3 deferred).** Failure mode: if the per-`NumExpr` grid is
   *not* folded in, depth-2 products threshold on the coarse seed grid, the fit is under-measured, and the
   escalation under-fires — the §5 demonstration fails not because the mechanism is wrong but because the
   grid is coarse. *Blast radius:* the acquired-only-via-the-meta-action test. *Mitigation:* §5 Q3
   recommends folding the per-`NumExpr` grid into this move for exactly this reason; if the reviewer defers
   it, the demonstration task must be chosen so the seed grid *does* contain the split (e.g. true threshold
   at `0.5`, a seed point) — named explicitly so it is a decision, not a silent confound.
5. **Depth-field residency drift (§2).** Failure mode: `current_num_depth` on `AgentState` and the
   `max_num_depth` kwarg threaded to `enumerate_programs` diverge — a call site enumerates the live belief
   at a stale depth. *Blast radius:* the live belief and the lookahead disagree on the alphabet.
   *Mitigation:* single authority (§4 dual-residency ledger) — `AgentState.current_num_depth` is the only
   stored value; every enumeration reads it; the kwarg is transient. *Pre-emptive grep*
   `grep -rn 'enumerate_programs\|_grammar_marginal_log_loss' apps/julia/grid_world/ src/program_space/`
   before the PR: each live-belief call site threads `max_num_depth = state.current_num_depth`; each
   lookahead call site threads the candidate depth. List each hit and its disposition.

## 7. Verification cadence

```
julia test/test_escalate_depth.jl        # exact-VOI shape, Invariant-3 agreement, acquired-via-op, price monotonicity
julia test/test_feature_arithmetic.jl    # the max_num_depth=1 floor still == (this move adds no floor atom)
julia test/test_threshold_explore.jl     # explore_grammar lookahead unchanged (max_num_depth kwarg defaults to 1)
julia test/test_feature_discovery.jl     # explore_features unchanged (sibling tier)
julia test/test_grid_world_meta.jl       # the meta-loop gains :gw_escalate_depth on the exact tier
```

Full `test/test_*.jl` green before commit; lint self-test + `check apps/`. **Skin smoke optional** —
arithmetic depth is a brain-decided meta-action (VOI-scored, host-wired into the EU-max loop exactly as
`explore_grammar`/`explore_features`/`:gw_deepen` are, not a host toggle); it touches no serialised wire
path. If this move feeds the dominance benchmark (§6 risk 1), the dominance gate is re-run with
`:gw_escalate_depth` in the `eu_max` policy and the AUC gap reported — a **halt-the-line** condition if the
op degrades the gate. Halt-the-line on any drift in the pre-change enumeration, any full-suite failure, or
any regression in the dominance gap — the branch never sleeps red.
