# ConstSlot design — the learnable numeric constant (the `{+,×,const}` density rung)

> STATUS: DRAFT — awaiting ratification. No code.
>
> Exploration-budget arc, the fast-follow to the feature-arithmetic move (that doc's §5 Q3, deferred and
> named). Master plan: `docs/exploration-budget/master-plan.md` (§3.1 the selection/generation seam; the
> fine-before-coarse escalation ladder; SPEC §1.3 the complexity prior). Predecessor on this branch: the
> coefficient-free NumExpr sublayer (`src/program_space/types.jl` `NumExpr` hierarchy — `FeatureRef`,
> `Times`, `Plus`, `Minus`, `Div`, `AQ`, `Neg`; commit 09bab94). **This move is GOVERNED by the
> `decision-free-combinator` precedent** (CLAUDE.md slug index + `docs/precedents.md`): `ConstSlot` is
> the precedent's named first application, the exact place a baked threshold could re-enter through the
> back door. Read the feature-arithmetic design **entirely** — especially §1 (theory-claim scoping),
> §5 Q3 (this move's charter), §6 risk 2 (the const-slot back door) — before reviewing this doc.
> Authored 2026-07-02.

---

## 1. Purpose

Add **one** node to the numeric sublayer — `ConstSlot`, a `NumExpr` leaf whose value ranges over
**data-derived candidates**, enumerated and complexity-priced exactly as a threshold is. This closes the
coefficient gap the feature-arithmetic move left open by design: today the sublayer can say `A × B > t`
(a coefficient-free product) but **not** `a·x + b > t` (an affine rule with a *learnable slope*). With
`ConstSlot` in the leaf alphabet, `{+, ×, const}` over features is complete — the same `{+,×,const}` that
Stone–Weierstrass makes **polynomially dense** in the continuous predicates. This is the theory claim the
coefficient-free move deliberately did **not** borrow (feature-arithmetic §1: *"It is not
Stone–Weierstrass-complete — that theorem needs the constants … reserved for when the learnable-constant
slot lands"*). This move earns it — and states its honest limits (density ≠ learnability; the claim is
about what the basis *can name at sufficient depth/grid*, not what the enumerator *finds within a budget*).

The load-bearing subtlety, which §5 Q1 owns and which distinguishes this from a threshold: a threshold sits
at a **comparison side** and has a natural grid — the observed feature values. A `ConstSlot` in a
**coefficient position** (`a·x`) multiplies a feature and has **no such natural grid**: the "right" `a`
depends on the units and scale of `x` and of whatever it is compared against. The whole legitimacy of this
move rests on deriving those candidates **from the data**, not baking them — because a `ConstSlot` with a
literal value is precisely the `decision-free-combinator`-illegal object (an injected answer, not a prior).
§5 Q1 develops four honest derivation options and recommends one.

**What unblocks.** Affine and higher-order polynomial predicate boundaries (`a·x + b·y > t`, `x² − a > t`)
become priced, enumerable hypotheses — the last expressiveness rung before genuinely novel sensory
dimensions. And the arc's headline theory claim moves from "multiplicative-interaction compactness" (what
the coefficient-free move earned) to "polynomial density" (what this move earns), with both scoping caveats
carried honestly.

**Depth/breadth is earned, not toggled — inherited from the predecessor.** As with `max_num_depth`, a
`ConstSlot` grid is a breadth multiplier, and its *use* is a meta-decision. `ConstSlot` is **off at the
floor** (empty candidate set ⇒ the leaf never enumerates ⇒ the coefficient-free enumeration is reproduced
bit-for-bit); a VOI-scored meta-action turns it on when the residual cannot be explained coefficient-free.
This is the same autonomy thesis the feature-arithmetic move states for arithmetic depth (a discovery is
*earned*, never *configured*), and it is the resource-rational answer to the `breadth × grid` cost (§6).

## 2. Files touched (the code PR this doc gates — no code here)

**`src/program_space/types.jl`** — modification (the `NumExpr` block, ~`types.jl:37-100`).

```julia
struct ConstSlot <: NumExpr
    value::Float64          # a DATA-DERIVED candidate value, never a literal chosen at definition time
end
```

- `show_num(e::ConstSlot) = string(e.value)` — renders the fitted value (e.g. `"0.25"`).
- `num_equal(a::ConstSlot, b::ConstSlot) = a.value == b.value`; the `num_equal(::NumExpr, ::NumExpr) = false`
  fallback already handles cross-type.
- `num_feature_refs!(acc, ::ConstSlot) = acc` — a constant references no feature (the one method with an
  empty body; **no generic fallback exists**, so omitting it fails loud, per the existing comment at
  `types.jl:88` — the method must be added).
- **The legitimacy comment** in the `NumExpr` docstring (`types.jl:29-35`) extends: `ConstSlot` is a
  decision-free combinator *with a free-parameter slot* — the slot's value is data-fit by the threshold
  machinery, never a literal. Cross-reference the precedent.

**`src/program_space/enumeration.jl`** — modification.
- `num_complexity(e::ConstSlot) = 1 + <grid-choice term>`. §5 Q3 decides the exact charge; the recommended
  form is `num_complexity(::ConstSlot) = 1` (one symbol) with the `log₂ |grid|` "which candidate" bits
  **left to the predictive marginal likelihood** (the SPEC §1.3 predictive route, exactly as the threshold
  grid is), so a `ConstSlot`'s node cost is grid-size-invariant like `GTExpr`'s.
- `_num_exprs_by_depth` (`enumeration.jl:94-130`): `ConstSlot` candidates enter the **depth-1** leaf set
  alongside `FeatureRef` — a constant is a depth-1 numeric expression, so `a·x` (`Times(ConstSlot, FeatureRef)`)
  is a depth-2 combination, correctly. The candidate values come from a new argument
  `const_grid::Vector{Float64} = Float64[]` (the floor is empty ⇒ no `ConstSlot` enumerated). Determinism:
  the grid is sorted before it enters the leaf set.
- **The degeneracy guards** (`enumeration.jl:114-125`) extend: `ConstSlot × ConstSlot`, `ConstSlot ± ConstSlot`,
  `Neg(ConstSlot)`, `ConstSlot / ConstSlot` are all constant-valued and must be excluded (a compound of only
  constants is a single fitted number — collapse it to one `ConstSlot`, or better, forbid the combination so
  the enumeration never emits an arithmetic tree whose leaves are all constants). §5 Q4 owns the exact rule.

**`src/program_space/compilation.jl`** — modification. `compile_num(e::ConstSlot) = (_f, _ts) -> e.value`
— a closure ignoring features and temporal state, returning the fitted constant. One method; total by
construction.

**`src/program_space/exploration.jl`** — modification (the load-bearing file). A new
`_const_candidates(g, observations) → Vector{Float64}` implementing the §5 Q1 recommended derivation, and
the `explore_*` meta-action that turns the grid on (the `ConstSlot`-enable meta-action, VOI-scored — §6).
This is where the "data-derived, never baked" contract is *mechanised*: the candidate set is a function of
the observation buffer, exactly as `_threshold_candidates` (`exploration.jl:65-82`) is.

**`src/program_space/perturbation.jl`** — modification. `extract_subtrees` / `expr_equal` already treat a
`GTExpr`'s `lhs` as an opaque leaf (feature-arithmetic §2); `ConstSlot` rides inside that opacity, no change
needed for this move. Named limitation (inherited): a recurring `Times(ConstSlot, FeatureRef)` is not yet
abstracted into a nonterminal.

**`src/Credence.jl`** — export `ConstSlot`; add `const_grid` (or the chosen candidate-source kwarg) to the
`enumerate_programs` surface, defaulted empty (the floor).

**New test** `test/test_constslot.jl` — see §6.

## 3. Behaviour preserved

The move is **semantically neutral at the floor by construction**: with an empty `ConstSlot` candidate set,
`_num_exprs_by_depth` emits no `ConstSlot` leaf, so the enumeration, complexity, prior log-weights, and
posterior are bit-identical to the coefficient-free predecessor. The pin:

- **Floor equivalence** — `enumerate_programs(g, d; max_num_depth = k, const_grid = Float64[])` reproduces
  the predecessor's program **count**, per-program `show_expr`, `expr_complexity`, and prior log-weights
  `==` for every `k`. This is the `const_grid = []` analogue of the feature-arithmetic move's
  `max_num_depth = 1` pin (that move's `test/fixtures/feature_arithmetic_lift_v1.tsv` spine).
- **Complexity of a bare threshold unchanged** — `expr_complexity(GTExpr(FeatureRef(:a), t)) == 1` still,
  because no existing program contains a `ConstSlot`.

Tolerances per template: strata-1 unit `isapprox(atol = 1e-14)` (enumeration counts/complexities are exact
integers/strings — `==`); strata-3 end-to-end `isapprox(rtol = 1e-10)`. Halt-the-line at greater drift.
Existing suites green after the change: `test_feature_arithmetic`, `test_program_space`,
`test_threshold_explore`, `test_feature_discovery`, `test_grid_world_meta`.

## 4. Worked end-to-end example — the affine rule a coefficient-free basis cannot name

**Task.** A grid_world grammar `g` over `{:x_norm, :y_norm}`. The true rule is *"enemy iff
`2·x_norm − y_norm > 0.3`"* — an affine half-plane with a **non-unit slope** `2` on `x_norm`. The
coefficient-free sublayer can form `Minus(FeatureRef(:x_norm), FeatureRef(:y_norm))` — i.e. `x − y > t`, a
half-plane at **slope 1** — but it **cannot tilt the boundary to slope 2** at any depth: without a
learnable coefficient, every difference of features is a fixed-slope line. The staircase of axis-aligned
`And(GT,GT)` tiles approximates the tilted half-plane only with many rectangles (the same compactness gap
the product example showed, now for a *slope* the basis structurally cannot express).

**Candidate derivation (module `program_space`, `_const_candidates`, the §5 Q1 recommendation — ratios).**
From the observation buffer, the recommended derivation proposes coefficient candidates as **ratios of
observed feature values** (the regression-slope shape, §5 Q1 option C): for the residual-misfit
observations, the ratios `Δy/Δx` between adjacent points on the misclassified boundary cluster near `2`, so
`2.0` enters the candidate grid (alongside other observed ratios). The grid is data-derived — no literal is
baked; on a task whose true slope were `0.5`, the ratios would cluster there instead.

**Enumeration (module `program_space`, `enumerate_programs(g, max_depth; max_num_depth = 2,
const_grid = _const_candidates(...))`).** Depth-1 leaves now include `ConstSlot(2.0)` (and the other
candidate constants) alongside `FeatureRef(:x_norm)`, `FeatureRef(:y_norm)`. Depth-2 combinations include
`Times(ConstSlot(2.0), FeatureRef(:x_norm))`; a depth-3 combination is
`Minus(Times(ConstSlot(2.0), FeatureRef(:x_norm)), FeatureRef(:y_norm))`. Phase-1 atoms threshold it:
`GTExpr(Minus(Times(ConstSlot(2.0), FeatureRef(:x_norm)), FeatureRef(:y_norm)), 0.3)` (`0.3` from the
observed-value threshold grid of that compound `NumExpr`, feature-arithmetic §5 Q4). Phase 2 emits
`P★ = IfExpr(that GTExpr, enemy, food)`.

**Complexity + prior (module `program_space`, via `enumerate_programs_as_measure`).** With the recommended
`num_complexity(::ConstSlot) = 1`:
`num_complexity(Minus(Times(ConstSlot, FeatureRef), FeatureRef)) = 1(minus) + [1(times) + 1(const) + 1(ref)] + 1(ref) = 5`,
so `expr_complexity(P★) = 1(if) + 5 + 1 + 1 = 8`. The axis-aligned staircase that approximates the tilted
half-plane costs far more (each tile a 3-symbol `And(GT,GT)`, joined by `Or`s — the same `2^{Δcomplexity}`
prior gulf the product example computed). The `log₂|const_grid|` "which candidate" bits are **not** in this
node cost (the SPEC §1.3 predictive route, §5 Q3): the finer the coefficient grid, the more the predictive
marginal likelihood must justify the specific `2.0` against the coarser alternatives — Bayesian Occam on the
grid, not a prior penalty.

**Condition (module `ontology`, `condition`).** On data from the true rule, `P★` both *fits* (it names the
tilted boundary exactly) and is *short*. A fixed-slope `x − y > t` misfits (wrong tilt); the staircase fits
but is dominated by the prior. **Result:** `expect(posterior, u_action)` (module `ontology`) selects the
correct action on the tilted half-plane the coefficient-free grammar provably could not name — a hypothesis
inexpressible before `ConstSlot`, findable after. Owner of each step: engine (`program_space` enumeration +
complexity; `ontology` condition/expect); the host supplies observations and executes actions — no host
arithmetic, and critically **no baked coefficient**: `2.0` is a data-derived candidate, priced by the same
machinery as every threshold.

## 5. Open design questions (the genuine forks for review)

### Q1 — candidate derivation for a *coefficient* position (THE load-bearing question)

Feature-arithmetic §5 Q3 named this genuinely harder than a threshold and left it open. A threshold sits at
a comparison side and has a natural grid (observed feature values, midpoints between them —
`_threshold_candidates`). A coefficient `a` in `a·x` multiplies a feature and has **no natural grid**: the
"right" `a` depends on the scale of `x` and of what it is compared against. Four honest derivations, each
developed with its shape and its failure mode:

- **(A) Observed values of the *other* operand (units-matched).** In `ConstSlot × FeatureRef(:x)`, propose
  the constant from the observed values of `:x` (or its reciprocals). *Shape:* a coefficient in the units of
  `1/x` roughly cancels `x`'s scale. *Failure:* it is a **type error dressed as a heuristic** — a coefficient
  is dimensionless-per-the-target, not in the operand's units; matching `:x`'s observed values gives
  constants near `x`'s magnitude, which is the wrong scale for a *multiplier*. Rejected as primary: it
  answers "what value does `x` take," not "what slope relates `x` to the target."

- **(B) Ratios of observed values (the regression-slope shape).** Propose coefficients as ratios `v_i / v_j`
  of observed feature values across the buffer — for the two-feature affine case, the slope that makes the
  boundary pass through the misclassified points is a ratio of coordinate differences (`Δy/Δx`). *Shape:*
  this is **the least-squares slope discretised** — the data-derived answer to "what multiplier relates
  these two features on the decision boundary." *Failure:* the ratio set is `O(n²)` in observed values
  (breadth), and ratios spanning many orders of magnitude need deduplication/binning to stay a finite,
  tight candidate set. But it is *derived from the data*, decision-free, and has the right dimensional
  character (a ratio of two feature values is exactly the slope units). **This is the recommendation** —
  see below.

- **(C) A log-spaced dimensionless grid (scale-free but decision-smelling).** A fixed geometric grid
  `{…, 1/4, 1/2, 1, 2, 4, …}` covering plausible coefficient magnitudes. *Shape:* scale-free, cheap, finite,
  deterministic. *The precedent question this option forces (and must engage head-on):* **is a fixed log
  grid a baked decision (`decision-free-combinator`-illegal) or a data-refinable seed (blessed like
  `THRESHOLDS`)?** The honest answer: a **fixed, data-independent** log grid is *closer to illegal* than
  `THRESHOLDS` is blessed. `THRESHOLDS` is blessed because (i) it is a **seed the explore path refines
  against the residual** (`explore_grammar` inserts observed-value midpoints — it does not stay fixed), and
  (ii) it is on the **predictive route** (SPEC §1.3), so a grid point only survives if the data's evidence
  favours it. A log coefficient grid that is *never refined from data* fails test (i) — it injects the
  modeller's guess about plausible coefficient magnitudes, which is exactly "a number the data should have
  determined." It is blessable **only** if paired with a data-refinement path (insert observed ratios,
  option B, into the log grid as the explore step) — at which point it is B with a log seed, not a
  standalone option. So: **C is not independently legal**; it is admissible only as a *seed* for B's
  refinement, mirroring how `THRESHOLDS` seeds `explore_grammar`'s midpoint refinement.

- **(D) Residual-derived candidates (the Move-3 midpoint mechanism transposed).** Compute, per candidate
  compound `NumExpr`, the coefficient value that would make the *current residual-misfit* observations flip
  to correct — the analogue of Move 3's "a threshold matters only where it crosses an observation," lifted
  to "a coefficient matters only where it changes a misclassification." *Shape:* the tightest, most
  directly-VOI-aligned set — it proposes exactly the coefficients that resolve the observed misfit. *Failure:*
  it is **circular for a general polynomial** — solving "what `a` flips this point" requires fixing the rest
  of the expression (the threshold, the other coefficients), so for a multi-coefficient expression the
  candidate for each slot depends on the others, and the joint derivation is a small optimisation, not an
  enumeration. Tractable for the **single-coefficient affine case** (`a·x + b`, one slot at a time),
  intractable-as-stated for the general multivariate polynomial.

**Recommendation: (B) ratios of observed values, seeded/bounded by a log grid (C-as-seed), with (D) as the
single-coefficient refinement.** Reasoning, strongest first:

1. **B has the correct dimensional character and is the discretised regression slope** — a coefficient
   relating two features *is* a ratio of their values, so B derives the candidate from the data in the right
   units. A is a units error; C alone injects the modeller's magnitude guess; B is the only option whose raw
   candidates are decision-free *and* dimensionally correct.
2. **B is a strict generalisation of the threshold mechanism, not a new machine.** `_threshold_candidates`
   already takes "midpoints between adjacent observed values"; B takes "ratios between observed values" — the
   same "candidates live at the data, priced by the predictive" contract, one operator up. This keeps the
   `decision-free-combinator` legitimacy argument **identical** to the threshold's (data-derived, complexity-
   priced, refined against the residual), which is the whole point of the precedent.
3. **The `O(n²)` breadth is controlled by the same VOI gate that controls arithmetic depth** (§6): the
   `ConstSlot`-enable meta-action pays the ratio-grid cost only when the residual warrants a coefficient, and
   a log-scale dedup/bin (C-as-seed) bounds the grid to a tight, deterministic set. D refines it further for
   the single-coefficient case.

*The counter to weigh:* B's `O(n²)` ratio set is genuinely broader than the threshold's `O(n)` midpoint set,
and for the multivariate case the "which ratios are slopes vs. noise" question is real. A reviewer who wants
the **narrowest defensible first move** could ratify **D-restricted-to-single-coefficient-affine** as the
initial `ConstSlot`, with B (general polynomial coefficients) as its own fast-follow — trading expressiveness
for a tighter, provably-terminating candidate set. That is the sharpest fork in this doc.

### Q2 — where `ConstSlot` may appear (any position, or restricted?)

A `ConstSlot` in a **comparison-side scalar** position (`x > ConstSlot(c)`) buys **nothing new** — that is
exactly what the existing threshold already does (`GTExpr(FeatureRef(:x), c)` with `c` from the grid). The
new expressiveness is *only* inside arithmetic: `ConstSlot × FeatureRef` (a coefficient), `ConstSlot + …`
(an intercept — though an intercept folds into the threshold: `a·x + b > t ≡ a·x > t − b`, so a `Plus`-side
constant is redundant with the threshold and should likely be forbidden to avoid a degenerate degree of
freedom). *Recommendation: restrict `ConstSlot` to **multiplicative (coefficient) positions*** — inside
`Times` (and, if poles are wanted, `Div`/`AQ` denominators). Forbid it as a bare comparison side (the
threshold covers it) and as a `Plus`/`Minus` operand at the top level (the threshold absorbs the intercept).
*Breadth cost per position:* each admitted position multiplies the enumeration by `|const_grid|`; restricting
to coefficient positions is both the expressiveness win *and* the breadth-minimising choice — they coincide,
which is why the recommendation is confident. *Counter:* a nested intercept (`a·x + b·y` where neither folds
into the single threshold) is genuinely new for **two or more** coefficient terms — so "forbid `Plus`-side
constants" is right for the *top-level* intercept but the multi-term case needs `Plus` of two
`Times(ConstSlot, FeatureRef)` terms, which the multiplicative-position rule already permits. The rule is
therefore "`ConstSlot` only as a `Times`/`Div`/`AQ` operand," not "only in `Times`."

### Q3 — complexity accounting: what does a `ConstSlot` cost, and how does it interact with `num_complexity`?

Two sub-questions. **(a) The node cost.** Options: `num_complexity(::ConstSlot) = 1` (one symbol, grid-choice
bits on the predictive) vs `= 1 + log₂|const_grid|` (charge the "which candidate" bits in the prior).
*Recommendation: `= 1`, the predictive route*, for exact parity with the threshold: `expr_complexity(GTExpr)`
under-counts the `log₂ n` threshold-choice term deliberately (SPEC §1.3 margin — "an under-count … exact and
harmless while the grid is fixed … the predictive marginal likelihood charges it"), and a `ConstSlot` grid is
the *same shape of object* as a threshold grid, so it takes the *same route*. Charging `log₂|grid|` in the
prior for the constant but not for the threshold would be an inconsistency the SPEC margin explicitly warns
against ("a finer partition is not free … does not need, and must not introduce, a separate fineness axis").
**The condition for legality is the same as the threshold's:** the `ConstSlot` candidates must be scored on
the **predictive marginal likelihood** (Bayesian Occam integrates the parameter — a finer grid wins only if
its evidence beats the coarser grid's), *not* a point/max-likelihood score (which has no Occam and would
chase unbounded grid refinement, at which point the SPEC says the prior route becomes mandatory). §6 pins
this: the `ConstSlot` enumeration must feed the marginal-likelihood-scored posterior, never a point estimate.
**(b) Interaction with `max_num_depth`.** A `ConstSlot` is a depth-1 leaf, so `ConstSlot × FeatureRef` is
depth-2 — the same depth budget that gates products gates coefficients. No new depth axis; the coefficient
rides the existing `max_num_depth` ladder. This is clean and is not really open — surfaced only to confirm
the reviewer agrees a constant is depth-1 (a literal is atomic), not depth-0 or a special case.

### Q4 — the all-constant-subtree degeneracy (a correctness guard, mildly open)

`Times(ConstSlot(2), ConstSlot(3))` is the constant `6`; `Neg(ConstSlot(2))` is `−2`; any arithmetic tree
whose leaves are *all* `ConstSlot`s is a single fitted number that should be one `ConstSlot`, not a compound.
If not guarded, the enumeration emits redundant, higher-complexity spellings of the same constant (wasted
breadth, and a mild prior distortion — the constant `6` appears both as `ConstSlot(6)` if `6` is a candidate
and as `Times(ConstSlot(2),ConstSlot(3))` at higher cost). *Recommendation: forbid any combinator all of
whose arguments are constant-valued* — a structural predicate `all_const(e)` checked in `_num_exprs_by_depth`
that drops the combination (mirroring the existing `x−x`/`x/x` degeneracy guards, `enumeration.jl:114-125`).
*Open sub-point:* whether to *drop* such combinations or *collapse* them to a single `ConstSlot` of the
computed value. Dropping is simpler and loses nothing (the value, if useful, is already a candidate in the
grid); collapsing risks introducing a constant *not* in the data-derived grid (a `decision-free-combinator`
smell — a value the data did not propose). *Recommendation: drop, never collapse* — collapsing manufactures a
candidate outside the data-derived set, which is exactly the injection the precedent forbids. This is the
subtle place the back door could reopen, so it is called out as its own question rather than buried in §2.

### Q5 — the Stone–Weierstrass claim: precise statement and honest limits

This move *earns* the density claim the coefficient-free move reserved (feature-arithmetic §1). It must be
stated **precisely** and **limited honestly**, or it over-claims exactly as §1 warned against.

*Precise statement.* The set `{+, ×, const}` over the feature terminals generates the **polynomial algebra**
in those features. Stone–Weierstrass: the polynomials are **dense in the continuous functions** on a compact
feature domain (uniform norm) — for any continuous `f` and any `ε > 0`, there is a polynomial `p` with
`sup |f − p| < ε`. Applied to predicates: any continuous decision boundary (the level set of a continuous
`f`) is **approximable to arbitrary accuracy** by a polynomial super-level set `{p > t}` — which is exactly a
`GTExpr(<polynomial NumExpr>, t)`. So the basis, *with constants*, can name any continuous predicate boundary
to arbitrary precision; without constants it cannot (the coefficient-free move's honest ceiling).

*Practical meaning.* Any continuous predicate boundary is approximable at **sufficient depth and grid** — the
expressiveness ceiling of the arc rises from "multiplicative interactions" to "arbitrary continuous
boundaries."

*Honest limits (mandatory — the claim is dense, not omnipotent):*
1. **Density ≠ learnability.** Stone–Weierstrass guarantees a polynomial *exists*; it does not guarantee the
   **enumerator finds it within a budget**. The approximating polynomial may need high degree (deep
   `max_num_depth`) and a fine coefficient grid — both breadth the VOI cascade pays for only on demand (§6).
   The theorem is about the basis's reach, the enumeration is about the search's reach; they are different,
   and the doc must not let the first imply the second.
2. **Compactness assumption.** Stone–Weierstrass needs a **compact** domain; the feature domain is compact
   only if features are bounded (grid_world's `[0,1]`-normalised features are, but the claim is scoped to
   bounded features — unbounded features (a raw `count`) need care).
3. **Approximation, not exactness, for non-polynomial boundaries.** A genuinely non-polynomial boundary (a
   `sin`, a hard threshold) is *approached*, not *named exactly*, and the approximation cost (degree) can be
   high. The move buys density, not a finite exact basis for all continuous predicates.
4. **This is a claim about the *predicate basis*, not the whole agent.** It says the hypothesis space can
   *express* the boundary; whether the agent *acquires* it depends on data, prior, and the VOI-scored depth
   escalation — the standard density-vs-realisability gap.

*Recommendation:* state the claim in the code PR's paper-facing note (and the arc status) with all four
limits attached, and add the **density-demonstration task** (§6) as the empirical witness that the basis
crosses the threshold the coefficient-free move could not. The claim is a genuine earn; the limits are what
keep it from being the over-claim §1 forbade.

## 6. Risk + mitigation

1. **The back door (the legitimacy surface — this move IS the `decision-free-combinator` first application).**
   *Failure mode:* a `ConstSlot` whose value is a literal (or a fixed non-refined grid, Q1 option C-alone, or
   a collapsed all-const subtree, Q4) injects an **answer**, not a prior — the precise object the precedent
   forbids. *Blast radius:* the complexity prior's neutrality (SPEC §1.3) — a baked coefficient launders a
   decision through the prior's support, corrupting every posterior that ranges over it. *Mitigation, in
   order:* (a) the candidate set is a **function of the observation buffer** (`_const_candidates`, Q1 option
   B), mechanically data-derived exactly as `_threshold_candidates` is — the legitimacy argument is
   *identical* to the threshold's, by construction; (b) **the structural no-literal test** (feature-arithmetic
   §6 risk 2's "no `NumExpr` primitive definition contains a literal cutoff") **extends** to assert no
   `ConstSlot` is constructed with a compile-time-constant value in `src/` — the only `ConstSlot(...)` call
   sites are in `_const_candidates` (data-derived) and tests; (c) Q4's **drop-never-collapse** rule keeps the
   grid the sole source of constant values; (d) the precedent's review checklist (already ratified) is the
   human backstop. No pragma — the precedent is documentation-only + the structural test.

2. **Breadth: the `grid × depth` multiplication goes another rung up.** *Failure mode:* the coefficient grid
   multiplies an already-broad `O((features·ops)^d)` enumeration by `|const_grid|` per admitted position — and
   Q1 option B's ratio grid is `O(n²)` in observed values. *Blast radius:* `enumerate_programs` wall-clock and
   meta-level lookahead cost — the feature-arithmetic move's named breadth risk, compounded. *Mitigation:*
   (a) the **`ConstSlot`-enable meta-action** (§1) is the resource-rational control — the grid is empty at the
   floor and turns on only when the residual cannot be explained coefficient-free (breadth on demand, VOI-
   scored, exactly as `explore_grammar`/escalate-arithmetic-depth are); (b) Q2's **restriction to
   coefficient positions** minimises admitted positions; (c) Q1's log-scale dedup/bin bounds the ratio grid to
   a tight deterministic set. **If a run hangs or exact enumeration is too slow, STOP and report — no silent
   approximation** (master-plan performance clause; `feedback_no_silent_approximations`). The breadth-wall
   benchmark (feature-arithmetic §6.1) measures where this sits on the `(basis-richness, depth, grid)`
   frontier.

3. **Point-estimate scoring would break the fineness-Occam (the SPEC §1.3 trap).** *Failure mode:* if the
   `ConstSlot` grid is scored by a **max/point** likelihood rather than the **predictive marginal**, there is
   no Occam on grid fineness (SPEC §1.3: "a point-estimate … has no Occam and would chase unbounded
   refinement") — the enumerator would prefer ever-finer coefficient grids that merely fit noise. *Blast
   radius:* the whole legitimacy of the predictive route (§5 Q3) — the reason `num_complexity(::ConstSlot)=1`
   is sound. *Mitigation:* the `ConstSlot` posterior is conditioned through the **existing marginal-likelihood
   path** (the same `condition`/`enumerate_programs_as_measure` the threshold grid rides); a test asserts a
   noise-only finer grid does **not** win (its predictive does not beat the coarser grid's) — the empirical
   witness that the predictive route holds for constants as it does for thresholds. If the SPEC §1.3 predictive
   route cannot be honoured for a particular derivation, the prior route (`num_complexity = 1 + log₂|grid|`)
   becomes **mandatory** (§5 Q3(a)) — named, so the choice is a decision, not a drift.

4. **The density claim over-reaches into learnability.** *Failure mode:* the paper-facing note reads
   "Stone–Weierstrass ⇒ the agent learns any continuous boundary," collapsing density into realisability —
   the exact over-claim feature-arithmetic §1 warned against, one move later. *Blast radius:* the arc's
   headline honesty (`feedback_achieved_value_not_a_cap` — never state a reachable-in-principle as a
   guaranteed-in-practice). *Mitigation:* §5 Q5's four honest limits ship *with* the claim; the density-
   demonstration task witnesses expressibility-crossing, not learnability, and is captioned as such.

5. **Behaviour drift from the new leaf.** *Failure mode:* `ConstSlot` enters an enumeration path and shifts a
   pre-existing program's complexity/order. *Blast radius:* the floor-equivalence pin (§3). *Pre-emptive grep:*
   `grep -rn 'NumExpr\|_num_exprs_by_depth\|num_complexity\|compile_num' src/ test/` — each site dispositioned
   for the new `ConstSlot` method (add-a-method, not modify-existing; the `num_feature_refs!` empty-body method
   is the one that would fail loud if omitted). *Backstop:* the `const_grid = []` floor pin halts the line on
   any drift in the coefficient-free enumeration.

## 7. Verification cadence

```
julia test/test_constslot.jl            # floor equivalence (== at const_grid=[]), complexity, affine discovery,
                                        #   no-literal structural test, predictive-route (noise grid doesn't win),
                                        #   density demonstration, determinism
julia test/test_feature_arithmetic.jl   # coefficient-free enumeration bit-stable after the ConstSlot leaf lands
julia test/test_program_space.jl        # base enumeration unchanged
julia test/test_threshold_explore.jl    # Move 3 threshold refinement untouched
julia test/test_feature_discovery.jl    # Move 4 selection untouched
julia test/test_grid_world_meta.jl      # host meta-loop unchanged (ConstSlot off at the floor)
```

**Capture pins.** A `test/fixtures/constslot_floor_v1.tsv` pins the coefficient-free enumeration's
(count, `show_expr`, complexity, log-prior) PRE-`ConstSlot` (captured at the predecessor SHA per
`test/fixtures/README.md`), asserted `==` post-change at `const_grid = []`, mirroring
`feature_arithmetic_lift_v1.tsv`. The **no-literal structural test** and the **density-demonstration task**
(the affine `2·x − y > 0.3` of §4, unreachable coefficient-free, acquired once the coefficient grid is on)
are the two named §6 witnesses.

Full `test/test_*.jl` green before commit; lint self-test + `check apps/` (the `decision-free-combinator`
slug already in the index — this move is its first *code* application, no new slug). Skin smoke **optional**
— no wire verb added (`ConstSlot`, like arithmetic depth, is a brain-decided meta-action, VOI-scored into the
EU-max loop, not a host toggle; it touches no serialised path). Halt-the-line on any drift in the
coefficient-free floor enumeration or any full-suite failure — the branch never sleeps red.
