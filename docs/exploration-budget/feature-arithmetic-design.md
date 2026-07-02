# Feature-arithmetic design — the numeric sublayer (`GT`/`LT` over expressions, `× + − ÷ neg`)

> Exploration-budget arc, post-Move-5 thematic move (the next rung of the §3.1 escalation ladder).
> Design-doc-before-code; ratify before any code lands. Master plan:
> `docs/exploration-budget/master-plan.md` (§3.1 the selection/generation seam; the fine-before-coarse
> escalation `refine thresholds → add features → construct arithmetic-derived features → (far future)
> continuous features`; SPEC §1.3 the complexity prior). Discharges the frontier **Move 4 named and
> parked** (`move-4-design.md §8.1`, §5 Q1: *"arithmetic combinations — `red × blue` — … the deferred
> §3.1 floor,"* "Move 4b or later," requiring *"a feature-arithmetic AST extension and a brain proposer"*).
> Predecessors on master: Moves 1–5, the #174 compression-removal follow-up. Authored 2026-07-01.

---

## 1. Purpose

Grow the agent's **predicate** alphabet one rung above feature selection: let a comparison threshold a
**real-valued expression over features** (`(A×B) > t`, `(A−B) > t`, `(A/B) > t`), not only a raw feature
(`A > t`). This is the next rung of the master-plan §3.1 escalation ladder, and it discharges the exact
frontier Move 4 parked.

**The gap is structural, and Move 4 located it precisely.** Today `GTExpr(feature::Symbol, threshold)`
thresholds a *raw named feature* (`types.jl:15-23`), so the grammar expresses conjunctive combinations
(`And(GT,GT)`) but **not** arithmetic ones (`A×B`). Move 4 §8.1 split §3.1's "combinations of existing
features" on exactly this line — conjunctive/disjunctive combinations are *already grammar-expressible*
(the program-space's job); arithmetic products are *"the only genuinely new dimension."* The reason they
were deferred, restated in the recon that opened this move: `GT`/`LT` take a *feature* where they should
take a *real-valued expression over features*, and `+`/`×` aren't in the alphabet.

**The dissolution of the "creative floor" (the head of this move).** Move 4 filed products under "the
creative floor requiring a brain proposer," which conflated two separable things:

- the **alphabet gap** — `×`/`÷` aren't in the basis, so a product is *inexpressible at any depth*; and
- the **breadth problem** — *which* products to try, once they are expressible.

The arithmetic head fixes the alphabet gap only. With `×`/`÷` in the alphabet a product is **priced,
enumerable compositional structure** the complexity prior scores exactly like any other subtree — *no
proposer, no creative floor.* What Move 4 called "a brain proposer" is the breadth-reducer (a recognition
model): efficiency, not correctness; the **measure-first** successor (§6.1), needed *because* arithmetic
raises breadth (`breadth^depth`), not needed for the mechanism to be sound. Enumeration under the prior is
correct-but-broad; a proposer would make it fast without touching the posterior. So this move is the
settled completion of a ratified deferral, with a sharp before/after (the comparison operators' argument
slot generalises from feature-ref to numeric expression; `+ × − ÷ neg` enter; everything above stays the
brain's), and the residual creative floor shrinks — honestly — to *genuinely novel sensory dimensions*
(features not expressible as a rational function of existing terminals), not eliminated.

**Theory-claim scoping (attribution honesty — a move does not borrow a later move's theorem).** The
**coefficient-free** first move (this doc's code PR) is justified by **multiplicative-interaction
compactness alone**: `A×B > t` is one 3-symbol predicate the complexity prior favours over the staircase
of `And(GT,GT)` tiles that approximates the same hyperbola (§4). It is **not** Stone–Weierstrass-complete —
that theorem needs the constants (`{+,×,const}` are dense; `{+,×}` over bare feature-refs is not). The
density / approximation-completeness claim is reserved for when the learnable-constant slot lands (§5 Q3).
The const-free move is fully justified on compactness grounds and must not over-claim.

**What unblocks.** Products, differences, and ratios of existing features become EU-max-enumerable
hypotheses, priced by the existing complexity prior and conditioned by the existing `condition` — the
capstone the fine-before-coarse ladder was climbing toward. It also makes the search **compact-but-broad**,
which turns the report's tractability question live for Credence specifically — named as the immediate
empirical successor (§6.1), not built here.

**Depth is earned, not toggled (the arc's autonomy thesis).** `max_num_depth` (§2) is the enumeration
lever, but its *value is a meta-decision, not a host setting*: raising arithmetic depth is a VOI-scored
exploration op — a generalisation of `:gw_deepen`, scored against the belief's residual exactly as
threshold-refinement (`explore_grammar`) and feature-addition (`explore_features`) are. The static default
`max_num_depth = 1` is only the behaviour-preserving *floor*; an **escalate-arithmetic-depth meta-action**
raises it on demand — when the residual cannot be explained without a product. That meta-action is a
**named successor** (§6.1): the first code PR ships expressibility at the floor, but the doc frames depth as
brain-earned from the start, because a host-toggled depth would make this the one rung of the arc where a
discovery is *configured* rather than *earned*. It is also the resource-rational answer to breadth (§6.1):
dynamic escalation pays the `O((features·ops)^d)` cost only when a product is warranted, where a static high
cap pays it on every enumeration.

## 2. Files touched (the follow-up code PR — coefficient-free first move)

**`src/program_space/types.jl`** — modification. A **numeric sublayer**, a *separate* abstract type
(numeric-valued, distinct from the boolean/action `ProgramExpr` layer — Invariant 3, single-responsibility
representations; recommended over `NumExpr <: ProgramExpr`, which would let a numeric node type-check where
a predicate is expected):

```julia
abstract type NumExpr end
struct FeatureRef <: NumExpr; feature::Symbol; end            # the reified raw-feature read
struct Times <: NumExpr; left::NumExpr; right::NumExpr; end   # ×
struct Plus  <: NumExpr; left::NumExpr; right::NumExpr; end   # +
struct Minus <: NumExpr; left::NumExpr; right::NumExpr; end   # −  (gives A>B via (A−B)>0)
struct Div   <: NumExpr; left::NumExpr; right::NumExpr; end   # ÷  (total; operator per §5 Q2)
struct Neg   <: NumExpr; child::NumExpr; end                  # unary negation
```

Generalise the comparison slot (`types.jl:15-23`): `GTExpr`/`LTExpr` carry `lhs::NumExpr` instead of
`feature::Symbol`. Today's atom becomes `GTExpr(FeatureRef(feat), t)` — a pure lift. `ConstSlot` (the
learnable numeric constant) is **designed** in §5 Q3 and deferred to a fast-follow; the first move is
coefficient-free.

**`src/program_space/enumeration.jl`** — modification.
- `num_complexity(::NumExpr)`: `FeatureRef = 1`; `Times/Plus/Minus/Div = 1 + left + right`; `Neg = 1 +
  child`. And `expr_complexity(e::GTExpr) = num_complexity(e.lhs)` (was `= 1`, `enumeration.jl:12-13`),
  `LTExpr` likewise; `_expanded` mirrored (`enumeration.jl:29-30`). See §3 — this keeps a bare-feature
  atom at cost 1.
- `enumerate_programs` (`enumeration.jl:63-145`): a new `max_num_depth::Int = 1` kwarg. Phase-1 atom
  generation (`:72-81`) builds a depth-bounded `NumExpr` set (depth 1 = `FeatureRef(f)` for each feature;
  depth d = the arithmetic combinators over depth-`<d` `NumExpr`s), then forms `GTExpr(nexpr, t)` /
  `LTExpr(nexpr, t)` over `nexpr × g.thresholds`. `max_num_depth = 1` ⇒ only `FeatureRef` ⇒ the atom set
  is the lifted image of today's (§3) — the **behaviour-preserving floor**. Higher depth is *not* a host
  toggle: the escalate-arithmetic-depth meta-action (§1; named successor, §6.1) raises it by residual VOI,
  exactly as `explore_grammar` refines thresholds. The first code PR ships enumeration parameterised by
  `max_num_depth` at the floor; the meta-action that drives it is the immediate successor.

**`src/program_space/compilation.jl`** — modification. `compile_num(::NumExpr) → (features, ts) ->
Float64` (six methods); `compile_expr(e::GTExpr)` (`compilation.jl:25-29`) becomes
`lhs_fn = compile_num(e.lhs); (f, ts) -> lhs_fn(f, ts) > e.threshold`. `compile_num(FeatureRef)` is the
current feature read `(f, _ts) -> get(f, feat, 0.0)`, so a lifted bare-feature atom compiles to a
bit-identical closure.

**`src/program_space/exploration.jl`** — modification. Threshold candidacy (`_threshold_candidates`,
~`:56-82`) generalises from per-feature observed values to **observed values of the `NumExpr`**: evaluate
the candidate `NumExpr` over the buffer via `compile_num`, take the residual-mass midpoints exactly as
Move 3 does. `explore_grammar` (~`:273-322`) is otherwise unchanged — it refines the grid of whatever
`NumExpr`s the grammar carries. §5 Q4 owns how a per-`NumExpr` grid attaches.

**`src/program_space/perturbation.jl`** — the AST walks that reach into predicates gain `NumExpr`
awareness where they must: `extract_subtrees` / `expr_equal` (`perturbation.jl:~65-114,471-504`) treat a
`GTExpr`'s `lhs` as an opaque leaf for now (arithmetic subtrees are not yet abstracted — a named
limitation, §6). A future `collect_feature_refs!` (deferred with `:remove_feature`, #174) must recurse
into `NumExpr` to collect `FeatureRef.feature` — noted, not built here.

**`src/Credence.jl`** — export the `NumExpr` hierarchy (`FeatureRef, Times, Plus, Minus, Div, Neg`) and
`compile_num`; add the `max_num_depth` kwarg to the `enumerate_programs` surface (`Credence.jl:75-82`).

**New test** `test/test_feature_arithmetic.jl` —
- the behaviour-preserving lift: `enumerate_programs(g, d; max_num_depth = 1)` reproduces the pre-change
  enumeration's count, per-program `show_expr`, complexity, and prior log-weights `==` (§3);
- complexity accounting: `expr_complexity(GTExpr(Times(FeatureRef(:a), FeatureRef(:b)), t)) == 3`; a bare
  `GTExpr(FeatureRef(:a), t)` still `== 1`;
- product discovery: a task whose true rule is `(A×B) > t`, unreachable by the colour-grammar staircase
  within the depth cap, is acquired once `max_num_depth ≥ 2` (§4);
- determinism: reproducible enumeration order under the sorted-feature / fixed-grid convention.

## 3. Behaviour preserved (the capture-before-refactor spine)

The lift is **semantically neutral by construction**. `expr_complexity(GTExpr) = num_complexity(lhs)` with
`num_complexity(FeatureRef) = 1` keeps a bare-feature threshold at **cost 1** (the threshold- and
comparison-free convention of Move 3 Q1(b) — a threshold adds no symbol; the comparison is bundled), so
every existing program keeps an identical complexity, identical prior log-weight, identical posterior after
conditioning, and a `compile_num(FeatureRef)`-identical closure. Only new arithmetic programs (present only
when `max_num_depth ≥ 2`) appear.

**The pin is on the observable projection, not struct identity.** The AST *representation* changed
(`GTExpr` now wraps a `FeatureRef` where it held a bare `Symbol`) — that reshape is the point (Invariant 3:
the AST is the structural-analysis representation; its shape is not a behaviour). So capture-before-refactor
pins the *semantic* projection PRE-change and asserts `==` post-change at `max_num_depth = 1`:

- program **count** and per-program `show_expr` **canonical string** (`(gt :a 0.5)` renders identically —
  `show_expr(GTExpr)` prints `(gt <show_num(lhs)> t)` with `show_num(FeatureRef(:a)) = ":a"`);
- per-program **complexity** and **prior log-weight** (`enumerate_programs_as_measure`);
- **posterior** weights after a fixed conditioning sequence.

Existing test construction sites that build `GTExpr(:a, t)` (a bare `Symbol`) get the mechanical lift to
`GTExpr(FeatureRef(:a), t)` (the §6 grep); their *semantic* assertions stay `==`.
`test_program_space`, `test_threshold_explore`, `test_feature_discovery`, `test_grid_world_meta` green
after that lift. Strata tolerances per template: unit `isapprox(atol = 1e-14)`; end-to-end
`isapprox(rtol = 1e-10)` — halt-the-line at greater drift.

## 4. Worked end-to-end example

**Task.** A grid_world grammar `g` over `{:x_norm, :y_norm}` (host extracts both every step). The true rule
is *"enemy iff `x_norm × y_norm > 0.25`"* — a far-corner region (the super-level set of a hyperbola).
Axis-aligned thresholds cannot name it: any single `And((gt :x_norm a), (gt :y_norm b))` tile is a
rectangle; covering the hyperbola needs a disjunction of many tiles.

**Enumeration (module `program_space`, `enumerate_programs(g, max_depth; max_num_depth = 2)`).** Phase 1
builds the depth-2 `NumExpr` set, which now contains `Times(FeatureRef(:x_norm), FeatureRef(:y_norm))`;
Phase-1 atoms include `GTExpr(Times(FeatureRef(:x_norm), FeatureRef(:y_norm)), 0.25)` (threshold from the
observed products' residual-midpoints, §5 Q4); Phase 2 emits
`P★ = IfExpr(GTExpr(Times(FeatureRef(:x_norm), FeatureRef(:y_norm)), 0.25), enemy, food)`.

**Complexity + prior (module `program_space`, `complexity.jl` via `enumerate_programs_as_measure`).**
`expr_complexity(P★) = 1(if) + num_complexity(Times(FeatureRef,FeatureRef)) + 1 + 1 = 1 + 3 + 1 + 1 = 6`.
The 5-tile staircase `IfExpr(Or(And(GT,GT), Or(And(GT,GT), …)), enemy, food)` costs
`1(if) + (5·3 + 4)(pred: five `And(GT,GT)` tiles + four `Or`s) + 1 + 1 = 22`. Log-prior `−complexity·log 2`:
the product outscores the staircase by `(22 − 6)·log 2 ≈ 11.1` nats — a prior ratio `2^16 ≈ 65 000`. And the
staircase (predicate depth ≈ 6+) sits *past a modest `max_depth` cap*, so it may not be enumerated at all,
while `P★` is program-depth 2.

**Condition (module `ontology`, `condition`).** On data drawn from the true rule, `P★` both *fits* (its
likelihood is high — it names the region) and is *short* (its prior is high). The staircase, even where
enumerated and fitting, carries the `2^16` prior penalty and is dominated; where it is past the cap, `P★`
is the only compact fit. **Result:** `expect(posterior, u_action)` (module `ontology`) selects the correct
action in the corner the colour/axis-aligned grammar provably could not name — a hypothesis that was
*inexpressible at any depth* before, and *findable at depth 2* after. The findability win is the prior
moving the true model from "long staircase past the cap" to "3-symbol predicate near the root."

Owner of each step: engine (`program_space` enumeration + complexity; `ontology` condition/expect); the
host only supplies observations and executes the action — no host arithmetic.

## 5. Open design questions (the genuine forks for review)

### Q1 — the rung boundary (stop at arithmetic; fold is a separate later rung)
Generalise the comparison slot and add `+ × − ÷ neg`; **bounded aggregation (fold) is not smuggled in**,
and unbounded recursion is host-gated and out of scope. The temporal operators (`Persists/Changed/Since`)
are *already* a domain-specialised fold-over-time (conjunction over a bounded window, `compilation.jl:60-97`)
— evidence the fold rung is real but **separate**: a general parameterised fold over finite structures is
the next rung, not this one. *Recommendation: stop at arithmetic.* The failure mode of a basis-enrichment
move is scope-creep up the expressiveness ladder; the report gives the exact rung to stop at, and
multiplicative-interaction compactness is the whole win this move needs to claim.

### Q2 — division semantics: protected division vs analytic quotient (load-bearing — it decides what the basis can *say*)
Naive protected division (`x/0 = 0.0`) introduces a **discontinuity** at the denominator's zero, and the GP
literature's long experience is that evolved/enumerated programs **exploit the `x/0` cliff** as spurious
short structure — the exact failure the complexity prior is meant to prevent. The fix is the **analytic
quotient** `AQ(x, y) = x / √(1 + y²)` (Ni, Drieberg & Rockett, 2013): total, smooth, artifact-free, a
documented generalisation improvement over protected division. But AQ **has no poles**, so it cannot
compactly express genuine asymptotic structure (an inverse-square law; a ratio that truly blows up as its
denominator vanishes). The fork is **domain-keyed**: true (protected) division buys pole/asymptote
compactness at the cost of the artifact; AQ buys smoothness at the cost of poles. **Resolved (2026-07-01): the two-operator answer
is operative, not AQ-only.** Some target features carry genuine poles — a rate is `count / time`, an
inverse-distance collision term is `1 / wall_dist`, both blowing up as the denominator vanishes, and both
exactly the kind of governance/spatial feature this basis is for. So: **AQ is the default** for a general
ratio with no expected pole (artifact-free smoothness where you have no reason to want a pole); **true
(protected) division is exposed** where a pole is real, with the `x/0` artifact guard documented. The
complexity prior then selects, per hypothesis, whichever operator is shorter for the data.

### Q3 — the learnable-constant slot: mechanism + phasing (the load-bearing constraint)
`ConstSlot` is where a baked threshold could re-enter through the back door. *Recommendation:* a constant
is **mechanically a threshold** — its value ranges over **data-derived candidates**, enumerated,
complexity-priced (`log₂ n` bits per grid choice, SPEC §1.3 margin), i.e. *"data-fit by the same machinery
that fits every other parameter,"* never a literal in a primitive. Open sub-question: the candidate
derivation for a *coefficient* position (observed values? ratios of observed values? a log-spaced grid?) is
genuinely harder than a threshold, which sits at observed feature values. *Phasing:* the first code move
ships **coefficient-free** combinators (`× + − ÷ neg` over `FeatureRef` + the existing data-derived
thresholds — the full products/ratios/differences compactness win, pure enumeration, zero
continuous-parameter machinery); `ConstSlot` is a fast-follow once its candidate derivation ratifies, and
it is the move that earns the Stone–Weierstrass density claim (§1). The legitimacy rule (below) is codified
*now* regardless, governing `ConstSlot` when it lands.

### Q4 — threshold attachment to compound `NumExpr`s
A per-`NumExpr` grid computed from the observed values of that expression (generalising
`g.thresholds[feat]`, `types.jl:137-158`), vs a single shared grid. *Recommendation: per-`NumExpr`
observed-value grid* (the Move-3 residual-midpoint mechanism, evaluated on the compiled `NumExpr`) — keeps
compound thresholds learnable and data-derived, and inherits Move 3's complexity-invariance (the grid is
not charged; the fineness-Occam rides the marginal likelihood).

> **ADDENDUM (2026-07-02, first code PR — flagged for ratification in that PR's review).**
> Implementation surfaced a coupling this section and §2's exploration.jl item under-analysed:
> refining a per-`NumExpr` grid requires knowing *which* compound expressions are in play, and
> that set is depth-escalation state — `max_num_depth` is an enumeration kwarg, not grammar
> state, so `explore_grammar` (whose signature carries no numeric depth) cannot yet name the
> compound candidates to refine. The first code PR therefore ships the floor: compound
> expressions threshold over the seed grid (`_num_threshold_grid`, enumeration.jl) and are
> UNREFINABLE by Move 3's Symbol-keyed mechanism — a NAMED, TESTED limitation
> (`test_feature_arithmetic.jl` §3 pins it consciously), not a silent one. Q4's ratified
> per-`NumExpr` observed-value grid moves to the **escalate-arithmetic-depth design** (which
> owns depth residency and therefore the compound-candidate set); the grid-key/storage
> question (`Grammar.thresholds` is `Symbol`-keyed; a `NumExpr`-keyed extension touches the
> persistence schema and its commit-pinned fixtures) is an open question of that doc. §4's
> worked-example threshold `0.25` is reachable only after that lands; at the floor, the
> example's product is enumerated at the seed grid (`0.3` separates the example's data).

### Q5 — home: exploration-budget move vs new arc
*Recommendation: a thematic doc in the exploration-budget arc* (this doc). It is the next rung of that
arc's own §3.1 escalation ladder and discharges the arc's own parked frontier; the recon confirmed the
paused **dominance benchmark assumes a fixed basis**, so enriching reopens that gate — but that is
intrinsic to the rung, not a reason to split off a new arc. Sequencing is handled by running dominance
first (§6.1 / master-plan sequencing).

## 6. Risk + mitigation

1. **Breadth strain — the tractability question goes live for Credence specifically (the named successor).**
   Today search is compact-but-*shallow*; arithmetic makes it compact-but-*broad* — the depth-`d` `NumExpr`
   count is `O((features · ops)^d)`, multiplied by the threshold grid, and that feeds Phase-1 atoms. This is
   where prior+VOI enumeration begins to strain (the report's `breadth^depth`; DreamCoder's evidence).
   *Blast radius:* `enumerate_programs` wall-clock and the meta-level lookahead cost.
   *Mitigation + hand-off — two complementary successors, both named:*
   - **The escalate-arithmetic-depth meta-action is the resource-rational breadth control** (§1) — it *chooses*
     the depth, paying the `O((features·ops)^d)` cost only when the residual warrants a product (breadth on
     demand), where the static `max_num_depth` cap pays it on *every* enumeration. The static default 1 is only
     the behaviour floor / worst-case bound; dynamic escalation is the actual control. This is a named successor,
     not built in the first code PR.
   - **The breadth-wall benchmark** *measures* tractability *at* a depth — where enumeration sits on the
     `(basis-richness, depth)` frontier; that is where the recognition-model proposal re-enters **on evidence,
     not faith**. Named, not built.

   The two are complementary: one chooses the depth, the other measures the cost of a chosen depth. If a run
   hangs or exact enumeration is too slow, STOP and report (no silent approximation) — per the master plan's
   performance clause.
2. **The const-slot back door (the legitimacy surface).** A baked coefficient would inject an *answer*, not
   a prior. *Mitigation:* the **constitutional companion** — a new precedent slug, candidate
   `decision-free-combinator`, added to CLAUDE.md's slug index + `docs/precedents.md` **in the code PR**:
   *primitives added to the open-vocabulary roster must be decision-free combinators (total,
   domain-independent, parameter-free or free-parameter-slot-with-data-fit); a primitive that bakes a
   numeric threshold or a decision the data should determine is illegitimate — it injects an answer, not a
   prior. Learnable constants are data-fit by the same machinery as thresholds, never literals. First
   application: `ConstSlot` (Invariants 1+2; SPEC §1.3).* The doc **proposes** it (honouring master-plan §6
   *"no new constitutional text — stop and report"*); the principle is ratified in conversation, the slug
   lands with the code PR that first needs it. It also retroactively blesses the existing `THRESHOLDS` seed
   grid as decision-free. A test asserts no `NumExpr` primitive definition contains a literal cutoff.
3. **Division totality (Koza closure).** The chosen operator must be total so
   enumeration/compilation/scoring stay well-defined — resolved in §5 Q2 (protected vs analytic quotient);
   once chosen, the semantics are fixed in `compile_num(Div, …)`.
4. **Behaviour drift from the signature change.** `GTExpr`/`LTExpr` change from `Symbol` to `NumExpr` in
   the LHS. *Pre-emptive grep* `grep -rn 'GTExpr(\|LTExpr(' src/ test/ apps/` before the PR; each hit
   dispositioned as the mechanical `feat → FeatureRef(feat)` lift. Blast radius: construction sites in
   `enumeration.jl`, `exploration.jl`, and the test corpus. The `max_num_depth = 1` `==` pin (§3) is the
   backstop — any semantic drift in the pre-change enumeration halts the line.
5. **Arithmetic subtrees are not yet abstracted (named limitation).** `perturb_grammar` compresses
   repeated `ProgramExpr` subtrees; a frequently-recurring `NumExpr` (e.g. `Times(FeatureRef(:a),
   FeatureRef(:b))`) is treated as an opaque leaf and not abstracted into a nonterminal in this move.
   Acceptable — abstraction over the numeric sublayer is a follow-on, not a correctness gap (the product is
   still enumerable and priced; it is merely not yet compressible). And the follow-on is more than symbol-
   saving: abstracting a recurring `Times(A, B)` into a nonterminal turns an enumerable product into a
   **learned named composite feature** — the real terminus of the §3.1 ladder (feature *construction*
   completed, not just compression). Noted so it is a decision, not a silent omission, and so the follow-on
   is understood as closing the feature-construction story.

## 7. Verification cadence

```
julia test/test_feature_arithmetic.jl    # lift (== at max_num_depth=1), complexity, product discovery, determinism
julia test/test_program_space.jl         # enumeration bit-stable after the mechanical GTExpr/LTExpr lift
julia test/test_threshold_explore.jl     # Move 3 threshold refinement over the generalised NumExpr slot
julia test/test_feature_discovery.jl     # Move 4 selection untouched
julia test/test_grid_world_meta.jl       # host meta-loop unchanged (arithmetic opt-in; default max_num_depth=1)
```

Full `test/test_*.jl` green before commit; lint self-test + `check apps/`. Skin smoke **optional** — no
wire verb added (arithmetic depth is a brain-decided meta-action — VOI-scored, host-*wired* into the EU-max
loop exactly as `explore_grammar`/`explore_features` are, not a host toggle; `NumExpr` touches no serialised
path in the first move). Halt-the-line on any drift in the pre-change enumeration or any full-suite failure
— the branch never sleeps red.
