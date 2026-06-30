# Move 5 design — the combined single-currency `argmax`: attempt, and the honest finding

> Exploration-budget arc, Move 5 (the capstone). Master plan: `docs/exploration-budget/master-plan.md`
> §4 ("attempt, expect to stop") + Q5. This doc is the **attempt**. It reaches the wall the master plan
> predicted — but the wall is in a sharper place than §4's prose, and the refinement is the deliverable.
> Per the arc discipline this design doc lands and is ratified before any code; the recommended code
> footprint is small-to-zero (a naming/closing move), which OQ-5.2 puts to the reviewer.

## 0. Ratification (2026-06-30, author)

Ratified in conversation:
- **OQ-5.1 — RATIFIED.** Refine master-plan §4 + Q5 from "two currencies (prior nats vs utility)" to
  **one currency (Δ log-evidence), two fidelities** (cheap prior-only surrogate vs re-conditioned exact);
  the invariant is the **compute-tier cascade** (EU-max, Russell–Wefald), not a currency incommensurability.
- **OQ-5.3 — DEFER + NAME.** The dominance benchmark (thesis prediction §3.2) is the one outstanding
  empirical gate; it is **deferred to a paper-gated task and named as such** in the master-plan status,
  not built here. (Discovery §3.1 and graceful degradation §3.3 are already unit-validated.)
- **OQ-5.2 → (b):** ship the two docstring precision notes (`net_voc`, `explore_features`) naming the
  surrogate / exact instances; **no** `net_evidence_voc` merge (it would blur the §2 prior/belief seam).
- **OQ-5.4 → stand-alone:** Move 5 lands as its own move (the §4 refinement deserves its own status line),
  not a Move-4 addendum.

The OQ prose in §5 is retained as the rationale of record.

## 1. Purpose

Move 5 as scoped in the master plan (§4): **attempt the combined single-currency `argmax` over the whole
meta-action space** — compression (`:add_rule` / `:remove_rule` / `:remove_feature`, priced by `net_voc`)
*and* exploration (`explore_grammar` thresholds, `explore_features` features, priced by lookahead VOI) —
and either close the headline ("the metalevel is one `argmax EU`") or **name the residue as a permanent
frontier with the reasoning** (the Phase-5 precedent: do not force a fake common currency).

The master plan **predicted stop**, and stop is correct. But doing the attempt honestly relocates the
wall, and the relocation matters:

> **Master plan §4 reads the gap as "two currencies — compression in prior nats, exploration in
> utility — that cannot merge."** The attempt finds that framing imprecise on both labels. There is
> **one** currency — **Δ log-evidence** (`Δlog P(g) + Δlog P(data|g)`, the log joint) — and
> `explore_features` already *sums its two terms in a single `net_value`*. Compression and exploration
> are not two currencies; they are **two fidelities of the one currency**: a cheap depth-one
> **prior-only surrogate** (compression's `net_voc`, the likelihood term dropped because the prior-only
> signature cannot afford to measure it) versus an expensive re-conditioned **exact** evaluation
> (exploration's lookahead). The thing that must not be flattened is the **compute-tier cascade**
> between those fidelities — and the cascade is *itself* EU-max (Russell–Wefald on the cost of
> evaluation), not a currency incommensurability.

So Move 5 **closes the currency question** (one currency, named and shown) and **names the residue
precisely** (the cheap-surrogate/exact-lookahead fidelity cascade is the permanent structure; and
*separately*, the genuine log-evidence→realised-ΔEU conversion is the standing frontier, shared equally
by both classes so it does not separate them). What unblocks: the arc's headline is settled, with the
master-plan §4 framing upgraded from "two currencies" to "one currency, two fidelities."

## 2. Files touched

This is a naming/closing move. The **only required** artifact is this design doc plus a precision edit to
the master plan; the code touches are all OQ-gated (OQ-5.2) and behaviour-preserving.

- `docs/exploration-budget/move-5-design.md` — **new** (this doc).
- `docs/exploration-budget/master-plan.md` §4 + Q5 — **modify** (required): upgrade "two currencies
  (prior nats vs utility)" to "one currency (Δ log-evidence), two fidelities (prior-only surrogate vs
  re-conditioned exact); the cascade is the invariant." This is a *refinement of a ratified plan* —
  surfaced for ratification as OQ-5.1, not applied unilaterally.
- `src/program_space/perturbation.jl` `net_voc` docstring (~296-315) — **modify, OQ-5.2** (optional):
  one sentence naming `net_voc` as the **Δℓ-dropped prior-only-surrogate instance** of the shared
  Δ log-evidence VOC, pointing at `explore_features` as the exact general instance. Pure comment.
- `src/program_space/exploration.jl` `explore_features` docstring (~316-340) — **modify, OQ-5.2**
  (optional): one sentence naming its `Δℓ + complexity_logprior(Δc)` as the **exact general instance**
  of the same functional, with `explore_grammar` the `Δprior=0` instance and `net_voc` the surrogate.
- **No new code file. No `net_evidence_voc` merge** (rejected in §4 below — it would blur the §2
  prior/belief seam). **No behaviour change** anywhere.

The dominance benchmark (the one un-validated empirical gate, §6) is **out of scope here pending OQ-5.3**
— it is an empirical/paper deliverable, a different kind of work from this conceptual close.

## 3. Behaviour preserved

Move 5 changes no arithmetic and selects no action differently. There is nothing to compute-equivalence
because there is no computational change:

- If Move 5 ships **doc-only** (recommended, §5): the entire `test/test_*.jl` suite is `==` unchanged;
  the assertion is "no source byte under `src/` changed," verified by `git diff --stat src/`.
- If OQ-5.2 elects the docstring precision notes: docstrings only; `julia test/test_voc_gate.jl`,
  `test_threshold_explore.jl`, `test_feature_discovery.jl`, `test_grid_world_meta.jl` stay `==` green
  (capture-before-refactor is trivial — no executable line moves).

There is deliberately **no** `net_evidence_voc` refactor whose equivalence we would have to pin (§4
rejects it), so the strata-1/2/3 tolerance ladder of the template does not apply: this move has no
behaviour-preserving code transform to bound.

## 4. Worked end-to-end example — the three meta-action classes through the one currency

The centrepiece claim is "one currency, three instances, two fidelities." Trace a concrete grammar with
all three classes available. Let `g` have features `{food, enemy}`, one rule `R`, and a residual buffer
`obs` on which the posterior mispredicts.

**The one functional (the currency).** For any grammar edit `g → g'`, the honest value is the change in
log joint evidence, minus the compute spent to evaluate it:

```
net_evidence_voc(g → g') = [ Δlog P(g) + Δlog P(data | g) ] − compute_cost
                         = [ complexity_logprior(Δcomplexity; λ=log2)  +  (mll(obs|g) − mll(obs|g')) ] − cc
                            └────────── prior term ──────────┘     └──────── likelihood term ───────┘
```

Now the three classes, each a special case — and the fidelity each can afford:

1. **Feature discovery** `:add_feature` (owner: `explore_features`, `exploration.jl:342`). `g' = g ∪
   {moved}`. Both terms non-zero: `Δcomplexity = +1` (prior term `−log2`), `Δℓ = mll(obs|g) −
   mll(obs|g') > 0` (a feature the posterior needed). `explore_features` computes
   `net_value((baseline − mll) + complexity_logprior(+1; λ=log2), cc)` — **the exact, general instance,
   both terms, re-conditioned.** Fidelity: **exact (expensive)**.

2. **Threshold refinement** `:modify_threshold` (owner: `explore_grammar`, `exploration.jl:273`). `g'` =
   `g` with a finer grid on `food`. A threshold constant is complexity-invariant
   (`expr_complexity(::GTExpr)=1`), so `Δcomplexity = 0` → prior term vanishes; only `Δℓ` remains.
   `explore_grammar` computes `net_value(baseline − mll, cc)` — **the exact `Δprior = 0` instance,
   re-conditioned.** Fidelity: **exact (expensive)**.

3. **Compression** `:add_rule` (owner: `net_voc` via `perturb_grammar`, `perturbation.jl:314`). `g'` = `g`
   with a frequent subtree abbreviated as a nonterminal. `Δcomplexity < 0` (the dictionary shrinks the
   description) → prior term `log2·Δsymbols > 0`. **The likelihood term is *not zero*** — abbreviating
   reweights the mixture toward compressible programs, which shifts `mll`. But `perturb_grammar` has **no
   belief, no `obs`, no re-conditioning** (the §2 prior-only signature), so it **cannot measure the
   likelihood term and drops it**, scoring `net_value(complexity_logprior(−Δsymbols; λ=log2), cc)` =
   `log2·Δsymbols − cc`. The docstring already names this honestly: *"achievable EU is unaffordable
   depth-one (Russell–Wefald); the affordable value-**proxy** is the change in the complexity prior."*
   Fidelity: **prior-only surrogate (cheap)**.

**What the trace shows.** The same `net_evidence_voc` row, read at three settings of (which terms, what
fidelity): feature = (both, exact), threshold = (likelihood only, exact), compression = (prior only,
**surrogate**). The currency never changes — `explore_features` *proves* prior nats and predictive nats
add coherently (they are both log-evidence; a unit mismatch would make that `+` type-incoherent). So
**the single currency exists and is already in the code.**

**Why the flat argmax is still wrong — the cascade is EU-max, not a workaround.** Suppose Move 5 built the
honest flat argmax: rank *every* candidate by the *exact* `net_evidence_voc`. Then compression's
likelihood term must be measured too — i.e. compression must be re-conditioned — making the cheap screen
as expensive as the lookahead. Russell–Wefald forbids exactly this: the **cost of evaluating** a
candidate nets against its value, and you do not pay the expensive exact evaluation when a cheap
**surrogate-positive** compression win is in hand. The saturation gate (`compression_exhausted ∧
plateau`, Move 2) is the *already-EU-derived* form of that meta-meta-decision: spend the expensive exact
lookahead only when (a) no cheap surrogate-positive compression remains and (b) the residual predicts the
lookahead will pay. So the "combined `argmax`" exists — but as a **two-tier cascade** (cheap prior-only
surrogate screen → expensive exact lookahead, ordered by saturation), each tier an `argmax` in the one
shared log-evidence currency. A flat one-shot `argmax` is not *more* unified; it is the cascade with its
EU-optimal evaluation-cost ordering deleted.

**The genuine residue (named, not closed).** None of the three is realised **ΔEU** (object-level decision
value, A2). All are **log-evidence** (information / model improvement). Converting log-evidence → ΔEU
needs decision-lookahead — the `net_voi` form (re-decide, measure the EU gain), which is depth-≥1 *over
actions* on top of the depth-≥1 over grammars. That conversion is the standing frontier — but it is
**shared equally** by compression and exploration (both improve the model; neither's model-improvement is
yet cashed into decision value), so it is **not** the wall that separates the two classes. The wall that
separates them is the **fidelity/compute-tier** boundary above. Two distinct frontiers, cleanly named.

## 5. Open design questions

1. **OQ-5.1 — Ratify the §4 refinement, or defend the original "two currencies."** The attempt finds
   master-plan §4's "compression in prior nats vs exploration in utility, cannot merge" imprecise: it is
   one currency (Δ log-evidence — `explore_features` sums both terms), two **fidelities** (prior-only
   surrogate vs re-conditioned exact), and the invariant is the **compute-tier cascade** (EU-max via
   Russell–Wefald), not a currency incommensurability; "utility" mislabels exploration's predictive-nats
   too. **Recommendation: ratify the refinement** and edit §4 + Q5 to the "one currency, two fidelities"
   framing — it is more correct (the `+` in `explore_features` is the proof), it strengthens rather than
   weakens the "don't flatten" conclusion (Russell–Wefald is a sharper reason than "different units"),
   and it adds **no constitutional text** (it is a precision edit to an arc plan, not to CLAUDE.md/SPEC).
   Counter to weigh: a ratified master plan changing its central §4 wording is not free; if the reviewer
   reads the surrogate-vs-exact distinction as *already implied* by §4's "depth-one prior-priced," then
   the edit is a clarification, not a correction, and could stay a footnote rather than a rewrite.

2. **OQ-5.2 — Code footprint: doc-only, or the two docstring precision notes?** The single currency is
   already in the code (shared `net_value`; `explore_features` sums the terms). The options are (a)
   **doc-only** — this design doc + the §4 edit, no `src/` change; or (b) **+ two docstring notes** on
   `net_voc` and `explore_features` naming them the surrogate / exact-general instances of the one
   functional, so the relationship is legible at the call sites. **Recommendation: (b), the two docstring
   notes** — they cost nothing, change no behaviour, and put the move's finding where the next reader of
   `net_voc` will see it (executable-documentation spirit), without the rejected merge. A full
   `net_evidence_voc` refactor routing all three through one function is **rejected**, not deferred:
   compression is prior-only by signature and exploration belief-aware (§2), so a shared functional would
   blur the prior/belief seam the architecture deliberately draws — the `measure-as-view` lesson
   (carrier-free vs carrier-bound stays split). Counter: even docstring edits touch `src/` on a move
   whose honest footprint might be zero; if the reviewer prefers the cleanest possible "naming move,"
   (a) is defensible and the notes move into this doc only.

3. **OQ-5.3 — The dominance benchmark (prediction §3.2): fold into Move 5, or name-and-defer?** Of the
   thesis's three empirical gates, **discovery** (§3.1) and **graceful degradation** (§3.3) are validated
   by unit tests (`test_feature_discovery`, `test_threshold_explore` §3c's continuous explore/no-op flip
   at `cc = Δℓ`); **dominance** — beating both random exploration and a fixed-schedule baseline — has **no
   comparative benchmark** in the suite. It is the one outstanding gate. **Recommendation: name it as the
   single outstanding empirical gate and treat it as a separate paper-gated deliverable, NOT part of this
   conceptual close** — the currency finding stands or falls on the analysis in §4, not on a benchmark,
   and a dominance figure is benchmark/figure work (closer to the `paper-as-gating-artifact` track) that
   would balloon a naming move. Fold it in **iff** this arc is feeding a specific paper now (then Move 5
   becomes "conceptual close + empirical capstone"); otherwise defer with the gate named. Reviewer
   decides based on whether exploration-budget feeds paper1-4 this cycle.

4. **OQ-5.4 — Does Move 5 stand alone, or fold into Move 4 retroactively (master plan's "May fold into
   Move 4 (OQ-5)")?** Given the recommended doc-only/near-zero footprint, Move 5 could be recorded as a
   §-appendix to Move 4 rather than its own move. **Recommendation: stand alone as Move 5.** The attempt
   *refines a ratified plan* (OQ-5.1) and *names two distinct frontiers* — that is a substantive close
   deserving its own ratifiable artifact and its own master-plan status line, even at near-zero code.
   Folding it into Move 4 would bury the §4 refinement in a move that has already landed. Counter: if
   OQ-5.1 resolves to "clarification, not correction" and OQ-5.2 to "doc-only," the move is light enough
   that a reviewer could reasonably prefer it as a Move-4 addendum.

## 6. Risk + mitigation

- **Over-claiming the unification (the headline risk).** Failure mode: reading "one currency" as "the
  flat single `argmax` is fine, build it." Blast radius: would motivate replacing the cheap compression
  screen with an exact re-conditioned evaluation — destroying the resource-rational cascade the whole arc
  rests on (the precise harm the master plan warned of). Mitigation: §4's worked example makes the
  flat-argmax-is-wrong argument explicitly and EU-theoretically (Russell–Wefald on evaluation cost), and
  the **recommendation ships no merge** — the cascade stays exactly as built; nothing in this move
  changes a selection path. The `average-not-collapse` precedent is adjacent but distinct (that forbids
  collapsing a *posterior* to its MAP to drive a decision; here we forbid collapsing a *two-fidelity
  cascade* to a flat exact argmax) — no pragma site, because no code asserts the flattening.

- **Contradicting a ratified master plan without consent.** Failure mode: silently rewriting §4. Blast
  radius: the master plan is the arc's durable record; an unratified edit erodes it. Mitigation: the §4
  edit is **gated on OQ-5.1** and applied only on ratification; until then this doc carries the refinement
  and the master plan is untouched. No CLAUDE.md/SPEC text changes (the arc's hard constraint): the
  refinement lives entirely in arc-local docs.

- **Scope creep via the dominance benchmark.** Failure mode: a naming move silently growing into a
  benchmark/figure build. Blast radius: Move 5 stops being bisectable/small and the conceptual close gets
  held hostage to figure tuning. Mitigation: OQ-5.3 makes the fold-in an explicit, defaulted-to-defer
  decision; the gate is **named** either way (no silent cap on the thesis — the outstanding gate is
  recorded, not buried).

## 7. Verification cadence

- **Doc-only path (recommended):** `git diff --stat src/ apps/` shows zero lines; lint self-test
  `python tools/credence-lint/credence_lint.py` + `check apps/` pass (no new pragmas, slug index intact).
  No skin smoke needed — Move 5 changes nothing crossing the wire (template: skin smoke optional for
  Moves 1/2/5/8).
- **If OQ-5.2 (b) docstring notes land:** additionally run the arc suite to confirm docstrings did not
  disturb executable lines — `julia test/test_voc_gate.jl && julia test/test_threshold_explore.jl &&
  julia test/test_feature_discovery.jl && julia test/test_grid_world_meta.jl &&
  julia test/test_compression_removal.jl && julia test/test_saturation.jl` — all `==` green. Then the
  full `test/test_*.jl` suite green before commit.
- **Halt-the-line:** any test failure, or any non-zero `src/` diff on the doc-only path, is a halt — the
  branch never sleeps red, and a "naming move" that mutated behaviour is a contradiction to stop and
  investigate, not to patch forward.
