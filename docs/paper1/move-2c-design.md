# Move B2c design doc — reliability learning under inferred category uncertainty

Status: design doc (code deferred until the open questions in §5 resolve).
Supersedes the rejected (γ) MAP assumption (master-plan §4a, corrected
2026-05-31).

## 1. Purpose

B2c wires the B2b category classifier (`apps/julia/qa_benchmark/
category_inference.jl`) into the agent under the Phase-B "fair
conditions": the category is **inferred, not given**, and every agent
sees the *same* soft posterior π. The Bayesian agent then both **decides**
(VOI/EU) and **learns** (reliability update) under that category
uncertainty. The load-bearing decision this doc settles is *how the
reliability update is performed* — and from first principles it is
**exact Bayesian conditioning**, not the MAP collapse an earlier draft
proposed. Once settled, B2c unblocks B3 (tools-only slice) and B4
(fairness-equalised prompting + re-run).

Scope refinement vs the master plan: the master plan's B2/B4 wired a
*given* category (`rel_betas[t, cat_idx]`). B2c replaces that index with
the inferred posterior π entering **both** decision and update. Real
question-bank embeddings remain gated (master plan §5); B2c specifies and
tests the mechanism on synthetic embeddings/posteriors, and the real-
embedding wiring lands with B3/B4.

## 2. The model, from first principles

Per A1–A3 and Invariant 1 (uncertainty is encoded in the hypothesis
space so `condition` can learn it):

- Reliability hypothesis: `θ_{t,c}` = P(tool t correct | true category c),
  one Beta per (tool, category) — the existing `rel_betas[t, c]`.
- Per question q the true category `C_q` is latent; the classifier
  supplies the soft posterior `π_q = P(C_q = · | embedding_q)` (frozen,
  from the LOO classifier — it is *evidence*, not a belief that the tool
  outcomes update).
- Observation: outcome `o ∈ {correct, wrong}` for tool t on q, with
  `P(o = correct | θ, C_q = c) = θ_{t,c}`. Marginalising the latent
  category: `P(correct | θ) = Σ_c π_{q,c} · θ_{t,c}`.

The **decision side** wants the category-marginalised reliability
belief: a `MixturePrevision([Beta(θ_{t,c}) for c], log π_q)` fed into
`eu`/`voi`/`expect`. This is exact and needs no approximation — it
replaces the `rel_betas[t, cat_idx]` argument at the decision sites.

The **update side** is `condition` against the likelihood
`Σ_c π_{q,c} θ_{t,c}`. Because that likelihood is a *sum* over c, it is
not conjugate to a product of independent Betas: each observation splits
the belief into a K-component mixture (one component per category, with
that category's Beta incremented), so the exact posterior after `n`
observations of tool t is a mixture of up to `K^n` product-of-Betas
components. **This is the exact posterior** — `condition` produces it.

## 3. Tractability is metacomputation, not a bolted-on approximation

`K^n` growth is a *computational cost*, and per the spec's "On
metareasoning" the choice of how much to compute is itself an EU decision
made by the same reasoner — not an escape hatch from Invariant 1. The
computational-strategy space for the reliability posterior:

| Strategy | What it keeps | Cost | Exactness |
|---|---|---|---|
| **Exact** | all `K^n` components | unbounded | exact |
| **Pruned** | top-`m` components by weight (`prune`/`truncate`, `ontology.jl:1471`) | `O(m·K)` / obs | exact-up-to-tail-mass |
| **ADF-collapse** | project to one product-of-Betas each step (assumed-density filtering) | `O(K)` / obs | mean-field |

The **fractional-pseudocount** update — `α_{t,c} += π_c·[correct]`,
`β_{t,c} += π_c·[wrong]` — is the ADF-collapse special case (the moment-
matched projection back to independent Betas; it is *not* "prune to one
component", which would be MAP-like). All three are computational
strategies the metacomputation can select by EU; none is hard-coded.
This keeps the whole spectrum *inside* Invariant 1: the Julia execution
layer implements the strategies (mixture `condition` + `prune`/`truncate`,
or a fractional-evidence conjugate update), and the DSL specifies the EU
choice among them — exactly the metareasoning posture the constitution
already mandates.

## 4. Files touched (B2c code; deferred until §5 resolves)

- `src/kernels.jl:19` — **(conditional on §5 OQ1)** add a
  `WeightedBernoulli <: LeafFamily` (carries no params; the weight rides
  on the observation) for the ADF-collapse fast path. *New type.*
- `src/conjugate.jl:21` — register `maybe_conjugate`/`update` for
  `(BetaPrevision, WeightedBernoulli)`: `α += w·o, β += w·(1-o)` with
  fractional `w`. Sits beside the existing unit-count BetaBernoulli; does
  not modify it. *Modification.*
- `apps/julia/qa_benchmark/host.jl` — decision sites (`:66, 84, 113,
  134`) build the category-marginalised `MixturePrevision` over π and
  pass it where `rel_betas[t, cat_idx]` is passed today; update site
  (`:177`) conditions under category uncertainty (mixture or
  fractional). The given-category read `cat_idx = findfirst(==(q.category)
  …)` (`:58`) is replaced by the classifier's soft posterior. *Modification.*
- `apps/julia/qa_benchmark/agent.bdsl` — **(conditional on §5 OQ3)** the
  answer-kernel/reliability-kernel consume the marginalised reliability;
  likely no change if the host owns the `MixturePrevision` construction.
- `test/test_qa_benchmark_category_update.jl` — reliability-update tests.
  *New file.*

The real-embedding wiring (`environment.jl` question-bank embeddings) is
**not** in B2c — it lands with B3/B4 per the embedding gate.

## 5. Behaviour preserved

B2c deliberately *changes* the v1 benchmark numbers (inferred category,
soft update); it does not preserve them. What it must preserve is the
**degenerate reduction**: when π is one-hot (`π_c = 1` for the true c),
both decision and update must reduce *exactly* to today's code.

- Decision: a one-hot `MixturePrevision` over π equals `rel_betas[t, c]`
  (single component) — assert `==` on `mean`/`weights`.
- Update: the soft update on one-hot π equals the current
  `condition(rel_betas[t,c], RELIABILITY_KERNEL, o)` unit-count update —
  assert `==` on `(α, β)`.

Tolerances: degenerate-reduction `==` (integer pseudocounts); fractional
arithmetic `rtol = 1e-12`; exact-mixture worked example `==` on the
closed-form weights/parameters in §6.

Capture-canonical discipline: capture the current host's
`(α, β)` sequence for a fixed seed *before* the change; assert `==`
on the one-hot path post-change (per `feedback_capture_canonical_before_
refactor`).

## 6. Worked end-to-end example

Two categories, `π = [0.7, 0.3]`, tool t, `rel_betas[t,1] = Beta(2,3)`,
`rel_betas[t,2] = Beta(1,1)`; observe **correct**.

**Exact (mixture `condition`).** Prior is the single product
`Beta(θ₁;2,3)·Beta(θ₂;1,1)` (weight 1). Likelihood `0.7·θ₁ + 0.3·θ₂`:

```
posterior ∝ 0.7·[θ₁·Beta(θ₁;2,3)]·Beta(θ₂;1,1)
          + 0.3·Beta(θ₁;2,3)·[θ₂·Beta(θ₂;1,1)]
```

Using `θ·Beta(θ;α,β) = (α/(α+β))·Beta(θ;α+1,β)`:
`θ₁·Beta(θ₁;2,3) = 0.4·Beta(θ₁;3,3)`, `θ₂·Beta(θ₂;1,1) = 0.5·Beta(θ₂;2,1)`.

```
∝ 0.28·[Beta(3,3)·Beta(1,1)]  +  0.15·[Beta(2,3)·Beta(2,1)]
```

→ a **2-component mixture**, normalised weights `0.28/0.43 = 0.651` and
`0.15/0.43 = 0.349`. Owner: `condition(::MixturePrevision, k, obs)`
(`ontology.jl:1121`) with the `Σ_c π_c θ_{t,c}` kernel; the per-component
Beta increments are the existing conjugate path.

**ADF-collapse (fractional pseudocounts).** Update each independently:
`rel[1] → Beta(2.7, 3)`, `rel[2] → Beta(1.3, 1)`. Single product
component. Owner: `condition(::BetaPrevision, WeightedBernoulli kernel,
(o, π_c))` → `update` in `conjugate.jl`. The marginal `E[θ₁]`: exact
mixture `0.651·(3/6) + 0.349·(2/5) = 0.465` vs ADF `2.7/5.7 = 0.474` —
close, not equal: the mean-field discrepancy made explicit.

**Degenerate check (`π = [1,0]`).** Exact: likelihood `θ₁`, posterior is
the single component `Beta(3,3)·Beta(1,1)`. ADF: `rel[1] → Beta(3,3)`,
`rel[2] → Beta(1,1)`. Both equal today's unit-count update
`Beta(2,3) --correct--> Beta(3,3)`. `==`.

## 7. Open design questions

1. **Is the strategy spectrum (exact / pruned / ADF) one mechanism under
   metacomputation, or do we ship a single fixed strategy for Paper 1?**
   Framing all three as EU-selected computational strategies is the
   constitution-pure answer, but it requires a cost model in the utility.
   The alternative is to implement *one* strategy now (recommended:
   start from exact mixture + a retention cap, with ADF as the cap=1
   degenerate) and defer the EU-over-strategies selector. Argue whether
   Paper 1 needs the full metacomputation or a named, fixed default.

2. **Default strategy + retention budget.** If a fixed default: exact
   mixture with top-`m` pruning (what `m`? chosen how — fixed, or a
   tail-mass threshold via `prune`'s `threshold`?), or ADF-collapse
   (tractable, on the conjugate fast path, but mean-field)? The
   Performance-Problems protocol governs: if a 50-question run with the
   exact/pruned path exceeds a wall-clock budget, **halt and report** —
   do not silently downgrade to ADF.

3. **Residency of the decision-side marginalisation.** Does the host
   build the `MixturePrevision` over π and pass it (bdsl unchanged — D3
   stays true), or does `agent.bdsl` take π and the per-category Betas
   and marginalise? Invariant 2 says the `MixturePrevision` is the
   declared structure; recommendation is host-builds, bdsl-consumes, but
   confirm against the answer-kernel's current signature.

4. **Substrate-change discipline for `WeightedBernoulli`.** Adding a
   `LeafFamily` + conjugate pair is sanctioned vocabulary growth
   (constitution: "named distributions … may be added"), but it touches
   `src/`. Does it ride in the B2c code PR, or land as its own
   substrate PR first (capture-canonical, stratum-2 conjugate test)
   ahead of the host wiring? This is the one place the B2b "no new
   `LeafFamily`/`ConjugatePrevision`" refusal is lifted — that refusal
   lived under the now-reversed MAP regime.

## 8. Risk + mitigation

- **Mixture explosion** (`K^n`). Blast radius: a benchmark run hangs/OOMs.
  Mitigation: the metacomputation/retention strategy of §3 + a wall-clock
  budget on the 50-question run; per Performance-Problems, exceeding it
  **halts and reports** rather than silently collapsing to ADF.
- **Degenerate regression.** The soft path must reduce to v1 on one-hot
  π. Mitigation: capture-canonical `(α,β)` sequence pre-change; assert
  `==` (§5).
- **Invariant 1 (topological).** The fractional update must be a
  *registered* `condition`/`update`, never host arithmetic on weights.
  Mitigation: `WeightedBernoulli` goes through `maybe_conjugate`; lint
  `check apps/`; no `rel_betas[...].alpha += …` anywhere host-side.
- **Grep step (pre-PR).** `grep -rn 'rel_betas\|cat_idx\|q.category'`
  across `apps/julia/qa_benchmark/` and list each hit's disposition
  (decision-site marginalisation / update-site soft-condition / replaced
  by classifier posterior / no change).
- **Fairness asymmetry.** Both agents must receive the *identical* soft
  posterior; the LLM prompt surface (B4) must mirror it. Mitigation: the
  B4 symmetry audit; out of scope here but named.

## 9. Verification cadence

At end of the B2c **code** PR (not this design PR):

- `julia test/test_qa_benchmark_category_update.jl` — degenerate-equals-v1
  (`==`), fractional arithmetic (`rtol 1e-12`), exact-mixture worked
  example weights (`==`), determinism.
- `julia test/test_qa_benchmark_category_inference.jl` — B2b, must stay
  green (unaffected).
- If the substrate change lands here: the stratum-2 conjugate test for
  `(BetaPrevision, WeightedBernoulli)` (`_dispatch_path === :conjugate`
  before value).
- `python tools/credence-lint/credence_lint.py check apps/` — clean.
- Halt-the-line: any red suite halts; no "fix in next commit".

This design PR itself is docs-only — no test/lint impact.
