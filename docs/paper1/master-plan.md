# Master plan — Paper 1 methodology revisions (Phase B)

Branch: `paper1/methodology`.

This is the durable in-repo copy of the branch's master plan. Phase B
operationalises the new Paper 1 thesis: that Bayesian VOI tool selection
occupies a non-empty region of the cost-performance Pareto frontier under
fair conditions. Phase B1 (this document + the draft PR that ships it)
opens the work and surfaces design questions; Phase B2–B5 implement them
in sequence.

Modelled on `docs/posture-3/master-plan.md` per `CLAUDE.md` §"Multi-move
branches: design-doc before code".

---

## 1. Context

The v1 benchmark (`apps/julia/qa_benchmark/`) shipped with two structural
fairness asymmetries that, taken together, make the comparison ambiguous
about what is actually being measured:

1. **Categories are given asymmetrically.** The Bayesian agent receives
   `Question.category` as a perfect oracle
   (`apps/julia/qa_benchmark/host.jl:58` →
   `rel_betas[t, cat_idx]`). LLM agents see only question text and four
   candidate strings (`apps/julia/qa_benchmark/llm_agent.jl:206-209`). The
   Bayesian agent's "knows which tool to use for this category" is
   structural; the LLMs' is inferential. This is not a bug — it was a
   modelling choice that exposed the reliability matrix's structure
   directly to the Bayesian agent — but it makes the empirical comparison
   read on different axes for different agents.
2. **Parametric world knowledge available across the question bank.** Many
   of the 50 questions (factual: 15, recent_events: 8, misconceptions: 7)
   are answerable from frontier-LLM world knowledge alone, without tool
   calls. Haiku 4.5 answers ~30/50 with no tools (`papers/RESULTS.md`).
   The benchmark cannot isolate the value of *tool selection* from the
   value of *parametric recall*.

The Phase A bootstrap (`papers/paper1/bootstrap-results.md`, 2026-05-04)
sharpened the picture under v1 conditions: the greedy ablation — query
the highest E[reliability] tool once, submit (no VOI, no abstention) —
beats the full Bayesian agent at Δ = +25.75, 95% CI [+1.15, +51.10],
p = 0.0386, paired N = 20. The result is marginally significant; the lower
CI bound is barely positive. But its sign is exactly what we should expect
under v1 conditions: when the category is given, the marginal value of
VOI's tool-by-tool exploration shrinks (the reliability matrix already
concentrates on the right tool for this category), and abstention can
hurt when the best-tool mean is already high enough that submission EU is
positive on average.

Phase B's response is methodological, not algorithmic. We change the
conditions under which the framework is evaluated; we do not change the
framework itself.

### 1.1 The trajectory of three re-framings

This is Paper 1's **second** re-framing in short succession. The
trajectory matters and must not be quietly overwritten:

- **Original (Feb 2026 draft):** "Bayesian +129.5 vs best LLM +10.8" —
  Bayesian dominates LLMs even under unequal conditions. Recorded as the
  now-obsolete framing in `papers/CLAUDE.md` lines 13-34.
- **Interim (March 2026 redesign):** "Bayesian is the principled,
  zero-cost approach; frontier LLMs win on raw score but at API cost."
  Pivot triggered by Haiku 4.5 +445.5 vs Bayesian +163.7 in the redesigned
  benchmark (`papers/RESULTS.md`). This framing is what `papers/CLAUDE.md`
  currently reflects under "Updated headline results".
- **Phase B (now):** "Bayesian VOI tool selection occupies a non-empty
  region of the cost-performance Pareto frontier under fair conditions."
  Pivot triggered by the Phase A bootstrap result and recognition of the
  v1 fairness asymmetries.

Each pivot was driven by a specific empirical observation that
invalidated a quantitative claim of the previous framing — frontier-LLM
dominance for the first pivot, the greedy-ablation result for the second.
This is deliberate epistemic progress, not drift. `papers/NOTES.md`
documents the trajectory explicitly as part of Phase B's pivot record;
future revisions of Paper 1 should not collapse the history into a single
"current" framing without a corresponding record of why.

---

## 2. Thesis

> **Bayesian VOI tool selection occupies a non-empty region of the
> cost-performance Pareto frontier under fair conditions.**

"Fair conditions" means:

(a) Categories are not given to any agent — all agents must infer
category from question content.

(b) The benchmark includes a slice where parametric world knowledge
cannot meaningfully contribute, isolating tool selection from parametric
recall.

(c) LLM prompts receive the same inferred-category information that the
Bayesian agent uses, in equivalent form.

A non-empty Pareto region is the load-bearing claim. Bayesian need not
dominate; it needs to occupy points on the frontier that no other agent
strictly dominates. If after B5 it does not, Paper 1 reports that
honestly — the methodology is designed to be willing to falsify the
thesis.

---

## 3. Move sequence (B2–B5)

Each move opens with a design-doc PR (`docs/paper1/move-N-design.md`,
modelled on `docs/posture-3/DESIGN-DOC-TEMPLATE.md` adapted for the
methodology context) before the implementation PR. Each design doc must
include an "Open design questions" section per the established
convention. The OQs surfaced in §4 below resolve in conversation between
the author (Guy) and Claude Code; the per-move design docs record the
resolutions and open any new ones that surface during implementation.

### B2 — Category inference

Implement the Bayesian category-inference component decided by OQ1, OQ2,
and OQ3. Resolve the architectural home (environment / body / brain),
the parametric form (full Bayesian / conjugate / MAP+Laplace), and the
calibration set. Emit a per-question category posterior (or hard label,
depending on OQ5) consumable by all agents. This is the load-bearing
module — without it, fairness cannot be enforced for either the Bayesian
agent or the LLMs.

**Deliverable:** category-inference module + tests + calibration
evaluation showing posterior calibration on held-out questions.

**Depends on:** OQ1, OQ2, OQ3 resolved. No other move.

**Design doc:** `docs/paper1/move-2-design.md` (B2a — surfaces evidence
for OQ1 + OQ2 joint resolution; splits OQ2(b) into (b-NB) Gaussian
Naive Bayes and (b-PG) Pólya-Gamma multinomial logistic per the B2a
finding that the master plan's sub-bullet is ambiguous).

### B3 — Tools-only slice

Construct a slice of ~50 questions where parametric world knowledge cannot
meaningfully contribute, in proportions resolved by OQ4. Augment or
replace `QUESTION_BANK` per the architectural decision (in-place
augmentation vs. parallel slice). Verify that frontier-LLM-with-no-tools
accuracy is at chance (~0.25 on 4-choice) on this slice — if it isn't,
the slice does not do what the methodology requires.

**Deliverable:** question content + held-out validation pass + a script
that runs frontier LLM without tools to confirm the slice's
parametric-recall floor.

**Depends on:** B2 at the calibration level (tools-only-slice questions
must carry category labels for B2's calibration; depending on OQ3's
resolution this may or may not couple the moves tightly). OQ4, OQ6
resolved.

### B4 — Fairness-equalised LLM prompting

Equip `apps/julia/qa_benchmark/llm_agent.jl` with the OQ5-resolved
category-information surface: hard label, soft distribution, with or
without per-tool reliability profile. Audit symmetry: whatever the
Bayesian agent reads from B2 must reach the LLM in equivalent form, with
no extra information leaking in either direction.

**Deliverable:** updated `llm_agent.jl` + symmetry audit doc + per-prompt
diff against the v1 prompts.

**Depends on:** B2 (category inference output), OQ5 resolved.

### B5 — Re-run benchmark and Pareto analysis

Run all agents (Bayesian, frontier LLM, local LLM, baselines, ablations)
under the new conditions. Generate the cost-performance Pareto plot.
Verify the thesis: Bayesian occupies a non-empty Pareto region. If it
does not, do not paper over it — Paper 1 either reframes again or reports
the negative result honestly.

**Deliverable:** new tables in `papers/RESULTS.md`, updated paired-
bootstrap results, Pareto plot, explicit frontier-membership statement.
Paper 1 LaTeX rewrite (Phase D) is downstream of B5; B5 stops at the
empirical artefact.

**Depends on:** B2, B3, B4 all complete and integrated. OQ7 resolved.

---

## 4. Open design questions

The five questions from the Phase B1 prompt, plus two surfaced while
reading the codebase. The author (Guy) resolves each in conversation; the
master plan updates as questions close. This is a floor, not a ceiling —
B2/B3/B4 design docs may surface more.

### OQ1 — where does category inference live architecturally?

`SPEC.md` §4.2 says the observation model is learned, and §6 puts
perception in the body. But the paper requires that every agent see the
same category inference for fairness. Three candidates:

- **(a) Environment-side, deterministic per question.** The environment
  ships each question with a pre-computed category label (possibly with
  confidence). All agents see the same labels. Trivially fair, but the
  paper's "Bayesian multinomial logistic" claim evaporates if the labels
  are deterministic — there is nothing Bayesian about a lookup table.
- **(b) Environment-side, learned once on a held-out calibration set,
  frozen and shared.** All agents see the same posterior category
  distribution per question. The Bayesian claim survives. But this
  conflicts with the embodiment doctrine — perception is supposed to live
  in the body, and freezing the posterior at training time means the
  category-inference component cannot learn during evaluation.
- **(c) Body-side, identical implementation across agents.** Most aligned
  with `SPEC.md` §6. Each agent infers categories from features
  independently. Fairness depends on the agents using identical inference
  machinery, audited per evaluation.

The trade-off is between architectural purity (c) and fairness
auditability (b). Each option also has a different relationship with the
"learnable observation model" claim in `SPEC.md` §4.2, which Guy may want
to weigh in resolution.

> **Resolved (2026-05-31): (c) body-side, identical implementation.**
> Each agent runs the *same* category-inference machinery on the
> question's embedding; fairness is enforced by identical
> implementation, audited per evaluation. Aligns with `SPEC.md` §6
> (perception in the body). The shared module is
> `apps/julia/qa_benchmark/category_inference.jl` (B2b).

### OQ2 — what is the form of the Bayesian multinomial logistic?

The paper claim is "Bayesian." Three candidates:

- **(a) Full Bayesian.** Dirichlet prior on coefficients; posterior via
  HMC or variational Bayes. Strongest Bayesian story; highest
  implementation cost (HMC infrastructure not present in the repo).
- **(b) Conjugate where possible.** Pólya-Gamma augmentation for binary
  logistic, extended to multinomial via stick-breaking. Principled,
  conjugate, more code. *Sub-bullet (Phase B1 pre-flight observation):*
  in the credence DSL idiom, a categorical-likelihood version of (b)
  reduces to a `MixtureMeasure` of `CategoricalMeasure` over category-
  conditional feature distributions plus a `Kernel` from features to
  category — `condition`/`expect`/`voi` exist already and would do the
  inference without new primitives. This may materially lower the cost
  estimate of (b) relative to (a) and (c) inside this codebase. It is
  *not* a reason to pre-resolve toward (b); a Pólya-Gamma multinomial is
  still more code than (c), the conjugacy story for the credence-DSL
  reduction needs verification, and (a)/(c) have independent merits
  (richest Bayesian story for (a); cleanest conjugate-free path for (c)).
  Surface it as an input to OQ2's resolution, not a resolution.
- **(c) MAP estimation with informative priors, posterior via Laplace
  approximation.** Cheap and defensible; the Bayesian claim is weaker
  but legitimate (point estimate + Hessian-derived uncertainty).

The trade-off is between the strength of the Bayesian story and
implementation cost. The answer affects both B2's design doc and how
aggressively §3 of the paper can lean on "principled Bayesian inference"
language.

> **Resolved (2026-05-31): (b-NB) Gaussian Naive Bayes on embeddings
> with a Dirichlet class prior.** The B2a design doc
> (`docs/paper1/move-2-design.md` §3.1) verifies this reduces to
> existing credence primitives — per-(class, dim) NormalGamma conjugate
> updates plus a Dirichlet class prior, Student-t marginal predictive —
> with no new primitives. The Pólya-Gamma multinomial-logistic path
> (b-PG, §3.2) is rejected: it needs a Pólya-Gamma augmentation
> primitive that does not exist and is out of scope for Paper 1. See
> the classifier-naming addendum below — "Naive Bayes", not
> "discriminative multinomial logistic".

### OQ3 — calibration set design

Category inference is a perception model. It needs to be calibrated.
Where does the calibration data come from?

- **(a) Hold out a fraction of the existing question bank for
  calibration.** Reduces the test set; introduces possible distribution
  mismatch with the new tools-only slice (B3).
- **(b) Generate calibration data separately** (synthetic, or external
  question banks like MMLU's category labels). Avoids cannibalising the
  test set; introduces train/test distribution mismatch.
- **(c) Cross-validated inference (LOO or k-fold).** No held-out set;
  every question gets an inference from a model trained on all others.
  Cleanest fairness story; more compute per run.

(c) is most defensible if compute allows. The choice also bears on B2/B3
coupling: if (c), B3's tools-only-slice questions need category labels
ahead of B2's evaluation pass; if (a)/(b), the coupling is looser.

> **Resolved (2026-05-31): (c) cross-validated leave-one-out.** Every
> question receives a category inference from a classifier fit on all
> *other* questions — no held-out set, cleanest fairness story.
> Implemented as `loo_category_inference` (B2b.3). This couples B2/B3:
> the tools-only slice's questions must carry category labels ahead of
> B2's evaluation pass (B3 owns that).

### OQ4 — tools-only slice composition

The slice must contain ~50 questions where parametric world knowledge
cannot meaningfully contribute. Three candidate sources:

- **(a) Synthetic arithmetic at high precision** (e.g. 7+-digit
  multiplication). Calculator-tool dominates trivially; LLMs error
  predictably. Simple to generate; possibly too on-the-nose — reviewers
  may read the slice as engineered rather than principled.
- **(b) Post-cutoff events** (calibrated to the latest frontier model's
  cutoff date — verify current cutoff for Haiku 4.5 / Claude Opus 4.7
  before assuming). Realistic; web-search-tool dominates. Hardest to
  scale; freshness decays as models update.
- **(c) Novel facts in a fictional knowledge base** (e.g. "the GDP of
  [fictional country] is X" — the simulated KB has the answer; LLMs
  cannot know it). Simulated already; fits the existing benchmark shape.
  Tool-routing logic is what the slice is testing.

A mix is probably best. The author resolves the proportions. Beware: the
slice's mean per-tool reliability must remain calibrated against the
existing reliability matrix — generating a slice where every question
favours one tool over-trivialises the comparison.

> **Status: open — B3 territory.** Resolved when the tools-only slice
> is designed. Not required for B2b.

### OQ5 — fairness equalisation specifics

LLM prompts must receive equivalent category information to whatever the
Bayesian agent's posterior uses. What does this look like concretely?

- **Hard category label:** "this question is in category C." Easiest to
  audit for symmetry; weakest signal.
- **Soft category distribution:** "P(category=C₁)=0.7, P(category=C₂)=0.2,
  P(category=C₃)=0.1." Symmetric with a Bayesian agent that takes the
  full posterior into VOI. Harder to audit (LLMs may underweight the tail
  of the distribution in ways that the Bayesian agent doesn't).
- **Plus per-tool reliability profile** (e.g. "for category C, the per-
  tool expected reliabilities are A: 0.70, B: 0.92, C: 0.25, D: 0.65").
  The Bayesian agent reads this from `rel_betas`; passing it to the LLM
  closes that asymmetry too. But it raises a separate question: is this
  step too far? If the LLM is given the reliability profile, what is left
  for it to reason about? The paper's claim is about *tool selection
  under uncertainty*; if uncertainty is removed, it is no longer a
  comparison.

The author resolves. Be aware that OQ5's resolution determines what B4
audits.

> **Resolved (2026-05-31): soft category distribution.** Both agents
> receive the full posterior `P(category=·)` per question, not a hard
> label — the symmetric surface for VOI. The Bayesian agent's
> *internal* use of that posterior is governed by the reliability-update
> note below (it conditions exactly under category uncertainty — the
> full posterior, not a MAP collapse), and the information surface
> handed to every agent is identical and soft.
> Per-tool reliability profiles are *not* exposed (the third
> sub-option is too far — it removes the very uncertainty the paper is
> about).

### OQ6 — slice integration: parallel evaluation track or unified benchmark?

(Surfaced in B1 reading.) Two architectural shapes for B3's slice:

- **(a) Parallel evaluation track.** The tools-only slice is a separate
  benchmark run from the existing 50-question bank. Each slice has its
  own table in `papers/RESULTS.md`; the Pareto analysis (B5) reports two
  Pareto plots, one per slice. Cleanest signal-isolation; halves
  per-slice statistical power.
- **(b) Unified benchmark with stratified analysis.** The tools-only
  slice augments the existing question bank into a single ~100-question
  evaluation; per-stratum analysis (factual / numerical / tools-only / …)
  reports per-slice Pareto membership. Higher per-question N for
  marginals; introduces the question of how to weight strata if proportions
  differ from a target population.

Affects B3's deliverable and B5's Pareto-plot shape. The author resolves;
(a) is probably cleanest unless the power calculation in OQ7 forces (b).

> **Status: open — B3 territory.** Resolved alongside OQ4 when the
> slice's integration shape is decided. Not required for B2b.

### OQ7 — statistical power for the Pareto-region claim

(Surfaced in B1 reading.) The Phase A bootstrap reports CIs ~50 points
wide for ~25-point effects at N = 20 paired seeds. A "Bayesian occupies a
non-empty Pareto region" claim is a positional claim, not a magnitude
claim — it needs distinguishability against the agents that bound the
frontier from below and above. Under v1's seed budget, the CIs for
Bayesian vs. greedy were [+1.15, +51.10] for a +25.75 effect; for
Bayesian vs. Haiku they were [-306.10, -257.50] for −281.80. The lower
end of the CI matters: if it crosses zero against the agent that
dominates Bayesian on cost (e.g. some lower-cost ablation), the
Pareto-region claim is unsupported.

- **(a) Keep N = 20 paired seeds; widen the agent comparison field.** Cheap;
  may be insufficient if the relevant Pareto-frontier comparison has a
  small effect.
- **(b) Increase to N = 50–100 paired seeds.** Reduces CI width by ~√(50/20).
  More compute (especially for LLM agents — Haiku 4.5 cost $3.24 for
  N = 20).
- **(c) Pre-register the power calculation:** specify the agent pair and
  effect size that the test must distinguish, derive N from a pilot run.

(c) is the rigorous version. (b) is the safe version. The choice depends
on Guy's appetite for benchmarking compute and how high the bar is for
the Pareto-membership claim to be defensible.

> **Resolved (2026-05-31): N = 100 paired seeds for cheap agents,
> N = 50 for LLM agents.** Narrows the CIs enough for the positional
> Pareto-membership claim while keeping LLM API spend bounded. Recorded
> here for the B4/B5 seed budget; *not* load-bearing for B2b. Revisit
> against a pilot power calculation (option (c)) if the relevant
> frontier comparison turns out to have a small effect.

---

## 4a. Phase B2b resolutions and addenda (2026-05-31)

The B2b session (`paper1/category-inference`) resolves OQ1, OQ2, OQ3,
OQ5, and OQ7 as recorded inline above, and adds three notes.

**Classifier naming (paper-language correction).** The category-
inference component implemented in B2b is **Gaussian Naive Bayes on
sentence embeddings with a Dirichlet class prior** — a *generative*
model — not a *discriminative* multinomial logistic. Paper language
(and `credence.tex` §3.3, when written) must describe it as such. This
rename rides along with the Phase D rewrite of `credence.tex`; nothing
currently on master mis-names it.

**Reliability updates under category uncertainty — exact conditioning,
not MAP (revised 2026-05-31).** An earlier draft of this section proposed
collapsing the category posterior to its MAP value for the Bayesian
agent's reliability updates, on the grounds that `src/conjugate.jl`'s
Beta-Bernoulli `update` coerces evidence to unit pseudocounts. That is
**rejected**: MAP discards information the agent has paid for (cf.
`feedback_no_discard_posterior_info`) and weakens the paper's
principled-Bayesian claim — the substrate limitation is a reason to
extend the substrate, not to approximate. From first principles
(A3: learning is conditioning; Invariant 1: only `condition` changes
weights, and uncertainty is encoded in the hypothesis space so
`condition` can learn it), the agent carries the category uncertainty in
the reliability hypothesis space and updates by **exact conditioning**.
The soft posterior π enters **both** the decision (VOI marginalised over
categories) and the update. The exact posterior is a mixture
(`P(correct|θ)=Σ_c π_c θ_{t,c}` against a product-of-Betas prior is
non-conjugate → a K-way split per observation); its growth is a
*computational* cost handled by **metacomputation** (EU over
computational strategies — the spec's "On metareasoning"), not by a
bolted-on approximation. The mechanism, the substrate extension it needs,
and the depth of the metacomputation for Paper 1 are specified in the B2c
design doc (`docs/paper1/move-2c-design.md`). Both agents still receive
the identical soft posterior (OQ5); the agent's exact internal update is
not an asymmetry in the information surface.

**B2b execution split.** B2b lands as a split: **B2b.1–B2b.3**
(this master-plan addendum, the `category_inference` module, and LOO
calibration) ship in PR `paper1/category-inference`; **B2b.4–B2b.6**
(host + `llm_agent` wiring — "B2c") follow in a separate PR. The split
point is forced by the embedding gate: everything past B2b.3 needs real
question-bank embeddings, which the B2a design doc (§5) defers to a
later step, and which collide with the no-`sentence-transformers`
dependency constraint. B2b ships and tests on synthetic embeddings.

---

## 5. Out of scope

Explicit list. Drift to Paper 3, embeddings to Paper 3, and so on. If
something is tempting to slip in, it goes here instead.

- **Drift / non-stationarity** — Paper 3. Phase B keeps the v1 stationary
  reliability assumption; non-stationary tool reliability is the topic of
  Paper 3 ("Adaptive Bayesian Tool Reliability Tracking for Non-
  Stationary LLM Agents", `papers/PAPERS-STRATEGY.md` Paper 3).
- **Joint inference of category and tool-reliability** — Paper 3
  territory. The Phase B category inference is independent of the
  reliability matrix; learning category structure that *reduces* the
  reliability matrix's effective dimensionality (e.g. discovering that
  "factual" and "misconceptions" share a reliability stage for a
  particular tool) is CEG-shaped work and belongs in Paper 6 / Paper 3.
- **Embedding-features-into-reliability** — Paper 3 / Paper 6. If
  category inference produces embeddings as a side-effect, those
  embeddings do not feed back into `rel_betas[t, cat_idx]` indexing in
  Phase B. The reliability matrix stays category-indexed.
- **Information-geometric / cheap-design connections** — Paper 4.
- **Preference learning from implicit feedback / extended CIRL** — Paper 5.
- **CEG model selection / staged-tree connections** — Paper 6.
- **Programs-as-options / closed-loop policy framework** — Paper 7.
- **Paper 1 LaTeX rewrite** — Phase D, after B5. Phase B does not edit
  `papers/paper1/credence.tex`.
- **DSL primitive additions** — none expected. If category inference can
  be expressed in existing primitives (`condition`, `expect`, `voi`,
  `MixtureMeasure`, `CategoricalMeasure`, `Kernel`), B2 uses them. Adding
  primitives for B2 would be a violation of `feedback_dsl_optimization_invisible`
  and would warrant its own design discussion outside this branch.
- **Re-evaluating the v1 ablations** — Phase A is closed; Phase B does
  not relitigate `ablation_no_voi` / `ablation_no_learning` / `ablation_
  no_abstain` results. Those are recorded in `papers/paper1/bootstrap-
  results.md` and stand. Phase B introduces new conditions, not new
  ablations.

---

## 6. Done criteria

Phase B is finished when all of the following hold:

- B2, B3, B4, B5 implementation PRs all merged.
- All seven OQs (5 from the prompt + OQ6/OQ7 from B1 reading + any
  surfaced during B2–B5) resolved and recorded in this master plan.
- New benchmark numbers in `papers/RESULTS.md` reflecting fair conditions.
- Pareto plot generated and committed (location TBD by B5 design doc).
- Frontier-membership statement: "Bayesian occupies the Pareto frontier
  in region X / does not occupy a non-empty Pareto region" — explicit,
  verifiable from the new RESULTS.md tables.
- Updated paired-bootstrap results in `papers/paper1/bootstrap-results.md`.
- Phase D (paper LaTeX rewrite) ready to begin: B5's deliverables are
  sufficient material for the paper rewrite, and the rewrite plan can be
  scoped without further empirical work.

If after B5 the thesis is not supported (Bayesian does not occupy a non-
empty Pareto region under fair conditions), Paper 1 either reframes again
with the trajectory recorded honestly per §1.1, or reports the negative
result. Phase B does not paper over a failure to confirm.
