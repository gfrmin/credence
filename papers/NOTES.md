# Paper Notes for Claude Code

## What this paper is

An architecture paper presenting Credence, a Bayesian decision-theoretic framework for LLM agent tool selection. NOT merely a benchmark paper. The benchmark results are evidence for an architectural argument.

## Key framing decisions

### Positioning
- Credence is the FIRST framework to apply formal value-of-information (VOI) calculations to LLM tool selection
- Closest competitors: RAFA (Bayesian regret bounds, ICML 2024), DeLLMa (EU maximisation, ICLR 2025), MACLA (Beta posteriors for procedure selection, AAMAS 2026)
- None of these combines: per-tool beliefs + VOI computation + EU maximisation + posterior updating
- The gap is wide open — no peer-reviewed paper at a top venue does what Credence does

### Three headline results
1. **Accuracy paradox**: Bayesian agent scores +129.5 vs best LLM agent's +10.8, despite LOWER accuracy (62.6% vs 76.4%). The highest-accuracy agent scores 12× less. Higher accuracy + undisciplined querying = marginal value.
2. **Prompting trap**: Three LLM variants form a gradient — Bare (-160.5) → ReAct (-15.3) → ReAct+S+H (+10.8). Each prompting technique improves accuracy but never closes the score gap. This empirically falsifies the hypothesis that prompting can substitute for decision theory.
3. **Graceful degradation**: Under drift, single-best-tool collapses (-69.0 delta), Bayesian agent barely notices (-21.8 with forgetting). No change-detection heuristic needed.

### Connections to author's prior work
- Beta-Bernoulli reliability tracking is a specialisation of the Dirichlet-categorical conjugate framework from Freeman & Smith (2011a) CEG model selection
- Exponential forgetting is the power steady model from Freeman & Smith (2011b) dynamic staged trees
- Brain/body/environment decomposition follows Ay (2015) geometric design principles
- LLM treated as "prosthetic" (noisy sensor) — cheap design principle

### Tone
- Academic, not polemical. The blog posts ("The Bitter Lesson Has No Utility Function") are provocative; the paper should let the results speak.
- Do NOT use the subtitle "Or, Why Your AI Agent Is an Expensive Flowchart" — that was from the February draft and is too informal for arXiv

### Non-negotiable design principles (from the Credence spec)
- No exploration bonuses or loop detection hacks
- Ground truth for learning only from positive rewards
- Prior beliefs about action success = 1/N, not 0.5
- VOI > cost is the query criterion, NOT (VOI - cost) > EU
- Brain decides WHAT, body decides HOW. LLM is a prosthetic, not a decision-maker.

## TODO before arXiv submission

1. [ ] Fill in ablation table (Table 4) from results/RESULTS.md
2. [ ] Expand bibliography to ~35 references (see reference list below)
3. [ ] Rewrite Section 2 (Related Work) with three subsections
4. [ ] Optional: run LangChain experiments with frontier model (5 seeds, Claude Sonnet via API)
5. [ ] Verify every technical claim against the code
6. [ ] Final proofread and consistency check

## References to add

### Must-cite (reviewer will look for these)
- Liu et al. (2024) "Reason for Future, Act for Now" (RAFA), ICML 2024, arXiv 2309.17382
- Liu et al. (2025) "DeLLMa: Decision Making Under Uncertainty with LLMs", ICLR 2025, arXiv 2402.02392  
- Forouzandeh et al. (2026) "MACLA: Memory-Augmented Continual Learning Agent", AAMAS 2026, arXiv 2512.18950
- Kapoor et al. (2025) "AI Agents That Matter", TMLR 2025
- Schick et al. (2023) "Toolformer", NeurIPS 2023
- Shinn et al. (2023) "Reflexion", NeurIPS 2023
- Zhou et al. (2024) "LATS: Language Agent Tree Search", ICML 2024, arXiv 2310.04406
- arXiv 2601.06407 "Value of Information: A Framework for Human-Agent Communication"

### Should-cite (shows field awareness)
- arXiv 2602.11541 "INTENT: Budget-Constrained Agentic LLMs" (Feb 2026)
- Zheng et al. (2024) "BTP: Budget-Constrained Tool Learning with Planning", ACL 2024, arXiv 2402.15960
- De Sabbata et al. (2024) "Rational Metareasoning for LLMs", NeurIPS 2024 workshop, arXiv 2410.05563
- Wang et al. (2024) "Reasoning in Token Economies", EMNLP 2024, arXiv 2406.06461
- Erol et al. (2025) "Cost-of-Pass", arXiv 2504.13359
- Wang et al. (2025) "Theory of Agent", ICML 2025 position paper, arXiv 2506.00886
- Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
- Chen, Zaharia & Zou (2023) "How is ChatGPT's behavior changing over time?", arXiv 2307.09009
- Wang et al. (2023) LLM agent survey, arXiv 2308.11432
- Montúfar, Zahedi & Ay (2015) "A Theory of Cheap Control in Embodied Systems", PLOS Comp Bio
- Chaloner & Verdinelli (1995) "Bayesian Experimental Design: A Review", Statistical Science
- Ibrahim & Chen (2003) "On Optimality Properties of the Power Prior", JASA
- Smith & Freeman (2011) "Distributional Kalman Filters", J. Forecasting [YOUR OWN PAPER — currently missing]
- Harrison & Stevens (1976) "Bayesian Forecasting", JRSS-B
- Berger (1985) "Statistical Decision Theory and Bayesian Analysis" [better reference for complete class theorem than Wald 1950]

### Already cited — keep
- Ay (2015), Freeman & Smith (2011a, 2011b), Smith & Anderson (2008)
- Howard (1966), Raiffa & Schlaifer (1961), Russell & Wefald (1991)
- Savage (1954), Wald (1950), Solomonoff (1964)
- Hadfield-Menell et al. (2016), Sutton et al. (1999)
- West & Harrison (1997), Yao et al. (2023), Chase (2022)
- Pfeifer & Bongard (2006), Kaelbling et al. (1998), Boutilier et al. (1999), DeGroot (1984)

## arXiv submission details
- Primary: cs.AI
- Cross-list: cs.LG, cs.CL
- License: CC BY 4.0

---

## Phase B (May 2026): methodology pivot

Phase B is the second methodology re-framing of Paper 1 in short
succession. The trajectory is recorded here as deliberate epistemic
progress so it does not silently overwrite. Branch:
`paper1/methodology`. Master plan: `docs/paper1/master-plan.md`.

### Trajectory of three re-framings

**1. Original (Feb 2026 draft, now obsolete).** "Bayesian +129.5 vs best
LLM +10.8 — Bayesian dominates LLMs even under unequal conditions." This
framing is the one preserved in the "Three headline results" section
above (lines 16-19 of this file). Empirically invalidated by the March
2026 benchmark redesign (frontier-LLM comparison).

**2. Interim (March 2026 redesign).** "Bayesian is the principled, zero-
cost approach; frontier LLMs win on raw score but at API cost." Pivot
triggered by Haiku 4.5 +445.5 vs Bayesian +163.7 in the redesigned
benchmark — frontier LLMs answer 30+/50 questions from world knowledge
alone (`papers/RESULTS.md`). This framing is what `papers/CLAUDE.md`
"Updated headline results" currently reflects (lines 13-34). The bullet
above this section ("Three headline results") is *not* updated to this
framing — it preserves the original framing for historical record. Phase
D's paper rewrite will resolve the inconsistency by editing the LaTeX
source rather than these notes.

**3. Phase B (May 2026, current).** "Bayesian VOI tool selection
occupies a non-empty region of the cost-performance Pareto frontier
under fair conditions." Pivot triggered by the Phase A bootstrap result
(`papers/paper1/bootstrap-results.md`, 2026-05-04) and recognition of
the v1 fairness asymmetries. Phase B implements the new conditions in
Moves B2–B5 on branch `paper1/methodology`.

Each pivot was driven by a specific empirical observation that
invalidated a quantitative claim of the previous framing:
- Pivot 1 → 2: frontier-LLM dominance.
- Pivot 2 → 3: greedy-ablation result + v1 fairness asymmetries.

The trajectory is deliberate. Future revisions of Paper 1 should not
collapse the history into a single "current" framing without a
corresponding record of the empirical observation that drove the
collapse.

### Phase B pivot rationale

The v1 framing ("Bayesian is principled at zero cost") survives as far
as it goes — Bayesian is +163.7 at $0 cost and 0.016s/q; that is true
and remains a real cost-performance datum. But the framing under-claims
relative to the framework's actual value proposition. VOI tool selection
is not just "the best you can do for free" — under the conditions where
parametric world knowledge cannot substitute for tool selection, and
where category-routing is something the agent has to do rather than
something handed to it, the Bayesian framework is *the* principled
mechanism for the problem. Phase B's job is to find the conditions under
which that claim is empirically defensible, and report honestly if it
isn't.

### Phase B4 outcome (2026-06): the bet was refuted — and that is the result

Phase B's bet (above) was that fair conditions would vindicate VOI: remove the
given-category advantage and the marginal value of VOI's tool-by-tool reasoning
would show. **The data refuted it.** Under inferred categories the greedy>VOI gap
*widened* (+25.7 → +39.2) — VOI is more sensitive to category-inference noise
than greedy. The honest, stronger result is the inversion: cost-efficiency is
earned by the **belief substrate** (reliability learning + category inference),
not the VOI action layer. Myopic VOI is a Bayesian *learner*, not an *explorer*,
and loses to optimistic-greedy at this horizon and mix; gating experiments cap
any action-policy gain at ≤+16 over greedy (known-θ ceiling 306, reachable ~205,
horizon-locked), while the inference lever is 40–53. Thesis broadened to
"Bayesian tool selection," VOI scoped to its cheap-and-dominant-tool niche;
genre = analysis/architecture (arXiv cs.AI). This is Pivot 3 → 4, driven (per the
discipline above) by the specific observation that fair conditions *widened*
rather than closed the greedy–VOI gap. Full locked argument + tables:
`papers/RESULTS.md`; OQ5 reversed to no-π-injection (master-plan B4/B5 section).

### Greedy ablation result interpretation

Phase A's bootstrap (`papers/paper1/bootstrap-results.md`, 2026-05-04):
Greedy − Bayesian Δ = +25.75, 95% CI [+1.15, +51.10], p = 0.0386, paired
N = 20.

**Honest framing:**
- Marginally significant. p = 0.0386 is one chance in twenty-six of a
  false positive at the 0.05 threshold. Lower CI bound +1.15 — barely
  positive. Magnitude is uncertain even where the sign is.
- *Expected* under v1 conditions, not surprising. When the category is
  given (the v1 setup), the marginal value of VOI's tool-by-tool
  exploration shrinks: the reliability matrix already concentrates on
  the right tool for this category, and abstention can hurt when the
  best-tool mean reliability is already high enough that submission EU
  is positive on average. The greedy ablation exploits exactly this
  structure.
- The framework's value emerges under fair conditions (Phase B's
  thesis), not under v1 conditions. The v1 result does not refute the
  framework; it refutes the v1 *evaluation*.
- Avoid dishonest framings: "Bayesian was wrong" (no — under different
  conditions it isn't), "the bootstrap was underpowered" (CI is wide but
  the test was correctly specified), "greedy is just an artifact"
  (`precedent:baseline-comparison` sanctions greedy as a deliberate
  non-Bayesian baseline; it is not an artifact).

### Amin (2026) and MACLA (2026) positioning

**MACLA (Forouzandeh et al., AAMAS 2026, arXiv 2512.18950)** — already
documented in this file's Positioning section (top of file) and Section 2
rewrite plan (`papers/CLAUDE.md` §2.1). The single most relevant
Bayesian mechanism in the LLM-agent literature: Beta posteriors for
procedure selection. Phase B does not change MACLA's positioning — the
gap remains "MACLA selects multi-step procedures, not individual tools,
and does not compute VOI." Phase B's category-inference component does
not encroach on MACLA's procedure-selection territory.

**Amin (2026)** — flagged in earlier conversation as a positioning
concern; specific reference and concern details not yet recorded in this
file. Placeholder for Guy to expand at next opportunity. If Amin (2026)
is the same memory cited in earlier Posture/Paper-1 sessions, it likely
warrants its own subsection here once the citation surfaces. Not load-
bearing for Phase B's methodology design — Phase B can proceed without
this resolved — but should be in place before Phase D (paper rewrite).

### Phase B scope and out-of-scope

**Modelling notes for the write-up (B2b, 2026-05-31; revised).** The
category-inference component (B2) is **Gaussian Naive Bayes on sentence
embeddings with a Dirichlet class prior** — a generative model, not a
discriminative multinomial logistic; describe it as such in §3. Both
agents receive the *identical soft category posterior*. The Bayesian
agent then updates tool reliability by **exact conditioning under
category uncertainty** — the full posterior enters both the decision
(VOI marginalised over categories) and the update — **not** a MAP
collapse (an earlier draft proposed MAP; rejected, since it discards
information the agent paid for and weakens the principled-Bayesian
claim). The exact posterior is a mixture; its growth is handled by
metacomputation (EU over computational strategies), per the spec's "On
metareasoning". **Resolved:** the runtime update is the full-posterior-
weighted (resource-rational) one — credit each category by π_c via a new
`WeightedBernoulli` conjugate family — which uses the whole posterior and
tracks the exact mixture to third-decimal accuracy. Mechanism + substrate
in the B2c design doc (`docs/paper1/move-2c-design.md`); see also
`master-plan.md` §4a.

In scope (this branch):
- Category inference (B2)
- Tools-only slice (B3)
- Fairness-equalised LLM prompting (B4)
- Re-run benchmark + Pareto analysis (B5)

Out of scope (deferred to other Papers / Phase D):
- Drift / non-stationarity → Paper 3
- Joint category+reliability inference → Paper 3 / Paper 6
- Embedding-into-reliability → Paper 3 / Paper 6
- Information-geometric / cheap-design connections → Paper 4
- CIRL / preference learning → Paper 5
- CEG / staged-tree connections → Paper 6
- Programs-as-options → Paper 7
- LaTeX rewrite → Phase D
- DSL primitive additions → not expected; if category inference cannot
  be expressed in existing primitives that is itself a finding worth
  separate discussion

See `docs/paper1/master-plan.md` §5 for the canonical out-of-scope list.
