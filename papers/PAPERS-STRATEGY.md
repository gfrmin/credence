# Credence Paper Programme: Instructions for Claude Code

## Context

Guy has a single 14-page arXiv draft (`credence-paper.tex` or similar) that currently tries to be four papers at once. The task is to:

1. **Strip Paper 1 down** to a focused, publication-ready flagship
2. **Extract material** into stub files for Papers 2–4, each with the relevant text, references, and notes on what new work is needed
3. **Document Papers 5–7** as future stubs only (no material yet)

The current draft is called "Credence: A Bayesian Decision-Theoretic Framework for LLM Agent Tool Selection" and contains results from benchmark experiments (stationary, drift, ablation), connections to chain event graphs, Ay's embodied agent framework, CIRL alignment, and the accuracy paradox.

---

## The Seven Papers

### Paper 1: "Credence: Bayesian Decision-Theoretic Tool Selection for LLM Agents"
**Status:** Draft exists. Needs trimming, not expansion.
**Target:** arXiv cs.AI (cross-list cs.LG, cs.CL) April 2026. Venue: AISTATS 2027 or UAI 2027.
**Length:** 10 pages + references.

**The one sentence:** VOI-based tool selection, derived from first principles, outperforms LLM-prompted reasoning because decision theory is not a prompting problem.

**What stays:**
- The five axioms (§3.1) — these are architecturally load-bearing
- The belief model (§3.3) — Beta-Bernoulli, conjugate updating, initialisation
- The VOI decision rule (§3.4) — the core contribution
- Category inference paragraph (currently in §3.4)
- Experimental setup (§4) — all of it
- Stationary results (§5.1, Table 2) — the main result
- Ablation study (§5.3, Table 4) — shows it's not one trick
- The "accuracy paradox" discussion (§6 first two paragraphs)
- The "prompting trap" analysis (§5.1 final paragraphs) — the three LLM variants gradient
- Proposition 1 (VOI non-increasing) — but tighten the proof or demote to "Observation"
- Related work (§2) — keep the full competitive landscape (RAFA, DeLLMa, MACLA, INTENT, BTP, Kapoor, De Sabbata, Wang et al. 2025). This is where you establish the gap.
- Brief future work pointing to the broader Credence architecture

**What gets CUT from Paper 1 (moved to other papers):**

1. **CEG material → Paper 3.** Remove all references to chain event graphs, staged trees, florets, Freeman & Smith 2011a. Remove the paragraph in §2.3 connecting Beta-Bernoulli to Dirichlet-categorical CEG framework. Remove the §6 paragraph about "structurally analogous to stage parameters in a chain event graph." Keep Freeman & Smith 2011b ONLY as a citation for the exponential forgetting mechanism (one sentence).

2. **Drift experiment → Paper 3.** Remove Table 3 and §5.2 entirely. Replace with a single paragraph: "In supplementary experiments, we verify that the Bayesian updating mechanism provides natural robustness to distributional drift without explicit change detection; full results will appear in [Paper 3 citation]." If Paper 3 isn't yet on arXiv, just say "detailed drift analysis is deferred to future work."

3. **Ay's embodied agent framework → Paper 4.** Remove the two paragraphs in §6 discussing cheap design, embodied universal approximation, the kernel correspondence, the feature vector analogy. Replace with ONE sentence in the discussion: "The computational asymmetry — 0.07s per question for the decision layer vs. 5.65s for the LLM agent — instantiates the cheap design principle of Ay [2015], wherein a controller need not be a universal approximator if it is sufficient for the embodiment constraints; we formalise this connection in forthcoming work." Keep Ay [2015] in the bibliography. Remove Montúfar et al. [2015] and Pfeifer & Bongard [2006] — they belong in Paper 4.

4. **Extended CIRL/alignment discussion → Paper 5.** Remove the future work paragraph about preference learning from implicit feedback. The CIRL axiom (Axiom 5) stays because it defines the objective. The speculative material about what it enables goes.

5. **Dynamic staged tree connections → Paper 3.** Remove references to power steady model, distributional Kalman filtering, Smith & Freeman 2011, Harrison & Stevens 1976, Ibrahim & Chen 2003, Adams & MacKay 2007. These all belong in Paper 3 where the forgetting mechanism is the main contribution.

6. **The "brain/body/environment" §3.2 framing.** Keep it but shorten. Currently it's half a page establishing the Ay connection. Reduce to one short paragraph that says: "We decompose the agent into a decision layer (maintains beliefs, selects actions), tools including the LLM (noisy information channels), and the environment (questions and ground truth). The LLM is treated as a sensor, not a decision-maker." Drop the Markov kernel notation β : W → ΔS — that's Paper 4's job.

**Known issues to fix:**
- **Tables 2 and 4 are inconsistent.** Table 2 shows Credence at +129.5; Table 4 shows "Full Credence" at +112.6. Either re-run the ablation with the same seeds as the main experiment, or add a footnote explaining they are from different runs. Reviewers WILL notice.
- **Oracle agent** is described in §4.2 but absent from Table 2. Either add it or remove it from §4.2.
- **Proposition 1 proof** is hand-wavy. The asymptotic argument (as α+β → ∞) establishes the limit but not monotonicity. Either: (a) prove monotonicity properly for the Beta-Bernoulli case (it follows from the variance formula), or (b) weaken to "Observation 1" and state it as an empirical property verified in experiments.
- **Abstract** claims "+129.5 points compared to +10.8 for the best LLM agent" — verify this matches Table 2 exactly after any re-runs.

**References after trimming (target ~25):**
Keep: Ay 2015, Berger 1985, Boutilier et al. 1999, Chaloner & Verdinelli 1995, Chase 2022, Chen et al. 2023, De Sabbata et al. 2024, DeGroot 1984, Dong et al. 2026, Erol et al. 2025, Forouzandeh et al. 2026, Hadfield-Menell et al. 2016, Howard 1966, Kaelbling et al. 1998, Kapoor et al. 2025, Liu et al. 2024 (RAFA), Liu et al. 2025 (DeLLMa), Liu et al. 2026 (INTENT), Raiffa & Schlaifer 1961, Russell & Wefald 1991, Savage 1954, Schick et al. 2023, Shinn et al. 2023, Solomonoff 1964, Sutton 2019, Wald 1950, Wang et al. 2023, Wang et al. 2024, Wang et al. 2025, Yao et al. 2023, Zheng et al. 2024, Zhou et al. 2024.
Remove: Freeman & Smith 2011a, Harrison & Stevens 1976, Ibrahim & Chen 2003, Adams & MacKay 2007, Smith & Freeman 2011, Montúfar et al. 2015, Pfeifer & Bongard 2006, Smith & Anderson 2008, Sutton et al. 1999, Thwaites et al. 2009.
Keep Freeman & Smith 2011b ONLY if the one-sentence forgetting mention remains.

---

### Paper 2: "The Accuracy Paradox in Tool-Using LLM Agents"
**Status:** Material exists in Paper 1. Needs its own framing.
**Target:** arXiv May 2026. Venue: EMNLP 2026 or AAAI 2027.
**Audience:** Agent evaluation community, not Bayesian methods community.

**Core claim:** Optimising agent accuracy without accounting for tool costs is provably suboptimal. We demonstrate this empirically with three LLM agent variants that form a monotone gradient: more sophisticated prompting → higher accuracy → lower net value.

**Material to extract from current Paper 1:**
- The accuracy paradox discussion (§6 first paragraph, eq. 9)
- The prompting trap analysis (§5.1 final paragraphs)
- Table 2 (stationary results)
- The three LLM variant comparison (§4.2 LLM agents description)

**New material needed:**
- Frame against accuracy paradox literature: Chen et al. 2024 (RLHF), Bedi et al. 2025 (clinical), Intervention Paradox (arXiv 2602.03338), Reasoning in Token Economies (EMNLP 2024)
- Formal proof that accuracy-maximisation ≠ utility-maximisation when costs > 0 (straightforward)
- Pareto frontier analysis (accuracy vs. score) — plot all agents on this
- Ideally: run at least one frontier LLM (GPT-4o or Claude Sonnet) to show the paradox persists even with much better perception. This pre-empts the main criticism.
- Discussion of implications for agent benchmarks (connect to Kapoor et al. 2025)

**This paper cites Paper 1** for the framework and experimental setup, but contributes to a different literature (agent evaluation, not agent architecture).

---

### Paper 3: "Adaptive Bayesian Tool Reliability Tracking for Non-Stationary LLM Agents"
**Status:** Seed material in Paper 1's drift experiment. Needs substantial expansion.
**Target:** arXiv August 2026. Venue: AISTATS 2027 or Bayesian Analysis.
**Audience:** Bayesian methods community, applied statistics.

**Core claim:** LLM APIs are empirically non-stationary (cite Chen et al. 2023). Classical Bayesian forgetting mechanisms — power priors, exponential discounting, changepoint detection — provide principled solutions that current agent frameworks lack entirely.

**Material to extract from current Paper 1:**
- §3.3 non-stationarity paragraph and eq. (4)
- §5.2 drift experiment (Table 3)
- §6 paragraph on dynamic staged trees connection
- All references to: Freeman & Smith 2011a,b, Smith & Freeman 2011, Harrison & Stevens 1976, Ibrahim & Chen 2003, Adams & MacKay 2007, West & Harrison 1997

**New material needed:**
- Extended drift experiments: multiple drift types (gradual, sudden, cyclic, correlated across tools)
- Comparison of forgetting mechanisms: fixed λ vs. learned λ vs. BOCD (Adams & MacKay) vs. BAM (Nassar et al. 2022)
- Learn λ online (this is the Class 2 multi-process model from Guy's thesis — the principled version)
- Explicit connection to CEG stage learning: each tool-category pair = a floret, Beta prior is the natural conjugate, AHC algorithm discovers which tool-category pairs share reliability stages
- Connection to power steady model (Freeman & Smith 2011b) and distributional Kalman filtering (Smith & Freeman 2011)
- Empirical evidence of real API non-stationarity (Chen et al. 2023, plus the 2026 study showing daily/weekly periodic patterns)

**This paper cites Paper 1** for the base framework, then extends the belief model to handle non-stationarity rigorously.

---

### Paper 4: "Cheap Design for LLM Agents: An Information-Geometric Perspective"
**Status:** Theoretical. Needs mathematical development.
**Target:** arXiv October 2026. Venue: UAI 2027 or Entropy journal.
**Audience:** Theoretical AI, information geometry.

**Core claim:** The Credence decision layer is an embodied universal approximator in the sense of Ay [2015] for the constraint set defined by tool reliability profiles, achieving the full behavioural repertoire with exponentially fewer parameters than a universal approximator (the LLM).

**Material to extract from current Paper 1:**
- §3.2 brain/body/environment decomposition (expand, don't trim)
- §6 paragraphs on cheap design, Ay's framework, feature vector analogy
- References: Ay 2015, Montúfar et al. 2015, Pfeifer & Bongard 2006

**New material needed (primarily theoretical):**
- Explicit kernel correspondence: map Credence components onto Ay's (β, π, α, φ) Markov kernels
- Define the constraint set Σ for tool reliability profiles
- State and prove: "The Beta-posterior VOI controller is a sufficient model (Ay Definition 1) for Σ"
- Connect to Ay's Proposition 1: the exponential family parameterisation maps onto softmax tool selection; the feature vectors are tool-category reliability profiles; dimensionality reduction from |C|·|A| to T×C×2
- Connect minimum-entropy result (Ay eq. 17) to empirical finding that Credence uses fewer tools
- Discuss controlled RBM (Ay Proposition 4) as suggestive for the full Credence DSL architecture: expression trees as hidden nodes, CRP clustering as learning interaction structure
- Address limits: Ay assumes finite state sets; Credence has continuous belief states. Ay doesn't address learning Σ; Credence learns it online.

**Stretch goal:** Contact Ay, Montúfar, or Zahedi about collaboration. This is their framework applied to a novel domain.

---

### Paper 5: "CIRL Alignment-by-Construction for Bayesian Agents" (DEFERRED)
**Status:** Axiom 5 exists. No experiments.
**Needs:** Experiments showing CIRL axiom produces different (better) behaviour than non-CIRL agent in adversarial or misalignment scenarios. Preference learning from implicit feedback (thumbs up/down in Telegram training loop). Probably depends on the full Julia implementation being more mature.
**Earliest realistic target:** 2027.

### Paper 6: "Chain Event Graphs for Tool Reliability Discovery" (DEFERRED)
**Status:** Theoretical connection identified. No implementation.
**Needs:** Implement AHC algorithm to discover tool-category stages from execution data. Show that automatic stage discovery recovers the hand-specified category structure. Ideal collaboration with Warwick CEG group (Leonelli, Shenvi, or Jim Smith).
**Earliest realistic target:** 2027.

### Paper 7: "Programs as Options: Closed-Loop Policies for Bayesian Agents" (DEFERRED)
**Status:** Julia implementation underway (grid world: 10 tests, email agent: 28 tests).
**Needs:** Convincing real-world domain evaluation beyond grid world. The email agent with polling execution, per-step conditioning, and meta-actions is the prototype. Connect to Sutton et al. 1999 options framework formally.
**Earliest realistic target:** Late 2027.

---

## Immediate Task: Refactoring the Current Draft

### Step 1: Create paper stubs
Create four directories: `paper1/`, `paper2/`, `paper3/`, `paper4/`. Copy the current draft into `paper1/`. Create stub `.tex` files in `paper2/`, `paper3/`, `paper4/` containing:
- A title
- An empty abstract with a one-line comment summarising the claim
- A `% EXTRACTED FROM PAPER 1:` section with the relevant paragraphs copied verbatim
- A `% NEW MATERIAL NEEDED:` section listing what's missing
- A bibliography with the relevant references

### Step 2: Strip Paper 1
Working in `paper1/`, make the cuts listed above under "What gets CUT from Paper 1." After each cut, leave a `% [Moved to Paper N]` comment so the provenance is clear.

### Step 3: Fix known issues
- Reconcile Tables 2 and 4 (check results files, re-run ablation if needed)
- Add Oracle to Table 2 or remove from §4.2
- Tighten Proposition 1 or demote to Observation
- Verify abstract numbers match tables

### Step 4: Final check
Paper 1 should be ~10 pages, ~25 references, with a clean arc:
Problem → Framework (axioms, VOI) → Experiment (accuracy paradox, prompting trap, ablation) → Conclusion

No CEGs. No staged trees. No information geometry. No drift tables. No Markov kernel notation. One sentence on cheap design. One sentence on forgetting. The CIRL axiom stays because it defines the objective, but no speculation about what it enables.

---

## Design Principles (Non-Negotiable)

These apply to ALL papers in the programme:

- All agent behaviour derived from first principles, never engineered
- No exploration bonuses, no loop detection, no ad-hoc heuristics
- Ground truth for sensor learning based on positive rewards only
- Prior beliefs about action success = 1/N, not 0.5
- VOI > cost, not (VOI - cost) > EU
- Brain decides WHAT; body decides HOW
- LLM = prosthetic (noisy sensor), never decision-maker
- No silent fallbacks (EU failures are bugs)
- Each paper is a quantum of knowledge: one clear claim, proven, with implications stated
