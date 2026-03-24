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
1. **Accuracy paradox**: Bayesian agent scores +112.6 vs LangChain ReAct's -8.0, despite LOWER accuracy (59.6% vs 63.7%). Higher accuracy + undisciplined querying = negative value.
2. **Prompting trap**: Enhanced LangChain with cost-awareness prompting scores WORST (-68.2). More sophisticated prompts make agents more expensive, not more economical. This empirically falsifies the hypothesis that prompting can substitute for decision theory.
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
