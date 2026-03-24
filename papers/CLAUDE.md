# CLAUDE.md — Paper 1 Instructions

## Context

You are helping co-author an academic paper: "Credence: Bayesian Decision-Theoretic Tool Selection for LLM Agents." The LaTeX source is at `paper/credence.tex`. Framing decisions, reference lists, and positioning notes are at `paper/NOTES.md`. Read NOTES.md thoroughly before making any changes to the paper.

The author is Guy Freeman (guy@gfrm.in), an independent researcher with a PhD in Statistics from Warwick (Bayesian graphical models, chain event graphs) and 10+ years of production ML experience. He is currently on contract at Booking.com designing agentic flows.

## The paper's argument

This is an **architecture paper**, not merely a benchmark paper. The benchmark results are evidence for a deeper claim: that LLM tool selection is fundamentally a decision-theoretic problem, not a perception problem, and that principled Bayesian methods outperform prompting-based approaches for structural reasons that no amount of prompt engineering can overcome.

Three headline results:
1. **Accuracy paradox**: +112.6 vs -8.0 despite lower accuracy (59.6% vs 63.7%)
2. **Prompting trap**: Enhanced LangChain scores worst (-68.2) — cost-awareness prompting backfires
3. **Graceful degradation**: Bayesian agent absorbs drift (-21.8 delta) while single-best collapses (-69.0)

## What needs doing (in priority order)

### 1. Fill in the ablation table

Table 4 (`\label{tab:ablation}`) has placeholder values. Extract exact numbers from `results/RESULTS.md` (or run `python run_ablation.py` if needed). The table should show:

| Variant | Score | Δ vs Full | Effect |
|---------|-------|-----------|--------|
| Full Credence | +112.6 | — | — |
| No VOI (always query) | [exact] | [exact] | [describe] |
| No categories (flat prior) | [exact] | [exact] | [describe] |
| No abstention | [exact] | [exact] | [describe] |
| Fixed reliability (p=0.5) | [exact] | [exact] | [describe] |

After filling in numbers, add 1-2 sentences of analysis after the table discussing which component contributes most.

### 2. Expand the bibliography

The paper currently has ~20 references. It needs ~35-40. Add the references listed under "Must-cite" and "Should-cite" in `paper/NOTES.md`.

For each new reference:
- Add a `\bibitem` entry in the `thebibliography` environment (the paper uses hand-written bibs, not .bib files)
- Use the same formatting conventions as existing entries
- Verify author names, year, venue, and title are correct — do NOT hallucinate citation details. If unsure, leave a `% TODO: verify` comment.

### 3. Rewrite Section 2 (Related Work)

The current Section 2 is too thin. Restructure it into three subsections:

**§2.1 Principled frameworks for LLM agents**
Position against:
- RAFA (Liu et al., ICML 2024): Only provably principled LLM agent framework. Bayesian adaptive MDPs with √T regret. But addresses planning in unknown environments, not per-tool reliability or VOI for individual tool calls.
- DeLLMa (Liu et al., ICLR 2025): EU maximisation for LLM decisions. Closest in spirit to Credence's EU framework. But no belief updating, no VOI, no tool reliability tracking.
- MACLA (Forouzandeh et al., AAMAS 2026): Uses Beta posteriors for procedure selection — the single most relevant Bayesian mechanism. But selects multi-step procedures, not individual tools, and does not compute VOI.

The key sentence: "To our knowledge, no published work combines per-tool Bayesian belief tracking, value-of-information computation, and expected utility maximisation for LLM tool selection."

**§2.2 Cost-aware agent approaches**
Cover the five tiers of increasing formality:
- Prompt-based: expose budget in prompt (limited effectiveness — our Enhanced LangChain result shows this fails)
- RL reward shaping: penalise tool calls in reward
- Planning/optimisation: BTP (Zheng et al., ACL 2024), SayCanPay
- Sequential decision-making: INTENT (arXiv 2602.11541) — the most formally rigorous
- Decision-theoretic: Credence (this paper) — the first to compute VOI

Also cite: AI Agents That Matter (Kapoor et al., TMLR 2025), Reasoning in Token Economies (Wang et al., EMNLP 2024), Cost-of-Pass (Erol et al., 2025), Theory of Agent position paper (Wang et al., ICML 2025).

**§2.3 Theoretical foundations** (keep shorter — this sets up §3)
- Embodied agent design: Ay (2015), Montúfar et al. (2015) — cheap design principle
- Graphical models: Freeman & Smith (2011a, 2011b) — CEG model selection and dynamic staged trees
- VOI: Howard (1966), Russell & Wefald (1991), the VOI for Human-Agent Communication paper (2601.06407)
- CIRL: Hadfield-Menell et al. (2016)

### 4. Other improvements

- Remove the subtitle "Or, Why Your AI Agent Is an Expensive Flowchart" if it's still there — too informal for arXiv
- Add Smith & Freeman (2011) "Distributional Kalman Filters" J. Forecasting to the bibliography and cite it in the non-stationarity discussion alongside West & Harrison (1997)
- In the Limitations paragraph, add: "Running the LangChain agents with frontier models (GPT-4o, Claude Sonnet) would likely improve their accuracy but would not address the structural inability to compute value of information — the prompting trap result already demonstrates this with the Enhanced variant."
- Verify the `\date` is set to `\today` or a specific month/year

## What NOT to do

- Do NOT change the five axioms or the mathematical framework in §3. These are settled.
- Do NOT add exploration bonuses, loop detection, or any "pragmatic" modifications to the algorithm description. These are explicitly forbidden by design.
- Do NOT change the scoring system or experimental setup — these are fixed from the benchmark.
- Do NOT weaken claims. The paper says "no published work" combines these elements — the landscape analysis confirms this is true as of March 2026. State it confidently.
- Do NOT make the paper longer than 12-13 pages (excluding references). Concision matters.
- Do NOT convert to a .bib file — keep the hand-written `thebibliography`. arXiv handles self-contained .tex files cleanly.

## Compilation

```bash
cd paper/
pdflatex -interaction=nonstopmode credence.tex
pdflatex -interaction=nonstopmode credence.tex  # twice for references
```

Check for warnings about undefined citations after adding new `\bibitem` entries.

## Style notes

- British spelling throughout (behaviour, formalise, maximise, colour)
- Use `\credence{}` macro for the framework name (renders as smallcaps CREDENCE)
- Equations are numbered; refer to them with `\cref{eq:...}`
- Tables use booktabs (toprule, midrule, bottomrule)
- The tone is measured and academic — let the results do the talking
