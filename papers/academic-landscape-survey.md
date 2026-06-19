# Credence occupies a wide-open space in LLM agent theory

**The academic landscape reveals that formal Bayesian decision theory has barely been applied to LLM agent tool selection, despite extensive work on heuristic and engineering-driven agent frameworks.** Credence's combination of Beta-Bernoulli belief tracking, value-of-information (VOI) calculations, expected utility maximisation, and axiomatic derivation has no peer-reviewed counterpart. The framework sits at the intersection of at least six distinct research gaps, each large enough to sustain a standalone publication. Below is a systematic assessment of each research area, identifying the closest existing work, the specific gap Credence fills, and the publication opportunities.

---

## 1. No peer-reviewed paper applies formal Bayesian decision theory to LLM tool selection

The most important finding across all ten research areas is simple: **no published paper at a top venue (NeurIPS, ICML, ICLR, AISTATS, UAI, AAAI) applies formal VOI-based Bayesian decision theory to LLM agent tool selection.** The field overwhelmingly relies on prompt engineering, RL reward shaping, or MCTS-style planning.

**Closest existing work:**

- **MACLA** (Forouzandeh et al., AAMAS 2026, arXiv 2512.18950): Maintains **Beta posteriors** over procedure success rates and uses expected-utility scoring for procedure selection in LLM agents. Bayesian selection is the single most impactful ablation (−7.8% seen, −9.1% unseen when removed). This is the closest peer-reviewed paper, but it selects *procedures* (multi-step skill sequences), not individual tools, and does not compute VOI.
- **Thompson Sampling for life sciences agents** (arXiv 2512.03065): Uses Beta-Bernoulli conjugate priors with Thompson Sampling for tool selection in a domain-specific agent, achieving 15–30% improvement over baselines. This is a bandit approach, not full decision-theoretic VOI.
- **RAFA** (Liu et al., ICML 2024, arXiv 2309.17382): The most principled LLM agent framework to date — casts agent reasoning as Bayesian adaptive MDPs and proves a **√T regret bound**. However, RAFA addresses planning in unknown environments, not per-tool reliability tracking or VOI for individual tool calls.
- **DeLLMa** (Liu et al., ICLR 2025, arXiv 2402.02392): Applies classical expected utility maximisation to LLM decision-making via multi-step inference (identify factors, enumerate states, forecast probabilities, maximise EU). This is the closest in spirit to Credence's EU framework but targets general decisions, not tool selection, and lacks belief updating or VOI.

**Gap Credence fills:** The unique combination of (a) per-tool Beta-Bernoulli reliability tracking, (b) VOI computation before each tool query, (c) EU maximisation for action selection, and (d) posterior updating from tool responses has **zero counterpart** in the peer-reviewed literature. MACLA demonstrates the value of Bayesian selection as an ablation; Credence provides the complete decision-theoretic architecture.

**Publishability:** This is clearly a **flagship paper** — a full venue paper at NeurIPS, ICML, or ICLR. The core Bayesian tool selection framework with VOI is the central contribution.

---

## 2. Cost-aware frameworks are abundant but almost none are decision-theoretic

Over **25 papers** explicitly model costs in LLM agent tool use, spanning five tiers of increasing formality:

| Tier | Approach | Examples |
|------|----------|----------|
| Prompt-based | Expose budget in prompt | BATS (arXiv 2511.17006), BCAS (arXiv 2603.08877) |
| RL reward shaping | Penalise tool calls in reward | OTC-PO (arXiv 2504.14870), CATP-LLM (ICCV 2025) |
| Planning/optimisation | Pre-plan under budget | BTP (ACL 2024, arXiv 2402.15960), SayCanPay (AAAI 2024) |
| Sequential decision-making | Online planning with cost anticipation | INTENT (arXiv 2602.11541), BAVT (arXiv 2603.12634) |
| Decision-theoretic/Bayesian | VOI computation, posterior updating | HAL paper (hal-05480691), CaMVo (NeurIPS 2025) |

**Key finding:** The most formally rigorous cost-aware paper is **INTENT** (Liu et al., Feb 2026), which formalises budget-constrained tool use as sequential decision-making with priced, stochastic tools. It uses a learned world model for planning — principled but not Bayesian. **BTP** (Zheng et al., ACL 2024) uses dynamic programming for budget allocation. Neither maintains posterior beliefs or computes VOI.

The **Theory of Agent (ToA)** position paper (Wang et al., ICML 2025, arXiv 2506.00886) argues that agents should invoke tools "only when epistemically necessary" and defines a knowledge boundary concept, but provides no computational mechanism. A Bayesian VOI framework provides exactly the mechanism ToA calls for.

**Gap Credence fills:** No existing system **jointly models** (a) uncertainty over the current answer, (b) expected information gain from each available tool, (c) monetary cost of the tool call, and (d) opportunity cost of budget spent. The entire cost-aware literature uses either learned models, heuristic rewards, or prompt-level awareness. The decision-theoretic approach — computing VOI before each call and acting only when VOI exceeds cost — is essentially unstudied in peer-reviewed work.

**Publishability:** The cost-benefit analysis component is a strong standalone contribution, especially paired with the finding that prompt-based cost awareness fails. A **systems paper** comparing Credence's principled approach against the prompt-based (BATS), RL-based (OTC-PO), and planning-based (BTP, INTENT) approaches would be compelling for AAAI or an agent workshop.

---

## 3. VOI has been applied to LLM agents only for communication, not tool selection

The classical VOI literature (Howard 1966; Raiffa & Schlaifer 1961; Russell & Wefald 1991) is deep and well-established. Modern Bayesian experimental design (Rainforth et al., 2024) and active learning (BALD) implement VOI principles under different names. The critical question is whether VOI has reached LLM agents.

**Three papers apply formal VOI or VOI-adjacent concepts to LLMs:**

- **VOI for Human-Agent Communication** (arXiv 2601.06407, Jan 2026): The first paper to directly apply formal VOI to LLM agents. Agents compute expected utility gain of asking clarifying questions vs. communication cost. Tested across 4 domains with strong results. **This is the single most relevant VOI paper** but addresses the ask-vs-act decision, not tool selection.
- **Rational Metareasoning for LLMs** (De Sabbata, Sumers & Griffiths, NeurIPS 2024 Workshop, arXiv 2410.05563): Applies Value of Computation (VOC) to LLM reasoning chains. Achieves 20–37% token reduction. Addresses *internal* computation ("should the LLM think more?"), not *external* tool calls.
- **Uncertainty of Thoughts (UoT)** (Hu et al., NeurIPS 2024): Uses information-gain rewards for LLM question-asking via uncertainty-aware simulation. Semi-formal VOI; acknowledges Bayesian updating would be theoretically ideal but doesn't implement it.

**Gap Credence fills:** No paper applies formal VOI to the specific decision of **whether to call a tool/API**. The ask-vs-act paper (2601.06407) provides the closest template but addresses human communication cost, not tool API cost. Credence's VOI computation — "will the expected EU improvement from this tool's response exceed its cost?" — is distinct and novel.

**Publishability:** A paper specifically on "Value of Information for LLM Agent Tool Selection" bridging classical VOI theory, modern BED, and practical LLM agent evaluation could target **UAI or AISTATS** (more theory-friendly venues) or the **Bayesian Analysis** journal.

---

## 4. The accuracy paradox has close relatives but Credence's version is novel

Several papers identify paradoxes structurally related to Credence's finding that **59.6% accuracy with ~1.1 tool calls beats 63.7% accuracy with ~3.22 tool calls**:

- **"The Accuracy Paradox in RLHF"** (Chen et al., 2024, arXiv 2410.06554) formally names the "accuracy paradox" — moderately accurate reward models outperform highly accurate ones for language model training. The formal structure is identical (higher component accuracy → worse system performance) but the domain differs.
- **"The Optimization Paradox in Clinical AI"** (Bedi et al., 2025, arXiv 2506.06574) shows 85.5% component accuracy yielded only 67.7% diagnostic accuracy in multi-agent clinical systems, significantly below a system with lower component accuracy.
- **"The Intervention Paradox"** (arXiv 2602.03338) demonstrates that LLM critics with high offline accuracy can **degrade** agent performance by up to 26 percentage points when deployed as mid-execution interventions.
- **"AI Agents That Matter"** (Kapoor et al., 2024, TMLR 2025) is the most influential paper in this space. It demonstrates that accuracy-only evaluation is misleading and proposes accuracy-cost Pareto analysis, but does not show that lower accuracy yields higher net value.
- **"Reasoning in Token Economies"** (Wang et al., EMNLP 2024) shows that complex reasoning strategies (Multi-Agent Debate, Reflexion) **become worse with more compute budget** — a direct demonstration that more resources harm performance.

**Gap Credence fills:** The *specific* accuracy paradox for tool-using agents — where a principled decision to abstain from tool use produces higher net value than an agent with higher accuracy but undisciplined tool use — has **not been formally identified or characterised**. Moreover, Credence's finding that **adding cost-awareness via prompt engineering made things worse** (−68.2 points) appears entirely novel. No published work shows that explicitly instructing an agent about costs produces perverse incentives.

**Publishability:** The accuracy paradox finding plus the prompt-engineering failure result constitute a strong **empirical contribution paper**, potentially for EMNLP or an evaluation workshop. Combined with Cost-of-Pass (Erol et al., 2025) and Kapoor et al.'s Pareto framework, this could motivate a new **agent evaluation metric** paper proposing net utility scoring.

---

## 5. Bayesian forgetting for tool reliability has no counterpart in agent systems

The classical toolkit for non-stationary Bayesian inference is well-developed: **power priors** (Ibrahim & Chen 1998, 2003), **Bayesian Online Changepoint Detection** (Adams & MacKay 2007), **exponential forgetting** (Moens & Zénon 2019), **Bayes with Adaptive Memory** (Nassar et al. 2022), and a recent unifying paper on **Bayesian forgetting as a posterior operator** (2025). Separately, there is growing empirical evidence that LLM APIs are non-stationary — Chen, Zaharia & Zou (2023) showed GPT-4 prime number accuracy plummeted from **97.6% to 2.4%** between March and June 2023, and a 2026 study found **systematic daily and weekly periodic patterns** accounting for ~20% of performance variability even in fixed model snapshots.

Current agent frameworks handle tool reliability through retry logic, static timeouts, and fallback tools — entirely reactive, never adaptive. **ReliabilityBench** (arXiv 2601.06112) evaluates agent reliability under faults but proposes no adaptive solution. The **Agent Drift** paper (arXiv 2601.04170) formalises drift taxonomy and proposes an Agent Stability Index but offers no Bayesian tracking mechanism.

**Gap Credence fills:** No published work applies power-prior-style Bayesian forgetting to LLM tool reliability tracking. The classical methods exist; the LLM reliability problem exists; the bridge between them does not. Credence's exponential forgetting (λ = 0.95) on Beta posteriors is a natural application of Ibrahim & Chen's framework to a novel domain where the posteriors are computationally trivial (closed-form Beta updates).

**Publishability:** A standalone paper on "Adaptive Bayesian Tool Reliability Tracking for LLM Agents" bridging the power prior literature with the agent drift literature could target **Bayesian Analysis**, **JRSS-B**, or **AISTATS**. The empirical evidence of API non-stationarity provides strong motivation.

---

## 6. Embodied design principles and CIRL remain entirely disconnected from LLM agents

**Ay's geometric framework:** Ay (2015) and Montúfar, Ghazi-Zahedi & Ay (2015) formalised the **cheap design principle** — that embodiment constraints render many policies equivalent, enabling exponentially simpler controllers — and **morphological computation** — the extent to which the body/environment offloads computation from the brain. The LLM agent literature informally echoes these ideas (Lilian Weng's 2023 "brain/planning/memory/tools" decomposition, the MAP brain-inspired planner in Nature Communications 2025) but **no work formally connects Ay's information-geometric framework to LLM agent architecture**.

**CIRL:** Hadfield-Menell et al.'s (2016) cooperative inverse reinforcement learning defines a two-player game where the robot's payoff equals the human's actual reward but the robot doesn't initially know it. CIRL is widely cited in alignment surveys but **has never been applied to LLM tool-using agents**. The closest practical work is "apprehensive agents" (Nature Scientific Reports 2024), which embeds alignment into the utility function architecture, and "Machines that halt" (Nature Scientific Reports 2025), which argues alignment should be an architectural guarantee. Neither addresses LLM agents specifically.

**Gap Credence fills:** Credence's axiomatic approach — deriving all behaviour from five axioms including a **CIRL alignment commitment** — would be the first agent framework to make alignment a mathematical axiom rather than a training-time or guardrail constraint. This is a fundamentally different approach than RLHF/DPO (imposed during training) or safety filters (imposed at runtime). Additionally, formalising the LLM-as-brain, tools-as-body, environment-as-world decomposition using Ay's framework and showing that tool/environment structure enables the "cheap" agent design is entirely novel.

**Publishability:** Each of these represents a separate paper:
- **CIRL for LLM agents** with axiomatic alignment → target AAMAS, AAAI, or an alignment workshop (FAccT, SafeAI)
- **Information-geometric design for LLM agents** applying Ay's cheap design principle → target a theory venue (UAI, AISTATS) or Entropy journal

---

## 7. Chain Event Graphs and the options framework are untouched territory for LLM agents

**CEGs:** Jim Q. Smith's group at Warwick has built a rich literature on Chain Event Graphs (Smith & Anderson 2008; Collazo, Görgen & Smith 2018; recent work by Leonelli, Varando, Yu, Strong). CEGs model asymmetric sequential processes where different paths through an event tree can share identical future transition probabilities — these shared positions are called **stages**. CEGs have been applied to decision support (nuclear, food security, forensics), causal inference, and system reliability (Yu & Smith 2024). The staged tree ML literature is growing rapidly (Leonelli & Varando at AISTATS 2023; clustering-based stage learning, March 2026). However, **zero papers apply CEGs to AI/LLM agent decision-making**.

The connection between CEG stage learning and tool reliability tracking is natural but unmade: stages correspond to contexts where a tool behaves identically. Learning stages from tool execution data would automatically discover when tools are interchangeable versus when context matters — a form of principled context-dependent tool reliability clustering.

**Options framework:** Sutton, Precup & Singh's (1999) options framework defines temporally extended actions as ⟨initiation set, internal policy, termination condition⟩ triples. A growing hierarchical RL + LLM literature exists — **GLIDER** (ICML 2025) uses two-level hierarchies with flexible temporal abstraction, **HiPER** (arXiv 2602.16165) separates planning from execution, **ArCHer** (arXiv 2402.19446) operates at utterance and token time scales. However, **no paper formally models LLM tools as options** in Sutton's sense. The phrase "programs as options" does not appear in the literature.

**Gap Credence fills:** Formalising tool calls as options (with reliability-dependent termination conditions) and connecting CEG stage structure to tool reliability contexts would bridge two well-established theoretical frameworks (CEGs and options) to the LLM agent domain for the first time.

**Publishability:** These are two distinct papers:
- **CEGs for agent tool selection** → target PGM conference, Bayesian Analysis, or AISTATS
- **Programs as options** → target ICML or NeurIPS (main conference or workshops on hierarchical RL)

---

## 8. The competing landscape is overwhelmingly heuristic

The dominant LLM agent frameworks lack formal theoretical foundations:

| Framework | Venue | Theory | Tool selection | Cost model |
|-----------|-------|--------|---------------|------------|
| ReAct (Yao et al.) | ICLR 2023 | None (cognitive inspiration) | Prompt-based | None |
| Toolformer (Schick et al.) | NeurIPS 2023 | Perplexity heuristic | Learned (self-supervised) | Implicit |
| Reflexion (Shinn et al.) | NeurIPS 2023 | Verbal RL (no guarantees) | Prompt-based | None |
| LATS (Zhou et al.) | ICML 2024 | MCTS (partially principled) | Tree search | Budget parameter |
| LangChain/LangGraph | Industry | None | Prompt-based | None |

Only **three frameworks** have genuinely formal theoretical foundations: RAFA (Bayesian adaptive MDPs with √T regret), DeLLMa (expected utility maximisation), and DecisionFlow (utility-theoretic reasoning). Credence is positioned to be the **fourth**, with the unique distinction of maintaining per-tool beliefs and computing VOI — capabilities absent from all three existing principled frameworks.

The **HAL Bayesian Control paper** (hal-05480691) explicitly argues that "Bayes has not yet reshaped LLM training" and that agents need Bayesian belief states calibrated against measurable outcomes. Credence provides exactly this.

---

## Publishable paper opportunities and their target venues

Based on the gap analysis, at least **seven distinct papers** can be carved from the Credence framework:

**Paper 1 — The flagship.** "Bayesian Decision-Theoretic Tool Selection for LLM Agents." The core framework: Beta-Bernoulli beliefs, VOI computation, EU maximisation, axiomatic derivation. Benchmark against ReAct and cost-aware baselines. **Target: NeurIPS, ICML, or ICLR.**

**Paper 2 — The accuracy paradox.** "The Accuracy Paradox in Tool-Using Agents: Why Less Accurate Agents Produce More Value." Formalise the phenomenon, show prompt-based cost awareness fails, propose net utility evaluation. **Target: EMNLP, AAAI, or TMLR.**

**Paper 3 — Adaptive tool reliability.** "Bayesian Forgetting for Non-Stationary Tool Reliability in LLM Agents." Bridge power priors to API drift, demonstrate adaptation when tool reliability changes. **Target: AISTATS, UAI, or Bayesian Analysis.**

**Paper 4 — VOI for tool selection.** "Value of Information for LLM Agent Information Gathering." Formal VOI analysis connecting Howard (1966), Russell & Wefald (1991), and modern BED to the tool-calling decision. **Target: UAI or AISTATS.**

**Paper 5 — CIRL alignment-by-construction.** "Alignment by Axiom: Cooperative Inverse Reinforcement Learning in LLM Agent Frameworks." First application of CIRL to practical tool-using agents with alignment as an architectural guarantee. **Target: AAMAS, AAAI, or SafeAI workshop.**

**Paper 6 — CEGs for tool reliability.** "Chain Event Graphs for Sequential Tool Selection in AI Agents." Connect stage learning to context-dependent tool reliability discovery. **Target: PGM conference, Bayesian Analysis, or AISTATS.**

**Paper 7 — Programs as options.** "Tools as Temporally Extended Actions: An Options Framework for LLM Agent Design." Formalise tool calls using Sutton's options framework with reliability-dependent termination. **Target: ICML or NeurIPS workshop on hierarchical RL.**

---

## Essential papers to cite for field awareness

The following papers should appear in any Credence publication to demonstrate comprehensive awareness, organised by relevance:

**Decision-theoretic foundations for LLM agents:** RAFA (Liu et al., ICML 2024) for the only provable Bayesian regret bounds; DeLLMa (Liu et al., ICLR 2025) for EU maximisation in LLM decision-making; DecisionFlow (Chen et al., EMNLP 2025) for utility-theoretic reasoning; the VOI human-agent communication paper (arXiv 2601.06407) for the closest application of VOI to agents; HAL Bayesian Control (hal-05480691) for the argument that agents need Bayesian belief states; Rational Metareasoning for LLMs (De Sabbata et al., 2024) for Value of Computation in LLM reasoning.

**Dominant agent frameworks to position against:** ReAct (Yao et al., ICLR 2023); Toolformer (Schick et al., NeurIPS 2023); Reflexion (Shinn et al., NeurIPS 2023); LATS (Zhou et al., ICML 2024).

**Cost-aware agent work:** AI Agents That Matter (Kapoor et al., TMLR 2025); INTENT (arXiv 2602.11541); BTP (Zheng et al., ACL 2024); Cost-of-Pass (Erol et al., 2025); Reasoning in Token Economies (Wang et al., EMNLP 2024); Theory of Agent (Wang et al., ICML 2025 position paper).

**Bayesian tool/procedure selection:** MACLA (Forouzandeh et al., AAMAS 2026); Thompson Sampling for life sciences agents (arXiv 2512.03065); BIRD (Feng et al., ICLR 2025) for Bayesian inference in LLM decisions.

**Non-stationarity and reliability:** Ibrahim & Chen (2003) for power priors; Adams & MacKay (2007) for BOCD; Chen, Zaharia & Zou (2023) for LLM API drift evidence; ReliabilityBench (arXiv 2601.06112); Agent Drift (arXiv 2601.04170).

**Classical theory:** Howard (1966) for VOI foundations; Russell & Wefald (1991) for metareasoning; Sutton, Precup & Singh (1999) for options; Hadfield-Menell et al. (2016) for CIRL; Ay (2015) and Montúfar et al. (2015) for embodied design; Smith & Anderson (2008) for CEGs; Raiffa & Schlaifer (1961) for Bayesian decision theory.

**Surveys:** Wang et al. (2023) LLM agent survey (arXiv 2308.11432); Foundation Models for Decision Making (Yang et al., 2023); Augmented Language Models (Mialon et al., TMLR 2023); Agentic LLM survey (Plaat et al., 2025).

**Accuracy paradox relatives:** The Accuracy Paradox in RLHF (Chen et al., 2024); The Optimization Paradox (Bedi et al., 2025); The Intervention Paradox (arXiv 2602.03338); Accuracy vs. Accuracy impossibility result (arXiv 2505.16494).

---

## Why the gap exists and what it means

The LLM agent community has been dominated by a **capability-first culture** — prompt engineering, fine-tuning, and scaling have produced dramatic improvements without formal theory. ReAct's thought-action-observation loop, despite having no theoretical guarantees, powers the majority of deployed agents. The decision theory community, conversely, has not engaged with LLM agents as a problem domain.

Credence sits in the intersection of these two communities. Its five axioms provide what no existing framework offers: a **complete derivation** of agent behaviour from first principles, where every action (query, abstain, submit) follows from Bayesian conditioning, expected utility integration, EU maximisation, a Solomonoff complexity prior, and a CIRL alignment commitment. The closest competitor, RAFA, proves regret bounds in Bayesian adaptive MDPs but does not model per-tool reliability, compute VOI for individual tool calls, or address costs explicitly. DeLLMa applies EU maximisation but does not maintain beliefs or track tool reliability over time.

The practical implication is stark: Credence's **+112.6 net points versus ReAct's −8.0** using one-third the tool calls demonstrates that undergraduate probability theory, properly applied, outperforms the industry-standard agent architecture by a margin that no amount of prompt engineering can close. The enhanced LangChain variant's −68.2 score is perhaps the most important result — it empirically falsifies the hypothesis that prompting can substitute for principled decision theory.