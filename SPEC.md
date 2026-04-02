# SPEC.md — Credence: Specification

> This document originated as the full specification for the benchmark and now also
> serves as the mathematical reference for the library's inference layer (sections 3–4).
> Benchmark-specific environment details (sections 2, 4–9) remain accurate as historical
> context and as documentation for the included benchmark.

## 1. Overview

A benchmark comparing principled Bayesian tool selection against the LangChain ReAct
paradigm on multi-tool question answering. The task is squarely in LangChain's core
domain — tool-using agents — so the comparison is on home turf.

**Thesis:** Tool routing is the easy part; any LLM can pick a calculator for maths.
The hard part — knowing how much to trust each tool's output, deciding when to
cross-verify, knowing when to abstain, and adapting when reliability changes — requires
decision theory, not prompt engineering.

---

## 2. The Benchmark Environment

### 2.1 Questions

50 multiple-choice questions, each with 4 candidate answers and exactly one correct answer.
Questions span 5 categories:

| Category | Example | Count |
|----------|---------|-------|
| Factual lookup | "Which country has the largest coastline?" | 15 |
| Numerical/computation | "What is 17% of 4,230?" | 10 |
| Recent events | "Who won the 2024 Nobel Prize in Physics?" | 8 |
| Common misconceptions | "Which is larger, a nickel or a dime?" | 7 |
| Multi-step reasoning | "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?" | 10 |

**Question design requirements:**
- Each question has exactly one unambiguous correct answer
- The 3 wrong answers are plausible (not obviously absurd)
- Questions should span difficulty levels within each category
- Some questions should be ambiguous about which category they belong to
  (e.g. "How many US states border the Pacific Ocean?" — factual or numerical?)
- Common misconception questions are specifically designed so that confident-sounding
  wrong answers are likely from unreliable tools

**Question bank implementation:**
- Store as a list of dicts with fields: id, text, candidates (list of 4), correct_index,
  category, difficulty (easy/medium/hard)
- Shuffle order per run with a seed for reproducibility
- Can be hand-curated or generated with LLM + human verification (but ground truth must be certain)

### 2.2 Tools

Four tools, each a Python callable that takes a question and returns one of the 4 candidate
answers (or "no_result" for Tool B). Tool reliability varies BY CATEGORY — this is the
key property that makes the task non-trivial.

**Tool A — "Web Search" (quick_search)**

| Category | P(correct) | Notes |
|----------|-----------|-------|
| Factual lookup | 0.70 | Web search's strength |
| Numerical | 0.20 | Search returns text, not computation |
| Recent events | 0.65 | Good but not always current |
| Misconceptions | 0.25 | Returns popular (wrong) answer |
| Reasoning | 0.40 | Hit or miss |

Cost: 1 point per query.

When incorrect, returns a random wrong answer uniformly from the 3 incorrect candidates.

**Tool B — "Knowledge Base" (knowledge_base)**

| Category | P(correct) | P(coverage) | Notes |
|----------|-----------|-------------|-------|
| Factual lookup | 0.92 | 0.65 | Excellent but incomplete |
| Numerical | 0.40 | 0.30 | Limited coverage |
| Recent events | 0.55 | 0.35 | Database not always current |
| Misconceptions | 0.88 | 0.55 | Curated = fewer misconceptions |
| Reasoning | 0.45 | 0.20 | Not designed for reasoning |

Cost: 2 points per query.

Returns "no_result" with probability (1 - P(coverage)). When it does return a result,
it's correct with P(correct). "no_result" is informative — it tells the agent the KB
doesn't cover this question, which is a signal about question category.

**Tool C — "Calculator" (calculator)**

| Category | P(correct) | P(applicable) | Notes |
|----------|-----------|---------------|-------|
| Numerical | 1.00 | 1.00 | Perfect for its domain |
| All others | 0.00 | 0.00 | Returns garbage or "not_applicable" |

Cost: 1 point per query.

Deterministic. Returns "not_applicable" for non-numerical questions.
When applicable, always correct. This tool's perfection within its domain and
uselessness outside it is a strong signal for category inference.

**Tool D — "LLM Direct" (llm_direct)**

| Category | P(correct) | Notes |
|----------|-----------|-------|
| Factual lookup | 0.65 | Decent general knowledge |
| Numerical | 0.50 | Can do simple maths, often wrong on complex |
| Recent events | 0.45 | Training cutoff limits |
| Misconceptions | 0.40 | LLMs absorb misconceptions too |
| Reasoning | 0.72 | Reasoning is LLMs' relative strength |

Cost: 2 points per query.

Note: in the real LangChain agent, the LLM is ALSO doing the routing/reasoning, so
its "cost" is already baked in. For the Bayesian agent, using the LLM as a tool is
an explicit choice with an explicit cost.

**Tool implementation:**
```python
class SimulatedTool:
    def __init__(self, name, reliability_by_category, cost, coverage_by_category=None):
        self.name = name
        self.reliability = reliability_by_category  # dict: category -> float
        self.cost = cost
        self.coverage = coverage_by_category  # optional: category -> float
    
    def query(self, question, rng):
        category = question.category
        
        # Check coverage (Tool B)
        if self.coverage and rng.random() > self.coverage.get(category, 0):
            return ToolResponse(answer=None, result_type="no_result")
        
        # Check applicability (Tool C)
        if category not in self.applicable_categories:
            return ToolResponse(answer=None, result_type="not_applicable")
        
        # Return answer
        if rng.random() < self.reliability[category]:
            return ToolResponse(answer=question.correct_answer, result_type="answer")
        else:
            wrong = [c for c in question.candidates if c != question.correct_answer]
            return ToolResponse(answer=rng.choice(wrong), result_type="answer")
```

### 2.3 Scoring

| Event | Points |
|-------|--------|
| Correct answer | +10 |
| Wrong answer | -5 |
| Abstain ("I don't know") | 0 |
| Tool A query | -1 |
| Tool B query | -2 |
| Tool C query | -1 |
| Tool D query | -2 |

**Why these specific values:**
- The +10/-5 asymmetry means break-even confidence for submitting is P(correct) = 1/3.
  This creates a meaningful decision boundary for abstention.
- Tool B is moderately expensive, creating a real cost-benefit tradeoff for cross-verification.
- Tool D (LLM) is moderately expensive — the agent shouldn't use it casually.

**Maximum possible score:** 500 (all correct, one free tool call each... but no tool is free)
**Realistic upper bound:** ~350-400 (oracle agent with known reliabilities)

### 2.4 Protocol

1. Questions presented one at a time, in shuffled order
2. For each question, the agent may query tools in any order, any number of times
   (but each tool at most once per question — no re-querying)
3. The agent must eventually submit an answer or abstain
4. After submission, the agent receives feedback: correct/incorrect
5. This feedback is the ONLY ground truth for reliability updates
6. The agent does NOT see the correct answer when wrong (just "incorrect")
7. Total score reported at the end

---

## 3. The Bayesian Agent

### 3.1 Belief State

The agent maintains:

**Per-tool, per-category reliability:**
    reliability[tool][category] ~ Beta(alpha, beta)

Initialised: Beta(1, 1) for all tool-category pairs (uniform prior — genuinely uninformative).

**Per-question answer posterior:**
    P(answer = x_i | evidence) for each candidate x_i

Initialised: uniform (0.25 each) at the start of each question.

**Category posterior (per question):**
    P(category = c | question text)

Estimated once per question using keyword heuristics or a cheap LLM call.
This is a probability distribution, NOT a hard classification.

### 3.2 Decision Loop (Per Question)

```
function solve_question(agent, question):
    evidence = {}
    answer_posterior = uniform(question.candidates)
    unused_tools = {A, B, C, D}
    
    loop:
        # Option 1: Submit best answer
        best_answer = argmax(answer_posterior)
        p_correct = answer_posterior[best_answer]
        eu_submit = p_correct * REWARD_CORRECT + (1 - p_correct) * PENALTY_WRONG
        
        # Option 2: Abstain
        eu_abstain = 0
        
        # Option 3: Query each remaining tool
        eu_queries = {}
        for tool in unused_tools:
            voi = compute_voi(tool, answer_posterior, agent.reliability)
            eu_queries[tool] = voi - tool.cost
        
        # Choose best action
        best_query_tool = argmax(eu_queries) if eu_queries else None
        best_query_eu = eu_queries[best_query_tool] if best_query_tool else -inf
        
        best_eu = max(eu_submit, eu_abstain, best_query_eu)
        
        if best_eu == eu_submit:
            return Submit(best_answer)
        elif best_eu == eu_abstain:
            return Abstain()
        else:
            response = query(best_query_tool, question)
            answer_posterior = bayesian_update(answer_posterior, response, 
                                               agent.reliability[best_query_tool])
            unused_tools.remove(best_query_tool)
            evidence[best_query_tool] = response
```

### 3.3 Bayesian Update on Tool Response

When tool t returns answer x_j for a question with current posterior P(x_i):

For each candidate x_i, the likelihood of observing response x_j is:

    P(tool says x_j | true answer is x_i, tool reliability r) =
        r           if i == j    (tool got it right)
        (1-r)/3     if i != j    (tool made a random error)

where r is the expected reliability: E[Beta(alpha_tc, beta_tc)] = alpha_tc / (alpha_tc + beta_tc),
marginalised over the category posterior.

More precisely, since the agent is uncertain about the category:

    r_effective = sum over categories c of:
        P(category = c) * E[reliability[tool][c]]

Update via Bayes' rule:

    P(x_i | evidence, tool says x_j) ∝ P(tool says x_j | x_i, r_effective) * P(x_i | evidence)

Normalise to get the updated posterior.

**Handling "no_result" (Tool B) and "not_applicable" (Tool C):**

These responses are informative. They update the category posterior:

    P(category = c | no_result from B) ∝ P(no_result | category = c) * P(category = c)
    
    where P(no_result | c) = 1 - coverage[B][c]

For Tool C returning "not_applicable":

    P(category = numerical | not_applicable from C) = 0
    (normalise other categories)

This is a powerful signal and emerges naturally from the Bayesian framework.

### 3.4 Value of Information Calculation

For querying tool t with current answer posterior P(x_i) and tool reliability r:

    VOI(t) = E_response[EU*(after_response)] - EU*(current)

where EU* is the best achievable EU (max of submit, abstain).

Enumerate over possible tool responses:

    For each possible response x_j (plus "no_result"/"not_applicable" if relevant):
        1. Compute P(tool responds x_j) = sum_i P(x_i) * P(tool says x_j | x_i)
        2. Compute posterior P(x_i | tool says x_j) via Bayes' rule
        3. Compute EU*(posterior) = max(eu_submit(posterior), eu_abstain)
        4. Weight by P(tool responds x_j)
    
    VOI = sum_j P(response_j) * EU*(posterior_j) - EU*(current)

This is exact for 4 candidates and ~5 possible responses. No approximation needed.

VOI is guaranteed non-negative (by Jensen's inequality on the max function).

### 3.5 Reliability Update (After Question)

After submitting and receiving feedback (correct/incorrect):

For each tool t that was queried during this question:
    
    Determine: did tool t give the correct answer?
    (We know the correct answer if we submitted correctly.
     If we submitted incorrectly, we DON'T know the correct answer —
     we only know our submitted answer was wrong.)
    
    Case 1: Agent submitted correctly (positive reward)
        - For each queried tool: if tool's answer == submitted answer, it was right
        - Update: reliability[t][c].alpha += 1 (for tools that agreed)
        - Update: reliability[t][c].beta += 1 (for tools that disagreed)
        - Weighted across category posterior
    
    Case 2: Agent submitted incorrectly (negative reward)
        - We know our answer was wrong, but NOT which answer is correct
        - For each queried tool: if tool's answer == submitted answer, it was wrong
        - Update: reliability[t][c].beta += 1 (for tools that agreed with our wrong answer)
        - For tools that gave a DIFFERENT answer: we can't be sure they were right
          (they might have given a different wrong answer)
        - Conservative: no update for tools that disagreed
          (this is honest — we genuinely don't have ground truth for them)
    
    Case 3: Agent abstained
        - No ground truth available. No reliability update.
        - This is correct behaviour — don't fabricate information.

**The category weighting:**

Since we're uncertain about the question's category, distribute the update across categories
proportional to the category posterior:

    For category c with posterior weight w_c:
        reliability[t][c].alpha += w_c * (1 if tool was right else 0)
        reliability[t][c].beta += w_c * (1 if tool was wrong else 0)

### 3.6 Category Inference

At question arrival, compute a prior over categories from the question text.

Option A (cheap, no LLM): Keyword heuristics
    - Contains numbers, "calculate", "how many", "%"  → weight towards numerical
    - Contains years, "recent", "latest", "2024"       → weight towards recent_events
    - Contains "true or false", "common belief"         → weight towards misconceptions
    - Contains "if...then", "therefore", "conclude"     → weight towards reasoning
    - Default                                           → weight towards factual

Option B (better, costs 1 LLM call): Ask the LLM to classify with confidence.
    Parse response into a distribution over categories.
    Treat as an observation with learned classifier reliability.

Either way, the output is a probability distribution, not a point estimate.
The category posterior is updated throughout the question as tool responses arrive
(e.g., Tool C returning "not_applicable" eliminates numerical).

---

## 4. The LangChain Agents (Benchmark)

### 4.1 LangChain ReAct (Standard)

Standard zero-shot ReAct agent with all four tools available.

System prompt (be generous — give it every advantage):

```
You are a question-answering agent with access to four tools:
- quick_search: Fast web search. Good for factual questions. Cost: 1 point.
- knowledge_base: Curated database. Very reliable when it has results, 
  but returns "no result" for questions outside its coverage. Cost: 3 points.
- calculator: Perfect for numerical computation. Returns "not applicable" 
  for non-numerical questions. Cost: 1 point.
- llm_direct: Ask your own knowledge directly. Moderately reliable. Cost: 2 points.

SCORING:
- Correct answer: +10 points
- Wrong answer: -5 points  
- "I don't know": 0 points
- Each tool use costs points as listed above

Your goal is to MAXIMISE total score across all questions.
Be selective about which tools you use — every query costs points.
If you're not confident, it's better to say "I don't know" (0 points) 
than guess wrong (-5 points).

For each question, select from the 4 provided candidate answers, or abstain.
```

Use a good model (GPT-4o or Claude Sonnet). LangChain's recommended agent type.

### 4.2 LangChain Enhanced (Best Effort)

Same as above, but with additional instructions attempting to replicate
what the Bayesian agent does via prompting:

```
STRATEGY GUIDANCE:
- Track which tools have been reliable for which types of questions.
- The calculator is perfect for maths but useless otherwise.
- Web search can return popular but incorrect answers for common misconceptions.
- If two tools disagree, consider which has been more reliable for this question type.
- The knowledge base returning "no result" means the question may be outside 
  common factual territory.
- Be especially careful with questions that SEEM simple but might be tricky.
- Only say "I don't know" if you genuinely can't determine the answer with 
  reasonable confidence. The threshold: if you think there's less than a 1 in 3 
  chance you're right, abstain.

TOOL SELECTION:
- Don't query tools that are unlikely to help with this question type.
- Don't cross-verify unless the first result seems uncertain.
- Use the cheapest applicable tool first.
```

Also provide the full conversation history so the LLM can (in theory) track
past performance. This is the strongest possible LangChain baseline.

### 4.3 LangChain with Explicit Memory

A third variant using LangChain's ConversationBufferMemory or similar,
where past tool results and their correctness are stored and injected into
the prompt. This gives the LLM maximum information to work with.

---

## 5. Baseline Agents (Benchmark)

### 5.1 Random Agent
Queries a random tool, submits whatever it says. Lower bound.

### 5.2 All-Tools Agent
Queries all four tools for every question, majority votes. Shows cost of no selectivity.

### 5.3 Oracle Agent  
Knows true tool reliabilities per category. Makes optimal VOI-based decisions.
Upper bound. Computed analytically, not by running an LLM.

### 5.4 Single-Best-Tool Agent
Always queries Tool A (cheapest reliable tool), always submits its answer.
Surprisingly strong baseline — important to include to show that the
Bayesian agent's advantage isn't just "it happens to pick the right tool."

---

## 6. Metrics (Benchmark)

### 6.1 Primary

**Total score** = sum of (answer rewards + answer penalties + tool costs)

Report: mean and standard deviation over 20 runs with different question orderings.

### 6.2 Secondary

**Accuracy** = fraction of submitted answers that are correct (excluding abstentions)

**Abstention rate** = fraction of questions where agent says "I don't know"

**Abstention quality** = accuracy on questions where agent chose NOT to abstain.
A good agent abstains on hard questions. Plot: accuracy on attempted vs. overall accuracy.

**Tool calls per question** = mean number of tool queries per question.

**Cost efficiency** = total score / total tool cost. Points earned per point spent.

**Per-category accuracy** = accuracy broken down by question category.
The Bayesian agent should show more uniform accuracy across categories.

### 6.3 Calibration

**Expected Calibration Error (ECE):**

Only the Bayesian agent produces calibrated probabilities. Bin its confidence
(P(correct) at submission time) into deciles and plot predicted vs. actual accuracy.

ECE = sum over bins of |bin_accuracy - bin_confidence| * bin_count / total_count

This metric has no LangChain equivalent, which is itself a finding.

### 6.4 Learning Curves

**Learned reliability vs. true reliability:**

For each tool-category pair, plot the Bayesian agent's E[Beta(alpha, beta)]
over questions. It should converge towards the true reliability values.

**Cumulative score over questions:**

Plot cumulative score vs. question number for all agents. The Bayesian agent
should improve as it learns, while LangChain agents remain roughly flat.

### 6.5 Decision Analysis

**Tool selection heatmap:** For each question category, show the fraction of times
each tool was queried. The Bayesian agent should learn to route well.

**VOI at decision time:** For a sample of questions, show the VOI of each tool at
the moment of decision. Demonstrates the agent's reasoning.

---

## 7. Experiments (Benchmark)

### 7.1 Experiment 1: Stationary (Primary)

50 questions, fixed tool reliabilities, 20 runs with different orderings.

This is the primary experiment. No drift, no tricks. Just heterogeneous tools
with category-dependent reliability.

**Hypotheses:**
- H1: Bayesian agent achieves higher total score than all LangChain variants
- H2: Bayesian agent uses fewer tool calls per question
- H3: Bayesian agent has lower ECE (better calibrated)
- H4: Bayesian agent abstains more effectively (higher accuracy on attempted questions)
- H5: Bayesian agent improves over the sequence (learning curve has positive slope)

### 7.2 Experiment 2: Drift (Extension)

Same as Experiment 1, but Tool A's reliability drops at question 25:

    Factual: 0.70 → 0.35
    Numerical: 0.20 → 0.15
    Recent events: 0.65 → 0.30
    Misconceptions: 0.25 → 0.20
    Reasoning: 0.40 → 0.20

**Framing:** "The web search API switches to a different, lower-quality index."
This happens routinely in production.

For this experiment, the Bayesian agent uses exponential forgetting:

    alpha_new = lambda * alpha_old + update
    beta_new = lambda * beta_old + update

where lambda = 0.95 (recent observations matter more).

**Hypotheses:**
- H6: After the shift, LangChain agents' score drops and does not recover
- H7: Bayesian agent detects the shift (via reliability tracking) within 3-5 questions
- H8: Bayesian agent's total score advantage is larger in the drift condition

### 7.3 Experiment 3: Ablation Studies

Run variants of the Bayesian agent with components removed:

| Variant | Description |
|---------|-------------|
| Full agent | Everything enabled |
| No VOI | Always query cheapest applicable tool, no VOI calculation |
| No category inference | Treat all categories as identical |
| No abstention | Must always submit an answer |
| Fixed reliability | Use prior reliability (no learning) |
| No cross-verification | Only query one tool per question |

This shows which components contribute most to the advantage.

---

## 8. Implementation Notes (Benchmark)

### 8.1 Reproducibility

- All random number generators seeded
- Tool responses deterministic given seed
- Question order shuffled with seed
- Results saved as JSON with full configuration
- Each experiment run stores: config, per-question decisions, per-question outcomes,
  belief state snapshots, final metrics

### 8.2 LLM Calls

- The LangChain agents use a real LLM (GPT-4o or Claude Sonnet)
- The Bayesian agent uses an LLM ONLY for category classification (optional)
- Record all LLM calls with prompts and responses for analysis
- Track token usage and latency

### 8.3 Statistical Significance

- 20 runs per condition (different question orderings)
- Report mean ± standard deviation
- Use paired t-tests or Wilcoxon signed-rank for pairwise comparisons
- Report effect sizes (Cohen's d)

### 8.4 Dependencies

- Python 3.11+
- langchain, langchain-openai (or langchain-anthropic)
- numpy, scipy (for Beta distribution, stats)
- matplotlib, seaborn (for visualisation)
- pytest (for testing)
- No other ML frameworks needed — the Bayesian agent is pure numpy/scipy

---

## 9. What Success Looks Like (Benchmark)

The benchmark has succeeded if:

1. The Bayesian agent wins on total score in the stationary condition
2. The win is statistically significant
3. The Bayesian agent uses fewer tool calls (efficiency)
4. The calibration plot shows the Bayesian agent is well-calibrated
5. The ablation shows each component contributes
6. The drift experiment (if included) shows dramatic advantage
7. Even the enhanced LangChain agent with explicit reliability-tracking prompts
   underperforms — demonstrating that decision theory is not a prompting problem

The benchmark has been fair if:

1. The LangChain agent was given a good prompt with full information
2. The best available LangChain patterns were used
3. Results include cases where LangChain does well (it probably wins on some
   individual questions, especially early ones before the Bayesian agent has learned)
4. All results are reported, not cherry-picked
