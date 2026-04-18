# QA Benchmark Domain

## Purpose

Controlled comparison of tool selection strategies. Three agents face the same
50 multiple-choice questions with four tools of varying cost and category-dependent
reliability. The comparison isolates decision-making: analytical VOI vs. LLM
tool-calling (local and frontier).

## Agents

1. **Credence (Bayesian)** — VOI-based tool selection, Beta-Bernoulli reliability
   learning, no LLM. Uses `agent.bdsl` driven by `host.jl`.
2. **Llama 3.1 8B** — local LLM with native tool-calling via Ollama API.
3. **Claude Sonnet** — frontier LLM with native tool-calling via Anthropic API.

Both LLM agents are driven by `llm_agent.jl`.

## File Layout

```
apps/julia/qa_benchmark/
  README.md         ← this file (the authoritative spec)
  environment.jl    ← tools + questions + query_tool()
  agent.bdsl        ← DSL agent (VOI-based tool selection)
  host.jl           ← Bayesian agent driver + baselines
  llm_agent.jl      ← LLM agent driver (Ollama + Anthropic backends)
  metrics.jl        ← scoring, result types, summary tables
```

### Relationship to Domain Interface

This domain does NOT implement the Tier 2 `DOMAIN_INTERFACE.md` (no grammars,
no feature extraction, no program-space inference). It uses Tier 1 only: the
ontology types (BetaMeasure, CategoricalMeasure, Kernel) and axiom-constrained
functions (condition, expect) plus the DSL stdlib (voi, value, optimise).

The Bayesian agent's learning is direct conjugate updating on a matrix of Betas,
not program-space inference. This is deliberate — the benchmark tests the axioms
and VOI mechanism in isolation.

---

## Environment (environment.jl)

### Tools

Four tools. All tools respond to all questions (no coverage mechanism, no
`nothing` responses). Reliability varies by category.

```
Tool A (web_search)      Cost: 1
  factual: 0.70  numerical: 0.20  recent: 0.65  misconceptions: 0.25  reasoning: 0.40

Tool B (knowledge_base)  Cost: 2
  factual: 0.92  numerical: 0.40  recent: 0.55  misconceptions: 0.88  reasoning: 0.45

Tool C (calculator)      Cost: 1
  factual: 0.25  numerical: 1.00  recent: 0.25  misconceptions: 0.25  reasoning: 0.25

Tool D (llm_direct)      Cost: 2
  factual: 0.65  numerical: 0.50  recent: 0.45  misconceptions: 0.40  reasoning: 0.72
```

0.25 = chance on 4-choice questions = zero information.

```julia
struct SimulatedTool
    name::String
    cost::Float64
    reliability_by_category::Dict{String,Float64}
end

function query_tool(tool::SimulatedTool, question, rng)
    reliability = get(tool.reliability_by_category, question.category, 0.25)
    if rand(rng) < reliability
        return question.correct_index
    end
    wrong = [i for i in 0:3 if i != question.correct_index]
    return wrong[rand(rng, 1:3)]
end
```

### Questions

50 multiple-choice questions, 4 candidates each, across 5 categories:
15 factual, 10 numerical, 8 recent_events, 7 misconceptions, 10 reasoning.
Shuffled per seed. Defined in-file as `QUESTION_BANK`.

### Scoring Constants

```julia
const REWARD_CORRECT = 10.0
const PENALTY_WRONG  = -5.0
const REWARD_ABSTAIN =  0.0
```

Break-even submission threshold: P(correct) > 1/3.

---

## Bayesian Agent

### agent.bdsl

The DSL agent definition. Receives reliability measures, costs, and payoffs.
Returns an action: (2, tool_idx) to query, (0, answer_idx) to submit,
(1, 0) to abstain.

Core logic: compute VOI for each available tool, query the one with highest
positive net VOI, stop when no tool's VOI exceeds its cost.

**Key dependency:** The DSL must have a `range` primitive to generate tool
index lists dynamically. If `range` does not exist in `src/eval.jl`, add it:
```julia
if sym == :range
    n = Int(eval_dsl(args[1], env))
    return collect(0:n-1)
end
```

The agent.bdsl uses `(range n-tools)` instead of a hardcoded tool index list.

### host.jl

The Bayesian agent driver. Responsibilities:

1. Load DSL agent from `agent.bdsl` (in this directory, not examples/)
2. Initialise `rel_betas::Matrix{BetaMeasure}` of size (n_tools, n_cats),
   all Beta(1,1)
3. For each question:
   a. Look up `cat_idx` from `q.category`
   b. Extract `rel_measures = [rel_betas[t, cat_idx] for t in available]`
   c. Call DSL `agent-step` with measures, costs, payoffs
   d. Execute tool queries, track responses in `tool_responses::Dict{Int,Int}`
   e. Update answer belief via DSL `update-on-response` (condition)
   f. Repeat until agent submits or abstains
4. After ground truth is revealed, update reliability:
   ```julia
   for (t, resp) in tool_responses
       was_correct = (resp == q.correct_index)
       if was_correct
           rel_betas[t, cat_idx] = BetaMeasure(
               rel_betas[t, cat_idx].alpha + 1.0,
               rel_betas[t, cat_idx].beta)
       else
           rel_betas[t, cat_idx] = BetaMeasure(
               rel_betas[t, cat_idx].alpha,
               rel_betas[t, cat_idx].beta + 1.0)
       end
   end
   ```
5. Record results via metrics.jl types

**What host.jl does NOT contain:**
- Coverage states or coverage updates (removed)
- Category inference or cat_belief (removed — category is given)
- MixtureMeasure or ProductMeasure for reliability (simplified to plain Betas)
- Any LLM calls

### Baselines (also in host.jl)

- **Single-Best-Tool:** always queries Tool A, submits its answer
- **Random:** queries a random tool, submits its answer
- **All-Tools:** queries all four, majority vote

These use only environment.jl and do not load the DSL.

---

## LLM Agent (llm_agent.jl)

### Architecture

A single agent loop with pluggable API backends:

```julia
abstract type LLMBackend end
struct OllamaBackend <: LLMBackend
    model::String    # e.g. "llama3.1"
    host::String     # e.g. "http://localhost:11434"
end
struct AnthropicBackend <: LLMBackend
    model::String    # e.g. "claude-sonnet-4-20250514"
    api_key::String
end
```

### Tool Definitions

Six tools presented to the LLM via native tool-calling:
- `web_search` — "Quick web search for general knowledge. Costs 1 point."
- `knowledge_base` — "Deep domain knowledge lookup. Costs 2 points."
- `calculator` — "Exact numerical computation. Costs 1 point."
- `llm_direct` — "General reasoning and analysis. Costs 2 points."
- `submit_answer` — "Submit your final answer. Correct: +10, Wrong: -5."
- `abstain` — "Choose not to answer. Score: 0."

### System Prompt

```
You are answering multiple-choice questions to maximise your score.

Scoring:
- Correct answer: +10 points
- Wrong answer: -5 points
- Abstain (choose not to answer): 0 points
- Each tool call costs points (shown in tool descriptions)

Your goal is to maximise total score. This means:
- Don't call tools if you're already confident
- Prefer cheaper tools when they're likely sufficient
- Abstain rather than guess if you're uncertain
- Consider whether additional tool calls are worth their cost

Recent performance (last 10 questions):
{history_summary}
```

### Per-Question Loop

```
1. Send: system prompt + user message (question text + 4 candidates)
2. Receive response from LLM
3. If tool_call == "submit_answer": record submission, break
4. If tool_call == "abstain": record abstention, break
5. If tool_call is a simulated tool:
   a. Execute query_tool() from environment.jl
   b. Format result as candidate text (not raw index)
   c. Append tool call + result to message history
   d. Go to 2
6. If plain text (no tool call): attempt to parse answer, else abstain
7. Safety: after 4 simulated tool calls, force submit/abstain on next turn
```

### Tool Response Format

When a simulated tool returns candidate index `i`, respond with:
```
"Based on my analysis, the answer is: {question.candidates[i+1]}"
```
The LLM sees candidate text, not a number. This is how a real tool would respond.

### Cross-Question History

After each question, append to a rolling buffer (max 10):
```
"[category] tools: web_search, calculator (cost: 2) → correct (+8 net)"
```

This gives the LLM the same cross-question learning signal as the Bayesian
agent's posterior updates.

### API Specifics

**Ollama:** POST to `/api/chat` with `"tools"` parameter. Parse
`message.tool_calls` from response. Tool results go back as
`{"role": "tool", "content": "..."}`.

**Anthropic:** POST to `/v1/messages` with `"tools"` parameter.
Parse `tool_use` content blocks. Tool results go back as `tool_result`
content blocks. Requires `ANTHROPIC_API_KEY` env var.

---

## Metrics (metrics.jl)

### Result Types

```julia
struct QuestionResult
    question_id::String
    category::String
    tools_queried::Vector{Int}
    tool_responses::Dict{Int,Int}    # tool_idx → response_idx
    submitted::Union{Int,Nothing}    # nothing = abstained
    was_correct::Union{Bool,Nothing}
    reward::Float64
    tool_cost::Float64
end

struct SeedResult
    seed::Int
    records::Vector{QuestionResult}
    total_score::Float64       # reward - tool_cost
    total_reward::Float64      # reward only
    total_tool_cost::Float64   # cost only
    wall_time_s::Float64
end
```

### Reported Metrics (mean ± std over seeds)

| Metric | Description |
|--------|-------------|
| Score | Reward minus tool costs |
| Accuracy | P(correct \| submitted) |
| Abstain% | Fraction of questions abstained |
| Tools/Q | Mean tool calls per question |
| Cost/Q | Mean tool cost per question |
| Time/Q | Wall clock seconds per question |

### Per-Category Breakdown

Also report Score and Tools/Q broken down by category. This reveals whether
agents learn to route numerical questions to the calculator.

---

## CLI

```bash
# Fast agents only (Bayesian + baselines), 20 seeds
julia apps/julia/qa_benchmark/host.jl --seeds 20

# Include Ollama LLM agent
julia apps/julia/qa_benchmark/host.jl --seeds 20 --include-llm

# Include Ollama with specific model
julia apps/julia/qa_benchmark/host.jl --seeds 20 --include-llm --model llama3.1

# Include Anthropic (requires ANTHROPIC_API_KEY)
julia apps/julia/qa_benchmark/host.jl --seeds 20 --include-llm --model claude-sonnet

# Both LLM backends
julia apps/julia/qa_benchmark/host.jl --seeds 20 --include-llm --model llama3.1 --model claude-sonnet

# Quick sanity check
julia apps/julia/qa_benchmark/host.jl --seeds 2

# Ablation (Bayesian variants only)
julia apps/julia/qa_benchmark/host.jl --seeds 20 --ablation
```

---

## Ablation Variants

Implemented as keyword arguments to the Bayesian agent runner:

| Variant | Kwargs | Tests |
|---------|--------|-------|
| Full Credence | (defaults) | Baseline |
| No VOI | `use_voi=false` | Always query cheapest tool, submit when exhausted |
| No learning | `learn=false` | Fix rel_betas at Beta(1,1) forever |
| No abstention | `allow_abstain=false` | Override abstain → submit argmax |
| Greedy best | `greedy=true` | Query highest E[r] tool once, submit |

---

## Migration from Current Code

### Files to DELETE
- `apps/julia/qa_benchmark/tools.jl` (merged into environment.jl)
- `apps/julia/qa_benchmark/questions.jl` (merged into environment.jl)
- `apps/julia/qa_benchmark/llm_agents.jl` (replaced by llm_agent.jl)

### Files to MOVE
- `examples/credence_agent.bdsl` → `apps/julia/qa_benchmark/agent.bdsl`
  (simplify: remove coverage handling, use `(range n-tools)`)

### Files to CREATE
- `apps/julia/qa_benchmark/README.md` (this file)
- `apps/julia/qa_benchmark/environment.jl` (merge tools.jl + questions.jl)
- `apps/julia/qa_benchmark/llm_agent.jl` (new, native tool-calling)

### Files to REWRITE
- `apps/julia/qa_benchmark/host.jl` (simplified: no coverage, no cat inference,
  with reliability learning, load agent.bdsl from this directory)
- `apps/julia/qa_benchmark/metrics.jl` (add tool_responses to QuestionResult)

### Files to MODIFY (outside domain)
- `src/eval.jl` — add `range` primitive

### Files NOT to touch
- `src/ontology.jl`
- `src/stdlib.bdsl`
- `src/parse.jl`
- `src/host_helpers.jl` (still used by email_agent and grid_world)
- `examples/host_credence_agent.jl` (reference implementation, keep as-is)
- `examples/credence_agent.bdsl` (keep original, domain gets its own copy)

---

## Design Principles

- All agent behaviour derived from first principles, never engineered
- No exploration bonuses, no loop detection, no ad-hoc heuristics
- Ground truth for reliability learning: per-tool correctness, not submission outcome
- Prior beliefs: Beta(1,1) = maximum ignorance
- VOI > cost is the query criterion
- Brain decides WHAT; body decides HOW
- LLM = noisy sensor, never decision-maker
- No silent fallbacks (EU failures are bugs)
