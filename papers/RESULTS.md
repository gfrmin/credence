# Benchmark Results

All experiments run from the Julia benchmark at `apps/julia/qa_benchmark/host.jl`.
Results stored in SQLite at `apps/julia/qa_benchmark/results/benchmark.db`.
20 seeds, 50 questions per seed, 4 simulated tools with category-dependent reliability.

**Important change from previous version:** Tool responses are pre-generated per seed
so all agents see identical (question, tool) outputs. No RNG interleaving with agent
decisions. LLM agents use native tool-calling (structured API), not text parsing.

## Agents

- **bayesian** — VOI-based tool selection, Beta-Bernoulli reliability learning, no LLM.
  Uses `agent.bdsl` driven by `host.jl`. Cost: $0.
- **claude-haiku-4-5-20251001** — Anthropic Haiku 4.5 with native tool-calling.
  Cost: $1/M input, $5/M output tokens.
- **llama3.1** — Llama 3.1 8B via Ollama with native tool-calling. Cost: $0 (local).
- **single_best** — Always queries Tool A (web_search, cost 1), submits its answer.
- **random** — Queries a random tool, submits its answer.
- **all_tools** — Queries all 4 tools, majority vote.

## Main Results (20 seeds)

| Agent | Score | Accuracy | Abstain% | Tools/Q | Sim Cost/Q | Time/Q | API Cost (total) |
|---|---|---|---|---|---|---|---|
| claude-haiku-4-5-20251001 | +445.5 ± 13.9 | 0.975 | 0.000 | 0.59 | 0.71 | 2.67s | $3.24 |
| bayesian | +163.7 ± 51.7 | 0.647 | 0.045 | 0.96 | 1.25 | 0.016s | $0.00 |
| llama3.1 | +79.3 ± 36.7 | 0.594 | 0.229 | 0.90 | 1.44 | 1.56s | $0.00 |
| random | +55.4 ± 48.7 | 0.509 | 0.000 | 1.00 | 1.53 | ~0s | $0.00 |
| single_best | +44.2 ± 50.1 | 0.459 | 0.000 | 1.00 | 1.00 | ~0s | $0.00 |
| all_tools | -67.0 ± 65.7 | 0.644 | 0.000 | 4.00 | 6.00 | ~0s | $0.00 |

## Key Findings

### 1. Frontier LLM dominates on raw score

Haiku 4.5 scores +445.5, nearly 3x the Bayesian agent (+163.7). It answers 97.5% of
questions correctly while using fewer tools (0.59/q) than any other agent. It answers
30+/50 questions with no tools at all — pure world knowledge.

### 2. Bayesian agent is the best zero-cost strategy

Among agents that don't call an LLM API, the Bayesian agent dominates at +163.7,
beating random (+55.4) by 3x and single_best (+44.2) by 3.7x. It achieves this
through principled tool selection (VOI), reliability learning, and selective abstention.

### 3. The cost-performance tradeoff is stark

| Comparison | Score | API Cost | Latency |
|---|---|---|---|
| Haiku 4.5 | 445.5 | $3.24 / 1000q | 2.67s/q |
| Bayesian | 163.7 | $0.00 | 0.016s/q |

Bayesian gets 37% of Haiku's score at 0% of its cost and 0.6% of its latency.
At scale (millions of tool-selection decisions), this matters.

### 4. Local LLM underperforms Bayesian

llama3.1 8B scores +79.3 — half the Bayesian agent. It abstains too aggressively
(22.9% vs 4.5%) and has lower accuracy (59.4% vs 64.7%). Native tool-calling helps
vs the old text-parsing approach, but a local 8B model lacks the world knowledge to
compete with either the Bayesian agent's principled tool selection or a frontier
model's raw accuracy.

### 5. Accuracy paradox holds among non-LLM agents

all_tools has the highest accuracy among non-LLM agents (64.4%) but the worst score
(-67.0) because querying all 4 tools costs 6.0 per question, destroying the margin.

## Querying the Results

All per-question data is in SQLite:

```bash
sqlite3 apps/julia/qa_benchmark/results/benchmark.db
```

```sql
-- Summary by agent
SELECT agent, COUNT(*) as seeds,
       ROUND(AVG(total_score),1) as avg_score,
       ROUND(SUM(total_api_cost_usd),4) as total_api_cost
FROM runs GROUP BY agent ORDER BY avg_score DESC;

-- Per-category accuracy by agent
SELECT r.agent, q.category,
       ROUND(AVG(q.was_correct)*100,1) as accuracy_pct
FROM questions q JOIN runs r ON q.run_id=r.id
WHERE q.submitted IS NOT NULL
GROUP BY r.agent, q.category ORDER BY r.agent, q.category;

-- Compare agents on same seed
SELECT agent, total_score, wall_time_s, total_api_cost_usd
FROM runs WHERE seed=0 ORDER BY total_score DESC;

-- Questions Haiku got wrong
SELECT r.seed, q.question_id, q.category, q.tools_queried
FROM questions q JOIN runs r ON q.run_id=r.id
WHERE r.agent='claude-haiku-4-5-20251001' AND q.was_correct=0;

-- Bayesian vs Haiku on same questions (seed 0)
SELECT b.question_id, b.category,
       b.was_correct as bayes_correct, b.tool_cost as bayes_cost,
       h.was_correct as haiku_correct, h.api_cost_usd as haiku_api
FROM questions b
JOIN runs rb ON b.run_id=rb.id AND rb.agent='bayesian' AND rb.seed=0
JOIN runs rh ON rh.agent='claude-haiku-4-5-20251001' AND rh.seed=0
JOIN questions h ON h.run_id=rh.id AND h.question_id=b.question_id;

-- Total API spend across all agents
SELECT agent, ROUND(SUM(total_api_cost_usd),4) as total_usd
FROM runs GROUP BY agent ORDER BY total_usd DESC;
```

## Changes from Previous Results

The old RESULTS.md reported different numbers from a different benchmark configuration:
- Old: coverage mechanism (tools could return `nothing`), text-based LLM parsing,
  3 LLM variants (bare/ReAct/ReAct+S+H), interleaved RNG
- New: no coverage (all tools always respond), native tool-calling, pre-generated
  response table, Haiku 4.5 frontier model added

The old headline ("Bayesian +129.5 vs best LLM +10.8") no longer holds against
frontier models. The new headline: Bayesian is the optimal zero-cost strategy,
but frontier LLMs with native tool-calling and world knowledge can outperform it
— at a dollar cost.

## Not Yet Run

- **Ablation variants** (no VOI, no learning, no abstention, greedy) — not yet
  implemented in the redesigned host.jl. Old ablation data (from Python benchmark)
  has been deleted.
- **Sonnet 4.6** — not yet run (~$10 estimated for 20 seeds at 3x Haiku pricing).
- **Non-stationary (drift) scenario** — not yet implemented in new benchmark.
  Old drift results are no longer valid (different tool specs, coverage mechanism).
