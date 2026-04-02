# Tool Routing via EU Maximisation

Replace LangChain/LangGraph's opaque LLM-based routing with principled Value of Information calculations.

## The Problem

LangChain routes tools by asking an LLM "which tool should I use?" This is:
- **Expensive**: Routing LLM calls cost ~$0.006-$0.009/question
- **Slow**: 1-1.5s routing overhead per question
- **Opaque**: No explanation of why a tool was chosen
- **Static**: Same routing behaviour on query 1 and query 10,000

## The Solution

credence provides EU-maximising tool selection: VOI calculations, learned reliability tables, explicit cost/benefit tradeoffs. Routing cost: $0, <1ms.

## Quick Start

```bash
python examples/tool_routing/demo.py
python examples/tool_routing/demo.py --latency-weight 0.01 --num 50
```

## Tools

| Tool | Cost | Latency | Best For |
|------|------|---------|----------|
| calculator | $0.000 | 1ms | Numerical (95% reliable) |
| cheap_llm (Haiku) | $0.0003 | 200ms | General (55-60% reliable) |
| expert_llm (Sonnet) | $0.001 | 400ms | Reasoning (85%), misconceptions (80%) |
| web_search (Perplexity) | $0.005 | 800ms | Recent events (90%), factual (80%) |

## How It Works

Each question, the agent:
1. Infers the question category (factual, numerical, reasoning, etc.)
2. Computes VOI for each tool: "how much would querying this tool improve my expected score?"
3. Subtracts effective cost: `net = VOI - monetary_cost - latency_weight * latency`
4. Queries the tool with highest positive net VOI, or submits if no tool is worth querying
5. After feedback, updates the reliability table — learning which tools work for which categories

## Comparison

| Dimension | credence | LangGraph ReAct |
|-----------|----------|-----------------|
| Routing cost | **$0** (numpy) | $0.006-$0.009/q (LLM) |
| Routing latency | **<1ms** | 1-1.5s |
| Learning | Yes | No |
| Deterministic | Yes | No |
| Explainable | VOI numbers | "I think..." |

See [credence-router](https://github.com/[org]/credence-router) for a full library with real API tools and benchmarks.
