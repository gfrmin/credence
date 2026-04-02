# credence-router

Transparent, cost-optimal tool routing via expected-utility maximisation.

Instead of asking an LLM to pick the right tool (and paying for that routing call),
credence-router uses Bayesian decision theory to route questions to the cheapest
reliable tool — with zero routing cost and sub-millisecond routing latency.

## Benchmark (50 questions, simulated tools)

| Agent | Accuracy | Cost | Tools/Q |
|-------|----------|------|---------|
| **credence-router** | **79.3%** | **$0.005** | **1.06** |
| langgraph-react | 60.9% | $0.312 | 1.96 |
| always-best | 74.0% | $0.250 | 1.00 |
| always-cheapest | 44.0% | $0.000 | 1.00 |
| random | 56.0% | $0.078 | 1.00 |

credence-router achieves higher accuracy than LangGraph at **1.6% of the cost**,
because it learns which tools are reliable for which question categories
and only queries expensive tools when the expected value of information justifies it.

## Installation

```bash
pip install credence-router
```

With optional API backends:

```bash
pip install credence-router[anthropic]     # Claude (Haiku/Sonnet)
pip install credence-router[perplexity]    # Perplexity web search
pip install credence-router[all]           # everything
```

## Quick start

```python
from credence_router import Router
from credence_router.tools.calculator import CalculatorTool
from credence_router.tools.claude import ClaudeTool

# Define your tool palette — the router learns which to use
router = Router(tools=[
    CalculatorTool(),           # free, instant, numerical only
    ClaudeTool("haiku"),        # $0.0003/query, general-purpose
    ClaudeTool("sonnet"),       # $0.001/query, higher accuracy
])

# Route a question — returns Answer with choice, confidence, cost, trace
answer = router.solve(
    question="What is 15% of 240?",
    candidates=("32", "36", "40", "44"),
)
print(answer.choice_text)    # "36"
print(answer.tools_used)     # ("calculator",)
print(answer.monetary_cost)  # 0.0

# Report ground truth so the router learns
router.report_outcome(correct=True)

# Save/load learned reliability across sessions
router.save_state("router_state.json")
```

## How it works

Each question goes through a Value of Information (VOI) loop:

1. **Classify** the question into a category (factual, numerical, reasoning, ...)
2. **Calculate** the expected utility of submitting now vs. querying each tool
3. **Query** the tool with highest net VOI (if any tool's VOI exceeds its cost)
4. **Update** beliefs with the tool's response, repeat from step 2
5. **Submit** when no tool's VOI justifies its cost

The router maintains a reliability table — P(correct | tool, category) — learned
from outcome feedback. It starts with prior coverage estimates and refines them
with every `report_outcome()` call.

## CLI

### Benchmark

```bash
# Run full benchmark with simulated tools
credence-router bench --run --simulate

# Explain specific questions (1-indexed)
credence-router bench --run --simulate --explain 1,5,10

# Show learned reliability table
credence-router bench --run --simulate --show-reliability
```

### Route a single question

```bash
# With real tools (needs ANTHROPIC_API_KEY)
credence-router route "What is the capital of France?" -o "London" "Paris" "Berlin" "Madrid"

# Force simulated tools
credence-router route "What is 2+2?" -o "3" "4" "5" "6" --simulate
```

## API

### `Router(tools, categories=..., scoring=..., latency_weight=...)`

The main entry point. Takes a list of `Tool` instances and routes questions optimally.

- **`tools`**: List of objects implementing the `Tool` protocol
- **`categories`**: Tuple of category names (default: 5 built-in categories)
- **`scoring`**: `ScoringRule(reward_correct, penalty_wrong, reward_abstain)`
- **`latency_weight`**: Cost per second of latency (default: 0.0)

### `Tool` protocol

```python
class Tool(Protocol):
    name: str
    cost: float       # $/query
    latency: float    # seconds

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None: ...
    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]: ...
```

### `Answer`

Returned by `router.solve()`:

- `choice`: candidate index (or `None` if abstained)
- `choice_text`: the answer string
- `confidence`: posterior probability
- `tools_used`: tuple of tool names queried
- `monetary_cost` / `effective_cost`: cost breakdown
- `reasoning`: human-readable VOI trace

## Try it: LangGraph vs credence-router (Ollama)

Run a side-by-side comparison locally with zero API costs:

```bash
uv sync --extra demo
ollama pull llama3.1
uv run python examples/langgraph_comparison.py
```

This runs 8 questions through both LangGraph's ReAct agent (LLM-based routing)
and credence-router's drop-in replacement (VOI-based routing), using `llama3.1`
via Ollama. The output shows LLM calls saved and wall-clock speedup per question.

The only code change:

```diff
- from langgraph.prebuilt import create_react_agent
+ from credence_router.compat import create_react_agent
```

## Why not LangGraph?

LangGraph uses an LLM to decide which tool to call. This means:

- **You pay for routing.** Every question costs an LLM call just to pick the tool.
- **Routing is opaque.** The LLM's tool selection can't be inspected or debugged.
- **No learning.** The router doesn't improve with feedback.
- **Over-querying.** ReAct agents call ~2 tools per question on average.

credence-router replaces the routing LLM with a closed-form VOI calculation:

- **Zero routing cost.** Tool selection is a matrix multiply, not an API call.
- **Fully transparent.** Every decision comes with a VOI trace you can inspect.
- **Learns from feedback.** Reliability estimates improve with every outcome.
- **Minimal tool calls.** Only queries tools when the expected value justifies the cost.

## License

AGPL-3.0-only
