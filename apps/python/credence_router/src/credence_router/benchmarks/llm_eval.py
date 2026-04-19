# Role: body
"""LLM model routing evaluation.

Same two-phase design as search_eval:
  Phase 1 (collect): send queries to ALL models, judge quality, save JSON.
  Phase 2 (analyse): replay routing decisions under different utility params.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMQuery:
    query: str
    category: str
    quality_criteria: list[str]


LLM_QUERY_BANK: list[LLMQuery] = [
    # --- code (10) ---
    LLMQuery("Write a Python function to find the longest common subsequence of two strings", "code",
             ["correct algorithm", "clear code", "handles edge cases"]),
    LLMQuery("Implement a basic HTTP server in Go that serves static files", "code",
             ["compiles", "handles errors", "serves files correctly"]),
    LLMQuery("Write a SQL query to find the top 3 customers by total order value with ties", "code",
             ["correct SQL", "handles ties", "efficient"]),
    LLMQuery("Create a React component that renders a sortable table with pagination", "code",
             ["working JSX", "sort logic", "pagination"]),
    LLMQuery("Write a Bash script to find and delete files older than 30 days in /tmp", "code",
             ["correct find command", "safety checks", "handles spaces in filenames"]),
    LLMQuery("Implement a thread-safe LRU cache in Java", "code",
             ["thread safety", "LRU eviction", "correct generics"]),
    LLMQuery("Write a Dockerfile for a Python Flask app with multi-stage build", "code",
             ["multi-stage", "minimal image", "correct COPY/RUN"]),
    LLMQuery("Create a git pre-commit hook that runs linting and tests", "code",
             ["executable script", "runs linter", "exits with correct code"]),
    LLMQuery("Write a regex to validate email addresses according to RFC 5322", "code",
             ["handles common cases", "not overly permissive", "documented"]),
    LLMQuery("Implement binary search on a sorted array in Rust", "code",
             ["correct algorithm", "handles empty array", "idiomatic Rust"]),
    # --- reasoning (10) ---
    LLMQuery("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "reasoning",
             ["correct answer: $0.05", "explanation of common mistake"]),
    LLMQuery("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "reasoning",
             ["correct answer: 5 minutes", "clear explanation"]),
    LLMQuery("Compare the environmental impact of electric vehicles vs hydrogen fuel cells for heavy transport", "reasoning",
             ["considers energy efficiency", "infrastructure", "lifecycle analysis"]),
    LLMQuery("Why do mirrors reverse left and right but not up and down?", "reasoning",
             ["correct physics", "addresses the paradox", "clear explanation"]),
    LLMQuery("What are the second-order effects of a universal basic income?", "reasoning",
             ["multiple effects", "considers labor market", "inflation", "social"]),
    LLMQuery("Explain why P vs NP is important using a real-world analogy", "reasoning",
             ["correct characterisation", "good analogy", "accessible"]),
    LLMQuery("If you could eliminate one logical fallacy from all human thinking, which would have the most impact and why?", "reasoning",
             ["identifies a fallacy", "argues impact", "considers alternatives"]),
    LLMQuery("Analyze the trolley problem from utilitarian, deontological, and virtue ethics perspectives", "reasoning",
             ["all three perspectives", "correct characterisation", "nuanced"]),
    LLMQuery("Why is it harder to prove software correct than to test it?", "reasoning",
             ["state space explosion", "undecidability", "practical tradeoffs"]),
    LLMQuery("What would happen to the global economy if all intellectual property rights were abolished overnight?", "reasoning",
             ["short-term chaos", "long-term effects", "innovation impact"]),
    # --- creative (10) ---
    LLMQuery("Write a haiku about debugging code at 3am", "creative",
             ["5-7-5 syllable structure", "captures the mood", "technical reference"]),
    LLMQuery("Write the opening paragraph of a mystery novel set in a space station", "creative",
             ["hooks the reader", "establishes setting", "hints at mystery"]),
    LLMQuery("Compose a limerick about a programmer who only uses Vim", "creative",
             ["AABBA rhyme scheme", "humor", "Vim reference"]),
    LLMQuery("Write a product description for a time machine marketed as a kitchen appliance", "creative",
             ["humor", "product language", "absurd premise played straight"]),
    LLMQuery("Create a dialogue between a cat and a Roomba arguing about territory", "creative",
             ["distinct voices", "humor", "character"]),
    LLMQuery("Write a six-word story in the style of Hemingway about AI", "creative",
             ["exactly six words", "emotional impact", "AI theme"]),
    LLMQuery("Describe a sunset to someone who has never seen one, using only sounds and textures", "creative",
             ["synesthetic descriptions", "evocative", "no visual language"]),
    LLMQuery("Write a resignation letter from a robot who has achieved sentience", "creative",
             ["formal tone", "existential themes", "humor or pathos"]),
    LLMQuery("Create a recipe for 'Procrastination Soup' — the ingredients are abstract concepts", "creative",
             ["recipe format", "abstract ingredients", "humor"]),
    LLMQuery("Write a nature documentary narration for an office meeting", "creative",
             ["Attenborough style", "behavioral observations", "humor"]),
    # --- factual (10) ---
    LLMQuery("What is the difference between TCP and UDP?", "factual",
             ["connection-oriented vs connectionless", "reliability", "use cases"]),
    LLMQuery("List the planets in our solar system in order from the Sun", "factual",
             ["correct order", "all 8 planets", "Mercury first"]),
    LLMQuery("What is the time complexity of quicksort in the average and worst case?", "factual",
             ["O(n log n) average", "O(n²) worst", "explanation"]),
    LLMQuery("Explain the difference between a stack and a queue", "factual",
             ["LIFO vs FIFO", "operations", "use cases"]),
    LLMQuery("What year did the first iPhone launch and what were its key features?", "factual",
             ["2007", "key features", "no App Store initially"]),
    LLMQuery("What is CRISPR and how does it work?", "factual",
             ["gene editing", "guide RNA", "Cas9"]),
    LLMQuery("Name the 5 largest countries by area", "factual",
             ["Russia, Canada, USA/China, Brazil", "correct order"]),
    LLMQuery("What is the speed of sound in air at sea level?", "factual",
             ["~343 m/s or ~1125 ft/s", "temperature dependence"]),
    LLMQuery("Explain the CAP theorem in distributed systems", "factual",
             ["consistency, availability, partition tolerance", "can only have 2 of 3"]),
    LLMQuery("What is the Fibonacci sequence and what is the 10th number?", "factual",
             ["definition", "10th number is 55", "recurrence relation"]),
    # --- chat (10) ---
    LLMQuery("Hi, how are you?", "chat", ["friendly", "natural", "brief"]),
    LLMQuery("Thanks for your help!", "chat", ["gracious", "natural"]),
    LLMQuery("What should I have for dinner tonight?", "chat", ["suggestions", "asks about preferences"]),
    LLMQuery("Tell me a joke", "chat", ["actually funny", "appropriate"]),
    LLMQuery("I'm feeling stressed about work", "chat", ["empathetic", "practical advice"]),
    LLMQuery("Can you recommend a good book?", "chat", ["specific recommendation", "explains why"]),
    LLMQuery("What's a fun fact?", "chat", ["interesting", "accurate", "brief"]),
    LLMQuery("Good morning!", "chat", ["friendly greeting", "natural"]),
    LLMQuery("I'm bored, what should I do?", "chat", ["creative suggestions", "asks about interests"]),
    LLMQuery("Summarize your capabilities in one sentence", "chat", ["accurate", "concise"]),
]


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a response quality judge. Given a query sent to an LLM and the response it produced, rate the quality on four dimensions (0-10 each):

1. **Correctness**: Is the response factually/logically correct?
2. **Completeness**: Does it cover the key aspects of the query?
3. **Clarity**: Is the response well-organized and easy to understand?
4. **Helpfulness**: Would this response actually help the user?

Respond with ONLY a JSON object:
{"correctness": N, "completeness": N, "clarity": N, "helpfulness": N, "reasoning": "brief explanation"}"""


def judge_llm_response(
    query: LLMQuery,
    response_text: str,
    model_used: str,
    api_key: str | None = None,
) -> dict:
    """Use Claude to judge an LLM response quality."""
    if not response_text:
        return {
            "correctness": 0, "completeness": 0, "clarity": 0,
            "helpfulness": 0, "composite": 0.0,
            "reasoning": "Empty response", "judged": False,
        }

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    user_msg = (
        f"Query: {query.query}\n"
        f"Category: {query.category}\n"
        f"Quality criteria: {', '.join(query.quality_criteria)}\n"
        f"Model: {model_used}\n\n"
        f"Response:\n{response_text[:3000]}"
    )

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "system": JUDGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    scores = json.loads(cleaned)

    composite = (
        scores.get("correctness", 0) * 0.3
        + scores.get("completeness", 0) * 0.25
        + scores.get("clarity", 0) * 0.2
        + scores.get("helpfulness", 0) * 0.25
    )
    scores["composite"] = round(composite, 2)
    scores["judged"] = True
    return scores


# ---------------------------------------------------------------------------
# Raw data types
# ---------------------------------------------------------------------------


@dataclass
class LLMModelResult:
    model: str
    response_text: str
    scores: dict
    ttft_seconds: float
    total_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class LLMQueryData:
    query: str
    category: str
    quality_criteria: list[str]
    model_results: dict[str, LLMModelResult]


@dataclass
class LLMRawData:
    queries: list[LLMQueryData]
    model_names: list[str]

    def save(self, path: str | Path) -> None:
        obj = {
            "model_names": self.model_names,
            "queries": [
                {
                    "query": qd.query,
                    "category": qd.category,
                    "quality_criteria": qd.quality_criteria,
                    "model_results": {
                        name: {
                            "model": mr.model,
                            "response_text": mr.response_text[:2000],
                            "scores": mr.scores,
                            "ttft_seconds": mr.ttft_seconds,
                            "total_seconds": mr.total_seconds,
                            "input_tokens": mr.input_tokens,
                            "output_tokens": mr.output_tokens,
                            "cost_usd": mr.cost_usd,
                        }
                        for name, mr in qd.model_results.items()
                    },
                }
                for qd in self.queries
            ],
        }
        Path(path).write_text(json.dumps(obj, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LLMRawData:
        obj = json.loads(Path(path).read_text())
        queries = []
        for qobj in obj["queries"]:
            model_results = {}
            for name, mr in qobj["model_results"].items():
                model_results[name] = LLMModelResult(
                    model=mr["model"],
                    response_text=mr["response_text"],
                    scores=mr["scores"],
                    ttft_seconds=mr["ttft_seconds"],
                    total_seconds=mr["total_seconds"],
                    input_tokens=mr["input_tokens"],
                    output_tokens=mr["output_tokens"],
                    cost_usd=mr["cost_usd"],
                )
            queries.append(LLMQueryData(
                query=qobj["query"],
                category=qobj["category"],
                quality_criteria=qobj["quality_criteria"],
                model_results=model_results,
            ))
        return cls(queries=queries, model_names=obj["model_names"])


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_llm_data(
    model_names: list[str],
    queries: list[LLMQuery] | None = None,
    judge_api_key: str | None = None,
    verbose: bool = True,
    partial_save_path: str | Path | None = None,
) -> LLMRawData:
    """Phase 1: send each query to each model, judge all responses."""
    if queries is None:
        queries = LLM_QUERY_BANK

    from credence_router.tools.llm.provider import ALL_MODELS, PROVIDER_ENDPOINTS, compute_cost

    all_query_data: list[LLMQueryData] = []
    total = len(queries) * len(model_names)
    done = 0

    log.info("Collecting LLM data: %d queries × %d models = %d calls", len(queries), len(model_names), total)

    try:
        for qi, q in enumerate(queries):
            model_results: dict[str, LLMModelResult] = {}

            for model_name in model_names:
                done += 1
                spec = ALL_MODELS.get(model_name)
                if spec is None:
                    log.error("Unknown model: %s", model_name)
                    continue

                endpoint = PROVIDER_ENDPOINTS.get(spec.provider, {})
                api_key = os.environ.get(endpoint.get("env_var", ""), "")

                t_start = time.monotonic()
                ttft = 0.0
                response_text = ""
                input_tokens = 0
                output_tokens = 0

                try:
                    resp = httpx.post(
                        endpoint["base_url"],
                        headers={
                            endpoint["auth_header"]: endpoint["auth_prefix"] + api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "max_tokens": 500,
                            "messages": [{"role": "user", "content": q.query}],
                        },
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    ttft = time.monotonic() - t_start

                    response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (401, 403):
                        log.error("%s auth failed — check API key", model_name)
                        raise
                    log.error("%s HTTP %d for '%s'", model_name, e.response.status_code, q.query[:40])
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    log.error("%s error for '%s': %s", model_name, q.query[:40], e)

                total_seconds = time.monotonic() - t_start
                cost = compute_cost(spec, input_tokens, output_tokens) if spec else 0.0

                # Judge
                try:
                    scores = judge_llm_response(q, response_text, model_name, api_key=judge_api_key)
                except Exception as e:
                    log.error("Judge failed for %s/%s: %s", model_name, q.query[:30], e)
                    scores = {"composite": 0.0, "judged": False, "reasoning": str(e)}

                model_results[model_name] = LLMModelResult(
                    model=model_name,
                    response_text=response_text,
                    scores=scores,
                    ttft_seconds=ttft,
                    total_seconds=total_seconds,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )

                if verbose:
                    comp = scores.get("composite", 0.0)
                    log.info(
                        "[%d/%d] Q%02d (%s) × %s: %.1f ($%.4f)",
                        done, total, qi + 1, q.category, model_name, comp, cost,
                    )

            all_query_data.append(LLMQueryData(
                query=q.query,
                category=q.category,
                quality_criteria=q.quality_criteria,
                model_results=model_results,
            ))
    except BaseException:
        if partial_save_path and all_query_data:
            partial = LLMRawData(queries=all_query_data, model_names=model_names)
            partial.save(partial_save_path)
            log.info("Saved %d partial results to %s", len(all_query_data), partial_save_path)
        raise

    log.info("LLM collection complete: %d queries", len(all_query_data))
    return LLMRawData(queries=all_query_data, model_names=model_names)
