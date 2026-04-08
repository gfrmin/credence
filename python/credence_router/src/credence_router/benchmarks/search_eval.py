"""Search provider routing evaluation.

Compares Credence's Bayesian routing against static baselines using real
search providers and an LLM judge for quality scoring.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from credence_router.search_router import SEARCH_CATEGORIES, SearchRouter
from credence_router.tool import SearchResult, SearchTool


# ---------------------------------------------------------------------------
# Query bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchQuery:
    """A search query with quality criteria for LLM judging."""

    query: str
    category: str
    quality_criteria: list[str]


QUERY_BANK: list[SearchQuery] = [
    # --- factual (15) ---
    SearchQuery("population of Tokyo 2025", "factual", ["current population figure", "metropolitan vs city proper distinction"]),
    SearchQuery("boiling point of ethanol in celsius", "factual", ["78.37 or approximately 78 degrees"]),
    SearchQuery("who wrote One Hundred Years of Solitude", "factual", ["Gabriel Garcia Marquez"]),
    SearchQuery("distance from Earth to Mars in kilometers", "factual", ["varies, average ~225 million km"]),
    SearchQuery("what is the GDP of Germany 2024", "factual", ["approximately 4 trillion USD"]),
    SearchQuery("chemical formula for table salt", "factual", ["NaCl"]),
    SearchQuery("when was the Eiffel Tower built", "factual", ["1887-1889", "opened 1889"]),
    SearchQuery("what programming language is the Linux kernel written in", "factual", ["C", "some Rust recently"]),
    SearchQuery("how many bones in the adult human body", "factual", ["206"]),
    SearchQuery("what is the speed of light in meters per second", "factual", ["299,792,458 m/s"]),
    SearchQuery("capital of New Zealand", "factual", ["Wellington"]),
    SearchQuery("who discovered penicillin", "factual", ["Alexander Fleming", "1928"]),
    SearchQuery("what is the half-life of carbon-14", "factual", ["5,730 years"]),
    SearchQuery("largest ocean on Earth by area", "factual", ["Pacific Ocean"]),
    SearchQuery("melting point of iron in celsius", "factual", ["1538 degrees celsius"]),
    # --- recent_events (10) ---
    SearchQuery("latest Federal Reserve interest rate decision 2026", "recent_events", ["most recent rate", "date of decision"]),
    SearchQuery("who won the 2026 FIFA World Cup", "recent_events", ["winning country or upcoming status"]),
    SearchQuery("latest SpaceX Starship launch results", "recent_events", ["most recent launch date", "success/failure"]),
    SearchQuery("current Bitcoin price today", "recent_events", ["recent price figure", "approximate USD value"]),
    SearchQuery("most recent Nobel Prize in Physics winner", "recent_events", ["winner name", "year", "research area"]),
    SearchQuery("latest major earthquake 2026", "recent_events", ["location", "magnitude", "date"]),
    SearchQuery("current US inflation rate", "recent_events", ["CPI percentage", "recent month/quarter"]),
    SearchQuery("who is the current UN Secretary General", "recent_events", ["name", "term information"]),
    SearchQuery("latest iPhone model released", "recent_events", ["model name/number", "key features"]),
    SearchQuery("recent major AI model releases 2026", "recent_events", ["model names", "companies", "capabilities"]),
    # --- technical (10) ---
    SearchQuery("how to use Rust async traits", "technical", ["async fn in traits", "syntax example", "stabilization status"]),
    SearchQuery("PostgreSQL JSONB indexing strategies", "technical", ["GIN index", "expression index", "performance considerations"]),
    SearchQuery("kubernetes pod autoscaling based on custom metrics", "technical", ["HPA configuration", "metrics server", "custom metrics adapter"]),
    SearchQuery("Python asyncio vs threading for IO bound tasks", "technical", ["GIL considerations", "performance comparison", "use cases"]),
    SearchQuery("how to implement a B-tree in C", "technical", ["node structure", "insertion algorithm", "splitting"]),
    SearchQuery("WebAssembly SIMD instructions tutorial", "technical", ["v128 type", "supported operations", "browser support"]),
    SearchQuery("git rebase vs merge best practices", "technical", ["when to use each", "team workflow implications"]),
    SearchQuery("implementing OAuth 2.0 PKCE flow", "technical", ["code verifier", "code challenge", "authorization endpoint"]),
    SearchQuery("eBPF for network monitoring tutorial", "technical", ["BPF programs", "maps", "tracing"]),
    SearchQuery("how to optimize Docker image size", "technical", ["multi-stage builds", "alpine base", "layer caching"]),
    # --- synthesis (10) ---
    SearchQuery("pros and cons of microservices vs monolith architecture", "synthesis", ["scalability", "complexity", "team size considerations"]),
    SearchQuery("comparison of React vs Svelte for web development 2026", "synthesis", ["performance", "developer experience", "ecosystem"]),
    SearchQuery("should I use SQL or NoSQL for my application", "synthesis", ["ACID vs eventual consistency", "query patterns", "scale considerations"]),
    SearchQuery("impact of remote work on software team productivity", "synthesis", ["research findings", "communication challenges", "benefits"]),
    SearchQuery("electric vehicles vs hydrogen fuel cells future", "synthesis", ["infrastructure", "efficiency", "cost trajectory"]),
    SearchQuery("best practices for API versioning", "synthesis", ["URL versioning", "header versioning", "tradeoffs"]),
    SearchQuery("functional programming vs object oriented for large codebases", "synthesis", ["maintainability", "testability", "team familiarity"]),
    SearchQuery("privacy implications of large language models", "synthesis", ["training data", "memorization", "regulatory considerations"]),
    SearchQuery("containerization vs serverless for startup backends", "synthesis", ["cost at scale", "cold starts", "vendor lock-in"]),
    SearchQuery("when to use GraphQL vs REST API", "synthesis", ["overfetching", "schema complexity", "caching"]),
    # --- local/niche (5) ---
    SearchQuery("best ramen restaurants in Shibuya Tokyo", "local", ["specific restaurant names", "specialties"]),
    SearchQuery("Tel Aviv coworking spaces with monthly plans", "local", ["specific space names", "pricing", "locations"]),
    SearchQuery("hiking trails near Queenstown New Zealand difficulty levels", "local", ["trail names", "difficulty ratings", "distances"]),
    SearchQuery("specialty coffee roasters in Melbourne Australia", "local", ["roaster names", "locations", "styles"]),
    SearchQuery("public transportation from Zurich airport to city center", "local", ["options", "duration", "cost"]),
]


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a search quality judge. Given a search query and the search results returned by a provider, rate the quality on four dimensions (0-10 each):

1. **Relevance**: Does the result actually answer or address the query?
2. **Freshness**: Is the information current and up-to-date? (Weight this heavily for recent_events queries)
3. **Completeness**: Does it cover the key aspects the user would expect?
4. **Source quality**: Are the sources authoritative and trustworthy?

Respond with ONLY a JSON object:
{"relevance": N, "freshness": N, "completeness": N, "source_quality": N, "reasoning": "brief explanation"}"""


def judge_search_result(
    query: SearchQuery,
    result: SearchResult | None,
    api_key: str | None = None,
) -> dict:
    """Use Claude to judge search result quality. Returns scores dict."""
    if result is None:
        return {
            "relevance": 0, "freshness": 0, "completeness": 0,
            "source_quality": 0, "composite": 0.0,
            "reasoning": "No result returned",
        }

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    user_msg = (
        f"Query: {query.query}\n"
        f"Category: {query.category}\n"
        f"Quality criteria: {', '.join(query.quality_criteria)}\n\n"
        f"Search results:\n{result.text[:3000]}\n\n"
        f"URLs: {', '.join(result.urls[:5])}"
    )

    try:
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
        scores = json.loads(text)

        # Composite: weight freshness more for recent_events
        weights = {"relevance": 0.3, "freshness": 0.2, "completeness": 0.3, "source_quality": 0.2}
        if query.category == "recent_events":
            weights = {"relevance": 0.25, "freshness": 0.35, "completeness": 0.25, "source_quality": 0.15}

        composite = sum(scores.get(k, 0) * w for k, w in weights.items())
        scores["composite"] = round(composite, 2)
        return scores
    except Exception as e:
        return {
            "relevance": 0, "freshness": 0, "completeness": 0,
            "source_quality": 0, "composite": 0.0,
            "reasoning": f"Judge error: {e}",
        }


# ---------------------------------------------------------------------------
# Evaluation result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Result for a single query from one solver."""

    query: str
    category: str
    provider: str
    scores: dict
    cost: float
    wall_time: float


@dataclass
class EvalResult:
    """Aggregate evaluation result for one solver."""

    solver_name: str
    results: list[QueryResult] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        composites = [r.scores.get("composite", 0.0) for r in self.results]
        return sum(composites) / len(composites) if composites else 0.0

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.results)

    @property
    def mean_latency(self) -> float:
        return sum(r.wall_time for r in self.results) / len(self.results) if self.results else 0.0

    def quality_by_category(self) -> dict[str, float]:
        by_cat: dict[str, list[float]] = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r.scores.get("composite", 0.0))
        return {cat: sum(vs) / len(vs) for cat, vs in by_cat.items()}

    def learning_curve(self, window: int = 10) -> tuple[float, float]:
        """Returns (early_quality, late_quality) for first and last window queries."""
        composites = [r.scores.get("composite", 0.0) for r in self.results]
        if len(composites) < 2 * window:
            return (self.mean_quality, self.mean_quality)
        early = sum(composites[:window]) / window
        late = sum(composites[-window:]) / window
        return (early, late)

    def provider_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.provider] = counts.get(r.provider, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Baseline solvers (search-specific)
# ---------------------------------------------------------------------------


class StaticBestSolver:
    """Always uses one fixed provider (simulates OpenClaw with explicit config)."""

    def __init__(self, search_tool: SearchTool):
        self._tool = search_tool
        self.name = f"static-{search_tool.name}"

    def route(self, query: str, **kwargs) -> tuple[str, SearchResult | None, float]:
        t = time.monotonic()
        result = self._tool.search(query)
        return self._tool.name, result, time.monotonic() - t

    def report_outcome(self, useful: bool) -> None:
        pass


class RoundRobinSolver:
    """Cycles through providers in order."""

    name = "round-robin"

    def __init__(self, search_tools: list[SearchTool]):
        self._tools = search_tools
        self._idx = 0

    def route(self, query: str, **kwargs) -> tuple[str, SearchResult | None, float]:
        tool = self._tools[self._idx % len(self._tools)]
        self._idx += 1
        t = time.monotonic()
        result = tool.search(query)
        return tool.name, result, time.monotonic() - t

    def report_outcome(self, useful: bool) -> None:
        pass


class RandomSearchSolver:
    """Picks a random provider per query."""

    name = "random-search"

    def __init__(self, search_tools: list[SearchTool], seed: int = 42):
        self._tools = search_tools
        self._rng = np.random.default_rng(seed)

    def route(self, query: str, **kwargs) -> tuple[str, SearchResult | None, float]:
        idx = int(self._rng.integers(len(self._tools)))
        tool = self._tools[idx]
        t = time.monotonic()
        result = tool.search(query)
        return tool.name, result, time.monotonic() - t

    def report_outcome(self, useful: bool) -> None:
        pass


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def run_search_eval(
    search_tools: list[SearchTool],
    queries: list[SearchQuery] | None = None,
    seed: int = 42,
    judge_api_key: str | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """Run the full search evaluation comparing Credence vs baselines.

    Returns a list of EvalResult, one per solver.
    """
    if queries is None:
        queries = QUERY_BANK

    rng = np.random.default_rng(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)

    # Build solvers
    credence = SearchRouter(search_tools)
    solvers: list[tuple[str, object]] = [
        ("credence-search", credence),
        ("round-robin", RoundRobinSolver(search_tools)),
        ("random-search", RandomSearchSolver(search_tools, seed=seed)),
    ]
    # Add static-best for each provider
    for tool in search_tools:
        solvers.append((f"static-{tool.name}", StaticBestSolver(tool)))

    all_results: list[EvalResult] = []

    for solver_name, solver in solvers:
        if verbose:
            print(f"\n--- {solver_name} ---")
        eval_result = EvalResult(solver_name=solver_name)

        for i, q in enumerate(shuffled):
            if isinstance(solver, SearchRouter):
                route_result = solver.route(q.query, category_hint=q.category)
                provider = route_result.provider
                search_result = route_result.result
                wall_time = route_result.wall_time
            else:
                provider, search_result, wall_time = solver.route(q.query)

            # Judge quality
            scores = judge_search_result(q, search_result, api_key=judge_api_key)
            cost = next((t.cost for t in search_tools if t.name == provider), 0.0)

            # Report outcome for learning
            useful = scores.get("composite", 0.0) >= 5.0
            solver.report_outcome(useful)

            eval_result.results.append(QueryResult(
                query=q.query,
                category=q.category,
                provider=provider,
                scores=scores,
                cost=cost,
                wall_time=wall_time,
            ))

            if verbose:
                marker = "+" if useful else "-"
                print(f"  [{marker}] Q{i + 1:02d} ({q.category}) -> {provider}: {scores.get('composite', 0):.1f}")

        all_results.append(eval_result)

    return all_results


def format_eval_table(results: list[EvalResult]) -> str:
    """Format evaluation results as a comparison table."""
    header = (
        f"{'Solver':<22s} {'Quality':>8s} {'Cost$':>8s} "
        f"{'Latency':>8s} {'Early':>7s} {'Late':>7s} {'Learn':>7s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        early, late = r.learning_curve()
        delta = late - early
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{r.solver_name:<22s} {r.mean_quality:>7.2f} "
            f"${r.total_cost:>6.3f} "
            f"{r.mean_latency:>6.1f}s "
            f"{early:>6.2f} {late:>6.2f} {sign}{delta:>5.2f}"
        )

    return "\n".join(lines)


def format_category_table(results: list[EvalResult]) -> str:
    """Format per-category quality breakdown."""
    categories = sorted({r.category for er in results for r in er.results})
    header = f"{'Solver':<22s} " + " ".join(f"{c:>12s}" for c in categories)
    sep = "-" * len(header)
    lines = [header, sep]

    for er in results:
        by_cat = er.quality_by_category()
        cols = " ".join(f"{by_cat.get(c, 0.0):>12.2f}" for c in categories)
        lines.append(f"{er.solver_name:<22s} {cols}")

    return "\n".join(lines)


def format_provider_table(results: list[EvalResult]) -> str:
    """Format provider distribution for each solver."""
    all_providers = sorted({r.provider for er in results for r in er.results})
    header = f"{'Solver':<22s} " + " ".join(f"{p:>10s}" for p in all_providers)
    sep = "-" * len(header)
    lines = [header, sep]

    for er in results:
        dist = er.provider_distribution()
        cols = " ".join(f"{dist.get(p, 0):>10d}" for p in all_providers)
        lines.append(f"{er.solver_name:<22s} {cols}")

    return "\n".join(lines)
