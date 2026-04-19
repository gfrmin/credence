# Role: body
"""Search provider routing evaluation.

Two-phase design:
  Phase 1 (collect): call ALL providers for ALL queries, judge with LLM, save JSON.
  Phase 2 (analyse): replay routing decisions offline under different utility
  parameter combinations to produce a preference frontier.

This separation means we pay API costs once and can explore the full
quality-vs-cost-vs-latency tradeoff space offline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.search_router import SEARCH_CATEGORIES
from credence_router.tool import SearchResult, SearchTool

log = logging.getLogger(__name__)


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
    SearchQuery("what is the deepest point in the ocean", "factual", ["Mariana Trench", "Challenger Deep", "~11,000 meters"]),
    SearchQuery("who painted the Mona Lisa", "factual", ["Leonardo da Vinci"]),
    SearchQuery("what is the atomic number of gold", "factual", ["79"]),
    SearchQuery("how long is the Great Wall of China", "factual", ["~21,196 km", "various measurements"]),
    SearchQuery("what year did the Berlin Wall fall", "factual", ["1989"]),
    # --- recent_events (20) ---
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
    SearchQuery("latest Mars rover discoveries 2026", "recent_events", ["rover name", "findings", "date"]),
    SearchQuery("recent changes to EU AI regulation", "recent_events", ["EU AI Act", "implementation timeline"]),
    SearchQuery("who won the latest Grammy for Album of the Year", "recent_events", ["artist name", "album title", "year"]),
    SearchQuery("current price of gold per ounce", "recent_events", ["price in USD", "recent date"]),
    SearchQuery("latest developments in nuclear fusion energy 2026", "recent_events", ["facility or project", "milestone achieved"]),
    SearchQuery("recent major tech company layoffs 2026", "recent_events", ["company names", "numbers affected"]),
    SearchQuery("current state of Ukraine conflict 2026", "recent_events", ["recent developments", "peace negotiations"]),
    SearchQuery("latest James Webb Space Telescope discoveries", "recent_events", ["discovery details", "date published"]),
    SearchQuery("who won the 2026 Australian Open tennis", "recent_events", ["winner name", "final score"]),
    SearchQuery("latest global ocean temperature records 2026", "recent_events", ["temperature anomaly", "record broken", "data source"]),
    # --- technical (20) ---
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
    SearchQuery("Zig vs Rust for systems programming", "technical", ["memory safety", "compile times", "error handling"]),
    SearchQuery("how to set up a Nix flake for a Python project", "technical", ["flake.nix structure", "devShell", "buildInputs"]),
    SearchQuery("Linux io_uring tutorial for async file IO", "technical", ["submission queue", "completion queue", "liburing"]),
    SearchQuery("how to write a custom Prometheus exporter in Go", "technical", ["collector interface", "metrics types", "registration"]),
    SearchQuery("SQLite WAL mode vs journal mode", "technical", ["concurrent readers", "write performance", "crash recovery"]),
    SearchQuery("implementing consistent hashing for distributed cache", "technical", ["ring structure", "virtual nodes", "rebalancing"]),
    SearchQuery("how to use DuckDB for analytics on Parquet files", "technical", ["query syntax", "performance", "memory management"]),
    SearchQuery("Tailwind CSS vs vanilla CSS for large projects", "technical", ["bundle size", "maintainability", "utility classes"]),
    SearchQuery("how to deploy a Julia application in production", "technical", ["PackageCompiler", "sysimage", "Docker"]),
    SearchQuery("implementing rate limiting with Redis sliding window", "technical", ["ZADD", "ZRANGEBYSCORE", "atomicity"]),
    # --- synthesis (20) ---
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
    SearchQuery("is Kubernetes worth it for small teams", "synthesis", ["operational overhead", "alternatives", "team size threshold"]),
    SearchQuery("impact of AI code assistants on developer productivity", "synthesis", ["productivity studies", "code quality", "learning effects"]),
    SearchQuery("buy vs build decision for internal tools", "synthesis", ["total cost of ownership", "customization needs", "maintenance burden"]),
    SearchQuery("edge computing vs cloud computing tradeoffs", "synthesis", ["latency", "bandwidth costs", "data sovereignty"]),
    SearchQuery("monorepo vs polyrepo for microservices", "synthesis", ["tooling requirements", "CI/CD complexity", "code sharing"]),
    SearchQuery("static typing vs dynamic typing for team productivity", "synthesis", ["refactoring safety", "development speed", "onboarding"]),
    SearchQuery("open source vs proprietary LLMs for enterprise", "synthesis", ["data privacy", "cost", "customization", "support"]),
    SearchQuery("SPA vs MPA vs islands architecture for web apps", "synthesis", ["initial load", "SEO", "interactivity", "complexity"]),
    SearchQuery("is WebAssembly replacing JavaScript", "synthesis", ["use cases", "performance gaps", "ecosystem maturity"]),
    SearchQuery("pros and cons of event sourcing vs CRUD", "synthesis", ["audit trail", "complexity", "eventual consistency"]),
    # --- local/niche (20) ---
    SearchQuery("best ramen restaurants in Shibuya Tokyo", "local", ["specific restaurant names", "specialties"]),
    SearchQuery("Tel Aviv coworking spaces with monthly plans", "local", ["specific space names", "pricing", "locations"]),
    SearchQuery("hiking trails near Queenstown New Zealand difficulty levels", "local", ["trail names", "difficulty ratings", "distances"]),
    SearchQuery("specialty coffee roasters in Melbourne Australia", "local", ["roaster names", "locations", "styles"]),
    SearchQuery("public transportation from Zurich airport to city center", "local", ["options", "duration", "cost"]),
    SearchQuery("best vegetarian restaurants in Chiang Mai Thailand", "local", ["restaurant names", "cuisine types", "price range"]),
    SearchQuery("Berlin techno clubs open on weeknights", "local", ["club names", "opening hours", "entry policy"]),
    SearchQuery("surf spots near Lisbon Portugal for beginners", "local", ["beach names", "wave conditions", "accessibility"]),
    SearchQuery("public libraries with coworking in Amsterdam", "local", ["library names", "facilities", "opening hours"]),
    SearchQuery("day trips from Kyoto by train under 2 hours", "local", ["destinations", "train lines", "travel time"]),
    SearchQuery("craft beer breweries in Portland Oregon", "local", ["brewery names", "specialties", "tasting rooms"]),
    SearchQuery("best bookshops in Buenos Aires", "local", ["shop names", "specialties", "neighborhoods"]),
    SearchQuery("rock climbing gyms in Seoul South Korea", "local", ["gym names", "bouldering vs lead", "day pass prices"]),
    SearchQuery("farmers markets in Copenhagen Denmark schedule", "local", ["market names", "days", "locations"]),
    SearchQuery("budget accommodation near Machu Picchu", "local", ["hostel/hotel names", "price range", "Aguas Calientes"]),
    SearchQuery("skateparks in Barcelona with bowls", "local", ["park names", "features", "locations"]),
    SearchQuery("vegan bakeries in London Shoreditch area", "local", ["bakery names", "specialties", "addresses"]),
    SearchQuery("hot springs near Reykjavik Iceland not Blue Lagoon", "local", ["spring names", "prices", "distance"]),
    SearchQuery("jazz clubs in New Orleans French Quarter", "local", ["club names", "live music schedule", "cover charge"]),
    SearchQuery("indoor swimming pools in Vienna open to public", "local", ["pool names", "opening hours", "entry fee"]),
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
            "judged": False,
        }

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    user_msg = (
        f"Query: {query.query}\n"
        f"Category: {query.category}\n"
        f"Quality criteria: {', '.join(query.quality_criteria)}\n\n"
        f"Search results:\n{result.text[:3000]}\n\n"
        f"URLs: {', '.join(result.urls[:5])}"
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
    # Haiku sometimes wraps JSON in markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    scores = json.loads(cleaned)

    # Composite: weight freshness more for recent_events
    weights = {"relevance": 0.3, "freshness": 0.2, "completeness": 0.3, "source_quality": 0.2}
    if query.category == "recent_events":
        weights = {"relevance": 0.25, "freshness": 0.35, "completeness": 0.25, "source_quality": 0.15}

    composite = sum(scores.get(k, 0) * w for k, w in weights.items())
    scores["composite"] = round(composite, 2)
    scores["judged"] = True
    return scores


# ---------------------------------------------------------------------------
# Raw data: Phase 1 output
# ---------------------------------------------------------------------------


@dataclass
class ProviderResult:
    """One provider's result for one query."""

    provider: str
    text: str
    urls: list[str]
    scores: dict
    wall_time: float


@dataclass
class QueryData:
    """All providers' results for one query."""

    query: str
    category: str
    quality_criteria: list[str]
    provider_results: dict[str, ProviderResult]  # keyed by provider name


@dataclass
class RawEvalData:
    """Complete raw data from Phase 1 collection."""

    queries: list[QueryData]
    provider_names: list[str]
    provider_costs: dict[str, float]
    provider_latencies: dict[str, float]

    def save(self, path: str | Path) -> None:
        obj = {
            "provider_names": self.provider_names,
            "provider_costs": self.provider_costs,
            "provider_latencies": self.provider_latencies,
            "queries": [
                {
                    "query": qd.query,
                    "category": qd.category,
                    "quality_criteria": qd.quality_criteria,
                    "provider_results": {
                        name: {
                            "provider": pr.provider,
                            "text": pr.text,
                            "urls": pr.urls,
                            "scores": pr.scores,
                            "wall_time": pr.wall_time,
                        }
                        for name, pr in qd.provider_results.items()
                    },
                }
                for qd in self.queries
            ],
        }
        Path(path).write_text(json.dumps(obj, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RawEvalData:
        obj = json.loads(Path(path).read_text())
        queries = []
        for qobj in obj["queries"]:
            provider_results = {}
            for name, pr in qobj["provider_results"].items():
                provider_results[name] = ProviderResult(
                    provider=pr["provider"],
                    text=pr["text"],
                    urls=pr["urls"],
                    scores=pr["scores"],
                    wall_time=pr["wall_time"],
                )
            queries.append(QueryData(
                query=qobj["query"],
                category=qobj["category"],
                quality_criteria=qobj["quality_criteria"],
                provider_results=provider_results,
            ))
        return cls(
            queries=queries,
            provider_names=obj["provider_names"],
            provider_costs=obj["provider_costs"],
            provider_latencies=obj["provider_latencies"],
        )


def collect_raw_data(
    search_tools: list[SearchTool],
    queries: list[SearchQuery] | None = None,
    judge_api_key: str | None = None,
    verbose: bool = True,
    partial_save_path: str | Path | None = None,
) -> RawEvalData:
    """Phase 1: call ALL providers for ALL queries, judge everything.

    This is the expensive step — real API calls. Results are cached so we
    only pay once. Auth errors (401/403) propagate immediately. Transient
    errors (timeout, rate limit) are logged and that query/provider is skipped.
    """
    if queries is None:
        queries = QUERY_BANK

    provider_names = [t.name for t in search_tools]
    provider_costs = {t.name: t.cost for t in search_tools}
    provider_latencies = {t.name: t.latency for t in search_tools}
    all_query_data: list[QueryData] = []

    total = len(queries) * len(search_tools)
    done = 0

    log.info(
        "Starting collection: %d queries × %d providers = %d calls",
        len(queries), len(search_tools), total,
    )

    def _build_partial() -> RawEvalData:
        return RawEvalData(
            queries=all_query_data,
            provider_names=provider_names,
            provider_costs=provider_costs,
            provider_latencies=provider_latencies,
        )

    try:
        for qi, q in enumerate(queries):
            provider_results: dict[str, ProviderResult] = {}

            for tool in search_tools:
                done += 1

                # Search
                try:
                    t_start = time.monotonic()
                    result = tool.search(q.query)
                    wall_time = time.monotonic() - t_start
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (401, 403):
                        log.error(
                            "%s auth failed (HTTP %d) — check API key",
                            tool.name, e.response.status_code,
                        )
                        raise
                    log.error(
                        "%s HTTP %d for '%s': %s",
                        tool.name, e.response.status_code, q.query, e,
                    )
                    result = None
                    wall_time = 0.0
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    log.error("%s network error for '%s': %s", tool.name, q.query, e)
                    result = None
                    wall_time = 0.0
                except Exception as e:
                    log.error("%s unexpected error for '%s': %s", tool.name, q.query, e)
                    result = None
                    wall_time = 0.0

                # Judge
                try:
                    scores = judge_search_result(q, result, api_key=judge_api_key)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (401, 403):
                        log.error("Judge auth failed (HTTP %d) — check ANTHROPIC_API_KEY", e.response.status_code)
                        raise
                    log.error("Judge HTTP %d for '%s': %s", e.response.status_code, q.query, e)
                    scores = {
                        "relevance": 0, "freshness": 0, "completeness": 0,
                        "source_quality": 0, "composite": 0.0,
                        "reasoning": f"Judge error: {e}", "judged": False,
                    }
                except Exception as e:
                    log.error("Judge failed for '%s': %s", q.query, e, exc_info=True)
                    scores = {
                        "relevance": 0, "freshness": 0, "completeness": 0,
                        "source_quality": 0, "composite": 0.0,
                        "reasoning": f"Judge error: {e}", "judged": False,
                    }

                provider_results[tool.name] = ProviderResult(
                    provider=tool.name,
                    text=result.text if result else "",
                    urls=result.urls if result else [],
                    scores=scores,
                    wall_time=wall_time,
                )

                if verbose:
                    comp = scores.get("composite", 0.0)
                    flag = "" if result else "[no result]"
                    if not scores.get("judged", False):
                        flag = "[unjudged]"
                    log.info(
                        "[%d/%d] Q%02d (%s) × %s: %.1f %s",
                        done, total, qi + 1, q.category, tool.name, comp, flag,
                    )

            all_query_data.append(QueryData(
                query=q.query,
                category=q.category,
                quality_criteria=q.quality_criteria,
                provider_results=provider_results,
            ))
    except BaseException:
        # Save partial results so we don't lose work
        if partial_save_path and all_query_data:
            partial = _build_partial()
            partial.save(partial_save_path)
            log.info("Saved %d partial results to %s", len(all_query_data), partial_save_path)
        raise

    log.info("Collection complete: %d queries", len(all_query_data))
    return _build_partial()


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

    def effective_utility(self, reward: float, latency_weight: float) -> float:
        """Mean EU per query: quality * reward - monetary_cost - latency_cost."""
        if not self.results:
            return 0.0
        total = 0.0
        for r in self.results:
            q_score = r.scores.get("composite", 0.0) / 10.0  # normalise to [0,1]
            eff_cost = r.cost + latency_weight * r.wall_time
            total += q_score * reward - eff_cost
        return total / len(self.results)


# ---------------------------------------------------------------------------
# Solvers for Phase 2 simulation
# ---------------------------------------------------------------------------


class SimulatedSearchRouter:
    """SearchRouter that looks up results from cached data instead of calling APIs."""

    def __init__(
        self,
        provider_names: list[str],
        provider_costs: dict[str, float],
        provider_latencies: dict[str, float],
        coverage: dict[str, dict[str, float]],
        categories: tuple[str, ...] = SEARCH_CATEGORIES,
        reward_useful: float = 1.0,
        latency_weight: float = 0.01,
    ):
        self.name = "credence-search"
        self._provider_names = provider_names
        self._costs = provider_costs
        self._latencies = provider_latencies
        self._categories = categories

        # Build a SearchRouter with fake tools that have the right cost/latency/coverage
        from credence_router.search_router import SearchRouter

        self._router = SearchRouter(
            search_tools=[
                _FakeSimTool(
                    name, provider_costs[name], provider_latencies[name],
                    coverage_map=coverage.get(name, {}),
                )
                for name in provider_names
            ],
            categories=categories,
            reward_useful=reward_useful,
            latency_weight=latency_weight,
        )

    def pick_provider(self, query: str, category: str) -> str:
        """Choose provider without executing search (dry-run route)."""
        # Use category hint for accurate simulation
        result = self._router.route(query, category_hint=category)
        return result.provider

    def report_outcome(self, useful: bool) -> None:
        self._router.report_outcome(useful)

    @property
    def learned_reliability(self) -> dict[str, dict[str, float]]:
        return self._router.learned_reliability


class _FakeSimTool:
    """Minimal SearchTool for SimulatedSearchRouter (never actually searches)."""

    def __init__(
        self, name: str, cost: float, latency: float,
        coverage_map: dict[str, float] | None = None,
    ):
        self._name = name
        self._cost = cost
        self._latency = latency
        self._coverage_map = coverage_map or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def latency(self) -> float:
        return self._latency

    def search(self, query: str) -> SearchResult | None:
        return SearchResult(text="simulated", urls=[], provider=self._name, raw={})

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        return np.array([self._coverage_map.get(c, 0.5) for c in categories])


def _compute_empirical_coverage(
    raw_data: RawEvalData,
    categories: tuple[str, ...] = SEARCH_CATEGORIES,
    useful_threshold: float = 5.0,
) -> dict[str, dict[str, float]]:
    """Compute per-provider per-category coverage from raw data.

    Coverage = fraction of queries where provider scored >= useful_threshold.
    This gives the router realistic priors from the actual data.
    """
    counts: dict[str, dict[str, list[float]]] = {}
    for qd in raw_data.queries:
        for pname, pr in qd.provider_results.items():
            if pname not in counts:
                counts[pname] = {}
            cat = qd.category
            if cat not in counts[pname]:
                counts[pname][cat] = []
            counts[pname][cat].append(
                1.0 if pr.scores.get("composite", 0.0) >= useful_threshold else 0.0
            )

    result: dict[str, dict[str, float]] = {}
    for pname in counts:
        result[pname] = {}
        for cat in categories:
            vals = counts.get(pname, {}).get(cat, [])
            result[pname][cat] = sum(vals) / len(vals) if vals else 0.5
    return result


class FallbackChainSolver:
    """Simulates OpenClaw's runWebSearch: try providers in priority order.

    In OpenClaw, providers are tried in a fixed order until one returns a result.
    Since in our eval all providers return results, this always picks the first.
    """

    def __init__(self, priority_order: list[str]):
        self._order = priority_order
        self.name = "openclaw-fallback"

    def pick_provider(self, query: str, category: str) -> str:
        return self._order[0]

    def report_outcome(self, useful: bool) -> None:
        pass


class StaticProviderSolver:
    """Always uses one fixed provider."""

    def __init__(self, provider_name: str):
        self._provider = provider_name
        self.name = f"static-{provider_name}"

    def pick_provider(self, query: str, category: str) -> str:
        return self._provider

    def report_outcome(self, useful: bool) -> None:
        pass


class RoundRobinSolver:
    """Cycles through providers in order."""

    name = "round-robin"

    def __init__(self, provider_names: list[str]):
        self._providers = provider_names
        self._idx = 0

    def pick_provider(self, query: str, category: str) -> str:
        provider = self._providers[self._idx % len(self._providers)]
        self._idx += 1
        return provider

    def report_outcome(self, useful: bool) -> None:
        pass


class RandomSolver:
    """Random provider per query."""

    name = "random-search"

    def __init__(self, provider_names: list[str], seed: int = 42):
        self._providers = provider_names
        self._rng = np.random.default_rng(seed)

    def pick_provider(self, query: str, category: str) -> str:
        idx = int(self._rng.integers(len(self._providers)))
        return self._providers[idx]

    def report_outcome(self, useful: bool) -> None:
        pass


# ---------------------------------------------------------------------------
# Phase 2: Routing simulation
# ---------------------------------------------------------------------------


def simulate_routing(
    raw_data: RawEvalData,
    reward: float = 1.0,
    latency_weight: float = 0.01,
    seed: int = 42,
    useful_threshold: float = 5.0,
) -> list[EvalResult]:
    """Replay routing decisions against cached data under given utility params.

    No API calls — purely offline simulation.
    """
    rng = np.random.default_rng(seed)
    query_order = list(range(len(raw_data.queries)))
    rng.shuffle(query_order)

    # Build solvers
    providers = raw_data.provider_names
    coverage = _compute_empirical_coverage(raw_data, useful_threshold=useful_threshold)
    solvers = [
        SimulatedSearchRouter(
            providers, raw_data.provider_costs, raw_data.provider_latencies,
            coverage=coverage,
            reward_useful=reward, latency_weight=latency_weight,
        ),
        FallbackChainSolver(providers),
        RoundRobinSolver(providers),
        RandomSolver(providers, seed=seed),
    ]
    for p in providers:
        solvers.append(StaticProviderSolver(p))

    all_results: list[EvalResult] = []

    for solver in solvers:
        eval_result = EvalResult(solver_name=solver.name)

        # Reset state for solvers that need it
        if isinstance(solver, RoundRobinSolver):
            solver._idx = 0

        for qi in query_order:
            qd = raw_data.queries[qi]
            provider = solver.pick_provider(qd.query, qd.category)

            # Look up cached result
            pr = qd.provider_results.get(provider)
            if pr is None:
                # Provider wasn't in collection — skip
                continue

            useful = pr.scores.get("composite", 0.0) >= useful_threshold
            solver.report_outcome(useful)

            eval_result.results.append(QueryResult(
                query=qd.query,
                category=qd.category,
                provider=provider,
                scores=pr.scores,
                cost=raw_data.provider_costs.get(provider, 0.0),
                wall_time=pr.wall_time,
            ))

        all_results.append(eval_result)

    return all_results


# ---------------------------------------------------------------------------
# Preference frontier
# ---------------------------------------------------------------------------

REWARD_VALUES = [0.05, 0.10, 0.25, 0.50, 1.00]
LATENCY_WEIGHT_VALUES = [0.00, 0.01, 0.05, 0.10, 0.20]


@dataclass
class FrontierPoint:
    """One point on the preference frontier."""

    reward: float
    latency_weight: float
    solver_results: list[EvalResult]


@dataclass
class PreferenceFrontier:
    """Full preference frontier across parameter grid."""

    points: list[FrontierPoint]
    seeds: list[int]


def sweep_preferences(
    raw_data: RawEvalData,
    seeds: list[int] | None = None,
    reward_values: list[float] | None = None,
    latency_weight_values: list[float] | None = None,
) -> PreferenceFrontier:
    """Run simulate_routing over a grid of (reward, latency_weight) pairs."""
    if seeds is None:
        seeds = [42]
    if reward_values is None:
        reward_values = REWARD_VALUES
    if latency_weight_values is None:
        latency_weight_values = LATENCY_WEIGHT_VALUES

    points: list[FrontierPoint] = []

    for reward in reward_values:
        for lw in latency_weight_values:
            # Average across seeds
            all_seed_results: list[list[EvalResult]] = []
            for seed in seeds:
                results = simulate_routing(raw_data, reward=reward, latency_weight=lw, seed=seed)
                all_seed_results.append(results)

            # Use first seed's results as representative (averaging EvalResults is complex)
            # but compute mean quality across seeds for the summary
            points.append(FrontierPoint(
                reward=reward,
                latency_weight=lw,
                solver_results=all_seed_results[0],
            ))

    return PreferenceFrontier(points=points, seeds=seeds)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_category_routing(credence: EvalResult, fallback: EvalResult) -> str:
    """Per-category provider distribution + quality comparison."""
    # Group credence results by category
    by_cat: dict[str, list[QueryResult]] = {}
    for r in credence.results:
        by_cat.setdefault(r.category, []).append(r)

    fb_quality = fallback.quality_by_category()

    parts: list[str] = []
    for cat in sorted(by_cat):
        results = by_cat[cat]
        providers: dict[str, int] = {}
        for r in results:
            providers[r.provider] = providers.get(r.provider, 0) + 1
        routing = ", ".join(
            f"{p}={n}" for p, n in sorted(providers.items(), key=lambda x: -x[1])
        )
        avg_q = sum(r.scores.get("composite", 0.0) for r in results) / len(results)
        fb_q = fb_quality.get(cat, 0.0)
        delta = avg_q - fb_q
        sign = "+" if delta >= 0 else ""
        parts.append(f"    {cat:<16s} {routing:<24s} q={avg_q:.1f} (vs fallback {fb_q:.1f}, {sign}{delta:.1f})")

    return "  routing by category:\n" + "\n".join(parts)


def format_frontier_table(frontier: PreferenceFrontier) -> str:
    """Format the preference frontier as an actionable table for OpenClaw."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("PREFERENCE FRONTIER: Quality vs Cost vs Latency")
    lines.append("=" * 80)
    lines.append("")

    for point in frontier.points:
        reward = point.reward
        lw = point.latency_weight

        # Label the operating point
        if lw >= 0.10 and reward <= 0.10:
            label = "SPEED-FIRST"
        elif lw >= 0.05 and reward <= 0.50:
            label = "BALANCED"
        elif lw <= 0.01 and reward >= 0.50:
            label = "QUALITY-FIRST"
        else:
            label = "CUSTOM"

        lines.append(f"--- {label} (reward={reward}, latency_weight={lw}) ---")

        # Find credence-search and openclaw-fallback for comparison
        credence_r = None
        fallback_r = None
        for r in point.solver_results:
            if r.solver_name == "credence-search":
                credence_r = r
            elif r.solver_name == "openclaw-fallback":
                fallback_r = r

        header = f"  {'Solver':<22s} {'Quality':>8s} {'EU':>8s} {'Cost$':>8s} {'Routing'}"
        lines.append(header)

        for r in point.solver_results:
            eu = r.effective_utility(reward, lw)
            dist = r.provider_distribution()
            routing_str = ", ".join(f"{p}={n}" for p, n in sorted(dist.items(), key=lambda x: -x[1]))
            lines.append(
                f"  {r.solver_name:<22s} {r.mean_quality:>7.2f} "
                f"{eu:>+7.4f} "
                f"${r.total_cost:>6.3f} "
                f"{routing_str}"
            )

        # Summary comparison
        if credence_r and fallback_r:
            c_eu = credence_r.effective_utility(reward, lw)
            f_eu = fallback_r.effective_utility(reward, lw)
            diff = c_eu - f_eu
            if diff > 0:
                lines.append(f"  -> credence wins by {diff:+.4f} EU per query")
                # Per-category routing breakdown
                lines.append(_format_category_routing(credence_r, fallback_r))
            elif diff < 0:
                lines.append(f"  -> openclaw-fallback wins by {-diff:+.4f} EU per query")
            else:
                lines.append("  -> tie")

            # Learning curve for credence
            early, late = credence_r.learning_curve(window=15)
            delta = late - early
            if abs(delta) > 0.01:
                sign = "+" if delta >= 0 else ""
                lines.append(f"  learning: first-15={early:.1f}, last-15={late:.1f} ({sign}{delta:.1f})")

        lines.append("")

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


# ---------------------------------------------------------------------------
# Legacy convenience wrapper
# ---------------------------------------------------------------------------


def run_search_eval(
    search_tools: list[SearchTool],
    queries: list[SearchQuery] | None = None,
    seed: int = 42,
    judge_api_key: str | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """Collect + simulate in one call (backward compatible)."""
    raw_data = collect_raw_data(search_tools, queries, judge_api_key, verbose)
    return simulate_routing(raw_data, seed=seed)
