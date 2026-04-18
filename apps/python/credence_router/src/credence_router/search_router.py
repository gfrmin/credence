"""SearchRouter: Bayesian web search provider routing via EU maximisation.

Provider selection is a direct EU maximisation over learned reliability:
for each provider, EU = P(useful | category) * reward - cost. The category
belief comes from keyword inference. Reliability is updated from outcome
feedback via Beta conjugate updating (numpy).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from credence_router.categories import make_keyword_category_infer_fn
from credence_router.tool import SearchResult, SearchTool

SEARCH_CATEGORIES = ("factual", "recent_events", "technical", "synthesis", "local")

_SEARCH_CATEGORY_PATTERNS = {
    "recent_events": __import__("re").compile(
        r"\b(202[0-9]|recent|latest|current|this year|last year|today|now|"
        r"who won the 20|just released|new.*202)\b",
        __import__("re").IGNORECASE,
    ),
    "technical": __import__("re").compile(
        r"\b(how to|implement|tutorial|code|programming|algorithm|"
        r"API|SDK|framework|library|docker|kubernetes|git|SQL|"
        r"async|deploy|optimize|debug|compile)\b",
        __import__("re").IGNORECASE,
    ),
    "synthesis": __import__("re").compile(
        r"\b(pros and cons|comparison|vs|versus|should I use|"
        r"best practices|trade.?offs?|impact of|implications|"
        r"when to use|advantages)\b",
        __import__("re").IGNORECASE,
    ),
    "local": __import__("re").compile(
        r"\b(restaurant|cafe|hotel|transport|near|in .*(city|town)|"
        r"coworking|hiking|trail|neighborhood|district|"
        r"Tokyo|London|Paris|Berlin|Melbourne|Zurich|Tel Aviv)\b",
        __import__("re").IGNORECASE,
    ),
}


@dataclass(frozen=True)
class SearchRouteResult:
    """Result of routing a search query through a provider."""

    query: str
    provider: str
    result: SearchResult | None
    confidence: float
    wall_time: float
    reasoning: str
    decision_trace: tuple[dict, ...] = field(default_factory=tuple)


class SearchRouter:
    """Bayesian web search provider routing.

    Learns which search provider is most reliable per query category.
    Provider selection is direct EU maximisation:

        EU(provider_i) = sum_c P(category=c) * P(useful | provider_i, c) * reward - cost_i

    where P(useful | provider_i, c) is a learned Beta posterior, and P(category=c)
    comes from keyword-based inference.
    """

    name = "credence-search"

    def __init__(
        self,
        search_tools: list[SearchTool],
        categories: tuple[str, ...] = SEARCH_CATEGORIES,
        latency_weight: float = 0.01,
        reward_useful: float = 1.0,
    ):
        self._search_tools = search_tools
        self._categories = categories
        self._latency_weight = latency_weight
        self._reward_useful = reward_useful
        self._provider_names = tuple(t.name for t in search_tools)

        # Category inference
        self._category_infer = make_keyword_category_infer_fn(
            categories,
            patterns=_SEARCH_CATEGORY_PATTERNS,
            default_category="factual",
        )

        # Per-provider per-category Beta parameters (alpha, beta)
        # Start with coverage priors: tools that declare high coverage
        # get a mildly informative prior
        n_cats = len(categories)
        self._alpha = np.ones((len(search_tools), n_cats))
        self._beta = np.ones((len(search_tools), n_cats))

        for i, tool in enumerate(search_tools):
            cov = tool.coverage(categories)
            # Use coverage as pseudo-counts: high coverage → mild prior toward reliable
            self._alpha[i] = 1.0 + cov
            self._beta[i] = 1.0 + (1.0 - cov)

        # Track last route for outcome reporting
        self._last_provider_idx: int | None = None
        self._last_category_dist: NDArray[np.float64] | None = None

    def _effective_cost(self, tool_idx: int) -> float:
        tool = self._search_tools[tool_idx]
        return tool.cost + self._latency_weight * tool.latency

    def _reliability_mean(self, tool_idx: int) -> NDArray[np.float64]:
        """E[P(useful | provider, category)] from Beta posterior."""
        return self._alpha[tool_idx] / (self._alpha[tool_idx] + self._beta[tool_idx])

    def _compute_eu(self, cat_dist: NDArray[np.float64]) -> NDArray[np.float64]:
        """EU for each provider given category distribution."""
        n_tools = len(self._search_tools)
        eus = np.empty(n_tools)
        for i in range(n_tools):
            rel = self._reliability_mean(i)
            expected_reward = float(np.dot(cat_dist, rel)) * self._reward_useful
            eus[i] = expected_reward - self._effective_cost(i)
        return eus

    def route(
        self,
        query: str,
        category_hint: str | None = None,
    ) -> SearchRouteResult:
        """Route a search query to the optimal provider and execute it."""
        t_start = time.monotonic()

        # Infer category distribution
        if category_hint and category_hint in self._categories:
            cat_dist = np.zeros(len(self._categories))
            cat_dist[self._categories.index(category_hint)] = 1.0
        else:
            cat_dist = self._category_infer(query)

        self._last_category_dist = cat_dist

        # EU maximisation over providers
        eus = self._compute_eu(cat_dist)
        provider_idx = int(np.argmax(eus))
        self._last_provider_idx = provider_idx

        provider = self._search_tools[provider_idx]

        # Build reasoning trace
        lines = ["Provider EU:"]
        for i, tool in enumerate(self._search_tools):
            rel = self._reliability_mean(i)
            marker = " <-" if i == provider_idx else ""
            lines.append(
                f"  {tool.name}: EU={eus[i]:+.4f} "
                f"(rel={np.dot(cat_dist, rel):.3f}, cost={self._effective_cost(i):.4f}){marker}"
            )
        cat_str = ", ".join(
            f"{c}={p:.2f}" for c, p in zip(self._categories, cat_dist) if p > 0.01
        )
        lines.append(f"Category: [{cat_str}]")

        trace_dicts = tuple(
            {"provider": t.name, "eu": float(eus[i]), "selected": i == provider_idx}
            for i, t in enumerate(self._search_tools)
        )

        # Execute the actual search
        search_result = provider.search(query)

        wall_time = time.monotonic() - t_start
        return SearchRouteResult(
            query=query,
            provider=provider.name,
            result=search_result,
            confidence=float(np.max(eus)),
            wall_time=wall_time,
            reasoning="\n".join(lines),
            decision_trace=trace_dicts,
        )

    def report_outcome(self, useful: bool) -> None:
        """Report whether the last search result was useful.

        Updates the Beta posterior for the selected provider, weighted
        by the category distribution from the query.
        """
        if self._last_provider_idx is None or self._last_category_dist is None:
            return

        idx = self._last_provider_idx
        cat_dist = self._last_category_dist

        # Update Beta parameters weighted by category distribution
        if useful:
            self._alpha[idx] += cat_dist
        else:
            self._beta[idx] += cat_dist

    @property
    def learned_reliability(self) -> dict[str, dict[str, float]]:
        """Current learned reliability per provider per category."""
        result: dict[str, dict[str, float]] = {}
        for i, tool in enumerate(self._search_tools):
            rel = self._reliability_mean(i)
            result[tool.name] = {
                cat: float(rel[j]) for j, cat in enumerate(self._categories)
            }
        return result

    def save_state(self, path: str | Path) -> None:
        """Persist learned state to disk."""
        state = {
            "alpha": self._alpha.tolist(),
            "beta": self._beta.tolist(),
            "provider_names": list(self._provider_names),
            "categories": list(self._categories),
        }
        Path(path).write_text(json.dumps(state))

    def load_state(self, path: str | Path) -> None:
        """Restore learned state from disk."""
        state = json.loads(Path(path).read_text())
        self._alpha = np.array(state["alpha"])
        self._beta = np.array(state["beta"])
