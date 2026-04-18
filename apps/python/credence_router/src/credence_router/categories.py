"""Category definitions and inference functions.

Re-exports benchmark categories from credence, and provides generic
keyword-based and Router-backed category inference functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from credence_agents.environment.categories import (
    CATEGORIES,
    NUM_CATEGORIES,
)

__all__ = [
    "CATEGORIES",
    "NUM_CATEGORIES",
    "make_keyword_category_infer_fn",
    "make_router_category_infer_fn",
]


_BENCHMARK_PATTERNS: dict[str, re.Pattern] = {
    "numerical": re.compile(
        r"(?:\bcalcul|\bcomput|\bhow many\b|\bhow much\b|\bwhat is \d|"
        r"\bsquare root\b|\bsum of\b|\barea\b|\bradius\b|"
        r"\binvest|\btip on\b|\d+\s*[\+\-\*\/\%\^]\s*\d+|\d+%)",
        re.IGNORECASE,
    ),
    "recent_events": re.compile(
        r"\b(202[0-9]|recent|latest|current|this year|last year|"
        r"who won the 20|hosted the 20|released .* in 20)\b",
        re.IGNORECASE,
    ),
    "misconceptions": re.compile(
        r"(?:\btrue or false\b|\bcommon belief\b|\bmyth\b|"
        r"\bdo .* really\b|\bis it true\b|\bdoes .* have a\b|"
        r"\bvisible from space\b|percent of the brain|"
        r"\bmemory span\b|\bmother reject|\bonly use\b)",
        re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"\b(if .* then|therefore|conclude|logic|implies|"
        r"can we conclude|probability|minimum number|"
        r"how long does it take .* machines|"
        r"all but|overtake|missing dollar|"
        r"counterfeit)\b",
        re.IGNORECASE,
    ),
}


def make_keyword_category_infer_fn(
    categories: tuple[str, ...],
    patterns: dict[str, re.Pattern | list[re.Pattern]] | None = None,
    default_category: str | None = None,
    match_boost: float = 9.0,
    count_matches: bool = False,
) -> Callable[[str], NDArray[np.float64]]:
    """Return a keyword-based category inference function.

    Given a question, returns a probability distribution over categories
    based on regex pattern matches.

    Args:
        categories: The category names.
        patterns: Mapping from category name to regex pattern(s). Each key
            must be present in ``categories``. Values may be a single compiled
            pattern or a list of patterns (any match triggers the boost).
        default_category: If no patterns match, this category gets a mild
            (+1.0) boost. If None, the distribution stays uniform on no match.
        match_boost: Weight added to a category when its pattern matches.
        count_matches: If False (default), any match gives a flat boost.
            If True, boost is proportional to total match count across all
            patterns for that category (``match_boost * total_matches``).
    """
    if patterns is None:
        patterns = _BENCHMARK_PATTERNS
        default_category = default_category or "factual"

    n = len(categories)
    cat_index = {name: i for i, name in enumerate(categories)}

    # Normalise patterns to lists
    pattern_lists: dict[str, list[re.Pattern]] = {}
    for cat_name, pat in patterns.items():
        if isinstance(pat, list):
            pattern_lists[cat_name] = pat
        else:
            pattern_lists[cat_name] = [pat]

    def infer(question_text: str) -> NDArray[np.float64]:
        weights = np.ones(n, dtype=np.float64)
        for cat_name, pats in pattern_lists.items():
            if cat_name in cat_index:
                if count_matches:
                    total = sum(len(p.findall(question_text)) for p in pats)
                    weights[cat_index[cat_name]] += match_boost * total
                elif any(p.search(question_text) for p in pats):
                    weights[cat_index[cat_name]] += match_boost
        if default_category is not None and weights.max() == 1.0:
            if default_category in cat_index:
                weights[cat_index[default_category]] += 1.0
        return weights / weights.sum()

    return infer


LLM_CATEGORIES = ("code", "reasoning", "creative", "factual", "chat")

_LLM_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "code": re.compile(
        r"(?:```|\bdef\b|\bclass\b|\bfunction\b|\bimport\b|\breturn\b|"
        r"\.py\b|\.js\b|\.ts\b|\.rs\b|\.go\b|\.jl\b|"
        r"\bcode\b|\bimplement\b|\brefactor\b|\bdebug\b|\bbug\b|\bfix\b|"
        r"\bAPI\b|\bSDK\b|\bCLI\b|\bgit\b|\bdocker\b|\bkubernetes\b|"
        r"\bsyntax\b|\bcompile\b|\btest\b|\bdeploy\b)",
        re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"\b(why|explain|compare|contrast|what if|suppose|"
        r"pros and cons|trade.?off|implication|consequence|"
        r"should I|which is better|analyze|evaluate|"
        r"cause|effect|reason|because|therefore|however)\b",
        re.IGNORECASE,
    ),
    "creative": re.compile(
        r"\b(write|draft|compose|create|design|brainstorm|"
        r"story|poem|essay|letter|email|blog|article|"
        r"imagine|invent|generate|suggest names|tagline|slogan|"
        r"rewrite|rephrase|tone|style)\b",
        re.IGNORECASE,
    ),
    "factual": re.compile(
        r"\b(what is|who is|when did|where is|how many|how much|"
        r"define|definition|list|name the|capital of|"
        r"population|founded|invented|discovered)\b",
        re.IGNORECASE,
    ),
}


def make_llm_category_infer_fn() -> Callable[[str], NDArray[np.float64]]:
    """Return a category inference function for LLM query routing."""
    return make_keyword_category_infer_fn(
        LLM_CATEGORIES,
        patterns=_LLM_CATEGORY_PATTERNS,
        default_category="chat",
    )


def make_router_category_infer_fn(
    categories: tuple[str, ...],
    keyword_patterns: dict[str, re.Pattern],
    llm_classifier=None,
    latency_weight: float = 0.0,
) -> Callable[[str], NDArray[np.float64]]:
    """Return a category inference function backed by a Router.

    Creates a Router whose candidates are the categories themselves.
    Keyword tools vote for free; an optional LLM classifier tool is
    only invoked when VOI exceeds its cost.

    Args:
        categories: The category names (also used as candidates).
        keyword_patterns: {category_name: compiled_regex} for keyword tools.
        llm_classifier: Optional Tool implementing the Tool protocol that
            classifies questions into categories. Gated by VOI.
        latency_weight: $/second penalty for tool latency.

    Returns:
        A callable ``(question_text: str) -> NDArray`` returning a probability
        distribution over categories.
    """
    from credence_router.router import Router
    from credence_router.tools.keyword_category import KeywordCategoryTool

    tools = [KeywordCategoryTool(cat, pat, categories) for cat, pat in keyword_patterns.items()]
    if llm_classifier is not None:
        tools.append(llm_classifier)

    router = Router(
        tools=tools,
        categories=("classification",),
        latency_weight=latency_weight,
    )

    def infer(question_text: str) -> NDArray[np.float64]:
        answer = router.solve(question_text, candidates=categories)
        return np.array(answer.answer_posterior)

    # Expose the router for state persistence
    infer.router = router  # type: ignore[attr-defined]
    return infer
