"""IF situation categories and category inference function."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from credence_router.categories import make_keyword_category_infer_fn

CATEGORIES: tuple[str, ...] = ("exploration", "puzzle", "inventory", "dialogue", "combat")
NUM_CATEGORIES: int = len(CATEGORIES)

# Keyword patterns per category (index-aligned with CATEGORIES).
_CATEGORY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "exploration": [
        re.compile(r"\b(dark|passage|room|door|north|south|east|west|up|down|corridor|hall)\b", re.I),
    ],
    "puzzle": [
        re.compile(r"\b(locked|key|lever|button|switch|mechanism|puzzle|open|close|insert)\b", re.I),
    ],
    "inventory": [
        re.compile(r"\b(take|drop|carry|holding|pick up|put down|wearing|remove)\b", re.I),
    ],
    "dialogue": [
        re.compile(r"\b(says?|asks?|tells?|speak|talk|reply|replies|greet|hello)\b", re.I),
    ],
    "combat": [
        re.compile(r"\b(attack|kill|fight|sword|troll|monster|hit|slash|wound|dead)\b", re.I),
    ],
}


def infer_category_hint(obs: object) -> str | None:
    """Infer a category hint from structured observation state.

    Returns a category name if confident, else None. The hint gets a +9.0 boost
    in credence's solve_question.
    """
    # Check inventory for puzzle-related items
    inventory = getattr(obs, "inventory", ())
    if inventory:
        puzzle_items = re.compile(r"\b(key|lever|gem|ring|orb|crystal|rod|wand)\b", re.I)
        for item in inventory:
            if puzzle_items.search(item):
                return "puzzle"

    text = getattr(obs, "text", "")
    location = getattr(obs, "location", "") or ""
    combined = f"{text} {location}"

    # Combat signals are high-priority
    if re.search(r"\b(attack|fight|monster|troll|sword|kill)\b", combined, re.I):
        return "combat"
    # Dialogue signals
    if re.search(r"\b(says?|asks?|tells?|speak|talk)\b", combined, re.I):
        return "dialogue"
    # Exploration signals from location
    if re.search(r"\b(corridor|passage|path|trail|road|bridge)\b", combined, re.I):
        return "exploration"

    return None


def make_if_category_infer_fn(
    categories: tuple[str, ...] = CATEGORIES,
) -> Callable[[str], NDArray[np.float64]]:
    """Return a function that classifies game-state text into a category distribution.

    Uses keyword matching with count-proportional weighting via the shared
    make_keyword_category_infer_fn with count_matches=True.
    """
    return make_keyword_category_infer_fn(
        categories, _CATEGORY_PATTERNS, match_boost=2.0, count_matches=True,
    )
