# Role: body
"""Sorting weights for a display-only top-K log line is allowed with pragma."""
import logging
from skin.client import SkinClient

log = logging.getLogger(__name__)


def log_top_k(skin: SkinClient, state_id: str, k: int) -> None:
    w = skin.weights(state_id)
    pairs = list(enumerate(w))
    # credence-lint: allow — precedent:sort-for-display — result is printed, not used for branching
    pairs.sort(key=lambda p: p[1], reverse=True)
    for i, wi in pairs[:k]:
        log.info("  [%d] %.4f", i, wi)
