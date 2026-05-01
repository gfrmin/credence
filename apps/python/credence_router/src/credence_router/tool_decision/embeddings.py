# Role: embed tools and look up nearest neighbours for prior smoothing.
"""Tool embedding (cache by content hash) + kNN-weighted Beta prior.

The embedding text concatenates name + description + parameter schema JSON.
The cache key is SHA-256 of that text. The prior for an unseen (model, tool)
query is a similarity-weighted average over the K nearest observed
(model, tool) cells whose embeddings are stored.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from credence_router.tool_decision.state import ToolDecisionState


def tool_content_hash(spec: dict) -> str:
    """Stable SHA-256 over the canonical JSON of name+description+parameters."""
    canonical = json.dumps(
        {
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _embedding_text(spec: dict) -> str:
    return "\n".join(
        [
            f"name: {spec.get('name', '')}",
            f"description: {spec.get('description', '')}",
            f"parameters: {json.dumps(spec.get('parameters', {}), sort_keys=True)}",
        ]
    )


def embed_tool(
    spec: dict,
    *,
    state: ToolDecisionState,
    embed_fn: Callable[[str], NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Return the embedding for a tool spec, caching in state."""
    h = tool_content_hash(spec)
    cached = state.get_embedding(h)
    if cached is not None:
        return cached
    vec = embed_fn(_embedding_text(spec)).astype(np.float32)
    state.put_embedding(h, vec)
    return vec


def knn_smoothed_prior(
    *,
    target_embedding: NDArray[np.float32],
    model_id: str,
    state: ToolDecisionState,
    k: int,
    tool_name_to_hash: dict[str, str] | None = None,
) -> tuple[float, float]:
    """kNN-weighted Beta prior for an unseen (model, tool) query.

    Pulls from cells of the same model_id (we don't share across models in v0).
    Weight is max(cosine_similarity, 0); pure-orthogonal neighbours contribute zero.
    Falls back to (1.0, 1.0) when no positive-weight neighbours exist.
    """
    name_to_hash = tool_name_to_hash or {}

    candidates: list[tuple[float, float, float]] = []  # (weight, alpha, beta)
    for m, t in state.iter_observed_pairs():
        if m != model_id:
            continue
        h = name_to_hash.get(t)
        if h is None:
            continue
        emb = state.get_embedding(h)
        if emb is None:
            continue
        denom = float(np.linalg.norm(target_embedding) * np.linalg.norm(emb)) + 1e-9
        cos = float(target_embedding @ emb) / denom
        weight = max(cos, 0.0)
        if weight <= 0.0:
            continue
        a, b = state.get_beta(m, t)
        candidates.append((weight, a, b))

    if not candidates:
        return 1.0, 1.0

    candidates.sort(reverse=True)
    top = candidates[:k]
    total_w = sum(w for w, _, _ in top)
    if total_w <= 0.0:
        return 1.0, 1.0
    alpha = sum(w * a for w, a, _ in top) / total_w
    beta = sum(w * b for w, _, b in top) / total_w
    return alpha, beta
