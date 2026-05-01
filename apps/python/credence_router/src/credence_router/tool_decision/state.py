# Role: persistent store for Beta posteriors and tool embeddings.
"""Per-(model_id, tool_name) Beta posterior + content-hashed embedding cache.

Storage: a single JSON file. Embeddings are stored as base64-encoded float32
arrays inside the JSON to keep the on-disk footprint a single file (consistent
with credence-router's existing single-file state convention).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class ToolDecisionState:
    """In-memory mutable store backed by a JSON file."""

    def __init__(self, path: Path):
        self._path = path
        # Key: f"{model_id}\t{tool_name}" -> [alpha, beta]
        self._betas: dict[str, list[float]] = {}
        # Key: content_hash -> base64 of float32 bytes
        self._embeddings: dict[str, str] = {}

    # ---- Beta posterior ----

    def get_beta(self, model_id: str, tool_name: str) -> tuple[float, float]:
        key = f"{model_id}\t{tool_name}"
        ab = self._betas.get(key, [1.0, 1.0])
        return ab[0], ab[1]

    def update(self, model_id: str, tool_name: str, *, approved: bool) -> None:
        key = f"{model_id}\t{tool_name}"
        ab = self._betas.setdefault(key, [1.0, 1.0])
        if approved:
            ab[0] += 1.0
        else:
            ab[1] += 1.0

    def iter_observed_pairs(self) -> Iterator[tuple[str, str]]:
        for key in self._betas:
            model_id, tool_name = key.split("\t", 1)
            yield model_id, tool_name

    # ---- Embedding cache ----

    def put_embedding(self, content_hash: str, embedding: NDArray[np.float32]) -> None:
        self._embeddings[content_hash] = base64.b64encode(
            embedding.astype(np.float32).tobytes()
        ).decode("ascii")

    def get_embedding(self, content_hash: str) -> NDArray[np.float32] | None:
        b64 = self._embeddings.get(content_hash)
        if b64 is None:
            return None
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()

    # ---- Persistence ----

    def save(self) -> None:
        payload = {"betas": self._betas, "embeddings": self._embeddings}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload))

    def load(self) -> None:
        if not self._path.exists():
            return
        payload = json.loads(self._path.read_text())
        self._betas = {k: list(v) for k, v in payload.get("betas", {}).items()}
        self._embeddings = dict(payload.get("embeddings", {}))
