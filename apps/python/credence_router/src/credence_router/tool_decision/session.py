# Role: derive a stable session_id from the messages prefix.
"""Hash everything except the latest user message → session_id.

Rationale: each new request from a single conversation appends one user
message; everything before it is the stable history. Hashing the prefix
gives the same id across requests in the same conversation, and a different
id when history diverges (branching, restart).
"""
from __future__ import annotations

import hashlib
import json


def derive_session_id(messages: list[dict]) -> str:
    prefix = _prefix_excluding_trailing_user(messages)
    canonical = json.dumps(prefix, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _prefix_excluding_trailing_user(messages: list[dict]) -> list[dict]:
    if not messages:
        return []
    if messages[-1].get("role") == "user":
        return list(messages[:-1])
    return list(messages)
