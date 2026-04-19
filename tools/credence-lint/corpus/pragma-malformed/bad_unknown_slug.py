# Role: body
"""Unknown slug — the slug must match a CLAUDE.md precedent."""
from skin.client import SkinClient


def show(skin: SkinClient, state_id: str) -> str:
    w = skin.weights(state_id)
    # credence-lint: allow — precedent:made-up-slug — no such precedent
    return f"{round(w[0] * 100, 1)}%"
