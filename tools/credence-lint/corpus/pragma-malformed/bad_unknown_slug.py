# Role: body
"""Unknown slug — the slug must match a CLAUDE.md precedent."""
from skin.client import SkinClient


def show(skin: SkinClient, state_id: str) -> str:
    return f"{round(skin.weights(state_id)[0] * 100, 1)}%"  # credence-lint: allow — precedent:made-up-slug — no such precedent
