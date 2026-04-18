"""World protocol and state types for Interactive Fiction environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Observation:
    """What the agent observes after taking an action."""

    text: str
    score: int
    location: str | None = None
    inventory: tuple[str, ...] = ()
    objective: str | None = None
    intermediate_reward: float = 0.0


# Opaque snapshot — each World implementation defines its own internal format.
StateSnapshot = Any


@runtime_checkable
class World(Protocol):
    """Abstract interface for an IF environment."""

    def reset(self) -> Observation:
        """Start or restart the game. Returns initial observation."""
        ...

    def step(self, action: str) -> tuple[Observation, float, bool]:
        """Execute an action. Returns (observation, reward, done)."""
        ...

    def valid_actions(self) -> list[str]:
        """Return the list of valid actions in the current state."""
        ...

    def save(self) -> StateSnapshot:
        """Snapshot the current game state (for peek-without-consuming)."""
        ...

    def restore(self, snapshot: StateSnapshot) -> None:
        """Restore a previously saved game state."""
        ...
