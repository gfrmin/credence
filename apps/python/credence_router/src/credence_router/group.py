# Role: body
"""RouterGroup: multi-router state management."""

from __future__ import annotations

import json
from pathlib import Path

from credence_router.router import Router


class RouterGroup:
    """Manages multiple named Router instances with unified state persistence.

    Saves and loads all routers' learned state to/from a single JSON file,
    enabling cross-session learning across multiple routing domains.
    """

    def __init__(self, routers: dict[str, Router]):
        self._routers = routers

    def __getitem__(self, name: str) -> Router:
        return self._routers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._routers

    def __iter__(self):
        return iter(self._routers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._routers.keys())

    def save_state(self, path: str | Path) -> None:
        """Persist all router states to a single JSON file."""
        state = {name: router.save_state_dict() for name, router in self._routers.items()}
        Path(path).write_text(json.dumps(state, indent=2))

    def load_state(self, path: str | Path) -> None:
        """Restore all router states from a JSON file."""
        path = Path(path)
        if not path.exists():
            return
        state = json.loads(path.read_text())
        for name, router_state in state.items():
            if name in self._routers:
                self._routers[name].load_state_dict(router_state)

    def save_state_dict(self) -> dict:
        """Return all router states as a plain dict."""
        return {name: router.save_state_dict() for name, router in self._routers.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore all router states from a dict."""
        for name, router_state in state.items():
            if name in self._routers:
                self._routers[name].load_state_dict(router_state)
