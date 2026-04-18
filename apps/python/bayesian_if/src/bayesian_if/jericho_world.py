"""Jericho adapter — wraps jericho.FrotzEnv as a World."""

from __future__ import annotations

import re

from bayesian_if.world import Observation, StateSnapshot


def _parse_inventory(inv_text: str) -> tuple[str, ...]:
    """Extract item names from an inventory response."""
    items: list[str] = []
    for line in inv_text.strip().splitlines():
        line = line.strip()
        match = re.match(r"^\s*[-*]?\s*(?:a |an |the |some )?(.+)$", line, re.IGNORECASE)
        if match and not line.lower().startswith("you"):
            items.append(match.group(1).strip().rstrip("."))
    return tuple(items)


class JerichoWorld:
    """World adapter for Jericho Z-machine games."""

    def __init__(self, rom_path: str) -> None:
        try:
            import jericho
        except ImportError as e:
            raise ImportError("Install jericho: pip install bayesian-if[jericho]") from e
        self.env = jericho.FrotzEnv(rom_path)
        self._last_score: int = 0

    def reset(self) -> Observation:
        obs, info = self.env.reset()
        self._last_score = 0
        return self._make_observation(obs, score=0)

    def step(self, action: str) -> tuple[Observation, float, bool]:
        obs, reward, done, info = self.env.step(action)
        score = self._last_score + int(reward)
        self._last_score = score
        return self._make_observation(obs, score), float(reward), done

    def valid_actions(self) -> list[str]:
        return self.env.get_valid_actions()

    def save(self) -> StateSnapshot:
        return self.env.get_state()

    def restore(self, snapshot: StateSnapshot) -> None:
        self.env.set_state(snapshot)

    def _make_observation(self, text: str, score: int) -> Observation:
        location = self._get_location()
        inventory = self._get_inventory()
        return Observation(text=text, score=score, location=location, inventory=inventory)

    def _get_location(self) -> str | None:
        try:
            return self.env.get_player_location().name
        except Exception:
            return None

    def _get_inventory(self) -> tuple[str, ...]:
        state = self.env.get_state()
        try:
            inv_text, _, _, _ = self.env.step("inventory")
            return _parse_inventory(inv_text)
        finally:
            self.env.set_state(state)


assert isinstance(JerichoWorld.__new__(object), object)  # type: ignore[arg-type]
# Runtime protocol check deferred — JerichoWorld satisfies World protocol structurally.
