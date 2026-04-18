"""TextWorld adapter — wraps textworld environments as a World."""

from __future__ import annotations

from bayesian_if.world import Observation, StateSnapshot


class TextWorldWorld:
    """World adapter for Microsoft TextWorld environments."""

    def __init__(self, game_file: str) -> None:
        try:
            import textworld
        except ImportError as e:
            raise ImportError("Install textworld: pip install bayesian-if[textworld]") from e

        self._tw = textworld
        request_infos = textworld.EnvInfos(
            admissible_commands=True,
            score=True,
            max_score=True,
            inventory=True,
            description=True,
            location=True,
            objective=True,
            won=True,
            lost=True,
            policy_commands=True,
            intermediate_reward=True,
        )
        self.env = textworld.start(game_file, request_infos)
        self._last_state = None
        self._game_file = game_file

    def reset(self) -> Observation:
        state = self.env.reset()
        self._last_state = state
        return self._make_observation(state)

    def step(self, action: str) -> tuple[Observation, float, bool]:
        state, score, done = self.env.step(action)
        prev_score = self._last_state.score if self._last_state else 0
        reward = float(score - prev_score)
        self._last_state = state
        return self._make_observation(state), reward, done

    def valid_actions(self) -> list[str]:
        if self._last_state is None:
            return []
        return list(self._last_state.get("admissible_commands", []))

    def save(self) -> StateSnapshot:
        return self.env.copy()

    def restore(self, snapshot: StateSnapshot) -> None:
        self.env = snapshot

    @property
    def max_score(self) -> int:
        if self._last_state is None:
            return 0
        return self._last_state.get("max_score", 0) or 0

    @property
    def policy_commands(self) -> list[str]:
        if self._last_state is None:
            return []
        return list(self._last_state.get("policy_commands", []) or [])

    @property
    def game_won(self) -> bool:
        if self._last_state is None:
            return False
        return bool(self._last_state.get("won", False))

    @property
    def objective(self) -> str | None:
        if self._last_state is None:
            return None
        return self._last_state.get("objective", None) or None

    def _make_observation(self, state) -> Observation:
        text = state.feedback
        description = state.get("description", "")
        location = self._parse_location(description)
        inv_text = state.get("inventory", "")
        inventory = self._parse_inventory(inv_text) if inv_text else ()
        score = state.score
        objective = state.get("objective", None) or None
        intermediate_reward = float(state.get("intermediate_reward", 0) or 0)
        return Observation(
            text=text, score=score, location=location, inventory=inventory,
            objective=objective, intermediate_reward=intermediate_reward,
        )

    @staticmethod
    def _parse_location(description: str) -> str | None:
        """Extract location name from TextWorld description header like '-= Attic =-'."""
        import re

        match = re.match(r"\s*-=\s*(.+?)\s*=-", description)
        return match.group(1) if match else None

    @staticmethod
    def _parse_inventory(inv_text: str) -> tuple[str, ...]:
        items: list[str] = []
        for line in inv_text.strip().splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                items.append(line.lstrip("-* ").strip())
        return tuple(items)
