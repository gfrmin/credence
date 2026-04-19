# Role: body
"""Information-gathering tools for IF — each returns a recommended action index."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from credence_agents import ToolConfig

from bayesian_if.categories import CATEGORIES
from bayesian_if.world import Observation, World

# Known IF verbs for action parsing.
IF_VERBS = frozenset({
    "go", "take", "open", "examine", "push", "pull", "turn", "read",
    "drop", "put", "close", "unlock", "insert", "look", "inventory",
    "wait", "eat", "drink", "wear", "remove", "give", "ask", "tell",
    "attack", "tie", "cut", "climb", "enter", "exit", "search",
})


class IFTool(ABC):
    """Base class for IF information-gathering tools."""

    name: str
    cost: float

    @abstractmethod
    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        failed_actions: set[str] | None = None,
    ) -> int | None:
        """Query this tool and return a recommended action index, or None."""
        ...

    def to_tool_config(self, categories: tuple[str, ...] = CATEGORIES) -> ToolConfig:
        """Convert to a credence ToolConfig with per-category coverage."""
        return ToolConfig(
            cost=self.cost,
            coverage_by_category=self._coverage(categories),
        )

    @abstractmethod
    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        """Return P(tool returns an answer | category) for each category."""
        ...


# ---------------------------------------------------------------------------
# Phase 3: Principled action matching
# ---------------------------------------------------------------------------

def _parse_action(action: str) -> tuple[str | None, list[str]]:
    """Split an IF action into (verb, object_words).

    Known IF verbs are recognised; everything after the verb is objects.
    If the first word is not a known verb, verb is None and all words are objects.
    """
    words = action.lower().split()
    if not words:
        return None, []
    verb = words[0] if words[0] in IF_VERBS else None
    objects = words[1:] if verb else words
    return verb, objects


def _score_actions(
    valid_actions: list[str],
    verb: str | None,
    nouns: list[str],
    objective_nouns: list[str] | None = None,
) -> int | None:
    """Score each valid action against a recommended verb + nouns.

    - Verb match: +3.0 (high weight — "take key" beats "examine key" when tool said "take")
    - Objective noun match: +2.0 per noun (between verb and regular noun priority)
    - Object match: +1.0 per noun, using word-boundary matching (\\bkey\\b not substring)
    - Returns argmax index if any action scores > 0, else None.
    """
    scores: list[float] = []
    for action in valid_actions:
        a_verb, a_objects = _parse_action(action)
        score = 0.0
        if verb and a_verb == verb:
            score += 3.0
        obj_text = " ".join(a_objects)
        if objective_nouns:
            for noun in objective_nouns:
                if re.search(r"\b" + re.escape(noun) + r"\b", obj_text, re.I):
                    score += 2.0
        for noun in nouns:
            if re.search(r"\b" + re.escape(noun) + r"\b", obj_text, re.I):
                score += 1.0
        scores.append(score)
    best_score = max(scores) if scores else 0.0
    if best_score <= 0.0:
        return None
    tied = [i for i, s in enumerate(scores) if s == best_score]
    return random.choice(tied)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _best_action_matching(
    valid_actions: list[str], keywords: list[str]
) -> int | None:
    """Return the index of the valid action best matching the keywords, or None."""
    best_idx: int | None = None
    best_score = 0
    for i, action in enumerate(valid_actions):
        action_lower = action.lower()
        score = sum(1 for kw in keywords if kw.lower() in action_lower)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _extract_keywords(text: str) -> list[str]:
    """Extract notable words from descriptive text."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "to", "of", "and", "you", "it", "that", "this",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _extract_verb(text: str) -> str | None:
    """Extract the first IF verb from descriptive text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for word in words:
        if word in IF_VERBS:
            return word
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class LookTool(IFTool):
    """Peek at 'look' output in a save/restore bracket."""

    name = "look"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        failed_actions: set[str] | None = None,
    ) -> int | None:
        snapshot = world.save()
        try:
            obs, _, _ = world.step("look")
            keywords = _extract_keywords(obs.text)
            # Incorporate location for directional disambiguation
            if observation.location:
                keywords.append(observation.location.lower())
            # Extract verbs from look text for better action matching
            verb = _extract_verb(obs.text)
            obj_nouns = _extract_keywords(observation.objective) if observation.objective else None
            return _score_actions(valid_actions, verb=verb, nouns=keywords,
                                  objective_nouns=obj_nouns)
        finally:
            world.restore(snapshot)

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        # Look is most useful for exploration, somewhat for puzzle
        coverage = {
            "exploration": 0.9,
            "puzzle": 0.5,
            "inventory": 0.3,
            "dialogue": 0.4,
            "combat": 0.4,
        }
        return np.array([coverage.get(c, 0.5) for c in categories])


class ExamineTool(IFTool):
    """Examine the most novel visible object in a save/restore bracket."""

    name = "examine"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        failed_actions: set[str] | None = None,
    ) -> int | None:
        targets = self._pick_targets(observation, valid_actions, history=history, max_targets=3)
        if not targets:
            return None

        obj_nouns = _extract_keywords(observation.objective) if observation.objective else None
        all_keywords: list[str] = []

        for target in targets:
            snapshot = world.save()
            try:
                obs, _, _ = world.step(f"examine {target}")
                all_keywords.extend(_extract_keywords(obs.text))
                all_keywords.append(target)
            finally:
                world.restore(snapshot)

        return _score_actions(valid_actions, verb=None, nouns=all_keywords,
                              objective_nouns=obj_nouns)

    @staticmethod
    def _pick_targets(
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        max_targets: int = 3,
    ) -> list[str]:
        """Pick up to max_targets objects to examine — prefer novel items."""
        candidates: list[str] = []
        for action in valid_actions:
            match = re.match(
                r"(?:take|examine|open|push|pull|turn|read)\s+(.+)", action, re.I
            )
            if match:
                candidates.append(match.group(1).strip())

        # Filter out targets that appear in recent history actions
        if history and candidates:
            tried = {act.lower() for act, _ in history}
            novel = [c for c in candidates if not any(c.lower() in t for t in tried)]
            if novel:
                candidates = novel

        if candidates:
            return candidates[:max_targets]

        # Consider inventory items as examination targets
        if observation.inventory:
            return list(observation.inventory[:max_targets])
        # Fallback: look for nouns in observation text
        nouns = re.findall(r"\b([A-Z][a-z]+)\b", observation.text)
        return [n.lower() for n in nouns[:max_targets]] if nouns else []

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        coverage = {
            "exploration": 0.5,
            "puzzle": 0.9,
            "inventory": 0.6,
            "dialogue": 0.3,
            "combat": 0.3,
        }
        return np.array([coverage.get(c, 0.5) for c in categories])


class InventoryTool(IFTool):
    """Check inventory in a save/restore bracket."""

    name = "inventory"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        failed_actions: set[str] | None = None,
    ) -> int | None:
        obj_nouns = _extract_keywords(observation.objective) if observation.objective else None
        # Use structured inventory when available
        if observation.inventory:
            items = [item.lower() for item in observation.inventory]
            return _score_actions(valid_actions, verb=None, nouns=items,
                                  objective_nouns=obj_nouns)
        # Fallback: save/restore bracket
        snapshot = world.save()
        try:
            obs, _, _ = world.step("inventory")
            keywords = _extract_keywords(obs.text)
            return _score_actions(valid_actions, verb=None, nouns=keywords,
                                  objective_nouns=obj_nouns)
        finally:
            world.restore(snapshot)

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        coverage = {
            "exploration": 0.2,
            "puzzle": 0.5,
            "inventory": 0.9,
            "dialogue": 0.1,
            "combat": 0.4,
        }
        return np.array([coverage.get(c, 0.3) for c in categories])


class LLMAdvisorTool(IFTool):
    """Ask an LLM which action to take."""

    name = "llm_advisor"
    cost = 0.02

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        model: str = "llama3.1",
    ) -> None:
        if generate_fn is not None:
            self._generate = generate_fn
        else:
            from bayesian_if.ollama import ollama_generate

            self._generate = lambda prompt: ollama_generate(prompt, model=model)

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
        failed_actions: set[str] | None = None,
    ) -> int | None:
        if not valid_actions:
            return None

        actions_str = "\n".join(f"  {i}: {a}" for i, a in enumerate(valid_actions))

        # Build rich context from structured observation + history
        context_parts: list[str] = []
        if observation.location:
            context_parts.append(f"Current location: {observation.location}")
        if observation.inventory:
            context_parts.append(f"Inventory: {', '.join(observation.inventory)}")
        if observation.objective:
            context_parts.append(f"Objective: {observation.objective}")
        if history:
            history_lines = [f"  > {act} -> {res}" for act, res in history]
            context_parts.append("Actions taken so far:\n" + "\n".join(history_lines))
        if failed_actions:
            context_parts.append(
                "Actions already tried without success: "
                + ", ".join(sorted(failed_actions))
                + "\nAvoid repeating these."
            )

        context = "\n".join(context_parts)

        prompt = f"You are playing a text adventure game.\n\nCurrent situation:\n{observation.text}\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += (
            f"Available actions:\n{actions_str}\n\n"
            f"Which action number is best? Reply with ONLY the number."
        )

        try:
            response = self._generate(prompt)
            # Extract first integer from response
            match = re.search(r"\d+", response)
            if match:
                idx = int(match.group())
                if 0 <= idx < len(valid_actions):
                    return idx
        except Exception:
            pass
        return None

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        # LLM has broad but imperfect coverage — the agent learns the truth
        return np.full(len(categories), 0.7)


DEFAULT_TOOLS: list[IFTool] = [LookTool(), ExamineTool(), InventoryTool()]
"""Default tools (no LLM). Add LLMAdvisorTool separately when Ollama is available."""
