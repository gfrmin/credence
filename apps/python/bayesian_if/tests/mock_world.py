"""Mock IF world for testing: 3 rooms, 2 objects, a goal."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from bayesian_if.world import Observation


@dataclass
class MockState:
    """Internal state for the mock world."""

    room: int = 0  # 0=start, 1=hallway, 2=treasure_room
    inventory: list[str] = field(default_factory=list)
    key_taken: bool = False
    chest_open: bool = False
    score: int = 0
    done: bool = False
    visited_rooms: set[int] = field(default_factory=lambda: {0})


ROOMS = {
    0: {
        "name": "Start Room",
        "desc": "A plain room with a door to the north. A rusty key sits on the table.",
        "objects": ["key", "table"],
    },
    1: {
        "name": "Hallway",
        "desc": "A long dark corridor. Doors lead south and north.",
        "objects": [],
    },
    2: {
        "name": "Treasure Room",
        "desc": "A locked chest sits in the corner of the room. The door leads south.",
        "objects": ["chest"],
    },
}


def _valid_actions(state: MockState) -> list[str]:
    """Compute valid actions for the current state."""
    actions = ["look", "wait", "inventory"]
    if state.room == 0:
        actions.append("go north")
        if not state.key_taken:
            actions.append("take key")
    elif state.room == 1:
        actions.extend(["go south", "go north"])
    elif state.room == 2:
        actions.append("go south")
        if not state.chest_open and "key" in state.inventory:
            actions.append("open chest")
    return actions


def _step(state: MockState, action: str) -> tuple[str, float]:
    """Apply action to state, return (text, reward)."""
    action = action.lower().strip()

    if action == "look":
        room = ROOMS[state.room]
        return room["desc"], 0.0

    if action == "wait":
        return "Time passes.", 0.0

    if action == "inventory":
        if state.inventory:
            return "You are carrying: " + ", ".join(state.inventory), 0.0
        return "You are empty-handed.", 0.0

    if action == "take key" and state.room == 0 and not state.key_taken:
        state.key_taken = True
        state.inventory.append("key")
        state.score += 5
        return "You pick up the rusty key.", 5.0

    if action == "go north":
        if state.room == 0:
            state.room = 1
            reward = 1.0 if 1 not in state.visited_rooms else 0.0
            state.visited_rooms.add(1)
            if reward > 0:
                state.score += 1
            return ROOMS[1]["desc"], reward
        elif state.room == 1:
            state.room = 2
            reward = 1.0 if 2 not in state.visited_rooms else 0.0
            state.visited_rooms.add(2)
            if reward > 0:
                state.score += 1
            return ROOMS[2]["desc"], reward
    elif action == "go south":
        if state.room == 1:
            state.room = 0
            return ROOMS[0]["desc"], 0.0
        elif state.room == 2:
            state.room = 1
            return ROOMS[1]["desc"], 0.0

    if action == "open chest" and state.room == 2 and "key" in state.inventory:
        state.chest_open = True
        state.score += 10
        state.done = True
        return "You unlock the chest and find treasure! You win!", 10.0

    return "Nothing happens.", 0.0


class MockWorld:
    """A 3-room mock IF world for testing."""

    def __init__(self) -> None:
        self._state = MockState()

    def reset(self) -> Observation:
        self._state = MockState()
        room = ROOMS[0]
        return Observation(
            text=room["desc"],
            score=0,
            location=room["name"],
            inventory=(),
        )

    def step(self, action: str) -> tuple[Observation, float, bool]:
        text, reward = _step(self._state, action)
        room = ROOMS[self._state.room]
        obs = Observation(
            text=text,
            score=self._state.score,
            location=room["name"],
            inventory=tuple(self._state.inventory),
        )
        return obs, reward, self._state.done

    def valid_actions(self) -> list[str]:
        return _valid_actions(self._state)

    def save(self) -> MockState:
        return copy.deepcopy(self._state)

    def restore(self, snapshot: MockState) -> None:
        self._state = copy.deepcopy(snapshot)
