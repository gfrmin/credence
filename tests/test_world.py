"""Tests for the World protocol and MockWorld."""

from bayesian_if.world import Observation, World
from tests.mock_world import MockWorld


def test_mock_world_satisfies_protocol():
    world = MockWorld()
    assert isinstance(world, World)


def test_reset_returns_observation():
    world = MockWorld()
    obs = world.reset()
    assert isinstance(obs, Observation)
    assert obs.score == 0
    assert obs.location == "Start Room"
    assert "door" in obs.text.lower() or "room" in obs.text.lower()


def test_valid_actions_at_start():
    world = MockWorld()
    world.reset()
    actions = world.valid_actions()
    assert "look" in actions
    assert "take key" in actions
    assert "go north" in actions


def test_step_take_key():
    world = MockWorld()
    world.reset()
    obs, reward, done = world.step("take key")
    assert reward == 5.0
    assert not done
    assert obs.score == 5
    assert "key" in obs.inventory


def test_step_navigate():
    world = MockWorld()
    world.reset()
    obs, _, _ = world.step("go north")
    assert obs.location == "Hallway"
    obs, _, _ = world.step("go north")
    assert obs.location == "Treasure Room"


def test_win_game():
    world = MockWorld()
    world.reset()
    world.step("take key")
    world.step("go north")   # +1 exploration
    world.step("go north")   # +1 exploration
    obs, reward, done = world.step("open chest")
    assert done
    assert reward == 10.0
    assert obs.score == 17  # 5 (key) + 1 + 1 (exploration) + 10 (chest)


def test_save_restore():
    world = MockWorld()
    world.reset()
    world.step("take key")
    snapshot = world.save()

    # Move forward
    world.step("go north")
    obs_after_move, _, _ = world.step("look")
    assert "Hallway" in (obs_after_move.location or "")

    # Restore
    world.restore(snapshot)
    actions = world.valid_actions()
    assert "take key" not in actions  # already taken before save
    assert "go north" in actions
    obs, _, _ = world.step("look")
    assert obs.location == "Start Room"


def test_save_restore_isolation():
    """Modifying restored state doesn't affect the snapshot."""
    world = MockWorld()
    world.reset()
    snapshot = world.save()
    world.step("take key")  # modifies state
    world.restore(snapshot)
    # Key should not be taken
    assert "take key" in world.valid_actions()
