"""Tests for IF information-gathering tools."""

from bayesian_if.tools import (
    ExamineTool,
    InventoryTool,
    LLMAdvisorTool,
    LookTool,
    _best_action_matching,
    _extract_keywords,
    _extract_verb,
    _parse_action,
    _score_actions,
)
from bayesian_if.world import Observation
from tests.mock_world import MockWorld


# ---------------------------------------------------------------------------
# Legacy helpers (backward compat)
# ---------------------------------------------------------------------------

def test_extract_keywords():
    text = "A rusty key sits on the old wooden table."
    kws = _extract_keywords(text)
    assert "rusty" in kws
    assert "key" in kws
    assert "table" in kws
    # Stop words excluded
    assert "the" not in kws
    assert "on" not in kws


def test_best_action_matching():
    actions = ["go north", "take key", "look", "wait"]
    assert _best_action_matching(actions, ["key", "rusty"]) == 1
    assert _best_action_matching(actions, ["north", "door"]) == 0
    assert _best_action_matching(actions, ["zzz"]) is None


# ---------------------------------------------------------------------------
# Phase 3: _parse_action + _score_actions
# ---------------------------------------------------------------------------

def test_parse_action_known_verb():
    verb, objects = _parse_action("take key")
    assert verb == "take"
    assert objects == ["key"]


def test_parse_action_multi_word_object():
    verb, objects = _parse_action("open rusty chest")
    assert verb == "open"
    assert objects == ["rusty", "chest"]


def test_parse_action_go_direction():
    verb, objects = _parse_action("go north")
    assert verb == "go"
    assert objects == ["north"]


def test_parse_action_unknown_verb():
    verb, objects = _parse_action("frobnicate widget")
    assert verb is None
    assert objects == ["frobnicate", "widget"]


def test_parse_action_empty():
    verb, objects = _parse_action("")
    assert verb is None
    assert objects == []


def test_score_actions_verb_plus_noun():
    actions = ["go north", "take key", "examine key"]
    assert _score_actions(actions, "take", ["key"]) == 1


def test_score_actions_word_boundary():
    actions = ["take monkey", "take key"]
    assert _score_actions(actions, None, ["key"]) == 1


def test_score_actions_verb_plus_direction():
    actions = ["go north", "go south"]
    assert _score_actions(actions, "go", ["north"]) == 0


def test_score_actions_no_match():
    actions = ["go north", "look"]
    assert _score_actions(actions, None, ["zzz"]) is None


def test_score_actions_tied_scores_not_always_first():
    """When scores are tied, random tie-breaking should not always pick index 0."""
    actions = ["go north", "go south", "go east", "go west"]
    results = {_score_actions(actions, "go", []) for _ in range(100)}
    # All four actions have score 3.0 (verb match only); should see variety
    assert len(results) > 1, f"Tie-breaking always returned the same index: {results}"


def test_score_actions_verb_only():
    actions = ["take key", "examine key"]
    # When verb matches but no noun info, prefer verb match
    assert _score_actions(actions, "take", []) == 0


def test_score_actions_monkey_vs_key():
    """'key' should NOT match 'monkey' with word-boundary matching."""
    actions = ["take monkey", "take key", "go north"]
    result = _score_actions(actions, None, ["key"])
    assert result == 1


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------

def test_look_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = world.valid_actions()

    tool = LookTool()
    result = tool.query(world, obs, actions)
    # Should return a valid index or None
    assert result is None or (0 <= result < len(actions))


def test_look_tool_does_not_consume_turn():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = world.valid_actions()

    LookTool().query(world, obs, actions)

    # State should be unchanged
    actions_after = world.valid_actions()
    assert actions == actions_after


def test_examine_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A rusty key sits on the Table.", score=0, location="Start Room")
    actions = world.valid_actions()

    tool = ExamineTool()
    result = tool.query(world, obs, actions)
    assert result is None or (0 <= result < len(actions))


def test_inventory_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    world.step("take key")
    obs = Observation(text="You have a key.", score=5, location="Start Room", inventory=("key",))
    actions = world.valid_actions()

    tool = InventoryTool()
    result = tool.query(world, obs, actions)
    assert result is None or (0 <= result < len(actions))


def test_inventory_tool_does_not_consume_turn():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()

    InventoryTool().query(world, obs, actions)

    world_actions = world.valid_actions()
    assert actions == world_actions


def test_llm_advisor_with_mock():
    """LLM advisor with a mock generate function."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room with a key.", score=0)
    actions = world.valid_actions()

    # Mock LLM that always returns "1"
    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "1")
    result = tool.query(world, obs, actions)
    assert result == 1


def test_llm_advisor_handles_bad_response():
    """LLM advisor handles non-numeric responses gracefully."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()

    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "I think you should explore")
    result = tool.query(world, obs, actions)
    assert result is None


def test_llm_advisor_handles_out_of_range():
    """LLM advisor handles out-of-range indices."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()

    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "999")
    result = tool.query(world, obs, actions)
    assert result is None


def test_tool_config_conversion():
    """Tools convert to credence ToolConfig correctly."""
    tool = LookTool()
    config = tool.to_tool_config()
    assert config.cost == 0.0
    assert len(config.coverage_by_category) == 5
    assert all(0.0 <= c <= 1.0 for c in config.coverage_by_category)


# ---------------------------------------------------------------------------
# Phase 1: Structured observation in tools
# ---------------------------------------------------------------------------

def test_inventory_tool_uses_structured_inventory():
    """InventoryTool should use observation.inventory directly to match actions."""
    world = MockWorld()
    world.reset()
    world.step("take key")

    obs = Observation(
        text="Hallway.", score=5, location="Start Room", inventory=("key",)
    )
    # Valid actions include "go north" — "key" should NOT match "go north"
    # but if there were an action like "unlock door with key", it would match
    actions = ["go north", "go south", "open chest"]
    tool = InventoryTool()
    result = tool.query(world, obs, actions)
    # "key" doesn't word-boundary match any of these action objects, so None
    assert result is None


def test_inventory_tool_matches_action_with_item():
    """InventoryTool matches inventory items to actions using word boundaries."""
    world = MockWorld()
    world.reset()
    world.step("take key")

    obs = Observation(
        text="Room.", score=5, location="Treasure Room", inventory=("key",)
    )
    actions = ["go south", "unlock chest with key", "look"]
    tool = InventoryTool()
    result = tool.query(world, obs, actions)
    assert result == 1  # "key" matches "unlock chest with key"


def test_inventory_tool_falls_back_to_save_restore():
    """InventoryTool falls back to save/restore when inventory is empty."""
    world = MockWorld()
    world.reset()

    obs = Observation(text="A room.", score=0, location="Start Room", inventory=())
    actions = world.valid_actions()

    tool = InventoryTool()
    # Should not crash — falls back to save/restore "inventory" command
    result = tool.query(world, obs, actions)
    assert result is None or (0 <= result < len(actions))


def test_examine_tool_considers_inventory_items():
    """ExamineTool._pick_targets should consider inventory items."""
    obs = Observation(
        text="A bare room.", score=5, location="Room", inventory=("golden key",)
    )
    # No examinable verbs in actions
    actions = ["go north", "go south", "wait"]
    targets = ExamineTool._pick_targets(obs, actions)
    assert "golden key" in targets


def test_look_tool_incorporates_location():
    """LookTool should incorporate location into keyword scoring."""
    world = MockWorld()
    world.reset()
    # The mock world's Start Room description mentions "north"
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = world.valid_actions()

    tool = LookTool()
    result = tool.query(world, obs, actions)
    # Should return a valid index (location is used as additional keyword)
    assert result is None or (0 <= result < len(actions))


# ---------------------------------------------------------------------------
# Phase 2: LLM prompt context
# ---------------------------------------------------------------------------

def test_llm_prompt_includes_location():
    """LLM prompt should include location when available."""
    prompts: list[str] = []

    def capture_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "0"

    world = MockWorld()
    world.reset()
    obs = Observation(
        text="A dark room.", score=0, location="Kitchen",
        inventory=("lamp",), objective="Find the treasure.",
    )
    actions = ["go north", "take key"]

    tool = LLMAdvisorTool(generate_fn=capture_prompt)
    tool.query(world, obs, actions)

    assert len(prompts) == 1
    assert "Kitchen" in prompts[0]
    assert "lamp" in prompts[0]
    assert "Find the treasure" in prompts[0]


def test_llm_prompt_includes_history():
    """LLM prompt should include recent history when provided."""
    prompts: list[str] = []

    def capture_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "0"

    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = ["go north", "look"]
    history = [("go south", "You arrive at a garden."), ("take lamp", "Taken.")]

    tool = LLMAdvisorTool(generate_fn=capture_prompt)
    tool.query(world, obs, actions, history=history)

    assert len(prompts) == 1
    assert "go south" in prompts[0]
    assert "garden" in prompts[0]
    assert "take lamp" in prompts[0]


# ---------------------------------------------------------------------------
# Ollama URL normalization
# ---------------------------------------------------------------------------

def test_ollama_normalize_bare_ip():
    from bayesian_if.ollama import _normalize_base_url

    assert _normalize_base_url("100.114.52.102") == "http://100.114.52.102:11434"


def test_ollama_normalize_ip_with_port():
    from bayesian_if.ollama import _normalize_base_url

    assert _normalize_base_url("192.168.1.1:8080") == "http://192.168.1.1:8080"


def test_ollama_normalize_full_url():
    from bayesian_if.ollama import _normalize_base_url

    assert _normalize_base_url("http://localhost:11434") == "http://localhost:11434"


def test_ollama_normalize_trailing_slash():
    from bayesian_if.ollama import _normalize_base_url

    assert _normalize_base_url("http://localhost:11434/") == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Phase 2: ExamineTool novelty filtering
# ---------------------------------------------------------------------------

def test_examine_tool_avoids_recently_examined():
    """ExamineTool should skip targets that appear in recent history."""
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = ["examine insect", "examine key", "go north"]
    history = [("examine insect", "It's a bug.")]

    targets = ExamineTool._pick_targets(obs, actions, history=history)
    assert targets[0] == "key"  # "insect" was already examined


def test_examine_tool_fallback_when_all_examined():
    """ExamineTool should still return something when all targets were tried."""
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = ["examine insect", "examine key", "go north"]
    history = [("examine insect", "It's a bug."), ("examine key", "A rusty key.")]

    targets = ExamineTool._pick_targets(obs, actions, history=history)
    # All examined → falls back to full candidates list
    assert targets[0] in ("insect", "key")


# ---------------------------------------------------------------------------
# Phase 4: LLM prompt includes failed actions
# ---------------------------------------------------------------------------

def test_llm_prompt_includes_failed_actions():
    """LLM prompt should mention failed actions to avoid repeating them."""
    prompts: list[str] = []

    def capture_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "0"

    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = ["go north", "take key"]
    failed = {"examine insect", "take insect"}

    tool = LLMAdvisorTool(generate_fn=capture_prompt)
    tool.query(world, obs, actions, failed_actions=failed)

    assert len(prompts) == 1
    assert "examine insect" in prompts[0]
    assert "take insect" in prompts[0]
    assert "Avoid repeating" in prompts[0]


# ---------------------------------------------------------------------------
# Objective-aware scoring
# ---------------------------------------------------------------------------

def test_score_actions_with_objective_nouns():
    """Objective nouns should boost matching actions."""
    actions = ["take key", "go north", "examine table"]
    # Without objective
    idx1 = _score_actions(actions, verb=None, nouns=["table"])
    assert idx1 is not None
    assert actions[idx1] == "examine table"

    # With objective mentioning "key"
    idx2 = _score_actions(actions, verb=None, nouns=["table"], objective_nouns=["key"])
    assert idx2 is not None
    assert actions[idx2] == "take key"  # key gets +2 objective + 0 noun vs table's +1 noun


# ---------------------------------------------------------------------------
# Verb extraction
# ---------------------------------------------------------------------------

def test_extract_verb_from_text():
    """_extract_verb should find the first IF verb in text."""
    assert _extract_verb("You should take the key from the table.") == "take"
    assert _extract_verb("The door is locked.") is None
    assert _extract_verb("Look around the room carefully.") == "look"


# ---------------------------------------------------------------------------
# Multi-examine
# ---------------------------------------------------------------------------

def test_examine_tool_multi_targets():
    """ExamineTool._pick_targets should return multiple targets."""
    obs = Observation(text="A room.", score=0, location="Room")
    actions = ["examine key", "examine chest", "examine table", "go north"]
    targets = ExamineTool._pick_targets(obs, actions, max_targets=3)
    assert len(targets) == 3
    assert "key" in targets
    assert "chest" in targets
    assert "table" in targets
