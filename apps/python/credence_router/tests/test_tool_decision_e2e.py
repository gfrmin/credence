# Role: end-to-end test of the tool-decision mode against the FastAPI app.
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CREDENCE_TOOL_DECISION", "1")
    monkeypatch.setenv("CREDENCE_TOOL_DECISION_STATE", str(tmp_path / "td.json"))
    # ask_cost < 0 so that voi_ask (≈0.333 at max-entropy) beats eu_stop (0.5)
    # on cold start: voi_ask - ask_cost ≈ 0.333 + 0.3 = 0.633 > 0.5 → ASK.
    # After one approval (α=2,β=1): voi_ask ≈ 0.222, voi_ask - ask_cost ≈ 0.522
    # < eu_execute ≈ 0.665 → EXECUTE.  This exercises both paths deterministically.
    monkeypatch.setenv("CREDENCE_ASK_COST", "-0.3")
    # No real API keys → available_models() returns [] so the Julia skin
    # process is never spawned, and embed_fn falls back to pseudo-embeddings.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CREDENCE_EMBEDDING_KEY", raising=False)

    from credence_router import server
    from credence_router.tool_decision.decide import Action, DecideInputs

    # Stub out the LLM forwarder + model selector at module scope so the test
    # doesn't require any upstream provider.
    monkeypatch.setattr(
        server,
        "_default_llm_fn",
        lambda: (
            lambda messages, tools, model_id: {
                "model_id": model_id,
                "text": "I'll run a shell command.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    }
                ],
                "usage_cost": 0.001,
            }
        ),
    )
    monkeypatch.setattr(
        server, "_default_select_model_fn", lambda: (lambda m, t: "stub-model")
    )
    # Stub decide_fn with a pure-Python mirror of tool_decider.bdsl so tests
    # don't require juliacall.  Logic: ASK when (voi_ask - ask_cost) > best
    # non-ask EU, otherwise argmax of [execute, substitute, stop].
    def _py_decide(inputs: DecideInputs) -> Action:
        eus = inputs.action_eus
        best_non_ask_eu = max(eus[0], eus[1], eus[2])
        if (inputs.voi_ask - inputs.ask_cost) > best_non_ask_eu:
            return Action.ASK
        non_ask = [eus[0], eus[1], eus[2]]
        best_idx = non_ask.index(max(non_ask))
        return Action(best_idx)

    monkeypatch.setattr(server, "_default_decide_fn", lambda: _py_decide)
    with TestClient(server.app) as c:
        yield c


def _bash_tool():
    return {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }


def test_first_call_with_diffuse_prior_triggers_ask(client):
    """Cold start: voi_ask - ask_cost beats eu_stop → ASK path."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "delete /tmp/foo"}],
            "tools": [_bash_tool()],
        },
    )
    assert response.status_code == 200
    msg = response.json()["choices"][0]["message"]
    # With a cold-start prior, voi_ask is at the maximum and should beat
    # the proposed action's EU. Assert ask-path: text contains the prompt
    # hint and no tool_calls are emitted.
    assert "approve?" in msg["content"].lower()
    assert msg.get("tool_calls") in (None, [])


def test_after_yes_reply_posterior_concentrates(client, tmp_path: Path):
    # Turn 1 — cold start ask.
    client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [_bash_tool()],
        },
    )
    # Turn 2 — user replies "yes" to the ask. The pipeline should observe
    # this and then proceed with EXECUTE.
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "Before I call `Bash`, approve? (y/n, or correct me)",
                },
                {"role": "user", "content": "yes"},
            ],
            "tools": [_bash_tool()],
        },
    )
    msg = response.json()["choices"][0]["message"]
    # On a second turn after a confirmed yes, the tool call should be emitted.
    assert msg.get("tool_calls"), f"expected tool_calls, got message: {msg}"
    assert msg["tool_calls"][0]["function"]["name"] == "Bash"


def test_interruption_in_history_updates_posterior_negatively(client):
    # An assistant tool_call that the user interrupted (no tool result).
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "run something"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "stop, never mind"},
            ],
            "tools": [_bash_tool()],
        },
    )
    assert response.status_code == 200
    # State should now have one β-bump on (stub-model, Bash) due to interruption.
    from credence_router import server

    assert server._tool_decision_state is not None
    a, b = server._tool_decision_state.get_beta("stub-model", "Bash")
    assert b >= 2.0
