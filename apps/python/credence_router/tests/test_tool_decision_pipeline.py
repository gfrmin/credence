# Role: pipeline orchestration tests with stubbed LLM and stubbed decide.
from __future__ import annotations

from pathlib import Path

import numpy as np

from credence_router.tool_decision.decide import Action
from credence_router.tool_decision.pipeline import PipelineConfig, run_pipeline
from credence_router.tool_decision.state import ToolDecisionState


def _stub_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(8).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def _llm_returns_tool_call(name: str = "Bash"):
    def fake_llm(messages, tools, model_id):
        return {
            "model_id": model_id,
            "text": "I'll run a shell command.",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": name, "arguments": "{}"}}
            ],
            "usage_cost": 0.001,
        }
    return fake_llm


def _llm_returns_no_tool():
    def fake_llm(messages, tools, model_id):
        return {
            "model_id": model_id,
            "text": "Done.",
            "tool_calls": [],
            "usage_cost": 0.001,
        }
    return fake_llm


def _decide_returns(action: Action):
    def fake_decide(inputs):
        return action
    return fake_decide


def _stub_select_model(messages, tools):
    return "stub-model"


def _bash_tool_spec():
    return {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }


def test_execute_path_returns_llm_tool_calls(tmp_path: Path):
    state = ToolDecisionState(path=tmp_path / "s.json")
    cfg = PipelineConfig(
        embed_fn=_stub_embed,
        llm_fn=_llm_returns_tool_call("Bash"),
        select_model_fn=_stub_select_model,
        decide_fn=_decide_returns(Action.EXECUTE),
        ask_cost=0.05,
        knn_k=3,
    )
    response = run_pipeline(
        messages=[{"role": "user", "content": "list files"}],
        tools=[_bash_tool_spec()],
        state=state,
        config=cfg,
    )
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "Bash"


def test_ask_path_returns_text_no_tool_calls(tmp_path: Path):
    state = ToolDecisionState(path=tmp_path / "s.json")
    cfg = PipelineConfig(
        embed_fn=_stub_embed,
        llm_fn=_llm_returns_tool_call("Bash"),
        select_model_fn=_stub_select_model,
        decide_fn=_decide_returns(Action.ASK),
        ask_cost=0.05,
        knn_k=3,
    )
    response = run_pipeline(
        messages=[{"role": "user", "content": "delete /tmp/foo"}],
        tools=[_bash_tool_spec()],
        state=state,
        config=cfg,
    )
    msg = response["choices"][0]["message"]
    assert msg.get("tool_calls") in (None, [])
    assert "approve" in msg["content"].lower() or "?" in msg["content"]


def test_stop_path_returns_assistant_no_tool_calls(tmp_path: Path):
    state = ToolDecisionState(path=tmp_path / "s.json")
    cfg = PipelineConfig(
        embed_fn=_stub_embed,
        llm_fn=_llm_returns_no_tool(),
        select_model_fn=_stub_select_model,
        decide_fn=_decide_returns(Action.STOP),
        ask_cost=0.05,
        knn_k=3,
    )
    response = run_pipeline(
        messages=[{"role": "user", "content": "done?"}],
        tools=[_bash_tool_spec()],
        state=state,
        config=cfg,
    )
    msg = response["choices"][0]["message"]
    assert msg.get("tool_calls") in (None, [])


def test_approval_reply_updates_posterior_yes(tmp_path: Path):
    state = ToolDecisionState(path=tmp_path / "s.json")
    cfg = PipelineConfig(
        embed_fn=_stub_embed,
        llm_fn=_llm_returns_tool_call("Bash"),
        select_model_fn=_stub_select_model,
        decide_fn=_decide_returns(Action.EXECUTE),
        ask_cost=0.05,
        knn_k=3,
    )

    # Turn 1: gateway emits an ask.
    cfg_ask = PipelineConfig(
        embed_fn=cfg.embed_fn,
        llm_fn=cfg.llm_fn,
        select_model_fn=cfg.select_model_fn,
        decide_fn=_decide_returns(Action.ASK),
        ask_cost=cfg.ask_cost,
        knn_k=cfg.knn_k,
    )
    run_pipeline(
        messages=[{"role": "user", "content": "list files"}],
        tools=[_bash_tool_spec()],
        state=state,
        config=cfg_ask,
    )

    # Turn 2: user approved. The pipeline must apply the +1 to alpha.
    msgs2 = [
        {"role": "user", "content": "list files"},
        {"role": "assistant",
         "content": "Before I call `Bash`, approve? (y/n, or correct me)"},
        {"role": "user", "content": "yes"},
    ]
    run_pipeline(messages=msgs2, tools=[_bash_tool_spec()], state=state, config=cfg)
    a, b = state.get_beta("stub-model", "Bash")
    assert a == 2.0  # 1 (prior) + 1 (yes)
    assert b == 1.0


def test_interruption_updates_posterior_no(tmp_path: Path):
    state = ToolDecisionState(path=tmp_path / "s.json")
    cfg = PipelineConfig(
        embed_fn=_stub_embed,
        llm_fn=_llm_returns_tool_call("Bash"),
        select_model_fn=_stub_select_model,
        decide_fn=_decide_returns(Action.EXECUTE),
        ask_cost=0.05,
        knn_k=3,
    )

    # History: assistant called Bash, user interrupted (no tool result).
    msgs = [
        {"role": "user", "content": "delete /tmp/foo"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "Bash", "arguments": "{}"}}]},
        {"role": "user", "content": "stop"},
    ]
    run_pipeline(messages=msgs, tools=[_bash_tool_spec()], state=state, config=cfg)
    a, b = state.get_beta("stub-model", "Bash")
    assert a == 1.0
    assert b == 2.0  # 1 (prior) + 1 (interruption)
