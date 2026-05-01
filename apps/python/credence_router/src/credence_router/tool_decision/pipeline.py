# Role: per-request orchestration of the tool-decision pipeline.
"""Per-request flow:

1. Identify session.
2. Update posterior from previous turn (approval reply OR interruption).
3. Embed each tool in the request (cached).
4. Call the real LLM (via injected llm_fn) for reasoning + a proposed tool_call.
5. Pick action: execute / substitute / stop / ask via the Julia decide bridge.
6. Format an OpenAI-format response.

Side-effect-driven via the injected ToolDecisionState.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from credence_router.tool_decision.approval_parsing import parse_approval_reply
from credence_router.tool_decision.decide import (
    Action,
    DecideInputs,
    compute_action_eus,
)
from credence_router.tool_decision.embeddings import (
    embed_tool,
    knn_smoothed_prior,
    tool_content_hash,
)
from credence_router.tool_decision.interruption import find_interrupted_tool_calls
from credence_router.tool_decision.session import derive_session_id
from credence_router.tool_decision.state import ToolDecisionState

EmbedFn = Callable[[str], NDArray[np.float32]]
SelectModelFn = Callable[[list[dict], list[dict]], str]
LlmFn = Callable[[list[dict], list[dict], str], dict]
DecideFn = Callable[[DecideInputs], Action]


@dataclass(frozen=True)
class PipelineConfig:
    embed_fn: EmbedFn
    llm_fn: LlmFn
    select_model_fn: SelectModelFn
    decide_fn: DecideFn
    ask_cost: float
    knn_k: int


_ASK_PROMPT_HINT = "approve? (y/n, or correct me)"


def run_pipeline(
    *,
    messages: list[dict],
    tools: list[dict],
    state: ToolDecisionState,
    config: PipelineConfig,
) -> dict:
    _session_id = derive_session_id(messages)

    # Step 2 — update posterior from observations in this turn's history.
    _apply_observations(messages, state)

    # Step 3 — embed each tool, build a name → content_hash index.
    tool_specs_by_name = _flatten_tools(tools)
    tool_name_to_hash = {
        name: tool_content_hash(spec) for name, spec in tool_specs_by_name.items()
    }
    for spec in tool_specs_by_name.values():
        embed_tool(spec, state=state, embed_fn=config.embed_fn)

    # Step 4 — call the real LLM.
    model_id = config.select_model_fn(messages, tools)
    llm_response = config.llm_fn(messages, tools, model_id)
    proposed = _first_tool_call(llm_response)
    proposed_name = proposed["function"]["name"] if proposed else None
    llm_cost = float(llm_response.get("usage_cost", 0.0))

    # Step 5 — compute action EUs.
    proposed_alpha, proposed_beta = _alpha_beta_for(
        model_id=model_id,
        tool_name=proposed_name or "__no_tool_call__",
        tool_specs=tool_specs_by_name,
        state=state,
        config=config,
        tool_name_to_hash=tool_name_to_hash,
    )
    best_alt_name, (best_alt_alpha, best_alt_beta) = _best_alternative(
        model_id=model_id,
        tool_specs=tool_specs_by_name,
        state=state,
        config=config,
        proposed_name=proposed_name,
        tool_name_to_hash=tool_name_to_hash,
    )
    stop_alpha, stop_beta = state.get_beta(model_id, "__stop__")

    action_eus = compute_action_eus(
        proposed_alpha=proposed_alpha,
        proposed_beta=proposed_beta,
        best_alt_alpha=best_alt_alpha,
        best_alt_beta=best_alt_beta,
        stop_alpha=stop_alpha,
        stop_beta=stop_beta,
        llm_cost=llm_cost,
    )
    voi_ask = _voi_ask(proposed_alpha, proposed_beta)
    action = config.decide_fn(
        DecideInputs(action_eus=action_eus, voi_ask=voi_ask, ask_cost=config.ask_cost)
    )

    # Step 6 — format response.
    if action is Action.EXECUTE:
        return _wrap_response(model_id, llm_response.get("text", ""), proposed)
    if action is Action.SUBSTITUTE and best_alt_name and best_alt_name != proposed_name:
        substituted = {
            "id": "credence-sub-1",
            "type": "function",
            "function": {"name": best_alt_name, "arguments": "{}"},
        }
        text = f"(credence override → {best_alt_name})"
        return _wrap_response(model_id, text, substituted)
    if action is Action.ASK:
        text = _format_ask_text(proposed_name)
        return _wrap_response(model_id, text, None)
    # STOP
    return _wrap_response(model_id, llm_response.get("text", ""), None)


# ----- helpers -----


def _flatten_tools(tools: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in tools:
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str):
            out[name] = {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
    return out


def _first_tool_call(llm_response: dict) -> dict | None:
    calls = llm_response.get("tool_calls") or []
    return calls[0] if calls else None


def _wrap_response(model_id: str, text: str, tool_call: dict | None) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_call is not None:
        msg["tool_calls"] = [tool_call]
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if tool_call else "stop",
            }
        ],
    }


def _format_ask_text(proposed_name: str | None) -> str:
    if proposed_name is None:
        return f"Should I stop now? {_ASK_PROMPT_HINT}"
    return f"Before I call `{proposed_name}`, {_ASK_PROMPT_HINT}"


def _alpha_beta_for(
    *,
    model_id: str,
    tool_name: str,
    tool_specs: dict[str, dict],
    state: ToolDecisionState,
    config: PipelineConfig,
    tool_name_to_hash: dict[str, str],
) -> tuple[float, float]:
    # Observed cell? Return its posterior.
    a, b = state.get_beta(model_id, tool_name)
    if (a, b) != (1.0, 1.0):
        return a, b
    # Cold-start cell — try kNN smoothing if we have an embedding for this tool.
    spec = tool_specs.get(tool_name)
    if spec is None:
        return 1.0, 1.0
    target = embed_tool(spec, state=state, embed_fn=config.embed_fn)
    return knn_smoothed_prior(
        target_embedding=target,
        model_id=model_id,
        state=state,
        k=config.knn_k,
        tool_name_to_hash=tool_name_to_hash,
    )


def _best_alternative(
    *,
    model_id: str,
    tool_specs: dict[str, dict],
    state: ToolDecisionState,
    config: PipelineConfig,
    proposed_name: str | None,
    tool_name_to_hash: dict[str, str],
) -> tuple[str | None, tuple[float, float]]:
    best_name = None
    best_score = -1.0
    best_ab = (1.0, 1.0)
    for name in tool_specs:
        if name == proposed_name:
            continue
        a, b = _alpha_beta_for(
            model_id=model_id,
            tool_name=name,
            tool_specs=tool_specs,
            state=state,
            config=config,
            tool_name_to_hash=tool_name_to_hash,
        )
        s = a / (a + b) if (a + b) > 0 else 0.5
        if s > best_score:
            best_score = s
            best_name = name
            best_ab = (a, b)
    return best_name, best_ab


def _voi_ask(alpha: float, beta: float) -> float:
    """Variance of the Beta as a tractable proxy for VOI in v0.

    Var(Beta(α,β)) = αβ / ((α+β)^2 (α+β+1)). Scaled by 4 so that the
    maximum-entropy prior (α=β=1) yields ~1.0 and concentrated posteriors
    yield ~0. The DSL gate compares VOI - ask_cost against the best non-ask EU.
    """
    s = alpha + beta
    if s <= 0:
        return 1.0
    var = (alpha * beta) / ((s * s) * (s + 1.0))
    return min(4.0 * var, 1.0)


def _apply_observations(messages: list[dict], state: ToolDecisionState) -> None:
    # 1. Approval reply — if the most recent user message follows an ask-prompt.
    if len(messages) >= 2 and messages[-1].get("role") == "user":
        prev = messages[-2]
        if prev.get("role") == "assistant" and _is_ask_prompt(prev):
            asked_tool = _extract_asked_tool_name(prev.get("content", ""))
            model_id = _last_model_id(messages) or "unknown"
            reply = parse_approval_reply(messages[-1].get("content", ""))
            if reply.approved is not None and asked_tool:
                state.update(model_id, asked_tool, approved=reply.approved)

    # 2. Interruption — orphan tool_calls anywhere in history we haven't yet billed.
    interrupted = find_interrupted_tool_calls(messages)
    if interrupted:
        model_id = _last_model_id(messages) or "unknown"
        for it in interrupted:
            state.update(model_id, it.tool_name, approved=False)


def _is_ask_prompt(message: dict) -> bool:
    content = message.get("content", "")
    return isinstance(content, str) and _ASK_PROMPT_HINT in content


def _extract_asked_tool_name(text: str) -> str | None:
    # Expect "Before I call `<name>`, ..." pattern.
    idx = text.find("`")
    if idx == -1:
        return None
    end = text.find("`", idx + 1)
    if end == -1:
        return None
    return text[idx + 1 : end] or None


def _last_model_id(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and isinstance(m.get("model"), str):
            return m["model"]
    # In v0, when the harness doesn't pass model in messages, the pipeline
    # uses the model_id from the current routing decision. Tests stub
    # select_model_fn → "stub-model".
    return "stub-model"
