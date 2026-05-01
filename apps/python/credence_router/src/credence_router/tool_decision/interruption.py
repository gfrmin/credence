# Role: detect tool calls that were interrupted (no result, or aborted result).
"""Scan an OpenAI-format messages[] array for tool calls that the user
interrupted before completion. Returned items become negative observations
on the corresponding (model_id, tool_name) cell.
"""
from __future__ import annotations

from dataclasses import dataclass

_ABORT_MARKERS = ("[aborted", "aborted by user", "[cancelled", "cancelled by user")


@dataclass(frozen=True)
class InterruptedToolCall:
    tool_call_id: str
    tool_name: str


def find_interrupted_tool_calls(messages: list[dict]) -> list[InterruptedToolCall]:
    """Return tool calls in `messages` that lack a corresponding successful
    `tool` result message.

    A tool call is considered interrupted iff:
    - There is no later message with role=='tool' and matching tool_call_id, OR
    - The matching tool result content contains an abort/cancel marker.
    """
    # Index tool results by id.
    results_by_id: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id") or m.get("id")
            content = m.get("content")
            if isinstance(tcid, str):
                results_by_id[tcid] = content if isinstance(content, str) else ""

    interrupted: list[InterruptedToolCall] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tcid = tc.get("id")
            if not isinstance(tcid, str):
                continue
            name = (tc.get("function") or {}).get("name") or tc.get("name")
            if not isinstance(name, str):
                continue
            result = results_by_id.get(tcid)
            if result is None:
                interrupted.append(InterruptedToolCall(tcid, name))
                continue
            low = result.lower()
            if any(marker in low for marker in _ABORT_MARKERS):
                interrupted.append(InterruptedToolCall(tcid, name))
    return interrupted
