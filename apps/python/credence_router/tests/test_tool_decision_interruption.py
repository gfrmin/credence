# Role: detect interrupted tool calls in OpenAI-format message history.
from __future__ import annotations

from credence_router.tool_decision.interruption import find_interrupted_tool_calls


def _assistant_with_tool_call(call_id: str, name: str):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool_result(call_id: str, content: str):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _user(text: str):
    return {"role": "user", "content": text}


def test_no_tool_calls_returns_empty():
    msgs = [_user("hi"), {"role": "assistant", "content": "hello"}]
    assert find_interrupted_tool_calls(msgs) == []


def test_completed_tool_call_returns_empty():
    msgs = [
        _user("read foo"),
        _assistant_with_tool_call("c1", "Read"),
        _tool_result("c1", "(file contents)"),
    ]
    assert find_interrupted_tool_calls(msgs) == []


def test_orphan_tool_call_detected():
    msgs = [
        _user("delete /tmp/foo"),
        _assistant_with_tool_call("c1", "Bash"),
        _user("stop! never mind"),  # user interrupted, no tool result
    ]
    found = find_interrupted_tool_calls(msgs)
    assert len(found) == 1
    assert found[0].tool_name == "Bash"
    assert found[0].tool_call_id == "c1"


def test_explicit_aborted_marker_detected():
    msgs = [
        _user("run something"),
        _assistant_with_tool_call("c1", "Bash"),
        _tool_result("c1", "[aborted by user]"),
    ]
    found = find_interrupted_tool_calls(msgs)
    assert len(found) == 1
    assert found[0].tool_name == "Bash"


def test_multiple_orphans_all_returned():
    msgs = [
        _user("do two things"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "Bash", "arguments": "{}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "Edit", "arguments": "{}"}},
            ],
        },
        _user("nope"),
    ]
    found = find_interrupted_tool_calls(msgs)
    names = sorted(x.tool_name for x in found)
    assert names == ["Bash", "Edit"]
