# Role: derive a stable session id from the messages prefix.
from __future__ import annotations

from credence_router.tool_decision.session import derive_session_id


def test_same_history_same_id():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "thanks"},
    ]
    a = derive_session_id(msgs)
    b = derive_session_id(msgs)
    assert a == b


def test_different_history_different_id():
    a = derive_session_id([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ])
    b = derive_session_id([
        {"role": "system", "content": "be clever"},
        {"role": "user", "content": "hi"},
    ])
    assert a != b


def test_appending_new_user_message_extends_same_session():
    base = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    extended = base + [{"role": "user", "content": "another q"}]
    assert derive_session_id(base) == derive_session_id(extended)


def test_changing_history_starts_new_session():
    a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    b = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo!"}]
    assert derive_session_id(a) != derive_session_id(b)
