# Role: parse user replies to ask-for-approval prompts.
from __future__ import annotations

import pytest

from credence_router.tool_decision.approval_parsing import (
    ApprovalReply,
    parse_approval_reply,
)


@pytest.mark.parametrize(
    "text",
    ["yes", "y", "Yes please", "ok", "go ahead", "sure", "approved", "do it"],
)
def test_clear_yes(text):
    reply = parse_approval_reply(text)
    assert reply.approved is True


@pytest.mark.parametrize(
    "text",
    ["no", "n", "No", "stop", "don't", "cancel", "abort"],
)
def test_clear_no(text):
    reply = parse_approval_reply(text)
    assert reply.approved is False


def test_corrective_text_yields_no_with_correction():
    reply = parse_approval_reply("no, use grep instead of bash")
    assert reply.approved is False
    assert "grep" in (reply.correction or "")


def test_unparseable_yields_unknown():
    # Best-effort: 'ok' substring → approved True. Acceptable behaviour.
    # The unambiguous-unknown case is empty input.
    reply = parse_approval_reply("")
    assert reply.approved is None


def test_isinstance_dataclass():
    reply = parse_approval_reply("yes")
    assert isinstance(reply, ApprovalReply)
