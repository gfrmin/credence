# Role: parse approval replies from raw user text.
"""Parse user responses to ask-for-approval prompts into yes/no/unknown.

v0: keyword matching with tightly scoped vocab. Phase 2: LLM-assisted parse
when keywords are ambiguous; richer correction extraction.
"""
from __future__ import annotations

from dataclasses import dataclass

_YES_TOKENS = frozenset(
    {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "approved", "go", "ahead", "do"}
)
_NO_TOKENS = frozenset(
    {"no", "n", "nope", "stop", "don't", "dont", "cancel", "abort", "halt", "skip"}
)


@dataclass(frozen=True)
class ApprovalReply:
    approved: bool | None
    correction: str | None = None


def parse_approval_reply(text: str) -> ApprovalReply:
    if not text or not text.strip():
        return ApprovalReply(approved=None)

    lowered = text.strip().lower()
    tokens = set(_tokenize(lowered))

    has_no = bool(tokens & _NO_TOKENS)
    has_yes = bool(tokens & _YES_TOKENS)

    if has_no and not has_yes:
        correction = _extract_correction(lowered)
        return ApprovalReply(approved=False, correction=correction)
    if has_yes and not has_no:
        return ApprovalReply(approved=True)
    if has_no and has_yes:
        # Mixed signals: lean to refusal (safer).
        correction = _extract_correction(lowered)
        return ApprovalReply(approved=False, correction=correction)
    return ApprovalReply(approved=None)


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    word = []
    for ch in text:
        if ch.isalpha() or ch == "'":
            word.append(ch)
        else:
            if word:
                out.append("".join(word))
                word = []
    if word:
        out.append("".join(word))
    return out


def _extract_correction(text: str) -> str | None:
    # Anything after the first comma or "instead" / "use" / "but" is a likely correction.
    for marker in (", ", " instead", " use ", " but ", "; "):
        idx = text.find(marker)
        if idx != -1 and idx + len(marker) < len(text):
            tail = text[idx + len(marker):].strip()
            if tail:
                return tail
    return None
