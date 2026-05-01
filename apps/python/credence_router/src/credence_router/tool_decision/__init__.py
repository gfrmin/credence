# Role: tool-decision mode for credence-router.
"""Tool-decision mode: intercept tool_calls in OpenAI chat completions
and route them through a credence Bayesian decision layer.

Activated by setting CREDENCE_TOOL_DECISION=1 in the server environment.
See docs/superpowers/plans/2026-05-01-credence-tool-decision-gateway.md.
"""
from __future__ import annotations
