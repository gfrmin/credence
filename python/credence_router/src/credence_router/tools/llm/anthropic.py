"""Anthropic Messages API provider for credence proxy.

Routes between Claude models (haiku, sonnet, opus) within the same API.
Supports streaming (SSE) passthrough.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.routing_domain import Observation

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Model definitions: name → (input_price_per_1k, output_price_per_1k, expected_latency_s)
ANTHROPIC_MODELS = {
    "claude-haiku-4-5-20251001": (0.0008, 0.004, 1.0),
    "claude-sonnet-4-6-20250514": (0.003, 0.015, 3.0),
    "claude-opus-4-6-20250514": (0.015, 0.075, 8.0),
}

# Short aliases
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250514",
    "opus": "claude-opus-4-6-20250514",
}

# Coverage priors per LLM category (code, reasoning, creative, factual, chat)
# Higher = model is expected to be more reliable for this category
_MODEL_COVERAGE = {
    "claude-haiku-4-5-20251001": np.array([0.5, 0.3, 0.4, 0.8, 0.9]),
    "claude-sonnet-4-6-20250514": np.array([0.8, 0.7, 0.7, 0.7, 0.7]),
    "claude-opus-4-6-20250514": np.array([0.9, 0.9, 0.9, 0.6, 0.5]),
}


def resolve_model(model: str) -> str:
    """Resolve a model alias to a full model ID."""
    return MODEL_ALIASES.get(model, model)


def available_models(api_key: str | None = None) -> list[str]:
    """Return list of available Anthropic model IDs."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return []
    return list(ANTHROPIC_MODELS.keys())


def model_cost(model: str) -> float:
    """Expected cost per request (using estimated 500 output tokens)."""
    model = resolve_model(model)
    if model not in ANTHROPIC_MODELS:
        return 0.01
    input_price, output_price, _ = ANTHROPIC_MODELS[model]
    # Estimate: 1000 input tokens + 500 output tokens
    return input_price + output_price * 0.5


def model_latency(model: str) -> float:
    """Expected latency in seconds."""
    model = resolve_model(model)
    if model not in ANTHROPIC_MODELS:
        return 3.0
    return ANTHROPIC_MODELS[model][2]


def model_coverage(model: str) -> NDArray[np.float64]:
    """Coverage prior per LLM category."""
    model = resolve_model(model)
    return _MODEL_COVERAGE.get(model, np.full(5, 0.5))


def extract_user_message(request_body: bytes) -> str:
    """Extract the last user message from an Anthropic Messages request."""
    data = json.loads(request_body)
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # content blocks — extract text blocks
                parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                return " ".join(parts)
    return ""


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute actual cost from observed token counts."""
    model = resolve_model(model)
    if model not in ANTHROPIC_MODELS:
        return 0.0
    input_price, output_price, _ = ANTHROPIC_MODELS[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1000.0


async def forward_streaming(
    request_body: bytes,
    model: str,
    api_key: str | None = None,
) -> AsyncIterator[tuple[bytes, Observation | None]]:
    """Forward a request to Anthropic and yield SSE chunks.

    Yields (chunk_bytes, None) for each SSE chunk during streaming.
    The final yield is (b"", Observation) with the outcome metrics.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    model = resolve_model(model)

    # Swap the model in the request body
    data = json.loads(request_body)
    data["model"] = model
    data["stream"] = True

    t_start = time.monotonic()
    ttft = 0.0
    input_tokens = 0
    output_tokens = 0
    completed = False
    error_type = None
    truncated = False

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                content=json.dumps(data),
                timeout=120.0,
            ) as response:
                if response.status_code in (401, 403):
                    error_type = "auth"
                    log.error("Anthropic auth failed (HTTP %d)", response.status_code)
                    raise httpx.HTTPStatusError(
                        f"Auth failed: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code == 429:
                    error_type = "rate_limit"
                    log.error("Anthropic rate limited")
                elif response.status_code >= 500:
                    error_type = "server_error"
                    log.error("Anthropic server error: %d", response.status_code)

                first_chunk = True
                async for chunk in response.aiter_bytes():
                    if first_chunk:
                        ttft = time.monotonic() - t_start
                        first_chunk = False
                    yield chunk, None

                    # Parse SSE events for usage info
                    for line in chunk.decode("utf-8", errors="replace").split("\n"):
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])
                                event_type = event_data.get("type", "")
                                if event_type == "message_delta":
                                    usage = event_data.get("usage", {})
                                    output_tokens = usage.get("output_tokens", output_tokens)
                                    stop = event_data.get("delta", {}).get("stop_reason")
                                    if stop == "end_turn":
                                        completed = True
                                    elif stop == "max_tokens":
                                        completed = True
                                        truncated = True
                                elif event_type == "message_start":
                                    usage = event_data.get("message", {}).get("usage", {})
                                    input_tokens = usage.get("input_tokens", 0)
                            except (json.JSONDecodeError, KeyError):
                                pass

                if error_type is None and not completed:
                    completed = True  # stream ended normally

        except httpx.TimeoutException:
            error_type = "timeout"
            log.error("Anthropic request timed out for model %s", model)
        except httpx.ConnectError as e:
            error_type = "timeout"
            log.error("Anthropic connection error: %s", e)

    total_seconds = time.monotonic() - t_start
    cost = compute_cost(model, input_tokens, output_tokens)

    observation = Observation(
        completed=completed and error_type is None,
        error_type=error_type,
        ttft_seconds=ttft,
        total_seconds=total_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        truncated=truncated,
        cost_usd=cost,
    )

    log.info(
        "Anthropic %s: %s in %.1fs (ttft=%.2fs, %d+%d tok, $%.4f)",
        model, "ok" if observation.useful else error_type or "failed",
        total_seconds, ttft, input_tokens, output_tokens, cost,
    )

    yield b"", observation
