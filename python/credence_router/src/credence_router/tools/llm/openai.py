"""OpenAI Chat Completions API provider for credence proxy.

Routes between OpenAI models (gpt-4o-mini, gpt-4o) within the same API.
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

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Model definitions: name → (input_price_per_1k, output_price_per_1k, expected_latency_s)
OPENAI_MODELS = {
    "gpt-4o-mini": (0.00015, 0.0006, 1.0),
    "gpt-4o": (0.0025, 0.01, 3.0),
}

# Coverage priors per LLM category (code, reasoning, creative, factual, chat)
_MODEL_COVERAGE = {
    "gpt-4o-mini": np.array([0.5, 0.3, 0.4, 0.7, 0.9]),
    "gpt-4o": np.array([0.8, 0.8, 0.7, 0.7, 0.7]),
}


def available_models(api_key: str | None = None) -> list[str]:
    """Return list of available OpenAI model IDs."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return []
    return list(OPENAI_MODELS.keys())


def model_cost(model: str) -> float:
    """Expected cost per request (using estimated 500 output tokens)."""
    if model not in OPENAI_MODELS:
        return 0.01
    input_price, output_price, _ = OPENAI_MODELS[model]
    return input_price + output_price * 0.5


def model_coverage(model: str) -> NDArray[np.float64]:
    """Coverage prior per LLM category."""
    return _MODEL_COVERAGE.get(model, np.full(5, 0.5))


def extract_user_message(request_body: bytes) -> str:
    """Extract the last user message from an OpenAI Chat Completions request."""
    data = json.loads(request_body)
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                return " ".join(parts)
    return ""


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute actual cost from observed token counts."""
    if model not in OPENAI_MODELS:
        return 0.0
    input_price, output_price, _ = OPENAI_MODELS[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1000.0


async def forward_streaming(
    request_body: bytes,
    model: str,
    api_key: str | None = None,
) -> AsyncIterator[tuple[bytes, Observation | None]]:
    """Forward a request to OpenAI and yield SSE chunks.

    Yields (chunk_bytes, None) for each SSE chunk during streaming.
    The final yield is (b"", Observation) with the outcome metrics.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")

    # Swap the model in the request body
    data = json.loads(request_body)
    data["model"] = model
    data["stream"] = True
    data["stream_options"] = {"include_usage": True}

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
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(data),
                timeout=120.0,
            ) as response:
                if response.status_code in (401, 403):
                    error_type = "auth"
                    log.error("OpenAI auth failed (HTTP %d)", response.status_code)
                    raise httpx.HTTPStatusError(
                        f"Auth failed: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code == 429:
                    error_type = "rate_limit"
                    log.error("OpenAI rate limited")
                elif response.status_code >= 500:
                    error_type = "server_error"
                    log.error("OpenAI server error: %d", response.status_code)

                first_chunk = True
                async for chunk in response.aiter_bytes():
                    if first_chunk:
                        ttft = time.monotonic() - t_start
                        first_chunk = False
                    yield chunk, None

                    # Parse SSE events for usage info
                    for line in chunk.decode("utf-8", errors="replace").split("\n"):
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                event_data = json.loads(line[6:])
                                # Usage in final chunk (stream_options.include_usage)
                                usage = event_data.get("usage")
                                if usage:
                                    input_tokens = usage.get("prompt_tokens", input_tokens)
                                    output_tokens = usage.get("completion_tokens", output_tokens)
                                # Check finish reason
                                choices = event_data.get("choices", [])
                                if choices:
                                    finish = choices[0].get("finish_reason")
                                    if finish == "stop":
                                        completed = True
                                    elif finish == "length":
                                        completed = True
                                        truncated = True
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass

                if error_type is None and not completed:
                    completed = True

        except httpx.TimeoutException:
            error_type = "timeout"
            log.error("OpenAI request timed out for model %s", model)
        except httpx.ConnectError as e:
            error_type = "timeout"
            log.error("OpenAI connection error: %s", e)

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
        "OpenAI %s: %s in %.1fs (ttft=%.2fs, %d+%d tok, $%.4f)",
        model, "ok" if observation.useful else error_type or "failed",
        total_seconds, ttft, input_tokens, output_tokens, cost,
    )

    yield b"", observation
