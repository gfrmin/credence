"""Unified LLM provider forwarder via OpenAI-compatible endpoints.

All major providers (Anthropic, OpenAI, Google) expose OpenAI-compatible
chat completion endpoints. This module forwards to any of them — the only
difference is the base URL and auth header.

The proxy accepts OpenAI Chat Completions format, Bayesian routing picks
the best model across all providers, and the request is forwarded unchanged
except for the model field.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import numpy as np
from numpy.typing import NDArray

from credence_router.routing_domain import Observation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model available through the proxy."""

    name: str
    provider: str  # "anthropic", "openai", "google"
    input_price_per_1k: float
    output_price_per_1k: float
    expected_latency: float
    # Coverage prior per LLM category (code, reasoning, creative, factual, chat)
    coverage: NDArray[np.float64]


# Provider endpoint config: base_url, auth_header_name, auth_env_var
PROVIDER_ENDPOINTS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "env_var": "OPENAI_API_KEY",
    },
}

# All available models
ALL_MODELS: dict[str, ModelSpec] = {
    "claude-haiku-4-5": ModelSpec(
        name="claude-haiku-4-5", provider="anthropic",
        input_price_per_1k=0.0008, output_price_per_1k=0.004,
        expected_latency=1.0,
        coverage=np.array([0.5, 0.3, 0.4, 0.8, 0.9]),
    ),
    "claude-sonnet-4-6": ModelSpec(
        name="claude-sonnet-4-6", provider="anthropic",
        input_price_per_1k=0.003, output_price_per_1k=0.015,
        expected_latency=3.0,
        coverage=np.array([0.8, 0.7, 0.7, 0.7, 0.7]),
    ),
    "claude-opus-4-6": ModelSpec(
        name="claude-opus-4-6", provider="anthropic",
        input_price_per_1k=0.015, output_price_per_1k=0.075,
        expected_latency=8.0,
        coverage=np.array([0.9, 0.9, 0.9, 0.6, 0.5]),
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini", provider="openai",
        input_price_per_1k=0.00015, output_price_per_1k=0.0006,
        expected_latency=1.0,
        coverage=np.array([0.5, 0.3, 0.4, 0.7, 0.9]),
    ),
    "gpt-4o": ModelSpec(
        name="gpt-4o", provider="openai",
        input_price_per_1k=0.0025, output_price_per_1k=0.01,
        expected_latency=3.0,
        coverage=np.array([0.8, 0.8, 0.7, 0.7, 0.7]),
    ),
}


def available_models() -> list[ModelSpec]:
    """Return models whose provider API key is available."""
    result = []
    for spec in ALL_MODELS.values():
        endpoint = PROVIDER_ENDPOINTS.get(spec.provider, {})
        env_var = endpoint.get("env_var", "")
        if os.environ.get(env_var):
            result.append(spec)
    return result


def model_cost(spec: ModelSpec) -> float:
    """Expected cost per request (estimated 500 output tokens)."""
    return spec.input_price_per_1k + spec.output_price_per_1k * 0.5


def compute_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """Compute actual cost from observed token counts."""
    return (input_tokens * spec.input_price_per_1k + output_tokens * spec.output_price_per_1k) / 1000.0


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


async def forward_streaming(
    request_body: bytes,
    model_name: str,
) -> AsyncIterator[tuple[bytes, Observation | None]]:
    """Forward an OpenAI-format request to the correct provider and yield SSE chunks.

    The request is forwarded unchanged except for the model field.
    Routes to the provider's OpenAI-compatible endpoint based on model name.
    """
    spec = ALL_MODELS.get(model_name)
    if spec is None:
        log.error("Unknown model: %s", model_name)
        yield b"", Observation(completed=False, error_type="unknown_model")
        return

    endpoint = PROVIDER_ENDPOINTS.get(spec.provider)
    if endpoint is None:
        log.error("Unknown provider: %s", spec.provider)
        yield b"", Observation(completed=False, error_type="unknown_provider")
        return

    api_key = os.environ.get(endpoint["env_var"], "")
    base_url = endpoint["base_url"]
    auth_value = endpoint["auth_prefix"] + api_key

    # Swap model, ensure streaming
    data = json.loads(request_body)
    data["model"] = model_name
    data["stream"] = True
    data["stream_options"] = {"include_usage": True}

    t_start = time.monotonic()
    ttft = 0.0
    input_tokens = 0
    output_tokens = 0
    completed = False
    error_type = None
    truncated = False
    response_parts: list[str] = []  # buffer response text for quality judging

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                base_url,
                headers={
                    endpoint["auth_header"]: auth_value,
                    "Content-Type": "application/json",
                },
                content=json.dumps(data),
                timeout=120.0,
            ) as response:
                if response.status_code in (401, 403):
                    error_type = "auth"
                    log.error("%s auth failed (HTTP %d)", spec.provider, response.status_code)
                    raise httpx.HTTPStatusError(
                        f"Auth failed: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code == 429:
                    error_type = "rate_limit"
                    log.error("%s rate limited", spec.provider)
                elif response.status_code >= 500:
                    error_type = "server_error"
                    log.error("%s server error: %d", spec.provider, response.status_code)

                first_chunk = True
                async for chunk in response.aiter_bytes():
                    if first_chunk:
                        ttft = time.monotonic() - t_start
                        first_chunk = False
                    yield chunk, None

                    # Parse SSE for usage + finish reason
                    for line in chunk.decode("utf-8", errors="replace").split("\n"):
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        try:
                            event = json.loads(line[6:])
                            # OpenAI format: usage in final chunk
                            usage = event.get("usage")
                            if usage:
                                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", input_tokens))
                                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", output_tokens))
                            # Extract content text for quality judging
                            choices = event.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    response_parts.append(content)
                                finish = choices[0].get("finish_reason")
                                if finish in ("stop", "end_turn"):
                                    completed = True
                                elif finish in ("length", "max_tokens"):
                                    completed = True
                                    truncated = True
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

                if error_type is None and not completed:
                    completed = True

        except httpx.TimeoutException:
            error_type = "timeout"
            log.error("Request timed out for %s/%s", spec.provider, model_name)
        except httpx.ConnectError as e:
            error_type = "timeout"
            log.error("Connection error for %s: %s", spec.provider, e)

    total_seconds = time.monotonic() - t_start
    cost = compute_cost(spec, input_tokens, output_tokens)

    response_text = "".join(response_parts)

    observation = Observation(
        completed=completed and error_type is None,
        error_type=error_type,
        ttft_seconds=ttft,
        total_seconds=total_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        truncated=truncated,
        cost_usd=cost,
        response_text=response_text,
    )

    log.info(
        "%s/%s: %s in %.1fs (ttft=%.2fs, %d+%d tok, $%.4f)",
        spec.provider, model_name,
        "ok" if observation.useful else error_type or "failed",
        total_seconds, ttft, input_tokens, output_tokens, cost,
    )

    yield b"", observation
