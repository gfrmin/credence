# Role: body
"""Minimal Ollama HTTP client."""

from __future__ import annotations

import os

def _normalize_base_url(raw: str) -> str:
    """Normalize OLLAMA_HOST into a full http:// URL."""
    if not raw.startswith("http"):
        host = raw
        port = "11434"
        if ":" in host:
            host, port = host.rsplit(":", 1)
        return f"http://{host}:{port}"
    return raw.rstrip("/")


_DEFAULT_BASE_URL = _normalize_base_url(
    os.environ.get("OLLAMA_HOST", "http://localhost:11434")
)


def ollama_generate(
    prompt: str,
    model: str = "llama3.1",
    temperature: float = 0.3,
    num_ctx: int = 8192,
    base_url: str = _DEFAULT_BASE_URL,
) -> str:
    """Send a generate request to Ollama and return the response text."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("Install httpx: pip install bayesian-if[ollama]") from e

    resp = httpx.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ollama_available(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Check whether Ollama is reachable."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
