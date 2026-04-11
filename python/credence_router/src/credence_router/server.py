"""HTTP server for credence proxy — Bayesian AI Gateway.

Unified proxy routing both search and LLM calls via EU maximisation.
One process, one port, one learned state.

LLM: single /v1/chat/completions endpoint (OpenAI format). Routes across
ALL providers (Anthropic, OpenAI, Google) via their OpenAI-compatible
endpoints. The client sends one request; the proxy picks the best model.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(title="credence-proxy", version="0.2.0")

# Global state (initialised on startup)
_search_router = None
_llm_domain = None
_state_path: Path | None = None
_llm_state_path: Path | None = None
_request_log: list[dict] = []


class RouteRequest(BaseModel):
    query: str
    category_hint: str | None = None


class OutcomeRequest(BaseModel):
    useful: bool
    domain: str = "search"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup():
    global _search_router, _llm_domain, _state_path, _llm_state_path

    # --- Search domain ---
    from credence_router.search_router import SearchRouter
    from credence_router.tool import SearchTool
    from credence_router.tools.web.duckduckgo import DuckDuckGoSearchTool

    search_tools: list[SearchTool] = [DuckDuckGoSearchTool()]

    if os.environ.get("BRAVE_API_KEY"):
        from credence_router.tools.web.brave import BraveSearchTool
        search_tools.append(BraveSearchTool())

    if os.environ.get("PERPLEXITY_API_KEY"):
        from credence_router.tools.web.perplexity import PerplexitySearchTool
        search_tools.append(PerplexitySearchTool())

    if os.environ.get("TAVILY_API_KEY"):
        from credence_router.tools.web.tavily import TavilySearchTool
        search_tools.append(TavilySearchTool())

    reward = float(os.environ.get("CREDENCE_REWARD", "0.25"))
    latency_weight = float(os.environ.get("CREDENCE_LATENCY_WEIGHT", "0.01"))

    _search_router = SearchRouter(search_tools, reward_useful=reward, latency_weight=latency_weight)

    # --- LLM domain ---
    _init_llm_domain()

    # --- State persistence ---
    _state_path = Path(os.environ.get("CREDENCE_STATE_PATH", "credence-state.json"))
    if _state_path.exists():
        try:
            _search_router.load_state(_state_path)
        except Exception as e:
            log.warning("Could not load search state: %s", e)

    _llm_state_path = Path(os.environ.get("CREDENCE_LLM_STATE_PATH", "credence-llm-state.bin"))
    if _llm_domain is not None and _llm_state_path.exists():
        try:
            _llm_domain.load_state(_llm_state_path)
        except Exception as e:
            log.warning("Could not load LLM state: %s", e)

    log.info(
        "Credence proxy started: search=%s, llm=%s",
        [t.name for t in search_tools],
        _llm_domain.provider_names if _llm_domain else "disabled (no API keys)",
    )


def _init_llm_domain():
    """Initialise the LLM routing domain with all available providers."""
    global _llm_domain

    from credence_router.tools.llm.provider import available_models, model_cost

    models = available_models()
    if not models:
        log.info("No LLM API keys found — LLM routing disabled")
        return

    from credence_agents.julia_bridge import CredenceBridge

    from credence_router.categories import LLM_CATEGORIES, make_llm_category_infer_fn
    from credence_router.routing_domain import RoutingDomain

    bridge = CredenceBridge()

    model_names = [m.name for m in models]
    costs = [model_cost(m) for m in models]

    reward = float(os.environ.get("CREDENCE_REWARD", "1.0"))

    _llm_domain = RoutingDomain(
        bridge=bridge,
        provider_names=model_names,
        costs=costs,
        categories=LLM_CATEGORIES,
        category_infer=make_llm_category_infer_fn(),
        reward=reward,
    )
    log.info("LLM domain initialised: models=%s", model_names)


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


@app.post("/route")
@app.post("/search")
def route_search(req: RouteRequest):
    result = _search_router.route(req.query, category_hint=req.category_hint)

    resp = {
        "provider": result.provider,
        "confidence": result.confidence,
        "wall_time": result.wall_time,
        "reasoning": result.reasoning,
    }
    if result.result is not None:
        resp["results"] = {
            "text": result.result.text[:2000],
            "urls": result.result.urls[:10],
        }
    return resp


# ---------------------------------------------------------------------------
# LLM quality judge (async, runs after response delivery)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """Rate this LLM response quality 0-10. Consider correctness, completeness, clarity, helpfulness.
Respond with ONLY a number (0-10), nothing else."""


async def _judge_and_update(
    user_message: str, response_text: str, model: str, observation,
):
    """Async quality judge. Runs after the client has received the response."""
    from credence_router.routing_domain import Observation

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 10,
                    "system": _JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": (
                        f"Query: {user_message[:500]}\n\nResponse ({model}):\n{response_text[:1500]}"
                    )}],
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            score_text = resp.json()["content"][0]["text"].strip()
            score = float(score_text.split()[0])
            score = max(0.0, min(10.0, score))
            normalised = score / 10.0

            log.info("Quality judge: %s → %.1f/10 for '%s...'", model, score, user_message[:30])

            # Queue continuous quality signal — processed on next route() call
            obs_with_quality = Observation(
                completed=observation.completed,
                error_type=observation.error_type,
                ttft_seconds=observation.ttft_seconds,
                total_seconds=observation.total_seconds,
                input_tokens=observation.input_tokens,
                output_tokens=observation.output_tokens,
                truncated=observation.truncated,
                cost_usd=observation.cost_usd,
                quality_score=normalised,
                response_text=response_text,
            )
            _llm_domain.queue_outcome(obs_with_quality)

    except Exception as e:
        log.error("Quality judge failed: %s", e)
        # Fall back to binary signal
        _llm_domain.queue_outcome(observation)


# ---------------------------------------------------------------------------
# LLM proxy — ONE endpoint, ALL providers
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Proxy OpenAI Chat Completions format with Bayesian model routing.

    Accepts OpenAI format. Routes across ALL providers (Anthropic, OpenAI, etc.)
    via their OpenAI-compatible endpoints. Picks the best model for the query,
    streams the response back, and updates beliefs from the outcome.
    """
    if _llm_domain is None:
        return {"error": "LLM domain not initialised. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."}, 503

    from credence_router.tools.llm.provider import extract_user_message, forward_streaming

    body = await request.body()
    user_message = extract_user_message(body)

    # Route: classify + EU-maximise, or use forced model
    forced = os.environ.get("CREDENCE_FORCE_MODEL")
    if forced:
        model = forced
        log.info("LLM forced: '%s...' → %s", user_message[:50], model)
    else:
        decision = _llm_domain.route(user_message)
        model = decision.provider_name
        log.info("LLM routing: '%s...' → %s", user_message[:50], model)

    async def generate():
        observation = None
        async for chunk, obs in forward_streaming(body, model):
            if obs is not None:
                observation = obs
            else:
                yield chunk
        if observation is not None:
            # Schedule async quality judge (runs after response is delivered)
            if not forced and observation.response_text and os.environ.get("ANTHROPIC_API_KEY"):
                asyncio.create_task(
                    _judge_and_update(user_message, observation.response_text, model, observation)
                )
            elif not forced:
                # No judge available — fall back to binary signal
                _llm_domain.queue_outcome(observation)
            _request_log.append({
                "user_message": user_message[:100],
                "model_selected": model,
                "input_tokens": observation.input_tokens,
                "output_tokens": observation.output_tokens,
                "cost_usd": observation.cost_usd,
                "ttft_seconds": observation.ttft_seconds,
                "total_seconds": observation.total_seconds,
                "useful": observation.useful,
                "forced": bool(forced),
            })
            # Persist LLM state after each request
            if _llm_state_path and not forced:
                try:
                    _llm_domain.save_state(_llm_state_path)
                except Exception as e:
                    log.warning("Could not save LLM state: %s", e)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Credence-Model": model,
        },
    )


# ---------------------------------------------------------------------------
# Outcome + state endpoints
# ---------------------------------------------------------------------------


@app.post("/outcome")
@app.post("/outcome/{domain}")
def report_outcome(req: OutcomeRequest, domain: str = "search"):
    if domain == "search":
        _search_router.report_outcome(req.useful)
    elif domain == "llm" and _llm_domain is not None:
        from credence_router.routing_domain import Observation
        _llm_domain.queue_outcome(
            Observation(completed=True, error_type=None if req.useful else "user_reported")
        )
    if _state_path:
        _search_router.save_state(_state_path)
    if _llm_state_path and _llm_domain is not None:
        try:
            _llm_domain.save_state(_llm_state_path)
        except Exception as e:
            log.warning("Could not save LLM state: %s", e)
    return {"status": "updated", "domain": domain}


@app.get("/state")
def get_state():
    state = {"search": _search_router.learned_reliability}
    if _llm_domain is not None:
        try:
            state["llm"] = _llm_domain.learned_reliability
        except Exception as e:
            state["llm"] = {"error": str(e)}
    return state


@app.get("/metrics")
def get_metrics():
    """Per-request metrics for eval."""
    return {"requests": _request_log}


@app.post("/metrics/clear")
def clear_metrics():
    """Clear request log for fresh eval run."""
    _request_log.clear()
    return {"status": "cleared"}


@app.get("/health")
def health():
    providers = {"search": list(_search_router.learned_reliability.keys())}
    if _llm_domain is not None:
        providers["llm"] = _llm_domain.provider_names
    return {"status": "ok", "providers": providers}


@app.get("/ready")
def readiness():
    """Readiness probe — validates Julia is responsive (for Docker/k8s)."""
    if _llm_domain is not None:
        try:
            _llm_domain._bridge.jl.seval("1+1")
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}, 503
    return {"status": "ready"}


def serve(host: str = "0.0.0.0", port: int = 8377):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
