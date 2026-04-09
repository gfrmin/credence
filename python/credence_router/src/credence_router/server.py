"""HTTP server for credence proxy — Bayesian AI Gateway.

Unified proxy routing both search and LLM calls via EU maximisation.
One process, one port, one learned state.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(title="credence-proxy", version="0.2.0")

# Global state (initialised on startup)
_search_router = None
_llm_domain = None
_state_path: Path | None = None


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
    global _search_router, _llm_domain, _state_path

    # --- Search domain (existing) ---
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
    if os.environ.get("ANTHROPIC_API_KEY"):
        _init_llm_domain()

    # --- State persistence ---
    _state_path = Path(os.environ.get("CREDENCE_STATE_PATH", "credence-state.json"))
    if _state_path.exists():
        try:
            _search_router.load_state(_state_path)
        except Exception as e:
            log.warning("Could not load search state: %s", e)

    log.info(
        "Credence proxy started: search=%s, llm=%s",
        [t.name for t in search_tools],
        "enabled" if _llm_domain else "disabled (no ANTHROPIC_API_KEY)",
    )


def _init_llm_domain():
    """Initialise the LLM routing domain."""
    global _llm_domain

    from credence_agents.inference.voi import ScoringRule, ToolConfig
    from credence_agents.julia_bridge import CredenceBridge

    from credence_router.categories import LLM_CATEGORIES, make_llm_category_infer_fn
    from credence_router.routing_domain import RoutingDomain
    from credence_router.tools.llm.anthropic import (
        ANTHROPIC_MODELS,
        model_cost,
        model_coverage,
    )

    bridge = CredenceBridge()

    model_names = list(ANTHROPIC_MODELS.keys())
    model_configs = [
        ToolConfig(cost=model_cost(m), coverage_by_category=model_coverage(m))
        for m in model_names
    ]

    scoring = ScoringRule(
        reward_correct=1.0,
        penalty_wrong=-0.5,
        reward_abstain=0.0,
    )

    _llm_domain = RoutingDomain(
        bridge=bridge,
        providers=model_configs,
        provider_names=model_names,
        categories=LLM_CATEGORIES,
        category_infer=make_llm_category_infer_fn(),
        scoring=scoring,
    )
    log.info("LLM domain initialised: models=%s", model_names)


# ---------------------------------------------------------------------------
# Search endpoints (existing)
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
# LLM proxy endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def proxy_anthropic_messages(request: Request):
    """Proxy Anthropic Messages API with Bayesian model routing.

    Receives an Anthropic-format request, classifies the query,
    EU-maximises over available models, swaps the model field,
    forwards to Anthropic, and streams the response back.
    """
    if _llm_domain is None:
        return {"error": "LLM domain not initialised (set ANTHROPIC_API_KEY)"}, 503

    from credence_router.tools.llm.anthropic import (
        extract_user_message,
        forward_streaming,
    )

    body = await request.body()
    user_message = extract_user_message(body)

    # Route: classify + EU-maximise
    decision = _llm_domain.route(user_message)
    log.info("LLM routing: '%s...' → %s", user_message[:50], decision.provider_name)

    # Stream response from selected model
    async def generate():
        observation = None
        async for chunk, obs in forward_streaming(body, decision.provider_name):
            if obs is not None:
                observation = obs
            else:
                yield chunk
        # Update beliefs after stream completes
        if observation is not None:
            _llm_domain.report_outcome(observation)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Credence-Provider": decision.provider_name,
            "X-Credence-Reasoning": decision.reasoning.replace("\n", " | "),
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
        _llm_domain.report_outcome(Observation(completed=True, error_type=None if req.useful else "user_reported"))
    if _state_path:
        _search_router.save_state(_state_path)
    return {"status": "updated", "domain": domain}


@app.get("/state")
def get_state():
    state = {
        "search": _search_router.learned_reliability,
    }
    if _llm_domain is not None:
        state["llm"] = _llm_domain.learned_reliability
    return state


@app.get("/health")
def health():
    providers = {
        "search": list(_search_router.learned_reliability.keys()),
    }
    if _llm_domain is not None:
        providers["llm"] = _llm_domain.provider_names
    return {"status": "ok", "providers": providers}


def serve(host: str = "0.0.0.0", port: int = 8377):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
