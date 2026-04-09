"""HTTP server for credence search routing.

Minimal FastAPI server exposing the SearchRouter as an HTTP service.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="credence-search", version="0.1.0")

# Global router instance (initialised on startup)
_router = None
_state_path: Path | None = None


class RouteRequest(BaseModel):
    query: str
    category_hint: str | None = None


class OutcomeRequest(BaseModel):
    useful: bool


@app.on_event("startup")
def startup():
    global _router, _state_path

    from credence_router.search_router import SearchRouter
    from credence_router.tool import SearchTool

    from credence_router.tools.web.duckduckgo import DuckDuckGoSearchTool

    tools: list[SearchTool] = [DuckDuckGoSearchTool()]

    if os.environ.get("BRAVE_API_KEY"):
        from credence_router.tools.web.brave import BraveSearchTool
        tools.append(BraveSearchTool())

    if os.environ.get("PERPLEXITY_API_KEY"):
        from credence_router.tools.web.perplexity import PerplexitySearchTool
        tools.append(PerplexitySearchTool())

    if os.environ.get("TAVILY_API_KEY"):
        from credence_router.tools.web.tavily import TavilySearchTool
        tools.append(TavilySearchTool())

    reward = float(os.environ.get("CREDENCE_REWARD", "0.25"))
    latency_weight = float(os.environ.get("CREDENCE_LATENCY_WEIGHT", "0.01"))

    _router = SearchRouter(tools, reward_useful=reward, latency_weight=latency_weight)

    _state_path = Path(os.environ.get("CREDENCE_STATE_PATH", "credence-search-state.json"))
    if _state_path.exists():
        _router.load_state(_state_path)


@app.post("/route")
def route(req: RouteRequest):
    result = _router.route(req.query, category_hint=req.category_hint)

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


@app.post("/outcome")
def outcome(req: OutcomeRequest):
    _router.report_outcome(req.useful)
    if _state_path:
        _router.save_state(_state_path)
    return {"status": "updated"}


@app.get("/state")
def state():
    return {
        "reliability": _router.learned_reliability,
    }


@app.get("/health")
def health():
    return {"status": "ok", "providers": list(_router.learned_reliability.keys())}


def serve(host: str = "0.0.0.0", port: int = 8377):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
