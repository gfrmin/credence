# Role: body
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
from starlette.background import BackgroundTask

log = logging.getLogger(__name__)

app = FastAPI(title="credence-proxy", version="0.2.0")

# Global state (initialised on startup)
_search_router = None
_llm_domain = None
_brain = None  # skin.client.SkinClient, spawned in _init_llm_domain
_state_path: Path | None = None
_llm_state_path: Path | None = None
_request_log: list[dict] = []

_tool_decision_state = None  # ToolDecisionState | None
_tool_decision_state_path = None  # Path | None
_tool_decision_enabled = False


class RouteRequest(BaseModel):
    query: str
    category_hint: str | None = None


class OutcomeRequest(BaseModel):
    useful: bool
    domain: str = "search"


# ---------------------------------------------------------------------------
# Tool-decision helper factories (stubs; completed in Task 11)
# ---------------------------------------------------------------------------


def _default_embed_fn():
    """Embedding via OpenAI-compatible /v1/embeddings on credence-router itself.

    For v0 we route through whichever provider has an embeddings endpoint;
    fall back to a deterministic local pseudo-embedding if no API key is set
    so tests / smoke runs don't require external creds.
    """
    import numpy as np

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CREDENCE_EMBEDDING_KEY")
    if api_key:
        import httpx as _httpx

        url = os.environ.get(
            "CREDENCE_EMBEDDING_URL",
            "https://api.openai.com/v1/embeddings",
        )
        model = os.environ.get("CREDENCE_EMBEDDING_MODEL", "text-embedding-3-small")
        client = _httpx.Client(timeout=10.0)

        def fn(text: str):
            r = client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": text},
            )
            r.raise_for_status()
            data = r.json()["data"][0]["embedding"]
            return np.asarray(data, dtype=np.float32)

        return fn

    log.warning("tool-decision: OPENAI_API_KEY not set, using pseudo-embeddings")

    def fn(text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(64).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    return fn


def _default_llm_fn():
    """Forward the chat-completions request to the upstream LLM.

    Wraps forward_streaming (async generator → raw SSE bytes) into the
    synchronous LlmFn contract: (messages, tools, model_id) → dict.

    forward_streaming yields (chunk_bytes, Observation | None).  We drain
    it in a fresh event loop running in a worker thread so we don't conflict
    with the FastAPI event loop that is already running on the calling thread.
    """
    import concurrent.futures
    import json as _json

    from credence_router.tools.llm.provider import forward_streaming

    def fn(messages: list[dict], tools: list[dict], model_id: str) -> dict:
        # Build a minimal OpenAI-format request body.
        body_dict: dict = {"model": model_id, "messages": messages, "stream": True}
        if tools:
            body_dict["tools"] = tools
        request_body = _json.dumps(body_dict).encode()

        async def _collect() -> dict:
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            usage_cost = 0.0

            async for chunk, obs in forward_streaming(request_body, model_id):
                if not chunk:
                    # Terminal tuple: (b"", Observation) — extract cost.
                    if obs is not None:
                        usage_cost = obs.cost_usd
                    continue
                # Parse SSE lines from the chunk.
                for line in chunk.decode("utf-8", errors="replace").split("\n"):
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    try:
                        event = _json.loads(line[6:])
                    except _json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if isinstance(content, str):
                            text_parts.append(content)
                        for tc in delta.get("tool_calls") or []:
                            _merge_tool_call(tool_calls, tc)

            return {
                "model_id": model_id,
                "text": "".join(text_parts),
                "tool_calls": tool_calls,
                "usage_cost": usage_cost,
            }

        # Run in a worker thread with its own event loop to avoid
        # "asyncio.run() cannot be called from a running event loop".
        def _run_in_thread():
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_collect())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_thread)
            return future.result()

    return fn


def _merge_tool_call(acc: list[dict], delta: dict) -> None:
    """Accumulate a streaming tool_call delta into acc."""
    idx = delta.get("index", 0)
    while len(acc) <= idx:
        acc.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    cur = acc[idx]
    if delta.get("id"):
        cur["id"] = delta["id"]
    fn_delta = delta.get("function", {})
    if fn_delta.get("name"):
        cur["function"]["name"] += fn_delta["name"]
    if fn_delta.get("arguments"):
        cur["function"]["arguments"] += fn_delta["arguments"]


def _default_select_model_fn():
    """Reuse credence-router's existing model-selection via _llm_domain.route().

    If _llm_domain is initialised, extract the last user message and route
    through the Bayesian EU-maximiser.  Falls back to CREDENCE_DEFAULT_MODEL
    (or claude-haiku-4-5) when the domain is unavailable (e.g. no API keys).
    """

    def fn(messages: list[dict], tools: list[dict]) -> str:
        if _llm_domain is None:
            return os.environ.get("CREDENCE_DEFAULT_MODEL", "claude-haiku-4-5")
        # Extract last user text for the routing domain (route() takes a string).
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    user_text = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                break
        decision = _llm_domain.route(user_text)
        return decision.provider_name

    return fn


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

    global _tool_decision_state, _tool_decision_state_path, _tool_decision_enabled
    _tool_decision_enabled = os.environ.get("CREDENCE_TOOL_DECISION", "0") == "1"
    if _tool_decision_enabled:
        from credence_router.tool_decision.state import ToolDecisionState

        _tool_decision_state_path = Path(
            os.environ.get("CREDENCE_TOOL_DECISION_STATE",
                           "credence-tool-decision-state.json")
        )
        _tool_decision_state = ToolDecisionState(path=_tool_decision_state_path)
        try:
            _tool_decision_state.load()
            log.info("tool-decision: loaded state from %s", _tool_decision_state_path)
        except Exception as e:  # noqa: BLE001
            log.warning("tool-decision: could not load state: %s", e)

    log.info(
        "Credence proxy started: search=%s, llm=%s",
        [t.name for t in search_tools],
        _llm_domain.provider_names if _llm_domain else "disabled (no API keys)",
    )


def _init_llm_domain():
    """Initialise the LLM routing domain with all available providers."""
    global _llm_domain, _brain

    from credence_router.tools.llm.provider import available_models, model_cost

    models = available_models()
    if not models:
        log.info("No LLM API keys found — LLM routing disabled")
        return

    # Import SkinClient via the routing_domain module which has already
    # located skin/ on sys.path. Resolve the Julia binary from juliapkg
    # so the skin subprocess can spawn without PATH mutation.
    from credence_router.routing_domain import SkinClient, _REPO_ROOT
    import juliapkg

    from credence_router.categories import LLM_CATEGORIES, make_llm_category_infer_fn
    from credence_router.routing_domain import RoutingDomain

    # Allow override: the container uses pyjuliapkg's julia (the only one
    # present); local dev machines may have a different julia binary with
    # a matching depot (CREDENCE_JULIA lets us choose).
    julia_exe = os.environ.get("CREDENCE_JULIA") or str(juliapkg.executable())
    server_path = _REPO_ROOT / "apps" / "skin" / "server.jl"
    _brain = SkinClient(
        julia=julia_exe,
        server_path=server_path,
        project=str(_REPO_ROOT),
    )
    _brain.initialize(dsl_files={"router": str(_REPO_ROOT / "examples" / "router.bdsl")})

    model_names = [m.name for m in models]
    costs = [model_cost(m) for m in models]

    reward = float(os.environ.get("CREDENCE_REWARD", "1.0"))

    _llm_domain = RoutingDomain(
        skin=_brain,
        provider_names=model_names,
        costs=costs,
        categories=LLM_CATEGORIES,
        category_infer=make_llm_category_infer_fn(),
        reward=reward,
    )
    log.info("LLM domain initialised: models=%s", model_names)


@app.on_event("shutdown")
def _shutdown_skin():
    global _brain
    if _brain is not None:
        try:
            _brain.shutdown()
        except Exception as e:
            log.warning("skin shutdown failed: %s", e)
        _brain = None

    if _tool_decision_state is not None and _tool_decision_state_path is not None:
        try:
            _tool_decision_state.save()
            log.info("tool-decision: saved state to %s", _tool_decision_state_path)
        except Exception as e:  # noqa: BLE001
            log.warning("tool-decision: could not save state: %s", e)


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
    if _tool_decision_enabled and _tool_decision_state is not None:
        import json as _json

        raw = await request.body()
        try:
            body_json = _json.loads(raw)
        except Exception:
            body_json = {}
        if body_json.get("tools"):
            from credence_router.tool_decision.pipeline import (
                PipelineConfig,
                run_pipeline,
            )
            from credence_router.tool_decision.decide import decide as julia_decide

            response = run_pipeline(
                messages=body_json.get("messages", []),
                tools=body_json.get("tools", []),
                state=_tool_decision_state,
                config=PipelineConfig(
                    embed_fn=_default_embed_fn(),
                    llm_fn=_default_llm_fn(),
                    select_model_fn=_default_select_model_fn(),
                    decide_fn=julia_decide,
                    ask_cost=float(os.environ.get("CREDENCE_ASK_COST", "0.05")),
                    knn_k=int(os.environ.get("CREDENCE_KNN_K", "3")),
                ),
            )
            return response

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

    obs_holder: dict = {}

    async def generate():
        async for chunk, obs in forward_streaming(body, model, obs_out=obs_holder):
            if obs is None:
                yield chunk

    async def after_stream():
        observation = obs_holder.get("observation")
        if observation is None:
            return
        if not forced and observation.response_text and os.environ.get("ANTHROPIC_API_KEY"):
            asyncio.create_task(
                _judge_and_update(user_message, observation.response_text, model, observation)
            )
        elif not forced:
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
        if _llm_state_path and not forced:
            try:
                _llm_domain.save_state(_llm_state_path)
            except Exception as e:
                log.warning("Could not save LLM state: %s", e)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Credence-Model": model},
        background=BackgroundTask(after_stream),
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
    """Readiness probe — validates the skin is responsive (for Docker/k8s)."""
    if _brain is not None:
        try:
            _brain.n_factors(_llm_domain._state_id)
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}, 503
    return {"status": "ready"}


def serve(host: str = "0.0.0.0", port: int = 8377):
    """Run the server."""
    import uvicorn

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(app, host=host, port=port)
