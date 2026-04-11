# credence-proxy

Bayesian AI gateway that routes LLM and search requests to the best provider for each query. Drop-in replacement for direct API calls — same OpenAI format in, better results out.

Instead of always calling the same model, credence-proxy learns which model works best for each type of query and routes automatically. It uses Bayesian expected utility maximisation, not heuristics or LLM-based routing.

**Eval results (OpenClaw agent framework):**

| Metric | Always Sonnet | Credence routing | Change |
|--------|--------------|-----------------|--------|
| Quality (0-10) | 6.56 | 7.80 | **+1.24** |
| Avg latency | 8.4s | 4.0s | **-52%** |
| Avg cost/request | $0.024 | $0.001 | **-96%** |

## Quick start

```bash
docker run -p 8377:8377 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  -v credence-data:/data \
  credence-proxy
```

Then point any OpenAI-compatible client at `http://localhost:8377/v1`:

```bash
curl http://localhost:8377/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain quicksort"}],
    "stream": true
  }'
```

The proxy picks the best model, streams the response, and updates its beliefs from the outcome.

## How it works

All inference runs in a Bayesian DSL ([Credence](https://github.com/gfrmin/credence)). The proxy holds one opaque state and calls three DSL functions:

1. **Classify** the query into categories (code, reasoning, creative, factual, chat)
2. **Decide** which model maximises expected utility: `EU = P(quality | model, category) * reward - cost`
3. **Observe** the outcome and update beliefs via Bayesian conditioning

The proxy learns a per-model, per-category quality distribution from continuous quality scores (0-10, judged asynchronously by a fast model). Both the reliability and the noise level of the judge are learned jointly.

There are no exploration bonuses, no epsilon-greedy, no heuristics. Only EU maximisation — which naturally explores when uncertainty is high, because the value of information is positive.

## Available models

The proxy routes across all models whose API key is provided:

| Model | Provider | Key |
|-------|----------|-----|
| claude-haiku-4-5 | Anthropic | `ANTHROPIC_API_KEY` |
| claude-sonnet-4-6 | Anthropic | `ANTHROPIC_API_KEY` |
| claude-opus-4-6 | Anthropic | `ANTHROPIC_API_KEY` |
| gpt-4o-mini | OpenAI | `OPENAI_API_KEY` |
| gpt-4o | OpenAI | `OPENAI_API_KEY` |

Set at least one provider key. The proxy only routes to models with available keys.

## Search routing

The proxy also routes web searches across providers:

```bash
curl -X POST http://localhost:8377/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest rust compiler release"}'
```

| Provider | Key | Always available |
|----------|-----|-----------------|
| DuckDuckGo | none | yes |
| Brave | `BRAVE_API_KEY` | no |
| Perplexity | `PERPLEXITY_API_KEY` | no |
| Tavily | `TAVILY_API_KEY` | no |

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (enables Claude models + quality judge) |
| `OPENAI_API_KEY` | — | OpenAI API key (enables GPT models) |
| `BRAVE_API_KEY` | — | Brave Search API key |
| `PERPLEXITY_API_KEY` | — | Perplexity API key |
| `TAVILY_API_KEY` | — | Tavily API key |
| `CREDENCE_REWARD` | `1.0` | Value of a quality point (higher = prefer quality over cost) |
| `CREDENCE_LATENCY_WEIGHT` | `0.01` | Cost per second of latency (higher = prefer faster models) |
| `CREDENCE_FORCE_MODEL` | — | Bypass routing: always use this model (for A/B testing) |
| `CREDENCE_STATE_PATH` | `/data/credence-state.json` | Search state file |
| `CREDENCE_LLM_STATE_PATH` | `/data/credence-llm-state.bin` | LLM state file |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible LLM proxy (streams SSE) |
| POST | `/search` | Search routing |
| GET | `/state` | Learned reliability per model per category |
| GET | `/metrics` | Per-request log (model, tokens, cost, latency) |
| POST | `/metrics/clear` | Reset request log |
| POST | `/outcome` | Manual outcome feedback |
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (validates Julia is loaded) |

The `X-Credence-Model` response header tells you which model was selected.

## Using with docker-compose

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Then:

```bash
docker-compose up -d
```

Learned state is persisted in a Docker volume (`credence-data`). It survives container restarts.

## Using with OpenClaw

Add a custom provider to `~/.openclaw-dev/openclaw.json`:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "credence": {
        "baseUrl": "http://localhost:8377/v1",
        "apiKey": "not-needed",
        "api": "openai-completions",
        "models": [
          {"id": "claude-sonnet-4-6", "name": "Claude Sonnet (via Credence)"}
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "credence/claude-sonnet-4-6"}
    }
  }
}
```

The `model` field in the config is cosmetic — the proxy ignores it and routes to whichever model maximises expected utility. You can use any model ID from the available models table.

## Using with any OpenAI-compatible client

Any client that supports a custom base URL works:

```python
# Python (openai SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8377/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
```

```typescript
// TypeScript (openai SDK)
const client = new OpenAI({
  baseURL: "http://localhost:8377/v1",
  apiKey: "not-needed",
});
```

## Monitoring

Check what the proxy has learned:

```bash
# Per-model, per-category reliability
curl http://localhost:8377/state | python -m json.tool

# Request log with model selection, tokens, cost, latency
curl http://localhost:8377/metrics | python -m json.tool
```

## Building from source

```bash
docker build -t credence-proxy .
```

Or without Docker (requires Julia 1.9+):

```bash
cd /path/to/credence
uv sync
PYTHON_JULIACALL_HANDLE_SIGNALS=yes credence-router serve
```

## Architecture

```
Client (OpenClaw, curl, any OpenAI client)
  │
  ▼
credence-proxy (FastAPI, port 8377)
  │
  ├── Classify query → category weights
  ├── EU maximise → pick best model (Julia DSL)
  ├── Forward to provider (Anthropic/OpenAI)
  ├── Stream response back to client
  ├── Async quality judge (Claude Haiku)
  └── Update beliefs via Bayesian conditioning (Julia DSL)
  │
  ▼
Anthropic API / OpenAI API
```

The Bayesian model (defined in `examples/router.bdsl`) maintains a joint distribution over reliability and judge concentration per model per category. Learning is Bayesian conditioning — no hyperparameters to tune, no exploration schedule, no decay.

## License

AGPL-3.0-only
