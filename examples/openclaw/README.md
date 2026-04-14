# OpenClaw + credence-proxy quickstart

Point OpenClaw at a Bayesian router that learns which model is worth using for each kind of query. On the benchmark that ships with the project, this lifts quality by **+1.24** (0–10 scale) while cutting latency by **52 %** and cost by **96 %** versus always-Sonnet.

The proxy is OpenAI-compatible. OpenClaw talks to it exactly the same way it talks to OpenAI.

## Prerequisites

- Docker and `docker compose`
- OpenClaw installed
- At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

## Three-step quickstart

```bash
cp .env.example .env
# edit .env — add your API key(s)
docker compose up -d
```

The image is pulled from `ghcr.io/gfrmin/credence-proxy:latest`. First start precompiles Julia; `docker compose ps` will flip the service to `healthy` in ~60 s.

## Wire OpenClaw to the proxy

Add this to `~/.openclaw-dev/openclaw.json`:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "credence": {
        "baseUrl": "http://localhost:8377/v1",
        "apiKey": "not-needed",
        "api": "openai-completions",
        "models": [{"id": "auto", "name": "Credence (auto-routed)"}]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "credence/auto"}
    }
  }
}
```

Restart OpenClaw. Requests now flow through the proxy; every completion updates the proxy's posterior over which model is reliable for which query category.

## Verify it's routing

Send a request directly to the proxy:

```bash
curl -s http://localhost:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "write a haiku about Bayesian inference"}]}' \
  | jq .
```

Then inspect the routing decision:

```bash
curl -s http://localhost:8377/metrics | jq '.[-1]'
```

You should see the selected model, the inferred category, the latency, and the cost. `curl -s http://localhost:8377/state | jq` returns the current reliability posterior for every (model, category) pair.

## Tuning

The defaults (`CREDENCE_REWARD=1.0`, `CREDENCE_LATENCY_WEIGHT=0.01`) are a good general-purpose starting point. For different workloads — cost-sensitive batch, latency-critical autocomplete, quality-first chat — see [**docs/adoption/tuning.md**](../../docs/adoption/tuning.md) for preset profiles and how to tell whether the proxy is actually learning useful distinctions.

## Stopping / state

```bash
docker compose down           # stop, keep learned state
docker compose down -v        # stop, wipe learned state (reset priors)
```

The learned posteriors live in the `credence-state` volume. A fresh container attached to the same volume resumes with everything it has learned.
