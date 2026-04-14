# Tuning credence-proxy

## The reward model

credence-proxy picks the action (model choice) that maximises expected utility. The utility of a chosen model on a given query is:

    EU = CREDENCE_REWARD · P(quality | model, category) − CREDENCE_LATENCY_WEIGHT · latency_seconds − dollar_cost

Three numbers, one equation. Neither `CREDENCE_REWARD` nor `CREDENCE_LATENCY_WEIGHT` is magical: they set the exchange rate between the three things you care about — getting a good answer, getting it quickly, and not spending too much to get it. Dollar cost is always measured in dollars. `CREDENCE_REWARD` decides how many dollars a quality point is worth; `CREDENCE_LATENCY_WEIGHT` decides how many dollars a second of waiting costs.

The defaults (`1.0` and `0.01`) mean: one expected quality point is worth a dollar, one second of latency is worth a cent. Good starting point; often not the right point for your workload.

## Preset profiles

### Quality-first

```
CREDENCE_REWARD=1.0
CREDENCE_LATENCY_WEIGHT=0.001
```

Use for chatbots and agents where a human is waiting for the answer anyway, so shaving seconds isn't worth sacrificing quality. Expensive models win more often.

### Cost-optimised

```
CREDENCE_REWARD=0.5
CREDENCE_LATENCY_WEIGHT=0.01
```

Use for bulk, batch, or background workloads. A quality point is still worth fifty cents, but not enough to pay Opus prices. Cheap models win unless the cheap model is genuinely unreliable for the category.

### Latency-critical

```
CREDENCE_REWARD=0.5
CREDENCE_LATENCY_WEIGHT=0.1
```

Use for autocomplete, inline suggestions, or anything where a long latency means the request is wasted (user moved on). A second of latency costs ten cents; even good models lose if they're slow.

## Is it actually tuned?

Two endpoints tell you whether the proxy is learning useful distinctions or just rubber-stamping one model.

`GET /state` returns the reliability posterior per (model, category) pair. After ~50 requests of traffic, categories where the proxy has converged will show one or two models with posterior mean above 0.8 and others well below. If every model in a category has reliability near the prior, you haven't given it enough data yet.

`GET /metrics` returns the per-request log. Scan the last 20 or 30 entries: if the same model is chosen regardless of the inferred category, one of two things is true — either `CREDENCE_LATENCY_WEIGHT` is so high that it swamps the quality signal (lower it), or `CREDENCE_REWARD` is so low that no quality difference can outweigh the cost of the more expensive model (raise it).

## How learning proceeds

Every completion contributes evidence. The quality judge (Claude Haiku) scores the response asynchronously, and the proxy updates its posterior over each model's reliability for the inferred query category. If you want to supply ground-truth quality signals yourself — from user feedback, from downstream task success — `POST /outcome` accepts them and they update the posterior the same way the judge's scores do.

Learned state persists to the `/data` volume. Resetting (`docker compose down -v`) wipes it; the next startup begins from uniform priors again.

## Back to the quickstart

[`examples/openclaw/README.md`](../../examples/openclaw/README.md)
