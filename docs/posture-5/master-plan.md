# Master plan — credence-proxy v0.1

Branch: TBD (opens when first design conversation lands).

## Context

Posture 4 completed the de Finettian migration: every layer of the codebase speaks Prevision directly, the Measure compatibility surface is retired, and the substrate has verified-clean container typing with lint coverage. The product surface — credence-proxy, a Bayesian AI gateway for LLM/search routing — runs against this foundation.

Posture 5 operationalises the proxy. The goal is a credence-proxy v0.1 release: benchmarked, cache-disciplined, and packaged for distribution.

## Provisional moves

These are drawn from the architectural-review conversation (2026-04-27) and are provisional. Posture 5's first design conversation will refine them.

### Move 0 — Cache-discipline audit on credence-router

Audit the Python credence-router package for cache-discipline gaps: are DSL results being cached in ways that drift from the posterior? Are there stale-state bugs in the long-running server process? The audit produces a findings document and a punch list.

### Move 1 — Real-workload benchmark methodology design

Design a benchmark methodology that measures credence-proxy against real workloads: latency, routing accuracy, cost. The methodology document specifies the workload, the metrics, the baselines, and the evaluation protocol.

### Move 2 — Benchmark execution

Execute the benchmark methodology from Move 1. Produce results, analysis, and a write-up.

### Move 3 — Release engineering for credence-proxy v0.1

Package credence-proxy for distribution: Docker image tagging, versioned releases, documentation for users, operational runbook.

### Move 4 — Distribution and write-up

Publish v0.1. Write the release announcement.

## Deferred to Posture 6

The personal-agent direction — `Connection` abstraction, Maildir reader, Telegram trainer, server loop, schema v4 — is deferred to a later posture (provisionally Posture 6). The design questions from the original Move 9 draft (event-form vs parametric-form convention for email observations, Telegram preference encoding, production persistence schema) require empirical evidence from the proxy's deployment to inform the brain/body interface design. See `docs/posture-6-prep/personal-agent-priors.md` for the preserved design priors.
