# credence-pi daemon

The brain. A Julia HTTP server that loads the five BDSL files under
`apps/credence-pi/bdsl/` at startup, holds the posterior as a `Measure`
in the loaded environment, and exposes two endpoints:

    POST /sensor    accepts one sensor event (tool-proposed,
                    user-responded, tool-completed); returns
                    {"ack": true, "event_id": "..."}.

    GET  /signals   long-lived Server-Sent Events stream;
                    each effector signal arrives as one SSE message.

The wire schema is fixed by [`../SPEC.md`](../SPEC.md). The daemon does
no probabilistic reasoning of its own — it loads the BDSL environment
and calls the public entry points (`decide-action`, `observe-response`,
`followup-after-response`).

## Run

    julia --project=<repo-root> apps/credence-pi/daemon/main.jl

`main.jl` is the standalone entrypoint: it loads `Credence`, starts the
HTTP server, and blocks until signalled. (`server.jl` is a module that
expects `Credence` already loaded in `Main` — it is `include`d by
`main.jl`, the tests, and the demo, not run directly.) Or run the
published image:

    docker run -d --name credence-pi --restart unless-stopped \
      -p 127.0.0.1:8787:8787 -v ~/.credence-pi:/root/.credence-pi \
      ghcr.io/gfrmin/credence-pi-daemon

Or via Compose (same daemon, restart-resilient across reboots):

    docker compose -f apps/credence-pi/docker-compose.yml up -d

Default bind is `127.0.0.1:8787`. Override via environment:
`CREDENCE_PI_HOST`, `CREDENCE_PI_PORT`, `CREDENCE_PI_BDSL_DIR`,
`CREDENCE_PI_LOG`. The observation log is appended to
`~/.credence-pi/observations.jsonl`; on startup the daemon replays the
log to reconstruct the posterior.

## Test

    julia --project=. apps/credence-pi/tests/julia/test_observation_log.jl
    julia --project=. apps/credence-pi/tests/julia/test_server.jl

## Observation log

Append-only JSONL at `~/.credence-pi/observations.jsonl`. One line per
sensor event:

    { "schema": "credence-pi/v1", "received_at": "...", "event": { ... } }

The daemon `fsync`s after each append; correctness over throughput.
Pass 1 has one writer per process — concurrent daemons are unsupported.
Pass 2 may bump the schema; see `SPEC.md` § "Decisions deferred to
Pass 2".
