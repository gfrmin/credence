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

    julia --project=. apps/credence-pi/daemon/server.jl

Default bind is `127.0.0.1:8787`. The observation log is appended to
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
