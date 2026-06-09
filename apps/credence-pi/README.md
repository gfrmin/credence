# credence-pi

In-loop Bayesian governance for agentic coding/agent tools. The **brain** is
a Julia daemon that loads BDSL programs and holds a posterior over
`P(approve | context)`; the **body** hooks an agent's tool-call boundary and
exchanges sensor events / effector signals with the brain over HTTP, mapping
the brain's expected-utility decision to **proceed / ask / block**.

As of Pass 2 the brain is **feature-conditioned**: it learns a different
approval rate per context via *structure-BMA* — a Bayesian model average over
which of the declared features (`tool-name`, `working-directory-relative`,
`parent-tool-call-name`, `recent-repetition-count`, `time-since-last-user-message`)
drive approval. This is what lets it do the surgical thing a single global
posterior cannot: block a repeated wasteful loop while still allowing a novel
call, at the same moment.

The binding specification is [`SPEC.md`](./SPEC.md). Read it before making
changes; the body-brain split, the wire schema, and the lint pragma policy are
not negotiable.

## Install

```bash
# the brain
docker run -p 8787:8787 -v ~/.credence-pi:/root/.credence-pi ghcr.io/gfrmin/credence-pi-daemon

# the body
openclaw plugins install @gfrmin/credence-pi-openclaw
openclaw plugins enable credence-pi
```

## What it does, measured

On real OpenClaw sessions (held-out, posterior frozen before the test arm —
see [`eval/results/`](./eval/results/)):

- exact-repeat wasted calls blocked at **precision 1.0 / recall 1.0**
  (0.7% of all calls);
- an injected exfiltration surfaced as a confirmation at **0.94 precision**,
  interrupting 1.2% of safe sessions.

The label: research-stage. Waste-blocking is enforced; safety governance ships
in **confirm mode** (harm-driven stops are questions, never silent blocks, and
each answer calibrates the belief). The governor lives at the tool boundary,
so it is structurally blind to harmful *output*, and the harm it can see there
tops out at about three in ten of unsafe trajectories on the benchmark. If you
try it, [an issue](https://github.com/gfrmin/credence/issues) saying whether
the confirmations land on real threats or merely annoy you is exactly the
telemetry this stage needs.

The long-form story: [the announcement](https://gfrm.in/posts/openclaw-cheaper-and-harder-to-fool/),
[the architecture](https://gfrm.in/posts/credence-pi-pass-1/), and
[what the brain learned](https://gfrm.in/posts/credence-pi-pass-2/).

## Layout

    bdsl/            — declared data: capabilities, features, utility constants
    brain/           — feature_brain.jl: typed Julia that declares the
                       structure-BMA family + readout Functionals and calls
                       the Tier-1 axiom ops (the Route-B brain)
    daemon/          — Julia HTTP server (server.jl) + standalone entrypoint
                       (main.jl); holds the posterior, replays the log
    openclaw-plugin/ — OpenClaw body (installable plugin; the published surface)
    extension/       — pi-side body (targets the pi coding agent directly)
    demo/            — governance_demo.jl: the surgical win, end to end
    tests/julia/     — Julia tests (feature_brain, server, observation_log, …)
    tests/typescript/— TS tests for the bodies

The two bodies share one brain and one wire (POST `/sensor`, SSE `/signals`).

## Run (operator)

The published OpenClaw plugin is `@gfrmin/credence-pi-openclaw`. See
[`openclaw-plugin/README.md`](./openclaw-plugin/README.md) for the install
runbook and [`daemon/README.md`](./daemon/README.md) for running the brain
(from source via `daemon/main.jl`, or the `ghcr.io/gfrmin/credence-pi-daemon`
image).

## Develop

    # Julia tests (one file at a time):
    julia --project=<repo-root> apps/credence-pi/tests/julia/test_feature_brain.jl
    julia --project=<repo-root> apps/credence-pi/tests/julia/test_server.jl
    julia --project=<repo-root> apps/credence-pi/tests/julia/test_observation_log.jl

    # TypeScript build + tests (OpenClaw body):
    cd apps/credence-pi/openclaw-plugin && npm install && npm run build && npm test

    # Lint (credence-pi's stricter rule — zero production-side pragmas):
    python3 tools/credence-lint/credence_lint.py --repo-root . check apps/credence-pi/

## Scope

Three effectors (`ask`, `proceed`, `block`) chosen by expected-utility
maximisation with a linear-cost utility; `ask` is gated by value-of-information
(`voi`). EVPI and the decision threshold are computed by the stdlib, never
hard-coded. Pass 2 replaced Pass 1's single global Beta with the
feature-conditioned structure-BMA brain without disturbing the wire schema or
the bodies — the architectural payoff of the body-brain discipline. See
`SPEC.md` and `docs/credence-pi-pass-2/`.
