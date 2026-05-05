# credence-pi

Pass-1 implementation of the body-brain governance loop for the pi
extension surface. The brain is a Julia daemon that loads BDSL programs
and holds a Beta posterior over `P(approve)`. The body is a TypeScript
extension that hooks pi's `tool_call` and exchanges sensor events /
effector signals with the brain over HTTP.

The binding specification is [`SPEC.md`](./SPEC.md). Read it before
making changes; the body-brain split, the wire schema, and the lint
pragma policy are not negotiable in Pass 1.

## Layout

    bdsl/        — capabilities, features, prior, kernel, decide
    daemon/      — Julia HTTP server holding the BDSL environment
    extension/   — TypeScript pi-side body
    tests/julia/ — Julia tests for BDSL + daemon
    tests/typescript/ — TS tests for manifest, features, client, hooks

See [`daemon/README.md`](./daemon/README.md) and
[`extension/README.md`](./extension/README.md) for run instructions.

## Develop

    # Julia tests (one file at a time):
    julia --project=. apps/credence-pi/tests/julia/test_bdsl.jl
    julia --project=. apps/credence-pi/tests/julia/test_observation_log.jl
    julia --project=. apps/credence-pi/tests/julia/test_server.jl
    julia --project=. apps/credence-pi/tests/julia/test_bdsl_primitives.jl

    # TypeScript build + tests:
    cd apps/credence-pi/extension
    npm install
    npm run build
    npm test

    # Lint:
    python3 tools/credence-lint/credence_lint.py --repo-root . check apps/credence-pi/

## Pass-1 scope

Single Beta(2,2) prior, no feature conditioning at decision time, three
effectors (`ask`, `proceed`, `block`). EVPI is computed inline by the
stdlib `voi`; nothing is hard-coded as a magic number. Pass 2 will
replace the global posterior with a CEG over features without
disturbing the wire schema or the body — that is the architectural
payoff of the Pass-1 discipline.

See `SPEC.md` § "Out of scope (Pass 1)" and "Decisions deferred to
Pass 2" for the line.
