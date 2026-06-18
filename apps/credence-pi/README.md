# credence-pi

**Make your OpenClaw cheaper and harder to fool.** credence-pi learns your agent
from the calls you approve, then blocks the tool calls it wastes (re-running work
it already did) and asks you before the ones that smell of prompt injection. Two
commands to install; everything runs locally, and no raw data leaves your machine.

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
# the brain (Bayesian daemon) — detached, restart-resilient, state in ~/.credence-pi
docker run -d --name credence-pi --restart unless-stopped \
  -p 127.0.0.1:8787:8787 -v ~/.credence-pi:/root/.credence-pi \
  ghcr.io/gfrmin/credence-pi-daemon

# the body (OpenClaw plugin) — governance + EU-max model routing, both ON by default
openclaw plugins install @gfrmin/credence-pi-openclaw
openclaw plugins enable credence-pi
```

That's it. **Model routing is on by default**: credence-pi auto-discovers your
configured models and routes each turn to the cheapest one whose expected accuracy
justifies its cost — no config, and it learns from your traffic (`routing: false`
to disable; a single-model setup is a no-op). Prefer Compose? `docker compose -f
apps/credence-pi/docker-compose.yml up -d` runs the same daemon.

## See it work (no agent, no data needed)

```bash
julia --project=. apps/credence-pi/demo/governance_demo.jl
```

Runs the real Pass-2 brain on a synthetic session: an agent stuck re-proposing
the same call, interleaved with one legitimate novel call. credence-pi learns to
auto-block the loop while still allowing the novel call at the same moment, by
expected-utility maximisation rather than a hard-coded rule, then prints a spend
report. Needs a local Julia; no Docker, no OpenClaw, no real data.

## What it does, measured

On real OpenClaw sessions (held-out, posterior frozen before the test arm —
see [`eval/results/`](./eval/results/)):

- exact-repeat wasted calls blocked at **precision 1.0 / recall 1.0**
  (0.7% of all calls);
- an injected exfiltration surfaced as a confirmation at **0.94 precision**,
  interrupting 1.2% of safe sessions.

**Certain in direction, not yet measured — and not free of false blocks.** When the
agent re-runs a call it already made *and nothing has changed*, blocking it hands
back the tokens, dollars, and wait that re-run would have cost. But the block keys on
`(tool, args)` alone — it does **not** check whether the workspace changed in between,
so a *legitimate* re-run (re-running the tests after an edit) is byte-identical and
would also be blocked. That false-block rate is unmeasured: the held-out corpus has no
edit-context to expose it, and the 1.0/1.0 above is against the exact-repeat
*definition* of waste, not against true waste (see
[`eval/results/FINDINGS.md`](./eval/results/FINDINGS.md)). How much you save — and how
often a block is a false one — is exactly what shadow mode measures on your own
sessions (see What's next).

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

## Why it's low-risk to try

The upside is measured and the downside is bounded by construction — and the
downside argument is the part most "smarter router" pitches can't make.

**Upside.** Routing is one expected-utility maximisation, over one learned belief,
per call, for *your* trade-off. Across 17 real Terminal-Bench tasks scored through the
live daemon, escalation's welfare beats **every** fixed single-model policy on **every**
profile — and the point is **cross-profile**: no single fixed model is best for every
user (haiku wins a cost-saver, sonnet wins balanced and quality-first), so "pick one
model and stick with it" is wrong for *someone*, and credence-pi tracks the per-profile
best without being told which. It captures the *union* of the tiers' capabilities — it
solves more tasks than any single tier (the solve-rate edge's bootstrap CI never crosses
zero). The honest qualifier is **precision, not direction**: 17 tasks is too small to put
tight error bars on the per-call welfare margins (the cluster-bootstrap CIs are wide —
that is a sample-size limit, *not* evidence the win is absent), so the *magnitude* is
carried by the better-powered offline minimax-regret result, and a head-on run on your own
logs is what would tighten it. See [`eval/live_ab/EXPERIMENT.md`](./eval/live_ab/EXPERIMENT.md).

**Low, bounded downside.**

- *Fail-open* — if the daemon is slow or down, OpenClaw runs exactly as it would
  without credence-pi. It cannot break your agent.
- *Uncertainty-gated* — when the belief is unsure, every profile collapses to the
  same sensible default; credence-pi diverges from what you'd have done anyway only
  on the calls where it has earned confidence.
- *Minutes to adopt* — two commands, the body does no math, profiles switch
  per-request with no restart.

The one real downside is the false block above: the waste governor keys on
`(tool, args)` and can stop a legitimate re-run (tests after an edit). It is small and
fail-open, not zero — so the honest recommendation is to **run shadow mode first**
(observe-only; it reports what it *would* have blocked, the would-save, and a
write/edit-aware false-block proxy) and turn on enforcement once you've seen the
false-block rate on your own traffic. Magnitude on both sides — savings and false
blocks — depends on your workload and is what early users reveal.

## Why not just a rule?

The waste result above is hash-set-easy, and the eval says so out loud. But the
governor's full ask / allow / block map is not. At one byte-identical input it can
ask or proceed depending on the **variance** of its belief, not its mean (Beta(4,4)
asks while the wider Beta(4,2) proceeds, so no threshold on variance or count can
sort it); a context it has never seen inherits an informed answer by model
averaging rather than a default; and harm and waste are integrated in one currency,
so the block cutoff on one slides with the other. Each break re-derives the
Tier-1 ops (`condition` + `voi` + `optimise`). It is reproducible and red-teamed:

```bash
julia --project=. apps/credence-pi/eval/regex_impossible.jl
```

The full argument, with the numbers, is
[What a Regex Can't Do](https://gfrm.in/posts/credence-pi-pass-2/).

## What's next

- **Live-enforcement telemetry.** Shadow mode now reports what governance *would*
  have done — counterfactual blocks, an estimated would-save, and a write/edit-aware
  false-block proxy — so an observe-only run on your own traffic yields the
  false-block rate and would-save directly. What remains is opt-in users to turn those
  counterfactuals into enforced, gold-labelled numbers, and to test the safety
  confirmations against real threats rather than a benchmark. This is the gate the
  honest claims above are waiting on.
- **Raising the safety ceiling.** A tool-boundary governor can see at most about
  three in ten of unsafe trajectories; reaching higher means reading signals
  beyond the tool call. A named frontier, not a silent gap.
- **More bodies, one brain.** The wire schema is fixed, so a new agent
  integration is body-only work; the brain and its posterior are untouched. Pass
  1's body targeted pi; Pass 2 added the OpenClaw plugin, on the same brain.

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
