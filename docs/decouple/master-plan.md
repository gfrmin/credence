# Decouple — master plan

> Branch family: `decouple/*`. Opened 2026-06-20; engine side **closed 2026-06-25**
> at protocol 1.12 (see "Moves" — all landed; remaining tail is cross-repo).
>
> Separate the Credence engine (DSL core + skin wire) from the apps that consume
> it, so each app pins a versioned engine artifact and talks the protocol-versioned
> wire instead of reaching into the source tree or embedding a brain in-process.

## The principle

**Credence is the engine; apps get data + a thin body, never a brain. If an app
wants its own brain, that is its problem — credence must not enable it.**

Four settled consequences:

1. **The wire is the only sanctioned consumption surface.** State stays
   server-side as opaque IDs, so Invariant 1 (single reasoner) holds *by
   construction* — a wire client physically cannot do probability arithmetic. We
   rely on the wire boundary **alone**; no shipped lint for app repos.
2. **No in-process embedding for external apps.** The juliacall path (`credence`
   bindings, `credence-agents` `BayesianAgent`) *is* a brain; it is demoted from a
   public consumption surface to engine-repo-internal only.
3. **Shared probabilistic machinery is promoted into engine stdlib** so domain
   brains are expressible as *declared BDSL data over the wire*, not app Julia.
4. **Per-consumer accretion.** Stdlib completes through consumer moves, not one
   big deliverable (precedent: Posture 4 stdlib accretion). Promote only what the
   current consumer needs.

## Final-state architecture

```
credence repo  =  ENGINE ONLY
  src/                  DSL core + stdlib (absorbs promoted machinery)
  apps/skin/            the wire: JSON-RPC, opaque state IDs, served over
                        stdio AND http off one handle_request dispatch
  ghcr.io/gfrmin/credence-skin   ← the versioned, pinnable artifact
  co-released product images (credence-proxy, credence-pi-daemon): allowed
    in-process embedding ONLY because they ship inside an engine-repo image

each app repo  =  DATA + THIN BODY
  *.bdsl                declared spaces / kernels / priors / utility (data)
  credence-skin-client  pure JSON-RPC wire client (no juliacall, no math)
```

The contract is the **skin protocol** (method set + opaque state IDs). **Transport
must match the state model:** the skin holds a single-tenant state registry, so
**stdio is the engine wire** (the OS process boundary is the tenancy boundary).
HTTP is reserved for product images with an external protocol obligation
(credence-proxy, credence-pi daemon) — *not* the skin; an HTTP skin would need
session isolation and is deferred until a real single-tenant-remote consumer needs
it. Public consumption surface = GHCR images + `credence-skin-client`. Zero
embedding packages.

## Moves — ALL LANDED ✅ (engine side closed 2026-06-25)

Execution order was reprioritized to 0 → 2 → 1 → 3 → 4 → 5 (rssfeed pulled ahead of
the life-agent pilot); listed below by move number.

| Move | Deliverable | Status |
|------|-------------|--------|
| **0 — Contract** | `credence-skin` image (stdio); `PROTOCOL_VERSION` split from engine semver + handshake (`-32010`); inline-BDSL `dsl_sources` (`.jl` plugin injection demoted to co-released-image-only); extracted `credence-skin-client`; demoted `credence`/`credence-agents`/`credence-router` from public PyPI; co-released-image exception in SPEC §6.9 + CLAUDE.md. (`serve_http` *not* built — stdio only.) See `move-0-design.md`. | ✅ PR #140 |
| **2 — rssfeed / MAUT** | Pure-functional MAUT ranker, wire-only over stdio. Uniform prevision-serialization (`params(p)` + `read_params` verb) + a family registry the BDSL `:family` surface reflects (`:normal/:soft/:weighted`); `examples/maut_demo.bdsl`. `apps/julia/rss/` → reference-only (program-space/PlackettLuce superseded). See `move-2-design.md`. | ✅ |
| **1 — life-agent pilot** | Repoint `brain.py` off `$CREDENCE_REPO` onto the pinned `credence-skin` image (`docker run -i` stdio, reusing `SubprocessTransport`); life-agent BDSL in-repo, inline via `dsl_sources`; host-side arithmetic-leak audit. Thin brain — no stdlib promotion. See `move-1-design.md`. | ✅ |
| **3 — credence-pi refit** | StructureBMA builder lifted into `src/structure_bma.jl`; `structure_bma` / `structure_observe` / `belief_at_context` skin verbs (protocol 1.2; body never transcribes the 2ⁿ prior); credence-pi BDSL stays data; TS wire client. See `move-3-design.md`. | ✅ |
| **4 — routing-brain lift** | Routing brain lifted into engine stdlib (`src/routing.jl`) + routing skin verbs (protocol 1.5); self-contained routing-stdlib oracle. See `move-4-design.md`. | ✅ |
| **5 — credence-pi extract** | credence-pi removed from `apps/`; rehomed to the standalone **credence-openclaw** repo as a pure TS wire consumer (own CI + production daemon image). See `move-5-design.md`. | ✅ |
| **engine-owns-grid (Phases A/B/C)** | Dispose the discretisation antipattern in the last consumer (life-agent body): the body declares CONTINUOUS beliefs + likelihood kernels, the engine owns ALL discretisation. **A** — continuous-τ reaction + `truncated_gaussian` support + sequential `QuadraturePrevision` conditioning, `discretised_gaussian` deleted (protocol 1.9/1.10); **B** — `MvQuadraturePrevision` + `margin_reaction` + `marginal` verb, metareasoned grid fidelity (no hard cap), `marginalise(shape)` retired (protocol 1.11); **C** — exact closed-form `RhoCategoricalPrevision` ρ-latent (Beta-moment sum, no grid), `label_prior`/`_RHO_GRID` disposed (protocol 1.12). Engine oracles: `test/test_quadrature_sequential_condition.jl`, `test/test_mv_quadrature.jl`, `test/test_rho_conjugate.jl`. | ✅ |

Each move = design-doc PR then code PR, per `docs/posture-4/DESIGN-DOC-TEMPLATE.md`.

**Arc status (2026-06-25): engine side COMPLETE at protocol 1.12.** All consumption is over
the versioned `credence-skin` wire; `credence-openclaw` is a pure wire consumer; the
`credence-skin` 1.12 image is published (CI-green on master `4b98c14`). The remaining tail is
**cross-repo**: land the life-agent `decouple/body` branch (the Phase A/B/C strict-body rewrite)
and pin the published 1.12 image digest in life-agent's `brain.py`.

## Residual risks (RESOLVED in the moves that hit them)

Three control-flow spots where "no app Julia" forced either thin multi-call wire
choreography or a small stdlib decision template — all settled in Moves 3–5 (the
credence-pi refit + routing-brain lift + extraction; credence-pi now lives in
`credence-openclaw` as a pure wire consumer, so these are exercised over the wire).
The rule that drove each decision: default to a stdlib template if the body would
otherwise do utility/probability arithmetic.

1. **Utility-coefficient assembly** (credence-pi `EU(block)=c·(1+m)(1−θ)−cλθ+H·θ_u`)
   — lean to a `linear_block_eu` decision template.
2. **VOI-as-constant action** (`decide`'s `ask` folds a `net_voi` scalar back into
   `optimise`) — two round-trips or a `decide_with_voi` template.
3. **EM confound coupling** (`routing_brain.jl route_outcome!` decode-then-couple)
   — thin body choreography (~4 skin calls, no math) or a `confounded_outcome_update`
   stdlib op.

None needs a new frozen type or an axiom change.
