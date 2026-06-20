# Decouple — master plan

> Branch family: `decouple/*`. Opened 2026-06-20.
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

## Moves

| Move | Branch | Deliverable |
|------|--------|-------------|
| **0 — Contract** ✅ | `decouple/contract` | **DONE (PR #140, merged).** `credence-skin` image (stdio); `PROTOCOL_VERSION` split from engine semver + handshake (`-32010`); inline-BDSL `dsl_sources` (`.jl` plugin injection demoted to co-released-image-only); extracted `credence-skin-client`; demoted `credence`/`credence-agents`/`credence-router` from public PyPI; co-released-image exception in SPEC §6.9 + CLAUDE.md. (`serve_http` *not* built — stdio only.) |
| **2 — rssfeed** (next, reprioritized ahead of Move 1) | `decouple/rssfeed` | MAUT ranker enablement, **wire-only over stdio**. Two protocol-shaped engine additions: a uniform prevision-serialization protocol (`params(p)` + `read_params` verb) and a family registry the BDSL `:family` surface reflects (`:normal/:soft/:weighted`). Domain-neutral `examples/maut_demo.bdsl` + end-to-end example + `/load/observe/score`→skin-verb mapping (rssfeed-side adapter). `apps/julia/rss/` → reference-only (program-space/PlackettLuce superseded; read-order is noise for a swipe-reader). See `move-2-design.md`. |
| **1 — life-agent pilot** (after rssfeed) | `decouple/life-agent` | Repoint `brain.py` off `$CREDENCE_REPO` onto the pinned `credence-skin` image (`docker run -i` stdio, reusing `SubprocessTransport`); move life-agent BDSL in-repo, passed inline as `dsl_sources`; audit for host-side probability arithmetic. Thin brain — no stdlib promotion. |
| **3 — credence-pi refit** | `decouple/credence-pi` | Lift StructureBMA builder into `src/structure_bma.jl`; add `structure_bma` / `structure_observe` / `belief_at_context` skin verbs (body never transcribes the 2ⁿ prior); credence-pi BDSL stays as data; TS wire client; bless the daemon as a co-released product image. |

Each move = design-doc PR then code PR, per `docs/posture-4/DESIGN-DOC-TEMPLATE.md`.

## Residual risks (decided per move, not plan-gating)

Three control-flow spots where "no app Julia" forces either thin multi-call wire
choreography or a small stdlib decision template. Default to a stdlib template if
the body would otherwise do utility/probability arithmetic:

1. **Utility-coefficient assembly** (credence-pi `EU(block)=c·(1+m)(1−θ)−cλθ+H·θ_u`)
   — lean to a `linear_block_eu` decision template.
2. **VOI-as-constant action** (`decide`'s `ask` folds a `net_voi` scalar back into
   `optimise`) — two round-trips or a `decide_with_voi` template.
3. **EM confound coupling** (`routing_brain.jl route_outcome!` decode-then-couple)
   — thin body choreography (~4 skin calls, no math) or a `confounded_outcome_update`
   stdlib op.

None needs a new frozen type or an axiom change.
