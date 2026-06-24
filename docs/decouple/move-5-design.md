# Decouple Move 5 — extract credence-pi → the credence-openclaw repo (pure wire consumer)

## 0. Final-state alignment

Moves 0–4 made credence-pi's *entire* reasoning surface wire-drivable: governance
(`structure_*` verbs, Move 3) and routing (`routing_*` verbs, Move 4) both run server-side
in the engine, reachable over the protocol-1.5 skin wire. The daemon, however, still
**embeds** the engine in-process (`daemon/main.jl` `using Credence`). Move 5 performs the
literal extraction the master plan deferred: credence-pi leaves the engine repo, is renamed
**credence-openclaw**, and its daemon is rewired as a **pure wire consumer** — it pins the
`credence-skin` image and drives the verbs, carrying **zero probabilistic Julia**. This
satisfies the consumption boundary for an *external* app (`SPEC §6.9`): the co-released-image
embedding exception is for engine-repo images only; a separate repo must use the wire.

## 1. Purpose & decisions

- **Architecture = B (pure wire consumer)** and **rename → credence-openclaw** (both locked
  this session). Option A (re-publishing an embeddable engine) is off the table — it
  contradicts Move 0's `Private :: Do Not Upload`.
- **Daemon language = TypeScript** (locked). It unifies the repo: the `openclaw-plugin` +
  `extension` are already TS and npm-published, and OpenClaw is a JS framework. The daemon
  becomes a TS HTTP/SSE server that drives `credence-skin` over a new **TS skin wire client**
  (JSON-RPC over stdio to `docker run -i credence-skin`). The repo ends up single-toolchain
  (TS) with no Julia.

## 2. Repos & files touched

### New repo: `credence-openclaw` (`~/git/credence-openclaw`, GitHub `gfrmin/credence-openclaw`, default branch `master`)
- **Stays (TS, unchanged in shape):** `openclaw-plugin/` (the body — HTTP→daemon, already
  arithmetic-free), `extension/` (test body).
- **Stays (declared DATA):** `bdsl/` (feature schema, roster, utility constants),
  `profiles/`, the brain `*.counts.json` artifacts (shipped INLINE to `routing_init`/
  `structure_*` over the wire), the eval harness, docs.
- **New (TS):**
  - `skin-client/` — a TS JSON-RPC-over-stdio client to a pinned `credence-skin` image (the
    daemon→skin analogue of the Python `credence_skin_client`; spawns `docker run -i
    credence-skin@<digest>`, holds opaque `state_id`/`model_id`/`rt_*` handles, exposes
    `initialize`/`structure_bma`/`structure_observe`/`structure_decide`/`routing_init`/
    `routing_decide`/`routing_escalate`/`routing_outcome`/`routing_belief`).
  - `daemon/` (TS) — the rewritten daemon: the HTTP/SSE server (`POST /sensor`, `GET
    /signals` SSE, `GET /report`), the observation log (append + replay — host-side,
    non-probabilistic), and the boot/per-event orchestration that calls the verbs. Replaces
    `daemon/{main,server}.jl` + `brain/{feature,routing}_brain.jl` (the brains are gone —
    their math is in the engine).
- **Deleted (the embedded Julia brain):** `brain/*.jl`, `daemon/*.jl`, `daemon/Dockerfile`
  (the multi-stage build that COPY'd the engine `src/`). The new `daemon/Dockerfile` is a TS
  image that runs the skin image as a pinned dependency (sidecar or spawned subprocess).

### Engine repo (`credence`, this branch `decouple/openclaw-extract`)
- **Remove** `apps/credence-pi/`.
- **Strip CI** (`.github/workflows/publish-image.yml`): the `unit-tests` credence-pi steps
  (Julia tests + extension + openclaw-plugin builds), the `daemon-smoke-build` job, the
  `publish-daemon` job, and the workflow name's "+ credence-pi daemon".
- **Repoint the two engine tests that reach into the departing app:**
  `test/test_sparse_structure_equivalence.jl` and `test/test_product_bma_routing.jl`
  `include` `apps/credence-pi/brain/feature_brain.jl` — repoint them onto the lifted
  `src/structure_bma.jl` / `src/routing.jl` (the substrate they test is now engine-resident),
  or fold their unique coverage into `test/test_structure_bma.jl` / `test/test_routing.jl`.
- **Docs:** update `SPEC.md` / `CLAUDE.md` / `docs/decouple/master-plan.md` mentions of the
  in-repo credence-pi to point at the external repo (the `credence-pi-daemon` co-released
  exception is retired — it's now an external wire consumer).

## 3. Behaviour preserved

The credence-openclaw daemon reproduces the in-repo Julia daemon's **decisions** through the
pinned `credence-skin` image: same `POST /sensor` → effector signal, same SSE `/signals`,
same `/report`, same boot reconstruction (warm counts + observation-log replay). Golden-trace
parity: a recorded sensor stream produces the same proceed/block/ask + route decisions
through the TS-wire daemon as through the Julia-embedding daemon (the verbs are the same
engine code; only the transport changed). The `openclaw-plugin` body is byte-unchanged.

## 4. Worked end-to-end example (the new boot + sensor path)

```
daemon boot:
  skin = spawn("docker run -i credence-skin@<digest>"); skin.initialize()
  gov  = skin.structure_bma({feature_names, feature_values, …from bdsl…})   → {model_id, state_id}
  rt   = skin.routing_init({roster, reward, warm_counts:<inline counts.json>, …})  → {routing_state_id}
  replay observation log: for each event → skin.structure_observe(...) / skin.routing_outcome(...)

POST /sensor {tool-proposed, features, proposed_call}:
  action = skin.structure_decide({model_id, state_id, features, cost, aversion, interrupt_cost, harm})
  route  = skin.routing_decide({routing_state_id, features, roster, profile})
  emit SSE signal(action, route); append to observation log; return ack
```
Zero arithmetic in the daemon — every probability/utility computation is a verb call.

## 5. Build sequencing (each a reviewable unit)

0. **Engine prerequisite — `structure_bma` gains inline `warm_counts`** (credence repo, this
   branch). Review finding: the daemon warm-seeds FOUR governance brains (waste/harm/tail +
   latency) via `reconstruct_posterior` (replays `observe` per count). `routing_init` already
   takes `warm_counts` inline and reconstructs server-side in one call, but `structure_bma`
   does not — so governance warm-seeding over the wire would be N `structure_observe`
   round-trips per brain (hundreds at boot). Add an optional `warm_counts` param to
   `structure_bma` (engine `build_structure_prior` + the existing
   `reconstruct_routing_tops_from_data`-style replay; the skin handler reconstructs from the
   inline counts), so governance warm-seeds in ONE server-side call, symmetric with routing.
   Additive → protocol 1.5 → 1.6. (The `/report` savings path needs no readback verb —
   `savings_report` is pure log IO, touches no beliefs.) Lands + republishes the skin before
   the daemon pins it; dev/test spawns this branch's `server.jl` directly.
1. **Stand up `credence-openclaw`** — create the repo, move the TS-stays + DATA, rename every
   load-bearing identifier (§ rename map). Repo builds (TS) with the brains/daemon-Julia
   removed (daemon stubbed). *Open Q: repo history — `git filter-repo` to preserve
   `apps/credence-pi` history, or a fresh repo with a provenance note?*
2. **TS skin client** — the JSON-RPC-over-stdio wrapper; unit-tested against a pinned
   `credence-skin` image (the verbs round-trip).
3. **TS daemon** — HTTP/SSE/log + the boot/per-event verb orchestration; golden-trace parity
   vs the Julia daemon's decisions.
4. **New-repo CI + publishing** — TS build/test, the daemon image (`FROM`/pins
   `credence-skin`), npm `@gfrmin/credence-openclaw`, GHCR `credence-openclaw-daemon`.
5. **Engine-repo cleanup** (this branch) — remove `apps/credence-pi/`, strip CI jobs, repoint
   the two engine tests, update docs. Engine CI stays green.

### Rename map (load-bearing identifiers, scout-confirmed)
`apps/credence-pi/` dir; plugin id `credence-pi`; npm `@gfrmin/credence-pi-openclaw` →
`@gfrmin/credence-openclaw`; image `credence-pi-daemon` → `credence-openclaw-daemon`;
`CREDENCE_PI_*` env → `CREDENCE_OPENCLAW_*`; `~/.credence-pi/` state dir →
`~/.credence-openclaw/`. (Cosmetic doc/comment mentions follow.)

## 6. Open design questions

1. **Image pinning / composition.** Digest-pin `credence-skin` (reproducibility). How does
   the daemon image run it — spawn `docker run -i` (needs docker-in-docker or a mounted
   socket), or compose the skin as a sidecar the daemon connects to over a pinned local
   transport? `docker run -i` stdio is simplest for a single-tenant daemon; a sidecar needs a
   non-stdio skin transport (the skin is stdio-only today — `serve_http.jl` was deferred).
   **Leaning:** the daemon spawns the skin as a child via the `command` seam (stdio), exactly
   as the Python client does; the daemon image bundles the skin image reference and a docker
   socket, or ships the skin binary. Resolve at build step 3.
2. **Repo history** (filter-repo vs fresh) — step 1.
3. **BDSL-as-data transcription.** The daemon must turn `bdsl/*.bdsl` (feature schema, roster,
   utility) into the JSON the verbs take. Parse the `.bdsl` in TS, or pre-bake a JSON config
   committed alongside? **Leaning:** pre-baked JSON config (the `.bdsl` stays as the
   human-authored source; a small build step emits the verb-payload JSON) — avoids a TS
   S-expression parser.
4. **`serve_http.jl` revisited?** If a sidecar transport is wanted (Q1), the skin needs HTTP;
   that reopens the Move-0-deferred `serve_http.jl`. Out of scope unless Q1 picks sidecar.

## 7. Verification cadence

- TS skin client unit tests (verbs round-trip against a pinned skin image).
- Daemon golden-trace parity: a recorded sensor stream → identical decisions vs the Julia
  daemon (capture the Julia daemon's decisions PRE-extraction as the oracle).
- `grep -rE 'using Credence|\.jl$' credence-openclaw/{daemon,skin-client}` finds **zero**
  probabilistic Julia in the consumer.
- New-repo CI green (TS build/test + daemon image smoke: initialize handshake through the
  pinned skin).
- Engine repo: CI green after `apps/credence-pi/` removal + the two tests repointed; lint
  `check apps/` clean on the smaller tree.

## 8. Reviewer checklist

- [ ] credence-openclaw carries no `*.jl` probabilistic code; the daemon drives only verbs.
- [ ] `credence-skin` image digest-pinned; daemon reproduces Julia-daemon decisions
      (golden trace).
- [ ] Every load-bearing `credence-pi` identifier renamed; existing npm/image consumers noted
      (one-time re-publish under the new name).
- [ ] Engine repo: `apps/credence-pi/` gone, CI jobs stripped, the two reach-in tests
      repointed onto `src/`, engine CI green.
- [ ] `openclaw-plugin` body unchanged (only the daemon it talks to changed).
