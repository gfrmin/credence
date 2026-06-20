# Decouple Move 0 — The contract

Design doc per `docs/posture-4/DESIGN-DOC-TEMPLATE.md`.

## 0. Final-state alignment

Move 0 establishes the pinnable boundary every later move depends on: a
protocol-versioned skin wire, an inline-source `initialize` that no longer reaches
into the host filesystem, a `credence-skin` image, an extracted
`credence-skin-client`, and the demotion of the juliacall embedding packages. It
converges the tip toward `master-plan.md` §"Final-state architecture" by making
the wire — not the source tree — the contract. Transient state left explicitly
unaligned after Move 0: (a) HTTP framing (`serve_http.jl`) is **deferred to Move 2**
when rssfeed is the first networked consumer to test it against — Move 0 ships
stdio-only, matching its only near-term consumer (life-agent via `docker run -i`);
(b) the path-based `dsl_files` param stays alive for co-released in-image consumers
(router) alongside the new inline `dsl_sources` — this is the co-released-image
exception, not drift; (c) `credence`/`credence-agents` remain importable
in-repo (router embeds them) — only their *public* surface is retired.

## 1. Purpose

Turn the skin from a source-tree-coupled, unversioned subprocess into a versioned,
pinnable engine artifact with a stable protocol, consumed only over the wire by a
thin client library that cannot host a brain.

## 2. Files touched

### Created
- `apps/skin/serve_http.jl` — **DEFERRED to Move 2** (named here for the record; not in this PR).
- `apps/skin/clients/python/` — extracted `credence-skin-client` package: `pyproject.toml`, `credence_skin_client/__init__.py` (the JSON-RPC plumbing + typed wrappers + ready/shutdown + protocol handshake, lifted from `apps/skin/client.py`).
- `Dockerfile.skin` (repo root) — `credence-skin` image: stage-1 depot warm-up mirroring `apps/credence-pi/daemon/Dockerfile`, entrypoint `julia --project=. apps/skin/server.jl` (stdio).

### Modified
- `apps/skin/server.jl` — add `const PROTOCOL_VERSION = "1.0"`; `handle_initialize` returns `{version, protocol, methods}`; protocol-major handshake → new error `-32010`; add inline `dsl_sources` handling; deprecate `plugins` for external use (`handle_initialize` lines 588–606); add `methods` list derived from the `handle_request` dispatch (lines 522–582).
- `apps/skin/client.py` — re-export from / thin-shim over `credence-skin-client`; add the handshake (send expected `protocol`).
- `apps/skin/protocol.md` — `Protocol-Version: 1.0` header + CHANGELOG; document `dsl_sources`, the handshake, `-32010`; mark `plugins` + path-`dsl_files` as co-released-image-only.
- `.github/workflows/publish-image.yml` — add `publish-skin` job (mirrors `publish-daemon`); add a CI step grepping protocol.md header == `PROTOCOL_VERSION`.
- `apps/python/credence_bindings/pyproject.toml`, `apps/python/credence_agents/pyproject.toml` — add `Private :: Do Not Upload` classifier + a README note: engine-repo-internal, not a public consumption surface.
- `CLAUDE.md` / `SPEC.md` — write the co-released-image exception: in-process embedding is permitted only for code co-released inside an engine-repo image; forbidden as a published library others import.

### Tests
- `apps/skin/test_skin.py` — assert `initialize` returns `{protocol, version, methods}`; assert `-32010` on a bad protocol major; assert `dsl_sources` loads an inline BDSL string equivalently to a path `dsl_files` load.
- `apps/skin/clients/python/` — a smoke test spawning the skin and round-tripping `initialize` + a `create_state`/`condition`/`weights` cycle through the extracted client.

## 3. Behaviour preserved

The method set and per-method semantics are unchanged — `handle_request`
dispatch (server.jl:522–582) is untouched except for the `initialize` return
shape and the new inline-source branch. Existing `test_skin.py` cases pass
unmodified except the `initialize` assertion (which now also sees `protocol`/`methods`).
`credence_router` keeps working in-process (path `dsl_files` against `examples/router.bdsl`
still resolves, because router runs co-released in the engine image).

## 4. Worked end-to-end example

External app, inline source, over stdio against the pinned image:

```
$ docker run --rm -i ghcr.io/gfrmin/credence-skin:0.1.0
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol":"1","dsl_sources":{"agent":"(defun ...)"}}}
{"jsonrpc":"2.0","id":1,"result":{"version":"0.1.0","protocol":"1.0","methods":["initialize","create_state",...]}}
{"jsonrpc":"2.0","id":2,"method":"create_state","params":{"type":"beta","alpha":1.0,"beta":1.0}}
{"jsonrpc":"2.0","id":2,"result":{"state_id":"s_1"}}
```

If the client sends `"protocol":"2"`, `initialize` returns
`{"error":{"code":-32010,"message":"protocol major mismatch: server 1, client 2"}}`
and the client refuses to proceed. The `credence-skin-client` `Skin.initialize()`
wrapper performs this handshake automatically from its compiled-against major.

## 5. Open design questions

1. **Inline source: new `dsl_sources` param, or overload `dsl_files` to sniff
   string-is-source vs string-is-path?** Recommend a *distinct* `dsl_sources`
   param: sniffing is exactly the kind of structural inference Invariant 2 warns
   against (a path that happens to parse as an s-expr, or vice versa, misfires
   silently). Distinct verbs keep the wire honest. Argue back if the duplication
   is judged worse than the sniff.
2. **Retire `plugins` now, or just document it internal-only?** No in-repo skin
   caller passes `plugins` (credence-pi runs its own daemon). Recommend: keep the
   param functional but document it co-released-image-only and add a one-line
   `log_msg` warning when used, rather than hard-removing — hard-removal is a
   Move-3 concern once credence-pi's refit confirms nothing depends on it.
3. **Protocol major = "1" while engine semver stays 0.1.0?** Recommend yes: the
   wire is the thing apps pin and it is stable enough to call v1; engine maturity
   is a separate axis. The split is the whole point of §2's two version fields.
4. **PyPI demotion: yank existing `credence`/`credence-agents` versions, or just
   stop publishing + mark internal?** CI never published them (manual only), so
   there is nothing to remove from CI. Recommend stop-publishing + `Private`
   classifier + README note, and **do not yank** existing versions — yanking is
   outward-facing and irreversible; defer that call to the author.

## 6. Risk + mitigation

- **Risk: the extracted client drifts from `apps/skin/client.py`.** Mitigation:
  `client.py` becomes a thin re-export, not a parallel copy — one implementation.
- **Risk: the `credence-skin` image's depot doesn't precompile the program-space
  path, so first-call latency spikes.** Mitigation: stage-1 warm-up runs a
  representative `create_state`/`condition` (mirror the credence-pi daemon
  Dockerfile, which already warms the brain).
- **Risk: inline `dsl_sources` and path `dsl_files` diverge in load semantics.**
  Mitigation: both route through the identical `load_dsl(source)`; the only
  difference is `read(path)` vs the inline string. Test asserts equivalence.
- **Review-process risk: demotion reads as a breaking change to PyPI users.**
  Mitigation: no yank; existing pins keep resolving; the change is additive
  (classifier + docs) until the author decides on yanking.

## 7. Verification cadence

`python -m apps.skin.test_skin` (skin smoke), the new `credence-skin-client`
smoke test, `julia test/test_*.jl` (engine core unchanged but run to confirm),
the credence-lint corpus self-test + `check apps/`, and a local
`docker build -f Dockerfile.skin` + `docker run -i` `initialize` round-trip.
`credence_router` pytest (it shares the `initialize` path).

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** Yes — Move 0 touches transport and
   packaging only; it adds no function returning a probabilistic `Float64`.
2. **Prevision-in-Measure or Measure-in-Prevision?** Neither — no ontology change.
3. **Opaque closure where declared structure fits?** No — and §5 Q1 explicitly
   rejects the string-sniff to *avoid* an inferred-structure hazard.
4. **`getproperty` override on a Prevision subtype?** No.
