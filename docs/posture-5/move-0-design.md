# Move 0 — Cache-discipline audit

## 0. Strategic context

Posture 4 closed at PR #76 with the de Finettian migration complete and the substrate verified clean. Posture 5 opens with credence-proxy v0.1 as the product target. Move 0 is a cache-discipline audit on the proxy: does the proxy's interposition between client and provider preserve the conditions under which prompt caching works?

The audit gates Move 1 (benchmark methodology design). If cache discipline is broken, Move 1's methodology must account for the breakage or the proxy must be repaired first. If cache discipline is clean, Move 1 proceeds against a known-clean substrate.

**Primary finding from inspection.** The proxy routes Anthropic traffic through `https://api.anthropic.com/v1/chat/completions` — the OpenAI-compatible endpoint — not Anthropic's native Messages API (`/v1/messages`). This discovery reframes the audit's entire substantive scope. Anthropic's user-directed `cache_control: {"type": "ephemeral"}` markers are a feature of the native Messages API; they are inapplicable on the OAI-compatible endpoint. The applicable caching mechanism is **automatic prefix-based caching**: identical message prefixes of sufficient length are cached server-side, per-model, with no client-side markers needed.

The audit question is therefore not "do `cache_control` markers survive proxy interposition?" but "does the proxy preserve the conditions for automatic prefix caching?"

## 1. Scope

Anthropic-routed traffic only. OpenAI's automatic caching is opaque to the user and the proxy doesn't interact with it directly; local-model traffic has no caching layer to break. The audit is meaningfully scoped to Anthropic's OAI-compatible endpoint.

Two substantive audit activities:

1. **Code-review plus empirical verification of request-mutation determinism** (modes a and b). Does the proxy's request transformation produce stable, byte-identical outbound bodies for identical inputs routed to the same model? Verified by code review of the mutation path and confirmed by a two-request diff test against a recording mock.

2. **Routing-distribution measurement** (mode c). When Bayesian routing distributes traffic across models, each model maintains its own prefix cache. The audit measures the routing distribution and same-model streak length across a stopgap multi-turn workload.

Two named outputs:

- **Cache-discipline verdict**: one of four options (clean / broken / broken-with-repair-path / broken-with-architectural-implications). The verdict is what Move 1's methodology design proceeds against.
- **Cache-observability-gap finding**: the proxy currently discards cache-related usage metadata from provider responses (`provider.py:218-220` parses only `prompt_tokens`/`completion_tokens`, not `cached_tokens`, `cache_creation_input_tokens`, or `cache_read_input_tokens`). This means that even if cache discipline is verified clean, *measuring* the cache's actual cost-savings benefit empirically requires either parsing those fields or attributing cost from request-side estimation. Move 1's benchmark methodology cannot answer "did the cache save us money?" without one of these mechanisms in place. This finding is a named constraint that Move 1's methodology design absorbs.

## 2. Out of scope

- OpenAI-side caching analysis.
- Local-model caching.
- Benchmark methodology design (Move 1).
- Production workload selection (Move 1).
- Any code changes to the proxy. If the audit surfaces a discipline failure, the repair is a separate PR.
- Cost-claim re-evaluation (Move 2's benchmark execution).
- Native Messages API migration design (named in §5.6 as a documented option-on-the-table; not designed here).

## 5. Design decisions

### 5.1 — Endpoint discovery and scope reframing

**The question.** What does "cache discipline" mean for credence-proxy?

**Finding.** The proxy targets the OAI-compatible endpoint (`/v1/chat/completions`), not the native Messages API (`/v1/messages`). Evidence: `PROVIDER_ENDPOINTS` dict at `provider.py:45-48` hardcodes `https://api.anthropic.com/v1/chat/completions` for the `"anthropic"` provider. The proxy sends `Authorization: Bearer <key>` and `Content-Type: application/json` headers — no `x-api-key`, no `anthropic-version`. The only place the native API appears is the quality judge (`server.py:204-221`), which is a separate code path.

**Consequence.** Anthropic's `cache_control: {"type": "ephemeral"}` markers are a content-block-level field in the native Messages API schema. They are not part of the OpenAI Chat Completions schema and are inapplicable on the OAI-compatible endpoint.

The applicable caching mechanism is **automatic prefix-based caching**: the provider caches identical message prefixes of sufficient length (1024 tokens for Claude models), per-model, with a 5-minute TTL. No client-side markers are needed or consumed. The audit's job is to confirm the proxy preserves the conditions for this automatic mechanism — primarily, that the proxy doesn't inject nondeterministic content into the message prefix.

**Scope statement.** The audit checks whether the proxy preserves automatic prefix caching conditions on Anthropic's OAI-compatible endpoint. The audit's findings are not generalisable to the native Messages API without separate work; if the proxy migrates to the native API in the future (see §5.6), the cache-discipline question must be re-audited against the native API's user-directed caching mechanism.

### 5.2 — Failure modes (reframed as mutation-determinism checks)

**The question.** Which failure modes does the audit check, given the endpoint reframing?

**(a) Prefix identity preservation.** Does the proxy's `json.loads` -> 3-field mutation -> `json.dumps` roundtrip produce identical output for identical input routed to the same model?

Inspection finding: yes. The entire request mutation is three lines at `provider.py:162-165`:

```python
data["model"] = model_name
data["stream"] = True
data["stream_options"] = {"include_usage": True}
```

`json.loads` returns a `dict` preserving the JSON key order as parsed (CPython 3.7+). The three mutations overwrite or add fixed-value keys: `model` is deterministic per routing decision; `stream` and `stream_options` are constants. `json.dumps(data)` re-serialises in insertion order. No other keys are added, removed, or modified. No conditional branches in `forward_streaming` inject variance into the request body.

Audit activity: code-review verification of the mutation path (substantially complete from this inspection), **plus** empirical confirmation via a two-request diff test (see §5.4). The diff test converts the verdict from "we read the code and believe it" to "we read the code and verified the runtime behaviour matches the read".

**(b) Nondeterministic cache-busting content.** Does the proxy inject timestamps, request IDs, UUIDs, or other varying content into the request body or headers?

Inspection finding: no. Headers sent by the proxy (`provider.py:181-183`):

```python
headers={
    endpoint["auth_header"]: auth_value,
    "Content-Type": "application/json",
}
```

`auth_value` is `"Bearer " + api_key` — constant per API key. No `User-Agent`, no `X-Request-ID`, no tracing metadata, no timestamps. The body mutations are the three deterministic fields above.

Audit activity: same code-review pass and diff test as mode (a).

**(c) Routing-induced cache fragmentation.** When Bayesian routing distributes traffic across N models (3 Anthropic: Haiku, Sonnet, Opus; 2 OpenAI: GPT-4o-mini, GPT-4o), each model maintains its own prefix cache. A conversation prefix cached on `claude-sonnet-4-6` provides zero benefit if the next turn routes to `claude-haiku-4-5`.

This is **not a discipline failure** — it is an inherent property of multi-model routing and is the proxy doing its job (selecting the EU-maximising model per request). But it has cost implications: cache-warming benefit is partially split across models, and the split's magnitude depends on the routing distribution's stability.

Audit activity: run 3-5 hand-constructed multi-turn workloads through the proxy, collect `_request_log` (server.py:299), compute:
- Per-model routing distribution across the workload.
- Same-model streak length distribution (what fraction of consecutive requests from the same conversation went to the same model?).
- Implied cache-fragmentation percentage.

### 5.3 — Verdict criterion

**The question.** What does the audit report, and how is it measured?

**Verdict.** Four-option:

- **"Cache discipline clean"** — modes (a) and (b) pass by code review and empirical diff test; mode (c) measured and acknowledged as a property of multi-model routing.
- **"Cache discipline broken"** — mode (a) or (b) reveals a prefix-busting mutation.
- **"Cache discipline broken with repair path"** — busting mutation identified, repair named and scoped.
- **"Cache discipline broken with architectural implications"** — requires a design conversation rather than a quick fix (e.g., automatic caching doesn't work on the OAI-compatible endpoint at all; or the proxy's architecture makes deterministic serialisation structurally fragile).

**Reference baseline** for modes (a)/(b): the same request body sent directly to the OAI-compatible endpoint via `httpx.post()` without proxy interposition. The proxy's outbound body should be byte-identical modulo the three expected mutations (`model`, `stream`, `stream_options`).

**Named outputs.** The report produces two artefacts:
1. The verdict (one of four options above).
2. The cache-observability-gap finding (see §1), which Move 1's methodology design absorbs as a constraint.

### 5.4 — Audit mechanism

**The question.** How does the audit observe the proxy's behaviour?

**Modes (a) and (b): code review plus required diff test.**

Code review: examine `forward_streaming` at `provider.py:131-271`, enumerate all conditional branches, confirm no prefix-busting variance. Substantially complete from this inspection.

Diff test (required, not optional): monkeypatch `PROVIDER_ENDPOINTS` in a pytest fixture to redirect Anthropic traffic to a local recording mock. Send two identical requests through the proxy. Diff the outbound JSON bodies recorded by the mock. Confirm they are byte-identical.

Implementation: `PROVIDER_ENDPOINTS` is a module-level dict read at request time (`forward_streaming` calls `PROVIDER_ENDPOINTS.get(spec.provider)` per invocation), so standard pytest `monkeypatch.setitem` works without proxy code changes. The recording mock is a minimal FastAPI app that captures the received request body and returns a valid streaming SSE response. Lives in the test directory alongside other test infrastructure.

**Mode (c): routing-distribution measurement.**

Run 3-5 hand-constructed multi-turn workloads through the proxy. Each workload: a system prompt, tool definitions, and 3-5 turns of user/assistant messages — shaped like real coding conversations, not synthetic categorical queries. Collect routing decisions from `_request_log`. Compute per-model distribution and same-model streak statistics.

The stopgap workloads are not statistically representative; they are representative enough to measure routing distribution. Move 1 designs the production workload.

### 5.5 — Reporting format

Report at `docs/posture-5/move-0-audit-report.md`. Structure:

- One-paragraph framing: audit purpose, endpoint discovery, scope.
- **§Mode (a)**: code-review finding plus diff-test result.
- **§Mode (b)**: code-review finding (same pass as mode a).
- **§Mode (c)**: per-model routing distribution table over the stopgap workloads; same-model streak length distribution; implied cache-fragmentation percentage.
- **§Cache observability gap**: the proxy discards cache-related usage metadata. `provider.py:218-220` parses only `prompt_tokens`/`input_tokens` and `completion_tokens`/`output_tokens`. Neither `prompt_tokens_details.cached_tokens` (OpenAI response format) nor `cache_creation_input_tokens`/`cache_read_input_tokens` (Anthropic response format) are extracted. Consequence: even with cache discipline verified clean, empirical measurement of cache cost-savings requires either parsing these fields or attributing cost from request-side estimation. This is a named constraint for Move 1.
- **§Verdict**: one of the four options from §5.3.
- **§Move 1 input**: methodology-relevant facts the benchmark design absorbs — the verdict, the cache-observability gap, the mode (c) routing distribution, and any other findings the audit surfaces.

### 5.6 — Native Messages API: documented option-on-the-table

**The question.** Should the proxy migrate from the OAI-compatible endpoint to Anthropic's native Messages API?

**Not answered here.** This section describes the capability gap; it does not argue for the migration.

The native Messages API (`/v1/messages`) exposes mechanisms the OAI-compatible endpoint does not:

- **User-directed caching**: `cache_control: {"type": "ephemeral"}` markers on content blocks, giving the client explicit control over what is cached and for how long.
- **Cache observability**: `cache_creation_input_tokens` and `cache_read_input_tokens` in the usage response, enabling empirical measurement of cache hit rates and cost savings.
- **Richer streaming events**: `message_start`, `content_block_start`, `content_block_delta` events with structured metadata.

Whether credence-proxy benefits materially from these capabilities depends on Move 1's methodology requirements and Move 2's benchmark findings. The migration is recorded here as an option-on-the-table, not a recommendation-to-act.

This is explicitly **not** framed as a separate Posture 5 move. If later moves (Move 1's methodology, Move 2's benchmarks) surface the native API's capabilities as load-bearing — e.g., if the benchmark methodology requires empirical cache-hit measurement that the OAI-compatible endpoint cannot provide — a future move addresses the migration. If they don't, the OAI-compatible endpoint is fine and this section closes as a documented option-not-taken.

## 6. Risks

**Risk 1: JSON serialisation determinism.** CPython 3.7+ preserves `dict` insertion order, but the Python language specification does not guarantee it across implementations. The proxy's deployment target is CPython (standard Docker `python:3.11` base image); the determinism assumption holds. If the deployment target ever shifts (e.g., to PyPy, or to a Rust reimplementation for latency reasons), the assumption needs revisiting. Mitigation: state the CPython dependency in the report; the assumption is captured for future-shift conversations.

**Risk 2: Stopgap workload representativeness.** Mode (c) measurement uses 3-5 hand-constructed multi-turn workloads. These are not statistically representative of production traffic; they are representative enough to measure routing distribution shape. Move 1 designs the production workload; mode (c) measurement with production-representative workloads is a Move 2 activity.

**Risk 3: Automatic caching may not apply to the OAI-compatible endpoint.** Anthropic's documentation primarily addresses the native Messages API. The OAI-compatible endpoint's automatic-caching behaviour may be: (a) the same as the native endpoint's automatic caching, (b) different but documented separately, or (c) undocumented and discoverable only by observation.

The audit's code-review phase checks Anthropic's current documentation for explicit statements about the OAI-compatible endpoint's caching behaviour. If the documentation is silent or ambiguous, the audit reports the ambiguity rather than assuming. A finding of "automatic caching's application to this endpoint is undocumented; observed behaviour suggests X but the assumption is provisional" is honest provenance for Move 1. If the documentation explicitly confirms automatic caching does *not* apply to the OAI-compatible endpoint, the verdict escalates to "cache discipline broken with architectural implications" and triggers a design conversation about the native API migration (§5.6 becomes load-bearing).

**Risk 4: Cache observability gap.** The proxy discards cache metadata from provider responses. Without empirical cache-hit data, mode (c) measurement is theoretical — routing distribution implies fragmentation but does not measure actual cache misses. This is a named audit output (see §1), not a blocker. Move 1's methodology design absorbs it as a constraint.

## 7. Test plan (audit execution shape)

1. **Documentation check** (Risk 3 gate): verify Anthropic's current documentation regarding the OAI-compatible endpoint's caching behaviour, using live web search rather than training-cutoff knowledge. If documentation explicitly confirms automatic caching does *not* apply to the OAI-compatible endpoint, the verdict escalates to "architectural implications" immediately and steps 2-4 are moot.

2. **Code-review verification** of modes (a) and (b): examine `forward_streaming` at `provider.py:131-271`, enumerate all conditional branches, confirm no prefix-busting variance. Document the exact mutation path and its determinism properties.

3. **Required diff test** for modes (a) and (b): monkeypatch `PROVIDER_ENDPOINTS` to point Anthropic traffic at a local recording mock. Send two identical OpenAI-format requests through the proxy (same system prompt, same messages, same tool definitions). Diff the outbound JSON bodies recorded by the mock. Confirm byte-identical. If not byte-identical, diff identifies the divergence — this is a mode (a) or (b) failure.

4. **Mode (c) routing-distribution measurement**: run 3-5 hand-constructed multi-turn workloads through the proxy. Each workload has a system prompt, tool definitions, and 3-5 turns. Collect `_request_log` entries. Compute per-model routing distribution and same-model streak length statistics.

5. **Write report** at `docs/posture-5/move-0-audit-report.md` per §5.5 structure.

## Closure

After the report lands:

- Move 1's design conversation absorbs the verdict and the cache-observability-gap finding as methodology inputs.
- The cache-observability gap (parsing `cached_tokens` / `cache_creation_input_tokens` from provider responses) is a candidate for early Move 1 infrastructure or a standalone item. It is not blocked on the native API migration — the OAI-compatible endpoint may return cache metadata in the OpenAI response format.
- §5.6 (native Messages API option) is on-the-table for later moves to pick up or leave, depending on whether Move 1/2 surface the native API's capabilities as load-bearing.
- Posture 5's master plan is provisional; refinement after Move 0's findings is appropriate.
