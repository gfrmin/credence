# Move 0 — Cache-discipline audit report

Posture 5 Move 0 audits whether credence-proxy preserves the conditions
under which Anthropic's prompt caching works. The audit's finding gates
Move 1's benchmark methodology design: a clean verdict means Move 1
proceeds against a known-clean substrate; an escalated verdict means a
design conversation before Move 1 begins. The design doc's §7 specifies
five audit steps; the documentation check (step 1) is sequenced first as
an early-escalation gate, because if prompt caching does not apply to the
proxy's endpoint, the remaining steps are moot.

## Documentation check (Risk 3 gate)

**Finding: prompt caching is explicitly not supported on the OAI-compatible
endpoint.**

Anthropic's [OpenAI SDK compatibility documentation](https://platform.claude.com/docs/en/api/openai-sdk)
states, under "Important OpenAI compatibility limitations > API behavior":

> Prompt caching is not supported, but it is supported in
> [the Anthropic SDK](/docs/en/api/client-sdks)

The same page's response-fields table confirms that `usage.prompt_tokens_details`
and `usage.completion_tokens_details` are "Always empty" on the OAI-compatible
endpoint — no cache-related usage metadata is returned regardless of request
shape.

The [prompt caching documentation](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching)
describes two caching modes (automatic and explicit), both available on "the
Claude API and Azure AI Foundry (preview)". The OAI-compatible endpoint is not
mentioned. The native Claude API is the only Anthropic-hosted path that
supports prompt caching.

**Pages consulted:**
- `https://platform.claude.com/docs/en/api/openai-sdk` — OpenAI SDK compatibility
  reference. Contains the explicit "Prompt caching is not supported" statement
  and the response-fields table.
- `https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching` —
  Prompt caching feature documentation. Describes automatic and explicit caching
  on the native Claude API; no mention of the OAI-compatible endpoint.

**Escalation.** Per the design doc's §6 Risk 3: documentation explicitly confirms
prompt caching does not apply to the OAI-compatible endpoint. The verdict
escalates to "Cache discipline broken with architectural implications". Audit
steps 2–4 (code-review verification, diff test, routing-distribution
measurement) are moot — there is no caching mechanism to preserve or fragment,
and no cache-related usage metadata to observe. §5.6 (native Messages API
migration) becomes load-bearing.

## Mode (a): prefix identity preservation

**Not applicable.** The OAI-compatible endpoint does not implement prompt caching.
The proxy's request-mutation determinism (3-field mutation at `provider.py:162-165`:
`model`, `stream`, `stream_options`) is irrelevant to caching because no caching
occurs on this endpoint regardless of request shape.

For the record: the mutation *is* deterministic (code inspection from the design
phase confirmed this), but determinism is a necessary condition for prefix
caching, not a sufficient one — and the endpoint doesn't support caching at all.

## Mode (b): nondeterministic cache-busting content

**Not applicable.** Same reason as mode (a). The proxy injects no nondeterministic
content (no timestamps, no request IDs, no UUIDs — headers are `Authorization`
and `Content-Type` only), but this property is moot when the endpoint doesn't
cache.

## Mode (c): routing-induced cache fragmentation

**Not applicable.** Cache fragmentation across models requires a caching mechanism
to fragment. Since the OAI-compatible endpoint does not cache, routing
distribution across models has no cache-related cost implication on the current
endpoint.

If the proxy migrates to the native Messages API (where prompt caching is
supported), routing-induced cache fragmentation becomes a live concern and
should be measured at that time. The design doc's mode (c) measurement
methodology (same-model streak length distribution over multi-turn workloads)
remains the right approach for that future measurement.

## Cache observability gap

The proxy discards cache-related usage metadata from provider responses.
`provider.py:218-220` parses only `prompt_tokens`/`input_tokens` and
`completion_tokens`/`output_tokens`. Neither `prompt_tokens_details.cached_tokens`
(OpenAI response format) nor `cache_creation_input_tokens`/`cache_read_input_tokens`
(Anthropic native format) are extracted.

On the current OAI-compatible endpoint, this is moot — the response-fields
table confirms `usage.prompt_tokens_details` is "Always empty". But if the
proxy migrates to the native Messages API, cache-related usage fields
(`cache_creation_input_tokens`, `cache_read_input_tokens`) become available and
the proxy will need to parse them to measure cache cost savings. This is a
constraint for Move 1's methodology design regardless of the migration path:
empirical cache-cost measurement requires either (a) migrating to the native
API and parsing cache fields, or (b) estimating cache savings from request-side
data without provider confirmation.

## Verdict

**Cache discipline broken with architectural implications.**

The proxy routes Anthropic traffic through the OAI-compatible endpoint
(`https://api.anthropic.com/v1/chat/completions`). Anthropic's documentation
explicitly states that prompt caching is not supported on this endpoint. No
caching — neither automatic nor user-directed — occurs on Anthropic-routed
traffic through the proxy.

This is not a bug in the proxy's request translation. The proxy's mutation path
is deterministic and well-behaved; the issue is the endpoint choice. The only
path to Anthropic prompt caching is migrating to the native Messages API
(`/v1/messages`), which supports both automatic caching (`cache_control` at the
request level) and explicit cache breakpoints (`cache_control` on individual
content blocks).

The design doc's §5.6 (native Messages API: documented option-on-the-table)
becomes load-bearing. The migration is no longer an option to evaluate after
Move 1/2 evidence; it is a prerequisite for the proxy to benefit from
Anthropic's prompt caching at all. Whether the migration is urgent depends on
how much of the proxy's cost-savings value proposition rests on cache cost
reduction vs. model-tier routing (selecting cheaper models when quality permits).

## Move 1 input

Move 1's benchmark methodology design absorbs the following findings:

1. **No Anthropic prompt caching on the current endpoint.** Any cost-savings
   claim attributable to prompt caching is an artefact of the endpoint choice,
   not a proxy feature. The proxy's cost savings come entirely from model-tier
   routing (selecting Haiku over Sonnet/Opus when EU-optimal).

2. **Native API migration is a prerequisite for cache savings.** If Move 1's
   benchmark methodology wants to measure cache-related cost savings, the proxy
   must first migrate to the native Messages API. This is a sequencing
   constraint: either the migration happens before or as part of Move 1, or
   Move 1's methodology explicitly excludes cache savings from its measurement
   scope.

3. **Cache observability requires native API fields.** Even after migration,
   the proxy must parse `cache_creation_input_tokens` and
   `cache_read_input_tokens` from the native API's usage response to measure
   cache cost savings empirically.

4. **Mode (c) measurement is deferred.** Routing-induced cache fragmentation
   becomes measurable only after migration to the native API. The design doc's
   measurement methodology (same-model streak length distribution) remains
   applicable; execution is deferred until caching is live.

5. **The proxy's request translation is well-behaved.** Despite the endpoint
   issue, the proxy's mutation path is deterministic and injects no
   nondeterministic content. If the proxy migrates to the native API, the
   request-translation layer is a sound foundation — the migration's work is in
   the API surface (message format, streaming protocol, header shape), not in
   fixing translation bugs.
