# answer-brain — Move 3 design (the life-agent capability bridge: retrieve / extract / probe over HTTP)

Follows `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Strategy is settled in `master-plan.md`; Move 1
(`move-1-design.md`) landed the native brain with parity; Move 2 (`move-2-design.md`) landed the
daemon's `decide` wire + the Python parity-boundary mapping, **deferring the bridge *service* to this
move** (move-2 §2 "Deferred to Move 3"). This move stands up that service — the second of the two
backends the pi-mono body drives — with no pi-mono and **no model in any loop**. The body + app +
end-to-end gate are the named successor, **Move 4**.

## 0. Final-state alignment

`master-plan.md` §"Architecture" is three processes: a pi-mono answering agent (TS) that proposes
`retrieve`/`extract`/`probe`/`answer`; the **answer-brain daemon** (Julia) that holds the candidate
posterior and governs answer/gather/ask/abstain by EU (Move 2's `POST /decide`); and **life-agent
capabilities** (Python) the agent's tools call. Move 2 gave the daemon its wire and froze the Python
parity mapping (`bridge/observations.py`). This move wraps life-agent's existing reads —
`route`/`retrieve`/`extract`/`probe_*` — in a **stateless HTTP service** so the body (Move 4) has a
second live, independently-tested backend whose `/extract` output is *exactly* the daemon `/decide`
input. Two backends, uniform HTTP; the body is the only process that talks to both.

Transient state, named so it is not drift:
- **The pi-mono body + app + e2e gate are deferred to Move 4.** They need a model in the loop, so
  they cannot be hermetically tested; and the model-choice-under-PII question (the eval runs over the
  owner's real corpus — §5 names it) is theirs to resolve, not this move's. What freezes here is the
  bridge's tool schema, so Move 4 is additive (write the body against a frozen seam), never a
  retraction.
- **The bridge exposes capabilities, not policy.** `gather.py`'s Stage-0 *orchestration* — the
  `_era_split` recency-decoupling (`core/gather.py:63`), the owner-scoped gather guard
  (`gather.py:141`), the `_top_candidates` fan-out (`gather.py:86`) — is the **throwaway driver**
  (`gather.py:39` "ports to the Julia answer-brain"). It does **not** move into the bridge; it
  becomes the brain's VOI job in Move 4 (recency/subject become *probes the brain selects when they
  discriminate*, `master-plan.md` §Why). The bridge's `/extract` takes `time_indexed` + `covariates`
  as **inputs**; it never decides them. A bridge that re-encoded the policy would be `gather.py`
  over HTTP — the wrong cut.
- **life-agent keeps driving the skin in production** (`core/gather.py`, `scripts/ask.py`); this move
  adds a *new process* beside the live ask path, it does not reroute it. The bridge **reuses** the
  reads (`route_question`, `_retrieve_set`, `observe_hits`, `probes.*`, `to_abstract_observations`);
  it rebuilds nothing, and touches no `pkm`.

## 1. Purpose

Stand up and test the **life-agent capability bridge** (`src/life_agent/bridge/server.py`): a
long-lived, **stateless** JSON-over-HTTP service (stdlib `http.server`, no new dependency) that
exposes life-agent's body-side reads as discrete endpoints, each a thin wrapper of an existing,
tested function. The brain's posterior is **not** here — the bridge gathers and shapes evidence;
the daemon (Move 2) decides. The bridge owns the **body side** of the parity boundary
(`route`/`retrieve`/`extract`/`probe`/`abstract`), so its `/extract` returns the candidate display
strings + the abstract integer observations the daemon consumes verbatim (`to_abstract_observations`,
the single-source mapping Move 2 froze).

**Why a service at all:** pi-mono ships **no MCP client** (verified `pi-mono@29c1504c`; move-2 §5 Q1),
its tool surface is `pi.registerTool(ToolDefinition)` with a TS `execute`. A per-call `uv run`
cold-start (seconds of DuckDB + model init) is rejected for an agent loop, so the reads live behind a
warm long-lived process the body `fetch`es — uniform with the already-HTTP daemon.

**Proof obligation:** (a) the HTTP contract — each endpoint's request/response shape, dispatch,
`400` on malformed input, `/ready` — is pinned hermetically (fake corpus / monkeypatched seams, no
model); (b) `/extract` returns precisely `to_abstract_observations`'s output (the brain stays
string-blind), proven by routing the endpoint and the function through the **same** assertion; (c)
statelessness — an interleaved request cannot perturb a repeat; (d) one **system** check (marked,
live local corpus + local Ollama, fully on-machine) exercises a real `retrieve → extract → probe →
extract` round-trip end-to-end.

## 2. Files touched

Created — `life-agent` (`src/life_agent/bridge/`):
- `server.py` — the stateless HTTP service. A warm boot opens one **read-only** DuckDB handle + the
  extraction client (Ollama) once (as `core/lookup.py` does per ask-session — move-2 §6), then serves:

  | endpoint | wraps (file:line) | request → response |
  |---|---|---|
  | `POST /route` | `lookup.route_question` | `{question}` → `{construct, time_indexed} ⎮ null` (null ⇒ narrative path, the brain's non-lookup case) |
  | `POST /retrieve` | `retrieval.retrieve_set` + `retrieval.build_query` (`core/retrieval.py`) | `{question, terms?, k}` → `{hits:[{artifact_cache_key, chunk_text, score, origin}]}` |
  | `POST /extract` | `lookup.observe_hits` (`lookup.py:447`) + `candidates_from` (`lookup.py:520`) + `to_abstract_observations` | `{question, hits, covariates?, time_indexed?, today?}` → `{candidates:[str], observations:[{reports,group,authority,subject_factor,time_factor}], rho, indeterminate}` |
  | `POST /probe/recency` | `probes.probe_recency` | `{hit_keys}` → `{doc_date:{key: iso⎮null}}` |
  | `POST /probe/subject` | `probes.probe_subject` (profile loaded server-side) | `{hit_keys}` → `{subject_state:{key: state}}` |
  | `POST /probe/authority` | `probes.probe_authority` | `{hits}` → `{authority:{key:[class, value]}}` |
  | `POST /probe/corroborate` | `probes.probe_corroborate` | `{question, leader_value, k?, exclude_keys?}` → `{hits:[…]}` |
  | `GET /utility` | the utility posterior's `u_bar()` (`utility.py:269`) | → `{u_bar:{u_correct,u_wrong,u_hedged,u_abstain,oracle_p,lambda_int,kappa_att}}` |
  | `GET /ready` | — | → `200 "ok"` (transport liveness, no reasoning) |

  Host/port from `LIFE_AGENT_BRIDGE_HOST` (`127.0.0.1`) / `LIFE_AGENT_BRIDGE_PORT` (`8798`, adjacent
  to the daemon's `8799`); localhost-bound so the owner's corpus never leaves the machine. **Holds no
  per-question state**: every endpoint is a pure function of (corpus, request) — the body holds the
  hit set + accumulated covariates and resends them, mirroring `/decide`'s discipline (move-2 §5
  Resolution). Adapts credence-pi's HTTP plumbing (`apps/credence-pi/daemon/server.jl`'s
  request-parse / JSON-response / `/ready` shape), **shedding** its SSE `/signals` stream + sensor
  queue (the answer path is synchronous request/response — move-2 §5 Q2).
- `bridge/__init__.py` — replace the "Move-3 service deferral" docstring with the service's contract.
- `bin/answer-bridge` — debug entrypoint (`python -m life_agent.bridge.server`), mirroring
  `bin/ask-live`; prints the bound address and the endpoint list.

**As-built refinements (Move 3 — author to confirm).** Three small, behaviour-preserving
departures from the literal plan, forced by an import boundary the plan hadn't surfaced — the
retrieval seam (`_retrieve_set`) lived in `scripts/ask.py`, and `src/` deliberately does not
import from `scripts/` (the boundary `core/matching.py:5` also keeps). All three are tested and
the full hermetic suite (916) stays green:
1. **New `src/life_agent/core/retrieval.py`** — `build_query` + `retrieve_set` (verbatim from
   `ask.py`). `ask.py` re-exports both (its callers and tests are unchanged), so there is still
   exactly ONE retrieval implementation (the §6 "no second read" obligation). This is the only
   edit to `ask.py` — its retrieval/expansion *math* is untouched; the functions just moved into
   the package so the bridge can reuse them.
2. **`/retrieve` takes `terms` as an INPUT** (the body supplies expansion), rather than the
   bridge calling the cloud `_expand_terms`. This is the same cut §0/§3 make for `/extract`'s
   covariates — expansion is a reformulation *policy* (the driver's job), and keeping it out
   holds the bridge model-free + cloud-free (§8), strengthening the localhost/PII posture.
   `_expand_terms` stays script-side; Move 4's body owns when to expand.
3. **Boot reuses `tasks/read.pkm_root()`** for root resolution (not `ask._pkm_root`), again to
   avoid the `src↛scripts` import; the bridge opens its own read-only catalogue handle with FTS.
   `BridgeServer` is single-threaded (the body drives one sequential loop) — concurrency, and
   with it duckdb thread-safety, is the Move-4 measurement §5 Q2 already defers.

Created — tests (`life-agent`):
- `tests/test_bridge_server.py` — **hermetic**: each endpoint's request/response shape over a fake
  corpus (seams monkeypatched), dispatch + `400`-on-malformed + `/ready`; the **single-source**
  assertion (`/extract` ≡ `to_abstract_observations` on the same observations — the brain-blindness
  proof); statelessness (an interleaved `/extract` does not perturb a repeat). No model, no network
  beyond loopback.
- `tests/test_bridge_server_live.py` — **one system check** (`-m system`): boot the service on the
  live corpus, `retrieve → extract → probe/corroborate → extract`, assert the union grows and the
  abstract observations are well-formed. Local only; opt-in (the default `-m 'not llm and not
  system'` excludes it).

No `src/Credence` change; no `apps/answer-brain/` change (the daemon is Move 2's, frozen); no
`src/pkm` change; no edit to `core/lookup.py` / `core/probes.py` / `scripts/ask.py` math (the bridge
*calls* them).

Deferred to Move 4 (named): `apps/answer-brain/extension/*` (the pi-mono TS body — the `tool_call`
governor blocking `answer` on a synchronous `/decide`, the `tool_result` observer accumulating the
evidence vector, the bridge-tool registrations); `apps/answer-brain/bdsl/*` for *answering*
(effectors answer/ask/abstain/gather; the candidate-set features); the minimal pi-mono answering app;
the end-to-end eval run + gate, and with it the **model-choice-under-PII** resolution (§5 Q4).

## 3. Behaviour preserved

The bridge introduces a process; it must change no result of any existing path.
- **The reads are reused, not reimplemented.** Each endpoint calls the named function; `ask`,
  `gather`, and the gate keep running unchanged. A divergence between a bridge response and a direct
  call is a serialisation bug, never a second implementation (there is one).
- **The parity boundary holds.** `/extract` returns only integers + floats (`reports`, `group`,
  `authority`, `subject_factor`, `time_factor`) + the candidate strings — never an `Observation`
  internal — via the same `to_abstract_observations` the daemon's fixtures encode. The daemon stays
  string-blind; the bridge owns all candidate canon + ancestry-group keying, Python-side.
- **`/extract` output is `/decide` input.** `{candidates, observations, rho}` from `/extract` + the
  body's `u_bar` (from `/utility`) compose directly into Move 2's `POST /decide
  {observations, candidates, rho, u_bar}` — the composition `dump_parity_fixtures.py:129‑142` already
  performs (`to_abstract_observations` → `lookup_posterior` over the same `rho`/`u_bar`).
- **Per-question isolation by construction.** Stateless ⇒ two questions interleaved in one process
  cannot perturb each other; the bridge holds no shared mutable belief (the §6 cross-talk risk move-2
  voided, voided here the same way).
- **PII stays server-side.** The owner profile (`owner.load_profile()`, `ask.py:633`) and the
  utility posterior (`$LIFE_AGENT_KB/utility/model.yaml`) are read **inside** the bridge; `/probe/
  subject` and `/utility` take no profile/utility over the wire. The body is identity-free; the
  service is localhost-bound.

## 4. Worked end-to-end example

The canonical mobile-number case (`master-plan.md` §Why) as the body would drive the bridge — synthetic
values, no real datum (this is a public repo):

```
POST /retrieve  { "question": "what's my mobile number?", "k": 20 }
→ { "hits": [ {"artifact_cache_key":"a_email_2015", "chunk_text":"…+852 5xxx…", "score":0.71, "origin":"mail/…"},
              {"artifact_cache_key":"a_ni_form",    "chunk_text":"…05x xxx…",   "score":0.42, "origin":"docs/…"}, … ] }

POST /extract   { "question":"what's my mobile number?", "hits":[…above…] }      # baseline, no covariates
→ { "candidates":["<stale-hk>","<current-il>"],
    "observations":[ {"reports":0,"group":0,"authority":0.90,"subject_factor":1.0,"time_factor":1.0}, … ],
    "rho":0.5, "indeterminate":[] }
```

The body POSTs `{candidates, observations, rho}` + `u_bar` (from `GET /utility`) to the daemon's
`POST /decide`; the corroboration-biased baseline returns `gather` (the stale value leads but below
the `u_wrong` bar). The body steers the LLM to gather, which calls:

```
POST /probe/corroborate { "question":"what's my mobile number?", "leader_value":"<current-il>",
                          "exclude_keys":["a_email_2015","a_ni_form"] }
→ { "hits":[ {"artifact_cache_key":"a_payslip_2025", …}, … ] }      # high-authority current records surface
POST /probe/recency     { "hit_keys":["a_email_2015","a_ni_form","a_payslip_2025"] }
→ { "doc_date":{"a_email_2015":"2015-06-02","a_ni_form":null,"a_payslip_2025":"2025-04-30"} }
POST /extract  { "question":"…", "hits":[union], "covariates":{"doc_date":{…}}, "time_indexed":true }
→ { "candidates":["<current-il>","<stale-hk>"], "observations":[ … time_factor decays the 2015 obs … ], "rho":0.5 }
```

A second `/decide` on the re-weighted abstract observations now reports `<current-il>`. **Every datum
the bridge returns is a read of the current corpus**; *which* probe to call and *when* to stop is the
brain's VOI decision (Move 4) — the bridge never decides, exactly as `/decide` never retrieves.

## 5. Open design questions

**Q1 — endpoint granularity: one per capability, or one generic `POST /tool {name, args}`.** A
generic dispatch is fewer handlers; per-capability endpoints are a 1:1 with the body's
`pi.registerTool` definitions, individually `curl`-able, and let each carry its own request schema +
`400`. **I recommend per-capability** (the table above): the body registers one tool per endpoint, so
the surfaces match, and a typed-per-endpoint contract is easier to pin and to evolve. Push back if you
want a single dispatch endpoint for a thinner server.

**Q2 — statelessness + state threading (load-bearing).** The bridge holds **no** per-question state;
the body holds the growing hit set + the accumulated covariates and resends them to `/extract` each
refinement — the same "the body holds the evidence and resends" discipline `/decide` adopted (move-2
§5 Resolution). The alternative is a session-keyed bridge that caches the hit set server-side (fewer
bytes per call) — but it reintroduces the mutable per-question state move-2 deliberately voided, and
its cross-talk failure mode. **I recommend stateless**, uniform with the daemon: both backends are
pure functions of their request, the body is the sole stateful coordinator. Push back if the wire
volume (resending hits) proves a real cost — but that is a Move-4 measurement, not a Move-3 guess.

**Q3 — capability vs policy (what the bridge *is*).** The bridge exposes raw capabilities;
`gather.py`'s policy (`_era_split`, the owner-scoped guard, the `_top_candidates` fan-out) does **not**
move into it — `/extract` takes `time_indexed` + `covariates` as inputs, it does not compute them.
That policy is the **brain's** VOI job in Move 4 (the redesign's whole thesis: recency/subject become
probes selected when they discriminate, not hard-coded covariates — `master-plan.md` §Why). **I
recommend the capability-only cut.** The cost is honest: until Move 4's brain reproduces the policy by
VOI, no single bridge call answers the mobile case end-to-end — the *driver* (Stage-0 `gather.py`, or
Move 4's brain) supplies the policy. This is the right boundary (the daemon owns decision; the bridge
owns evidence), but it is the one most worth your explicit assent, because it declines to make the
bridge a one-call answerer. Push back if you want the bridge to also expose a `gather.py`-equivalent
*policy* endpoint (a convenience that would duplicate, and risk drifting from, the brain).

**Q4 — the model-choice-under-PII question (flagged here, resolved in Move 4).** The e2e gate runs the
agent over the owner's **real** corpus; retrieved chunks carry PII (`$LIFE_AGENT_KB`). A cloud model
driving the loop would send that PII off-machine — disallowed. So Move 4's driver is constrained to
**local Ollama** (PII-safe) *or* a **deterministic scripted driver** (no model sees the corpus; tests
the brain's govern+steer in isolation, removing the LLM-tool-use confound). This move does not decide
it — it has no model in any loop — but I name it now so Move 4's scope is not a surprise. (My lean,
for Move 4: a scripted driver *is* the gate — confound-free, PII-safe, deterministic — with a local
Ollama run as a separate qualitative demonstration. Not decided here.)

**Resolution (author-approved, in conversation).** Q1 → **per-capability endpoints** (one HTTP
endpoint per read, 1:1 with the body's `pi.registerTool` definitions). Q2 → **stateless** (the body
holds the evidence and resends it; every endpoint is a pure function of (corpus, request), uniform
with `/decide`). Q3 → **capability-only** (the bridge exposes raw `retrieve`/`extract`/`probe`;
`gather.py`'s policy stays out and becomes the brain's VOI job in Move 4 — `/extract` takes
`time_indexed` + `covariates` as inputs, never computing them). All three Move-3 forks took the
recommended option; §2 is scoped accordingly. Q4 (model-choice-under-PII) is **named, not resolved**:
it is Move 4's, and is constrained to local Ollama or a deterministic scripted driver because the
gate runs over the owner's real corpus.

## 6. Risk + mitigation

- **A bridge endpoint silently re-implements a read** (drifts from `ask`/`gather`). *Mitigation:* each
  endpoint is a one-line call of the named function; the hermetic test asserts the bridge response
  equals the direct call. No logic in the handler beyond parse/serialise.
- **`/extract` diverges from the daemon's expected input** (a renamed field, a float rounded).
  *Caught by:* the single-source assertion — `/extract`'s `observations` is literally
  `to_abstract_observations`'s output (the move-2 fixtures' oracle), so the daemon receives the shape
  its parity tests pin.
- **Per-question cross-talk.** *Voided by design (§5 Q2):* stateless — each call rebuilds from its
  request; the isolation test pins it.
- **Python cold-start / per-call model spin-up** making the loop slow. *Mitigation:* warm long-lived
  process — read-only DuckDB handle + extraction client initialised once at boot (as `core/lookup.py`
  already does); per call is a read, not an init.
- **PII egress.** *Mitigation:* localhost-bound (`127.0.0.1`); profile + utility read server-side,
  never over the wire (§3); no endpoint returns more than the cited reads already expose to `ask`.
- **pkm-freeze violation.** *Mitigation:* the bridge is `life_agent`, calling existing reads; `pkm`
  is untouched (the move-2 Q1(a) reason, held).

## 7. Verification cadence

End-of-move (all green): `uv run pytest tests/test_bridge_server.py` (hermetic — contract, dispatch,
`400`, `/ready`, the single-source brain-blindness assertion, statelessness); `uv run pytest
tests/test_bridge.py` (Move 2's 7 mapping cases — the bridge consumes that frozen mapping, regression
it); `uv run pytest -m system tests/test_bridge_server_live.py` (the one live local round-trip, run
and reported explicitly — it touches the corpus + Ollama, so it is opt-in and its result is stated,
not assumed); `ruff` + `mypy --strict` clean on touched `life_agent/` src. No model is in the
hermetic loop, so those runs are deterministic and reported with exact counts; the system check's
non-determinism is disclosed where it is cited.

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** Not applicable to the bridge — it computes **no**
   posterior and **no** decision. It returns declared reads: hits (retrieval), the candidate strings +
   abstract-obs records (`to_abstract_observations`, declared data), the projected covariates (the
   probes' projections), and `u_bar()` (the utility posterior's own summary, computed in `utility.py`,
   not here). The decision stays in the daemon (`candidate_posterior` + `optimise`/`value`); the
   bridge transports its inputs.
2. **Prevision inside a Measure or vice-versa?** No new wrapping — the bridge holds no Credence object
   at all; it is a JSON service over Python reads.
3. **Opaque closure where declared structure fits?** No closures; the bridge dispatches named
   functions. It is stateless — it holds no decision object between requests.
4. **`getproperty` override on a Prevision subtype?** No — the bridge adds no methods to any `src/`
   type, in either repo.

## Reviewer checklist

- [x] §0 names transient state explicitly (body + app + e2e + model-choice deferred to Move 4; policy
      stays in the brain not the bridge; skin still drives prod).
- [x] §5 holds non-trivial questions with an argued position each (Q3 capability-vs-policy is the
      load-bearing cut; Q4 names the PII constraint it does *not* yet resolve) + author-approved
      Resolution.
- [x] §8 returns "no posterior here" on all four; the "the bridge decides nothing" claim is
      load-bearing and matches §2 (no `lookup_posterior`/`decide` call in any handler).
- [x] file:line / SHA citations present for current-state references (pi-mono no-MCP @29c1504c;
      `ask.py`/`lookup.py`/`probes.py`/`gather.py`/`utility.py` seams; credence-pi `server.jl` shape).
- [x] The move needs no later move to retract it (deferred files are additive against the frozen
      bridge schema; the bridge changes no existing path).
