# answer-brain — Move 2 design (Stage 2a: the wire surface + the capability bridge)

Follows `docs/posture-4/DESIGN-DOC-TEMPLATE.md`. Strategy is settled in `master-plan.md`; Stage 1
(`move-1-design.md`) landed the native brain with parity. This move stands up the two **backends** the
pi-mono body will drive — the daemon's decision wire and the life-agent capability bridge — and proves
the wire preserves Stage-1 parity. The pi-mono app + governor extension (the body that consumes both)
is the named successor, Move 3.

## 0. Final-state alignment

`master-plan.md` §"Target architecture" is three processes: a pi-mono answering agent (TS) that
proposes `retrieve`/`extract`/`probe`/`answer`; the **answer-brain daemon** (Julia) that holds the
candidate posterior and governs answer/gather/ask/abstain by EU; and **life-agent capabilities**
(Python) the agent's tools call. Stage 1 built the brain's *math*; this move gives it and the Python
capabilities a **wire**, so Move 3's body has two live, independently-tested backends to drive.

Transient state, named so it is not drift:
- **The pi-mono app + governor extension are deferred to Move 3.** They need both backends live to
  test end-to-end; building them now would be against an unproven wire. What freezes here is the
  body's contract — the daemon's `decide` request/response schema and the bridge's tool schema — so
  Move 3 is additive (write the body against a frozen seam), never a retraction of this move.
- **life-agent keeps driving the skin in production** (`core/gather.py`); this move adds a *bridge*
  process beside it, it does not reroute the live ask path. The bridge reuses life-agent's existing
  reads (`_retrieve_set`, `observe_hits`, `probes.*`); it rebuilds nothing.

## 1. Purpose

Stand up and parity-test two backends:

1. **The answer-brain daemon's decision wire** (`apps/answer-brain/daemon/server.jl`): a stateless
   HTTP surface over Stage-1's `candidate_posterior` + the `optimise`/`value` terminal decide
   (`decide_full`). Each `POST /decide` rebuilds the posterior from the request's full observation
   vector and returns the chosen effector + the reported candidate index. The brain math is
   unchanged — the wire only transports its inputs (abstract observations + Ū + channel params) and
   its outputs (effector, report index, credences). (`voi_gather`, the forward gather gate, exists
   from Stage 1; its answer-vs-gather arbitration wires in Move 3 with the body's probe set.)

2. **The life-agent parity boundary** (`src/life_agent/bridge/observations.py`): the **Python side**
   of the parity cut (`move-1-design.md` §"parity boundary") — `to_abstract_observations` maps
   grounded observations to **abstract integer observations** + the candidate display strings, the
   exact shape Stage 1's fixtures encode, so the brain stays string-blind. The HTTP bridge *service*
   wrapping this (with `retrieve`/`extract`/`probe_*` over a live corpus, pi-mono having **no MCP
   client** — §5 Q1) is deferred to Move 3, built and validated end-to-end with its pi-mono consumer:
   it cannot be hermetically tested without a model, and its endpoint shapes are fixed by the body's
   needs — building it now is plumbing against an unproven consumer (the reason Stage 1 likewise
   deferred `server.jl`).

**Proof obligation:** posting Stage-1's `stage0_parity.json` cases as `decide` *wire requests*
reproduces the native effector + EU + posterior to `atol=1e-9` (parity survives the wire); and
`to_abstract_observations`, on synthetic observations, reproduces the candidate-indexing /
ancestry-grouping / identity-collapse / covariate pass-through the fixtures were generated through.

## 2. Files touched

Created — `credence` (`apps/answer-brain/`):
- `daemon/server.jl` — HTTP, **stateless** (§5 Resolution). `POST /decide` `{question_id,
  observations[], candidates[], rho, u_bar, channel?} → {effector, report_index, value, credences[],
  p_none, eu}`; `GET /ready`. Each call rebuilds the posterior from the full observation vector, so
  no per-question state is kept server-side — isolation is by construction. Wraps
  `AnswerBrain.candidate_posterior` + the `optimise`/`value` decide (the report value is `optimise`'s
  action key, not an `argmax(weights)`); **adds no inference** — every numerical result is the
  brain's. Adapts `apps/credence-pi/daemon/server.jl` (the HTTP/JSON3 plumbing, the `/ready` probe),
  shedding the SSE machinery (§5 Q2) and the stateful posterior.
- `daemon/main.jl` — grow the Stage-1 skeleton to `start_daemon(; port, host)` and block;
  env-configurable (`ANSWER_BRAIN_HOST`/`ANSWER_BRAIN_PORT`). Stateless ⇒ nothing to load/replay on
  boot, unlike credence-pi's.
- `tests/julia/test_server.jl` — wire-parity: each `stage0_parity.json` case through
  `decide_response` reproduces the effector + EU + posterior (≡ Stage-1 native); the report index is
  the weight leader; statelessness (an interleaved question doesn't perturb a repeat); plus a live
  HTTP round-trip (`/ready`, `/decide`, malformed→400). **48 checks.**

Created — `life-agent` (separate repo + commit):
- `src/life_agent/bridge/observations.py` — `to_abstract_observations`: the **observation →
  abstract-obs** mapping (candidate canon via the existing `_candidate_key`, ancestry-group keying,
  covariate floats → the integer obs records). The **single source** of the parity boundary — the
  fixture oracle (`scripts/dump_parity_fixtures.py`, refactored to call it) and the Move-3 bridge
  both use it, so the brain's input shape is provably the fixtures'.
- `src/life_agent/bridge/__init__.py` — package docstring naming the Move-3 service deferral.
- `tests/test_bridge.py` — **7 hermetic cases** pinning the mapping (distinct→indexed, same-doc→one
  group, first-seen group order, date/format identity collapse, distinct-stay-separate, covariate
  pass-through, empty). No model, no corpus.

No `src/Credence` change; no `src/pkm` change; no edit to a landed migration or to `core/lookup.py`'s
math (the mapping *calls* it).

Deferred to Move 3 (named): the **life-agent bridge HTTP service** (`src/life_agent/bridge/server.py`
— `retrieve`/`extract`/`probe_*` over the live corpus, stdlib `http.server`); `apps/answer-brain/
extension/*` (the pi-mono TS body — `tool_call` governor + `tool_result` observer + the bridge-tool
registrations); the minimal pi-mono answering app/harness; the end-to-end eval run + gate.

## 3. Behaviour preserved

The reference is Stage-1's native brain (itself parity-locked to Stage-0 Python `@4b336db`). The wire
must not perturb it:
- **Effector + EU** from `POST /decide` == `AnswerBrain.terminal_decide` on the same inputs, exact
  effector / `atol=1e-9` EU. The fixtures are reused verbatim as request bodies; any divergence is a
  serialisation or state-keying bug, not a benign reassociation (no RNG on this path).
- **The parity boundary holds:** the daemon receives only integers + floats (reports, group,
  authority, subject_factor, time_factor) + Ū; it never sees a candidate string.
  `to_abstract_observations` owns all string-canon and group keying, Python-side — the single-source
  mapping the fixture oracle also uses.
- **Per-question isolation:** stateless ⇒ independent by construction — two `question_id`s decided in
  one process yield independent results; a decide on one cannot be perturbed by another's
  observations.

## 4. Worked end-to-end example

The `move-1-design.md` §4 case (one obs reporting candidate-0, `k=2`+NONE, Ū with `u_wrong=-5`) as a
wire round-trip:

```
POST /decide
{ "question_id": "q-demo",
  "observations": [ {"reports": 0, "group": 0, "authority": 0.9,
                     "subject_factor": 1.0, "time_factor": 1.0} ],
  "candidates": ["alpha", "bravo"],
  "u_bar": {"u_correct":1.0,"u_wrong":-5.0,"u_hedged":-0.5,"u_abstain":0.0,
            "oracle_p":0.9,"lambda_int":0.05,"kappa_att":0.0} }
→ 200
{ "effector": "abstain", "report_index": null, "value": null,
  "credences": [0.754, 0.082], "p_none": 0.164, "eu": 0.0 }
```

Server-side this is verbatim Stage 1: `candidate_posterior` builds the `CategoricalMeasure`, conditions
the PushOnly kernel (weights `≈[0.754,0.082,0.164]`), `decide_full` runs `optimise` over
`{report_0,report_1,hedge,ask,abstain}` and returns `abstain` because EU(report_0)=`0.754−5·0.246=
−0.476` < EU(abstain)=`0`. The `report_j` argmax stays inside `optimise` (Invariant 1); on a report,
`decide_full` returns `report_index = j*` (the action key `optimise` chose, not an `argmax(weights)`)
and the wire fills `value = candidates[j*]`. The leading candidate is *named in the response*
(`credences` aligned to `candidates`) so an abstain still surfaces the withheld leader — the body
renders it.

## 5. Open design questions

**Q1 — TS→Python transport.** The plan said "exposed via the dormant `src/pkm/mcp_server.py`
bridge", but pi-mono ships **no MCP client** (verified `pi-mono@12bb8dd2`: the only
`modelcontextprotocol` references are a package name in test files; the tool surface is
`pi.registerTool(ToolDefinition)` with a TS `execute`). So MCP-as-written is out, and a bridge
tool's `execute` must reach Python another way. Options: **(a) an HTTP service** (life-agent runs a
small JSON server; the pi-mono tools `fetch` it) — standard, debuggable, warm DuckDB/skin, a port to
manage; **(b) a stdio JSON-RPC subprocess** the app spawns (mirrors how life-agent already spawns the
Julia skin, `core/brain.py`) — no port, but the app owns a Python process lifecycle in TS. Both keep
Python warm (a per-call `uv run` cold-start — seconds of DuckDB+model init — is rejected for an agent
loop). **I recommend (a) HTTP:** the daemon is already HTTP, so the body talks to two uniform HTTP
backends; lifecycle is "start two services", trivially scriptable for the eval harness; and it keeps
pkm **frozen** (a *new* `life_agent` bridge, not life-agent tools bolted into pkm's MCP server, which
would violate the freeze). Push back if you'd rather the app own the Python lifecycle (b) for a
single-process-to-launch story.

**Q2 — daemon wire: synchronous request/response vs. the credence-pi SSE governor.** credence-pi's
`server.jl` is `POST /sensor` (fire-and-forget) + `GET /signals` (SSE push), because *its* decision
can defer across a user round-trip. The answer-brain's decision is **synchronous**: given the
observations gathered so far, decide *now*. The reactive loop is driven by the LLM's tool calls — the
governor's `tool_call` hook blocks the `answer` call and awaits one decision — so the brain never
needs to *push* a signal absent a tool call. **I recommend request/response** (`POST /decide`
returns the effector in the response body): it is YAGNI-correct, ~half the server code, and the
`ask-user` effector is realised by the body's own `ctx.ui.confirm` (pi-mono exposes it) rather than a
second async signal. The SSE governor is the faithful-to-credence-pi alternative and the right shape
*if* a future §16 always-on governor must push unprompted — but that is out of scope until the
answering brain is gate-certified (`master-plan.md` §"Out of scope"). Push back if you want the SSE
surface now for template symmetry.

**Q3 — Stage 2 as one move or two.** Stage 2 is a Julia daemon + a Python bridge + a TS app +
extension + an e2e eval — large, and the novel risk (does the wire preserve parity; does the bridge
reproduce the mapping) is **independent of pi-mono**. **I recommend this two-move split:** Move 2
(this doc) lands + tests both backends with no pi-mono dependency; Move 3 writes the body against the
frozen seam and runs the gate. This de-risks the wire before the agent-loop plumbing and gives a
green checkpoint that doesn't need a model in the loop. Alternative: one move, but then the first
green light requires the whole stack standing — more to debug at once, against the template's
"a move needs no later move to retract it" preference. Push back if you'd rather one move for a single
end-to-end milestone.

**Resolution (author-approved, in conversation).** Q1 → **(a) HTTP service** — a new `life_agent`
JSON-over-HTTP bridge (stdlib `http.server`, no new dependency); `pkm` is untouched. Q2 →
**request/response** (`POST /decide` returns the effector in the response body); the SSE governor is
deferred with the §16 always-on governor. Q3 → **two moves** — this move lands both backends +
wire-parity with no pi-mono dependency; Move 3 writes the body against the frozen seam and runs the
gate. All three are the recommended options; this move is scoped accordingly.

A build-time refinement (recorded here, not a re-decision, surfaced when reading the Stage-1 brain
surface): **`/decide` is stateless.** `candidate_posterior` rebuilds the posterior from the full
observation vector each call — a handful of `condition` calls, cheap — so no per-question posterior is
kept server-side. The body holds the accumulated evidence and resends it each decision; the daemon
holds the decision *policy*, not the belief state. This makes §3's per-question isolation hold *by
construction* and voids §6's state-leak risk. The chosen report *value* comes from `optimise`'s
returned action key (`act ≤ k ⇒ report candidate act−1`), never an `argmax(weights)` in the daemon —
Invariant-1-pure; the body uses `credences` only to *display* the withheld leader. The observation log
stays append-only for audit/replay and is wired to a live session in Move 3.

## 6. Risk + mitigation

- **Wire serialisation drift** (a float rounded, a field renamed). *Caught by:* the fixture
  wire-parity test at `atol=1e-9` — the same oracle Stage 1 used, now exercised through JSON3.
- **Per-question cross-talk.** *Voided by design (§5 Resolution):* `/decide` is stateless — each call
  rebuilds the posterior from the request's full observation vector, so there is no shared mutable
  state to leak across `question_id`s. The isolation test still pins it (two `question_id`s decided in
  one process yield independent results).
- **Bridge mapping divergence** from the parity oracle. *Mitigation:* there is **one** mapping
  function (`bridge/observations.py`), called by both the fixture oracle
  (`scripts/dump_parity_fixtures.py`) and the Move-3 bridge — divergence is impossible by
  construction; the 7 hermetic cases pin its output.
- **pkm-freeze violation** by extending its MCP server. *Mitigation:* Q1(a) puts the bridge in
  `life_agent`, not `pkm`; pkm is untouched.
- **Python cold-start** making the loop slow. *Mitigation:* the bridge is a warm long-lived process
  (Q1); DuckDB read-only handle + `shared_brain()` initialised once at boot, as `core/lookup.py`
  already does per ask-session.

## 7. Verification cadence

End-of-move (all green): `julia --project=. apps/answer-brain/tests/julia/test_server.jl` (48 checks:
wire parity + isolation + a live HTTP round-trip — the §4 smoke, automated); the Stage-1 regression
`test_answer_brain.jl` (57 checks — the `decide_full` refactor is safe); `uv run pytest
tests/test_bridge.py` (7 hermetic) in life-agent; `ruff` + `mypy --strict` clean on touched
`life_agent/` src; `tools/credence-lint/credence_lint.py check apps/answer-brain` clean (6 files). No
model is in this move's loop, so every run is deterministic and reported with exact counts.

## 8. de Finettian discipline self-audit

1. **Every numerical query through `expect`?** Yes — unchanged from Stage 1. `server.jl` calls
   `candidate_posterior` + `decide_full` (`optimise`/`value`); it computes no probability itself. It
   serialises their outputs (`credences`, `eu`, `effector`, the report index) — transport, not
   inference. The parity-boundary mapping computes **no** posterior at all: it returns declared data
   (the candidate strings + the abstract-obs records).
2. **Prevision inside a Measure or vice-versa?** No new wrapping — the daemon holds the same
   `CategoricalMeasure` per question; the wire carries plain JSON.
3. **Opaque closure where declared structure fits?** No new closures; the kernels/utilities are
   Stage 1's declared `Kernel`/`Tabular`. The daemon is stateless — it holds no decision object
   between requests at all.
4. **`getproperty` override on a Prevision subtype?** No — neither backend adds methods to `src/`
   types.

## Reviewer checklist

- [x] §0 names transient state explicitly (deferred bridge service + extension + app + e2e; skin
      still drives prod).
- [x] §5 holds three non-trivial questions with an argued position each (+ author-approved
      Resolution).
- [x] §8 returns "yes" on all four; the mapping's "no inference" claim is the load-bearing one.
- [x] file:line / SHA citations present for current-state references (pi-mono no-MCP @12bb8dd2;
      credence-pi `server.jl`; life-agent `lookup.py`/`dump_parity_fixtures.py`).
- [x] The move needs no later move to retract it (deferred files are additive against a frozen seam).
