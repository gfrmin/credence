# Personal-agent design priors (provisional)

> **Status:** provisional, dated 2026-04-27. Source: architectural-review conversation.
>
> This document was originally drafted as Posture 4's Move 9 under the assumption that personal-agent work was the next product direction. The MVP framing has since shifted to credence-proxy v0.1 (see `docs/posture-5/master-plan.md`), and this document's design questions — Connection abstract type, event-form vs parametric-form convention, Telegram preference encoding — are deferred until empirical evidence from the proxy's deployment informs the brain/body interface design. The priors here are preserved as forensic record, not as commitment.

---

# Original Move 9 design doc — Body work

## 0. Final-state alignment

Move 9 delivers the first consumer of the Posture 4 foundation outside the reconstruction's own tests. The email agent with Maildir sync, Ollama enrichment, Telegram training loop, and production persistence demonstrates that the Prevision-primary architecture (Moves 1–8) is usable for a real application, not merely correct for synthetic benchmarks. After this move, the `apps/julia/email_agent/` directory is a complete brain-side application backed by body-side connections, as specified in `SPEC.md` §6.3. The `Connection` abstraction is concrete, the server loop runs, and the belief state round-trips through production persistence (schema v4). Transient state: Move 9 does not deliver calendar/files/tasks connections — those are explicitly deferred per `master-plan.md` line 246.

## 1. Purpose

Implement the body-work move: formalise the `Connection` abstraction, wire Gmail via Maildir (mbsync), add Telegram bot for user-feedback training loop, build the server loop with polling execution and meta-actions, and land production persistence (schema v4). This closes the "email is the book" loop: email arrives → feature extraction → Prevision update → action selection → user reaction → further Prevision update.

## 2. Files touched

### Created

- `apps/julia/email_agent/connection.jl` — `Connection` abstract type + Gmail and LLM connection implementations
- `apps/julia/email_agent/maildir.jl` — Maildir reader: parse email files from `~/Maildir/INBOX/new/` and `cur/`, extract `Email` structs
- `apps/julia/email_agent/telegram.jl` — Telegram bot: send proposed actions, receive 👍/👎 reactions, encode as observations
- `apps/julia/email_agent/server.jl` — Server loop: poll Maildir for new emails, run decision loop per email, dispatch Telegram notifications, handle feedback
- `apps/julia/email_agent/production_persistence.jl` — Schema v4 save/load: full `AgentState` + connection registries + cost model + Telegram state

### Modified

- `apps/julia/email_agent/host.jl` — Refactored: decision loop extracted to a reusable `decide_and_act(state, features, connections)` function callable from both `host.jl` (simulation) and `server.jl` (production)
- `apps/julia/email_agent/live.jl` — Updated: interactive CLI driver uses the refactored `decide_and_act` API
- `apps/julia/email_agent/features.jl` — Extended: `extract_features(::Email)` gains a `from_maildir::Bool` parameter for real emails (vs synthetic corpus)
- `apps/julia/email_agent/llm_prosthetic.jl` — Minor: `LLMConnection` wrapper implementing the `Connection` interface
- `apps/julia/email_agent/state_persistence.jl` — Updated: delegates to `production_persistence.jl` for v4 schema; v3 load remains for migration
- `src/persistence.jl` — Extended: v4 schema marker; v3→v4 migration path

### Test files

- `apps/julia/email_agent/test_server.jl` — End-to-end smoke test: stubbed Maildir + stubbed Ollama + stubbed Telegram → belief state evolves correctly
- `test/fixtures/posture-4-move-9/` — v4 production state fixture (first capture)

## 3. Behaviour preserved

Move 9 is additive — it does not modify the Tier 1 DSL core or the execution layer. The existing `host.jl` simulation and `live.jl` interactive driver continue to work against the refactored API. All Move 0 capture invariants hold (Move 9 does not touch `src/ontology.jl` beyond what was already modified in Moves 1–8).

The refactoring of `host.jl` to extract `decide_and_act` is a pure extraction — the simulation loop calls the same function the server loop calls, so numerical results of the simulation are identical.

## 4. Worked end-to-end example

A new email arrives in `~/Maildir/INBOX/new/`:

```
From: boss@company.com
Subject: URGENT: Q3 budget review — action required
Date: Mon, 28 Apr 2026 09:15:00 +0100
Content-Type: text/plain

Please review the attached budget spreadsheet and respond by EOD.
```

**Step 1: Maildir polling** (`server.jl`)
Server loop detects new file in `~/Maildir/INBOX/new/`. Calls `GmailConnection.extract(filepath)`.

**Step 2: Feature extraction** (`connection.jl` → `features.jl` → `maildir.jl`)
`GmailConnection.extract` parses the email file, constructs an `Email` struct, calls `extract_features(email)`. Result:

```julia
features = Dict{Symbol, Float64}(
    :subject_has_urgent_kw => 1.0,
    :subject_has_action_kw => 1.0,
    :subject_has_money_kw  => 1.0,
    :sender_is_noreply     => 0.0,
    :sender_is_bulk_domain => 0.0,
    # ... 20+ more features
)
```

**Step 3: LLM enrichment** (`llm_prosthetic.jl`)
`LLMConnection.extract(features)` calls Ollama to detect additional features:
```julia
enriched = Dict(:llm_has_urgent_signal => 0.95, :llm_has_action_request => 0.88, ...)
merge!(features, enriched)
```

**Step 4: Conditioning on features** (`host.jl` → `decide_and_act`)
For each declared Event in the connection registry, the server evaluates it against the feature dict. Events that fire are conditioned:

```julia
# FeatureEquals(:subject_has_urgent_kw, 1.0) fires
state.belief = condition(state.belief, TagSet(urgent_tag_indices))

# FeatureInterval(:llm_has_action_request, 0.5, 1.0) fires
k = answer_kernel(state, :llm_has_action_request)  # Kernel from reliability state
state.belief = condition(state.belief, k, 0.88)
```

Feature-dictionary observations use both condition forms per §5.2: binary features that are structurally known (`:subject_has_urgent_kw` is 0 or 1 by construction) use event-form; continuous LLM-enriched features (`:llm_has_urgent_signal` with noise) use parametric-form.

**Step 5: Action selection** (`host.jl` → `decide_and_act`)
EU maximisation over domain + meta actions:

```julia
action, eu = optimise(state.belief, all_actions, preference_fn)
# action = :flag_urgent, eu = 3.2
```

**Step 6: Telegram notification** (`telegram.jl`)
Server sends proposed action to Telegram:
```
📧 From: boss@company.com
📋 URGENT: Q3 budget review — action required
🤖 Agent recommends: FLAG_URGENT (confidence: 0.89)
👍 Approve  👎 Override
```

**Step 7: User reaction** (`telegram.jl`)
User taps 👍. Server receives callback, encodes as observation:

```julia
# Approve = observation 1.0 on the preference kernel
state.belief = condition(state.belief, preference_kernel, 1.0)
```

**Step 8: Action execution** (`server.jl`)
Execute the approved action primitives: `add_label_urgent`, `move_to_priority`, `notify_user` via JMAP or Maildir flag operations.

**Step 9: Persistence** (`production_persistence.jl`)
Save updated state to `~/.credence/state.jls` (schema v4).

## 5. Open design questions

### 5.1 Connection: abstract type with required methods, or struct with function fields?

SPEC.md §6.3 shows a concrete `struct Connection` with function fields (`extract::Function`, `execute!::Function`). The prompts suggest abstract type with required methods as a prior.

**Option (a): Abstract type with required methods.**
```julia
abstract type Connection end
extract(c::Connection, event) = error("implement extract for $(typeof(c))")
execute!(c::Connection, context, action) = error("implement execute! for $(typeof(c))")
features(c::Connection) = error("implement features for $(typeof(c))")
events(c::Connection) = error("implement events for $(typeof(c))")
```
Pro: Julia-idiomatic dispatch. Each connection is its own type with method specialisation. Con: traits in Julia are fragile — no compile-time verification that a concrete type implements all required methods.

**Option (b): `@kwdef struct` with function fields.**
```julia
@kwdef struct Connection
    name::Symbol
    features::Vector{Symbol}
    actions::Vector{Symbol}
    events::Vector{Event}
    extract::Function
    execute!::Function
end
```
Pro: SPEC.md-aligned, self-documenting (fields enumerate the interface), no dispatch fragility. Con: `Function` fields defeat Invariant 2's declared-structure discipline — opaque closures where typed dispatch would fit.

**My prior:** option (a). The `extract::Function` field in SPEC.md was written before Invariant 2's closure discipline was codified. The abstract type pattern aligns with Move 7's `LikelihoodFamily` subtypes — each connection is a declared type with methods. The "traits are fragile" concern is mitigated by the end-to-end test that exercises all required methods.

### 5.2 Event-form vs parametric-form condition: when does each apply?

The body work uses both condition forms. The question is the convention for choosing between them.

**Convention I propose:** Binary features known by construction (`:subject_has_urgent_kw` is 0.0 or 1.0, derived from string matching) use event-form: `condition(belief, TagSet(indices))`. Continuous features with measurement noise (Ollama enrichment outputs like `:llm_has_urgent_signal = 0.88`) use parametric-form: `condition(belief, kernel, observation)`, where the kernel encodes the LLM's reliability per feature.

The distinction is observational, not semantic: event-form conditions on structural predicates (the world is in a known state), parametric-form conditions on noisy observations (the world generated data through a channel with known noise characteristics). This matches the Move 7 §5.1 elevation: event-form is a peer primary, not derived from parametric-form.

**Open question for review:** Is this distinction sufficient, or should there be a third form for "soft binary" features — e.g., `:sender_frequency` which is continuous but derived from exact counts (no measurement noise)?

### 5.3 Telegram preference encoding: Event or Kernel-obs?

User reactions (👍/👎) update the agent's belief about whether its actions are correct. Two encodings:

**Option (a): Kernel-obs pair.** 👍 = observation 1.0, 👎 = observation 0.0, conditioned through a BetaBernoulli kernel over a "user-approval" parameter. Each domain action has an associated approval-rate Beta. This is the standard reliability-learning pattern from `credence_agents`.

**Option (b): Event.** 👍 = `TagSet` event that fires the "correct-action" hypothesis components. 👎 = complement event. This conditions the mixture weights directly, boosting programs whose recommended action the user approved.

**My prior:** option (a). The reliability-learning pattern is well-tested (Moves 1–8, the entire `credence_agents` benchmark). Option (b) is more principled (the user's reaction conditions the program hypothesis space directly) but requires mapping from reaction → which mixture components predicted this action, which is fragile if the program space is large. Option (a) is the affordance-learning mechanism from SPEC.md §6.8: the agent learns what each action achieves through conditioning on outcomes.

**The CIRL angle:** The master plan references "CIRL prior over utility parameters." The full CIRL formulation would have the user's reaction condition a joint prior over (action-correctness, utility-parameter-vector). This is architecturally clean but adds a dimension to the state space. My prior: defer the full CIRL formulation to a follow-up. Move 9 uses option (a) — per-action approval rates — which is CIRL in the degenerate case where the utility function is "user approves."

## 6. Risk + mitigation

**Risk 1: Move 9 scope is too large.** The master plan flags this explicitly (line 284). Mitigation: the design doc proposes a 9a/9b split point. 9a = Maildir + feature extraction + persistence + refactored host API. 9b = Telegram + server loop + preference learning. 9a is self-contained and testable without external dependencies (stubbed Maildir, no Telegram). 9b requires Telegram API credentials and a running Ollama instance.

**Risk 2: Maildir format parsing is fragile.** Real-world email files have encoding issues, multipart MIME, embedded images, etc. Mitigation: Move 9 parses headers only (From, Subject, Date, Content-Type). Body text is not parsed — the LLM prosthetic handles body understanding. Header parsing is well-specified (RFC 2822) and the existing `Email` struct uses only header-derived features.

**Risk 3: Telegram bot webhook requires a public endpoint.** Mitigation: use Telegram's long-polling API (`getUpdates`) instead of webhooks. Long-polling works from behind NAT without a public endpoint. The server loop already polls Maildir; adding Telegram polling is a second poll in the same loop.

**Risk 4: Production persistence schema v4 may not be forward-compatible.** Mitigation: v4 is a first capture with a fixture. The "never regenerate" rule applies post-Move-9 (per `claude-code-prompts.md` line 111). Schema versioning field stays for forward compatibility.

**Risk 5: `decide_and_act` extraction from host.jl may break the simulation.** Mitigation: the simulation test suite (`host.jl` with synthetic corpus) runs before and after the extraction. Numerical results must be identical (seeded RNG, same seed).

## 7. Verification cadence

End-of-PR verification:

1. `julia test/test_core.jl` — DSL core unchanged
2. `julia apps/julia/email_agent/test_server.jl` — end-to-end smoke test (stubbed Maildir + Ollama + Telegram)
3. `julia apps/julia/email_agent/host.jl` — simulation still runs (numerical output unchanged)
4. `python tools/credence-lint/credence_lint.py check apps/` — 0 violations
5. `python tools/credence-lint/credence_lint.py test` — corpus 14/10/5
6. `PYTHON_JULIACALL_HANDLE_SIGNALS=yes uv run pytest apps/python/` — Python tests unaffected
7. Skin smoke test: `uv run python apps/skin/test_skin.py` — skin unchanged
8. Production persistence round-trip: save state → load state → compare

## 8. de Finettian discipline self-audit

1. **Is every numerical query in this move routed through `expect`?** Yes. The decision loop calls `optimise` (which calls `expect`) for action selection. Feature extraction produces raw observables; the brain queries beliefs via `expect`. No direct arithmetic on Prevision parameters in application code.

2. **Does this move hold a Prevision inside a Measure, or a Measure inside a Prevision?** No. The email agent constructs `MixturePrevision` of `ProductPrevision` of `BetaPrevision` (post-Move-8 vocabulary). `CategoricalMeasure` stays for the action space (principled exception). No new Measure↔Prevision nesting.

3. **Does this move introduce an opaque closure where a declared structure would fit?** The `Connection.extract` and `Connection.execute!` methods are typed dispatch (option (a) in §5.1), not opaque closures. The Telegram reaction encoding is a BetaBernoulli kernel (declared likelihood family), not a closure. The Ollama prompt is an opaque string, but it is a body concern (prosthetic), not a brain concern — it does not enter any axiom-constrained function.

4. **Does this move add a `getproperty` override on any Prevision subtype?** No.

## 9. Commit sequence

1. **Refactor `host.jl`: extract `decide_and_act`.** Pure extraction; simulation output unchanged.
2. **`connection.jl` + `maildir.jl`: Connection abstraction + Maildir reader.** Gmail connection implements the interface. Test: parse a test Maildir directory.
3. **`telegram.jl`: Telegram bot.** Long-polling `getUpdates`, send/receive messages, reaction encoding. Test: stubbed HTTP responses.
4. **`server.jl`: Server loop.** Polls Maildir + Telegram, dispatches to `decide_and_act`, executes actions. Test: end-to-end with stubs.
5. **`production_persistence.jl`: Schema v4.** Save/load full production state. Test: round-trip fixture.
6. **Documentation + lint.** Update `apps/julia/email_agent/CLAUDE.md` if present.

### Proposed 9a/9b split

If the move is too large for a single PR:
- **9a** (commits 1–2, 5): Refactored host + Connection + Maildir + persistence. Self-contained, testable without external APIs.
- **9b** (commits 3–4, 6): Telegram + server loop. Requires Telegram API token.
