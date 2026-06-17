# answer-brain extension (the body)

The pi-mono **body** for the answer-brain: the TS extension that runs the govern+steer
loop over the two HTTP backends — the **daemon** (`POST /decide`, the single reasoner)
and the life-agent **bridge** (`/route`/`/retrieve`/`/extract`/`/probe/*`/`/utility`, the
evidence). It stands on the shared `@credence/brain-body` governor lib (move-4-design §2A);
only the transport (`decideWithGather`) and the effector/feature impls are answer-brain's.

## Shape

The LLM drives three tools (pi-mono is reactive — the brain governs, it does not drive):

```
retrieve_documents(question) → bridge /route + /retrieve + /probe/recency   (accumulate)
extract_candidates()         → bridge /extract → candidates + era_split      (accumulate)
answer(value)                → GOVERNED on tool_call:
      compose evidence + Ū → POST /decide → effector:
        report/hedge → ALLOW, rewrite the answer to the brain's value
        gather(recency) → enact INTERNALLY (re-extract with recency, re-decide)
        ask_clarify  → BLOCK, steer to ask the owner
        abstain      → BLOCK, name the withheld leader (never a confident wrong)
```

The daemon decides every step; the body transports its inputs and enacts its outputs. No
posterior or EU is computed here. The body has **no BDSL parser** — it fetches the
effector/feature vocabulary from the daemon's `GET /manifest` and verifies it implements
it (fail closed at startup). On an unreachable daemon the governor **fails closed** (it
withholds rather than emit an ungoverned answer).

## Files

- `src/tools.ts` — register the tools, accumulate evidence, install the governor (`govern`).
- `src/daemon.ts` — `decideWithGather` (the injected `askBrain`): `/decide` + the internal
  recency-gather loop; the daemon-action→effector mapping + human-facing param composition.
- `src/bridge.ts` — the life-agent capability-bridge client.
- `src/effectors.ts` — answer / ask-user / abstain / gather (the `tool_call` outcomes).
- `src/features.ts` — the four posterior-shape extractors (era-split + owner-scoped are the
  body's to project; dispersion + leader-band are daemon-authoritative display proxies).
- `src/index.ts` — the factory: manifest verify + wiring.
- `src/pi.ts` — the minimal `PiLike` slice of the pi API (so the body tests without pi-mono).

## Develop

```bash
npm install
npm run build   # tsc --noEmit
npm test        # node --test against a fake daemon + fake bridge (no pi-mono, no corpus)
```

The end-to-end run (a real local-Ollama agent over the real corpus) lives in `../app`.
