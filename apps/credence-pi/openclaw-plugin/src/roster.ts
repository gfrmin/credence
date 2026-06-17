// roster.ts — discover the user's MODEL ROSTER from the OpenClaw config and turn it into the
// per-request routing roster the credence-pi brain routes over.
//
// before_model_resolve hands the plugin only the prompt, NOT the candidate models. But
// register(api) carries api.config (the resolved OpenClawConfig), which lists the configured
// providers + models + per-Mtok pricing. The BODY reads it (IO is the body's job; the brain
// stays config-agnostic and just receives the roster as data) and sends the roster in each
// route-request, so routing is over the user's ACTUAL models — not a hardcoded list.
//
// Cost coordinate: the brain's EU is reward·P(correct) − cost, with cost a per-CALL dollar
// figure. We derive it as output-$/Mtok × NOMINAL_OUTPUT_TOKENS — the same scale the shipped
// default roster uses (haiku 5×700e-6 = 0.0035, sonnet 0.0105, opus 0.0175). The nominal is a
// shared weakly-informative per-call token estimate; being shared across models it sets only
// the overall cost SCALE (calibrated against the profile reward), not the relative ordering.
// The principled refinement — a learned per-model E[tokens/call] (the tb turns-belief move) —
// is a follow-up; this is its cold-start.

import type {
  OpenClawConfigSlice,
  OpenClawModelDefinitionSlice,
} from "./openclaw-types.js";
import type { PriceTable } from "./cost.js";

export interface RoutingModel {
  name: string;
  provider: string;
  model: string; // provider-facing model id
  cost: number; // per-call USD = output-$/Mtok × NOMINAL_OUTPUT_TOKENS
}

// Shared per-call output-token estimate that turns output-$/Mtok into a per-call $. Matches
// the shipped default roster's calibration (bdsl costs == output-$/Mtok × 700).
export const NOMINAL_OUTPUT_TOKENS = 700;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Resolve a model's output price (USD per million tokens): prefer the user's OWN configured
// price, else the plugin price table (built-in defaults + `pricing` overrides), matched the
// same way cost.ts does (exact id, provider/id ref, then longest family-substring).
function resolveOutputPricePerM(
  provider: string,
  def: OpenClawModelDefinitionSlice,
  priceTable: PriceTable,
): number | undefined {
  if (typeof def.cost?.output === "number") return def.cost.output;
  const id = def.id.toLowerCase();
  if (priceTable[id]) return priceTable[id].output;
  const ref = `${provider}/${def.id}`.toLowerCase();
  if (priceTable[ref]) return priceTable[ref].output;
  let best: { key: string; out: number } | undefined;
  for (const [key, p] of Object.entries(priceTable)) {
    const re = new RegExp(`(^|[^a-z0-9])${escapeRegExp(key)}`);
    if (re.test(id) && (best === undefined || key.length > best.key.length)) {
      best = { key, out: p.output };
    }
  }
  return best?.out;
}

/**
 * Build the routing roster from the OpenClaw config. Returns the PRICEABLE models across the
 * configured providers, narrowed to the agent's allow-list (agents.defaults.models) when one
 * is configured. Models with no resolvable price are skipped — the brain can't make a
 * cost-aware EU decision without a cost. The caller routes only when ≥2 remain (else it leaves
 * OpenClaw's model alone), so an undiscoverable roster degrades to no-op, never mis-routing.
 */
export function buildRoster(
  config: OpenClawConfigSlice | undefined,
  priceTable: PriceTable,
): RoutingModel[] {
  const providers = config?.models?.providers;
  if (!providers || typeof providers !== "object") return [];

  // Allow-list (model refs "provider/id"); when present, route only among these.
  const allow = config?.agents?.defaults?.models;
  const allowSet =
    allow && typeof allow === "object"
      ? new Set(Object.keys(allow).map((r) => r.toLowerCase()))
      : undefined;

  const roster: RoutingModel[] = [];
  for (const [provider, pc] of Object.entries(providers)) {
    for (const def of pc?.models ?? []) {
      if (!def || typeof def.id !== "string") continue;
      const ref = `${provider}/${def.id}`.toLowerCase();
      if (allowSet && !allowSet.has(ref) && !allowSet.has(def.id.toLowerCase())) {
        continue;
      }
      const outPerM = resolveOutputPricePerM(provider, def, priceTable);
      if (outPerM === undefined) continue; // unpriceable ⇒ not routable
      roster.push({
        name: def.name ?? def.id,
        provider,
        model: def.id,
        cost: (outPerM / 1_000_000) * NOMINAL_OUTPUT_TOKENS,
      });
    }
  }
  return roster;
}
