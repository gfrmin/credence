// cost.ts — reconstruct per-turn USD from llm_output token counts.
//
// OpenClaw does not hand plugins a dollar figure (only token counts +
// model id on llm_output). The host computes USD internally via
// calculateCost(model, usage), but does not expose it to plugins, and we
// stay dependency-free (no `openclaw` runtime import). So we reconstruct
// USD from a small, CONFIG-OVERRIDABLE price table (USD per million
// tokens). The built-in numbers are approximate defaults — the operator
// should verify/override via the plugin's `pricing` config for an exact
// dollars-saved figure. When a model can't be priced, usd is null and the
// Move-2 surface falls back to token counts.

import type { LlmOutputEvent } from "./openclaw-types.js";

export interface ModelPrice {
  input: number; // USD / million input tokens
  output: number; // USD / million output tokens
  cacheRead?: number; // USD / million cache-read tokens
  cacheWrite?: number; // USD / million cache-write tokens
}

export type PriceTable = Record<string, ModelPrice>;

// Built-in defaults, keyed by a lowercase family substring matched against
// the model id. Approximate; override via config `pricing`. Ordered by
// specificity is not required — we pick the longest matching key.
export const DEFAULT_PRICES: PriceTable = {
  // Anthropic 4.x, verified 2026-06 (platform.claude.com): input / output /
  // cache-read / 5m-cache-write, USD per million tokens. (The prior table was
  // stale — opus 15/75 and haiku 0.8/4 were the retired generation.)
  "claude-opus": { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  "claude-sonnet": { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-haiku": { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-4o": { input: 2.5, output: 10 },
  "gpt-4.1": { input: 2, output: 8 },
  "o3": { input: 2, output: 8 },
  "gemini-2.5-pro": { input: 1.25, output: 10 },
  "gemini-2.5-flash": { input: 0.3, output: 2.5 },
};

export interface TurnCost {
  usd: number | null;
  total_tokens: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read: number | null;
  cache_write: number | null;
  model: string | null;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function resolvePrice(
  model: string | undefined,
  table: PriceTable,
): ModelPrice | undefined {
  if (!model) return undefined;
  const m = model.toLowerCase();
  // Exact match first.
  if (table[m]) return table[m];
  // Longest key that appears as a SEGMENT of the id — anchored at the
  // start or after a non-alphanumeric separator. So "o3" matches
  // "o3-mini" and "openai/o3" but NOT "foo3"; longest match wins.
  let best: { key: string; price: ModelPrice } | undefined;
  for (const [key, price] of Object.entries(table)) {
    const re = new RegExp(`(^|[^a-z0-9])${escapeRegExp(key)}`);
    if (re.test(m) && (best === undefined || key.length > best.key.length)) {
      best = { key, price };
    }
  }
  return best?.price;
}

/** Merge operator `pricing` config over the built-in table. */
export function buildPriceTable(overrides: unknown): PriceTable {
  const table: PriceTable = { ...DEFAULT_PRICES };
  if (overrides && typeof overrides === "object") {
    for (const [k, v] of Object.entries(overrides as Record<string, unknown>)) {
      if (v && typeof v === "object") {
        const o = v as Record<string, unknown>;
        const input = typeof o.input === "number" ? o.input : undefined;
        const output = typeof o.output === "number" ? o.output : undefined;
        if (input !== undefined && output !== undefined) {
          table[k.toLowerCase()] = {
            input,
            output,
            cacheRead: typeof o.cacheRead === "number" ? o.cacheRead : undefined,
            cacheWrite:
              typeof o.cacheWrite === "number" ? o.cacheWrite : undefined,
          };
        }
      }
    }
  }
  return table;
}

export function computeTurnCost(
  event: LlmOutputEvent,
  table: PriceTable,
): TurnCost {
  const u = event.usage ?? {};
  const input = u.input ?? 0;
  const output = u.output ?? 0;
  const cacheRead = u.cacheRead ?? 0;
  const cacheWrite = u.cacheWrite ?? 0;
  const total = u.total ?? input + output + cacheRead + cacheWrite;
  const model = event.model ?? null;

  const price = resolvePrice(event.model, table);
  let usd: number | null = null;
  if (price) {
    usd =
      (input * price.input +
        output * price.output +
        cacheRead * (price.cacheRead ?? price.input) +
        cacheWrite * (price.cacheWrite ?? price.input)) /
      1_000_000;
  }

  return {
    usd,
    total_tokens: total,
    input_tokens: input,
    output_tokens: output,
    cache_read: cacheRead,
    cache_write: cacheWrite,
    model,
  };
}
