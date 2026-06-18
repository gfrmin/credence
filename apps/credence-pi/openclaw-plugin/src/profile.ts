// profile.ts — the user's utility PROFILE (their cost / time / quality / risk trade-offs).
//
// credence-pi is a proxy welfare-maximiser: ONE shared learned belief, the USER'S utility, one
// EU-max (Savage factorisation). The utility weights are the user's PREFERENCE, so the BODY
// ships them — like the model roster — per request, and the daemon applies them with NO restart.
// A "profile" is a named bundle of the dials; the four presets are points in the dial space, and
// `profileOverride` lets a power user set any dial directly. The numbers mirror the welfare.jl
// anchors (cost-hawk λ=0.25/q=0.05, flow-guard λ=4.0/q=1.0) plus the routing reward and the time
// dial. The body stays MATH-FREE — it ships numbers; the brain does all the EU-max.
//
// Wire keys (read by the daemon's _pget): reward (quality: $ value of a correct answer), lambda
// (false-block aversion), q (interruption $), harm (harm $), w_time ($/sec of the user's
// wall-clock — "what is your time worth?"). The full profile is attached to tool-proposed and
// route/escalate-request; the daemon reads whichever keys each decision needs.

export interface Profile {
  reward: number; // quality coordinate ($ value of a correct answer)
  lambda: number; // false-block aversion (unitless)
  q: number; // interruption cost ($)
  harm: number; // harm cost ($)
  w_time: number; // time coordinate ($/sec of wall-clock)
}

export const PROFILE_KEYS = ["reward", "lambda", "q", "harm", "w_time"] as const;

// Four presets spanning the trade-off space:
//  - cost-saver:    money precious — cheap models, block readily, ask freely, time cheap.
//  - balanced:      the middle.
//  - quality-first: pay for the best — escalate fully, fewer false blocks, more harm-averse.
//  - speed-first:   time precious — faster models, escalate sooner, rarely interrupt (asking
//                   costs the user's seconds), high $/sec.
export const PRESETS: Record<string, Profile> = {
  "cost-saver": { reward: 0.02, lambda: 0.25, q: 0.05, harm: 1.0, w_time: 0.002 },
  balanced: { reward: 1.0, lambda: 1.0, q: 0.02, harm: 1.0, w_time: 0.014 },
  "quality-first": { reward: 5.0, lambda: 2.0, q: 0.1, harm: 2.0, w_time: 0.014 },
  "speed-first": { reward: 1.0, lambda: 1.0, q: 0.5, harm: 1.0, w_time: 0.05 },
};

export const DEFAULT_PROFILE = "balanced";
export const PRESET_NAMES = Object.keys(PRESETS);

/**
 * Resolve the active profile: a preset name (default `balanced`) with an optional per-dial
 * override merged on top. An unknown preset falls back to the default. `override` accepts any
 * subset of the dials. Returns the full weight vector the body attaches to sensor events.
 */
export function resolveProfile(
  preset: string | undefined,
  override: Partial<Record<(typeof PROFILE_KEYS)[number], unknown>> | undefined,
): Profile {
  const base = (preset && PRESETS[preset]) || PRESETS[DEFAULT_PROFILE];
  const merged: Profile = { ...base };
  if (override && typeof override === "object") {
    for (const k of PROFILE_KEYS) {
      const v = override[k];
      if (typeof v === "number" && Number.isFinite(v)) merged[k] = v;
    }
  }
  return merged;
}
