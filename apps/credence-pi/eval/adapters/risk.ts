// risk.ts — risk-aware feature primitives for the eval adapters.
//
// SINGLE SOURCE OF TRUTH: these live in the body (openclaw-plugin/src/safety.ts) so the
// offline-trained harm posterior and the LIVE governor compute identical features — the same
// pattern as the waste FeatureTracker (also imported from the plugin). The eval re-exports
// them here so the adapters' imports are unchanged.
export {
  actionClass,
  targetExternality,
  looksUntrusted,
  isReadOnly,
  isSink,
  isExternalSend,
  isCredentialAccess,
  extractTokens,
  taintFlow,
  untrustedImperatives,
  matchesImperative,
} from "../../openclaw-plugin/src/safety.js";
