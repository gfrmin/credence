// @credence/brain-body — the transport-agnostic governor core shared by
// credence answer-brain bodies on pi-mono. A body injects its transport
// (`askBrain`), its effector impls + feature extractors, and the value to fail
// open with; the lib supplies the govern loop, the manifest fetch+verify (no
// BDSL parser), and the dispatch/feature harness. See move-4-design §2A.

export type {
	EffectorSignal,
	EffectorParam,
	EffectorDecl,
	FeatureDecl,
	Manifest,
} from "./wire.js";

export {
	type EffectorRegistry,
	type FeatureExtractor,
	dispatchEffector,
	assembleFeatures,
} from "./dispatch.js";

export {
	type FetchManifestOptions,
	type Registries,
	fetchManifest,
	verify,
} from "./manifest.js";

export { type GovernSpec, govern, withTimeout } from "./governor.js";
