// wire.ts — the shapes that cross the brain↔body seam, shared by every
// credence answer-brain body. Deliberately tiny and transport-free: a body
// reaches its daemon over whatever transport it likes (answer-brain's
// synchronous POST /decide; credence-pi's async SSE bus), but the *manifest*
// it verifies against and the *effector signal* it dispatches are the same
// shape. No posterior, no EU, no policy lives here — only the daemon computes
// those (move-4-design §2 single-reasoner invariant).

/**
 * The brain's chosen effector and its parameters, as the (injected) transport
 * surfaces it to the governor. The `effector` name matches a key in the body's
 * effector registry (= a name declared in the manifest); the transport is
 * responsible for mapping any daemon-internal action vocabulary onto the
 * manifest's effector names before the signal reaches `dispatch`.
 */
export interface EffectorSignal {
	effector: string;
	parameters: Record<string, unknown>;
}

// ── Manifest (served by the daemon's GET /manifest; the body has NO parser) ──
// These mirror the JSON the library parser emits (apps/answer-brain/daemon/
// manifest.jl) and replace credence-pi's body-side BDSL parser.

export interface EffectorParam {
	name: string;
	type: string;
}

export interface EffectorDecl {
	name: string;
	parameters: EffectorParam[];
}

export interface FeatureDecl {
	name: string;
	spaceName: string;
}

export interface Manifest {
	effectors: EffectorDecl[];
	features: FeatureDecl[];
}
