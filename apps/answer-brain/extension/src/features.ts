// features.ts — the four posterior-shape feature extractors (features.bdsl). They
// read the accumulated evidence and return a kebab-case space member. Two are the
// body's to PROJECT (the daemon cannot see them):
//
//   candidates-era-split  the bridge projected it (raw obs + doc_date); the body forwards it
//   owner-scoped          a "my X" question (sent to /decide; daemon v0 reserves it)
//
// The other two the daemon computes authoritatively from the posterior it builds; the body's
// versions here are DISPLAY proxies off the daemon's last returned credences (move-4-design
// §2C). All four are registered so the manifest `verify` passes (an extractor per declared
// feature); only era-split + owner-scoped cross the wire to /decide.

import type { FeatureExtractor } from "@credence/brain-body";

import type { Evidence } from "./types.js";

const dispersion: FeatureExtractor<Evidence> = (ev) => {
	if (!ev.lastCredences.length) return "moderate";
	const max = Math.max(...ev.lastCredences);
	if (max >= 0.7) return "sharp";
	if (max >= 0.4) return "moderate";
	return "dispersed";
};

const leaderBand: FeatureExtractor<Evidence> = (ev) => {
	if (!ev.lastCredences.length) return "near-bar";
	const max = Math.max(...ev.lastCredences);
	if (max >= 0.8) return "above-bar";
	if (max >= 0.5) return "near-bar";
	return "below-bar";
};

const eraSplit: FeatureExtractor<Evidence> = (ev) => (ev.eraSplit ? "yes" : "no");

const ownerScoped: FeatureExtractor<Evidence> = (ev) => (ev.ownerScoped ? "yes" : "no");

export const extractors: Record<string, FeatureExtractor<Evidence>> = {
	"posterior-dispersion": dispersion,
	"leader-credence-band": leaderBand,
	"candidates-era-split": eraSplit,
	"owner-scoped": ownerScoped,
};
