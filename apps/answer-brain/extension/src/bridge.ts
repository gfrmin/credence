// bridge.ts — the body's client for the life-agent capability bridge (move-3).
// Each method is a thin POST/GET to one bridge endpoint; the bridge gathers and
// shapes evidence, the body composes it. Errors throw (the governor fail-opens
// around the whole loop); these calls are loopback and fast.

import type { ExtractResult, Hit, RouteResult } from "./types.js";

export interface BridgeClient {
	route(question: string): Promise<RouteResult | null>;
	retrieve(question: string, terms: string, k: number): Promise<Hit[]>;
	probeRecency(hitKeys: string[]): Promise<Record<string, string | null>>;
	extract(
		question: string,
		hits: Hit[],
		covariates: { doc_date?: Record<string, string | null>; subject_state?: Record<string, string> },
		timeIndexed: boolean,
	): Promise<ExtractResult>;
	utility(): Promise<Record<string, number>>;
}

export interface BridgeOptions {
	timeoutMs?: number;
	fetchImpl?: typeof fetch;
}

export function createBridgeClient(baseUrl: string, opts: BridgeOptions = {}): BridgeClient {
	const base = baseUrl.replace(/\/+$/, "");
	const timeoutMs = opts.timeoutMs ?? 60_000;
	const fetchImpl = opts.fetchImpl ?? fetch;

	async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
		const ctrl = new AbortController();
		const timer = setTimeout(() => ctrl.abort(), timeoutMs);
		try {
			const resp = await fetchImpl(`${base}${path}`, {
				method,
				headers: body === undefined ? undefined : { "Content-Type": "application/json" },
				body: body === undefined ? undefined : JSON.stringify(body),
				signal: ctrl.signal,
			});
			if (!resp.ok) {
				throw new Error(`bridge ${method} ${path} → ${resp.status}`);
			}
			return (await resp.json()) as T;
		} finally {
			clearTimeout(timer);
		}
	}

	return {
		async route(question) {
			// /route returns JSON null for the narrative path (not a typed lookup).
			return await call<RouteResult | null>("POST", "/route", { question });
		},
		async retrieve(question, terms, k) {
			const { hits } = await call<{ hits: Hit[] }>("POST", "/retrieve", { question, terms, k });
			return hits;
		},
		async probeRecency(hitKeys) {
			const { doc_date } = await call<{ doc_date: Record<string, string | null> }>(
				"POST",
				"/probe/recency",
				{ hit_keys: hitKeys },
			);
			return doc_date;
		},
		async extract(question, hits, covariates, timeIndexed) {
			return await call<ExtractResult>("POST", "/extract", {
				question,
				hits,
				covariates,
				time_indexed: timeIndexed,
			});
		},
		async utility() {
			const { u_bar } = await call<{ u_bar: Record<string, number> }>("GET", "/utility");
			return u_bar;
		},
	};
}
