// daemon-client.ts — body-side HTTP client for the credence-pi daemon.
//
// shared-source: apps/credence-pi/extension/src/client.ts
//   This is a vendored copy (verbatim behaviour) so the OpenClaw plugin
//   is a self-contained, installable package. The pi-extension body and
//   this OpenClaw body MUST keep the same daemon wire (POST /sensor,
//   GET /signals SSE). If you change one, change the other; client.test.ts
//   in the extension covers the reference behaviour.
//
// Two responsibilities, both fail-open:
//   postSensor(event)              — POST /sensor with a timeout; never
//                                    throws, never hangs; logs once on
//                                    failure and returns {ok:false}.
//   connectSignalsStream(onSignal) — streaming GET /signals consumer;
//                                    parses `data:` SSE frames; auto-
//                                    reconnects with exponential backoff
//                                    until close().

export interface SignalEnvelope {
  signal_type: string;
  signal_id: string;
  in_response_to: string;
  effector: string;
  parameters: Record<string, unknown>;
}

export type Logger = (msg: string, err?: unknown) => void;

export interface ClientOptions {
  baseUrl: string;
  timeoutMs?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  logger?: Logger;
}

export interface PostResult {
  ok: boolean;
}

export interface DaemonClient {
  postSensor: (event: object) => Promise<PostResult>;
  connectSignalsStream: (
    onSignal: (sig: SignalEnvelope) => void,
  ) => SignalsConnection;
}

export interface SignalsConnection {
  close: () => void;
  done: Promise<void>;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_INITIAL_BACKOFF_MS = 500;
const DEFAULT_MAX_BACKOFF_MS = 30_000;

export function createDaemonClient(opts: ClientOptions): DaemonClient {
  const baseUrl = opts.baseUrl.replace(/\/+$/, "");
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const initialBackoff = opts.initialBackoffMs ?? DEFAULT_INITIAL_BACKOFF_MS;
  const maxBackoff = opts.maxBackoffMs ?? DEFAULT_MAX_BACKOFF_MS;
  const log: Logger =
    opts.logger ??
    ((m, e) => (e === undefined ? console.warn(m) : console.warn(m, e)));

  return {
    postSensor: (event) => postSensor(baseUrl, event, timeoutMs, log),
    connectSignalsStream: (onSignal) =>
      connectSignalsStream(baseUrl, onSignal, initialBackoff, maxBackoff, log),
  };
}

async function postSensor(
  baseUrl: string,
  event: object,
  timeoutMs: number,
  log: Logger,
): Promise<PostResult> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(`${baseUrl}/sensor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
      signal: ctrl.signal,
    });
    try {
      await resp.text();
    } catch {
      /* noop */
    }
    if (!resp.ok) {
      log(`credence-pi: daemon /sensor returned status ${resp.status}; failing open`);
      return { ok: false };
    }
    return { ok: true };
  } catch (err) {
    log("credence-pi: daemon unreachable on /sensor; failing open", err);
    return { ok: false };
  } finally {
    clearTimeout(timer);
  }
}

function connectSignalsStream(
  baseUrl: string,
  onSignal: (sig: SignalEnvelope) => void,
  initialBackoff: number,
  maxBackoff: number,
  log: Logger,
): SignalsConnection {
  const ctrl = new AbortController();
  let closed = false;

  const done = (async () => {
    let backoff = initialBackoff;
    while (!closed) {
      try {
        await consumeOnce(baseUrl, onSignal, ctrl.signal, log);
        if (closed) break;
        log("credence-pi: /signals stream ended; reconnecting");
      } catch (err) {
        if (closed) break;
        log("credence-pi: /signals stream error; reconnecting", err);
      }
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, backoff);
        ctrl.signal.addEventListener(
          "abort",
          () => {
            clearTimeout(t);
            resolve();
          },
          { once: true },
        );
      });
      backoff = Math.min(backoff * 2, maxBackoff);
    }
  })();

  return {
    close: () => {
      closed = true;
      ctrl.abort();
    },
    done,
  };
}

async function consumeOnce(
  baseUrl: string,
  onSignal: (sig: SignalEnvelope) => void,
  signal: AbortSignal,
  log: Logger,
): Promise<void> {
  const resp = await fetch(`${baseUrl}/signals`, {
    method: "GET",
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`/signals returned ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        dispatchFrame(frame, onSignal, log);
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }
}

function dispatchFrame(
  frame: string,
  onSignal: (sig: SignalEnvelope) => void,
  log: Logger,
): void {
  for (const line of frame.split("\n")) {
    if (line.startsWith("data: ")) {
      const payload = line.slice(6);
      try {
        onSignal(JSON.parse(payload) as SignalEnvelope);
      } catch (err) {
        log("credence-pi: /signals dispatch dropped malformed frame", err);
      }
    }
  }
}
