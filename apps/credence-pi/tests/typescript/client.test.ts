// client.test.ts — step 6 of credence-pi: HTTP client unit tests.
//
// The load-bearing claim is fail-open: when the daemon is unreachable
// or slow, the body's POST /sensor must resolve quickly without
// throwing or hanging, with the failure logged. Pi proceeds without
// governance; pi must NEVER be blocked by the daemon's absence.
//
// Tests spin up tiny in-process HTTP servers via `node:http` to
// exercise the happy path, the timeout, and the SSE consumer; the
// fail-open case uses a port that is not bound to any server.

import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { createDaemonClient, type SignalEnvelope }
  from "../../extension/src/client.js";

// ── helpers ────────────────────────────────────────────────────────

interface CapturedPost { url: string; body: string; }

function withSensorServer<T>(
  fn: (baseUrl: string, posts: CapturedPost[]) => Promise<T>,
): Promise<T> {
  const posts: CapturedPost[] = [];
  const server = http.createServer((req, res) => {
    if (req.method !== "POST" || req.url !== "/sensor") {
      res.writeHead(404).end();
      return;
    }
    let body = "";
    req.on("data", chunk => { body += chunk; });
    req.on("end", () => {
      posts.push({ url: req.url!, body });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ack: true, event_id: "echoed" }));
    });
  });
  return new Promise((resolve, reject) => {
    server.listen(0, "127.0.0.1", async () => {
      const port = (server.address() as AddressInfo).port;
      try {
        const out = await fn(`http://127.0.0.1:${port}`, posts);
        resolve(out);
      } catch (e) {
        reject(e);
      } finally {
        server.close();
      }
    });
  });
}

function withSlowSensorServer<T>(
  fn: (baseUrl: string) => Promise<T>,
): Promise<T> {
  const server = http.createServer((req, res) => {
    // Never respond — test will time the client out.
    void req; void res;
  });
  return new Promise((resolve, reject) => {
    server.listen(0, "127.0.0.1", async () => {
      const port = (server.address() as AddressInfo).port;
      try {
        const out = await fn(`http://127.0.0.1:${port}`);
        resolve(out);
      } catch (e) {
        reject(e);
      } finally {
        // closeAllConnections on the server so the unfinished sockets
        // don't keep node alive past the test return.
        server.closeAllConnections?.();
        server.close();
      }
    });
  });
}

interface SSEHandle {
  baseUrl: string;
  push: (signal: object) => void;
  pushRaw: (frame: string) => void;
  close: () => void;
}

function startSseServer(): Promise<SSEHandle> {
  return new Promise((resolve) => {
    const writers: http.ServerResponse[] = [];
    const server = http.createServer((req, res) => {
      if (req.method !== "GET" || req.url !== "/signals") {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      });
      writers.push(res);
      req.on("close", () => {
        const i = writers.indexOf(res);
        if (i >= 0) writers.splice(i, 1);
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      resolve({
        baseUrl: `http://127.0.0.1:${port}`,
        push: (sig) => {
          for (const w of writers) {
            w.write(`data: ${JSON.stringify(sig)}\n\n`);
          }
        },
        pushRaw: (frame) => {
          for (const w of writers) w.write(frame);
        },
        close: () => {
          for (const w of writers) w.end();
          server.closeAllConnections?.();
          server.close();
        },
      });
    });
  });
}

// ── 1. POST /sensor happy path ─────────────────────────────────────

test("postSensor: happy path POSTs JSON body to /sensor and resolves", async () => {
  await withSensorServer(async (baseUrl, posts) => {
    const client = createDaemonClient({
      baseUrl, logger: () => { /* silenced */ },
    });
    await client.postSensor({
      event_type: "tool-proposed",
      event_id: "evt_1",
      features: { "tool-name": "bash" },
    });
    assert.equal(posts.length, 1);
    assert.equal(posts[0]!.url, "/sensor");
    const parsed = JSON.parse(posts[0]!.body);
    assert.equal(parsed.event_type, "tool-proposed");
    assert.equal(parsed.event_id, "evt_1");
    assert.equal(parsed.features["tool-name"], "bash");
  });
});

// ── 2. Fail-open: unreachable URL ──────────────────────────────────
//
// Load-bearing safety claim. Body must NOT throw, must NOT hang.

test("postSensor: unreachable daemon resolves quickly with logged warning", async () => {
  // 127.0.0.1:1 is reserved (tcpmux) and almost certainly closed; the
  // OS rejects the connection synchronously on Linux, so the test does
  // not need to lean on the per-request timeout to surface the failure.
  const logged: string[] = [];
  const client = createDaemonClient({
    baseUrl: "http://127.0.0.1:1",
    timeoutMs: 5_000,
    logger: (msg) => { logged.push(msg); },
  });
  const t0 = Date.now();
  await client.postSensor({ event_type: "tool-proposed", event_id: "x" });
  const elapsed = Date.now() - t0;
  assert.ok(elapsed < 4_000,
    `postSensor on unreachable URL took ${elapsed}ms; expected fast fail-open`);
  assert.ok(logged.some(m => /unreachable/.test(m)),
    `expected an unreachable warning; got ${JSON.stringify(logged)}`);
});

// ── 3. Fail-open: slow daemon (timeout) ────────────────────────────

test("postSensor: slow daemon times out without throwing or hanging", async () => {
  await withSlowSensorServer(async (baseUrl) => {
    const logged: string[] = [];
    const client = createDaemonClient({
      baseUrl,
      timeoutMs: 250,
      logger: (msg) => { logged.push(msg); },
    });
    const t0 = Date.now();
    await client.postSensor({ event_type: "tool-proposed", event_id: "x" });
    const elapsed = Date.now() - t0;
    assert.ok(elapsed < 2_000,
      `postSensor against slow daemon took ${elapsed}ms; expected ~timeoutMs + slack`);
    assert.ok(logged.some(m => /unreachable/.test(m)),
      `expected unreachable/timeout warning; got ${JSON.stringify(logged)}`);
  });
});

// ── 4. SSE: receives signals, dispatches parsed JSON to callback ───

test("connectSignalsStream: receives a signal and dispatches it", async () => {
  const sse = await startSseServer();
  const received: SignalEnvelope[] = [];
  const conn = createDaemonClient({
    baseUrl: sse.baseUrl,
    logger: () => { /* silenced */ },
  }).connectSignalsStream((sig) => { received.push(sig); });

  try {
    // Wait briefly for the consumer to register, then push.
    await new Promise(r => setTimeout(r, 50));
    sse.push({
      signal_type: "effector",
      signal_id: "sig_1",
      in_response_to: "evt_1",
      effector: "ask",
      parameters: { text: "go ahead?" },
    });
    for (let i = 0; i < 50 && received.length === 0; i++) {
      await new Promise(r => setTimeout(r, 20));
    }
    assert.equal(received.length, 1);
    assert.equal(received[0]!.effector, "ask");
    assert.equal(received[0]!.parameters.text, "go ahead?");
  } finally {
    conn.close();
    await conn.done;
    sse.close();
  }
});

// ── 5. SSE: malformed `data:` payload is logged and skipped ────────

test("connectSignalsStream: malformed JSON in a frame is logged, not thrown", async () => {
  const sse = await startSseServer();
  const received: SignalEnvelope[] = [];
  const logged: string[] = [];
  const conn = createDaemonClient({
    baseUrl: sse.baseUrl,
    logger: (msg) => { logged.push(msg); },
  }).connectSignalsStream((sig) => { received.push(sig); });

  try {
    await new Promise(r => setTimeout(r, 50));
    // Frame whose body is not valid JSON.
    sse.pushRaw("data: {not-json\n\n");
    // Then a valid frame to confirm the consumer kept going past the
    // malformed one rather than silently dying.
    sse.push({
      signal_type: "effector",
      signal_id: "sig_after",
      in_response_to: "evt",
      effector: "proceed",
      parameters: {},
    });
    for (let i = 0; i < 50 && received.length === 0; i++) {
      await new Promise(r => setTimeout(r, 20));
    }
    assert.equal(received.length, 1);
    assert.equal(received[0]!.effector, "proceed");
    assert.ok(logged.some(m => /malformed/.test(m)),
      `expected a malformed-frame log; got ${JSON.stringify(logged)}`);
  } finally {
    conn.close();
    await conn.done;
    sse.close();
  }
});
