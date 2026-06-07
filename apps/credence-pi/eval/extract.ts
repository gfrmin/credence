// extract.ts — CLI: transcripts -> normalized tool-call event stream (JSONL).
//
//   # native: each input FILE is one session
//   node --import tsx extract.ts --format native [--out events.jsonl] s1.jsonl s2.jsonl
//
//   # clawsbench: input file is a JSONL of records, each record is one session
//   node --import tsx extract.ts --format clawsbench [--harness openclaw] \
//        [--limit N] [--out events.jsonl] train.jsonl
//
// Output: NormalizedEvent JSONL (one line per tool call). "Record everything":
// every parsed call is emitted; a stderr summary reports counts, sessions with
// zero tool calls, and any filtering/limit applied (no silent drops).

import { createReadStream, readFileSync, writeFileSync, createWriteStream } from "node:fs";
import { createInterface } from "node:readline";
import { basename } from "node:path";
import { adaptNative } from "./adapters/native_openclaw.js";
import { adaptClawsbench, type ClawsbenchRecord } from "./adapters/clawsbench.js";
import type { NormalizedEvent } from "./types.js";

interface Args {
  format: string;
  out: string | null;
  harness: string | null;
  limit: number | null;
  inputs: string[];
}

function parseArgs(argv: string[]): Args {
  const a: Args = { format: "native", out: null, harness: null, limit: null, inputs: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === "--format") a.format = argv[++i];
    else if (t === "--out") a.out = argv[++i];
    else if (t === "--harness") a.harness = argv[++i];
    else if (t === "--limit") a.limit = parseInt(argv[++i], 10);
    else a.inputs.push(t);
  }
  return a;
}

function emit(out: string | null, lines: string[]): void {
  const body = lines.join("\n") + (lines.length ? "\n" : "");
  if (out) writeFileSync(out, body);
  else process.stdout.write(body);
}

// native: each file = one session, parsed whole.
function runNative(args: Args): void {
  const all: NormalizedEvent[] = [];
  const perFile: Array<{ file: string; calls: number }> = [];
  for (const file of args.inputs) {
    const sessionId = basename(file).replace(/\.jsonl$/, "");
    const records = readFileSync(file, "utf8")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => JSON.parse(l)) as Parameters<typeof adaptNative>[0];
    const events = adaptNative(records, sessionId, { corpus: "native", file });
    all.push(...events);
    perFile.push({ file: sessionId, calls: events.length });
  }
  emit(args.out, all.map((e) => JSON.stringify(e)));
  process.stderr.write(`extract(native): ${all.length} tool calls from ${args.inputs.length} session(s)\n`);
  for (const p of perFile) process.stderr.write(`  ${p.file}: ${p.calls}\n`);
}

// clawsbench: stream the big JSONL; each line is one session record.
async function runClawsbench(args: Args): Promise<void> {
  const outStream = args.out ? createWriteStream(args.out) : process.stdout;
  let sessions = 0,
    kept = 0,
    filtered = 0,
    empties = 0,
    calls = 0;
  for (const file of args.inputs) {
    const rl = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
    for await (const line of rl) {
      const t = line.trim();
      if (!t) continue;
      sessions += 1;
      let rec: ClawsbenchRecord;
      try {
        rec = JSON.parse(t);
      } catch {
        process.stderr.write("  warn: skipped malformed record line\n");
        continue;
      }
      if (args.harness && rec.harness !== args.harness) {
        filtered += 1;
        continue;
      }
      if (args.limit != null && kept >= args.limit) break;
      const events = adaptClawsbench(rec);
      kept += 1;
      if (events.length === 0) empties += 1;
      for (const e of events) {
        outStream.write(JSON.stringify(e) + "\n");
        calls += 1;
      }
    }
  }
  if (args.out) (outStream as ReturnType<typeof createWriteStream>).end();
  process.stderr.write(
    `extract(clawsbench): ${calls} tool calls from ${kept} kept session(s) ` +
      `(${sessions} seen, ${filtered} filtered by harness=${args.harness ?? "*"}, ` +
      `${empties} kept-but-toolless${args.limit != null ? `, limit=${args.limit}` : ""})\n`,
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.inputs.length === 0) {
    process.stderr.write(
      "usage: extract.ts --format native|clawsbench [--harness H] [--limit N] [--out f] <input.jsonl ...>\n",
    );
    process.exit(2);
  }
  if (args.format === "native") runNative(args);
  else if (args.format === "clawsbench") await runClawsbench(args);
  else throw new Error(`unknown format '${args.format}'`);
}

main();
