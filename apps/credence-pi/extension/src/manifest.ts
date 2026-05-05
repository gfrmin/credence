// manifest.ts — parse capabilities.bdsl and features.bdsl from
// apps/credence-pi/bdsl/, verify the body has implementations and
// extractors registered for every declared effector and feature.
// Body-side; the brain runs the full BDSL evaluator. Pass 1 manifests
// use only symbols and parens; no string literals, nested non-list
// forms, or macros. Pass 2 will need string-literal tokenisation if
// (parameters …) or (feature …) ever takes string defaults; tightest
// wins until then.

import { readFileSync } from "node:fs";

export interface EffectorParam { name: string; type: string; }
export interface EffectorDecl { name: string; parameters: EffectorParam[]; }
export interface FeatureDecl { name: string; spaceName: string; }

// Registry keys are kebab-case strings matching the BDSL declarations
// verbatim ("ask", "proceed", "block", "tool-name",
// "recent-repetition-count", …) — NOT camelCase. The verifier names
// missing keys in the same form so a developer can grep the manifest
// file directly.
export type Registry = Record<string, unknown>;
export type SExpr = string | SExpr[];

function tokenize(src: string): string[] {
  const tokens: string[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (c === ";") { while (i < src.length && src[i] !== "\n") i++; }
    else if (c === " " || c === "\t" || c === "\n" || c === "\r") { i++; }
    else if (c === "(" || c === ")") { tokens.push(c); i++; }
    else {
      let j = i;
      while (j < src.length && !" \t\n\r();".includes(src[j]!)) j++;
      tokens.push(src.slice(i, j));
      i = j;
    }
  }
  return tokens;
}

function read(tokens: string[]): SExpr[] {
  let pos = 0;
  const readOne = (): SExpr => {
    if (pos >= tokens.length) throw new Error("manifest: unexpected end of input");
    const tok = tokens[pos++]!;
    if (tok === "(") {
      const list: SExpr[] = [];
      while (pos < tokens.length && tokens[pos] !== ")") list.push(readOne());
      if (pos >= tokens.length) throw new Error("manifest: missing closing ')'");
      pos++;
      return list;
    }
    if (tok === ")") throw new Error("manifest: unexpected ')'");
    return tok;
  };
  const out: SExpr[] = [];
  while (pos < tokens.length) out.push(readOne());
  return out;
}

// Walker strategy: top-level-or-inside-list-only. Both manifest files
// wrap declarations in (define X (list ...)), so we descend through
// nested lists matching forms whose head equals `head`. Pass 2 may need
// to revisit if BDSL ever gains macro-like manifest constructs (e.g.
// (let ... (effector ...))); Pass 1 has no such case.
function collect<T>(exprs: SExpr[], head: string, parseOne: (form: SExpr[]) => T): T[] {
  const out: T[] = [];
  const visit = (e: SExpr): void => {
    if (!Array.isArray(e)) return;
    if (e.length > 0 && e[0] === head) { out.push(parseOne(e)); return; }
    for (const child of e) visit(child);
  };
  for (const e of exprs) visit(e);
  return out;
}

const fmt = (e: SExpr | undefined): string =>
  e === undefined ? "<undefined>"
    : Array.isArray(e) ? "(" + e.map(fmt).join(" ") + ")"
    : e;

function parseEffectorForm(form: SExpr[]): EffectorDecl {
  if (form.length < 2 || typeof form[1] !== "string") {
    throw new Error(`manifest: effector form missing name in ${fmt(form)}`);
  }
  const name = form[1];
  const parameters: EffectorParam[] = [];
  for (let i = 2; i < form.length; i++) {
    const clause = form[i]!;
    if (!Array.isArray(clause) || clause[0] !== "parameters") {
      throw new Error(`manifest: effector ${name}: expected (parameters ...), got ${fmt(clause)}`);
    }
    for (let j = 1; j < clause.length; j++) {
      const p = clause[j]!;
      if (!Array.isArray(p) || p.length !== 2 ||
          typeof p[0] !== "string" || typeof p[1] !== "string") {
        throw new Error(`manifest: effector ${name}: parameter must be (name type), got ${fmt(p)}`);
      }
      parameters.push({ name: p[0], type: p[1] });
    }
  }
  return { name, parameters };
}

function parseFeatureForm(form: SExpr[]): FeatureDecl {
  if (form.length !== 3 || typeof form[1] !== "string" || typeof form[2] !== "string") {
    throw new Error(`manifest: feature form must be (feature NAME SPACE), got ${fmt(form)}`);
  }
  return { name: form[1], spaceName: form[2] };
}

// Low-level access to the s-expr reader so consumers (the features-test)
// can walk forms beyond manifest's effector/feature surface.
export function parseSExprs(src: string): SExpr[] { return read(tokenize(src)); }

export function parseCapabilities(src: string): EffectorDecl[] {
  return collect(read(tokenize(src)), "effector", parseEffectorForm);
}

export function parseFeatures(src: string): FeatureDecl[] {
  return collect(read(tokenize(src)), "feature", parseFeatureForm);
}

export function readCapabilities(path: string): EffectorDecl[] {
  return parseCapabilities(readFileSync(path, "utf-8"));
}

export function readFeatures(path: string): FeatureDecl[] {
  return parseFeatures(readFileSync(path, "utf-8"));
}

export function verifyEffectors(decls: EffectorDecl[], registry: Registry): void {
  const missing = decls.filter(d => !(d.name in registry));
  if (missing.length > 0) {
    throw new Error(
      `manifest: no implementation registered for declared effector(s): ${missing.map(d => d.name).join(", ")}`,
    );
  }
}

export function verifyFeatures(decls: FeatureDecl[], registry: Registry): void {
  const missing = decls.filter(d => !(d.name in registry));
  if (missing.length > 0) {
    throw new Error(
      `manifest: no extractor registered for declared feature(s): ${missing.map(d => d.name).join(", ")}`,
    );
  }
}
